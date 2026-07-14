#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, Checkpoint, CheckpointStore,
    Edge, JsonMap, Operator, OperatorContext, PipelineBuilder, Port, PortEndpoint, Result,
    RunContext, Sink, Source, SourceItem, UdfRegistry,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

#[derive(Clone, Debug)]
pub enum Action {
    Pass,
    DelayPass(Duration),
    Fail(&'static str),
    CancelAndPass(CancellationToken),
    CancelAndFail(CancellationToken, &'static str),
    GatePass {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
    },
    MissingOutput,
    UnknownOutput,
    WrongSchema,
}

#[derive(Debug, Default)]
pub struct Probe {
    calls: AtomicUsize,
    snapshots: AtomicUsize,
    restores: AtomicUsize,
    resets: AtomicUsize,
    active: AtomicUsize,
    max_active: AtomicUsize,
    order: Mutex<Vec<String>>,
}

impl Probe {
    pub fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }

    pub fn restores(&self) -> usize {
        self.restores.load(Ordering::SeqCst)
    }

    pub fn snapshots(&self) -> usize {
        self.snapshots.load(Ordering::SeqCst)
    }

    pub fn resets(&self) -> usize {
        self.resets.load(Ordering::SeqCst)
    }

    pub fn max_active(&self) -> usize {
        self.max_active.load(Ordering::SeqCst)
    }

    pub fn order(&self) -> Vec<String> {
        self.order.lock().unwrap().clone()
    }

    fn enter(&self, name: &str) -> ActiveGuard<'_> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.order.lock().unwrap().push(name.to_owned());
        let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_active.fetch_max(active, Ordering::SeqCst);
        ActiveGuard(self)
    }
}

struct ActiveGuard<'a>(&'a Probe);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

pub struct TestOperator {
    name: String,
    inputs: Vec<Port>,
    outputs: Vec<Port>,
    action: Action,
    probe: Arc<Probe>,
    state: i64,
    mutate_state: bool,
    fail_restore: bool,
    fail_reset: bool,
}

impl TestOperator {
    pub fn transform(name: &str, action: Action, probe: Arc<Probe>) -> Self {
        Self {
            name: name.into(),
            inputs: vec![table_port("input", true)],
            outputs: vec![table_port("output", true)],
            action,
            probe,
            state: 0,
            mutate_state: false,
            fail_restore: false,
            fail_reset: false,
        }
    }

    pub fn ports(
        name: &str,
        inputs: Vec<Port>,
        outputs: Vec<Port>,
        action: Action,
        probe: Arc<Probe>,
    ) -> Self {
        Self {
            name: name.into(),
            inputs,
            outputs,
            action,
            probe,
            state: 0,
            mutate_state: false,
            fail_restore: false,
            fail_reset: false,
        }
    }

    pub const fn stateful(mut self) -> Self {
        self.mutate_state = true;
        self
    }

    pub const fn failing_restore(mut self) -> Self {
        self.fail_restore = true;
        self
    }

    pub const fn failing_reset(mut self) -> Self {
        self.fail_reset = true;
        self
    }
}

#[async_trait]
impl Operator for TestOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let _guard = self.probe.enter(&self.name);
        if self.mutate_state {
            self.state += 1;
        }
        match &self.action {
            Action::DelayPass(duration) => tokio::time::sleep(*duration).await,
            Action::Fail(message) => {
                return Err(CalcFlowError::Operator {
                    node_id: self.name.clone(),
                    message: (*message).into(),
                });
            }
            Action::CancelAndPass(token) => token.cancel(),
            Action::CancelAndFail(token, message) => {
                token.cancel();
                return Err(CalcFlowError::Operator {
                    node_id: self.name.clone(),
                    message: (*message).into(),
                });
            }
            Action::GatePass { started, release } => {
                started.notify_one();
                release.notified().await;
            }
            Action::MissingOutput => return Ok(BTreeMap::new()),
            Action::UnknownOutput => {
                return Ok(BTreeMap::from([(
                    "unknown".into(),
                    first_input(inputs)?.clone(),
                )]));
            }
            Action::WrongSchema => {
                return Ok(BTreeMap::from([("output".into(), string_batch(&["bad"]))]));
            }
            Action::Pass => {}
        }
        let batch = first_input(inputs)?.clone();
        let output_name = self.outputs.first().map_or("output", |port| port.name());
        Ok(BTreeMap::from([(output_name.into(), batch)]))
    }

    fn snapshot(&self) -> Result<Value> {
        self.probe.snapshots.fetch_add(1, Ordering::SeqCst);
        Ok(json!({"state": self.state}))
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        self.probe.restores.fetch_add(1, Ordering::SeqCst);
        if self.fail_restore {
            return Err(CalcFlowError::Format {
                message: format!("{} restore injected", self.name),
            });
        }
        self.state =
            state
                .get("state")
                .and_then(Value::as_i64)
                .ok_or_else(|| CalcFlowError::Format {
                    message: format!("{} state is invalid", self.name),
                })?;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.probe.resets.fetch_add(1, Ordering::SeqCst);
        if self.fail_reset {
            return Err(CalcFlowError::Format {
                message: format!("{} reset injected", self.name),
            });
        }
        self.state = 0;
        Ok(())
    }
}

fn first_input(inputs: &BTreeMap<String, Batch>) -> Result<&Batch> {
    inputs
        .values()
        .next()
        .ok_or_else(|| CalcFlowError::Operator {
            node_id: "test".into(),
            message: "test operator requires one input".into(),
        })
}

pub fn table_port(name: &str, required: bool) -> Port {
    Port::new(
        name,
        BatchKind::Table,
        required,
        Some(vec![Field::new("value", DataType::Int64, false)]),
    )
    .unwrap()
}

pub fn untyped_table_port(name: &str, required: bool) -> Port {
    Port::new(name, BatchKind::Table, required, None).unwrap()
}

pub fn int_batch(values: &[i64]) -> Batch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    let record =
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

pub fn string_batch(values: &[&str]) -> Batch {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Utf8,
        false,
    )]));
    let record =
        RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(values.to_vec()))]).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

pub fn stateful_plan(name: &str, probe: Arc<Probe>) -> Arc<calc_flow::ExecutionPlan> {
    Arc::new(
        PipelineBuilder::new(name)
            .unwrap()
            .add_node(
                "node",
                Box::new(TestOperator::transform("node", Action::Pass, probe).stateful()),
            )
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    )
}

pub fn partially_failing_reset_plan(
    name: &str,
    first_probe: Arc<Probe>,
    second_probe: Arc<Probe>,
) -> Arc<calc_flow::ExecutionPlan> {
    Arc::new(
        PipelineBuilder::new(name)
            .unwrap()
            .add_node(
                "first",
                Box::new(TestOperator::transform("first", Action::Pass, first_probe).stateful()),
            )
            .unwrap()
            .add_node(
                "second",
                Box::new(
                    TestOperator::transform("second", Action::Pass, second_probe)
                        .stateful()
                        .failing_reset(),
                ),
            )
            .unwrap()
            .connect(Edge::new(
                PortEndpoint::new("first", "output").unwrap(),
                PortEndpoint::new("second", "input").unwrap(),
            ))
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    )
}

#[derive(Default)]
pub struct MemoryCheckpointStore {
    checkpoint: Mutex<Option<Checkpoint>>,
    saves: AtomicUsize,
    deletes: AtomicUsize,
    fail_saves: AtomicUsize,
    fail_deletes: AtomicUsize,
    fail_deletes_after_remove: AtomicUsize,
}

impl MemoryCheckpointStore {
    pub fn with_checkpoint(checkpoint: Checkpoint) -> Self {
        Self {
            checkpoint: Mutex::new(Some(checkpoint)),
            ..Self::default()
        }
    }

    pub fn fail_next_saves(&self, count: usize) {
        self.fail_saves.store(count, Ordering::SeqCst);
    }

    pub fn fail_next_deletes(&self, count: usize) {
        self.fail_deletes.store(count, Ordering::SeqCst);
    }

    pub fn fail_next_deletes_after_remove(&self, count: usize) {
        self.fail_deletes_after_remove
            .store(count, Ordering::SeqCst);
    }

    pub fn checkpoint(&self) -> Option<Checkpoint> {
        self.checkpoint.lock().unwrap().clone()
    }

    pub fn saves(&self) -> usize {
        self.saves.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl CheckpointStore for MemoryCheckpointStore {
    async fn load(&self, _pipeline_name: &str) -> Result<Option<Checkpoint>> {
        Ok(self.checkpoint())
    }

    async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        self.saves.fetch_add(1, Ordering::SeqCst);
        if self
            .fail_saves
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(CalcFlowError::Format {
                message: "save injected".into(),
            });
        }
        *self.checkpoint.lock().unwrap() = Some(checkpoint.clone());
        Ok(())
    }

    async fn delete(&self, _pipeline_name: &str) -> Result<()> {
        self.deletes.fetch_add(1, Ordering::SeqCst);
        if self
            .fail_deletes
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(CalcFlowError::Format {
                message: "delete injected".into(),
            });
        }
        *self.checkpoint.lock().unwrap() = None;
        if self
            .fail_deletes_after_remove
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(CalcFlowError::Format {
                message: "delete after removal injected".into(),
            });
        }
        Ok(())
    }
}

pub struct QueueSource {
    items: VecDeque<SourceItem>,
    opens: Arc<Mutex<Vec<Option<Value>>>>,
}

impl QueueSource {
    pub fn new(items: Vec<SourceItem>) -> (Self, Arc<Mutex<Vec<Option<Value>>>>) {
        let opens = Arc::new(Mutex::new(Vec::new()));
        (
            Self {
                items: items.into(),
                opens: Arc::clone(&opens),
            },
            opens,
        )
    }
}

#[async_trait]
impl Source for QueueSource {
    async fn open(&mut self, cursor: Option<Value>) -> Result<()> {
        self.opens.lock().unwrap().push(cursor);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceItem>> {
        Ok(self.items.pop_front())
    }
}

pub fn source_item(values: &[i64], cursor: Value, sequence: u64) -> SourceItem {
    SourceItem {
        batch: int_batch(values),
        cursor: Some(cursor),
        sequence,
    }
}

pub struct RecordingSink {
    label: String,
    calls: Arc<Mutex<Vec<(String, String, usize)>>>,
    failures: Arc<AtomicUsize>,
}

pub struct GatedSink {
    started: Arc<tokio::sync::Notify>,
    release: Arc<tokio::sync::Notify>,
}

impl GatedSink {
    pub fn new(started: Arc<tokio::sync::Notify>, release: Arc<tokio::sync::Notify>) -> Self {
        Self { started, release }
    }
}

#[async_trait]
impl Sink for GatedSink {
    async fn write(&mut self, _batch: &Batch, _context: &RunContext) -> Result<()> {
        self.started.notify_one();
        self.release.notified().await;
        Ok(())
    }
}

impl RecordingSink {
    pub fn new(
        label: &str,
        calls: Arc<Mutex<Vec<(String, String, usize)>>>,
        failures: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            label: label.into(),
            calls,
            failures,
        }
    }
}

#[async_trait]
impl Sink for RecordingSink {
    async fn write(&mut self, batch: &Batch, context: &RunContext) -> Result<()> {
        self.calls.lock().unwrap().push((
            self.label.clone(),
            context.run_id().into(),
            batch.num_rows(),
        ));
        if self
            .failures
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |remaining| {
                (remaining > 0).then(|| remaining - 1)
            })
            .is_ok()
        {
            return Err(CalcFlowError::Format {
                message: format!("{} injected", self.label),
            });
        }
        Ok(())
    }
}
