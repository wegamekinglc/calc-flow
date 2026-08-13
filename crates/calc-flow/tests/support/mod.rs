#![allow(dead_code)]

use std::{
    collections::BTreeMap,
    fs::File,
    path::PathBuf,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchExecutionPlan, BatchKind, BatchMetadata, BatchOperator, BatchOperatorContext,
    CalcFlowError, CancellationToken, ExpressionOperator, JsonMap, OperatorMetadata,
    PipelineBuilder, Port, Result, SqlOperator, UdfRegistry,
};
use datafusion::arrow::{
    array::{Int64Array, StringArray},
    datatypes::{DataType, Field, Schema},
    ipc::reader::FileReader,
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
    GateOncePass {
        started: Arc<tokio::sync::Notify>,
        release: Arc<tokio::sync::Notify>,
        pending: Arc<std::sync::atomic::AtomicBool>,
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

impl OperatorMetadata for TestOperator {
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
}

#[async_trait]
impl BatchOperator for TestOperator {
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
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
            Action::GateOncePass {
                started,
                release,
                pending,
            } => {
                if pending.swap(false, Ordering::SeqCst) {
                    started.notify_one();
                    release.notified().await;
                }
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

pub fn read_v1_fixture(name: &str) -> Vec<RecordBatch> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/v1")
        .join(name);
    let reader = FileReader::try_new(File::open(path).unwrap(), None).unwrap();
    let schema = reader.schema();
    let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>().unwrap();
    if batches.is_empty() {
        vec![RecordBatch::new_empty(schema)]
    } else {
        batches
    }
}

pub fn v1_expression_plan(expression: &str) -> BatchExecutionPlan {
    PipelineBuilder::new("v1-expression")
        .unwrap()
        .add_node(
            "calculate",
            Box::new(
                ExpressionOperator::new("calculate", expression, Vec::new(), None, Vec::new())
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

pub fn v1_sql_plan() -> BatchExecutionPlan {
    PipelineBuilder::new("v1-sql")
        .unwrap()
        .add_node(
            "join",
            Box::new(
                SqlOperator::new(
                    "join",
                    "SELECT l.id, l.amount * r.rate AS total FROM l JOIN r ON l.id = r.id",
                    vec!["l".into(), "r".into()],
                    Vec::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

pub fn v1_identity_plan(name: &str) -> BatchExecutionPlan {
    PipelineBuilder::new(name)
        .unwrap()
        .add_node(
            "identity",
            Box::new(TestOperator::ports(
                "identity",
                vec![untyped_table_port("input", true)],
                vec![untyped_table_port("output", true)],
                Action::Pass,
                Arc::new(Probe::default()),
            )) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

pub fn v1_rollback_plan() -> BatchExecutionPlan {
    PipelineBuilder::new("v1-state-rollback")
        .unwrap()
        .add_node(
            "state",
            Box::new(EmptyStateFailingOperator::default()) as Box<dyn BatchOperator>,
        )
        .unwrap()
        .compile_batch(&UdfRegistry::new().snapshot())
        .unwrap()
}

struct EmptyStateFailingOperator {
    inputs: [Port; 1],
    outputs: [Port; 1],
    state: JsonMap,
}

impl Default for EmptyStateFailingOperator {
    fn default() -> Self {
        Self {
            inputs: [untyped_table_port("input", true)],
            outputs: [untyped_table_port("output", true)],
            state: JsonMap::new(),
        }
    }
}

impl OperatorMetadata for EmptyStateFailingOperator {
    fn name(&self) -> &'static str {
        "fail_after_state"
    }

    fn input_ports(&self) -> &[Port] {
        &self.inputs
    }

    fn output_ports(&self) -> &[Port] {
        &self.outputs
    }

    fn configuration(&self) -> JsonMap {
        JsonMap::new()
    }
}

#[async_trait]
impl BatchOperator for EmptyStateFailingOperator {
    async fn process(
        &mut self,
        _inputs: &BTreeMap<String, Batch>,
        _context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        self.state.insert("mutated".into(), json!(true));
        Err(CalcFlowError::Operator {
            node_id: "state".into(),
            message: "fail_after_state".into(),
        })
    }

    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Object(self.state.clone().into_iter().collect()))
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        self.state = state
            .as_object()
            .ok_or_else(|| CalcFlowError::Format {
                message: "state rollback fixture requires an object".into(),
            })?
            .clone()
            .into_iter()
            .collect();
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state.clear();
        Ok(())
    }
}
