use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use allocation_counter::{AllocationInfo, measure};
use async_trait::async_trait;
use datafusion::arrow::{
    array::{ArrayRef, Int64Array},
    record_batch::RecordBatch,
};
use serde_json::{Value, json};

use super::{
    CompiledOperator, Edge, ExecutionOptions, ExecutionPlan, PipelineBuilder, PortEndpoint,
};
use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionRuntime, JsonMap, Operator,
    OperatorContext, Port, Result, RunContext, UdfRegistry, operator::SignalAwareOperator,
    runtime::ControlMarker,
};

struct FixtureOperator {
    input_ports: [Port; 1],
    output_ports: [Port; 1],
}

impl FixtureOperator {
    fn new() -> Self {
        Self {
            input_ports: [Port::new("input", BatchKind::Table, true, None).unwrap()],
            output_ports: [Port::new("output", BatchKind::Table, true, None).unwrap()],
        }
    }
}

#[async_trait]
impl Operator for FixtureOperator {
    fn name(&self) -> &'static str {
        "signal-allocation-fixture"
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        BTreeMap::new()
    }

    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(inputs
            .values()
            .next()
            .cloned()
            .map(|batch| BTreeMap::from([("output".into(), batch)]))
            .unwrap_or_default())
    }
}

struct NoopSignalAwareProbe {
    control_calls: Arc<AtomicUsize>,
}

#[async_trait]
impl SignalAwareOperator for NoopSignalAwareProbe {
    async fn process_data(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _run: &RunContext,
        _datafusion: Option<&DataFusionRuntime>,
    ) -> Result<BTreeMap<String, Batch>> {
        Ok(inputs
            .values()
            .next()
            .cloned()
            .map(|batch| BTreeMap::from([("output".into(), batch)]))
            .unwrap_or_default())
    }

    async fn handle_control(
        &mut self,
        _marker: &ControlMarker,
        _context: &RunContext,
    ) -> Result<()> {
        self.control_calls.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        if state.is_null() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "no-op signal-aware probe state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[derive(Clone, Copy)]
enum SignalAllocationKind {
    Data,
    Control,
}

fn endpoint(node_id: &str, port: &str) -> PortEndpoint {
    PortEndpoint::new(node_id, port).unwrap()
}

fn table_batch(value: i64) -> Batch {
    let record_batch = RecordBatch::try_from_iter(vec![(
        "value",
        Arc::new(Int64Array::from(vec![value])) as ArrayRef,
    )])
    .unwrap();
    Batch::table(vec![record_batch], BatchMetadata::default()).unwrap()
}

fn single_node_plan(name: &str) -> ExecutionPlan {
    PipelineBuilder::new(name)
        .unwrap()
        .add_node("aware", Box::new(FixtureOperator::new()))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn resolve_signal_allocation_output(output: impl AsRef<std::path::Path>) -> std::path::PathBuf {
    let output = output.as_ref();
    if output.is_absolute() {
        return output.to_path_buf();
    }
    assert!(
        !output
            .components()
            .any(|component| component == std::path::Component::ParentDir),
        "relative signal allocation output cannot contain parent components"
    );

    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    if output.starts_with("target") {
        workspace.join(output)
    } else {
        workspace.join("target").join(output)
    }
}

fn write_signal_allocation_reference(kind: SignalAllocationKind) {
    let output = std::env::var_os("SIGNAL_ALLOCATION_OUTPUT");
    let (warmup_dispatches, measured_dispatches) =
        output.as_ref().map_or((0_u64, 1_u64), |_| (1_000, 10_000));
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let control_calls = Arc::new(AtomicUsize::new(0));
    let (allocation, completed_dispatches) = match kind {
        SignalAllocationKind::Data => measure_signal_aware_data(
            &runtime,
            warmup_dispatches,
            measured_dispatches,
            Arc::clone(&control_calls),
        ),
        SignalAllocationKind::Control => measure_signal_aware_control(
            &runtime,
            warmup_dispatches,
            measured_dispatches,
            &control_calls,
        ),
    };
    assert_eq!(completed_dispatches, measured_dispatches);

    if let Some(output) = output {
        let name = match kind {
            SignalAllocationKind::Data => "signal_aware_data",
            SignalAllocationKind::Control => "signal_aware_control",
        };
        let report = json!({
            "name": name,
            "warmup_dispatches": warmup_dispatches,
            "requested_dispatches": measured_dispatches,
            "completed_dispatches": completed_dispatches,
            "control_handler_calls": control_calls.load(Ordering::SeqCst),
            "current_thread_runtime": true,
            "raw": {
                "count_total": allocation.count_total,
                "count_current": allocation.count_current,
                "count_max": allocation.count_max,
                "bytes_total": allocation.bytes_total,
                "bytes_current": allocation.bytes_current,
                "bytes_max": allocation.bytes_max,
            },
        });
        let output = resolve_signal_allocation_output(output);
        std::fs::create_dir_all(output.parent().unwrap()).unwrap();
        std::fs::write(output, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    }
}

fn measure_signal_aware_data(
    runtime: &tokio::runtime::Runtime,
    warmup_dispatches: u64,
    measured_dispatches: u64,
    control_calls: Arc<AtomicUsize>,
) -> (AllocationInfo, u64) {
    let plan = single_node_plan("signal-aware-data-allocation");
    runtime.block_on(async {
        *plan.nodes[0].operator.lock().await =
            CompiledOperator::SignalAware(Box::new(NoopSignalAwareProbe { control_calls }));
    });
    let input = table_batch(42);
    let input_column = Arc::clone(input.table_payload().unwrap().batches()[0].column(0));
    runtime.block_on(async {
        for _ in 0..warmup_dispatches {
            dispatch_signal_aware_data(&plan, &input, &input_column).await;
        }
    });
    measure(|| {});
    let mut completed = 0_u64;
    let allocation = measure(|| {
        runtime.block_on(async {
            for _ in 0..measured_dispatches {
                dispatch_signal_aware_data(&plan, &input, &input_column).await;
                completed += 1;
            }
        });
    });
    (allocation, completed)
}

async fn dispatch_signal_aware_data(plan: &ExecutionPlan, input: &Batch, input_column: &ArrayRef) {
    let result = plan
        .execute(
            BTreeMap::from([("input".into(), input.clone())]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();
    let output_column = result.outputs["output"].table_payload().unwrap().batches()[0].column(0);
    assert!(Arc::ptr_eq(output_column, input_column));
}

fn measure_signal_aware_control(
    runtime: &tokio::runtime::Runtime,
    warmup_dispatches: u64,
    measured_dispatches: u64,
    control_calls: &Arc<AtomicUsize>,
) -> (AllocationInfo, u64) {
    let plan = PipelineBuilder::new("signal-aware-control-allocation")
        .unwrap()
        .add_node("aware", Box::new(FixtureOperator::new()))
        .unwrap()
        .add_node("successor", Box::new(FixtureOperator::new()))
        .unwrap()
        .connect(Edge::new(
            endpoint("aware", "output"),
            endpoint("successor", "input"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    runtime.block_on(async {
        *plan.nodes[0].operator.lock().await =
            CompiledOperator::SignalAware(Box::new(NoopSignalAwareProbe {
                control_calls: Arc::clone(control_calls),
            }));
        for _ in 0..warmup_dispatches {
            plan.dispatch_control(
                "input",
                ControlMarker::watermark(),
                ExecutionOptions::default(),
            )
            .await
            .unwrap();
        }
    });
    measure(|| {});
    let mut completed = 0_u64;
    let allocation = measure(|| {
        runtime.block_on(async {
            for _ in 0..measured_dispatches {
                plan.dispatch_control(
                    "input",
                    ControlMarker::watermark(),
                    ExecutionOptions::default(),
                )
                .await
                .unwrap();
                completed += 1;
            }
        });
    });
    assert_eq!(
        control_calls.load(Ordering::SeqCst),
        usize::try_from(warmup_dispatches + measured_dispatches).unwrap()
    );
    (allocation, completed)
}

#[test]
fn signal_aware_data_allocation_reference() {
    write_signal_allocation_reference(SignalAllocationKind::Data);
}

#[test]
fn signal_aware_control_allocation_reference() {
    write_signal_allocation_reference(SignalAllocationKind::Control);
}

#[test]
fn signal_allocation_relative_output_stays_under_workspace_target() {
    let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let expected = workspace.join("target/signal-allocation/reference.json");

    assert_eq!(
        resolve_signal_allocation_output("signal-allocation/reference.json"),
        expected
    );
    assert_eq!(
        resolve_signal_allocation_output("target/signal-allocation/reference.json"),
        expected
    );
}

#[test]
#[should_panic(expected = "relative signal allocation output cannot contain parent components")]
fn signal_allocation_relative_output_rejects_parent_components() {
    resolve_signal_allocation_output("../reference.json");
}
