use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex as StdMutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

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
    Batch, BatchKind, BatchMetadata, CalcFlowError, CancellationToken, DataFusionRuntime, JsonMap,
    Operator, OperatorContext, Port, Result, RunContext, UdfRegistry,
    operator::SignalAwareOperator,
    runtime::{ControlMarker, SharedControlMarker},
};

struct TopologyProbe {
    name: String,
    input_ports: Vec<Port>,
    output_ports: Vec<Port>,
    snapshots: Arc<AtomicUsize>,
}

impl TopologyProbe {
    fn new(name: &str, inputs: &[&str], snapshots: Arc<AtomicUsize>) -> Self {
        let inputs = inputs
            .iter()
            .map(|input| (*input, true))
            .collect::<Vec<_>>();
        Self::new_with_requirements(name, &inputs, snapshots)
    }

    fn new_with_requirements(
        name: &str,
        inputs: &[(&str, bool)],
        snapshots: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            name: name.into(),
            input_ports: inputs
                .iter()
                .map(|(input, required)| {
                    Port::new(input, BatchKind::Table, *required, None).unwrap()
                })
                .collect(),
            output_ports: vec![Port::new("output", BatchKind::Table, true, None).unwrap()],
            snapshots,
        }
    }
}

#[async_trait]
impl Operator for TopologyProbe {
    fn name(&self) -> &str {
        &self.name
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

    fn snapshot(&self) -> Result<Value> {
        self.snapshots.fetch_add(1, Ordering::SeqCst);
        Ok(Value::Null)
    }
}

struct SignalAwareProbe {
    handler_calls: Arc<AtomicUsize>,
    handled_kinds: Arc<StdMutex<Vec<String>>>,
}

struct OrderedSignalAwareProbe {
    events: Arc<StdMutex<Vec<String>>>,
}

struct FailingSignalAwareProbe;

struct StatefulSignalAwareProbe {
    state: usize,
    fail: Arc<AtomicBool>,
    observed: Arc<StdMutex<Vec<String>>>,
}

enum CancellationBehavior {
    ReturnCancelled,
    CancelThenFail(CancellationToken),
}

struct CancellationSignalAwareProbe {
    behavior: CancellationBehavior,
}

#[async_trait]
impl SignalAwareOperator for FailingSignalAwareProbe {
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
        Err(CalcFlowError::Format {
            message: "injected control failure".into(),
        })
    }

    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        if state.is_null() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "failing signal-aware probe state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl SignalAwareOperator for StatefulSignalAwareProbe {
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
        marker: &ControlMarker,
        _context: &RunContext,
    ) -> Result<()> {
        self.observed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(control_label(marker));
        self.state += 1;
        if self.fail.load(Ordering::SeqCst) {
            Err(CalcFlowError::Format {
                message: "injected stateful control failure".into(),
            })
        } else {
            Ok(())
        }
    }

    fn snapshot(&self) -> Result<Value> {
        Ok(json!(self.state))
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        self.state = state
            .as_u64()
            .and_then(|state| usize::try_from(state).ok())
            .ok_or_else(|| CalcFlowError::Format {
                message: "stateful signal-aware probe state must be an unsigned integer".into(),
            })?;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = 0;
        Ok(())
    }
}

#[async_trait]
impl SignalAwareOperator for CancellationSignalAwareProbe {
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
        context: &RunContext,
    ) -> Result<()> {
        match &self.behavior {
            CancellationBehavior::ReturnCancelled => Err(CalcFlowError::Cancelled {
                run_id: context.run_id().into(),
            }),
            CancellationBehavior::CancelThenFail(cancellation) => {
                cancellation.cancel();
                Err(CalcFlowError::Format {
                    message: "handler failed after cancellation".into(),
                })
            }
        }
    }

    fn snapshot(&self) -> Result<Value> {
        Ok(Value::Null)
    }

    fn restore(&mut self, state: &Value) -> Result<()> {
        if state.is_null() {
            Ok(())
        } else {
            Err(CalcFlowError::Format {
                message: "cancellation signal-aware probe state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl SignalAwareOperator for OrderedSignalAwareProbe {
    async fn process_data(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        _run: &RunContext,
        _datafusion: Option<&DataFusionRuntime>,
    ) -> Result<BTreeMap<String, Batch>> {
        let batch = inputs.values().next().unwrap();
        let values = batch.table_payload()?.batches()[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(format!("Data({})", values.value(0)));
        Ok(BTreeMap::from([("output".into(), batch.clone())]))
    }

    async fn handle_control(
        &mut self,
        marker: &ControlMarker,
        _context: &RunContext,
    ) -> Result<()> {
        self.events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(control_label(marker));
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
                message: "ordered signal-aware probe state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
}

#[async_trait]
impl SignalAwareOperator for SignalAwareProbe {
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
        marker: &ControlMarker,
        _context: &RunContext,
    ) -> Result<()> {
        self.handler_calls.fetch_add(1, Ordering::SeqCst);
        self.handled_kinds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(format!("{:?}", marker.kind()));
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
                message: "signal-aware probe state must be null".into(),
            })
        }
    }

    fn reset(&mut self) -> Result<()> {
        Ok(())
    }
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

fn control_label(marker: &ControlMarker) -> String {
    format!("{:?}({:?})", marker.kind(), marker.occurrence())
}

async fn assert_control_rejected_before_snapshot(
    plan: &ExecutionPlan,
    input: &str,
    snapshots: &AtomicUsize,
    expected_node: &str,
) -> String {
    let mut observed = Vec::new();
    let error = plan
        .dispatch_control_observed(
            input,
            ControlMarker::watermark(),
            ExecutionOptions::default(),
            |target, _ingress, _marker| observed.push(target.clone()),
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "control_input"
    ));
    assert!(error.to_string().contains(expected_node));
    assert_eq!(snapshots.load(Ordering::SeqCst), 0);
    assert!(observed.is_empty());
    error.to_string()
}

fn single_node_plan(name: &str) -> ExecutionPlan {
    PipelineBuilder::new(name)
        .unwrap()
        .add_node(
            "aware",
            Box::new(TopologyProbe::new(
                "aware",
                &["input"],
                Arc::new(AtomicUsize::new(0)),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap()
}

#[tokio::test]
async fn runtime_envelope_rejects_two_connected_inputs_before_snapshot() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("two-connected-inputs")
        .unwrap()
        .add_node(
            "source_a",
            Box::new(TopologyProbe::new(
                "source_a",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "source_b",
            Box::new(TopologyProbe::new(
                "source_b",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "merge",
            Box::new(TopologyProbe::new(
                "merge",
                &["left", "right"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("source_a", "output"),
            endpoint("merge", "left"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("source_b", "output"),
            endpoint("merge", "right"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let error = plan
        .dispatch_control(
            "source_a.input",
            ControlMarker::watermark(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "control_input"
    ));
    assert!(error.to_string().contains("merge"));
    assert_eq!(snapshots.load(Ordering::SeqCst), 0);

    plan.execute(
        BTreeMap::from([
            ("source_a.input".into(), table_batch(1)),
            ("source_b.input".into(), table_batch(2)),
        ]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn runtime_envelope_rejects_connected_plus_external_before_snapshot() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("connected-plus-external")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "merge",
            Box::new(TopologyProbe::new(
                "merge",
                &["left", "right"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("merge", "left"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_control_rejected_before_snapshot(&plan, "input", &snapshots, "merge").await;
}

#[tokio::test]
async fn runtime_envelope_rejects_diamond_before_snapshot() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("diamond")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "left",
            Box::new(TopologyProbe::new(
                "left",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "right",
            Box::new(TopologyProbe::new(
                "right",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "merge",
            Box::new(TopologyProbe::new(
                "merge",
                &["left", "right"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("left", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("right", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("left", "output"),
            endpoint("merge", "left"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("right", "output"),
            endpoint("merge", "right"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_control_rejected_before_snapshot(&plan, "input", &snapshots, "merge").await;
}

#[tokio::test]
async fn runtime_envelope_rejects_unique_active_path_into_multi_input_node() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("unique-active-path")
        .unwrap()
        .add_node(
            "active",
            Box::new(TopologyProbe::new(
                "active",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "active_middle",
            Box::new(TopologyProbe::new(
                "active_middle",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "inactive",
            Box::new(TopologyProbe::new(
                "inactive",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "merge",
            Box::new(TopologyProbe::new(
                "merge",
                &["active", "inactive"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("active", "output"),
            endpoint("active_middle", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("active_middle", "output"),
            endpoint("merge", "active"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("inactive", "output"),
            endpoint("merge", "inactive"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_control_rejected_before_snapshot(&plan, "active.input", &snapshots, "merge").await;
}

#[tokio::test]
async fn runtime_envelope_rejects_unconnected_optional_second_input() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("optional-second-input")
        .unwrap()
        .add_node(
            "multi",
            Box::new(TopologyProbe::new_with_requirements(
                "multi",
                &[("first", true), ("second", false)],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_control_rejected_before_snapshot(&plan, "first", &snapshots, "multi").await;
}

#[tokio::test]
async fn runtime_envelope_rejects_two_target_ports_from_same_source() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("same-source-two-targets")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "merge",
            Box::new(TopologyProbe::new(
                "merge",
                &["left", "right"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("merge", "left"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("merge", "right"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    assert_control_rejected_before_snapshot(&plan, "input", &snapshots, "merge").await;
}

#[tokio::test]
async fn runtime_envelope_selects_first_conflict_in_topological_order() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("stable-first-conflict")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "a_conflict",
            Box::new(TopologyProbe::new(
                "a_conflict",
                &["right", "left"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "b_conflict",
            Box::new(TopologyProbe::new(
                "b_conflict",
                &["left", "right"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("a_conflict", "left"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("b_conflict", "left"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let message =
        assert_control_rejected_before_snapshot(&plan, "input", &snapshots, "a_conflict").await;
    assert!(!message.contains("b_conflict"));
    assert!(message.find("a_conflict.left") < message.find("a_conflict.right"));
}

#[tokio::test]
async fn runtime_envelope_allows_disjoint_single_input_routes() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("disjoint-routes")
        .unwrap()
        .add_node(
            "source_a",
            Box::new(TopologyProbe::new(
                "source_a",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "source_b",
            Box::new(TopologyProbe::new(
                "source_b",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    plan.dispatch_control(
        "source_a.input",
        ControlMarker::watermark(),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();
}

#[tokio::test]
async fn runtime_envelope_transparent_and_aware_share_runtime_forwarding() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let handler_calls = Arc::new(AtomicUsize::new(0));
    let handled_kinds = Arc::new(StdMutex::new(Vec::new()));
    let plan = PipelineBuilder::new("transparent-aware")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "aware",
            Box::new(TopologyProbe::new(
                "aware",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "tail",
            Box::new(TopologyProbe::new(
                "tail",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("aware", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("aware", "output"),
            endpoint("tail", "input"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let aware = plan
        .nodes
        .iter()
        .find(|node| node.node_id == "aware")
        .unwrap();
    *aware.operator.lock().await = CompiledOperator::SignalAware(Box::new(SignalAwareProbe {
        handler_calls: Arc::clone(&handler_calls),
        handled_kinds: Arc::clone(&handled_kinds),
    }));
    let mut observed = Vec::new();

    plan.dispatch_control_observed(
        "input",
        ControlMarker::watermark(),
        ExecutionOptions::default(),
        |target, _ingress, _marker| observed.push(target.node_id.clone()),
    )
    .await
    .unwrap();

    assert_eq!(observed, ["root", "aware", "tail"]);
    assert_eq!(handler_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *handled_kinds
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        ["Watermark"]
    );
}

#[tokio::test]
async fn runtime_envelope_single_chain_preserves_occurrence_order() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let events = Arc::new(StdMutex::new(Vec::new()));
    let plan = PipelineBuilder::new("ordered-chain")
        .unwrap()
        .add_node(
            "aware",
            Box::new(TopologyProbe::new(
                "aware",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    *plan.nodes[0].operator.lock().await =
        CompiledOperator::SignalAware(Box::new(OrderedSignalAwareProbe {
            events: Arc::clone(&events),
        }));
    let watermark = ControlMarker::watermark();
    let watermark_label = control_label(&watermark);
    let epoch = ControlMarker::epoch();
    let epoch_label = control_label(&epoch);

    plan.execute(
        BTreeMap::from([("input".into(), table_batch(1))]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();
    plan.dispatch_control("input", watermark, ExecutionOptions::default())
        .await
        .unwrap();
    plan.execute(
        BTreeMap::from([("input".into(), table_batch(2))]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();
    plan.dispatch_control("input", epoch, ExecutionOptions::default())
        .await
        .unwrap();
    plan.execute(
        BTreeMap::from([("input".into(), table_batch(3))]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();

    assert_eq!(
        *events
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [
            "Data(1)".to_string(),
            watermark_label,
            "Data(2)".to_string(),
            epoch_label,
            "Data(3)".to_string(),
        ]
    );
}

#[tokio::test]
async fn runtime_envelope_fan_out_observes_once_per_target_consumption() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let mut builder = PipelineBuilder::new("fan-out-observation")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap();
    for branch in ["branch_a", "branch_b", "branch_c"] {
        builder = builder
            .add_node(
                branch,
                Box::new(TopologyProbe::new(
                    branch,
                    &["input"],
                    Arc::clone(&snapshots),
                )),
            )
            .unwrap()
            .connect(Edge::new(
                endpoint("root", "output"),
                endpoint(branch, "input"),
            ))
            .unwrap();
    }
    let plan = builder.compile(&UdfRegistry::new().snapshot()).unwrap();
    let root = plan
        .nodes
        .iter()
        .find(|node| node.node_id == "root")
        .unwrap();
    assert_eq!(
        root.outbound["output"]
            .iter()
            .map(|target| target.node_id.as_str())
            .collect::<Vec<_>>(),
        ["branch_a", "branch_b", "branch_c"]
    );
    let mut observed = Vec::<(String, SharedControlMarker)>::new();

    plan.dispatch_control_observed(
        "input",
        ControlMarker::watermark(),
        ExecutionOptions::default(),
        |target, _ingress, marker| observed.push((target.node_id.clone(), marker.clone())),
    )
    .await
    .unwrap();

    assert_eq!(
        observed
            .iter()
            .map(|(node_id, _)| node_id.as_str())
            .collect::<Vec<_>>(),
        ["root", "branch_a", "branch_b", "branch_c"]
    );
    assert!(
        observed
            .iter()
            .all(|(_, marker)| marker.shares_allocation(&observed[0].1))
    );
}

#[tokio::test]
async fn runtime_envelope_fan_out_failure_discards_unvisited_sibling() {
    let snapshots = Arc::new(AtomicUsize::new(0));
    let plan = PipelineBuilder::new("fan-out-failure")
        .unwrap()
        .add_node(
            "root",
            Box::new(TopologyProbe::new(
                "root",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "a_fail",
            Box::new(TopologyProbe::new(
                "a_fail",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "a_successor",
            Box::new(TopologyProbe::new(
                "a_successor",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .add_node(
            "b_sibling",
            Box::new(TopologyProbe::new(
                "b_sibling",
                &["input"],
                Arc::clone(&snapshots),
            )),
        )
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("a_fail", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("a_fail", "output"),
            endpoint("a_successor", "input"),
        ))
        .unwrap()
        .connect(Edge::new(
            endpoint("root", "output"),
            endpoint("b_sibling", "input"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let failing = plan
        .nodes
        .iter()
        .find(|node| node.node_id == "a_fail")
        .unwrap();
    *failing.operator.lock().await =
        CompiledOperator::SignalAware(Box::new(FailingSignalAwareProbe));
    let mut observed = Vec::new();

    let error = plan
        .dispatch_control_observed(
            "input",
            ControlMarker::watermark(),
            ExecutionOptions::default(),
            |target, _ingress, _marker| observed.push(target.node_id.clone()),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        CalcFlowError::Operator { ref node_id, .. } if node_id == "a_fail"
    ));
    assert_eq!(observed, ["root", "a_fail"]);
}

#[tokio::test]
async fn runtime_envelope_handler_error_maps_node_kind_and_occurrence() {
    let plan = single_node_plan("handler-error-mapping");
    *plan.nodes[0].operator.lock().await =
        CompiledOperator::SignalAware(Box::new(FailingSignalAwareProbe));

    for marker in [ControlMarker::watermark(), ControlMarker::epoch()] {
        let expected_kind = format!("{:?}", marker.kind());
        let expected_occurrence = format!("{:?}", marker.occurrence());
        let error = plan
            .dispatch_control("input", marker, ExecutionOptions::default())
            .await
            .unwrap_err();

        assert!(matches!(
            error,
            CalcFlowError::Operator { ref node_id, .. } if node_id == "aware"
        ));
        let message = error.to_string();
        assert!(message.contains(&expected_kind));
        assert!(message.contains(&expected_occurrence));
        assert!(message.contains("External"));
    }
}

#[tokio::test]
async fn runtime_envelope_handler_error_rolls_back_and_retry_is_clean() {
    let plan = single_node_plan("handler-rollback-retry");
    let fail = Arc::new(AtomicBool::new(true));
    let observed = Arc::new(StdMutex::new(Vec::new()));
    *plan.nodes[0].operator.lock().await =
        CompiledOperator::SignalAware(Box::new(StatefulSignalAwareProbe {
            state: 0,
            fail: Arc::clone(&fail),
            observed: Arc::clone(&observed),
        }));
    let failed = ControlMarker::watermark();
    let failed_label = control_label(&failed);

    plan.dispatch_control("input", failed, ExecutionOptions::default())
        .await
        .unwrap_err();
    assert_eq!(plan.snapshot().await.unwrap()["aware"], json!(0));

    fail.store(false, Ordering::SeqCst);
    let retry = ControlMarker::watermark();
    let retry_label = control_label(&retry);
    plan.dispatch_control("input", retry, ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.snapshot().await.unwrap()["aware"], json!(1));
    plan.execute(
        BTreeMap::from([("input".into(), table_batch(7))]),
        ExecutionOptions::default(),
    )
    .await
    .unwrap();

    assert_ne!(failed_label, retry_label);
    assert_eq!(
        *observed
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner),
        [failed_label, retry_label]
    );
}

#[tokio::test]
async fn runtime_envelope_post_handler_cancellation_wins() {
    let direct = single_node_plan("handler-returned-cancelled");
    *direct.nodes[0].operator.lock().await =
        CompiledOperator::SignalAware(Box::new(CancellationSignalAwareProbe {
            behavior: CancellationBehavior::ReturnCancelled,
        }));
    let direct_error = direct
        .dispatch_control(
            "input",
            ControlMarker::watermark(),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();
    assert!(matches!(direct_error, CalcFlowError::Cancelled { .. }));

    let post = single_node_plan("post-handler-cancelled");
    let cancellation = CancellationToken::new();
    *post.nodes[0].operator.lock().await =
        CompiledOperator::SignalAware(Box::new(CancellationSignalAwareProbe {
            behavior: CancellationBehavior::CancelThenFail(cancellation.clone()),
        }));
    let post_error = post
        .dispatch_control(
            "input",
            ControlMarker::epoch(),
            ExecutionOptions {
                cancellation,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(post_error, CalcFlowError::Cancelled { .. }));
}
