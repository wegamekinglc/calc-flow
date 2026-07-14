mod support;

use std::{
    collections::BTreeMap,
    sync::{Arc, atomic::AtomicBool},
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchKind, CalcFlowError, CancellationToken, Edge, ExecutionOptions, ExpressionOperator,
    JsonMap, Operator, OperatorContext, PipelineBuilder, Port, PortEndpoint, UdfKind, UdfReference,
    UdfRegistry,
};
use datafusion::{
    arrow::datatypes::{DataType, Field},
    common::ScalarValue,
    logical_expr::{ColumnarValue, Volatility, create_udf},
};
use serde_json::{Value, json};
use support::{Action, Probe, TestOperator, int_batch, table_port, untyped_table_port};

fn endpoint(node: &str, port: &str) -> PortEndpoint {
    PortEndpoint::new(node, port).unwrap()
}

fn edge(source: &str, target: &str) -> Edge {
    Edge::new(endpoint(source, "output"), endpoint(target, "input"))
}

fn one_node(action: Action, probe: Arc<Probe>) -> calc_flow::ExecutionPlan {
    PipelineBuilder::new("execute")
        .unwrap()
        .add_node(
            "node",
            Box::new(TestOperator::transform("node", action, probe)),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap()
}

fn inputs() -> BTreeMap<String, Batch> {
    BTreeMap::from([("input".into(), int_batch(&[1, 2, 3]))])
}

#[tokio::test]
async fn dropped_execute_restores_mutated_state_before_the_next_public_call() {
    let probe = Arc::new(Probe::default());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let plan = PipelineBuilder::new("cancelled execute")
        .unwrap()
        .add_node(
            "node",
            Box::new(
                TestOperator::transform(
                    "node",
                    Action::GateOncePass {
                        started: Arc::clone(&started),
                        release,
                        pending: Arc::new(AtomicBool::new(true)),
                    },
                    Arc::clone(&probe),
                )
                .stateful(),
            ),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let mut cancelled = Box::pin(plan.execute(inputs(), ExecutionOptions::default()));
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("operator gate did not suspend execute: {result:?}"),
    }
    drop(cancelled);

    assert_eq!(plan.snapshot().await.unwrap()["node"]["state"], json!(0));
    plan.execute(inputs(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(plan.snapshot().await.unwrap()["node"]["state"], json!(1));
    assert_eq!(probe.calls(), 2);
}

#[tokio::test]
async fn reset_recovers_an_unrestorable_dropped_direct_multi_input_operation() {
    let first = Arc::new(Probe::default());
    let second = Arc::new(Probe::default());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let plan = PipelineBuilder::new("forced direct reset")
        .unwrap()
        .add_node(
            "first",
            Box::new(
                TestOperator::transform(
                    "first",
                    Action::GateOncePass {
                        started: Arc::clone(&started),
                        release,
                        pending: Arc::new(AtomicBool::new(true)),
                    },
                    Arc::clone(&first),
                )
                .stateful()
                .failing_restore(),
            ),
        )
        .unwrap()
        .add_node(
            "second",
            Box::new(
                TestOperator::transform("second", Action::Pass, Arc::clone(&second)).stateful(),
            ),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let inputs = plan
        .external_inputs()
        .keys()
        .map(|name| (name.clone(), int_batch(&[1])))
        .collect();

    let mut cancelled = Box::pin(plan.execute(inputs, ExecutionOptions::default()));
    tokio::select! {
        () = started.notified() => {}
        result = &mut cancelled => panic!("operator gate did not suspend execute: {result:?}"),
    }
    drop(cancelled);

    plan.reset().await.unwrap();
    let state = plan.snapshot().await.unwrap();
    assert_eq!(state["first"]["state"], json!(0));
    assert_eq!(state["second"]["state"], json!(0));
    assert_eq!(first.resets(), 1);
    assert_eq!(second.resets(), 1);
}

struct DriftingOutputOperator {
    input_ports: Vec<Port>,
    output_ports: Vec<Port>,
}

impl DriftingOutputOperator {
    fn new() -> Self {
        Self {
            input_ports: vec![table_port("input", true)],
            output_ports: vec![table_port("output", true)],
        }
    }
}

#[async_trait]
impl Operator for DriftingOutputOperator {
    fn name(&self) -> &'static str {
        "drifting-output"
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
        _inputs: &BTreeMap<String, Batch>,
        _context: &OperatorContext<'_>,
    ) -> calc_flow::Result<BTreeMap<String, Batch>> {
        self.output_ports = vec![Port::new(
            "output",
            BatchKind::Table,
            true,
            Some(vec![Field::new("value", DataType::Utf8, false)]),
        )?];
        Ok(BTreeMap::from([(
            "output".into(),
            support::string_batch(&["drifted"]),
        )]))
    }
}

#[tokio::test]
async fn run_result_contains_named_outputs_timings_metrics_and_metadata() {
    let plan = PipelineBuilder::new("calculation")
        .unwrap()
        .add_node(
            "calculate",
            Box::new(
                ExpressionOperator::new("calculate", "plus_one = value + 1", vec![], None, vec![])
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let result = plan
        .execute(inputs(), ExecutionOptions::default())
        .await
        .unwrap();

    assert_eq!(result.outputs["output"].num_rows(), 3);
    assert_eq!(result.node_timings["calculate"].input_rows["input"], 3);
    assert_eq!(result.node_timings["calculate"].output_rows["output"], 3);
    assert!(result.node_timings["calculate"].duration_ns > 0);
    assert_eq!(result.datafusion_metrics.len(), 1);
    assert_eq!(
        result.datafusion_metrics[0].node_id.as_deref(),
        Some("calculate")
    );
    assert!(!result.metadata.run_id.is_empty());
    assert_eq!(result.metadata.pipeline_name, "calculation");
    assert_eq!(result.metadata.pipeline_fingerprint, plan.fingerprint());
}

#[tokio::test]
async fn external_inputs_reject_unknown_missing_kind_and_schema_without_calling_nodes() {
    let probe = Arc::new(Probe::default());
    let plan = one_node(Action::Pass, Arc::clone(&probe));
    for invalid in [
        BTreeMap::new(),
        BTreeMap::from([("unknown".into(), int_batch(&[1]))]),
    ] {
        assert!(
            plan.execute(invalid, ExecutionOptions::default())
                .await
                .is_err()
        );
    }
    let schema_error = plan
        .execute(
            BTreeMap::from([("input".into(), support::string_batch(&["bad"]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err()
        .to_string();
    assert!(schema_error.contains("node.input"), "{schema_error}");
    assert_eq!(probe.calls(), 0);
}

#[tokio::test]
async fn optional_external_input_may_be_absent_but_unknown_names_are_rejected() {
    let probe = Arc::new(Probe::default());
    let operator = TestOperator::ports(
        "optional",
        vec![untyped_table_port("input", false)],
        vec![untyped_table_port("output", false)],
        Action::MissingOutput,
        Arc::clone(&probe),
    );
    let plan = PipelineBuilder::new("optional")
        .unwrap()
        .add_node("optional", Box::new(operator))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let result = plan
        .execute(BTreeMap::new(), ExecutionOptions::default())
        .await
        .unwrap();
    assert!(result.outputs.is_empty());
    assert!(
        plan.execute(
            BTreeMap::from([("other".into(), int_batch(&[1]))]),
            ExecutionOptions::default(),
        )
        .await
        .is_err()
    );
    assert_eq!(probe.calls(), 1);
}

#[tokio::test]
async fn omitted_same_name_output_does_not_echo_the_external_input() {
    let probe = Arc::new(Probe::default());
    let operator = TestOperator::ports(
        "same-name",
        vec![untyped_table_port("shared", false)],
        vec![untyped_table_port("shared", false)],
        Action::MissingOutput,
        Arc::clone(&probe),
    );
    let plan = PipelineBuilder::new("same-name omitted output")
        .unwrap()
        .add_node("same-name", Box::new(operator))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let result = plan
        .execute(
            BTreeMap::from([("shared".into(), int_batch(&[1, 2, 3]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();

    assert!(result.outputs.is_empty());
    assert_eq!(probe.calls(), 1);
}

#[tokio::test]
async fn connected_same_name_output_must_be_produced_before_routing_downstream() {
    let probe = Arc::new(Probe::default());
    let producer = TestOperator::ports(
        "producer",
        vec![table_port("shared", false)],
        vec![table_port("shared", false)],
        Action::MissingOutput,
        Arc::clone(&probe),
    );
    let consumer = TestOperator::transform("consumer", Action::Pass, Arc::clone(&probe));
    let plan = PipelineBuilder::new("connected same-name output")
        .unwrap()
        .add_node("producer", Box::new(producer))
        .unwrap()
        .add_node("consumer", Box::new(consumer))
        .unwrap()
        .connect(Edge::new(
            endpoint("producer", "shared"),
            endpoint("consumer", "input"),
        ))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let error = plan
        .execute(
            BTreeMap::from([("shared".into(), int_batch(&[1, 2, 3]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();

    assert!(error.to_string().contains("required input consumer.input"));
    assert_eq!(probe.order(), vec!["producer"]);
}

#[tokio::test]
async fn produced_same_name_output_replaces_neither_input_nor_route_identity() {
    let probe = Arc::new(Probe::default());
    let operator = TestOperator::ports(
        "same-name",
        vec![untyped_table_port("shared", true)],
        vec![untyped_table_port("shared", true)],
        Action::Pass,
        Arc::clone(&probe),
    );
    let plan = PipelineBuilder::new("same-name produced output")
        .unwrap()
        .add_node("same-name", Box::new(operator))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let result = plan
        .execute(
            BTreeMap::from([("shared".into(), int_batch(&[1, 2, 3]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap();

    assert_eq!(result.outputs["shared"].num_rows(), 3);
    assert_eq!(probe.calls(), 1);
}

#[tokio::test]
async fn execution_validates_outputs_against_compile_time_port_snapshot() {
    let plan = PipelineBuilder::new("compile-time ports")
        .unwrap()
        .add_node("drift", Box::new(DriftingOutputOperator::new()))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let error = plan
        .execute(inputs(), ExecutionOptions::default())
        .await
        .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("output drift.output schema mismatch")
    );
}

#[tokio::test]
async fn execution_is_topological_and_validates_required_downstream_inputs() {
    let probe = Arc::new(Probe::default());
    let producer = TestOperator::ports(
        "producer",
        vec![table_port("input", true)],
        vec![table_port("output", false)],
        Action::MissingOutput,
        Arc::clone(&probe),
    );
    let consumer = TestOperator::transform("consumer", Action::Pass, Arc::clone(&probe));
    let plan = PipelineBuilder::new("topological")
        .unwrap()
        .add_node("consumer", Box::new(consumer))
        .unwrap()
        .add_node("producer", Box::new(producer))
        .unwrap()
        .connect(edge("producer", "consumer"))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let error = plan
        .execute(inputs(), ExecutionOptions::default())
        .await
        .unwrap_err();
    assert!(error.to_string().contains("required input consumer.input"));
    assert_eq!(probe.order(), vec!["producer"]);
}

#[tokio::test]
async fn missing_unknown_and_schema_invalid_outputs_are_rejected_and_rolled_back() {
    for action in [
        Action::MissingOutput,
        Action::UnknownOutput,
        Action::WrongSchema,
    ] {
        let probe = Arc::new(Probe::default());
        let operator = TestOperator::transform("node", action, Arc::clone(&probe)).stateful();
        let plan = PipelineBuilder::new("invalid output")
            .unwrap()
            .add_node("node", Box::new(operator))
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap();
        let before = plan.snapshot().await.unwrap();
        assert!(
            plan.execute(inputs(), ExecutionOptions::default())
                .await
                .is_err()
        );
        assert_eq!(plan.snapshot().await.unwrap(), before);
        assert_eq!(probe.restores(), 1);
    }
}

#[tokio::test]
async fn operator_and_datafusion_failures_restore_every_node() {
    let first_probe = Arc::new(Probe::default());
    let second_probe = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("rollback")
        .unwrap()
        .add_node(
            "first",
            Box::new(
                TestOperator::transform("first", Action::Pass, Arc::clone(&first_probe)).stateful(),
            ),
        )
        .unwrap()
        .add_node(
            "second",
            Box::new(
                TestOperator::transform("second", Action::Fail("boom"), Arc::clone(&second_probe))
                    .stateful(),
            ),
        )
        .unwrap()
        .connect(edge("first", "second"))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let before = plan.snapshot().await.unwrap();
    assert!(
        plan.execute(inputs(), ExecutionOptions::default())
            .await
            .is_err()
    );
    assert_eq!(plan.snapshot().await.unwrap(), before);
    assert_eq!(first_probe.restores(), 1);
    assert_eq!(second_probe.restores(), 1);

    let datafusion = PipelineBuilder::new("runtime rollback")
        .unwrap()
        .add_node(
            "bad_query",
            Box::new(
                ExpressionOperator::new("bad_query", "result = missing + 1", vec![], None, vec![])
                    .unwrap(),
            ),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let before = datafusion.snapshot().await.unwrap();
    assert!(
        datafusion
            .execute(inputs(), ExecutionOptions::default())
            .await
            .is_err()
    );
    assert_eq!(datafusion.snapshot().await.unwrap(), before);
}

#[tokio::test]
async fn cancellation_is_checked_immediately_before_and_after_each_operator_call() {
    let before_probe = Arc::new(Probe::default());
    let before_plan = one_node(Action::Pass, Arc::clone(&before_probe));
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let error = before_plan
        .execute(
            inputs(),
            ExecutionOptions {
                cancellation: cancelled,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(before_probe.calls(), 0);

    let token = CancellationToken::new();
    let first_probe = Arc::new(Probe::default());
    let second_probe = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("post cancellation")
        .unwrap()
        .add_node(
            "first",
            Box::new(
                TestOperator::transform(
                    "first",
                    Action::CancelAndPass(token.clone()),
                    Arc::clone(&first_probe),
                )
                .stateful(),
            ),
        )
        .unwrap()
        .add_node(
            "second",
            Box::new(TestOperator::transform(
                "second",
                Action::Pass,
                Arc::clone(&second_probe),
            )),
        )
        .unwrap()
        .connect(edge("first", "second"))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let before = plan.snapshot().await.unwrap();
    let error = plan
        .execute(
            inputs(),
            ExecutionOptions {
                cancellation: token,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
    assert_eq!(first_probe.calls(), 1);
    assert_eq!(second_probe.calls(), 0);
    assert_eq!(plan.snapshot().await.unwrap(), before);

    let precedence_token = CancellationToken::new();
    let precedence = one_node(
        Action::CancelAndFail(precedence_token.clone(), "operator also failed"),
        Arc::new(Probe::default()),
    );
    let error = precedence
        .execute(
            inputs(),
            ExecutionOptions {
                cancellation: precedence_token,
                ..ExecutionOptions::default()
            },
        )
        .await
        .unwrap_err();
    assert!(matches!(error, CalcFlowError::Cancelled { .. }));
}

#[tokio::test]
async fn restore_rejects_missing_and_extra_node_ids_before_mutating_any_node() {
    let first = Arc::new(Probe::default());
    let second = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("restore validation")
        .unwrap()
        .add_node(
            "first",
            Box::new(TestOperator::transform(
                "first",
                Action::Pass,
                Arc::clone(&first),
            )),
        )
        .unwrap()
        .add_node(
            "second",
            Box::new(TestOperator::transform(
                "second",
                Action::Pass,
                Arc::clone(&second),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    for invalid in [
        BTreeMap::from([("first".into(), json!({"state": 0}))]),
        BTreeMap::from([
            ("first".into(), json!({"state": 0})),
            ("second".into(), json!({"state": 0})),
            ("extra".into(), Value::Null),
        ]),
    ] {
        assert!(matches!(
            plan.restore(&invalid).await,
            Err(CalcFlowError::CheckpointMismatch { .. })
        ));
    }
    assert_eq!(first.restores(), 0);
    assert_eq!(second.restores(), 0);
}

#[tokio::test]
async fn snapshot_restore_and_reset_cover_every_node() {
    let first = Arc::new(Probe::default());
    let second = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("lifecycle")
        .unwrap()
        .add_node(
            "first",
            Box::new(TestOperator::transform(
                "first",
                Action::Pass,
                Arc::clone(&first),
            )),
        )
        .unwrap()
        .add_node(
            "second",
            Box::new(TestOperator::transform(
                "second",
                Action::Pass,
                Arc::clone(&second),
            )),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let state = plan.snapshot().await.unwrap();
    assert_eq!(
        state.keys().cloned().collect::<Vec<_>>(),
        ["first", "second"]
    );
    plan.restore(&state).await.unwrap();
    plan.reset().await.unwrap();
    assert_eq!((first.restores(), first.resets()), (1, 1));
    assert_eq!((second.restores(), second.resets()), (1, 1));
}

#[tokio::test]
async fn rollback_failure_preserves_original_failure_and_attempts_every_restore() {
    let bad = Arc::new(Probe::default());
    let good = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("rollback failure")
        .unwrap()
        .add_node(
            "bad_restore",
            Box::new(
                TestOperator::transform("bad_restore", Action::Pass, Arc::clone(&bad))
                    .stateful()
                    .failing_restore(),
            ),
        )
        .unwrap()
        .add_node(
            "failure",
            Box::new(
                TestOperator::transform(
                    "failure",
                    Action::Fail("original boom"),
                    Arc::clone(&good),
                )
                .stateful(),
            ),
        )
        .unwrap()
        .connect(edge("bad_restore", "failure"))
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();
    let error = plan
        .execute(inputs(), ExecutionOptions::default())
        .await
        .unwrap_err();
    let message = error.to_string();
    assert!(message.contains("original boom"), "{message}");
    assert!(message.contains("restore injected"), "{message}");
    assert_eq!(bad.restores(), 1);
    assert_eq!(good.restores(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_runs_are_serialized_across_the_whole_plan() {
    let probe = Arc::new(Probe::default());
    let plan = Arc::new(
        PipelineBuilder::new("serialized")
            .unwrap()
            .add_node(
                "first",
                Box::new(TestOperator::transform(
                    "first",
                    Action::DelayPass(Duration::from_millis(20)),
                    Arc::clone(&probe),
                )),
            )
            .unwrap()
            .add_node(
                "second",
                Box::new(TestOperator::transform(
                    "second",
                    Action::DelayPass(Duration::from_millis(20)),
                    Arc::clone(&probe),
                )),
            )
            .unwrap()
            .connect(edge("first", "second"))
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let first = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move { plan.execute(inputs(), ExecutionOptions::default()).await }
    });
    let second = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move { plan.execute(inputs(), ExecutionOptions::default()).await }
    });
    first.await.unwrap().unwrap();
    second.await.unwrap().unwrap();
    assert_eq!(probe.max_active(), 1);
    assert_eq!(probe.order(), vec!["first", "second", "first", "second"]);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lifecycle_operations_wait_for_the_whole_run_lock() {
    let first_probe = Arc::new(Probe::default());
    let second_probe = Arc::new(Probe::default());
    let started = Arc::new(tokio::sync::Notify::new());
    let release = Arc::new(tokio::sync::Notify::new());
    let plan = Arc::new(
        PipelineBuilder::new("run lifecycle lock")
            .unwrap()
            .add_node(
                "first",
                Box::new(TestOperator::transform(
                    "first",
                    Action::Pass,
                    Arc::clone(&first_probe),
                )),
            )
            .unwrap()
            .add_node(
                "second",
                Box::new(TestOperator::transform(
                    "second",
                    Action::GatePass {
                        started: Arc::clone(&started),
                        release: Arc::clone(&release),
                    },
                    Arc::clone(&second_probe),
                )),
            )
            .unwrap()
            .connect(edge("first", "second"))
            .unwrap()
            .compile(&UdfRegistry::new().snapshot())
            .unwrap(),
    );
    let run = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move { plan.execute(inputs(), ExecutionOptions::default()).await }
    });
    started.notified().await;
    let reset = tokio::spawn({
        let plan = Arc::clone(&plan);
        async move { plan.reset().await }
    });
    tokio::time::sleep(Duration::from_millis(10)).await;
    assert_eq!(first_probe.resets(), 0);
    assert!(!reset.is_finished());
    release.notify_one();
    run.await.unwrap().unwrap();
    reset.await.unwrap().unwrap();
    assert_eq!(first_probe.resets(), 1);
    assert_eq!(second_probe.resets(), 1);
}

#[tokio::test]
async fn execution_does_not_replace_or_modify_input_batches() {
    let probe = Arc::new(Probe::default());
    let plan = one_node(Action::Pass, probe);
    let original = inputs();
    let before_rows = original["input"].num_rows();
    let before_schema = Arc::clone(original["input"].table_payload().unwrap().schema());
    plan.execute(original.clone(), ExecutionOptions::default())
        .await
        .unwrap();
    assert_eq!(original["input"].num_rows(), before_rows);
    assert!(Arc::ptr_eq(
        original["input"].table_payload().unwrap().schema(),
        &before_schema
    ));
}

#[tokio::test]
async fn invalid_external_input_does_not_touch_operator_lifecycle() {
    let probe = Arc::new(Probe::default());
    let plan = PipelineBuilder::new("invalid before snapshot")
        .unwrap()
        .add_node(
            "node",
            Box::new(
                TestOperator::transform("node", Action::Pass, Arc::clone(&probe))
                    .stateful()
                    .failing_restore(),
            ),
        )
        .unwrap()
        .compile(&UdfRegistry::new().snapshot())
        .unwrap();

    let error = plan
        .execute(
            BTreeMap::from([("input".into(), support::string_batch(&["wrong schema"]))]),
            ExecutionOptions::default(),
        )
        .await
        .unwrap_err();

    assert!(matches!(
        error,
        CalcFlowError::Compile { message } if message.contains("schema mismatch")
    ));
    assert_eq!(probe.calls(), 0);
    assert_eq!(probe.snapshots(), 0);
    assert_eq!(probe.restores(), 0);
}

#[tokio::test]
async fn each_run_registers_only_compile_time_selected_native_udfs() {
    let selected = UdfReference::new("rust", "chosen", "1", UdfKind::DataFusionScalar).unwrap();
    let unselected = UdfReference::new("plugin", "chosen", "2", UdfKind::DataFusionScalar).unwrap();
    let mut registry = UdfRegistry::new();
    for (reference, value) in [(selected.clone(), 7_i64), (unselected, 99_i64)] {
        registry
            .register_datafusion(
                reference,
                Arc::new(create_udf(
                    "chosen",
                    vec![],
                    DataType::Int64,
                    Volatility::Immutable,
                    Arc::new(move |_| Ok(ColumnarValue::Scalar(ScalarValue::Int64(Some(value))))),
                )),
                0,
            )
            .unwrap();
    }
    let plan = PipelineBuilder::new("selected udf")
        .unwrap()
        .add_node(
            "calculate",
            Box::new(
                ExpressionOperator::new(
                    "calculate",
                    "result = chosen()",
                    vec![],
                    None,
                    vec![selected],
                )
                .unwrap(),
            ),
        )
        .unwrap()
        .compile(&registry.snapshot())
        .unwrap();

    for _ in 0..2 {
        let result = plan
            .execute(inputs(), ExecutionOptions::default())
            .await
            .unwrap();
        let table = result.outputs["output"].table_payload().unwrap();
        let values = table.batches()[0]
            .column_by_name("result")
            .unwrap()
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(values.value(0), 7);
    }
}
