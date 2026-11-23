use super::*;
use crate::pipeline::stream::{CompiledStreamOperator, StreamExecutionPlan};
use crate::{
    Batch, CancellationToken, EdgeCollector, Epoch, ExpressionOperator, JsonMap,
    OperatorStateSnapshot, StreamJobContext, StreamOperator, StreamOperatorContext,
    StreamRequirements, UdfRegistry,
};
use datafusion::arrow::{
    array::{Array, Int64Array, UInt64Array},
    compute::take,
    record_batch::RecordBatch,
};

fn compiled(selects: &[&[&str]], filter: Option<&str>) -> StreamExecutionPlan {
    let mut builder = graph(selects);
    if let Some(filter) = filter {
        let NodeOperator::Expression(original) = &builder.nodes["select0"].operator else {
            unreachable!()
        };
        let expression = ExpressionOperator::new(
            "select0",
            "",
            selects[0].iter().map(|value| (*value).into()).collect(),
            Some(filter.into()),
            vec![],
        )
        .unwrap()
        .with_ports(
            original.input_ports()[0].clone(),
            original.output_ports()[0].clone(),
        )
        .unwrap();
        builder.nodes.get_mut("select0").unwrap().operator = NodeOperator::Expression(expression);
    }
    let logical = builder.nodes["asof"].operator.configuration();
    let inputs = builder.nodes["asof"]
        .operator
        .input_ports()
        .iter()
        .map(|port| port.schema().cloned())
        .collect::<Vec<_>>();
    let registry = UdfRegistry::new().snapshot();
    let fingerprint = crate::pipeline::compile_graph(&builder, "stream", &registry)
        .unwrap()
        .fingerprint;
    let mut plan = builder
        .compile_stream(&registry, &StreamRequirements::default())
        .unwrap();
    assert_eq!(plan.fingerprint, fingerprint);
    let join = join(&mut plan);
    assert_eq!(
        join.configuration(),
        logical,
        "physical projection must not enter serialized Join configuration"
    );
    assert_eq!(
        join.input_ports()
            .iter()
            .map(|port| port.schema().cloned())
            .collect::<Vec<_>>(),
        inputs
    );
    plan
}

fn join(plan: &mut StreamExecutionPlan) -> &mut StreamJoinOperator {
    let node = plan
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "asof")
        .unwrap();
    let CompiledStreamOperator::StreamJoin(join) = &mut node.operator else {
        unreachable!()
    };
    join
}

async fn process(
    join: &mut StreamJoinOperator,
    input: Batch,
    side: &str,
    job: &StreamJobContext,
) -> Vec<Batch> {
    let mut collector = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        side,
        input,
        &StreamOperatorContext::new(job, "asof", None),
        &mut collector,
    )
    .await
    .unwrap();
    collector
        .drain("output")
        .into_iter()
        .map(|event| event.as_data().unwrap().clone())
        .collect()
}

async fn captured(
    join: &mut StreamJoinOperator,
    right: &Batch,
    job: &StreamJobContext,
    v1: bool,
) -> OperatorStateSnapshot {
    if v1 {
        join.set_checkpoint_v1_test_producer();
    }
    assert!(process(join, right.clone(), "right", job).await.is_empty());
    join.prepare_checkpoint_async(&StreamOperatorContext::new(job, "asof", None))
        .await
        .unwrap();
    join.checkpoint(Epoch::new(1).unwrap()).unwrap()
}

fn record(batch: &Batch) -> &RecordBatch {
    &batch.table_payload().unwrap().batches()[0]
}

fn left_input(schema: datafusion::arrow::datatypes::SchemaRef) -> Batch {
    let original = super::super::input(schema, false);
    let source = record(&original);
    let indices = UInt64Array::from(vec![0, 0]);
    let columns = vec![
        take(source.column(0).as_ref(), &indices, None).unwrap(),
        take(source.column(1).as_ref(), &indices, None).unwrap(),
        Arc::clone(source.column(2)),
        Arc::clone(source.column(3)),
    ];
    Batch::table(
        vec![RecordBatch::try_new(source.schema(), columns).unwrap()],
        crate::BatchMetadata::default(),
    )
    .unwrap()
}

async fn consumer(
    plan: &mut StreamExecutionPlan,
    index: usize,
    input: Batch,
    job: &StreamJobContext,
) -> Batch {
    let name = format!("select{index}");
    let node = plan
        .nodes
        .iter_mut()
        .find(|node| node.node_id == name)
        .unwrap();
    let mut collector = EdgeCollector::new(node.output_ports.values().cloned().collect());
    node.operator
        .process_data(
            "input",
            input,
            &StreamOperatorContext::new(job, &name, None),
            &mut collector,
        )
        .await
        .unwrap();
    collector.drain("output")[0].as_data().unwrap().clone()
}

async fn continuation(selects: &[&[&str]], filter: Option<&str>, v1: bool) {
    let mut plan = compiled(selects, filter);
    assert_eq!(
        join(&mut plan).output_ports()[0]
            .schema()
            .unwrap()
            .fields()
            .len(),
        if filter.is_some() { 3 } else { 0 }
    );
    let mut reference_selects = selects.to_vec();
    reference_selects.push(&["*"]);
    let mut full = compiled(&reference_selects, filter);
    assert_eq!(
        join(&mut full).output_ports()[0]
            .schema()
            .unwrap()
            .fields()
            .len(),
        8
    );
    let job = StreamJobContext::new(
        1,
        &plan.fingerprint,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let schema = Arc::clone(join(&mut plan).input_ports()[0].schema().unwrap());
    let right = super::super::input(Arc::clone(&schema), true);
    let left = left_input(schema);
    let untouched = record(&left).clone();
    let (snapshot, full_snapshot) = snapshots(&mut plan, &mut full, &right, &job, v1).await;
    assert_eq!(join(&mut plan).status(), join(&mut full).status());
    let mut cold = compiled(selects, filter);
    let migrated = recovery_pair(&mut cold, &mut full, &snapshot, &full_snapshot, &job).await;
    let narrowed = process(join(&mut cold), left.clone(), "left", &job).await;
    let reference = process(join(&mut full), left.clone(), "left", &job).await;
    assert_eq!(narrowed.len(), 1);
    assert_eq!(reference.len(), 1);
    assert_eq!(join(&mut cold).status(), join(&mut full).status());
    assert_eq!(record(&left), &untouched);
    assert_eq!(narrowed[0].num_rows(), 2);
    for index in 0..selects.len() {
        let actual = consumer(&mut cold, index, narrowed[0].clone(), &job).await;
        let expected = consumer(&mut full, index, reference[0].clone(), &job).await;
        assert_eq!(record(&actual), record(&expected));
        assert_eq!(
            actual.num_rows(),
            if filter.is_some() && index == 0 { 1 } else { 2 }
        );
        let value = record(&actual)
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>();
        if let Some(value) = value {
            assert!(!value.is_null(0));
        }
    }
    let pools = [&mut plan, &mut cold, &mut full]
        .map(|plan| join(plan).checkpoint_preload_test_pool().unwrap());
    drop((
        plan,
        cold,
        full,
        snapshot,
        full_snapshot,
        migrated,
        narrowed,
        reference,
        right,
        left,
    ));
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
}

async fn snapshots(
    projected: &mut StreamExecutionPlan,
    full: &mut StreamExecutionPlan,
    right: &Batch,
    job: &StreamJobContext,
    v1: bool,
) -> (OperatorStateSnapshot, OperatorStateSnapshot) {
    let snapshot = captured(join(projected), right, job, false).await;
    let legacy = captured(join(full), right, job, v1).await;
    assert_eq!(snapshot.inline_metadata["layout_version"], 3);
    assert_eq!(
        legacy.inline_metadata["layout_version"],
        if v1 { 1 } else { 2 }
    );
    for field in [
        "spec",
        "next_left_row_id",
        "next_right_row_id",
        "next_output_sequence",
        "metrics",
        "ended",
        "epoch",
    ] {
        assert_eq!(
            snapshot.inline_metadata[field], legacy.inline_metadata[field],
            "logical checkpoint field: {field}"
        );
    }
    (snapshot, legacy)
}

async fn recovery_pair(
    cold: &mut StreamExecutionPlan,
    full: &mut StreamExecutionPlan,
    snapshot: &OperatorStateSnapshot,
    legacy: &OperatorStateSnapshot,
    job: &StreamJobContext,
) -> OperatorStateSnapshot {
    let before = join(full).status();
    assert!(
        join(full).restore(snapshot).is_err(),
        "full state cannot restore omitted columns"
    );
    assert_eq!(join(full).status(), before);
    join(cold).restore(legacy).unwrap();
    assert_eq!(join(cold).status(), before);
    join(full).reset().unwrap();
    join(full).restore(legacy).unwrap();
    let epoch = Epoch::new(2).unwrap();
    assert!(join(cold).checkpoint(epoch).is_err());
    join(cold)
        .prepare_checkpoint_async(&StreamOperatorContext::new(job, "asof", None))
        .await
        .unwrap();
    let migrated = join(cold).checkpoint(epoch).unwrap();
    assert_eq!(migrated.inline_metadata["layout_version"], 3);
    assert!(
        migrated.inline_metadata["v2_inventory"]["deltas"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    join(cold).restore(&migrated).unwrap();
    migrated
}

pub(in crate::pipeline) async fn execution_and_restore() {
    for v1 in [false, true] {
        continuation(
            &[
                &[
                    "abs(right__value) + left__seq AS total",
                    "left__seq AS sequence",
                ],
                &["left__seq AS another"],
            ],
            Some("right__value > 499 AND left__key = 'A' AND left__seq = 7"),
            v1,
        )
        .await;
        continuation(&[&["42 AS constant_value"]], None, v1).await;
    }
}

async fn projected_prices(select: &str, filter: Option<&str>) -> RecordBatch {
    use datafusion::arrow::array::{Float64Array, StringArray, TimestampMicrosecondArray};
    let graph = hot_graph_with_filter(&["quote__sequence AS sequence", select], false, filter);
    let mut plan = graph
        .compile_stream(
            &UdfRegistry::new().snapshot(),
            &StreamRequirements::default(),
        )
        .unwrap();
    let job = StreamJobContext::new(
        1,
        &plan.fingerprint,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let operator = join(&mut plan);
    operator.enable_native_checkpoint().unwrap();
    let make = |side: usize, columns| {
        let schema = Arc::clone(operator.input_ports()[side].schema().unwrap());
        Batch::table(
            vec![RecordBatch::try_new(schema, columns).unwrap()],
            crate::BatchMetadata::default(),
        )
        .unwrap()
    };
    let right = make(
        1,
        vec![
            Arc::new(StringArray::from(vec!["A"])),
            Arc::new(Float64Array::from(vec![2.0])),
            Arc::new(UInt64Array::from(vec![0])),
            Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
        ],
    );
    let left = make(
        0,
        vec![
            Arc::new(TimestampMicrosecondArray::from(vec![0, 0]).with_timezone("UTC")),
            Arc::new(UInt64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["A", "A"])),
            Arc::new(Float64Array::from(vec![5.0, 12.0])),
        ],
    );
    assert!(process(operator, right, "right", &job).await.is_empty());
    let joined = process(operator, left, "left", &job).await;
    let node = plan
        .nodes
        .iter_mut()
        .find(|node| node.node_id == "result")
        .unwrap();
    let mut out = EdgeCollector::new(node.output_ports.values().cloned().collect());
    for batch in joined {
        node.operator
            .process_data(
                "input",
                batch,
                &StreamOperatorContext::new(&job, "result", None),
                &mut out,
            )
            .await
            .unwrap();
    }
    let outputs = out.drain("output");
    let records = outputs
        .iter()
        .flat_map(|event| event.as_data().unwrap().table_payload().unwrap().batches())
        .collect::<Vec<_>>();
    datafusion::arrow::compute::concat_batches(&records[0].schema(), records.iter().copied())
        .unwrap()
}

#[tokio::test]
async fn fused_join_preserves_the_filter_when_delivering_rows() {
    use datafusion::arrow::array::Float64Array;
    let result = projected_prices(
        "quote__price * dimension__factor AS value",
        Some("quote__price > 10"),
    )
    .await;
    assert_eq!(result.num_rows(), 1);
    assert_eq!(
        result
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0),
        2
    );
    assert_eq!(
        result
            .column(1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .value(0)
            .to_bits(),
        24.0_f64.to_bits()
    );
}

#[tokio::test]
async fn fused_join_retains_the_exact_quoted_output_alias() {
    let result = projected_prices(r#"(quote__price * dimension__factor) AS "Va lue""#, None).await;
    assert_eq!(result.num_rows(), 2);
    assert_eq!(result.schema().field(1).name(), "Va lue");
}
