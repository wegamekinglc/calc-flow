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
