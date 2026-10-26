use calc_flow::{
    DeliveryGuarantee, EdgeCollector, Epoch, OperatorMetadata, OperatorStateSnapshot,
    PipelineBuilder, SnapshotSqlOperator, StateBudget, StateSegment, StreamOperator,
    StreamOperatorContext, StreamRequirements, UdfRegistrySnapshot,
};

#[path = "snapshot_sql/runner.rs"]
mod runner;
#[path = "snapshot_sql/support.rs"]
mod support;
use support::*;

#[tokio::test]
async fn test_uninitialized_partial_and_initialized_roundtrips() {
    for initialized in 0..=2 {
        let mut original = operator(QUERY).await;
        let job = job();
        let context = StreamOperatorContext::new(&job, "valuation", None);
        let mut collector = EdgeCollector::new(original.output_ports().to_vec());
        if initialized > 0 {
            original
                .process_data("p", batch(&[&[1], &[2]]), &context, &mut collector)
                .await
                .unwrap();
        }
        if initialized > 1 {
            original
                .process_data("q", batch(&[&[1]]), &context, &mut collector)
                .await
                .unwrap();
        }
        collector.drain("output");
        let saved = original.checkpoint(Epoch::new(1).unwrap()).unwrap();
        assert!(saved.segments.contains_key("control"));
        let mut restored = operator(QUERY).await;
        restored.restore(&saved).unwrap();
        let carried = restored.checkpoint(Epoch::new(2).unwrap()).unwrap();
        assert_eq!(carried.segments, saved.segments);
        if initialized == 0 {
            restored
                .process_data("p", batch(&[&[1], &[2]]), &context, &mut collector)
                .await
                .unwrap();
            assert!(collector.drain("output").is_empty());
        }
        restored
            .process_data("q", batch(&[&[2]]), &context, &mut collector)
            .await
            .unwrap();
        assert_output(
            &mut collector,
            &oracle(QUERY, batch(&[&[1], &[2]]), batch(&[&[2]])).await,
        );
    }
}

fn change_control(
    snapshot: &OperatorStateSnapshot,
    edit: impl FnOnce(&mut serde_json::Value),
) -> OperatorStateSnapshot {
    let mut snapshot = snapshot.clone();
    let mut control: serde_json::Value =
        serde_json::from_slice(snapshot.segments["control"].bytes()).unwrap();
    edit(&mut control);
    let segment = StateSegment::new(serde_json::to_vec(&control).unwrap());
    snapshot
        .inline_metadata
        .insert("control_sha256".into(), segment.sha256().into());
    snapshot.segments.insert("control".into(), segment);
    snapshot
}

#[tokio::test]
async fn test_unknown_corrupt_and_incompatible_state_fail_atomically() {
    let mut operator = operator(QUERY).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    let good = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut unknown_segment = good.clone();
    unknown_segment
        .segments
        .insert("unknown".into(), StateSegment::new(vec![]));
    let mut missing = good.clone();
    missing.segments.remove("metadata-0");
    let mut corrupt = good.clone();
    corrupt
        .segments
        .insert("relation-0".into(), StateSegment::new(vec![0; 20]));
    let mut inline = good.clone();
    inline
        .inline_metadata
        .insert("old_sql_layout".into(), 3.into());
    let cases = [
        OperatorStateSnapshot::default(),
        unknown_segment,
        missing,
        corrupt,
        inline,
        change_control(&good, |v| v["state_kind"] = "sql".into()),
        change_control(&good, |v| v["state_layout"] = 2.into()),
        change_control(&good, |v| v["state_accounting"] = 2.into()),
        change_control(&good, |v| v["datafusion_version"] = "unknown".into()),
        change_control(&good, |v| v["identity"]["output"] = "wrong".into()),
        change_control(&good, |v| v["aliases"][0] = 500.into()),
        change_control(&good, |v| v["relations"][0]["rows"] = 42.into()),
        change_control(&good, |v| v["unknown"] = true.into()),
    ];
    for bad in cases {
        assert!(operator.restore(&bad).is_err());
        assert_eq!(
            operator
                .checkpoint(Epoch::new(2).unwrap())
                .unwrap()
                .segments,
            good.segments
        );
    }
}

#[tokio::test]
async fn test_restore_rechecks_current_row_and_byte_budgets() {
    let mut original = operator(QUERY).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(original.output_ports().to_vec());
    original
        .process_data("p", batch(&[&[1, 2]]), &context, &mut collector)
        .await
        .unwrap();
    original
        .process_data("q", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    let saved = original.checkpoint(Epoch::new(1).unwrap()).unwrap();
    for budget in [
        StateBudget::new(2, 1 << 20).unwrap(),
        StateBudget::new(10, 1).unwrap(),
    ] {
        let mut restored = operator(QUERY).await;
        restored.set_state_budget(budget).unwrap();
        assert!(restored.restore(&saved).is_err());
        restored
            .set_state_budget(StateBudget::new(10, 1 << 20).unwrap())
            .unwrap();
        restored.restore(&saved).unwrap();
        restored
            .process_data("q", batch(&[&[2]]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
    }
}

#[tokio::test]
async fn test_shared_relation_owner_is_encoded_once_and_rows_count_per_alias() {
    let mut operator = operator(QUERY).await;
    operator
        .set_state_budget(StateBudget::new(2, 1 << 20).unwrap())
        .unwrap();
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let shared = batch(&[&[1]]);
    operator
        .process_data("p", shared.clone(), &context, &mut collector)
        .await
        .unwrap();
    operator
        .process_data("q", shared, &context, &mut collector)
        .await
        .unwrap();
    let saved = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();
    assert_eq!(saved.segments.len(), 3);
    operator.reset().unwrap();
    operator.restore(&saved).unwrap();
    assert!(
        operator
            .set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
            .is_err()
    );
}

#[tokio::test]
async fn test_graph_rejects_batch_successors_and_exactly_once() {
    let udfs = UdfRegistrySnapshot::default();
    assert!(
        PipelineBuilder::new("snapshot")
            .unwrap()
            .add_node("v", operator(QUERY).await)
            .unwrap()
            .compile_batch(&udfs)
            .is_err()
    );
    let requirements = StreamRequirements {
        delivery: [("output".into(), DeliveryGuarantee::ExactlyOnce)].into(),
    };
    assert!(
        PipelineBuilder::new("snapshot")
            .unwrap()
            .add_node("v", operator(QUERY).await)
            .unwrap()
            .compile_stream(&udfs, &requirements)
            .is_err()
    );
    let plan = PipelineBuilder::new("snapshot")
        .unwrap()
        .add_node("v", operator(QUERY).await)
        .unwrap()
        .compile_stream(&udfs, &StreamRequirements::default())
        .unwrap();
    assert_eq!(plan.snapshot_source_binding_ids(), ["p", "q"]);
    let tuned = calc_flow::StreamRuntimeConfig {
        sql_snapshot_state_budget: Some(StateBudget::new(10, 1 << 20).unwrap()),
        ..calc_flow::StreamRuntimeConfig::default()
    };
    assert_ne!(
        plan.runtime_config_hash(&tuned).unwrap(),
        plan.runtime_config_hash(&calc_flow::StreamRuntimeConfig::default())
            .unwrap()
    );
    assert!(
        PipelineBuilder::new("snapshot")
            .unwrap()
            .add_node(
                "v",
                SnapshotSqlOperator::new("v", "SELECT * FROM p", vec!["p".into()], vec![]).unwrap()
            )
            .unwrap()
            .compile_stream(&udfs, &StreamRequirements::default())
            .is_err()
    );
    let project: calc_flow::OperatorSpec = serde_json::from_value(
        serde_json::json!({"kind": "snapshot_sql", "query": "SELECT * FROM p", "aliases": ["p"]}),
    )
    .unwrap();
    assert!(matches!(
        project,
        calc_flow::OperatorSpec::SnapshotSql { .. }
    ));
}

#[tokio::test]
async fn test_snapshot_output_cannot_feed_an_append_sql_successor() {
    use calc_flow::{BatchKind, Edge, Port, PortEndpoint, SqlOperator};
    let snapshot = operator(QUERY).await;
    let schema = snapshot.output_ports()[0].schema().unwrap();
    let fields = schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone())
        .collect::<Vec<_>>();
    let successor = SqlOperator::new(
        "after",
        "SELECT * FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap()
    .with_ports(
        vec![Port::new("events", BatchKind::Table, true, Some(fields.clone())).unwrap()],
        Port::new("output", BatchKind::Table, true, Some(fields)).unwrap(),
    )
    .unwrap();
    let result = PipelineBuilder::new("unsafe-successor")
        .unwrap()
        .add_node("v", snapshot)
        .unwrap()
        .add_node("after", Box::new(successor))
        .unwrap()
        .connect(Edge::new(
            PortEndpoint::new("v", "output").unwrap(),
            PortEndpoint::new("after", "events").unwrap(),
        ))
        .unwrap()
        .compile_stream(
            &UdfRegistrySnapshot::default(),
            &StreamRequirements::default(),
        );
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("connect directly to sinks")
    );
}

#[tokio::test]
async fn test_independent_output_keeps_its_exactly_once_compilation_contract() {
    let requirements = StreamRequirements {
        delivery: [("side.output".into(), DeliveryGuarantee::ExactlyOnce)].into(),
    };
    let old = calc_flow::SqlOperator::new(
        "side",
        "SELECT * FROM events",
        vec!["events".into()],
        vec![],
    )
    .unwrap();
    let plan = PipelineBuilder::new("independent-output")
        .unwrap()
        .add_node("v", operator(QUERY).await)
        .unwrap()
        .add_node("side", Box::new(old))
        .unwrap()
        .compile_stream(&UdfRegistrySnapshot::default(), &requirements)
        .unwrap();
    assert_eq!(
        plan.requirements().delivery["side.output"],
        DeliveryGuarantee::ExactlyOnce
    );
}

#[tokio::test]
async fn test_valid_digest_with_invalid_ipc_or_schema_still_fails_atomically() {
    use datafusion::arrow::{
        array::{Array, Int64Array},
        ipc::writer::FileWriter,
        record_batch::RecordBatch,
    };
    use std::sync::Arc;
    let mut operator = operator(QUERY).await;
    let job = job();
    let context = StreamOperatorContext::new(&job, "valuation", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    operator
        .process_data("p", batch(&[&[1]]), &context, &mut collector)
        .await
        .unwrap();
    let good = operator.checkpoint(Epoch::INITIAL).unwrap();
    let wrong = RecordBatch::try_from_iter(vec![(
        "other",
        Arc::new(Int64Array::from(vec![1])) as Arc<dyn Array>,
    )])
    .unwrap();
    let mut encoded = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut encoded, &wrong.schema()).unwrap();
        writer.write(&wrong).unwrap();
        writer.finish().unwrap();
    }
    for bytes in [b"invalid".to_vec(), b"ARROW1ARROW1".to_vec(), encoded] {
        let segment = StateSegment::new(bytes);
        let mut bad = good.clone();
        bad.segments.insert("relation-0".into(), segment.clone());
        let bad = change_control(&bad, |v| {
            v["relations"][0]["data_sha256"] = segment.sha256().into()
        });
        assert!(operator.restore(&bad).is_err());
        assert_eq!(
            operator.checkpoint(Epoch::INITIAL).unwrap().segments,
            good.segments
        );
    }
}
