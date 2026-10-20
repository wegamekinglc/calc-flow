use super::*;

fn cumulative_spec() -> RollingSpec {
    serde_json::from_value(json!({
        "configuration_version": 1,
        "state_layout_version": 4,
        "partition_by": ["symbol"], "event_time": "ts", "sequence_by": ["sequence"],
        "outputs": [
            {"kind":"cumulative_sum", "primitive_version":1, "input":"price", "output":"total", "min_periods":1},
            {"kind":"cumulative_min", "primitive_version":1, "input":"price", "output":"low", "min_periods":1},
            {"kind":"cumulative_max", "primitive_version":1, "input":"price", "output":"high", "min_periods":1},
            {"kind":"cumulative_variance", "primitive_version":1, "input":"price", "output":"variance", "min_periods":1, "ddof":1},
            {"kind":"cumulative_variance", "primitive_version":1, "input":"price", "output":"population", "min_periods":2, "ddof":0}
        ],
        "allowed_lateness_micros":0,
        "late_policy":{"kind":"error", "scope":"envelope"},
        "value_policy":"stateful_numeric_v1"
    })).unwrap()
}

fn drain(collector: &mut crate::EdgeCollector) -> Vec<RecordBatch> {
    collector
        .drain("output")
        .iter()
        .flat_map(|message| {
            message
                .as_data()
                .unwrap()
                .table_payload()
                .unwrap()
                .batches()
                .to_vec()
        })
        .collect()
}

fn remove_cumulative_summaries(
    snapshot: &crate::OperatorStateSnapshot,
) -> crate::OperatorStateSnapshot {
    use datafusion::arrow::{array::BooleanArray, compute::filter_record_batch};

    let mut snapshot = snapshot.clone();
    let (segment_id, segment) = snapshot.segments.iter().next().unwrap();
    let segment_id = segment_id.clone();
    let mut reader = FileReader::try_new(Cursor::new(segment.bytes()), None).unwrap();
    let record = reader.next().unwrap().unwrap();
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .unwrap();
    let keep = kinds
        .values()
        .iter()
        .map(|kind| *kind != 4)
        .collect::<BooleanArray>();
    let stripped = filter_record_batch(&record, &keep).unwrap();
    assert!(stripped.num_rows() < record.num_rows());
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, stripped.schema().as_ref()).unwrap();
        writer.write(&stripped).unwrap();
        writer.finish().unwrap();
    }
    let replacement = crate::StateSegment::new(bytes);
    let handle = &mut snapshot
        .inline_metadata
        .get_mut("segment_inventory")
        .unwrap()[0]["handle"];
    handle["byte_len"] = json!(replacement.bytes().len());
    handle["sha256"] = json!(replacement.sha256());
    snapshot.segments.insert(segment_id, replacement);
    snapshot
}

async fn pending_checkpoint(
    profile: RollingNumericalProfile,
    prefix: bool,
) -> (
    RollingOperator,
    crate::OperatorStateSnapshot,
    crate::StreamJobContext,
) {
    let mut spec = cumulative_spec();
    spec.numerical_profile = profile;
    let mut source = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    let job = crate::StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let rows = if prefix {
        vec![(1, "a", 1, Some(10.0)), (2, "a", 2, Some(20.0))]
    } else {
        source
            .set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
            .unwrap();
        vec![(2, "a", 2, Some(20.0))]
    };
    let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
    source
        .process_data(
            "input",
            Batch::table(vec![float64_fast_record(&rows)], BatchMetadata::default()).unwrap(),
            &context,
            &mut output,
        )
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(1), &context, &mut output)
        .await
        .unwrap();
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    (source, snapshot, job)
}

#[tokio::test]
async fn test_cumulative_snapshot_rejects_missing_prefix_inventory() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let (mut source, snapshot, job) = pending_checkpoint(profile, true).await;
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone())
                .unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        let corrupted = remove_cumulative_summaries(&snapshot);
        assert!(
            StreamOperator::restore(&mut restored, &corrupted).is_err(),
            "{profile:?}: restore accepted a processed prefix without any cumulative summaries"
        );
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        for operator in [&mut source, &mut restored] {
            operator
                .on_watermark(EventTime::from_micros(2), &context, &mut output)
                .await
                .unwrap();
        }
        let output = drain(&mut output);
        assert_eq!(output[0], output[1]);
        let total = output[1]
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total.value(0), 30.0);
    }
}

#[tokio::test]
async fn test_cumulative_snapshot_restores_pending_only_entity() {
    for profile in [
        RollingNumericalProfile::StableV1,
        RollingNumericalProfile::StableV2Preview,
    ] {
        let (mut source, snapshot, job) = pending_checkpoint(profile, false).await;
        let mut restored =
            RollingOperator::new("rolling", Arc::new(kernel_schema()), source.spec().clone())
                .unwrap();
        restored
            .set_state_budget(StateBudget::new(1, 1 << 20).unwrap())
            .unwrap();
        StreamOperator::restore(&mut restored, &snapshot).unwrap();
        assert!(restored.state.histories.by_entity.is_empty());
        let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
        let mut output = crate::EdgeCollector::new(source.output_ports().to_vec());
        for operator in [&mut source, &mut restored] {
            operator
                .on_watermark(EventTime::from_micros(2), &context, &mut output)
                .await
                .unwrap();
        }
        let output = drain(&mut output);
        assert_eq!(output[0], output[1]);
        let total = output[1]
            .column_by_name("total")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(total.value(0), 20.0);
    }
}

#[tokio::test]
async fn test_cumulative_snapshot_restores_summaries_and_pending_rows() {
    use crate::{CancellationToken, StreamJobContext};
    let spec = cumulative_spec();
    let mut source =
        RollingOperator::new("rolling", Arc::new(kernel_schema()), spec.clone()).unwrap();
    assert_eq!(source.state_layout_version(), 4);
    let job = StreamJobContext::new(
        7,
        TEST_FINGERPRINT,
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "rolling", None);
    let mut collector = crate::EdgeCollector::new(source.output_ports().to_vec());
    let first = float64_fast_record(&[
        (1, "a", 1, Some(1e12)),
        (1, "b", 1, None),
        (2, "a", 2, Some(1e12 + 2.0)),
        (2, "b", 2, Some(10.0)),
    ]);
    source
        .process_data(
            "input",
            Batch::table(vec![first], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    source
        .on_watermark(EventTime::from_micros(1), &context, &mut collector)
        .await
        .unwrap();
    drain(&mut collector);
    let snapshot = source.checkpoint(Epoch::new(1).unwrap()).unwrap();
    let mut restored = RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).unwrap();
    StreamOperator::restore(&mut restored, &snapshot).unwrap();
    let mut unknown = snapshot.clone();
    unknown
        .inline_metadata
        .insert("state_layout_version".into(), json!(99));
    assert!(StreamOperator::restore(&mut restored, &unknown).is_err());
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(1)));
    for operator in [&mut source, &mut restored] {
        operator
            .on_watermark(EventTime::from_micros(2), &context, &mut collector)
            .await
            .unwrap();
    }
    let output = drain(&mut collector);
    assert_eq!(output.len(), 2);
    assert_eq!(output[0], output[1]);
    let variance = output[1]
        .column_by_name("variance")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(variance.value(0), 2.0);
    assert!(variance.is_null(1));
    let low = output[1]
        .column_by_name("low")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(low.value(0), 1e12);
    assert_eq!(low.value(1), 10.0);
    let context = StreamOperatorContext::new(&job, "rolling", Some(EventTime::from_micros(2)));
    for operator in [&mut source, &mut restored] {
        let next = float64_fast_record(&[(3, "a", 3, Some(1e12 + 4.0)), (3, "b", 3, Some(20.0))]);
        operator
            .process_data(
                "input",
                Batch::table(vec![next], BatchMetadata::default()).unwrap(),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        operator
            .on_watermark(EventTime::from_micros(3), &context, &mut collector)
            .await
            .unwrap();
    }
    let output = drain(&mut collector);
    assert_eq!(output[0], output[1]);
    let variance = output[1]
        .column_by_name("variance")
        .unwrap()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(variance.value(0), 4.0);
    assert_eq!(variance.value(1), 50.0);
}

#[test]
fn test_cumulative_versions_and_ddof_fail_closed() {
    let mut spec = cumulative_spec();
    spec.state_layout_version = 2;
    assert!(RollingOperator::new("rolling", Arc::new(kernel_schema()), spec).is_err());
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    value["outputs"][0]["primitive_version"] = json!(99);
    assert!(
        RollingOperator::new(
            "rolling",
            Arc::new(kernel_schema()),
            serde_json::from_value(value).unwrap()
        )
        .is_err()
    );
    let mut value = serde_json::to_value(cumulative_spec()).unwrap();
    value["outputs"][3]["ddof"] = json!(2);
    assert!(
        RollingOperator::new(
            "rolling",
            Arc::new(kernel_schema()),
            serde_json::from_value(value).unwrap()
        )
        .is_err()
    );
}
