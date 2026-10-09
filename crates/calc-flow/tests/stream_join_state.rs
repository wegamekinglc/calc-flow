use std::sync::Arc;
use std::time::Duration;

use std::collections::{BTreeMap, BTreeSet};

use calc_flow::OperatorMetadata;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, CancellationToken, EdgeCollector, Epoch, EventTime,
    IngressProgress, IngressProgressSnapshot, IngressState, JoinStateLimits, JoinTimeBounds,
    JsonMap, OperatorStateSnapshot, StreamJobContext, StreamJoinOperator, StreamJoinSpec,
    StreamOperator, StreamOperatorContext, StreamingFailureReason,
};
use datafusion::arrow::array::Array as _;
use datafusion::arrow::array::{
    ArrayRef, Decimal32Array, Decimal64Array, FixedSizeListArray, Int32Array, Int64Array,
    ListArray, StringArray, StructArray, TimestampMicrosecondArray,
};
use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion::arrow::datatypes::{DataType, Field, Fields, Schema, TimeUnit};
use datafusion::arrow::record_batch::RecordBatch;

fn spec() -> StreamJoinSpec {
    StreamJoinSpec::inner(
        ["account_id"],
        ["account_id"],
        "ts",
        "ts",
        JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
        JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
    )
    .unwrap()
}

fn plain_schema(extra: Option<Field>) -> Arc<Schema> {
    let mut fields = vec![
        Field::new("account_id", DataType::Utf8, false),
        Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, None),
            false,
        ),
    ];
    if let Some(extra) = extra {
        fields.push(extra);
    }
    Arc::new(Schema::new(fields))
}

/// Version-1 key-encoding block for one non-null Utf8 key value:
/// tag(1) + tz length(4) + value length(4) + value bytes.
const ENCODED_UTF8_KEY_LEN: u64 = 1 + 4 + 4 + "account-01".len() as u64;

/// Feeds one left row carrying `payload` and returns the retained left bytes
/// reported by the checkpoint metrics.
async fn retained_bytes_for(payload_type: DataType, payload: ArrayRef) -> u64 {
    let mut operator = StreamJoinOperator::new(
        "match",
        plain_schema(Some(Field::new("payload", payload_type, true))),
        plain_schema(None),
        spec(),
    )
    .unwrap();
    let job = StreamJobContext::new(
        1,
        "fingerprint",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = RecordBatch::try_new(
        plain_schema(Some(Field::new(
            "payload",
            payload.data_type().clone(),
            true,
        ))),
        vec![
            Arc::new(StringArray::from(vec!["account-01"])),
            Arc::new(TimestampMicrosecondArray::from(vec![1_000_000_i64])),
            payload,
        ],
    )
    .unwrap();
    operator
        .process_data(
            "left",
            Batch::table(vec![batch], BatchMetadata::default()).unwrap(),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();
    let metadata: serde_json::Value = serde_json::to_value(&snapshot.inline_metadata).unwrap();
    metadata["metrics"]["left"]["retained_bytes"]
        .as_u64()
        .unwrap()
}

fn base_charge(payload_cells: u64) -> u64 {
    // 64 overhead + encoded key + 8 event time + 8 row id + payload cells.
    // The key and timestamp cells are themselves payload cells.
    64 + ENCODED_UTF8_KEY_LEN + 16 + payload_cells
}

const KEY_CELL: u64 = 1 + 4 + "account-01".len() as u64;
const TS_CELL: u64 = 1 + 8;

#[tokio::test]
async fn state_charge_uses_the_versioned_logical_table_for_list_payloads() {
    let values = Int64Array::from(vec![1, 2, 3]);
    let element = Field::new("element", DataType::Int64, true);
    let offsets = OffsetBuffer::from_lengths([3_usize]);
    let list = ListArray::new(Arc::new(element), offsets, Arc::new(values), None);
    // List cell: 1 validity + 4 prefix + three children (1 + 8 each).
    let list_cell = 1 + 4 + 3 * (1 + 8);
    assert_eq!(
        retained_bytes_for(
            DataType::List(Arc::new(Field::new("element", DataType::Int64, true))),
            Arc::new(list)
        )
        .await,
        base_charge(KEY_CELL + TS_CELL + list_cell)
    );
}

#[tokio::test]
async fn state_charge_covers_decimal32_and_decimal64_value_bytes() {
    let decimal32 = Decimal32Array::from(vec![7_i32])
        .with_precision_and_scale(9, 0)
        .unwrap();
    assert_eq!(
        retained_bytes_for(decimal_type_of_32(), Arc::new(decimal32)).await,
        base_charge(KEY_CELL + TS_CELL + 1 + 4)
    );
    let decimal64 = Decimal64Array::from(vec![7_i64])
        .with_precision_and_scale(18, 0)
        .unwrap();
    assert_eq!(
        retained_bytes_for(decimal_type_of_64(), Arc::new(decimal64)).await,
        base_charge(KEY_CELL + TS_CELL + 1 + 8)
    );
}

#[tokio::test]
async fn state_charge_walks_struct_children_in_schema_order() {
    let struct_array = StructArray::new(
        Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]),
        vec![
            Arc::new(Int32Array::from(vec![Some(1)])),
            Arc::new(StringArray::from(vec![Some("wxyz")])),
        ],
        None,
    );
    // Struct cell: own validity byte + (1 + 4) + (1 + 4 + 4).
    let struct_cell = 1 + 5 + 9;
    assert_eq!(
        retained_bytes_for(struct_array_type(), Arc::new(struct_array)).await,
        base_charge(KEY_CELL + TS_CELL + struct_cell)
    );
}

#[tokio::test]
async fn state_charge_charges_null_cells_one_validity_byte() {
    let null_payload = StringArray::from(vec![Option::<String>::None]);
    assert_eq!(
        retained_bytes_for(DataType::Utf8, Arc::new(null_payload)).await,
        base_charge(KEY_CELL + TS_CELL + 1)
    );
}

#[tokio::test]
async fn state_charge_covers_list_view_large_list_view_and_fixed_size_list() {
    let values = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
    let element = Field::new("element", DataType::Int64, true);
    // Three child cells of (1 validity + 8 value) bytes each.
    let children = 3 * (1 + 8);

    let list_view = datafusion::arrow::array::ListViewArray::new(
        Arc::new(element.clone()),
        ScalarBuffer::from(vec![0_i32]),
        ScalarBuffer::from(vec![3_i32]),
        values.clone(),
        None,
    );
    // ListView cell: 1 validity + 8 prefix + children.
    assert_eq!(
        retained_bytes_for(
            DataType::ListView(Arc::new(element.clone())),
            Arc::new(list_view),
        )
        .await,
        base_charge(KEY_CELL + TS_CELL + 1 + 8 + children)
    );

    let large_list_view = datafusion::arrow::array::LargeListViewArray::new(
        Arc::new(element.clone()),
        ScalarBuffer::from(vec![0_i64]),
        ScalarBuffer::from(vec![3_i64]),
        values.clone(),
        None,
    );
    // LargeListView cell: 1 validity + 16 prefix + children.
    assert_eq!(
        retained_bytes_for(
            DataType::LargeListView(Arc::new(element.clone())),
            Arc::new(large_list_view),
        )
        .await,
        base_charge(KEY_CELL + TS_CELL + 1 + 16 + children)
    );

    let fixed = FixedSizeListArray::new(Arc::new(element), 3, values, None);
    // FixedSizeList cell: 1 validity + children, with no prefix.
    assert_eq!(
        retained_bytes_for(
            DataType::FixedSizeList(Arc::new(Field::new("element", DataType::Int64, true)), 3),
            Arc::new(fixed),
        )
        .await,
        base_charge(KEY_CELL + TS_CELL + 1 + children)
    );
}

fn decimal_type_of_32() -> DataType {
    DataType::Decimal32(9, 0)
}

fn decimal_type_of_64() -> DataType {
    DataType::Decimal64(18, 0)
}

fn struct_array_type() -> DataType {
    DataType::Struct(Fields::from(vec![
        Field::new("a", DataType::Int32, true),
        Field::new("b", DataType::Utf8, true),
    ]))
}

fn state_operator(bounds_secs: u64) -> (StreamJoinOperator, StreamJobContext, EdgeCollector) {
    let operator = StreamJoinOperator::new(
        "match",
        plain_schema(None),
        plain_schema(None),
        StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "ts",
            "ts",
            JoinTimeBounds::new(
                Duration::from_secs(bounds_secs),
                Duration::from_secs(bounds_secs),
            )
            .unwrap(),
            JoinStateLimits::new(100_000, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let collector = EdgeCollector::new(operator.output_ports().to_vec());
    (
        operator,
        StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        ),
        collector,
    )
}

fn keyed_batch(values: &[(&str, i64)]) -> Batch {
    let batch = RecordBatch::try_new(
        plain_schema(None),
        vec![
            Arc::new(StringArray::from(
                values.iter().map(|(key, _)| *key).collect::<Vec<_>>(),
            )),
            Arc::new(TimestampMicrosecondArray::from(
                values.iter().map(|(_, ts)| *ts).collect::<Vec<_>>(),
            )),
        ],
    )
    .unwrap();
    Batch::table(vec![batch], BatchMetadata::default()).unwrap()
}

const SECOND: i64 = 1_000_000;

fn wire_count(bytes: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap())
}

fn assert_v2_base(snapshot: &OperatorStateSnapshot, side: &str, rows: u64) {
    let bytes = snapshot.segments[&format!("{side}-base")].bytes();
    assert_eq!(&bytes[..8], b"CFJIDX2\0");
    assert_eq!(wire_count(bytes, 16), rows);
    assert_eq!(wire_count(bytes, 24), 0);
    if rows == 0 {
        assert_eq!(bytes.len(), 32);
    }
}

fn assert_v2_delta(snapshot: &OperatorStateSnapshot, epoch: u64, side: &str) {
    let bytes = snapshot.segments[&format!("{side}-delta-{epoch}")].bytes();
    assert_eq!(&bytes[..8], b"CFJDIX2\0");
    assert_eq!(wire_count(bytes, 16), epoch);
    assert_eq!(wire_count(bytes, 24), 1);
    assert_eq!(wire_count(bytes, 32), 0);
}

fn assert_v2_history(
    snapshot: &OperatorStateSnapshot,
    anchor: u64,
    base_rows: [u64; 2],
    deltas: &[(u64, &str)],
    payload_rows: &[(&str, u64)],
) {
    let metadata = &snapshot.inline_metadata;
    assert_eq!(metadata["layout_version"], 2);
    let inventory = &metadata["v2_inventory"];
    assert_eq!(inventory["codec_version"], 2);
    assert_eq!(inventory["base_epoch"], anchor);
    let expected_deltas = deltas
        .iter()
        .map(|(epoch, side)| serde_json::json!({"epoch": epoch, "sides": [side]}))
        .collect::<Vec<_>>();
    assert_eq!(inventory["deltas"], serde_json::json!(expected_deltas));
    assert_v2_base(snapshot, "left", base_rows[0]);
    assert_v2_base(snapshot, "right", base_rows[1]);
    let mut names = BTreeSet::from(["left-base".to_owned(), "right-base".to_owned()]);
    for &(epoch, side) in deltas {
        assert_v2_delta(snapshot, epoch, side);
        names.insert(format!("{side}-delta-{epoch}"));
    }
    let mut actual_payload_rows = Vec::new();
    for entry in inventory["payloads"].as_array().unwrap() {
        let side = entry["side"].as_str().unwrap();
        let digest = entry["sha256"].as_str().unwrap();
        let name = format!("{side}-payload-{digest}");
        let segment = &snapshot.segments[&name];
        assert_eq!(segment.sha256(), digest);
        assert_eq!(
            entry["bytes"].as_u64().unwrap(),
            u64::try_from(segment.bytes().len()).unwrap()
        );
        assert_eq!(&segment.bytes()[..8], b"CFJPAY2\0");
        assert_eq!(
            entry["rows"].as_u64().unwrap(),
            wire_count(segment.bytes(), 16)
        );
        actual_payload_rows.push((side.to_owned(), wire_count(segment.bytes(), 16)));
        names.insert(name);
    }
    let mut expected_payload_rows = payload_rows
        .iter()
        .map(|(side, rows)| ((*side).to_owned(), *rows))
        .collect::<Vec<_>>();
    actual_payload_rows.sort_unstable();
    expected_payload_rows.sort_unstable();
    assert_eq!(actual_payload_rows, expected_payload_rows);
    assert_eq!(
        snapshot.segments.keys().cloned().collect::<BTreeSet<_>>(),
        names
    );
}

fn assert_shared_segments(previous: &OperatorStateSnapshot, current: &OperatorStateSnapshot) {
    for (name, segment) in &previous.segments {
        assert_eq!(&current.segments[name], segment);
        assert!(Arc::ptr_eq(
            &segment.bytes_arc(),
            &current.segments[name].bytes_arc()
        ));
    }
}

#[tokio::test]
async fn checkpoints_capture_only_dirty_deltas_and_restore_folds_them() {
    let (mut operator, job, mut collector) = state_operator(300);
    let context = StreamOperatorContext::new(&job, "match", None);

    operator
        .process_data(
            "left",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let first = operator.checkpoint(Epoch::INITIAL).unwrap();
    assert_v2_history(&first, 0, [0, 0], &[(1, "left")], &[("left", 1)]);

    operator
        .process_data(
            "right",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let second = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    // The epoch-2 snapshot carries the prepared epoch-1 delta plus the new
    // dirty right delta; no full-state re-encode happens at either epoch.
    assert_v2_history(
        &second,
        0,
        [0, 0],
        &[(1, "left"), (2, "right")],
        &[("left", 1), ("right", 1)],
    );
    assert_shared_segments(&first, &second);
    assert_eq!(operator.status().left.retained_bytes, 105);
    assert_eq!(operator.status().right.retained_bytes, 105);

    let (mut restored, job, collector_two) = state_operator(300);
    let context_two = StreamOperatorContext::new(&job, "match", None);
    let mut collector_two = collector_two;
    restored.restore(&second).unwrap();
    assert_eq!(restored.status(), operator.status());
    // The restored left row still matches a newly processed right row.
    restored
        .process_data(
            "right",
            keyed_batch(&[("a", 101 * SECOND)]),
            &context_two,
            &mut collector_two,
        )
        .await
        .unwrap();
    let outputs = collector_two.drain("output");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].as_data().unwrap().num_rows(), 1);
    assert_eq!(restored.status().left.retained_bytes, 105);
    assert_eq!(restored.status().right.retained_bytes, 210);
}

#[tokio::test]
async fn evicted_rows_leave_tombstones_that_survive_restore() {
    let (mut operator, job, mut collector) = state_operator(300);
    let context = StreamOperatorContext::new(&job, "match", None);
    operator
        .process_data(
            "left",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let _first = operator.checkpoint(Epoch::INITIAL).unwrap();

    // Advancing the right watermark past `left.ts + before` evicts the row.
    let progress = IngressProgressSnapshot::new(BTreeMap::from([(
        "right".into(),
        IngressProgress::new(
            IngressState::Active,
            Some(EventTime::from_micros(500 * SECOND)),
        ),
    )]));
    let job_progress = StreamOperatorContext::with_ingress_progress(&job, "match", None, progress);
    operator
        .on_ingress_progress("right", &job_progress)
        .await
        .unwrap();

    let second = operator.checkpoint(Epoch::new(2).unwrap()).unwrap();
    assert!(second.segments.contains_key("left-delta-2"));

    let (mut restored, job_two, _) = state_operator(300);
    let _ = &job_two;
    restored.restore(&second).unwrap();
    let context_two = StreamOperatorContext::new(&job_two, "match", None);
    let mut collector_three = EdgeCollector::new(restored.output_ports().to_vec());
    restored
        .process_data(
            "right",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context_two,
            &mut collector_three,
        )
        .await
        .unwrap();
    assert_eq!(collector_three.drain("output").len(), 0);
}

fn limited_operator(rows: u64) -> StreamJoinOperator {
    StreamJoinOperator::new(
        "match",
        plain_schema(None),
        plain_schema(None),
        StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "ts",
            "ts",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(rows, 134_217_728, 1_000_000).unwrap(),
        )
        .unwrap(),
    )
    .unwrap()
}

#[tokio::test]
async fn state_limit_failure_advances_only_its_own_counter_and_nothing_else() {
    let (mut operator, job, mut collector) = {
        let operator = limited_operator(1);
        let collector = EdgeCollector::new(operator.output_ports().to_vec());
        (
            operator,
            StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            collector,
        )
    };
    let context = StreamOperatorContext::new(&job, "match", None);

    operator
        .process_data(
            "left",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let before = operator.status();
    assert_eq!(before.left.retained_rows, 1);
    assert_eq!(before.emitted_match_rows, 0);

    let failure = operator
        .process_data(
            "left",
            keyed_batch(&[("b", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap_err();
    match failure {
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::JoinStateLimitExceeded,
            ..
        } => {}
        other => panic!("expected the typed state-limit reason, got {other:?}"),
    }

    let after = operator.status();
    assert_eq!(after.state_limit_failures, 1);
    assert_eq!(after.match_limit_failures, 0);
    assert_eq!(after.left.retained_rows, 1);
    assert_eq!(after.left.retained_bytes, before.left.retained_bytes);
    assert_eq!(after.right.retained_rows, 0);
    assert_eq!(after.emitted_match_rows, 0);
    // No output was emitted for the failed batch.
    assert!(collector.drain("output").is_empty());
}

#[tokio::test]
async fn watermark_equality_is_on_time_and_eviction_is_strict() {
    let (mut operator, job, mut collector) = {
        let operator = limited_operator(100);
        let collector = EdgeCollector::new(operator.output_ports().to_vec());
        (
            operator,
            StreamJobContext::new(
                1,
                "fingerprint",
                JsonMap::new(),
                None,
                CancellationToken::new(),
            ),
            collector,
        )
    };
    let context = StreamOperatorContext::new(&job, "match", None);
    operator
        .process_data(
            "left",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();

    // A right watermark exactly at the left row's `ts + after` boundary does
    // not evict the retained left row (inclusive bound, FR29 strictness).
    let at_boundary = IngressProgressSnapshot::new(BTreeMap::from([(
        "right".into(),
        IngressProgress::new(
            IngressState::Active,
            Some(EventTime::from_micros(160 * SECOND)),
        ),
    )]));
    operator
        .on_ingress_progress(
            "right",
            &StreamOperatorContext::with_ingress_progress(&job, "match", None, at_boundary),
        )
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 1);

    // One microsecond past the boundary evicts exactly (strict `<`).
    let past_boundary = IngressProgressSnapshot::new(BTreeMap::from([(
        "right".into(),
        IngressProgress::new(
            IngressState::Active,
            Some(EventTime::from_micros(160 * SECOND + 1)),
        ),
    )]));
    operator
        .on_ingress_progress(
            "right",
            &StreamOperatorContext::with_ingress_progress(&job, "match", None, past_boundary),
        )
        .await
        .unwrap();
    assert_eq!(operator.status().left.retained_rows, 0);
    assert_eq!(operator.status().left.evicted_rows, 1);
}

fn assert_compacted_v2_history(snapshot: &OperatorStateSnapshot) {
    assert_v2_history(
        snapshot,
        4,
        [4, 0],
        &[(5, "left")],
        &[("left", 4), ("left", 1)],
    );
}

#[tokio::test]
async fn compaction_survives_restore_checkpoint_restore_cycles() {
    let (mut operator, job, mut collector) = state_operator(300);
    let context = StreamOperatorContext::new(&job, "match", None);

    // Empty V2 bases anchor four dirty deltas until async preparation compacts them.
    let mut previous = None;
    for epoch in 1..=4_u64 {
        let stamp = 100_i64 + i64::try_from(epoch).unwrap() * SECOND;
        operator
            .process_data(
                "left",
                keyed_batch(&[("a", stamp)]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let pending = operator.checkpoint(Epoch::new(epoch).unwrap()).unwrap();
        let deltas = (1..=epoch).map(|epoch| (epoch, "left")).collect::<Vec<_>>();
        let payloads = vec![("left", 1); usize::try_from(epoch).unwrap()];
        assert_v2_history(&pending, 0, [0, 0], &deltas, &payloads);
        if let Some(previous) = &previous {
            assert_shared_segments(previous, &pending);
        }
        previous = Some(pending);
    }
    operator.prepare_checkpoint_async(&context).await.unwrap();
    operator
        .process_data(
            "left",
            keyed_batch(&[("a", 500 * SECOND)]),
            &context,
            &mut collector,
        )
        .await
        .unwrap();
    let compacted = operator.checkpoint(Epoch::new(5).unwrap()).unwrap();
    assert!(
        compacted.segments.contains_key("left-base"),
        "compaction must emit a canonical base: {:?}",
        compacted.segments.keys().collect::<Vec<_>>()
    );
    // The post-compaction dirty row is a delta on top of the fresh base.
    assert!(
        compacted.segments.contains_key("left-delta-5"),
        "{:?}",
        compacted.segments.keys().collect::<Vec<_>>()
    );

    assert_compacted_v2_history(&compacted);
    assert_eq!(operator.status().left.retained_rows, 5);
    assert_eq!(operator.status().left.retained_bytes, 525);

    // Restore from the compacted snapshot; the next checkpoint must still
    // carry the base plus any new deltas instead of an empty inventory.
    let (mut restored, job_two, _) = state_operator(300);
    let _ = &job_two;
    restored.restore(&compacted).unwrap();
    assert_eq!(restored.status(), operator.status());
    let after_restore = restored.checkpoint(Epoch::new(6).unwrap()).unwrap();
    assert!(
        after_restore.segments.contains_key("left-base"),
        "restore must carry the base forward: {:?}",
        after_restore.segments.keys().collect::<Vec<_>>()
    );

    assert_compacted_v2_history(&after_restore);
    assert_eq!(after_restore.segments, compacted.segments);
    assert_shared_segments(&compacted, &after_restore);

    // The chain restores again from the carried checkpoint.
    let (mut restored_again, job_three, collector_three) = state_operator(300);
    let _ = (&job_three, collector_three);
    restored_again.restore(&after_restore).unwrap();
    let final_snapshot = restored_again.checkpoint(Epoch::new(7).unwrap()).unwrap();
    assert!(final_snapshot.segments.contains_key("left-base"));
    assert_compacted_v2_history(&final_snapshot);
    assert_eq!(restored_again.status(), operator.status());
    assert_eq!(final_snapshot.segments, compacted.segments);
    assert_shared_segments(&after_restore, &final_snapshot);

    // The compacted state still matches newly processed right rows.
    let (mut matcher, job_four, _) = state_operator(300);
    let _ = job_four;
    matcher.restore(&final_snapshot).unwrap();
    let context_four = StreamOperatorContext::new(&job_four, "match", None);
    let mut matcher_collector = EdgeCollector::new(matcher.output_ports().to_vec());
    matcher
        .process_data(
            "right",
            keyed_batch(&[("a", 100 * SECOND)]),
            &context_four,
            &mut matcher_collector,
        )
        .await
        .unwrap();
    // Original stamps are 1..4 seconds plus 100 microseconds; all four match
    // the 100-second right row, while the 500-second row is outside.
    let matched_rows = matcher_collector
        .drain("output")
        .iter()
        .map(|message| message.as_data().unwrap().num_rows())
        .sum::<usize>();
    assert_eq!(matched_rows, 4);
}
