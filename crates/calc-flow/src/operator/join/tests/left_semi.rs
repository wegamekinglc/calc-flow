use super::*;
use crate::StreamingFailureReason;
use datafusion::arrow::{
    array::{ArrayRef, UInt8Array, UInt32Array},
    ipc::{reader::FileReader, writer::FileWriter},
};
use std::io::Cursor;

fn operator() -> StreamJoinOperator {
    let mut spec = spec();
    spec.join_type = StreamJoinType::LeftSemi;
    let mut op = StreamJoinOperator::new("match", left_schema(), right_schema(), spec).unwrap();
    op.enable_run_core();
    op
}

fn replace_index_column(
    snapshot: &mut OperatorStateSnapshot,
    page: &str,
    column: usize,
    value: ArrayRef,
) {
    let segment = &snapshot.segments[page];
    let record = FileReader::try_new(Cursor::new(segment.bytes()), None)
        .unwrap()
        .next()
        .unwrap()
        .unwrap();
    let mut columns = record.columns().to_vec();
    columns[column] = value;
    let record = RecordBatch::try_new(record.schema(), columns).unwrap();
    replace_index_record(snapshot, page, &record);
}

fn replace_index_record(snapshot: &mut OperatorStateSnapshot, page: &str, record: &RecordBatch) {
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &record.schema()).unwrap();
        writer.write(record).unwrap();
        writer.finish().unwrap();
    }
    snapshot
        .segments
        .insert(page.into(), StateSegment::new(bytes));
}

async fn assert_native_index_rejected(change: impl FnOnce(&mut OperatorStateSnapshot)) {
    let mut op = operator();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data(
        "left",
        left_batch_with_keys(vec![7, 8], vec![100, 200]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let mut store = crate::state::v4::V4Store::new(temporary.path().join("lineage"));
    let first = persist_native_epoch(&mut op, &mut store, 1).await;
    let mut original = crate::state::v4::native::load_native(&mut store, &first, "match").unwrap();
    let mut changed = original.clone();
    change(&mut changed);
    assert!(matches!(
        op.restore(&changed),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
    let second = persist_native_epoch(&mut op, &mut store, 2).await;
    let current = crate::state::v4::native::load_native(&mut store, &second, "match").unwrap();
    original.inline_metadata.insert("epoch".into(), 2.into());
    assert_eq!(current, original);
}

#[tokio::test]
async fn test_left_semi_restored_expiration_requires_the_owner_group() {
    assert_native_index_rejected(|changed| {
        replace_index_column(
            changed,
            "left/expiration",
            1,
            Arc::new(UInt32Array::from(vec![1, 0])),
        );
    })
    .await;
}

#[tokio::test]
async fn test_left_semi_restored_bucket_requires_a_complete_index_schema() {
    assert_native_index_rejected(|changed| {
        let partial =
            RecordBatch::try_from_iter([("kind", Arc::new(UInt8Array::from(vec![0])) as ArrayRef)])
                .unwrap();
        replace_index_record(changed, "left/group-0", &partial);
    })
    .await;
}

#[tokio::test]
async fn test_left_semi_managed_cuts_and_corrupt_witness_pages() {
    let mut original = operator();
    let job = job().with_checkpointing(false);
    let context = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(original.output_ports().to_vec());
    let times = [100_000_000, 1_000_000_000];
    original
        .process_data("left", left_batch(times.into()), &context, &mut output)
        .await
        .unwrap();
    let temporary = tempfile::tempdir().unwrap();
    let mut store = crate::state::v4::V4Store::new(temporary.path().join("lineage"));
    let first = persist_native_epoch(&mut original, &mut store, 1).await;
    original
        .process_data("right", right_batch(vec![times[0]]), &context, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    let second = persist_native_epoch(&mut original, &mut store, 2).await;
    let load =
        |store: &mut _, cut| crate::state::v4::native::load_native(store, cut, "match").unwrap();
    let previous = load(&mut store, &first);
    let current = load(&mut store, &second);
    assert_eq!(current.inline_metadata["layout_version"], 13);
    assert_ne!(
        previous.segments["left/group-0"].sha256(),
        current.segments["left/group-0"].sha256()
    );
    assert_eq!(
        previous.segments["left/expiration"].sha256(),
        current.segments["left/expiration"].sha256()
    );
    for (cut, sequence, expected) in [(&previous, 0, times.as_slice()), (&current, 1, &times[1..])]
    {
        let mut restored = operator();
        restored.restore(cut).unwrap();
        restored
            .process_data("right", right_batch(times.into()), &context, &mut output)
            .await
            .unwrap();
        let messages = output.drain("output");
        let batch = messages[0].as_data().unwrap();
        assert_eq!(batch.metadata().sequence(), sequence);
        assert_eq!(batch.num_rows(), expected.len());
        let rows = &batch.table_payload().unwrap().batches()[0];
        let actual = rows
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap();
        assert_eq!(actual.values().as_ref(), expected);
    }
    let mut restored = operator();
    restored.restore(&current).unwrap();
    let before = restored.status();
    for (page, col, values) in [
        (
            "right/group-0",
            0,
            Arc::new(UInt8Array::from(vec![2])) as ArrayRef,
        ),
        ("left/expiration", 0, Arc::new(UInt8Array::from(vec![2, 1]))),
        ("left/group-0", 0, Arc::new(UInt8Array::from(vec![8, 0]))),
        (
            "left/group-0",
            5,
            Arc::new(UInt32Array::from(vec![65_536, 1])),
        ),
        ("left/group-0", 5, Arc::new(UInt32Array::from(vec![1, 0]))),
    ] {
        let mut bad = current.clone();
        replace_index_column(&mut bad, page, col, values);
        assert!(
            matches!(
                restored.restore(&bad),
                Err(CalcFlowError::CheckpointMismatch { .. })
            ),
            "{page} column {col}"
        );
        assert_eq!(restored.status(), before);
    }
    let mut inner =
        StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
    let mut incompatible = current;
    incompatible
        .inline_metadata
        .insert("layout_version".into(), 12.into());
    incompatible
        .inline_metadata
        .insert("spec".into(), serde_json::to_value(spec()).unwrap());
    assert!(matches!(
        inner.restore(&incompatible),
        Err(CalcFlowError::CheckpointMismatch { .. })
    ));
}

#[tokio::test]
async fn test_left_semi_right_idle_boundary_and_eof_preserve_retained_witnesses() {
    let mut op = operator();
    let job = job();
    let open = StreamOperatorContext::new(&job, "match", None);
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![100_000_000]), &open, &mut output)
        .await
        .unwrap();
    for state in [IngressState::Idle, IngressState::Active] {
        let boundary = progress_context(
            &job,
            (IngressState::Active, None),
            (state, Some(160_000_000)),
        );
        op.on_ingress_progress("right", &boundary).await.unwrap();
        assert_eq!(op.status().left.retained_rows, 1);
    }
    op.process_data("right", right_batch(vec![160_000_000]), &open, &mut output)
        .await
        .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    let ended = progress_context(
        &job,
        (IngressState::Active, None),
        (IngressState::Ended, Some(160_000_000)),
    );
    op.on_ingress_progress("right", &ended).await.unwrap();
    assert_eq!(op.status().left.retained_rows, 0);
    op.process_data(
        "left",
        left_batch(vec![100_000_000, 150_000_000, 500_000_000]),
        &ended,
        &mut output,
    )
    .await
    .unwrap();
    let messages = output.drain("output");
    let batch = messages[0].as_data().unwrap();
    let values = batch.table_payload().unwrap().batches()[0]
        .column(1)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .unwrap();
    assert_eq!(values.values().as_ref(), &[100_000_000, 150_000_000]);
    assert_eq!(op.status().left.retained_rows, 0);
}

#[derive(Default)]
struct RejectSecond(Vec<Batch>);

#[async_trait::async_trait]
impl StreamCollector for RejectSecond {
    async fn emit(&mut self, _port: &str, batch: Batch) -> Result<()> {
        if self.0.len() == 1 {
            return Err(CalcFlowError::Internal {
                message: "second output rejected".into(),
            });
        }
        self.0.push(batch);
        Ok(())
    }
}

#[tokio::test]
async fn test_left_semi_failed_prefix_preserves_witnesses_for_retry() {
    let mut op = operator();
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None)
        .with_output_budget(EdgeBudget::new(1, 4096).unwrap());
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.process_data("left", left_batch(vec![10, 10]), &context, &mut output)
        .await
        .unwrap();
    let mut prefix = RejectSecond::default();
    let err = op
        .process_data("right", right_batch(vec![10, 10]), &context, &mut prefix)
        .await
        .unwrap_err();
    assert!(err.to_string().contains("second output rejected"));
    assert_eq!(prefix.0.len(), 1);
    assert_eq!(prefix.0[0].metadata().sequence(), 0);
    assert_eq!(op.status().emitted_match_rows, 0);
    assert_eq!(op.status().right.retained_rows, 0);
    op.process_data("right", right_batch(vec![10, 10]), &context, &mut output)
        .await
        .unwrap();
    let retry = output.drain("output");
    let sequences: Vec<_> = retry
        .iter()
        .map(|m| m.as_data().unwrap().metadata().sequence())
        .collect();
    assert_eq!(sequences, [0, 1]);
    op.process_data("right", right_batch(vec![10]), &context, &mut output)
        .await
        .unwrap();
    assert!(output.drain("output").is_empty());
}

#[tokio::test]
async fn test_left_semi_null_late_and_conversion_precedence() {
    use datafusion::arrow::array::TimestampSecondArray;
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true),
        Field::new("time", DataType::Timestamp(TimeUnit::Second, None), true),
    ]));
    let spec = StreamJoinSpec::new(
        StreamJoinType::LeftSemi,
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(100, 1_000_000, 1).unwrap(),
    )
    .unwrap();
    let mut op = StreamJoinOperator::new("match", schema.clone(), schema.clone(), spec).unwrap();
    op.enable_run_core();
    let batch = |keys, times| {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(TimestampSecondArray::from(times)),
        ];
        Batch::table(
            vec![RecordBatch::try_new(schema.clone(), columns).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap()
    };
    let job = job();
    let context = progress_context(
        &job,
        (IngressState::Active, Some(1_000_000)),
        (IngressState::Active, Some(1_000_000)),
    );
    let mut output = EdgeCollector::new(op.output_ports().to_vec());
    op.on_ingress_progress("left", &context).await.unwrap();
    op.on_ingress_progress("right", &context).await.unwrap();
    op.process_data(
        "left",
        batch(
            vec![None, None, Some(7), Some(7)],
            vec![None, Some(0), Some(0), Some(1)],
        ),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    let left = op.status().left;
    assert_eq!(
        (
            left.null_event_time_rows,
            left.null_key_rows,
            left.late_rows,
            left.retained_rows
        ),
        (1, 1, 1, 1)
    );
    let error = op
        .process_data(
            "right",
            batch(vec![None], vec![Some(i64::MAX)]),
            &context,
            &mut output,
        )
        .await
        .unwrap_err();
    assert!(matches!(
        error,
        CalcFlowError::OperatorReason {
            reason_code: StreamingFailureReason::JoinTimeConversionFailed,
            ..
        }
    ));
    assert_eq!(op.status().right.null_key_rows, 0);
    op.process_data(
        "right",
        batch(vec![None, Some(7), Some(7)], vec![Some(1), None, Some(1)]),
        &context,
        &mut output,
    )
    .await
    .unwrap();
    assert_eq!(output.drain("output")[0].as_data().unwrap().num_rows(), 1);
    let right = op.status().right;
    assert_eq!(
        (
            right.null_event_time_rows,
            right.null_key_rows,
            right.retained_rows
        ),
        (1, 1, 1)
    );
}
