use std::collections::BTreeMap;

use super::*;
use crate::operator::join::columnar::RowPayload;
use crate::{EventTime, IngressProgress, IngressProgressSnapshot, IngressState};

pub(super) async fn retained_sparse_repair() {
    let input = input();
    let original = input.table_payload().unwrap().batches()[0].clone();
    let schema = original.schema();
    let mut projected = operator(&schema, true);
    let mut full = operator(&schema, false);
    let job = StreamJobContext::new(
        1,
        "retained-sparse",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "selected-state", None);
    let mut actual = EdgeCollector::new(projected.output_ports().to_vec());
    let mut expected = EdgeCollector::new(full.output_ports().to_vec());
    projected
        .process_data("left", input.clone(), &context, &mut actual)
        .await
        .unwrap();
    full.process_data("left", input.clone(), &context, &mut expected)
        .await
        .unwrap();
    assert_eq!(projected.status(), full.status());
    let old = projected.state.left.at(0).record.clone();
    let alias = old.column(2).to_data().buffers()[0].clone();
    let progressed = progress(&job);
    projected
        .on_ingress_progress("right", &progressed)
        .await
        .unwrap();
    full.on_ingress_progress("right", &progressed)
        .await
        .unwrap();
    let watermark = projected
        .status()
        .right
        .watermark_micros
        .map(EventTime::as_micros);
    assert_eq!(projected.status(), full.status());
    check_replacement(&projected, &old);
    let probe = Batch::table(vec![original.slice(1023, 1)], BatchMetadata::default()).unwrap();
    projected
        .process_data("right", probe.clone(), &progressed, &mut actual)
        .await
        .unwrap();
    full.process_data("right", probe, &progressed, &mut expected)
        .await
        .unwrap();
    assert_eq!(projected.status(), full.status());
    assert_eq!(projected.status().emitted_match_rows, 1);
    compare_output(&mut actual, &mut expected);
    assert_eq!(input.table_payload().unwrap().batches()[0], original);
    projected
        .prepare_checkpoint_async(&progressed)
        .await
        .unwrap();
    let mut snapshot = projected.checkpoint(crate::Epoch::INITIAL).unwrap();
    assert_eq!(snapshot.inline_metadata["layout_version"], 3);
    let mut restored = operator(&schema, true);
    restored
        .restore_managed_snapshot(&mut snapshot, &job, None)
        .await
        .unwrap();
    assert_eq!(
        restored
            .status()
            .with_ingress_progress(progressed.ingress_progress()),
        projected.status(),
    );
    post_restore_gap(&mut restored, &mut full, &original, &progressed).await;
    let pools = [&mut projected, &mut full, &mut restored].map(|operator| {
        operator
            .runtime
            .runtime()
            .unwrap()
            .incremental_memory_pool()
    });
    drop((
        old, projected, full, restored, snapshot, actual, expected, context, progressed,
    ));
    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    assert!(
        futures::poll!(drain.as_mut()).is_pending(),
        "old sparse buffer still owns retirement"
    );
    drop(alias);
    assert!(drain.await.is_empty());
    drop(job);
    assert!(pools.iter().all(|pool| pool.reserved() == 0));
    assert_eq!(
        watermark,
        Some(1023),
        "sparse repair must publish the handler ingress progress"
    );
}

async fn post_restore_gap(
    restored: &mut StreamJoinOperator,
    full: &mut StreamJoinOperator,
    original: &RecordBatch,
    context: &StreamOperatorContext<'_>,
) {
    let source = Batch::table(
        vec![original.slice(0, 1), original.slice(1023, 1)],
        BatchMetadata::default(),
    )
    .unwrap();
    let mut actual = EdgeCollector::new(restored.output_ports().to_vec());
    let mut expected = EdgeCollector::new(full.output_ports().to_vec());
    restored
        .process_data("right", source.clone(), context, &mut actual)
        .await
        .unwrap();
    full.process_data("right", source, context, &mut expected)
        .await
        .unwrap();
    assert_eq!(restored.status(), full.status());
    assert_eq!(
        restored
            .state
            .right
            .iter()
            .map(|row| row.row_id)
            .collect::<Vec<_>>(),
        [0, 2]
    );
    assert_eq!(restored.state.next_right_row_id, 3);
    assert_eq!(restored.state.metrics.right.late_rows, 1);
    assert_eq!(restored.state.next_output_sequence, 2);
    assert_eq!(
        restored.state.right.at(1).charge,
        full.state.right.at(1).charge
    );
    compare_output(&mut actual, &mut expected);
}

fn progress(job: &StreamJobContext) -> StreamOperatorContext<'_> {
    StreamOperatorContext::with_ingress_progress(
        job,
        "selected-state",
        None,
        IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, None),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1023))),
            ),
        ])),
    )
}

fn input() -> Batch {
    let times = (0..1024)
        .map(|row| if row == 1023 { 1024 } else { row })
        .collect::<Vec<_>>();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from_iter_values(
            (0..1024).map(|_| "unused".repeat(128)),
        )),
        Arc::new(Int64Array::from((0..1024).collect::<Vec<_>>())),
        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
        Arc::new(Int64Array::from((0..1024).collect::<Vec<_>>())),
    ];
    let fields = ["unused", "key", "time", "value"]
        .into_iter()
        .zip(&arrays)
        .map(|(name, array)| Field::new(name, array.data_type().clone(), false))
        .collect::<Vec<_>>();
    let record = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
    Batch::table(vec![record], BatchMetadata::default()).unwrap()
}

fn operator(schema: &Arc<Schema>, projected: bool) -> StreamJoinOperator {
    let spec = StreamJoinSpec::inner(
        ["key"],
        ["key"],
        "time",
        "time",
        JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        JoinStateLimits::new(2048, 4_000_000, 2048).unwrap(),
    )
    .unwrap();
    let mut operator = StreamJoinOperator::new(
        "selected-state",
        Arc::clone(schema),
        Arc::clone(schema),
        spec,
    )
    .unwrap();
    if projected {
        operator.set_output_projection(vec![3, 7]).unwrap();
    }
    operator
}

fn check_replacement(join: &StreamJoinOperator, old: &RowPayload) {
    assert_eq!(join.state.left.len(), 1);
    let row = &join.state.left.at(0);
    assert_eq!((row.row_id, row.event_time.as_micros()), (1023, 1024));
    assert_eq!(row.record.num_columns(), 3);
    assert_eq!(row.record.offset(), 0);
    for column in [0, 2] {
        assert_eq!(
            row.record
                .column(column)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .value(0),
            1023
        );
    }
    assert_eq!(
        row.record
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .value(0),
        1024
    );
    assert_ne!(row.record.funded_owner(), old.funded_owner());
    let RowPayload::Shared { chunk: old, .. } = old else {
        panic!("native old chunk")
    };
    let RowPayload::Shared { chunk: new, .. } = &row.record else {
        panic!("native replacement")
    };
    assert!(!Arc::ptr_eq(old, new));
    // The dirty window derives from the live rows (spec FR3), so the
    // repaired row binds its own funded payload directly.
    assert_eq!(row.row_id, 1023);
    assert!(row.record.funded_owner().is_some());
    assert_eq!(row.record.offset(), 0);
}
