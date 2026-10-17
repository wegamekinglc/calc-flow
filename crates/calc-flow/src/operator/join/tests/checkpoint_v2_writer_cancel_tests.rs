use super::*;
use std::sync::{Mutex, mpsc};

struct ReleaseWorker(Option<mpsc::Sender<()>>);

impl ReleaseWorker {
    fn release(&mut self) {
        self.0.take().unwrap().send(()).unwrap();
    }
}

impl Drop for ReleaseWorker {
    fn drop(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

struct WriterGate {
    hook: checkpoint_v2::WriterTestHook,
    entered: tokio::sync::oneshot::Receiver<(usize, bool)>,
    release: ReleaseWorker,
}

fn writer_gate() -> WriterGate {
    let (entered, started) = tokio::sync::oneshot::channel();
    let entered = Mutex::new(Some(entered));
    let (release, wait) = mpsc::channel();
    let wait = Mutex::new(wait);
    let hook = Arc::new(move |credit: &MemoryReservation| {
        let native = std::thread::current()
            .name()
            .is_some_and(|name| name == "calc-flow-gather");
        entered
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send((credit.size(), native))
            .unwrap();
        wait.lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(10))
            .unwrap();
    });
    WriterGate {
        hook,
        entered: started,
        release: ReleaseWorker(Some(release)),
    }
}

async fn seed(context: &StreamOperatorContext<'_>) -> StreamJoinOperator {
    let mut join = operator();
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    let input = record(&[95], &[None], &["red"], vec![Some(vec![Some(1), None])]);
    join.process_data(
        "left",
        Batch::table(vec![input], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
    assert!(output.drain("output").is_empty());
    join
}

fn assert_pending(join: &StreamJoinOperator) {
    // The dirty window stays derivable: the one live left row sits above the
    // watermark and encodes as the next delta's only upsert (spec FR3).
    assert_eq!(join.state.deltas.watermark[0], 0);
    assert_eq!(join.state.deltas.dirty[0], 1);
    assert!(join.state.deltas.tombstones[0].is_empty());
    let row = join.state.left.at(0);
    assert_eq!(
        (row.row_id, row.event_time.as_micros(), row.charge),
        (0, 95, 136)
    );
    assert_eq!(row.encoded_key.as_slice(), KEY);
}

fn assert_uninstalled(
    join: &StreamJoinOperator,
    previous: &StreamJoinStatus,
    rows: *const StateRows,
) {
    assert_eq!(join.status(), *previous);
    assert_eq!(Arc::as_ptr(&join.state.left.0), rows);
    assert_eq!((join.state.left.len(), join.state.right.len()), (1, 0));
    assert_eq!(
        (
            join.state.next_left_row_id,
            join.state.next_right_row_id,
            join.state.next_output_sequence,
        ),
        (1, 0, 0)
    );
    assert_eq!(join.state.last_checkpoint_epoch, None);
    assert!(!join.state.ended);
    assert!(join.state.deltas.base.is_empty());
    assert!(join.state.deltas.segments.is_empty());
    assert_pending(join);
}

async fn cancelled_writer(service: &crate::runtime::streaming::gather_work::TestService) {
    let job = job().with_gather_owner(service.owner("v2-writer-cancel".into()));
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut join = seed(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let previous = join.status();
    let rows = Arc::downgrade(&join.state.left.0);
    let column = Arc::downgrade(join.state.left.at(0).record.column(0));
    let WriterGate {
        hook,
        entered,
        mut release,
    } = writer_gate();
    join.set_checkpoint_writer_test_hook(hook);
    let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    let (paid, native) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            entered = entered => entered.unwrap(),
            result = preparing.as_mut() => panic!("writer completed before release: {result:?}"),
        }
    })
    .await
    .unwrap();
    job.cancellation().cancel();
    drop(preparing);
    assert_uninstalled(&join, &previous, rows.as_ptr());
    drop(join);

    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    let pending = futures::poll!(drain.as_mut()).is_pending();
    let retained_inputs = rows.upgrade().is_some() && column.upgrade().is_some();
    let (_, _, attempt) = job.gather_owner().funding();
    let reserved = pool.reserved();
    release.release();
    let failures = tokio::time::timeout(Duration::from_secs(5), drain)
        .await
        .unwrap();
    assert!(native);
    assert!(paid > 0);
    assert!(pending);
    assert!(retained_inputs);
    assert!(attempt > 0);
    assert!(reserved >= attempt);
    assert!(failures.is_empty(), "{failures:?}");
    assert!(rows.upgrade().is_none());
    assert!(column.upgrade().is_none());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_v2_writer_cancel_retains_accepted_inputs_until_actual_drain() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(cancelled_writer(service));
    });
}

fn dirty_record() -> RecordBatch {
    let input = record(&[97], &[Some("ok")], &["red"], vec![Some(vec![Some(9)])]);
    let key: ArrayRef = Arc::new(Int64Array::from(vec![11]));
    RecordBatch::try_new(
        input.schema(),
        std::iter::once(key)
            .chain(input.columns()[1..].iter().cloned())
            .collect(),
    )
    .unwrap()
}

async fn seed_dirty(
    context: &StreamOperatorContext<'_>,
) -> (StreamJoinOperator, OperatorStateSnapshot) {
    let mut join = seed(context).await;
    let first = join.checkpoint(Epoch::INITIAL).unwrap();
    assert!(join.state.deltas.is_clean());
    let mut output = EdgeCollector::new(join.output_ports().to_vec());
    join.process_data(
        "left",
        Batch::table(vec![dirty_record()], BatchMetadata::default()).unwrap(),
        context,
        &mut output,
    )
    .await
    .unwrap();
    assert!(output.drain("output").is_empty());
    (join, first)
}

fn assert_dirty_uninstalled(
    join: &StreamJoinOperator,
    previous: &StreamJoinStatus,
    rows: *const StateRows,
) {
    assert_eq!(join.status(), *previous);
    assert_eq!(Arc::as_ptr(&join.state.left.0), rows);
    assert_eq!((join.state.left.len(), join.state.right.len()), (2, 0));
    assert_eq!(
        (
            join.state.next_left_row_id,
            join.state.next_right_row_id,
            join.state.next_output_sequence,
        ),
        (2, 0, 0)
    );
    assert_eq!(join.state.last_checkpoint_epoch, Some(Epoch::INITIAL));
    assert!(!join.state.ended);
    assert!(join.state.deltas.base.is_empty());
    assert!(join.state.deltas.segments.is_empty());
    // The dirty Delta row stays derivable above the watermark, reusing the
    // live row's key and record directly (spec FR3).
    assert_eq!(join.state.deltas.watermark[0], 1);
    assert_eq!(join.state.deltas.dirty[0], 1);
    assert!(join.state.deltas.tombstones[0].is_empty());
    let row = join.state.left.at(1);
    assert_eq!(
        (row.row_id, row.event_time.as_micros(), row.charge),
        (1, 97, 141)
    );
    assert_eq!(*row.record.view(), dirty_record());
    assert_eq!(
        row.encoded_key.as_slice(),
        encode_join_key_v1(&dirty_record(), 0, &[0]).unwrap()
    );
    assert!(Arc::ptr_eq(&row.encoded_key, join.state.left.key_at(1)));
    assert!(Arc::ptr_eq(
        row.record.column(0),
        join.state.left.at(1).record.column(0)
    ));
}

async fn cancelled_delta_writer(service: &crate::runtime::streaming::gather_work::TestService) {
    let job = job().with_gather_owner(service.owner("v2-delta-writer-cancel".into()));
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let (mut join, first) = seed_dirty(&context).await;
    let pool = join.runtime.runtime().unwrap().incremental_memory_pool();
    let previous = join.status();
    let rows = Arc::downgrade(&join.state.left.0);
    let column = Arc::downgrade(join.state.left.at(1).record.column(0));
    let key = Arc::downgrade(join.state.left.key_at(1));
    assert!(!Arc::ptr_eq(
        join.state.left.at(0).record.column(0),
        join.state.left.at(1).record.column(0)
    ));
    assert!(!Arc::ptr_eq(
        join.state.left.key_at(0),
        join.state.left.key_at(1)
    ));
    let segments = first
        .segments
        .values()
        .map(|segment| Arc::downgrade(&segment.bytes_arc()))
        .collect::<Vec<_>>();
    let owners = segments
        .iter()
        .map(std::sync::Weak::strong_count)
        .collect::<Vec<_>>();
    let WriterGate {
        hook,
        entered,
        mut release,
    } = writer_gate();
    join.set_checkpoint_writer_test_hook(hook);
    let mut preparing = Box::pin(join.prepare_checkpoint_async(&context));
    assert!(futures::poll!(preparing.as_mut()).is_pending());
    let (paid, native) = tokio::time::timeout(Duration::from_secs(5), async {
        tokio::select! {
            entered = entered => entered.unwrap(),
            result = preparing.as_mut() => panic!("writer completed before release: {result:?}"),
        }
    })
    .await
    .unwrap();
    job.cancellation().cancel();
    drop(preparing);
    assert_dirty_uninstalled(&join, &previous, rows.as_ptr());
    assert_eq!(
        segments
            .iter()
            .map(std::sync::Weak::strong_count)
            .collect::<Vec<_>>(),
        owners,
        "cancelled Delta preparation must preserve the current Base owners"
    );
    drop(join);
    drop(first);

    let mut drain = Box::pin(job.gather_owner().close_and_drain());
    let pending = futures::poll!(drain.as_mut()).is_pending();
    // The derived Delta work shares the live row state exactly like Base
    // work (spec FR3/FR8): its inputs stay alive until the actual drain.
    let retained_inputs =
        rows.upgrade().is_some() && column.upgrade().is_some() && key.upgrade().is_some();
    let retired_history = segments.iter().all(|segment| segment.upgrade().is_none());
    let (_, _, attempt) = job.gather_owner().funding();
    let reserved = pool.reserved();
    release.release();
    let failures = tokio::time::timeout(Duration::from_secs(5), drain)
        .await
        .unwrap();
    assert!(native);
    assert!(paid > 0);
    assert!(pending);
    assert!(
        retained_inputs,
        "the owned pending copy must retain its inputs"
    );
    assert!(retired_history);
    assert!(attempt > 0);
    assert!(reserved >= attempt);
    assert!(failures.is_empty(), "{failures:?}");
    assert!(column.upgrade().is_none());
    assert!(key.upgrade().is_none());
    let (home, generation, attempt) = job.gather_owner().funding();
    assert_eq!((generation, attempt), (0, 0));
    assert_eq!(pool.reserved(), home);
    drop(context);
    drop(job);
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_v2_delta_writer_cancel_retains_pending_inputs_until_actual_drain() {
    checkpoint_compaction_tests::isolated_checkpoint_test(|service, runtime| {
        runtime.block_on(cancelled_delta_writer(service));
    });
}
