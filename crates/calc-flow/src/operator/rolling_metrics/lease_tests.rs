use super::*;

#[derive(Default)]
struct Clock(Mutex<Duration>);

impl RollingMetricsClock for Clock {
    fn now(&self) -> Duration {
        *self.0.lock()
    }
}

impl Clock {
    fn advance(&self, nanos: u64) {
        *self.0.lock() += Duration::from_nanos(nanos);
    }
}

#[test]
fn dropped_callback_waits_for_numeric_join_and_freezes_its_first_outcome() {
    let clock = Arc::new(Clock::default());
    let store = RollingMetricsStore::with_clock(clock.clone());
    let cancellation = CancellationToken::new();
    let callback = store.begin(RollingCallback::Watermark, cancellation.clone());
    let recorder = callback.recorder();
    let numeric = recorder.stage(RollingStage::NumericUpdate);
    let lease = recorder.numeric_lease(7).expect("one work generation");
    clock.advance(5);
    drop(callback);
    assert_eq!(store.snapshot().watermark.interrupted, 0);
    cancellation.cancel();
    recorder.add(RollingWork::NumericRows, 999);
    drop(recorder.stage(RollingStage::HistoryMaintenance));
    drop(numeric);
    clock.advance(11);
    lease.settle(9, false);
    let snapshot = store.snapshot();
    assert!(!snapshot.overflowed);
    assert_eq!(snapshot.watermark.interrupted, 1);
    assert_eq!(snapshot.watermark.cancelled, 0);
    assert_eq!(snapshot.watermark.numeric_rows, 9);
    assert_eq!(
        snapshot.watermark.numeric_update_duration,
        Duration::from_nanos(5)
    );
    assert_eq!(snapshot.watermark.other_duration, Duration::from_nanos(11));
    assert_eq!(
        snapshot.watermark.callback_duration,
        Duration::from_nanos(16)
    );
}

#[test]
fn joined_numeric_work_does_not_publish_before_emit_and_commit_complete() {
    let clock = Arc::new(Clock::default());
    let store = RollingMetricsStore::with_clock(clock.clone());
    let callback = store.begin(RollingCallback::Data, CancellationToken::new());
    let recorder = callback.recorder();
    let numeric = recorder.stage(RollingStage::NumericUpdate);
    let lease = recorder.numeric_lease(1).expect("reserved generation");
    assert!(recorder.numeric_lease(2).is_none());
    clock.advance(2);
    lease.settle(8, false);
    assert_eq!(store.snapshot().data.succeeded, 0);
    assert_eq!(store.snapshot().data.numeric_rows, 0);
    drop(numeric);
    {
        let _emit = recorder.stage(RollingStage::SendWait);
        clock.advance(3);
    }
    recorder.add(RollingWork::OutputRowsPrepared, 8);
    callback.complete(&Ok(()));
    let snapshot = store.snapshot();
    assert_eq!(snapshot.data.succeeded, 1);
    assert_eq!(snapshot.data.interrupted, 0);
    assert_eq!(snapshot.data.numeric_rows, 8);
    assert_eq!(snapshot.data.output_rows_prepared, 8);
    assert_eq!(
        snapshot.data.numeric_update_duration,
        Duration::from_nanos(2)
    );
    assert_eq!(snapshot.data.send_wait_duration, Duration::from_nanos(3));
    assert_eq!(snapshot.data.callback_duration, Duration::from_nanos(5));
    assert!(recorder.numeric_lease(3).is_none());
}

#[test]
fn explicit_error_then_guard_drop_and_late_cancel_remain_failed_after_join() {
    let store = RollingMetricsStore::default();
    let cancellation = CancellationToken::new();
    let callback = store.begin(RollingCallback::End, cancellation.clone());
    let lease = callback.recorder().numeric_lease(2).expect("work lease");
    callback.complete(&Err(CalcFlowError::Internal {
        message: "selected error".into(),
    }));
    cancellation.cancel();
    assert_eq!(store.snapshot().end.failed, 0);
    lease.settle(17, false);
    let snapshot = store.snapshot();
    assert_eq!(snapshot.end.failed, 1);
    assert_eq!(snapshot.end.cancelled, 0);
    assert_eq!(snapshot.end.interrupted, 0);
    assert_eq!(snapshot.end.numeric_rows, 17);
}

#[test]
fn deferred_numeric_overflow_preserves_previous_complete_observation() {
    for local_overflow in [false, true] {
        let store = RollingMetricsStore::default();
        if !local_overflow {
            store.0.metrics.lock().data.numeric_rows = u64::MAX;
        }
        let callback = store.begin(RollingCallback::Data, CancellationToken::new());
        let lease = callback.recorder().numeric_lease(4).expect("work lease");
        callback.complete(&Ok(()));
        let mut previous = store.snapshot();
        lease.settle(2, local_overflow);
        previous.overflowed = true;
        assert_eq!(store.snapshot(), previous);
    }
}
