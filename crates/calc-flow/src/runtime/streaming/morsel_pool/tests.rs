use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
    thread,
    time::Duration,
};

use super::{MorselCost, MorselPool, MorselPoolConfig, MorselTicket, MorselUsage};
use crate::{CalcFlowError, Result};

fn config(workers: usize) -> MorselPoolConfig {
    MorselPoolConfig {
        workers,
        max_morsels: 4,
        max_rows: 1_024,
        max_bytes: 4_096,
        inline_rows: 0,
    }
}

fn cost(rows: usize, bytes: usize) -> MorselCost {
    MorselCost {
        rows,
        scratch_bytes: bytes / 2,
        output_bytes: bytes - bytes / 2,
    }
}

fn submission_error<T>(result: Result<MorselTicket<T>>) -> CalcFlowError {
    match result {
        Ok(_) => panic!("submission unexpectedly succeeded"),
        Err(error) => error,
    }
}

fn assert_cancelled<T>(result: Result<T>) {
    match result {
        Err(CalcFlowError::Cancelled { .. }) => {}
        Err(error) => panic!("expected cancellation, got {error}"),
        Ok(_) => panic!("expected cancellation, got success"),
    }
}

#[test]
fn closed_pool_rejects_inline_execution() {
    let mut pool = MorselPool::new(
        "closed-inline",
        MorselPoolConfig {
            inline_rows: 4,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    pool.close_and_join().unwrap();
    assert_cancelled(client.execute(1, cost(1, 8), |value, _| Ok(value)));
}

#[test]
fn admission_accounting_uses_checked_arithmetic() {
    let mut pool = MorselPool::new(
        "overflow",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 2,
            max_rows: usize::MAX,
            max_bytes: usize::MAX,
            inline_rows: 0,
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit((), cost(usize::MAX, 0), move |(), _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let error = submission_error(client.submit((), cost(1, 0), |(), _| Ok(())));
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel.rows"
    ));
    release_tx.send(()).unwrap();
    first.join().unwrap();
    pool.close_and_join().unwrap();
}

#[test]
fn nested_execution_is_rejected_across_pools() {
    let mut outer_pool = MorselPool::new("outer", config(1)).unwrap();
    let mut inner_pool = MorselPool::new("inner", config(1)).unwrap();
    let outer = outer_pool.client();
    let inner = inner_pool.client();
    let Err(error) = outer.execute((), cost(8, 32), move |(), _| {
        inner.execute(1, cost(1, 8), |value, _| Ok(value))
    }) else {
        panic!("cross-pool nested execution unexpectedly succeeded")
    };
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel_pool.nested"
    ));
    outer_pool.close_and_join().unwrap();
    inner_pool.close_and_join().unwrap();
}

#[test]
fn individual_row_and_byte_bounds_are_rejected_before_queueing() {
    let mut pool = MorselPool::new(
        "bounds",
        MorselPoolConfig {
            max_rows: 8,
            max_bytes: 64,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();

    let rows = submission_error(client.submit((), cost(9, 32), |(), _| Ok(())));
    assert!(matches!(
        rows,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel.rows"
    ));
    let bytes = submission_error(client.submit((), cost(8, 65), |(), _| Ok(())));
    assert!(matches!(
        bytes,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel.bytes"
    ));
    assert_eq!(pool.usage(), MorselUsage::default());

    pool.close_and_join().unwrap();
}

#[test]
fn count_admission_waits_for_capacity() {
    assert_aggregate_budget_blocks(
        MorselPoolConfig {
            max_morsels: 1,
            ..config(1)
        },
        cost(2, 32),
        cost(2, 32),
    );
}

#[tokio::test]
async fn async_admission_waits_for_capacity_without_blocking() {
    let mut pool = MorselPool::new(
        "async-admission",
        MorselPoolConfig {
            max_morsels: 1,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit(7, cost(2, 32), move |value, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(value)
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second_client = client.clone();
    let second = tokio::spawn(async move {
        second_client
            .submit_async(9, cost(2, 32), |value, _| Ok(value + 1))
            .await
            .unwrap()
    });
    tokio::task::yield_now().await;
    assert!(!second.is_finished(), "a full pool must hold admission");

    release_tx.send(()).unwrap();
    assert_eq!(first.join_async().await.unwrap(), 7);
    let second = second.await.unwrap();
    assert_eq!(second.join_async().await.unwrap(), 10);
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn row_admission_waits_for_capacity() {
    assert_aggregate_budget_blocks(
        MorselPoolConfig {
            max_morsels: 2,
            max_rows: 8,
            ..config(1)
        },
        cost(6, 32),
        cost(6, 32),
    );
}

#[test]
fn byte_admission_waits_for_capacity() {
    assert_aggregate_budget_blocks(
        MorselPoolConfig {
            max_morsels: 2,
            max_bytes: 64,
            ..config(1)
        },
        cost(2, 48),
        cost(2, 48),
    );
}

fn assert_aggregate_budget_blocks(
    config: MorselPoolConfig,
    first_cost: MorselCost,
    second_cost: MorselCost,
) {
    let mut pool = MorselPool::new("aggregate-bounds", config).unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit(7, first_cost, move |value, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(value)
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    let second_client = client.clone();
    let (calling_tx, calling_rx) = mpsc::channel();
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let second_thread = thread::spawn(move || {
        calling_tx.send(()).unwrap();
        let ticket = second_client
            .submit(9, second_cost, |value, _| Ok(value + 1))
            .unwrap();
        submitted_tx.send(ticket).unwrap();
    });
    calling_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert!(
        submitted_rx
            .recv_timeout(Duration::from_millis(50))
            .is_err(),
        "a full job budget must hold the next morsel outside the queue"
    );

    release_tx.send(()).unwrap();
    assert_eq!(first.join().unwrap(), 7);
    let second = submitted_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    assert_eq!(second.join().unwrap(), 10);
    second_thread.join().unwrap();
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn small_execute_runs_inline_without_pool_admission() {
    let mut pool = MorselPool::new(
        "inline",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 1,
            inline_rows: 4,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let large = client
        .submit(1, cost(8, 32), move |value, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(value)
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let before = pool.usage();

    let caller = thread::current().id();
    let (executed, value) = client
        .execute(21, cost(4, 32), |value, _| {
            Ok((thread::current().id(), value * 2))
        })
        .unwrap();
    assert_eq!(executed, caller);
    assert_eq!(value, 42);
    assert_eq!(pool.usage(), before);

    release_tx.send(()).unwrap();
    assert_eq!(large.join().unwrap(), 1);
    pool.close_and_join().unwrap();
}

#[tokio::test]
async fn async_ticket_join_does_not_block_the_caller() {
    let mut pool = MorselPool::new("async-join", config(1)).unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let ticket = client
        .submit(7, cost(8, 32), move |value, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(value * 2)
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    let mut joined = Box::pin(ticket.join_async());
    assert!(futures::poll!(&mut joined).is_pending());
    release_tx.send(()).unwrap();
    let value = tokio::time::timeout(Duration::from_secs(2), joined)
        .await
        .expect("async ticket join must settle")
        .unwrap();
    assert_eq!(value, 14);
    pool.close_and_join().unwrap();
}

#[test]
fn queued_ticket_cancellation_settles_without_running_work() {
    let mut pool = MorselPool::new(
        "queued-cancel",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 2,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit(3, cost(8, 32), move |value, _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(value)
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let second_started = Arc::clone(&started);
    let second = client
        .submit(4, cost(8, 32), move |value, _| {
            second_started.store(true, Ordering::SeqCst);
            Ok(value)
        })
        .unwrap();

    second.cancel();
    assert_cancelled(second.join());
    assert!(!started.load(Ordering::SeqCst));
    release_tx.send(()).unwrap();
    assert_eq!(first.join().unwrap(), 3);
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn pool_cancellation_signals_running_lease_and_teardown_joins() {
    let mut pool = MorselPool::new("running-cancel", config(1)).unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let ticket = client
        .submit((), cost(8, 32), move |(), lease| -> Result<usize> {
            entered_tx.send(()).unwrap();
            loop {
                lease.check()?;
                thread::yield_now();
            }
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    pool.cancellation().cancel();
    pool.close_and_join().unwrap();
    assert_cancelled(ticket.join());
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn abandoned_running_ticket_is_settled_and_refunded_by_teardown() {
    let mut pool = MorselPool::new("abandoned", config(1)).unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let ticket = client
        .submit((), cost(8, 32), move |(), lease| -> Result<usize> {
            entered_tx.send(()).unwrap();
            loop {
                lease.check()?;
                thread::yield_now();
            }
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();

    drop(ticket);
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn ordinary_failure_cancels_siblings_and_refunds_reservations() {
    let mut pool = MorselPool::new(
        "ordinary-failure",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 2,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let failed = client
        .submit((), cost(8, 32), move |(), _| -> Result<usize> {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Err(CalcFlowError::Internal {
                message: "injected ordinary morsel failure".into(),
            })
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let sibling_started = Arc::clone(&started);
    let sibling = client
        .submit(1, cost(8, 32), move |value, _| {
            sibling_started.store(true, Ordering::SeqCst);
            Ok(value)
        })
        .unwrap();

    release_tx.send(()).unwrap();
    let Err(error) = failed.join() else {
        panic!("injected failure unexpectedly succeeded")
    };
    assert!(matches!(
        error,
        CalcFlowError::Internal { ref message } if message.contains("injected ordinary")
    ));
    assert_cancelled(sibling.join());
    assert!(!started.load(Ordering::SeqCst));
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn panic_becomes_task_failure_cancels_siblings_and_refunds_reservations() {
    let mut pool = MorselPool::new(
        "panic-failure",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 2,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let failed = client
        .submit((), cost(8, 32), move |(), _| -> Result<usize> {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            panic!("injected morsel panic")
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let started = Arc::new(AtomicBool::new(false));
    let sibling_started = Arc::clone(&started);
    let sibling = client
        .submit(1, cost(8, 32), move |value, _| {
            sibling_started.store(true, Ordering::SeqCst);
            Ok(value)
        })
        .unwrap();

    release_tx.send(()).unwrap();
    let Err(error) = failed.join() else {
        panic!("injected panic unexpectedly succeeded")
    };
    assert!(matches!(
        error,
        CalcFlowError::TaskPanicked { ref message, .. } if message.contains("injected morsel panic")
    ));
    assert_cancelled(sibling.join());
    assert!(!started.load(Ordering::SeqCst));
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn nested_large_execute_is_rejected_without_waiting() {
    let mut pool = MorselPool::new(
        "nested-execute",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 1,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let nested = client.clone();

    let result = client.execute((), cost(8, 32), move |(), _| {
        nested.execute(1, cost(8, 32), |value, _| Ok(value))
    });
    let Err(error) = result else {
        panic!("nested pool execution unexpectedly succeeded")
    };
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel_pool.nested"
    ));
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn nested_ticket_join_is_rejected_without_waiting() {
    let mut pool = MorselPool::new(
        "nested-join",
        MorselPoolConfig {
            workers: 1,
            max_morsels: 2,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (ticket_tx, ticket_rx) = mpsc::channel::<MorselTicket<usize>>();
    let outer = client
        .submit((), cost(8, 32), move |(), _| {
            entered_tx.send(()).unwrap();
            ticket_rx.recv().unwrap().join()
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let inner = client.submit(9, cost(8, 32), |value, _| Ok(value)).unwrap();
    ticket_tx.send(inner).unwrap();

    let Err(error) = outer.join() else {
        panic!("nested ticket join unexpectedly succeeded")
    };
    assert!(matches!(
        error,
        CalcFlowError::InvalidArgument { ref field, .. } if field == "morsel_pool.nested"
    ));
    pool.close_and_join().unwrap();
    assert_eq!(pool.usage(), MorselUsage::default());
}

#[test]
fn settled_ticket_can_join_after_pool_drop() {
    let mut pool = MorselPool::new("ticket-drop", config(1)).unwrap();
    let client = pool.client();
    let (done_tx, done_rx) = mpsc::channel();
    let ticket = client
        .submit(7, cost(1, 8), move |value, _| {
            done_tx.send(()).unwrap();
            Ok(value)
        })
        .unwrap();
    done_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    pool.close_and_join().unwrap();
    drop(client);
    drop(pool);
    assert_eq!(ticket.join().unwrap(), 7);
}

#[test]
fn direct_cancellation_wakes_blocked_admission() {
    let mut pool = MorselPool::new(
        "cancel-admission",
        MorselPoolConfig {
            max_morsels: 1,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit((), cost(8, 32), move |(), _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second_client = client.clone();
    let (calling_tx, calling_rx) = mpsc::channel();
    let (submitted_tx, submitted_rx) = mpsc::channel();
    let second = thread::spawn(move || {
        calling_tx.send(()).unwrap();
        submitted_tx
            .send(second_client.submit((), cost(1, 8), |(), _| Ok(())))
            .unwrap();
    });
    calling_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    pool.cancellation().cancel();
    let outcome = submitted_rx.recv_timeout(Duration::from_secs(2));
    release_tx.send(()).unwrap();
    first.join().unwrap();
    second.join().unwrap();
    pool.close_and_join().unwrap();
    let result = outcome.expect("cancellation did not wake admission");
    assert_cancelled(match result {
        Ok(ticket) => ticket.join(),
        Err(error) => Err(error),
    });
}

#[test]
fn cancellation_synchronizes_with_admission_entering_its_wait() {
    let pool = MorselPool::new("cancel-at-wait-boundary", config(1)).unwrap();
    let state = super::lock(&pool.inner.state);
    let cancellation = pool.cancellation();
    let (started_tx, started_rx) = mpsc::channel();
    let (finished_tx, finished_rx) = mpsc::channel();
    let canceller = thread::spawn(move || {
        started_tx.send(()).unwrap();
        cancellation.cancel();
        finished_tx.send(()).unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let early_notification = finished_rx.recv_timeout(Duration::from_millis(250)).is_ok();

    let (state, wait) = pool
        .inner
        .settled
        .wait_timeout(state, Duration::from_secs(2))
        .unwrap();
    drop(state);
    canceller.join().unwrap();

    assert!(
        !early_notification,
        "cancellation notified before admission could wait"
    );
    assert!(
        !wait.timed_out(),
        "admission did not receive the cancellation"
    );
}

#[test]
fn queued_cancellation_contains_input_destructor_panics() {
    struct PanicDrop;
    impl Drop for PanicDrop {
        fn drop(&mut self) {
            panic!("morsel input drop panic")
        }
    }

    let mut pool = MorselPool::new(
        "drop-panic",
        MorselPoolConfig {
            max_morsels: 2,
            ..config(1)
        },
    )
    .unwrap();
    let client = pool.client();
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let first = client
        .submit((), cost(8, 32), move |(), _| {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();
    entered_rx.recv_timeout(Duration::from_secs(2)).unwrap();
    let second = client
        .submit(PanicDrop, cost(8, 32), |_, _| Ok(()))
        .unwrap();
    second.cancel();
    let Err(CalcFlowError::TaskPanicked { message, .. }) = second.join() else {
        panic!("destructor panic was not contained as a task failure")
    };
    assert!(message.contains("morsel input drop panic"));
    release_tx.send(()).unwrap();
    first.join().unwrap();
    pool.close_and_join().unwrap();
}
