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
