use std::sync::atomic::AtomicUsize;

use parking_lot::Mutex;

use super::*;
use crate::{
    CalcFlowError, CancellationToken, Result, runtime::streaming::supervisor::TaskSupervisor,
};

#[derive(Default)]
struct ParentWake(AtomicUsize);

impl Wake for ParentWake {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.0.fetch_add(1, SeqCst);
    }
}

#[derive(Default)]
struct ProbeState {
    polls: AtomicUsize,
    done: AtomicBool,
    self_wake: AtomicBool,
    waker: Mutex<Option<Waker>>,
    peer: Mutex<Option<Waker>>,
}

struct Probe {
    state: Arc<ProbeState>,
    _resource: Arc<()>,
}

impl Future for Probe {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        self.state.polls.fetch_add(1, SeqCst);
        *self.state.waker.lock() = Some(context.waker().clone());
        if self.state.self_wake.load(SeqCst) {
            context.waker().wake_by_ref();
        }
        if let Some(peer) = self.state.peer.lock().as_ref() {
            peer.wake_by_ref();
        }
        if self.state.done.load(SeqCst) {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}

fn probe() -> (Probe, Arc<ProbeState>, Weak<()>) {
    let state = Arc::new(ProbeState::default());
    let resource = Arc::new(());
    let weak = Arc::downgrade(&resource);
    (
        Probe {
            state: state.clone(),
            _resource: resource,
        },
        state,
        weak,
    )
}

fn prepared<F, G>(
    supervisor: &mut TaskSupervisor,
    first: F,
    second: G,
) -> super::super::PreparedPair<F, G>
where
    F: Future<Output = Result<()>> + Send + 'static,
    G: Future<Output = Result<()>> + Send + 'static,
{
    supervisor
        .prepare_pair_with_failure_signals(
            "operator:rolling",
            |_| first,
            "operator:project",
            |_| second,
        )
        .with_readiness()
}

#[test]
fn self_wake_survives_pending_and_each_child_polls_once_per_parent_poll() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, _) = probe();
    let (second, second_state, _) = probe();
    first_state.self_wake.store(true, SeqCst);
    second_state.self_wake.store(true, SeqCst);
    let mut driver = Box::pin(prepared(&mut supervisor, first, second).run(true));
    let wake_counter = Arc::new(ParentWake::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    for expected in 1..=3 {
        wake_counter.0.store(0, SeqCst);
        assert!(driver.as_mut().poll(&mut context).is_pending());
        assert_eq!(first_state.polls.load(SeqCst), expected);
        assert_eq!(second_state.polls.load(SeqCst), expected);
        assert!(
            wake_counter.0.load(SeqCst) > 0,
            "self-wake must request the next parent poll"
        );
    }
}

#[test]
fn upstream_wake_after_downstream_was_polled_is_kept_for_the_next_poll() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, _) = probe();
    let (second, second_state, _) = probe();
    let downstream = second_state.clone();
    let upstream = first_state.clone();
    let observer: Observer = Arc::new(move |phase, _, _| {
        if phase == Phase::BeforeChild(0) {
            *upstream.peer.lock() = downstream.waker.lock().clone();
        }
    });
    let mut driver = Box::pin(
        prepared(&mut supervisor, first, second)
            .observe(observer)
            .run(true),
    );
    let wake_counter = Arc::new(ParentWake::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(second_state.polls.load(SeqCst), 1);
    assert_eq!(first_state.polls.load(SeqCst), 1);
    assert!(
        wake_counter.0.load(SeqCst) > 0,
        "visited downstream must retain its wake"
    );
    wake_counter.0.store(0, SeqCst);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(second_state.polls.load(SeqCst), 2);
    assert_eq!(first_state.polls.load(SeqCst), 1);
    assert_eq!(wake_counter.0.load(SeqCst), 0);
}

#[test]
fn wake_at_each_poll_phase_is_consumed_or_schedules_the_next_poll() {
    for (phase, defer_to_next) in [
        (Phase::BeforeRegister, false),
        (Phase::Registered, false),
        (Phase::Polling, false),
        (Phase::BeforeChild(0), true),
        (Phase::AfterChild(0), true),
        (Phase::BeforeIdle, true),
        (Phase::Idle, true),
        (Phase::Checked, true),
    ] {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let (first, first_state, _) = probe();
        let (second, _, _) = probe();
        let armed = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let observer: Observer = {
            let armed = armed.clone();
            let fired = fired.clone();
            Arc::new(move |at, _, wakers| {
                if at == phase && armed.load(SeqCst) && !fired.swap(true, SeqCst) {
                    wakers[0].wake_by_ref();
                }
            })
        };
        let mut driver = Box::pin(
            prepared(&mut supervisor, first, second)
                .observe(observer)
                .run(true),
        );
        let old_parent = Arc::new(ParentWake::default());
        let old_waker = Waker::from(old_parent.clone());
        assert!(
            driver
                .as_mut()
                .poll(&mut Context::from_waker(&old_waker))
                .is_pending()
        );
        assert_eq!(first_state.polls.load(SeqCst), 1);
        if matches!(phase, Phase::BeforeChild(_) | Phase::AfterChild(_)) {
            first_state.waker.lock().as_ref().unwrap().wake_by_ref();
        }
        old_parent.0.store(0, SeqCst);
        let current_parent = Arc::new(ParentWake::default());
        let current_waker = Waker::from(current_parent.clone());
        let mut context = Context::from_waker(&current_waker);
        armed.store(true, SeqCst);
        assert!(driver.as_mut().poll(&mut context).is_pending());
        assert!(fired.load(SeqCst), "phase {phase:?} must be exercised");
        if defer_to_next {
            assert!(current_parent.0.load(SeqCst) > 0, "lost wake at {phase:?}");
            let before = first_state.polls.load(SeqCst);
            assert!(driver.as_mut().poll(&mut context).is_pending());
            assert_eq!(first_state.polls.load(SeqCst), before + 1, "{phase:?}");
        } else {
            assert_eq!(first_state.polls.load(SeqCst), 2, "{phase:?}");
        }
        if phase != Phase::BeforeRegister {
            assert_eq!(
                old_parent.0.load(SeqCst),
                0,
                "parent registration must be replaced at {phase:?}"
            );
        }
    }
}

#[test]
fn controlled_cross_thread_wake_at_poll_exit_cannot_be_lost() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, _) = probe();
    let (second, _, _) = probe();
    let fired = Arc::new(AtomicBool::new(false));
    let observer: Observer = {
        let fired = fired.clone();
        Arc::new(move |phase, _, wakers| {
            if phase == Phase::BeforeIdle && !fired.swap(true, SeqCst) {
                let waker = wakers[0].clone();
                std::thread::scope(|scope| scope.spawn(move || waker.wake()).join().unwrap());
            }
        })
    };
    let mut driver = Box::pin(
        prepared(&mut supervisor, first, second)
            .observe(observer)
            .run(true),
    );
    let wake_counter = Arc::new(ParentWake::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(first_state.polls.load(SeqCst), 1);
    assert!(wake_counter.0.load(SeqCst) > 0);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    assert_eq!(first_state.polls.load(SeqCst), 2);
}

#[test]
fn completed_child_late_wakes_do_not_repoll_or_reschedule_parent() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, _) = probe();
    let (second, second_state, _) = probe();
    first_state.done.store(true, SeqCst);
    let mut driver = Box::pin(prepared(&mut supervisor, first, second).run(true));
    let wake_counter = Arc::new(ParentWake::default());
    let waker = Waker::from(wake_counter.clone());
    let mut context = Context::from_waker(&waker);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    let completed_waker = first_state.waker.lock().clone().unwrap();
    wake_counter.0.store(0, SeqCst);
    for _ in 0..8 {
        completed_waker.wake_by_ref();
        assert!(driver.as_mut().poll(&mut context).is_pending());
    }
    assert_eq!(wake_counter.0.load(SeqCst), 0);
    assert_eq!(first_state.polls.load(SeqCst), 1);
    assert_eq!(second_state.polls.load(SeqCst), 1);
}

#[test]
fn completion_and_drop_release_parent_waker_shared_state_and_child_resources() {
    for complete in [false, true] {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let (first, first_state, first_resource) = probe();
        let (second, second_state, second_resource) = probe();
        let state = Arc::new(Mutex::new(Weak::<Shared>::new()));
        let observer: Observer = {
            let state = state.clone();
            Arc::new(move |_, shared, _| *state.lock() = Arc::downgrade(shared))
        };
        let mut driver = Box::pin(
            prepared(&mut supervisor, first, second)
                .observe(observer)
                .run(true),
        );
        let parent = Arc::new(ParentWake::default());
        let weak_parent = Arc::downgrade(&parent);
        let parent_waker = Waker::from(parent.clone());
        assert!(
            driver
                .as_mut()
                .poll(&mut Context::from_waker(&parent_waker))
                .is_pending()
        );
        let late_wakers = [
            first_state.waker.lock().clone().unwrap(),
            second_state.waker.lock().clone().unwrap(),
        ];
        if complete {
            first_state.done.store(true, SeqCst);
            second_state.done.store(true, SeqCst);
            for waker in &late_wakers {
                waker.wake_by_ref();
            }
            assert!(
                driver
                    .as_mut()
                    .poll(&mut Context::from_waker(&parent_waker))
                    .is_ready()
            );
        }
        drop(parent_waker);
        drop(parent);
        if !complete {
            assert!(
                weak_parent.upgrade().is_some(),
                "parked pair retains its registered parent"
            );
        }
        drop(driver);
        assert!(weak_parent.upgrade().is_none());
        assert!(state.lock().upgrade().is_none());
        assert!(first_resource.upgrade().is_none());
        assert!(second_resource.upgrade().is_none());
        for waker in late_wakers {
            waker.wake();
        }
        assert!(weak_parent.upgrade().is_none());
        assert!(state.lock().upgrade().is_none());
    }
}

#[test]
fn full_error_wrapper_is_repolled_after_settled_before_cancelling_its_peer() {
    let cancellation = CancellationToken::new();
    let mut supervisor = TaskSupervisor::new(cancellation.clone());
    let peer = cancellation.clone();
    let pair = prepared(
        &mut supervisor,
        async {
            Err(CalcFlowError::Internal {
                message: "rolling failure".into(),
            })
        },
        async move {
            peer.cancelled().await;
            Ok(())
        },
    );
    let mut driver = Box::pin(pair.run(true));
    let parent = Arc::new(ParentWake::default());
    let waker = Waker::from(parent.clone());
    let mut context = Context::from_waker(&waker);
    assert!(driver.as_mut().poll(&mut context).is_pending());
    assert!(supervisor.settled.lock()[&TaskId::new(0)].result.is_err());
    assert!(
        !cancellation.is_cancelled(),
        "wrapper is still in convergence yield"
    );
    assert!(parent.0.load(SeqCst) > 0, "full wrapper must remain ready");
    assert!(
        matches!(driver.as_mut().poll(&mut context), Poll::Ready(ids) if ids == [TaskId::new(0), TaskId::new(1)])
    );
    assert!(cancellation.is_cancelled());
    assert!(supervisor.settled.lock()[&TaskId::new(1)].result.is_ok());
    assert_eq!(
        supervisor.terminal_arbiter.primary_failures(),
        [TaskId::new(0)].into()
    );
}

#[tokio::test]
async fn ready_strategy_preserves_both_simultaneous_error_identities() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let pair = prepared(
        &mut supervisor,
        async {
            Err(CalcFlowError::Internal {
                message: "rolling failure".into(),
            })
        },
        async {
            Err(CalcFlowError::Internal {
                message: "projection failure".into(),
            })
        },
    );
    supervisor.spawn_prepared_pair(pair);
    assert_eq!(supervisor.task_count(), 2);
    assert_eq!(supervisor.physical_driver_count(), 1);
    let report = supervisor.join_all().await;
    assert_eq!(report.primary_errors().len(), 2);
    assert_eq!(
        report
            .errors
            .iter()
            .map(|error| error.task_id)
            .collect::<Vec<_>>(),
        [TaskId::new(0), TaskId::new(1)]
    );
    assert_eq!(supervisor.task_count(), 0);
}

#[tokio::test]
async fn ready_strategy_abort_preserves_settled_ok_error_and_panic() {
    for outcome in ["ok", "error", "panic"] {
        let mut supervisor = TaskSupervisor::new(CancellationToken::new());
        let (entered, observed) = tokio::sync::oneshot::channel();
        let pair = prepared(
            &mut supervisor,
            async move {
                entered.send(()).unwrap();
                match outcome {
                    "ok" => Ok(()),
                    "error" => Err(CalcFlowError::Internal {
                        message: "original rolling failure".into(),
                    }),
                    _ => panic!("original rolling panic"),
                }
            },
            std::future::pending(),
        );
        supervisor.spawn_prepared_pair(pair);
        observed.await.unwrap();
        supervisor.tasks.abort_all();
        let report = supervisor.join_all().await;
        let completed = report
            .errors
            .iter()
            .find(|error| error.task_id == TaskId::new(0));
        match outcome {
            "ok" => assert!(completed.is_none()),
            "error" => assert!(
                matches!(&completed.unwrap().error, CalcFlowError::Internal { message } if message == "original rolling failure")
            ),
            _ => assert!(
                matches!(&completed.unwrap().error, CalcFlowError::TaskPanicked { task_id: 0, message } if message == "original rolling panic")
            ),
        }
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.task_id == TaskId::new(1))
        );
        assert!(supervisor.settled.lock().is_empty());
        assert_eq!(supervisor.task_count(), 0);
    }
}

#[test]
fn wake_before_the_initial_parent_registration_is_not_lost() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, _) = probe();
    let (second, second_state, _) = probe();
    let observer: Observer = Arc::new(|phase, _, wakers| {
        if phase == Phase::BeforeRegister {
            wakers[0].wake_by_ref();
        }
    });
    let mut driver = Box::pin(
        prepared(&mut supervisor, first, second)
            .observe(observer)
            .run(true),
    );
    let parent = Arc::new(ParentWake::default());
    let waker = Waker::from(parent.clone());
    assert!(
        driver
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    assert_eq!(first_state.polls.load(SeqCst), 1);
    assert_eq!(second_state.polls.load(SeqCst), 1);
    assert_eq!(parent.0.load(SeqCst), 0);
}

#[tokio::test]
async fn actual_driver_abort_releases_shared_state_and_both_child_resources() {
    let mut supervisor = TaskSupervisor::new(CancellationToken::new());
    let (first, first_state, first_resource) = probe();
    let (second, second_state, second_resource) = probe();
    let state = Arc::new(Mutex::new(Weak::<Shared>::new()));
    let observer: Observer = {
        let state = state.clone();
        Arc::new(move |_, shared, _| *state.lock() = Arc::downgrade(shared))
    };
    let pair = prepared(&mut supervisor, first, second).observe(observer);
    supervisor.spawn_prepared_pair(pair);
    for _ in 0..32 {
        if first_state.polls.load(SeqCst) > 0 && second_state.polls.load(SeqCst) > 0 {
            break;
        }
        tokio::task::yield_now().await;
    }
    let retained = [
        first_state.waker.lock().clone().unwrap(),
        second_state.waker.lock().clone().unwrap(),
    ];
    assert!(state.lock().upgrade().is_some());
    supervisor.tasks.abort_all();
    let report = supervisor.join_all().await;
    assert_eq!(report.errors.len(), 2);
    assert_eq!(supervisor.task_count(), 0);
    assert!(state.lock().upgrade().is_none());
    assert!(first_resource.upgrade().is_none());
    assert!(second_resource.upgrade().is_none());
    for waker in retained {
        waker.wake();
    }
    assert!(state.lock().upgrade().is_none());
    assert_eq!(supervisor.task_count(), 0);
}
