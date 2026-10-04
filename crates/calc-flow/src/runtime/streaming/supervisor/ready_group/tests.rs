use std::sync::atomic::AtomicUsize;

use parking_lot::Mutex;

use super::*;

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
struct State {
    polls: AtomicUsize,
    done: AtomicBool,
    waker: Mutex<Option<Waker>>,
}

struct Child {
    id: TaskId,
    state: Arc<State>,
    _resource: Arc<()>,
}

impl Future for Child {
    type Output = TaskId;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<TaskId> {
        self.state.polls.fetch_add(1, SeqCst);
        *self.state.waker.lock() = Some(context.waker().clone());
        if self.state.done.load(SeqCst) {
            Poll::Ready(self.id)
        } else {
            Poll::Pending
        }
    }
}

struct Fixture {
    children: Vec<Pin<Box<dyn Future<Output = TaskId> + Send>>>,
    states: Vec<Arc<State>>,
    resources: Vec<Weak<()>>,
}

fn fixture() -> Fixture {
    let mut fixture = Fixture {
        children: vec![],
        states: vec![],
        resources: vec![],
    };
    for id in [0, 2, 3] {
        let state = Arc::new(State::default());
        let resource = Arc::new(());
        fixture.resources.push(Arc::downgrade(&resource));
        fixture.states.push(state.clone());
        fixture.children.push(Box::pin(Child {
            id: TaskId::new(id),
            state,
            _resource: resource,
        }));
    }
    fixture
}

#[test]
fn a13_three_member_real_native_wake_at_register_poll_and_idle_is_not_lost() {
    for (phase, deferred) in [
        (Phase::BeforeRegister, false),
        (Phase::Registered, false),
        (Phase::Polling, false),
        (Phase::BeforeChild(0), true),
        (Phase::AfterChild(0), true),
        (Phase::BeforeIdle, true),
        (Phase::Idle, true),
        (Phase::Checked, true),
    ] {
        let fixture = fixture();
        let armed = Arc::new(AtomicBool::new(false));
        let fired = Arc::new(AtomicBool::new(false));
        let observer: Observer = {
            let armed = armed.clone();
            let fired = fired.clone();
            Arc::new(move |at, _, wakers| {
                if at == phase && armed.load(SeqCst) && !fired.swap(true, SeqCst) {
                    let wake = wakers[0].clone();
                    std::thread::spawn(move || wake.wake()).join().unwrap();
                }
            })
        };
        let mut driver = Box::pin(run(fixture.children, Some(observer)));
        let parent = Arc::new(ParentWake::default());
        let waker = Waker::from(parent.clone());
        let mut context = Context::from_waker(&waker);
        assert!(driver.as_mut().poll(&mut context).is_pending());
        for state in &fixture.states {
            assert_eq!(state.polls.load(SeqCst), 1);
            state.waker.lock().as_ref().unwrap().wake_by_ref();
        }
        armed.store(true, SeqCst);
        parent.0.store(0, SeqCst);
        assert!(driver.as_mut().poll(&mut context).is_pending());
        assert!(fired.load(SeqCst), "{phase:?}");
        for state in &fixture.states {
            assert_eq!(state.polls.load(SeqCst), 2, "{phase:?}");
        }
        if deferred {
            assert!(parent.0.load(SeqCst) > 0, "{phase:?}");
        }
        assert!(driver.as_mut().poll(&mut context).is_pending());
        assert_eq!(
            fixture.states[0].polls.load(SeqCst),
            2 + usize::from(deferred)
        );
        assert_eq!(fixture.states[1].polls.load(SeqCst), 2);
        assert_eq!(fixture.states[2].polls.load(SeqCst), 2);
        for state in &fixture.states {
            state.done.store(true, SeqCst);
            state.waker.lock().as_ref().unwrap().wake_by_ref();
        }
        assert_eq!(
            driver.as_mut().poll(&mut context),
            Poll::Ready(vec![TaskId::new(0), TaskId::new(2), TaskId::new(3)])
        );
        drop(driver);
        assert!(
            fixture
                .resources
                .iter()
                .all(|weak| weak.upgrade().is_none())
        );
    }
}

#[test]
fn a13_three_member_retained_native_wakers_do_not_keep_dropped_driver_alive() {
    let fixture = fixture();
    let shared = Arc::new(Mutex::new(Weak::<Shared>::new()));
    let observer: Observer = {
        let shared = shared.clone();
        Arc::new(move |_, state, _| *shared.lock() = Arc::downgrade(state))
    };
    let mut driver = Box::pin(run(fixture.children, Some(observer)));
    let parent = Arc::new(ParentWake::default());
    let parent_weak = Arc::downgrade(&parent);
    let waker = Waker::from(parent);
    assert!(
        driver
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    let retained: Vec<_> = fixture
        .states
        .iter()
        .map(|state| state.waker.lock().clone().unwrap())
        .collect();
    drop(driver);
    drop(waker);
    std::thread::spawn(move || {
        for waker in retained {
            waker.wake();
        }
    })
    .join()
    .unwrap();
    assert!(shared.lock().upgrade().is_none());
    assert!(parent_weak.upgrade().is_none());
    assert!(
        fixture
            .resources
            .iter()
            .all(|weak| weak.upgrade().is_none())
    );
}
