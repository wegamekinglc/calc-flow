use super::*;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::{cell::Cell, sync::Mutex};

#[derive(Debug, Default)]
struct Fault {
    armed: bool,
    reject: Option<usize>,
    admissions: Vec<usize>,
}

#[derive(Debug)]
struct RestorePool {
    inner: GreedyMemoryPool,
    fault: Mutex<Fault>,
}

impl RestorePool {
    fn arm(&self, reject: Option<usize>) {
        *self.fault.lock().unwrap() = Fault {
            armed: true,
            reject,
            admissions: Vec::new(),
        };
    }

    fn finish(&self) -> Vec<usize> {
        std::mem::take(&mut *self.fault.lock().unwrap()).admissions
    }
}

impl fmt::Display for RestorePool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.inner, formatter)
    }
}

impl MemoryPool for RestorePool {
    fn name(&self) -> &'static str {
        "v2-restore-fault"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        let mut fault = self.fault.lock().unwrap();
        if fault.armed && bytes > 0 {
            fault.admissions.push(bytes);
            if fault.reject == Some(fault.admissions.len()) {
                return Err(datafusion::common::DataFusionError::ResourcesExhausted(
                    "restore admission rejected".into(),
                ));
            }
        }
        drop(fault);
        self.inner.try_grow(credit, bytes)
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

fn fixture() -> (StreamJoinOperator, OperatorStateSnapshot, Arc<RestorePool>) {
    let pool = Arc::new(RestorePool {
        inner: GreedyMemoryPool::new(16 << 20),
        fault: Mutex::new(Fault::default()),
    });
    let mut operator = operator();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    operator.restore(&snapshot(&operator)).unwrap();
    let target = history_controls::interleaved_snapshot(&operator);
    (operator, target, pool)
}

struct Previous {
    left: Arc<Vec<StoredRow>>,
    right: Arc<Vec<StoredRow>>,
    containers: Arc<checkpoint_v2::ContainerFunding>,
    metrics: StreamJoinStatus,
    paid: usize,
}

impl Previous {
    fn capture(operator: &StreamJoinOperator, pool: &RestorePool) -> Self {
        Self {
            left: Arc::clone(&operator.state.left.0),
            right: Arc::clone(&operator.state.right.0),
            containers: Arc::clone(operator.v2_containers.as_ref().unwrap()),
            metrics: operator.status(),
            paid: pool.reserved(),
        }
    }
    fn assert_retained(&self, operator: &StreamJoinOperator, pool: &RestorePool) {
        assert!(Arc::ptr_eq(&operator.state.left.0, &self.left));
        assert!(Arc::ptr_eq(&operator.state.right.0, &self.right));
        assert!(Arc::ptr_eq(
            operator.v2_containers.as_ref().unwrap(),
            &self.containers
        ));
        assert_eq!(operator.status(), self.metrics);
        assert_restored(operator);
        assert_eq!(pool.reserved(), self.paid);
    }
}

#[test]
fn test_v2_restore_rejects_every_admission_without_replacing_old_state() {
    let (mut operator, target, pool) = fixture();
    pool.arm(None);
    operator.restore(&target).unwrap();
    let admissions = pool.finish();
    operator.restore(&snapshot(&operator)).unwrap();
    let previous = Previous::capture(&operator, &pool);
    assert!(admissions.len() > 20);
    for reject in 1..=admissions.len() {
        pool.arm(Some(reject));
        let result = operator.restore(&target);
        let observed = pool.finish();
        assert!(result.is_err(), "admission {reject}");
        assert_eq!(observed, admissions[..reject]);
        previous.assert_retained(&operator, &pool);
    }
    operator.restore(&target).unwrap();
    assert_eq!(operator.state.left[0].row_id, 1);
    println!(
        "rejected and refunded all {} positive restore admissions",
        admissions.len()
    );
    drop((previous, operator, target));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn test_v2_restore_cancels_at_every_check_without_replacing_old_state() {
    let (mut operator, target, pool) = fixture();
    let count = Cell::new(0);
    operator
        .restore_v2_checked_for_test(&target, &|| {
            count.set(count.get() + 1);
            Ok(())
        })
        .unwrap();
    let checks = count.get();
    operator.restore(&snapshot(&operator)).unwrap();
    let previous = Previous::capture(&operator, &pool);
    for cancel in 1..=checks {
        count.set(0);
        let result = operator.restore_v2_checked_for_test(&target, &|| {
            count.set(count.get() + 1);
            if count.get() >= cancel {
                Err(CalcFlowError::Internal {
                    message: "restore check cancelled".into(),
                })
            } else {
                Ok(())
            }
        });
        assert!(result.is_err(), "check {cancel}/{checks}");
        assert!((cancel..=checks).contains(&count.get()));
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("restore check cancelled")
        );
        previous.assert_retained(&operator, &pool);
    }
    operator.restore(&target).unwrap();
    assert_eq!(operator.state.left[0].row_id, 1);
    println!("cancelled and refunded all {checks} restore checks");
    drop((previous, operator, target));
    assert_eq!(pool.reserved(), 0);
}
