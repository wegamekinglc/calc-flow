use super::*;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
pub(super) struct ProbePool {
    inner: GreedyMemoryPool,
    peak: AtomicUsize,
}

impl fmt::Display for ProbePool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("native-probe-funding")
    }
}

impl MemoryPool for ProbePool {
    fn name(&self) -> &'static str {
        "native-probe-funding"
    }
    fn grow(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.grow(credit, bytes);
        self.observe();
    }
    fn shrink(&self, credit: &MemoryReservation, bytes: usize) {
        self.inner.shrink(credit, bytes);
    }
    fn reserved(&self) -> usize {
        self.inner.reserved()
    }
    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
    fn try_grow(&self, credit: &MemoryReservation, bytes: usize) -> datafusion::common::Result<()> {
        self.inner.try_grow(credit, bytes)?;
        self.observe();
        Ok(())
    }
}

impl ProbePool {
    fn observe(&self) {
        self.peak.fetch_max(self.inner.reserved(), Ordering::SeqCst);
    }
    pub(super) fn reset_peak(&self) {
        self.peak.store(self.inner.reserved(), Ordering::SeqCst);
    }
    pub(super) fn peak(&self) -> usize {
        self.peak.load(Ordering::SeqCst)
    }
}

pub(super) fn operator(left: SchemaRef, right: SchemaRef) -> (StreamJoinOperator, Arc<ProbePool>) {
    let pool = Arc::new(ProbePool {
        inner: GreedyMemoryPool::new(1 << 30),
        peak: AtomicUsize::new(0),
    });
    let mut operator = StreamJoinOperator::new("match", left, right, spec()).unwrap();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    (operator, pool)
}

pub(super) fn preparation_peak(
    operator: &mut StreamJoinOperator,
    plan: &SidePlan,
    admitted: &dyn AdmittedRows,
    pool: &ProbePool,
) -> usize {
    let baseline = pool.reserved();
    pool.reset_peak();
    let keys = operator.native_probe_keys(plan, admitted).unwrap().unwrap();
    let peak = pool.peak() - baseline;
    drop(keys);
    assert_eq!(
        pool.reserved(),
        baseline,
        "uncommitted preparation must refund every owner"
    );
    peak
}

pub(super) fn alias_funding(keys: [&Arc<columnar::FramedKey>; 2]) -> usize {
    keys.into_iter()
        .filter_map(|key| key.funded_owner())
        .collect::<BTreeMap<_, _>>()
        .values()
        .sum()
}
