use super::*;
use crate::runtime::streaming::gather_work::ParallelCpuWork;

struct Work {
    input: Arc<FundedInput>,
    gates: [Arc<TypedGate>; 2],
    entered: std::sync::mpsc::Sender<usize>,
    error: bool,
}

impl ParallelCpuWork for Work {
    type Output = u64;

    fn unit_count(&self) -> usize {
        2
    }

    fn run(&self, ordinal: usize, stop: &GatherStop) -> crate::Result<u64> {
        self.entered.send(ordinal).unwrap();
        self.gates[ordinal].wait();
        stop.check()?;
        if self.error && ordinal == 0 {
            return Err(crate::CalcFlowError::Internal {
                message: "parallel unit error".into(),
            });
        }
        Ok(self.input.values[ordinal])
    }
}

struct Gates([Arc<TypedGate>; 2]);

impl Drop for Gates {
    fn drop(&mut self) {
        for gate in &self.0 {
            gate.release();
        }
    }
}

#[test]
fn parallel_units_overlap_and_return_in_ordinal_order() {
    parallel_case(false, false);
}

#[test]
fn abandoned_parallel_units_keep_sources_and_credit_until_drain() {
    parallel_case(true, false);
}

#[test]
fn parallel_unit_error_settles_siblings_and_refunds_credit() {
    parallel_case(false, true);
}

fn parallel_case(abandon: bool, error: bool) {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = job(901, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![13, 29],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let gates = Gates(std::array::from_fn(|_| Arc::new(TypedGate::default())));
    let (entered, receiver) = std::sync::mpsc::channel();
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new("operator:parallel".into()))
        .scope()
        .unwrap();
    let ticket = runtime
        .block_on(scope.submit_parallel_work(
            Arc::new(Work {
                input,
                gates: gates.0.clone(),
                entered,
                error,
            }),
            credit(&pool),
            GatherStop::from_job(&job),
        ))
        .unwrap();
    let mut started = (0..2)
        .filter_map(|_| receiver.recv_timeout(Duration::from_secs(2)).ok())
        .collect::<Vec<_>>();
    started.sort_unstable();
    let overlap = started == [0, 1];
    assert!(weak.upgrade().is_some());
    assert!(pool.reserved() >= 32_768);
    if abandon {
        drop(ticket);
        assert!(runtime.block_on(async {
            tokio::time::timeout(
                Duration::from_millis(25),
                job.gather_owner().close_and_drain(),
            )
            .await
            .is_err()
        }));
        assert!(weak.upgrade().is_some());
        drop(gates);
    } else {
        gates.0[1].release();
        gates.0[0].release();
        let output = runtime.block_on(ticket.finish());
        if error {
            assert!(
                matches!(output, Err(crate::CalcFlowError::Internal { message }) if message == "parallel unit error")
            );
        } else {
            let output = output.unwrap();
            assert_eq!(output.value, [13, 29]);
            assert!(output.credit.size() >= 32_768);
            drop(output);
        }
    }
    assert!(
        runtime
            .block_on(job.gather_owner().close_and_drain())
            .is_empty()
    );
    drop((scope, job, runtime));
    service.shutdown();
    assert!(overlap, "parallel work units executed serially");
    assert!(weak.upgrade().is_none());
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
}

#[test]
fn parallel_control_budget_rejects_before_workers_start() {
    let service = TestService::new(2, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let job = job(906, &service);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(32_768));
    let funded_drop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let input = Arc::new(FundedInput {
        values: vec![1, 2],
        pool: pool.clone(),
        funded_drop: funded_drop.clone(),
    });
    let weak = Arc::downgrade(&input);
    let gates = Gates(std::array::from_fn(|_| Arc::new(TypedGate::default())));
    let (entered, receiver) = std::sync::mpsc::channel();
    let scope = job
        .gather_owner()
        .client(GatherOperatorId::new("operator:parallel".into()))
        .scope()
        .unwrap();
    let result = runtime.block_on(scope.submit_parallel_work(
        Arc::new(Work {
            input,
            gates: gates.0.clone(),
            entered,
            error: false,
        }),
        credit(&pool),
        GatherStop::from_job(&job),
    ));
    assert!(matches!(
        result,
        Err(super::super::AdmissionFailure::Budget {
            stage: "attempt",
            ..
        })
    ));
    assert!(receiver.try_recv().is_err());
    assert!(weak.upgrade().is_none());
    assert!(funded_drop.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(pool.reserved(), 0);
    assert!(
        runtime
            .block_on(job.gather_owner().close_and_drain())
            .is_empty()
    );
    drop((scope, job, runtime));
    service.shutdown();
}
