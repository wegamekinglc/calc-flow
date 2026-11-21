pub(crate) mod multi;

use std::sync::{Arc, Barrier, Weak, mpsc};

use datafusion::{
    arrow::{
        array::{Array, ArrayRef, Int64Array},
        datatypes::{DataType, Field, Schema},
        record_batch::RecordBatch,
    },
    execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation},
};

use crate::runtime::streaming::gather_work::{
    GatherOperatorId, GatherPlan, GatherStop, OwnedCpuWork, TestService, WorkOutput,
};
use crate::{Result, StreamJobContext};

pub(crate) struct Probe {
    pub(crate) started: mpsc::Receiver<()>,
    pub(crate) release: Arc<Barrier>,
    pub(crate) source: Weak<dyn Array>,
    pub(crate) pool: Arc<dyn MemoryPool>,
}

pub(crate) fn materialization(job: StreamJobContext) -> (impl Future<Output = Result<()>>, Probe) {
    let record = record(&[1, 2, 3]);
    let source = Arc::downgrade(record.column(0));
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let reservation = MemoryConsumer::new("gather-lifecycle-bridge").register(&pool);
    reservation.try_grow(32_768).unwrap();
    let (started_tx, started) = mpsc::channel();
    let release = Arc::new(Barrier::new(2));
    let plan = BridgePlan {
        record,
        started: parking_lot::Mutex::new(Some(started_tx)),
        release: release.clone(),
        worker: None,
    };
    let future = async move {
        let context = crate::StreamOperatorContext::new(&job, "bridge", None);
        let client = context.gather_client(GatherOperatorId::new("operator:bridge".into()));
        let scope = client.scope()?;
        let ticket = scope
            .submit_work(plan, reservation, GatherStop::from_job(&job))
            .await?;
        let _ = ticket.finish().await?;
        Ok(())
    };
    (
        future,
        Probe {
            started,
            release,
            source,
            pool,
        },
    )
}

struct BridgePlan {
    record: Arc<RecordBatch>,
    started: parking_lot::Mutex<Option<mpsc::Sender<()>>>,
    release: Arc<Barrier>,
    worker: Option<Arc<WorkerProbe>>,
}

impl OwnedCpuWork for BridgePlan {
    type Output = Vec<ArrayRef>;

    fn control_bytes(&self) -> Result<usize> {
        Ok(256)
    }

    fn run(self, stop: &GatherStop) -> Result<Self::Output> {
        (0..self.column_count())
            .map(|ordinal| {
                stop.check()?;
                self.gather(ordinal, stop)
            })
            .collect()
    }
}

impl GatherPlan for BridgePlan {
    fn column_count(&self) -> usize {
        2
    }

    fn gather(&self, _ordinal: usize, _stop: &GatherStop) -> Result<ArrayRef> {
        if let Some(started) = self.started.lock().take() {
            let _ = started.send(());
            self.release.wait();
        }
        if let Some(worker) = &self.worker {
            worker.wait();
        }
        Ok(self.record.column(0).clone())
    }
}

pub(crate) fn ungated_plan(value: i64) -> Arc<dyn GatherPlan> {
    Arc::new(BridgePlan {
        record: record(&[value]),
        started: parking_lot::Mutex::new(None),
        release: Arc::new(Barrier::new(2)),
        worker: None,
    })
}

fn record(values: &[i64]) -> Arc<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Int64,
        false,
    )]));
    Arc::new(
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(values.to_vec()))]).unwrap(),
    )
}

fn output_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("left__value", DataType::Int64, false),
        Field::new("right__value", DataType::Int64, true),
    ]))
}

#[test]
fn gather_idle_pool_yields_capacity_with_live_context_and_unobserved_ticket() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let progressed = runtime.block_on(asof_pressure_case(&service));
    drop(runtime);
    service.shutdown();
    assert!(progressed);
}

async fn asof_pressure_case(service: &TestService) -> bool {
    let job_a = pressure_job(1, service);
    let job_b = pressure_job(2, service);
    let pool_a: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let pool_b: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let client_a = job_a
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()));
    let client_b = job_b
        .gather_owner()
        .client(GatherOperatorId::new("operator:asof".into()));
    let scope_a = client_a.scope().unwrap();
    let scope_b = client_b.scope().unwrap();
    let mut ticket_a = Some(
        scope_a
            .submit(
                ungated_plan(11),
                pressure_credit(&pool_a),
                GatherStop::from_job(&job_a),
            )
            .await
            .unwrap(),
    );
    let old_generation = ticket_a.as_ref().unwrap().generation();
    ticket_a.as_ref().unwrap().wait_settled().await.unwrap();
    assert_eq!(service.joined_workers(), 0);
    let mut pending_b = Box::pin(scope_b.submit(
        ungated_plan(22),
        pressure_credit(&pool_b),
        GatherStop::from_job(&job_b),
    ));
    let observed_b = tokio::time::timeout(std::time::Duration::from_secs(2), async {
        let ticket = pending_b.as_mut().await?;
        ticket.wait_settled().await?;
        Ok::<_, crate::CalcFlowError>(ticket)
    })
    .await;
    let progressed = observed_b.is_ok();
    if let Ok(ticket_b) = observed_b {
        let output_b = ticket_b.unwrap().finish().await.unwrap();
        assert_pressure_output(&output_b, 22);
        assert!(service.joined_workers() >= 1);
        assert_eq!(job_a.gather_owner().funding().1, 0);
        drop(output_b);
        let output_a = ticket_a.take().unwrap().finish().await.unwrap();
        assert_pressure_output(&output_a, 11);
        drop(output_a);
        let next_a = scope_a
            .submit(
                ungated_plan(33),
                pressure_credit(&pool_a),
                GatherStop::from_job(&job_a),
            )
            .await
            .unwrap();
        assert!(next_a.generation() > old_generation);
        let output_a = next_a.finish().await.unwrap();
        assert_pressure_output(&output_a, 33);
        drop(output_a);
    }
    drop(pending_b);
    drop(ticket_a);
    job_a.gather_owner().close_and_drain().await;
    job_b.gather_owner().close_and_drain().await;
    drop(scope_a);
    drop(scope_b);
    drop(job_a);
    drop(job_b);
    assert_eq!(pool_a.reserved(), 0);
    assert_eq!(pool_b.reserved(), 0);
    progressed
}

fn pressure_job(id: u64, service: &TestService) -> StreamJobContext {
    StreamJobContext::new(
        id,
        "asof",
        crate::JsonMap::new(),
        None,
        crate::CancellationToken::new(),
    )
    .with_gather_owner(service.owner(id.to_string().into()))
}

fn pressure_credit(pool: &Arc<dyn MemoryPool>) -> MemoryReservation {
    let credit = MemoryConsumer::new("asof-pressure").register(pool);
    credit.try_grow(32_768).unwrap();
    credit
}

fn assert_pressure_output(output: &WorkOutput<Vec<ArrayRef>>, expected: i64) {
    assert_eq!(output.value.len(), 2);
    assert!(output.credit.size() >= 32_768);
    for column in &output.value {
        let values = column.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(values.iter().collect::<Vec<_>>(), vec![Some(expected)]);
    }
}

pub(crate) struct RuntimeProbe {
    pub(crate) started: mpsc::Receiver<()>,
    pub(crate) worker: Arc<WorkerProbe>,
    pub(crate) source: Weak<dyn Array>,
    pub(crate) output_schema: Weak<Schema>,
    pub(crate) pool: Arc<dyn MemoryPool>,
}

pub(crate) struct WorkerProbe {
    started: parking_lot::Mutex<Option<mpsc::Sender<()>>>,
    released: parking_lot::Mutex<bool>,
    changed: parking_lot::Condvar,
}

impl WorkerProbe {
    fn wait(&self) {
        if let Some(started) = self.started.lock().take() {
            let _ = started.send(());
        }
        let mut released = self.released.lock();
        while !*released {
            self.changed.wait(&mut released);
        }
    }

    pub(crate) fn release(&self) {
        *self.released.lock() = true;
        self.changed.notify_all();
    }
}

pub(crate) fn runtime_materialization(
    job: StreamJobContext,
) -> (impl Future<Output = Result<()>>, RuntimeProbe) {
    let record = record(&[1, 2, 3]);
    let source = Arc::downgrade(record.column(0));
    let schema = output_schema();
    let output_schema = Arc::downgrade(&schema);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1_048_576));
    let reservation = MemoryConsumer::new("columns-only-lifecycle").register(&pool);
    reservation.try_grow(32_768).unwrap();
    let (started_tx, started) = mpsc::channel();
    let worker = Arc::new(WorkerProbe {
        started: parking_lot::Mutex::new(Some(started_tx)),
        released: parking_lot::Mutex::new(false),
        changed: parking_lot::Condvar::new(),
    });
    let plan = BridgePlan {
        record,
        started: parking_lot::Mutex::new(None),
        release: Arc::new(Barrier::new(2)),
        worker: Some(worker.clone()),
    };
    let future = async move {
        let observer_schema = schema;
        let context = crate::StreamOperatorContext::new(&job, "bridge", None);
        let client = context.gather_client(GatherOperatorId::new("operator:bridge".into()));
        let scope = client.scope()?;
        let ticket = scope
            .submit_work(plan, reservation, GatherStop::from_job(&job))
            .await?;
        let _ = ticket.finish().await?;
        drop(observer_schema);
        Ok(())
    };
    (
        future,
        RuntimeProbe {
            started,
            worker,
            source,
            output_schema,
            pool,
        },
    )
}
