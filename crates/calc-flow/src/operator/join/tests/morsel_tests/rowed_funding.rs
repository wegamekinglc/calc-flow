use super::*;
use crate::operator::join::{AdmittedRow, StateRows, StoredRow};
use crate::{CancellationToken, EventTime, JoinTimeBounds, JsonMap, StreamJobContext};
use datafusion::arrow::array::Int64Array;
use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};
use std::time::Duration;

fn source() -> Arc<RecordBatch> {
    let record = RecordBatch::try_from_iter([(
        "value",
        Arc::new(Int64Array::from(vec![7; 16_384])) as ArrayRef,
    )])
    .unwrap();
    Arc::new(record.slice(41, 2))
}

fn row(payload: RowPayload, ordinal: usize) -> AdmittedRow {
    AdmittedRow {
        record: payload,
        event_time: EventTime::from_micros(0),
        row_id: u64::try_from(ordinal).unwrap(),
        retain: false,
    }
}

fn inputs(rows: Vec<AdmittedRow>, opposite: RowPayload) -> Arc<ProbeInputs> {
    let mut state = StateRows::default();
    state.append(vec![StoredRow {
        record: opposite,
        event_time: EventTime::from_micros(0),
        row_id: 0,
        charge: 1,
        encoded_key: Arc::new(vec![].into()),
    }]);
    Arc::new(ProbeInputs {
        partition: super::super::super::partition::Partition::new(rows.len(), 1),
        admitted: rows.into(),
        keys: None,
        slots: Vec::new(),
        opposite: Some(Arc::new(state)),
        index: None,
        containers: None,
        bounds: JoinTimeBounds::new(Duration::ZERO, Duration::ZERO).unwrap(),
        incoming_is_left: true,
        control: None,
        released: None,
        hook: None,
        unit_hook: None,
    })
}

fn job() -> StreamJobContext {
    StreamJobContext::new(
        793,
        "rowed-funding",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    )
}

#[tokio::test]
async fn batch_parallel_output_rowed_funding_prices_backing_and_refunds_refusals() {
    let source = source();
    let payload = RowPayload::Rowed {
        parent: Arc::clone(&source),
        row: 0,
    };
    let inputs = inputs(vec![row(payload.clone(), 0)], payload);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let bytes = input_bytes(&inputs, &context).await.unwrap().unwrap();
    let backing = source.get_array_memory_size();
    assert!(backing > source.num_rows() * size_of::<i64>() * 100);
    assert!(
        bytes >= backing + 4096,
        "lease includes full slice backing and controls"
    );
    let base = 1_024;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(base + backing - 1));
    let scratch = MemoryConsumer::new("rowed-finite-refusal").register(&pool);
    assert!(
        fund_inputs(&inputs, &scratch, base, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(pool.reserved(), 0);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(base + bytes));
    let scratch = MemoryConsumer::new("rowed-full-backing").register(&pool);
    assert!(
        fund_inputs(&inputs, &scratch, base + bytes, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(
        pool.reserved(),
        0,
        "explicit finite pool declines the combined scratch/input reservation"
    );
    let funded = fund_inputs(&inputs, &scratch, base, &context)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(scratch.size(), base);
    assert_eq!(pool.reserved(), base + bytes);
    drop((funded, scratch));
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn cpu_aware_probe_large_scratch_reservation_uses_user_pool_and_refunds() {
    let source = source();
    let payload = RowPayload::Rowed {
        parent: source,
        row: 0,
    };
    let inputs = inputs(vec![row(payload.clone(), 0)], payload);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let base = 600 << 20;
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    let scratch = MemoryConsumer::new("large-user-scratch").register(&pool);
    let funded = fund_inputs(&inputs, &scratch, base, &context)
        .await
        .unwrap()
        .expect("a sufficient user pool permits scratch above the old fixed cap");
    assert!(pool.reserved() > base);
    drop((funded, scratch));
    assert_eq!(pool.reserved(), 0);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(base));
    let scratch = MemoryConsumer::new("limited-user-scratch").register(&pool);
    assert!(
        fund_inputs(&inputs, &scratch, base, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn cpu_aware_probe_large_parent_inventory_obeys_only_user_pool() {
    let source = source();
    let payload = RowPayload::Rowed {
        parent: source,
        row: 0,
    };
    let rows = (0..4_097)
        .map(|ordinal| row(payload.clone(), ordinal))
        .collect();
    let inputs = inputs(rows, payload);
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
    let scratch = MemoryConsumer::new("large-user-inputs").register(&pool);
    let funded = fund_inputs(&inputs, &scratch, 0, &context)
        .await
        .unwrap()
        .expect("parent inventory has no independent fixed byte cap");
    assert!(pool.reserved() > 512 << 20);
    drop((funded, scratch));
    assert_eq!(pool.reserved(), 0);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let scratch = MemoryConsumer::new("limited-user-inputs").register(&pool);
    assert!(
        fund_inputs(&inputs, &scratch, 0, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(pool.reserved(), 0);
}

#[tokio::test]
async fn batch_parallel_output_rowed_proof_rejects_legacy_and_unfunded_opposite() {
    let source = source();
    let rowed = RowPayload::Rowed {
        parent: Arc::clone(&source),
        row: 0,
    };
    let legacy = RowPayload::Legacy(source.slice(0, 1));
    let job = job();
    let context = StreamOperatorContext::new(&job, "match", None);
    let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 20));
    let scratch = MemoryConsumer::new("rowed-eligibility").register(&pool);
    let incoming_legacy = inputs(vec![row(legacy.clone(), 0)], rowed.clone());
    assert!(
        fund_inputs(&incoming_legacy, &scratch, 0, &context)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(pool.reserved(), 0);
    let pairs = vec![MatchedPair {
        pos: 0,
        opposite_index: 0,
    }]
    .into();
    for opposite in [rowed.clone(), legacy] {
        let inputs = inputs(vec![row(rowed.clone(), 0)], opposite);
        let funded = fund_inputs(&inputs, &scratch, 0, &context)
            .await
            .unwrap()
            .unwrap();
        assert!(!owned_payloads(&funded, &pairs, &GatherStop::from_job(&job)).unwrap());
        drop(funded);
        assert_eq!(pool.reserved(), 0);
    }
}

#[tokio::test]
async fn batch_parallel_output_rowed_inventory_yields_and_observes_cancellation() {
    tokio::spawn(async {
        let source = source();
        let payload = RowPayload::Rowed {
            parent: source,
            row: 0,
        };
        let rows = (0..5_000)
            .map(|ordinal| row(payload.clone(), ordinal))
            .collect();
        let inputs = inputs(rows, payload);
        let job = job();
        let context = StreamOperatorContext::new(&job, "match", None);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1 << 30));
        let scratch = MemoryConsumer::new("rowed-cancel").register(&pool);
        while tokio::task::coop::has_budget_remaining() {
            tokio::task::consume_budget().await;
        }
        let mut preflight = Box::pin(fund_inputs(&inputs, &scratch, 0, &context));
        assert!(futures::poll!(preflight.as_mut()).is_pending());
        job.cancellation().cancel();
        assert!(matches!(
            preflight.await,
            Err(crate::CalcFlowError::Cancelled { .. })
        ));
        assert_eq!(pool.reserved(), 0);
    })
    .await
    .unwrap();
}
