use super::*;
use crate::runtime::streaming::gather_work::TestService;
use std::time::Duration;

async fn blocked_tokio_preparation(
    service: &TestService,
    rows: usize,
    ordered: bool,
) -> (bool, bool) {
    let (mut operator, schema) = identity_fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("admission-cpu".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let blocker = tokio::task::spawn_blocking(move || {
        entered.send(()).unwrap();
        gate.recv_timeout(Duration::from_secs(10)).unwrap();
    });
    started.await.unwrap();
    let mut completed = true;
    for round in 0..2 {
        let record = repeated_key_batch(&schema, 0, rows);
        let record = if ordered {
            record
        } else {
            let indices = datafusion::arrow::array::UInt32Array::from_iter_values(
                (0..u32::try_from(rows).unwrap()).rev(),
            );
            datafusion::arrow::compute::take_record_batch(&record, &indices).unwrap()
        };
        let batch = Batch::table(vec![record.clone()], crate::BatchMetadata::default()).unwrap();
        let input = ValidatedInput {
            index: 0,
            watermark: None,
        };
        let result = tokio::time::timeout(
            Duration::from_secs(2),
            operator.prepare_admission(input, &batch, &context),
        )
        .await;
        let Ok(Ok(mut admitted)) = result else {
            match result {
                Ok(Err(error)) => eprintln!("left preparation failed: {error}"),
                Err(error) => eprintln!("left preparation timed out: {error}"),
                Ok(Ok(_)) => unreachable!(),
            }
            completed = false;
            break;
        };
        assert_eq!(admitted.accepted, rows as u64);
        let chunks = admitted.left_chunks.take().unwrap();
        assert_eq!(chunks.len(), 1);
        let (owner, data) = chunks.into_iter().next().unwrap().into_parts();
        assert_eq!(owner.record.as_ref(), &record);
        assert_eq!(data.times.as_ref(), vec![10; rows]);
        assert_eq!(data.sequences.len(), rows);
        assert_eq!(data.key_counts.iter().sum::<usize>(), rows);
        assert_eq!(service.joined_workers(), 0, "round {round}");
    }
    let used_pool = service.available_capacity().0 == 0;
    release.send(()).unwrap();
    blocker.await.unwrap();
    let failures = job.gather_owner().close_and_drain().await;
    assert!(failures.is_empty());
    drop(context);
    drop(job);
    if completed {
        assert_eq!(operator.status.state_rows, 0);
        assert_eq!(pool.reserved(), 0);
    }
    (completed, used_pool)
}

fn preparation_case(rows: usize, ordered: bool, expect_pool: bool) {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(1)
        .build()
        .unwrap();
    let (completed, used_pool) =
        runtime.block_on(blocked_tokio_preparation(&service, rows, ordered));
    drop(runtime);
    service.shutdown();
    assert!(
        completed,
        "left admission waits for Tokio blocking capacity"
    );
    assert_eq!(used_pool, expect_pool);
}

#[test]
fn small_left_admission_completes_without_a_cpu_worker() {
    preparation_case(32, true, false);
}

#[test]
fn unordered_left_admission_reuses_owned_cpu_workers() {
    preparation_case(1024, false, true);
}

#[test]
fn ordered_scalar_left_admission_avoids_cpu_worker() {
    preparation_case(1024, true, false);
}

#[test]
fn inline_preparation_bounds_rows_identity_bytes_and_order() {
    let rows = (0..4096)
        .map(|row| {
            (
                (
                    i64::from(row),
                    state::Encoding::from_slice(b"key"),
                    state::Encoding::from_slice(b"seq"),
                ),
                AdmissionRef {
                    batch_index: 0,
                    row,
                    key_index: 0,
                },
            )
        })
        .collect::<Vec<_>>();
    assert!(can_prepare_inline(&rows));
    let mut too_many = rows.clone();
    let mut next = rows[0].clone();
    next.0.0 = 4096;
    too_many.push(next);
    assert!(!can_prepare_inline(&too_many));
    let mut unordered = rows.clone();
    unordered.swap(0, 1);
    assert!(!can_prepare_inline(&unordered));
    let mut wide = rows;
    let key = state::Encoding::from_slice(&[1; 128]);
    for (order, _) in &mut wide {
        order.1 = key.clone();
    }
    assert!(!can_prepare_inline(&wide));
}

struct GateWork {
    work: LeftChunkWork,
    entered: tokio::sync::oneshot::Sender<()>,
    gate: std::sync::mpsc::Receiver<()>,
}

impl OwnedCpuWork for GateWork {
    type Output = PreparedInput;

    fn run(self, stop: &GatherStop) -> Result<PreparedInput> {
        stop.check()?;
        self.entered.send(()).unwrap();
        self.gate.recv_timeout(Duration::from_secs(10)).unwrap();
        self.work.run(stop)
    }
}

async fn abandoned_preparation(service: &TestService) {
    let (operator, schema) = identity_fixture();
    let pool = operator.runtime.pool.clone();
    let job = StreamJobContext::new(1, "asof", JsonMap::new(), None, CancellationToken::new())
        .with_gather_owner(service.owner("admission-abandon".into()));
    let context = StreamOperatorContext::new(&job, "asof", None);
    let (entered, started) = tokio::sync::oneshot::channel();
    let (release, gate) = std::sync::mpsc::channel();
    let record = repeated_key_batch(&schema, 0, 4096);
    let keys = state::encode_columns(&record, operator.spec.left().keys()).unwrap();
    let sequences = state::encode_columns(&record, operator.spec.left().sequence_by()).unwrap();
    let batches =
        vec![encode_payload(record, 0, 0, operator.payload_header_bytes[0], "asof").unwrap()];
    let weak = Arc::downgrade(&batches[0]);
    let rows = (0..4096)
        .map(|row| {
            (
                (10, keys.row(row), sequences.row(row)),
                AdmissionRef {
                    batch_index: 0,
                    row: u32::try_from(row).unwrap(),
                    key_index: 0,
                },
            )
        })
        .collect();
    let workspace = AdmissionWorkspace {
        _identity: operator.reserve_workspace(1 << 20).unwrap(),
        _payload: operator.reserve_workspace(0).unwrap(),
        _keys: None,
    };
    let work = LeftChunkWork {
        rows,
        _descriptor: operator.reserve_left_work(batches.len()).unwrap(),
        batches,
        workspace,
        side: operator.spec.left().clone(),
        name: operator.name.clone(),
    };
    let mut pending = Box::pin(operator.run_cpu_work(
        GateWork {
            work,
            entered,
            gate,
        },
        &context,
    ));
    assert!(futures::poll!(pending.as_mut()).is_pending());
    started.await.unwrap();
    drop(pending);
    let retained = weak.upgrade().is_some() && pool.reserved() >= 1 << 20;
    release.send(()).unwrap();
    assert!(job.gather_owner().close_and_drain().await.is_empty());
    drop(context);
    drop(job);
    assert!(
        retained,
        "abandoned preparation released its running Arrow owner or credit"
    );
    assert!(weak.upgrade().is_none());
    assert_eq!(pool.reserved(), 0);
    assert_eq!(operator.status.state_rows, 0);
}

#[test]
fn abandoned_left_preparation_retains_owner_until_native_drain() {
    let service = TestService::new(1, 1).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(abandoned_preparation(&service));
    drop(runtime);
    service.shutdown();
}
