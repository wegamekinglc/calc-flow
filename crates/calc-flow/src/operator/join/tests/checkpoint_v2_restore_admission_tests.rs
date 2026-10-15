use super::*;
use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryLimit, MemoryPool, MemoryReservation,
};
use std::{cell::Cell, sync::Mutex};

#[derive(Clone, Debug, Eq, PartialEq)]
struct Admission {
    consumer: String,
    bytes: usize,
    rejected: bool,
}

#[derive(Debug, Default)]
struct Fault {
    armed: bool,
    reject: Option<usize>,
    admissions: Vec<Admission>,
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

    fn finish(&self) -> Vec<Admission> {
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
            let rejected = fault.reject == Some(fault.admissions.len() + 1);
            fault.admissions.push(Admission {
                consumer: credit.consumer().name().to_owned(),
                bytes,
                rejected,
            });
            if rejected {
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

fn fixture() -> (
    StreamJoinOperator,
    OperatorStateSnapshot,
    Arc<RestorePool>,
    OperatorStateSnapshot,
) {
    let pool = Arc::new(RestorePool {
        inner: GreedyMemoryPool::new(16 << 20),
        fault: Mutex::new(Fault::default()),
    });
    let mut operator = operator();
    operator
        .runtime
        .initialize_benchmark_pool(pool.clone())
        .unwrap();
    let original = snapshot(&operator);
    operator.restore(&original).unwrap();
    let target = history_controls::interleaved_snapshot(&operator);
    (operator, target, pool, original)
}

struct Previous {
    left: Arc<StateRows>,
    right: Arc<StateRows>,
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

#[tokio::test]
async fn test_v2_restore_distinguishes_mandatory_and_dictionary_refusals() {
    let (mut operator, target, pool, original) = fixture();
    pool.arm(None);
    operator.restore(&target).unwrap();
    let admissions = pool.finish();
    assert!(admissions.len() > 20);
    drop((operator, target, original));
    assert_eq!(pool.reserved(), 0);
    let mut optional = 0;
    for reject in 1..=admissions.len() {
        optional += usize::from(Box::pin(check_rejected_admission(reject, &admissions)).await);
    }
    assert!(
        optional > 0,
        "must exercise a real optional dictionary refusal"
    );
    println!(
        "rejected {} mandatory restore admissions and installed SQL state after {optional} dictionary refusals",
        admissions.len() - optional
    );
}

fn assert_rejected_admission<'a>(
    observed: &'a [Admission],
    expected: &[Admission],
    reject: usize,
) -> &'a Admission {
    assert!(observed.len() >= reject);
    assert_eq!(&observed[..reject - 1], &expected[..reject - 1]);
    let actual = &observed[reject - 1];
    assert_eq!(actual.consumer, expected[reject - 1].consumer);
    assert_eq!(actual.bytes, expected[reject - 1].bytes);
    assert!(actual.bytes > 0);
    assert!(actual.rejected);
    assert_eq!(observed.iter().filter(|entry| entry.rejected).count(), 1);
    println!("admission {reject}: {actual:?}");
    actual
}

async fn check_rejected_admission(reject: usize, admissions: &[Admission]) -> bool {
    let (mut operator, target, pool, old_history) = fixture();
    let original = target.clone();
    let previous = Previous::capture(&operator, &pool);
    let writer = operator.v2_writer.observation_for_test();
    pool.arm(Some(reject));
    let result = operator.restore(&target);
    let observed = pool.finish();
    let actual = assert_rejected_admission(&observed, admissions, reject);
    let optional = actual.consumer == "sql-incremental:stream-join-dictionary";
    if optional {
        result.expect("a real optional dictionary refusal must install validated SQL state");
        assert!(!Arc::ptr_eq(&operator.state.left.0, &previous.left));
        assert!(!Arc::ptr_eq(&operator.state.right.0, &previous.right));
        assert!(!Arc::ptr_eq(
            operator.v2_containers.as_ref().unwrap(),
            &previous.containers
        ));
        assert_ne!(operator.v2_writer.observation_for_test(), writer);
        assert_sql_candidate(&operator);
        assert_candidate_history(&mut operator, &target);
    } else {
        assert!(result.is_err(), "mandatory admission {reject}: {actual:?}");
        assert_eq!(observed.len(), reject);
        previous.assert_retained(&operator, &pool);
        assert_eq!(operator.v2_writer.observation_for_test(), writer);
        assert_candidate_history(&mut operator, &old_history);
    }
    drop(previous);
    let continuation = if optional {
        Some(Box::pin(continue_sql_candidate(&mut operator)).await)
    } else {
        None
    };
    assert_eq!(target.inline_metadata, original.inline_metadata);
    assert_eq!(target.segments, original.segments);
    drop((operator, target, original, old_history));
    if let Some(job) = continuation {
        assert!(job.gather_owner().close_and_drain().await.is_empty());
        drop(job);
    }
    assert_eq!(pool.reserved(), 0);
    optional
}

fn assert_sql_candidate(operator: &StreamJoinOperator) {
    assert_eq!(
        (operator.state.left.len(), operator.state.right.len()),
        (2, 1)
    );
    assert_row(&operator.state.left.at(0), 1, 97, 135, "新", "green", None);
    assert_row(
        &operator.state.left.at(1),
        4,
        98,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_row(
        &operator.state.right.at(0),
        4,
        100,
        141,
        "ok",
        "red",
        Some(&[9]),
    );
    assert_eq!(operator.state.last_checkpoint_epoch, Epoch::new(9));
    assert_eq!(operator.state.next_left_row_id, 10);
    assert_eq!(operator.state.next_right_row_id, 5);
    assert_eq!(operator.state.next_output_sequence, 6);
    assert_eq!(operator.status().left.retained_bytes, 276);
    assert_eq!(operator.status().right.retained_bytes, 141);
    assert_eq!(operator.status().emitted_match_rows, 5);
    assert!(operator.key_dictionary_needs_rebuild);
    assert_eq!(operator.key_work.dictionary_grant_refusals, 1);
    assert_eq!(operator.key_dictionary.inventory(), (0, 0, 0, false));
    assert!(operator.state.left.1.is_none());
    assert!(operator.state.right.1.is_none());
    assert!(operator.state.deltas.pending.is_empty());
}

fn assert_candidate_history(operator: &mut StreamJoinOperator, target: &OperatorStateSnapshot) {
    let captured = operator.checkpoint(Epoch::new(10).unwrap()).unwrap();
    assert_eq!(
        captured.inline_metadata["metrics"],
        target.inline_metadata["metrics"]
    );
    assert_eq!(
        captured.inline_metadata["v2_inventory"],
        target.inline_metadata["v2_inventory"]
    );
    assert_eq!(captured.segments.len(), target.segments.len());
    for (name, segment) in &target.segments {
        let retained = &captured.segments[name];
        assert_eq!(retained.bytes(), segment.bytes());
        assert!(Arc::ptr_eq(&retained.bytes_arc(), &segment.bytes_arc()));
    }
}

async fn continue_sql_candidate(operator: &mut StreamJoinOperator) -> StreamJobContext {
    let job = job();
    let context = StreamOperatorContext::new(&job, "v2-match", None);
    let mut output = EdgeCollector::new(operator.output_ports().to_vec());
    let batch = Batch::table(
        vec![record(
            &[101],
            &[Some("ok")],
            &["red"],
            vec![Some(vec![Some(9)])],
        )],
        BatchMetadata::default(),
    )
    .unwrap();
    let queries = operator.key_work.sql_probe_table_builds;
    reset_join_work();
    operator
        .process_data("right", batch, &context, &mut output)
        .await
        .unwrap();
    assert_eq!(operator.key_work.sql_probe_table_builds, queries + 1);
    assert_eq!(operator.key_work.needs_rebuild_routes, 1);
    assert_eq!(join_work().native_range_visits, 0);
    assert!(operator.key_dictionary_needs_rebuild);
    assert_eq!(operator.status().emitted_match_rows, 7);
    assert_eq!(operator.status().left.retained_bytes, 276);
    assert_eq!(operator.status().right.retained_bytes, 282);
    assert_eq!(operator.state.next_right_row_id, 6);
    assert_eq!(operator.state.next_output_sequence, 7);
    let messages = output.drain("output");
    assert_eq!(messages.len(), 1);
    let batch = messages[0].as_data().unwrap();
    assert_eq!(batch.metadata().sequence(), 6);
    assert_eq!(batch.num_rows(), 2);
    let result = &batch.table_payload().unwrap().batches()[0];
    assert_eq!(result.num_columns(), 10);
    assert_eq!(
        result
            .column(1)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[97, 98]
    );
    assert_eq!(
        result
            .column(6)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .unwrap()
            .values(),
        &[101, 101]
    );
    drop((messages, output, context));
    job
}

#[test]
fn test_v2_restore_cancels_at_every_check_without_replacing_old_state() {
    let (mut operator, target, pool, original) = fixture();
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
    assert_eq!(operator.state.left.at(0).row_id, 1);
    println!("cancelled and refunded all {checks} restore checks");
    drop((previous, operator, target, original));
    assert_eq!(pool.reserved(), 0);
}
