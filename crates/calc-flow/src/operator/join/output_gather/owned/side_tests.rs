use super::OwnedSide;
use crate::operator::join::{columnar::RowPayload, join_work, reset_join_work};
use crate::{CancellationToken, JsonMap, StreamJobContext, StreamOperatorContext};
use datafusion::arrow::{
    array::Int64Array,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Capture,
    Promotion,
}

#[derive(Clone, Debug)]
struct Charge {
    phase: Phase,
    visits: usize,
    bytes: usize,
}

#[derive(Clone, Default)]
struct Work {
    charges: Vec<Charge>,
    promoted: usize,
}

struct Observed {
    work: Work,
    cancel_after_prefix: Option<(CancellationToken, usize)>,
}

thread_local! {
    static OBSERVED: std::cell::RefCell<Option<Observed>> = const {
        std::cell::RefCell::new(None)
    };
}

struct Observation;

impl Observation {
    fn new(cancel_after_prefix: Option<(CancellationToken, usize)>) -> Self {
        OBSERVED.with(|observed| {
            let mut observed = observed.borrow_mut();
            assert!(observed.is_none());
            *observed = Some(Observed {
                work: Work::default(),
                cancel_after_prefix,
            });
        });
        Self
    }

    fn work() -> Work {
        OBSERVED.with(|observed| observed.borrow().as_ref().unwrap().work.clone())
    }
}

impl Drop for Observation {
    fn drop(&mut self) {
        OBSERVED.with(|observed| *observed.borrow_mut() = None);
    }
}

pub(super) fn note_charge(phase: Phase, visits: usize, bytes: usize) {
    OBSERVED.with(|observed| {
        if let Some(observed) = observed.borrow_mut().as_mut() {
            observed.work.charges.push(Charge {
                phase,
                visits,
                bytes,
            });
        }
    });
}

pub(super) fn note_promoted(rows: usize) {
    OBSERVED.with(|observed| {
        if let Some(observed) = observed.borrow_mut().as_mut() {
            observed.work.promoted += rows;
            if let Some((cancellation, threshold)) = &observed.cancel_after_prefix
                && observed.work.promoted >= *threshold
            {
                cancellation.cancel();
            }
        }
    });
}

fn assert_paid_work(work: &Work, rows: usize, prefix: usize) {
    assert!(work.charges.iter().all(|charge| {
        (1..=64).contains(&charge.visits)
            && charge.bytes == 32 * charge.visits
            && charge.bytes <= 4096
    }));
    let visits = |phase| {
        work.charges
            .iter()
            .filter(|charge| charge.phase == phase)
            .map(|charge| charge.visits)
            .sum::<usize>()
    };
    let capture_visits = visits(Phase::Capture);
    assert!(capture_visits >= rows && capture_visits <= rows + 64);
    assert_eq!(visits(Phase::Promotion), prefix);
    assert_eq!(work.promoted, prefix);
}

#[tokio::test]
async fn test_single_capture_batches_quantum_work() {
    let parent = Arc::new(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from_iter_values(0..17))],
        )
        .unwrap(),
    );
    let offsets = (0..129).map(|row| (row * 7) % 17).collect::<Vec<usize>>();
    let payloads = offsets
        .iter()
        .map(|&row| RowPayload::Rowed {
            parent: Arc::clone(&parent),
            row,
        })
        .collect::<Vec<_>>();
    let job = StreamJobContext::new(
        1,
        "single-capture-work",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let _observation = Observation::new(None);
    reset_join_work();

    let captured = OwnedSide::capture(payloads.iter().map(Into::into), &context)
        .await
        .unwrap();
    let work = join_work();

    let OwnedSide::Single { source, rows } = &captured else {
        panic!("one shared parent must retain the single-source representation");
    };
    assert_eq!(source.columns().as_ptr(), parent.columns().as_ptr());
    assert_eq!(
        rows.values().as_ref(),
        offsets
            .iter()
            .map(|&row| u64::try_from(row).unwrap())
            .collect::<Vec<_>>()
    );
    drop(captured);
    drop(payloads);
    assert_eq!(Arc::strong_count(&parent), 1);
    assert!(
        work.quantum_steps <= offsets.len().div_ceil(64) + 1,
        "capture must charge bounded chunks; observed {} Quantum calls for {} rows",
        work.quantum_steps,
        offsets.len()
    );
    let paid = Observation::work();
    assert_paid_work(&paid, offsets.len(), 0);
    assert_eq!(paid.charges.len(), work.quantum_steps);
    assert_eq!(
        paid.charges
            .iter()
            .map(|charge| charge.visits)
            .sum::<usize>(),
        offsets.len()
    );
}

fn source(start: i64) -> Arc<RecordBatch> {
    Arc::new(
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(Int64Array::from_iter_values(start..start + 17))],
        )
        .unwrap(),
    )
}

fn payloads(parents: &[Arc<RecordBatch>; 2], prefix: usize, tail: usize) -> Vec<RowPayload> {
    (0..prefix + 1 + tail)
        .map(|index| RowPayload::Rowed {
            parent: Arc::clone(&parents[usize::from(index == prefix)]),
            row: (index * 7) % 17,
        })
        .collect()
}

#[tokio::test]
async fn test_capture_promotes_boundary_prefixes_without_losing_order_or_bounded_work() {
    for (prefix, tail) in [(63, 65), (64, 64), (65, 63), (193, 7)] {
        let parents = [source(0), source(100)];
        let payloads = payloads(&parents, prefix, tail);
        let job = StreamJobContext::new(
            1,
            "promoted-capture-work",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let _observation = Observation::new(None);
        reset_join_work();

        let captured = OwnedSide::capture(payloads.iter().map(Into::into), &context)
            .await
            .unwrap();
        let work = join_work();

        let OwnedSide::Multiple { sources, rows } = &captured else {
            panic!("source changes must promote the captured side");
        };
        assert_eq!(sources.len(), 2);
        for (source, parent) in sources.iter().zip(&parents) {
            assert_eq!(source.columns().as_ptr(), parent.columns().as_ptr());
        }
        assert_eq!(
            *rows,
            (0..payloads.len())
                .map(|index| (usize::from(index == prefix), (index * 7) % 17))
                .collect::<Vec<_>>()
        );
        assert_eq!(rows.capacity(), payloads.len());
        let paid = Observation::work();
        assert_paid_work(&paid, payloads.len(), prefix);
        assert_eq!(paid.charges.len(), work.quantum_steps);
        let maximum_calls = payloads.len().div_ceil(64) + prefix.div_ceil(64) + 2;
        drop(captured);
        drop(payloads);
        assert!(parents.iter().all(|parent| Arc::strong_count(parent) == 1));
        assert!(
            work.quantum_steps <= maximum_calls,
            "prefix {prefix} and tail {tail} require bounded capture/promotion; observed {} Quantum calls, maximum {maximum_calls}",
            work.quantum_steps,
        );
    }
}

async fn assert_cancelled_capture(prefix: usize, tail: usize, cancel_after: usize) {
    let parents = [source(0), source(100)];
    let payloads = payloads(&parents, prefix, tail);
    let job = StreamJobContext::new(
        1,
        "cancelled-capture-promotion",
        JsonMap::new(),
        None,
        CancellationToken::new(),
    );
    let context = StreamOperatorContext::new(&job, "match", None);
    let _observation = Observation::new(Some((job.cancellation().clone(), cancel_after)));
    reset_join_work();

    let result = OwnedSide::capture(payloads.iter().map(Into::into), &context).await;
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    let paid = Observation::work();
    assert_eq!(paid.promoted, cancel_after);
    let first_promotion = paid
        .charges
        .iter()
        .position(|charge| charge.phase == Phase::Promotion)
        .unwrap();
    assert!(
        paid.charges[first_promotion..]
            .iter()
            .all(|charge| charge.phase == Phase::Promotion)
    );
    assert_eq!(
        join_work().quantum_steps,
        paid.charges.len() + usize::from(cancel_after < prefix),
        "the held transition must never consume another capture charge after cancellation",
    );
    drop(payloads);
    assert!(parents.iter().all(|parent| Arc::strong_count(parent) == 1));
}

#[tokio::test]
async fn test_capture_cancellation_during_prefix_promotion_retires_local_sources() {
    assert_cancelled_capture(193, 7, 64).await;
}

#[tokio::test]
async fn test_capture_cancellation_before_held_row_resume_retires_local_sources() {
    assert_cancelled_capture(63, 0, 63).await;
}
