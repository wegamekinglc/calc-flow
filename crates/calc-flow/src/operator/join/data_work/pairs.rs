use super::super::{matches::Matches, native_lookup::scratch_error};
use crate::Result;
use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) struct PairOutput {
    pub(super) pairs: crate::operator::join::matches::Selection,
    pub(super) credit: MemoryReservation,
}

pub(super) enum FillFunding {
    Single(Mutex<Option<MemoryReservation>>),
    Fragments(Arc<MemoryReservation>),
}

impl FillFunding {
    pub(super) fn unit(&self) -> PairCredit {
        match self {
            Self::Single(credit) => PairCredit::Owned(
                credit
                    .lock()
                    .take()
                    .expect("single fill unit owns pair funding"),
            ),
            Self::Fragments(credit) => PairCredit::Shared {
                _credit: Arc::clone(credit),
            },
        }
    }
}

pub(super) enum PairCredit {
    Owned(MemoryReservation),
    Shared { _credit: Arc<MemoryReservation> },
}

pub(super) struct PairFragment {
    pub(super) pairs: crate::operator::join::matches::Selection,
    pub(super) credit: PairCredit,
}

impl PairFragment {
    fn into_output(self) -> Result<PairOutput> {
        match self {
            Self {
                pairs,
                credit: PairCredit::Owned(credit),
            } => Ok(PairOutput { pairs, credit }),
            other => {
                drop(other);
                Err(scratch_error("join"))
            }
        }
    }
}

pub(super) fn ordered_output(
    mut fragments: Vec<PairFragment>,
    credit: Option<MemoryReservation>,
    count: usize,
) -> Result<PairOutput> {
    let Some(credit) = credit else {
        debug_assert_eq!(fragments.len(), 1);
        return fragments.pop().expect("single fill result").into_output();
    };
    let incoming = fragments
        .iter()
        .try_fold(0usize, |total, fragment| {
            total.checked_add(fragment.pairs.incoming_capacity())
        })
        .ok_or_else(|| scratch_error("join"))?;
    let mut pairs = crate::operator::join::matches::Selection::with_capacity(count, incoming);
    for fragment in fragments {
        let PairFragment {
            pairs: fragment,
            credit: fragment_credit,
        } = fragment;
        pairs.append_view(fragment.view());
        drop(fragment);
        drop(fragment_credit);
    }
    debug_assert_eq!(pairs.len(), count);
    Ok(PairOutput { pairs, credit })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::join::{MatchedPair, matches::Selection};
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryConsumer, MemoryPool};

    #[test]
    fn test_fragment_merge_bounds_incoming_runs_by_input_rows() {
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(16_000_000));
        let count = 100_000;
        let master = MemoryConsumer::new("selection-merge").register(&pool);
        let fragments = (0..2)
            .map(|pos| {
                let mut pairs = Selection::with_capacity(count / 2, 1);
                pairs.extend((0..count / 2).map(|opposite_index| MatchedPair {
                    pos,
                    opposite_index,
                }));
                let credit = master.new_empty();
                credit.try_grow(count * 24).unwrap();
                PairFragment {
                    pairs,
                    credit: PairCredit::Shared {
                        _credit: Arc::new(credit),
                    },
                }
            })
            .collect();
        let credit = master.new_empty();
        credit.try_grow(count * 24).unwrap();
        crate::operator::join::matches::reset_pair_pushes();
        let measured = allocation_counter::measure(|| {
            let output = ordered_output(fragments, Some(credit), count).unwrap();
            assert_eq!(output.pairs.len(), count);
            assert_eq!(output.pairs.incoming_capacity(), 2);
            let positions = output.pairs.iter().map(|pair| pair.pos);
            assert!(positions.take(count / 2).all(|pos| pos == 0));
            assert!(
                output
                    .pairs
                    .iter()
                    .skip(count / 2)
                    .all(|pair| pair.pos == 1)
            );
        });
        assert!(
            measured.bytes_max <= (count * size_of::<usize>() + 128) as u64,
            "merge expanded input positions with fanout: {measured:?}"
        );
        assert_eq!(
            crate::operator::join::matches::pair_pushes(),
            0,
            "owner assembly must append existing runs rather than rebuild every pair"
        );
        assert_eq!(pool.reserved(), 0);
    }
}
