use super::super::{MatchedPair, native_lookup::scratch_error};
use crate::Result;
use datafusion::execution::memory_pool::MemoryReservation;
use parking_lot::Mutex;
use std::sync::Arc;

pub(super) struct PairOutput {
    pub(super) pairs: Vec<MatchedPair>,
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
    pub(super) pairs: Vec<MatchedPair>,
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
    let mut pairs = Vec::with_capacity(count);
    for fragment in fragments {
        pairs.extend(fragment.pairs);
        drop(fragment.credit);
    }
    debug_assert_eq!(pairs.len(), count);
    Ok(PairOutput { pairs, credit })
}
