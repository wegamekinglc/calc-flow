//! Incoming identity/runs plus opposite row references, without pair descriptors.

use super::MatchedPair;
use std::ops::Range;

pub(super) trait Matches: Sync {
    fn len(&self) -> usize;
    fn pair(&self, index: usize) -> MatchedPair;
}

impl dyn Matches + '_ {
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn first(&self) -> Option<MatchedPair> {
        (self.len() != 0).then(|| self.pair(0))
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = MatchedPair> + '_ {
        (0..self.len()).map(|index| self.pair(index))
    }
    pub(super) fn slice(&self, range: Range<usize>) -> Slice<'_> {
        Slice {
            matches: self,
            range,
        }
    }
}

pub(super) struct Slice<'a> {
    matches: &'a dyn Matches,
    range: Range<usize>,
}
impl Matches for Slice<'_> {
    fn len(&self) -> usize {
        self.range.len()
    }
    fn pair(&self, index: usize) -> MatchedPair {
        self.matches.pair(self.range.start + index)
    }
}

enum Incoming {
    Identity { start: usize },
    Runs(Vec<(usize, usize)>),
}

pub(super) struct Selection {
    incoming: Incoming,
    opposite: Vec<usize>,
    incoming_capacity: usize,
}

impl Selection {
    pub(super) fn with_capacity(matches: usize, incoming: usize) -> Self {
        Self {
            incoming: Incoming::Identity { start: 0 },
            opposite: Vec::with_capacity(matches),
            incoming_capacity: incoming.min(matches),
        }
    }
    pub(super) fn push(&mut self, pair: MatchedPair) {
        let length = self.opposite.len();
        match &mut self.incoming {
            Incoming::Identity { start } if length == 0 => *start = pair.pos,
            Incoming::Identity { start } if pair.pos != *start + length => {
                let mut runs = Vec::with_capacity(self.incoming_capacity.max(length + 1));
                runs.extend((0..length).map(|index| (index, *start + index)));
                if runs
                    .last()
                    .is_none_or(|(_, position)| *position != pair.pos)
                {
                    runs.push((length, pair.pos));
                }
                self.incoming = Incoming::Runs(runs);
            }
            Incoming::Runs(runs) => {
                if runs
                    .last()
                    .is_none_or(|(_, position)| *position != pair.pos)
                {
                    runs.push((length, pair.pos));
                }
            }
            Incoming::Identity { .. } => {}
        }
        self.opposite.push(pair.opposite_index);
    }
    pub(super) fn extend(&mut self, pairs: impl IntoIterator<Item = MatchedPair>) {
        for pair in pairs {
            self.push(pair);
        }
    }
    pub(super) fn incoming_capacity(&self) -> usize {
        self.incoming_capacity
    }
    pub(super) fn len(&self) -> usize {
        self.opposite.len()
    }
    pub(super) fn is_empty(&self) -> bool {
        self.opposite.is_empty()
    }
    pub(super) fn iter(&self) -> impl ExactSizeIterator<Item = MatchedPair> + '_ {
        (0..self.len()).map(|index| self.pair(index))
    }
}

impl Matches for Selection {
    fn len(&self) -> usize {
        self.opposite.len()
    }
    fn pair(&self, index: usize) -> MatchedPair {
        let pos = match &self.incoming {
            Incoming::Identity { start } => *start + index,
            Incoming::Runs(runs) => runs[runs.partition_point(|(first, _)| *first <= index) - 1].1,
        };
        MatchedPair {
            pos,
            opposite_index: self.opposite[index],
        }
    }
}

impl From<Vec<MatchedPair>> for Selection {
    fn from(pairs: Vec<MatchedPair>) -> Self {
        let mut result = Self::with_capacity(pairs.len(), pairs.len());
        result.extend(pairs);
        result
    }
}

impl Matches for Vec<MatchedPair> {
    fn len(&self) -> usize {
        Vec::len(self)
    }
    fn pair(&self, index: usize) -> MatchedPair {
        self[index]
    }
}
impl<const N: usize> Matches for [MatchedPair; N] {
    fn len(&self) -> usize {
        N
    }
    fn pair(&self, index: usize) -> MatchedPair {
        self[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_and_fanout_selections_do_not_allocate_pair_descriptors() {
        for fanout in [false, true] {
            let count = 100_000;
            let allocations = allocation_counter::measure(|| {
                let mut selection = Selection::with_capacity(count, if fanout { 1 } else { count });
                selection.extend((0..count).map(|index| MatchedPair {
                    pos: if fanout { 7 } else { index + 7 },
                    opposite_index: index,
                }));
                assert_eq!(selection.len(), count);
                assert_eq!(
                    selection.pair(count - 1).pos,
                    if fanout { 7 } else { count + 6 }
                );
                if !fanout {
                    assert!(matches!(
                        selection.incoming,
                        Incoming::Identity { start: 7 }
                    ));
                }
            });
            assert!(
                allocations.bytes_max <= (count * size_of::<usize>() + 64) as u64,
                "incoming identity or one fanout run must not allocate a position per match: {allocations:?}"
            );
        }
    }

    #[test]
    fn test_ranges_preserve_sparse_incoming_and_opposite_order_across_run_boundaries() {
        let expected = [(2, 9), (2, 3), (2, 5), (7, 1), (9, 8), (9, 0)];
        let selection: Selection = expected
            .iter()
            .map(|&(pos, opposite_index)| MatchedPair {
                pos,
                opposite_index,
            })
            .collect::<Vec<_>>()
            .into();
        let selected = (&selection as &dyn Matches).slice(1..5);
        let selected: &dyn Matches = &selected;
        assert_eq!(
            selected
                .iter()
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>(),
            expected[1..5]
        );
    }
}
