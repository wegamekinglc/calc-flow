//! Incoming identity/runs plus opposite row references, without pair descriptors.

use super::MatchedPair;
use std::{borrow::Cow, ops::Range};

#[cfg(test)]
thread_local! {
    static PAIR_PUSHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(in crate::operator::join) fn reset_pair_pushes() {
    PAIR_PUSHES.set(0);
}

#[cfg(test)]
pub(in crate::operator::join) fn pair_pushes() -> usize {
    PAIR_PUSHES.get()
}

pub(super) trait Matches: Sync {
    fn view(&self) -> View<'_>;
    fn len(&self) -> usize {
        self.view().len()
    }
    fn pair(&self, index: usize) -> MatchedPair {
        self.view().pair(index)
    }
}

impl dyn Matches + '_ {
    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub(super) fn first(&self) -> Option<MatchedPair> {
        (self.len() != 0).then(|| self.pair(0))
    }
    pub(super) fn iter(&self) -> Cursor<'_> {
        self.view().iter()
    }
    pub(super) fn slice(&self, range: Range<usize>) -> View<'_> {
        self.view().slice(range)
    }
}

/// Borrowed pairs; `Runs` starts with the run that contains index `base`.
#[derive(Clone)]
pub(super) enum View<'a> {
    Pairs(&'a [MatchedPair]),
    Identity {
        start: usize,
        opposite: &'a [usize],
    },
    Runs {
        runs: &'a [(usize, usize)],
        base: usize,
        opposite: &'a [usize],
    },
    Intervals(Cow<'a, [RangeMatch]>),
}

impl<'a> View<'a> {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Pairs(pairs) => pairs.len(),
            Self::Identity { opposite, .. } | Self::Runs { opposite, .. } => opposite.len(),
            Self::Intervals(ranges) => ranges.iter().map(|matched| matched.range.len()).sum(),
        }
    }

    pub(super) fn pair(&self, index: usize) -> MatchedPair {
        match self {
            Self::Pairs(pairs) => pairs[index],
            Self::Identity { start, opposite } => MatchedPair {
                pos: start + index,
                opposite_index: opposite[index],
            },
            Self::Runs {
                runs,
                base,
                opposite,
            } => MatchedPair {
                pos: runs[run_containing(runs, base + index)].1,
                opposite_index: opposite[index],
            },
            Self::Intervals(ranges) => {
                let base = ranges.first().map_or(0, |matched| matched.first);
                let index = base + index;
                let matched = &ranges[run_containing_ranges(ranges, index)];
                MatchedPair {
                    pos: matched.position,
                    opposite_index: matched.range.start + index - matched.first,
                }
            }
        }
    }

    pub(super) fn slice(self, range: Range<usize>) -> Self {
        match self {
            Self::Pairs(pairs) => Self::Pairs(&pairs[range]),
            Self::Identity { start, opposite } => Self::Identity {
                start: start + range.start,
                opposite: &opposite[range],
            },
            Self::Runs {
                runs,
                base,
                opposite,
            } => {
                let base = base + range.start;
                Self::Runs {
                    runs: &runs[run_containing(runs, base)..],
                    base,
                    opposite: &opposite[range],
                }
            }
            Self::Intervals(ranges) => {
                let mut sliced = Vec::new();
                for matched in ranges.iter() {
                    let first = matched.first.max(range.start);
                    let last = (matched.first + matched.range.len()).min(range.end);
                    if first < last {
                        sliced.push(RangeMatch {
                            first,
                            position: matched.position,
                            range: matched.range.start + first - matched.first
                                ..matched.range.start + last - matched.first,
                        });
                    }
                }
                Self::Intervals(Cow::Owned(sliced))
            }
        }
    }

    /// Visits pairs in order, advancing incoming runs instead of searching.
    pub(super) fn iter(self) -> Cursor<'a> {
        Cursor {
            view: self,
            index: 0,
            run: 0,
        }
    }
}

impl Matches for View<'_> {
    fn view(&self) -> View<'_> {
        self.clone()
    }
}

fn run_containing(runs: &[(usize, usize)], index: usize) -> usize {
    #[cfg(test)]
    tests::RUN_SEARCHES.with(|count| count.set(count.get() + 1));
    runs.partition_point(|(first, _)| *first <= index) - 1
}

fn run_containing_ranges(ranges: &[RangeMatch], index: usize) -> usize {
    ranges.partition_point(|matched| matched.first <= index) - 1
}

/// Whole-selection iteration in O(pairs + runs).
#[derive(Clone)]
pub(super) struct Cursor<'a> {
    view: View<'a>,
    index: usize,
    run: usize,
}

impl Iterator for Cursor<'_> {
    type Item = MatchedPair;

    #[inline]
    fn next(&mut self) -> Option<MatchedPair> {
        let index = self.index;
        let pair = match &self.view {
            View::Pairs(pairs) => *pairs.get(index)?,
            View::Identity { start, opposite } => MatchedPair {
                pos: start + index,
                opposite_index: *opposite.get(index)?,
            },
            View::Runs {
                runs,
                base,
                opposite,
            } => {
                let opposite_index = *opposite.get(index)?;
                while runs
                    .get(self.run + 1)
                    .is_some_and(|(first, _)| *first <= base + index)
                {
                    self.run += 1;
                }
                MatchedPair {
                    pos: runs[self.run].1,
                    opposite_index,
                }
            }
            View::Intervals(ranges) => {
                let matched = ranges.get(self.run)?;
                if index < matched.first {
                    self.index = matched.first;
                    return self.next();
                }
                if index >= matched.first + matched.range.len() {
                    self.run += 1;
                    return self.next();
                }
                MatchedPair {
                    pos: matched.position,
                    opposite_index: matched.range.start + index - matched.first,
                }
            }
        };
        self.index = index + 1;
        Some(pair)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len() - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for Cursor<'_> {}

enum Incoming {
    Identity { start: usize },
    Runs(Vec<(usize, usize)>),
}

#[derive(Clone, Debug)]
pub(super) struct RangeMatch {
    first: usize,
    position: usize,
    range: Range<usize>,
}

#[derive(Debug)]
enum Opposite {
    Indices(Vec<usize>),
    Ranges(Vec<RangeMatch>),
}

impl Opposite {
    fn len(&self) -> usize {
        match self {
            Self::Indices(indices) => indices.len(),
            Self::Ranges(ranges) => ranges.iter().map(|matched| matched.range.len()).sum(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn push_index(&mut self, position: usize, index: usize) {
        let first = self.len();
        match self {
            Self::Indices(indices) => indices.push(index),
            Self::Ranges(ranges) => ranges.push(RangeMatch {
                first,
                position,
                range: index..index + 1,
            }),
        }
    }

    fn push_range(&mut self, position: usize, range: Range<usize>) {
        let first = self.len();
        match self {
            Self::Indices(indices) if indices.is_empty() => {
                *self = Self::Ranges(vec![RangeMatch {
                    first: 0,
                    position,
                    range,
                }]);
            }
            Self::Indices(indices) => indices.extend(range),
            Self::Ranges(ranges) => ranges.push(RangeMatch {
                first,
                position,
                range,
            }),
        }
    }

    fn extend_values(
        &mut self,
        values: impl IntoIterator<Item = usize>,
        position: impl Fn(usize) -> usize,
    ) {
        if let Self::Indices(indices) = self {
            indices.extend(values);
            return;
        }
        let base = self.len();
        let mut current: Option<(usize, usize, usize)> = None;
        for (offset, value) in values.into_iter().enumerate() {
            if let Some((start, previous, first)) = current {
                if value == previous + 1 {
                    current = Some((start, value, first));
                    continue;
                }
                self.push_range(position(first), start..previous + 1);
            }
            current = Some((value, value, base + offset));
        }
        if let Some((start, previous, first)) = current {
            self.push_range(position(first), start..previous + 1);
        }
    }
}

pub(super) struct Selection {
    incoming: Incoming,
    opposite: Opposite,
    incoming_capacity: usize,
    opposite_capacity: usize,
}

impl Selection {
    pub(super) fn with_capacity(matches: usize, incoming: usize) -> Self {
        Self {
            incoming: Incoming::Identity { start: 0 },
            opposite: Opposite::Indices(Vec::new()),
            incoming_capacity: incoming.min(matches),
            opposite_capacity: matches,
        }
    }

    fn reserve_indices(&mut self) {
        let capacity = self.opposite_capacity;
        if let Opposite::Indices(indices) = &mut self.opposite
            && indices.capacity() == 0
        {
            indices.reserve(capacity);
        }
    }

    pub(super) fn push(&mut self, pair: MatchedPair) {
        #[cfg(test)]
        PAIR_PUSHES.set(PAIR_PUSHES.get() + 1);
        self.reserve_indices();
        self.append_incoming(pair.pos, self.opposite.len(), 1);
        self.opposite.push_index(pair.pos, pair.opposite_index);
    }

    fn append_incoming(&mut self, position: usize, length: usize, count: usize) {
        debug_assert_ne!(count, 0);
        match &mut self.incoming {
            Incoming::Identity { start } if length == 0 && count == 1 => *start = position,
            Incoming::Identity { start } if count != 1 || position != *start + length => {
                let mut runs = Vec::with_capacity(self.incoming_capacity.max(length + 1));
                runs.extend((0..length).map(|index| (index, *start + index)));
                if runs
                    .last()
                    .is_none_or(|(_, previous)| *previous != position)
                {
                    runs.push((length, position));
                }
                self.incoming = Incoming::Runs(runs);
            }
            Incoming::Runs(runs) => {
                if runs
                    .last()
                    .is_none_or(|(_, previous)| *previous != position)
                {
                    runs.push((length, position));
                }
            }
            Incoming::Identity { .. } => {}
        }
    }

    pub(super) fn append_run(&mut self, position: usize, opposite: impl Iterator<Item = usize>) {
        self.reserve_indices();
        let start = self.opposite.len();
        self.opposite.extend_values(opposite, |_| position);
        let count = self.opposite.len() - start;
        if count != 0 {
            self.append_incoming(position, start, count);
        }
    }

    pub(super) fn append_interval(&mut self, position: usize, range: Range<usize>) {
        if range.is_empty() {
            return;
        }
        let start = self.opposite.len();
        self.append_incoming(position, start, range.len());
        self.opposite.push_range(position, range);
    }

    pub(super) fn append_view(&mut self, view: View<'_>) {
        if view.len() == 0 {
            return;
        }
        match view {
            View::Pairs(mut pairs) => {
                while let Some(first) = pairs.first() {
                    let count = pairs
                        .iter()
                        .position(|pair| pair.pos != first.pos)
                        .unwrap_or(pairs.len());
                    self.append_run(
                        first.pos,
                        pairs[..count].iter().map(|pair| pair.opposite_index),
                    );
                    pairs = &pairs[count..];
                }
            }
            View::Identity { start, opposite } => {
                let offset = self.opposite.len();
                match &mut self.incoming {
                    Incoming::Identity { start: destination } if offset == 0 => {
                        *destination = start;
                    }
                    Incoming::Identity { start: destination } if start == *destination + offset => {
                    }
                    _ => {
                        for index in 0..opposite.len() {
                            self.append_incoming(start + index, offset + index, 1);
                        }
                    }
                }
                self.opposite
                    .extend_values(opposite.iter().copied(), |index| start + index - offset);
            }
            View::Runs {
                runs,
                base,
                opposite,
            } => self.append_runs(runs, base, opposite),
            View::Intervals(ranges) => {
                for matched in ranges.iter() {
                    self.append_interval(matched.position, matched.range.clone());
                }
            }
        }
    }

    fn append_runs(&mut self, runs: &[(usize, usize)], base: usize, opposite: &[usize]) {
        let offset = self.opposite.len();
        let end = base + opposite.len();
        for (index, &(first, position)) in runs.iter().enumerate() {
            if first >= end {
                break;
            }
            let start = first.max(base);
            let next = runs.get(index + 1).map_or(end, |&(next, _)| next.min(end));
            self.append_incoming(position, offset + start - base, next - start);
        }
        self.opposite
            .extend_values(opposite.iter().copied(), |index| {
                runs[run_containing(runs, index)].1
            });
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
    pub(super) fn iter(&self) -> Cursor<'_> {
        self.view().iter()
    }
}

impl Matches for Selection {
    fn view(&self) -> View<'_> {
        match &self.opposite {
            Opposite::Ranges(ranges) => View::Intervals(Cow::Borrowed(ranges)),
            Opposite::Indices(opposite) => match &self.incoming {
                Incoming::Identity { start } => View::Identity {
                    start: *start,
                    opposite,
                },
                Incoming::Runs(runs) => View::Runs {
                    runs,
                    base: 0,
                    opposite,
                },
            },
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
    fn view(&self) -> View<'_> {
        View::Pairs(self)
    }
}
impl<const N: usize> Matches for [MatchedPair; N] {
    fn view(&self) -> View<'_> {
        View::Pairs(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    thread_local! {
        pub(super) static RUN_SEARCHES: Cell<usize> = const { Cell::new(0) };
    }

    #[test]
    fn test_bulk_append_preserves_sliced_runs_and_identity_transitions() {
        let identity: Selection = (10..14)
            .map(|pos| MatchedPair {
                pos,
                opposite_index: pos * 2,
            })
            .collect::<Vec<_>>()
            .into();
        let fanout: Selection = [(13, 7), (13, 5), (13, 9), (17, 1), (17, 8), (19, 4)]
            .map(|(pos, opposite_index)| MatchedPair {
                pos,
                opposite_index,
            })
            .to_vec()
            .into();
        let raw = [(19, 3), (19, 2), (25, 6)].map(|(pos, opposite_index)| MatchedPair {
            pos,
            opposite_index,
        });
        let views = [
            identity.view().slice(1..4),
            fanout.view().slice(1..5),
            View::Pairs(&raw),
            fanout.view().slice(2..2),
            identity.view().slice(0..1),
        ];
        let expected = views
            .iter()
            .flat_map(|view| view.clone().iter())
            .map(|pair| (pair.pos, pair.opposite_index))
            .collect::<Vec<_>>();
        let mut selection = Selection::with_capacity(expected.len(), expected.len());
        reset_pair_pushes();
        selection.append_run(999, std::iter::empty());
        for view in views {
            selection.append_view(view);
        }
        assert_eq!(pair_pushes(), 0);
        assert_eq!(
            selection
                .iter()
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>(),
            expected
        );
        for (index, expected) in expected.into_iter().enumerate() {
            let pair = selection.pair(index);
            assert_eq!((pair.pos, pair.opposite_index), expected);
        }

        let mut contiguous = Selection::with_capacity(4, 4);
        contiguous.append_view(identity.view().slice(0..2));
        contiguous.append_view(identity.view().slice(2..4));
        assert!(matches!(
            contiguous.incoming,
            Incoming::Identity { start: 10 }
        ));
    }

    #[test]
    fn test_contiguous_intervals_avoid_per_match_index_storage() {
        let count = 100_000;
        let allocations = allocation_counter::measure(|| {
            let mut selection = Selection::with_capacity(count, 1);
            selection.append_interval(7, 0..count);
            assert_eq!(selection.len(), count);
            assert_eq!(selection.pair(0).opposite_index, 0);
            assert_eq!(selection.pair(count - 1).opposite_index, count - 1);
            let sliced = selection.view().slice(10..20);
            assert_eq!(sliced.len(), 10);
            assert_eq!(sliced.pair(0).opposite_index, 10);
            assert_eq!(sliced.pair(9).opposite_index, 19);
            let mut merged = Selection::with_capacity(10, 1);
            merged.append_view(sliced);
            assert_eq!(merged.len(), 10);
            assert_eq!(merged.pair(0).opposite_index, 10);
            assert_eq!(merged.pair(9).opposite_index, 19);
        });
        assert!(
            allocations.bytes_max < 4_096,
            "one interval must not allocate an index per match: {allocations:?}"
        );
    }

    #[test]
    fn test_bulk_fanout_append_reuses_existing_incoming_run_without_pair_storage() {
        let count = 100_000;
        reset_pair_pushes();
        let allocations = allocation_counter::measure(|| {
            let mut selection = Selection::with_capacity(count, 1);
            for first in (0..count).step_by(4_096) {
                selection.append_run(7, first..(first + 4_096).min(count));
            }
            assert_eq!(selection.len(), count);
            let Incoming::Runs(runs) = &selection.incoming else {
                panic!("fanout has a constant incoming run");
            };
            assert_eq!(runs.as_slice(), [(0, 7)]);
            assert!(
                selection
                    .iter()
                    .enumerate()
                    .all(|(index, pair)| { pair.pos == 7 && pair.opposite_index == index })
            );
        });
        assert_eq!(pair_pushes(), 0);
        assert!(
            allocations.bytes_max <= (count * size_of::<usize>() + 64) as u64,
            "bulk fanout must retain one incoming run: {allocations:?}"
        );
    }

    #[test]
    fn test_fanout_iteration_advances_runs_without_searching_per_pair() {
        let (incoming, fanout) = (1_000, 11);
        let expected = (0..incoming)
            .flat_map(|pos| {
                (0..fanout).map(move |offset| MatchedPair {
                    pos: pos * 3 + 1,
                    opposite_index: pos * fanout + offset,
                })
            })
            .collect::<Vec<_>>();
        let selection: Selection = expected.clone().into();
        let identities = |pairs: &mut dyn Iterator<Item = MatchedPair>| {
            pairs
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>()
        };
        let reference = |range: Range<usize>| identities(&mut expected[range].iter().copied());
        let matches: &dyn Matches = &selection;
        let slice = matches.slice(5..expected.len() - 7);
        let sliced: &dyn Matches = &slice;
        let boundary = matches.slice(fanout..expected.len() - 7);
        let boundary_matches: &dyn Matches = &boundary;
        RUN_SEARCHES.with(|count| count.set(0));
        assert_eq!(
            identities(&mut matches.iter()),
            reference(0..expected.len())
        );
        assert_eq!(
            identities(&mut selection.iter()),
            reference(0..expected.len())
        );
        assert_eq!(
            identities(&mut sliced.iter()),
            reference(5..expected.len() - 7)
        );
        // `fanout` is the first index of the second run, so this slice starts
        // exactly on a run boundary.
        assert_eq!(
            identities(&mut boundary_matches.iter()),
            reference(fanout..expected.len() - 7)
        );
        let searches = RUN_SEARCHES.with(Cell::get);
        assert!(
            searches <= 3,
            "whole-selection iteration must advance {incoming} incoming runs sequentially, \
             not search once per pair: {searches} searches for {} pairs",
            3 * expected.len()
        );
    }

    #[test]
    fn test_cursor_iteration_matches_random_access_pairs_in_every_shape() {
        let shapes = vec![
            Vec::new(),
            vec![MatchedPair {
                pos: 4,
                opposite_index: 9,
            }],
            (0..50)
                .map(|index| MatchedPair {
                    pos: index + 3,
                    opposite_index: index * 2,
                })
                .collect(),
            // Mixed fan-out (1..=4 matches per incoming row) exercises the
            // run-representation with uneven run lengths and run starts.
            (0..40)
                .flat_map(|pos| {
                    (0..=pos % 4).map(move |offset| MatchedPair {
                        pos: pos * 5 + 2,
                        opposite_index: pos * 7 + offset,
                    })
                })
                .collect::<Vec<_>>(),
        ];
        let identities = |pairs: &mut dyn Iterator<Item = MatchedPair>| {
            pairs
                .map(|pair| (pair.pos, pair.opposite_index))
                .collect::<Vec<_>>()
        };
        for expected in shapes {
            let selection: Selection = expected.clone().into();
            let matches: &dyn Matches = &selection;
            let random_access = |range: Range<usize>| {
                (0..range.len())
                    .map(|index| {
                        let pair = matches.pair(range.start + index);
                        (pair.pos, pair.opposite_index)
                    })
                    .collect::<Vec<_>>()
            };
            let len = expected.len();
            assert_eq!(identities(&mut matches.iter()), random_access(0..len));
            let bounds = [
                (0, len),
                (1.min(len), 3.min(len)),
                (len.saturating_sub(2), len),
            ];
            for (start, end) in bounds {
                let slice = matches.slice(start..end);
                let sliced: &dyn Matches = &slice;
                assert_eq!(
                    identities(&mut sliced.iter()),
                    random_access(start..end),
                    "sliced cursor must agree with pair() in {start}..{end}"
                );
            }
        }
    }

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
