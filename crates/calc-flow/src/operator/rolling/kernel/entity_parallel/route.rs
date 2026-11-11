use std::mem::size_of;

use super::{EntityAssignment, LaneRow};

#[derive(Clone, Copy, Debug)]
pub(crate) enum RoutingLayout {
    Bitmap { words: usize },
    Explicit,
}

impl RoutingLayout {
    pub(crate) fn choose(rows: usize, lanes: usize) -> Option<Self> {
        let words = rows
            .checked_div(64)?
            .checked_add(usize::from(!rows.is_multiple_of(64)))?;
        let bitmap = lanes
            .checked_mul(words)
            .and_then(|total| total.checked_mul(size_of::<u64>()));
        let explicit = rows.checked_mul(size_of::<LaneRow>());
        match (bitmap, explicit) {
            (Some(left), Some(right)) if left <= right => Some(Self::Bitmap { words }),
            (_, Some(_)) => Some(Self::Explicit),
            (Some(_), None) => Some(Self::Bitmap { words }),
            (None, None) => None,
        }
    }

    pub(crate) fn payload_bytes(self, rows: usize, lanes: usize) -> Option<usize> {
        match self {
            Self::Bitmap { words } => lanes.checked_mul(words)?.checked_mul(size_of::<u64>()),
            Self::Explicit => rows.checked_mul(size_of::<LaneRow>()),
        }
    }
}

#[derive(Clone)]
enum Storage {
    Bitmap {
        words: Vec<u64>,
        count: usize,
        universe: usize,
    },
    Explicit(Vec<LaneRow>),
}

#[derive(Clone)]
pub(crate) struct LaneRoute {
    lane: usize,
    expected: usize,
    storage: Storage,
}

impl LaneRoute {
    pub(crate) fn new(layout: RoutingLayout, lane: usize, rows: usize, expected: usize) -> Self {
        let storage = match layout {
            RoutingLayout::Bitmap { words } => Storage::Bitmap {
                words: vec![0; words],
                count: 0,
                universe: rows,
            },
            RoutingLayout::Explicit => Storage::Explicit(Vec::with_capacity(expected)),
        };
        Self {
            lane,
            expected,
            storage,
        }
    }

    pub(crate) fn push(&mut self, row: LaneRow) -> Option<()> {
        if self.len() >= self.expected {
            return None;
        }
        match &mut self.storage {
            Storage::Bitmap {
                words,
                count,
                universe,
            } => {
                let row = row.original_row as usize;
                if row >= *universe {
                    return None;
                }
                let word = words.get_mut(row / 64)?;
                let mask = 1_u64 << (row % 64);
                if *word & mask != 0 {
                    return None;
                }
                *word |= mask;
                *count += 1;
            }
            Storage::Explicit(rows) => rows.push(row),
        }
        Some(())
    }

    pub(crate) fn len(&self) -> usize {
        match &self.storage {
            Storage::Bitmap { count, .. } => *count,
            Storage::Explicit(rows) => rows.len(),
        }
    }

    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub(crate) fn complete(&self, lane: usize, universe: usize) -> bool {
        if self.lane != lane || self.len() != self.expected {
            return false;
        }
        match &self.storage {
            Storage::Bitmap {
                words,
                universe: total,
                ..
            } => {
                *total == universe
                    && words.len() == universe.div_ceil(64)
                    && (universe.is_multiple_of(64)
                        || words
                            .last()
                            .is_some_and(|word| word >> (universe % 64) == 0))
            }
            Storage::Explicit(_) => true,
        }
    }

    pub(crate) fn lane(&self) -> usize {
        self.lane
    }

    pub(crate) fn is_explicit(&self) -> bool {
        matches!(self.storage, Storage::Explicit(_))
    }

    pub(crate) fn cursor<'a>(
        &'a self,
        entity_ids: &'a [usize],
        assignments: &'a [EntityAssignment],
    ) -> RouteCursor<'a> {
        RouteCursor {
            route: self,
            entity_ids,
            assignments,
            index: 0,
            word: 0,
            base: 0,
            polled: false,
            consumed: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn payload_bytes(&self) -> usize {
        match &self.storage {
            Storage::Bitmap { words, .. } => words.capacity() * size_of::<u64>(),
            Storage::Explicit(rows) => rows.capacity() * size_of::<LaneRow>(),
        }
    }

    #[cfg(test)]
    pub(crate) fn replace_explicit(&mut self, rows: Vec<LaneRow>) -> &mut Vec<LaneRow> {
        self.storage = Storage::Explicit(rows);
        let Storage::Explicit(rows) = &mut self.storage else {
            unreachable!()
        };
        rows
    }

    #[cfg(test)]
    pub(crate) fn explicit_mut(&mut self) -> Option<&mut Vec<LaneRow>> {
        match &mut self.storage {
            Storage::Explicit(rows) => Some(rows),
            Storage::Bitmap { .. } => None,
        }
    }

    #[cfg(test)]
    pub(crate) fn bitmap_mut_for_test(&mut self) -> Option<(&mut Vec<u64>, &mut usize)> {
        match &mut self.storage {
            Storage::Bitmap { words, count, .. } => Some((words, count)),
            Storage::Explicit(_) => None,
        }
    }
}

pub(crate) enum RouteStep {
    Poll,
    Row(LaneRow),
    Invalid,
}

pub(crate) struct RouteCursor<'a> {
    route: &'a LaneRoute,
    entity_ids: &'a [usize],
    assignments: &'a [EntityAssignment],
    index: usize,
    word: u64,
    base: usize,
    polled: bool,
    consumed: usize,
}

impl RouteCursor<'_> {
    pub(crate) fn next_step(&mut self) -> Option<RouteStep> {
        let Storage::Bitmap {
            words, universe, ..
        } = &self.route.storage
        else {
            let Storage::Explicit(rows) = &self.route.storage else {
                unreachable!()
            };
            let row = rows.get(self.index).copied()?;
            self.index += 1;
            return Some(RouteStep::Row(row));
        };
        loop {
            if self.word != 0 {
                let bit = self.word.trailing_zeros() as usize;
                self.word &= self.word - 1;
                return Some(self.decode(self.base.checked_add(bit), *universe));
            }
            let word = *words.get(self.index)?;
            if self.index.is_multiple_of(4) && !self.polled {
                self.polled = true;
                return Some(RouteStep::Poll);
            }
            self.polled = false;
            let Some(base) = self.index.checked_mul(64) else {
                return Some(RouteStep::Invalid);
            };
            self.base = base;
            self.word = word;
            self.index += 1;
            #[cfg(test)]
            super::ROUTE_WORD_VISITS.set(super::ROUTE_WORD_VISITS.get() + 1);
        }
    }

    fn decode(&self, row: Option<usize>, universe: usize) -> RouteStep {
        let Some(row) = row.filter(|row| *row < universe) else {
            return RouteStep::Invalid;
        };
        let Some(assignment) = self
            .entity_ids
            .get(row)
            .and_then(|id| self.assignments.get(*id))
        else {
            return RouteStep::Invalid;
        };
        let (Ok(original_row), Ok(lane_entity)) =
            (u32::try_from(row), u32::try_from(assignment.local))
        else {
            return RouteStep::Invalid;
        };
        if assignment.lane != self.route.lane {
            return RouteStep::Invalid;
        }
        RouteStep::Row(LaneRow {
            original_row,
            lane_entity,
        })
    }

    pub(crate) fn next_row(&mut self) -> Option<Result<LaneRow, ()>> {
        loop {
            match self.next_step()? {
                RouteStep::Poll => {}
                RouteStep::Row(row) => {
                    self.consumed += 1;
                    return Some(Ok(row));
                }
                RouteStep::Invalid => return Some(Err(())),
            }
        }
    }

    pub(crate) fn consumed(&self) -> usize {
        self.consumed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn filled(rows: usize) -> LaneRoute {
        let mut route = LaneRoute::new(
            RoutingLayout::Bitmap {
                words: rows.div_ceil(64),
            },
            0,
            rows,
            rows,
        );
        for row in 0..rows {
            route
                .push(LaneRow {
                    original_row: u32::try_from(row).unwrap(),
                    lane_entity: 0,
                })
                .unwrap();
        }
        route
    }

    #[test]
    fn test_bitmap_route_crosses_word_boundaries_in_order_without_tail_rows() {
        for rows in [63, 64, 65] {
            let route = filled(rows);
            assert!(route.complete(0, rows));
            let ids = vec![0; rows];
            let assignments = [EntityAssignment {
                rows,
                lane: 0,
                local: 0,
            }];
            let mut cursor = route.cursor(&ids, &assignments);
            super::super::ROUTE_WORD_VISITS.set(0);
            let actual = std::iter::from_fn(|| cursor.next_row())
                .map(|row| row.unwrap().original_row)
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                (0..u32::try_from(rows).unwrap()).collect::<Vec<_>>()
            );
            assert_eq!(super::super::ROUTE_WORD_VISITS.get(), rows.div_ceil(64));
        }
    }

    #[test]
    fn test_route_layout_chooses_checked_payload_without_a_lane_ceiling() {
        assert!(matches!(
            RoutingLayout::choose(128, 4),
            Some(RoutingLayout::Bitmap { words: 2 })
        ));
        assert!(matches!(
            RoutingLayout::choose(128, 65),
            Some(RoutingLayout::Explicit)
        ));
        assert!(RoutingLayout::choose(usize::MAX, usize::MAX).is_none());
    }

    #[test]
    fn test_sparse_bitmap_cursor_emits_task_polls_without_off_lane_rows() {
        let mut route = LaneRoute::new(RoutingLayout::Bitmap { words: 16 }, 0, 1_024, 2);
        for row in [0, 1_023] {
            route
                .push(LaneRow {
                    original_row: row,
                    lane_entity: 0,
                })
                .unwrap();
        }
        let ids = vec![0; 1_024];
        let assignments = [EntityAssignment {
            rows: 2,
            lane: 0,
            local: 0,
        }];
        let mut cursor = route.cursor(&ids, &assignments);
        let mut owned = Vec::new();
        let mut polls = Vec::new();
        while let Some(step) = cursor.next_step() {
            match step {
                RouteStep::Poll => polls.push(cursor.index * 64),
                RouteStep::Row(row) => owned.push(row.original_row),
                RouteStep::Invalid => panic!("producer-qualified route"),
            }
        }
        assert_eq!(owned, [0, 1_023]);
        assert_eq!(polls, [0, 256, 512, 768]);
    }

    #[test]
    fn test_bitmap_cursor_rejects_bad_tail_and_stale_count_without_indexing_outside_input() {
        let mut route = filled(65);
        let Storage::Bitmap { words, count, .. } = &mut route.storage else {
            unreachable!()
        };
        *count -= 1;
        words[1] |= 1_u64 << 63;
        assert!(!route.complete(0, 65));
        let ids = vec![0; 65];
        let assignments = [EntityAssignment {
            rows: 65,
            lane: 0,
            local: 0,
        }];
        let mut cursor = route.cursor(&ids, &assignments);
        for row in 0..65 {
            assert_eq!(
                cursor.next_row().unwrap().unwrap().original_row as usize,
                row
            );
        }
        assert!(cursor.next_row().unwrap().is_err());
        assert!(cursor.next_row().is_none());
    }
}
