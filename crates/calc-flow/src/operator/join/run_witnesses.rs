use super::{Bucket, GroupId, Incoming, JoinSide, MORSEL_ROWS, Probe, RunState};
use crate::Result;
use datafusion::arrow::{array::BooleanArray, buffer::BooleanBuffer};
use std::{
    collections::{BTreeMap, BTreeSet},
    ops::Range,
};

#[cfg(test)]
thread_local! {
    pub(super) static ELIGIBILITY_READS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Default)]
pub(super) struct InputMatches {
    pub(super) count: usize,
    pub(super) witnesses: Witnesses,
}

#[derive(Default)]
pub(super) struct Witnesses {
    groups: BTreeMap<GroupId, BucketWitnesses>,
    left_cut: Option<(usize, u64)>,
}

#[derive(Default)]
struct BucketWitnesses {
    positions: BTreeSet<usize>,
    examined: Range<usize>,
}

impl Witnesses {
    pub(super) fn select(
        &mut self,
        probe: &mut Probe,
        position: usize,
        (group, bucket): (GroupId, &Bucket),
        window: Range<usize>,
        room: usize,
        check: &mut dyn FnMut() -> Result<()>,
    ) -> Result<()> {
        if self.groups.get(&group).is_some_and(|pending| {
            pending.examined.start <= window.start && window.end <= pending.examined.end
        }) {
            return Ok(());
        }
        let mut start = None;
        for (offset, entry) in bucket.active()[window.clone()].iter().enumerate() {
            if offset.is_multiple_of(MORSEL_ROWS) {
                check()?;
            }
            #[cfg(test)]
            ELIGIBILITY_READS.set(ELIGIBILITY_READS.get() + 1);
            let index = window.start + offset;
            if !entry.locator.witnessed
                && self
                    .groups
                    .entry(group)
                    .or_default()
                    .positions
                    .insert(bucket.head + index)
            {
                let first = *start.get_or_insert(index);
                if index + 1 - first > room.saturating_sub(probe.count) {
                    return probe.push(position, first..index + 1);
                }
            } else if let Some(first) = start.take() {
                probe.push(position, first..index)?;
            }
        }
        if let Some(first) = start {
            probe.push(position, first..window.end)?;
        }
        if let Some(pending) = self.groups.get_mut(&group) {
            pending.examined = window;
        }
        Ok(())
    }

    pub(super) fn seal(&mut self, state: &RunState) {
        if !self.groups.is_empty() {
            self.left_cut = Some((
                state.rows(JoinSide::Left),
                state.next_row_id(JoinSide::Left),
            ));
        }
    }

    pub(super) fn validate(&self, state: &RunState) -> Result<()> {
        if self.left_cut.is_some_and(|cut| {
            cut != (
                state.rows(JoinSide::Left),
                state.next_row_id(JoinSide::Left),
            )
        }) {
            return Err(super::run_error(
                "prepared left witness positions are stale",
            ));
        }
        Ok(())
    }

    pub(super) fn commit(self, state: &mut RunState) {
        let left = JoinSide::Left.index();
        for (group, selected) in self.groups {
            let bucket = state.sides[left]
                .buckets
                .get_mut(&group)
                .expect("prepared left bucket");
            for index in selected.positions {
                bucket.entries[index].locator.witnessed = true;
            }
            state.dirty.groups[left].insert(group);
        }
    }
}

pub(super) fn retain_unwitnessed(incoming: &mut Incoming, probe: &Probe) {
    if probe.spans.is_empty() {
        return;
    }
    let mut witnessed = probe.spans.iter().peekable();
    let retained = BooleanBuffer::collect_bool(incoming.len(), |row| {
        let has_witness = witnessed.next_if(|span| span.position == row).is_some();
        !has_witness && incoming.retained(row)
    });
    incoming.retain = BooleanArray::new(retained, None);
}

#[cfg(test)]
mod tests {
    use super::super::{LocatorEntry, RowLocator, RunId};
    use super::*;
    use crate::EventTime;

    #[test]
    fn test_already_witnessed_history_checks_cancellation_without_pending_storage() {
        let group = GroupId::new(0).unwrap();
        let rows = 3 * MORSEL_ROWS;
        let entries = (0..rows)
            .map(|row| LocatorEntry {
                time: EventTime::from_micros(0),
                row_id: u64::try_from(row).unwrap(),
                group,
                locator: RowLocator {
                    run: RunId(u64::try_from(row / MORSEL_ROWS).unwrap()),
                    row: u16::try_from(row % MORSEL_ROWS).unwrap(),
                    witnessed: true,
                },
            })
            .collect();
        let bucket = Bucket { entries, head: 0 };
        let mut witnesses = Witnesses::default();
        let mut probe = Probe::default();
        let mut checks = 0;
        let measured = allocation_counter::measure(|| {
            witnesses
                .select(&mut probe, 0, (group, &bucket), 0..rows, 1, &mut || {
                    checks += 1;
                    Ok(())
                })
                .unwrap();
        });
        assert_eq!(checks, 3);
        assert_eq!(probe.count, 0);
        assert!(witnesses.groups.is_empty());
        assert_eq!(measured.count_total, 0);
        checks = 0;
        let error = witnesses
            .select(&mut probe, 0, (group, &bucket), 0..rows, 1, &mut || {
                checks += 1;
                if checks == 2 {
                    Err(super::super::run_error("cancelled witness scan"))
                } else {
                    Ok(())
                }
            })
            .unwrap_err();
        assert_eq!(checks, 2);
        assert!(error.to_string().contains("cancelled witness scan"));
    }
}
