use super::{Candidate, CandidateMap, Result, checked_bytes, df_error};
use ahash::RandomState;
use hashbrown::HashMap;
use std::{alloc::Layout, mem::size_of};

pub(super) enum CandidateGroups {
    Generic(CandidateMap),
    Ranked(RankedCandidates),
}

pub(super) struct RankedCandidates {
    pub(super) values: Vec<Candidate>,
    ranks: HashMap<usize, usize, RandomState>,
    slots: Vec<usize>,
}

impl CandidateGroups {
    pub(super) fn new(ranked: bool) -> Self {
        if ranked {
            Self::Ranked(RankedCandidates {
                values: Vec::new(),
                ranks: HashMap::with_hasher(RandomState::new()),
                slots: Vec::new(),
            })
        } else {
            Self::Generic(CandidateMap::with_hasher(RandomState::new()))
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::Generic(groups) => groups.len(),
            Self::Ranked(groups) => groups.values.len(),
        }
    }

    pub(super) fn get(&self, slot: usize) -> Option<&Candidate> {
        match self {
            Self::Generic(groups) => groups.get(&slot),
            Self::Ranked(groups) => groups.ranks.get(&slot).map(|&rank| &groups.values[rank]),
        }
    }

    pub(super) fn get_mut(&mut self, slot: usize) -> Option<&mut Candidate> {
        match self {
            Self::Generic(groups) => groups.get_mut(&slot),
            Self::Ranked(groups) => groups
                .ranks
                .get(&slot)
                .map(|&rank| &mut groups.values[rank]),
        }
    }

    pub(super) fn generic_mut(&mut self) -> &mut CandidateMap {
        match self {
            Self::Generic(groups) => groups,
            Self::Ranked(_) => unreachable!("generic aggregate candidate storage"),
        }
    }

    pub(super) fn insert(&mut self, slot: usize, candidate: Candidate, name: &str) -> Result<()> {
        match self {
            Self::Generic(groups) => {
                groups.insert(slot, candidate);
            }
            Self::Ranked(groups) => groups.insert(slot, candidate, name)?,
        }
        Ok(())
    }

    pub(super) fn take_slots(&mut self, slots: Vec<usize>) {
        if let Self::Ranked(groups) = self
            && !groups.values.is_empty()
        {
            debug_assert_eq!(slots.len(), groups.values.len());
            groups.slots = slots;
        }
    }

    #[cfg(test)]
    pub(super) fn ranked_layout(&self) -> (usize, usize, usize, usize) {
        let Self::Ranked(groups) = self else {
            panic!("ranked candidate storage");
        };
        (
            groups.ranks.capacity(),
            groups.ranks.allocation_size(),
            groups.values.capacity(),
            groups.slots.capacity(),
        )
    }

    pub(super) fn values(&self) -> impl Iterator<Item = &Candidate> {
        let (generic, ranked) = match self {
            Self::Generic(groups) => (Some(groups), None),
            Self::Ranked(groups) => (None, Some(&groups.values)),
        };
        generic
            .into_iter()
            .flat_map(|groups| groups.values())
            .chain(ranked.into_iter().flatten())
    }

    pub(super) fn values_mut(&mut self) -> impl Iterator<Item = &mut Candidate> {
        let (generic, ranked) = match self {
            Self::Generic(groups) => (Some(groups), None),
            Self::Ranked(groups) => (None, Some(&mut groups.values)),
        };
        generic
            .into_iter()
            .flat_map(|groups| groups.values_mut())
            .chain(ranked.into_iter().flatten())
    }

    pub(super) fn iter(&self) -> impl Iterator<Item = (usize, &Candidate)> {
        let (generic, ranked) = match self {
            Self::Generic(groups) => (Some(groups), None),
            Self::Ranked(groups) => (None, Some(groups)),
        };
        generic
            .into_iter()
            .flat_map(|groups| groups.iter().map(|(&slot, candidate)| (slot, candidate)))
            .chain(
                ranked
                    .into_iter()
                    .flat_map(|groups| groups.slots.iter().copied().zip(&groups.values)),
            )
    }

    pub(super) fn into_iter(self) -> impl Iterator<Item = (usize, Candidate)> {
        let (generic, ranked) = match self {
            Self::Generic(groups) => (Some(groups), None),
            Self::Ranked(groups) => (None, Some(groups)),
        };
        generic.into_iter().flatten().chain(
            ranked
                .into_iter()
                .flat_map(|groups| groups.slots.into_iter().zip(groups.values)),
        )
    }
}

impl RankedCandidates {
    fn insert(&mut self, slot: usize, candidate: Candidate, name: &str) -> Result<()> {
        let rank = self.values.len();
        let count = rank
            .checked_add(1)
            .ok_or_else(|| df_error(name, "candidate count overflowed"))?;
        self.reserve(count, name)?;
        self.ranks.insert(slot, rank);
        self.values.push(candidate);
        Ok(())
    }

    fn reserve(&mut self, count: usize, name: &str) -> Result<()> {
        self.check_growth(count, name)?;
        self.ranks
            .try_reserve(1)
            .map_err(|error| df_error(name, error))?;
        self.values
            .try_reserve_exact(self.ranks.capacity() - self.values.len())
            .map_err(|error| df_error(name, error))?;
        self.check_capacity(count, name)
    }

    fn check_growth(&self, count: usize, name: &str) -> Result<()> {
        // The existing candidate container credit also owns transferred partial slots.
        let peak = growth_peak(count, self.ranks.capacity(), name)?;
        let paid = checked_bytes(
            0,
            [(
                count,
                4 * size_of::<(usize, Candidate)>() + size_of::<[usize; 8]>(),
            )],
            name,
        )?;
        if peak > paid {
            return Err(df_error(
                name,
                "ranked candidate growth exceeds container credit",
            ));
        }
        Ok(())
    }

    fn check_capacity(&self, count: usize, name: &str) -> Result<()> {
        let (_, expected) = table_capacity(count, name)?;
        if self.ranks.capacity() != expected || self.values.capacity() != expected {
            return Err(df_error(
                name,
                "ranked candidate capacity differs from prepaid layout",
            ));
        }
        Ok(())
    }
}

fn table_capacity(count: usize, name: &str) -> Result<(usize, usize)> {
    let buckets = match count {
        0 => return Ok((0, 0)),
        1..=3 => 4,
        4..=7 => 8,
        8..=14 => 16,
        _ => count
            .checked_mul(8)
            .and_then(|count| (count / 7).checked_next_power_of_two())
            .ok_or_else(|| df_error(name, "candidate table capacity overflowed"))?,
    };
    let capacity = if buckets < 8 {
        buckets - 1
    } else {
        buckets / 8 * 7
    };
    Ok((buckets, capacity))
}

fn table_bytes(capacity: usize, name: &str) -> Result<usize> {
    if capacity == 0 {
        return Ok(0);
    }
    let (buckets, _) = table_capacity(capacity, name)?;
    // hashbrown 0.17.1 TableLayout: at most 16-byte control alignment and tail.
    let bytes = checked_bytes(31, [(buckets, size_of::<(usize, usize)>() + 1)], name)?;
    Layout::from_size_align(bytes, 16).map_err(|error| df_error(name, error))?;
    Ok(bytes)
}

pub(super) fn growth_peak(count: usize, old_capacity: usize, name: &str) -> Result<usize> {
    let (_, capacity) = table_capacity(count, name)?;
    let old = if count > old_capacity {
        old_capacity
    } else {
        0
    };
    Layout::array::<Candidate>(capacity).map_err(|error| df_error(name, error))?;
    let slots = count
        .max(4)
        .checked_next_power_of_two()
        .ok_or_else(|| df_error(name, "candidate slot capacity overflowed"))?;
    checked_bytes(
        table_bytes(capacity, name)?,
        [
            (table_bytes(old, name)?, 1),
            (capacity, size_of::<Candidate>()),
            (old, size_of::<Candidate>()),
            (slots, size_of::<usize>()),
        ],
        name,
    )
}
