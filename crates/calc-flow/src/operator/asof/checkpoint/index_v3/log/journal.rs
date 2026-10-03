use super::super::{BatchKey, Encoding};
use crate::{
    Result,
    operator::asof::{checked, state::LeftOrder},
};
use datafusion::execution::memory_pool::MemoryReservation;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(in crate::operator::asof) enum Identity {
    Left(BatchKey),
    Right(LeftOrder),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(in crate::operator::asof) enum Version {
    Left {
        rows: u64,
        capacities: [usize; 6],
    },
    Right {
        tag: u8,
        payload: Option<(BatchKey, u32)>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(in crate::operator::asof) struct Change {
    pub identity: Identity,
    pub before: Option<Version>,
    pub after: Option<Version>,
}

impl Change {
    fn encodings(&self) -> impl Iterator<Item = &Encoding> {
        let encodings = match &self.identity {
            Identity::Left(_) => [None, None],
            Identity::Right((_, key, sequence)) => [Some(key), Some(sequence)],
        };
        encodings.into_iter().flatten()
    }

    pub(super) fn validate(&self) -> Result<()> {
        for version in [self.before, self.after].into_iter().flatten() {
            let valid = match (&self.identity, version) {
                (Identity::Left((0, _)), Version::Left { rows, .. }) => rows > 0,
                (
                    Identity::Right(_),
                    Version::Right {
                        tag: 1,
                        payload: Some(((1, _), _)),
                    }
                    | Version::Right {
                        tag: 0 | 2,
                        payload: None,
                    },
                ) => true,
                _ => false,
            };
            if !valid {
                return Err(super::super::mismatch(
                    "ASOF journal identity and version differ",
                ));
            }
        }
        if self.before == self.after {
            return Err(super::super::mismatch("ASOF journal change is empty"));
        }
        Ok(())
    }
}

#[derive(Default)]
pub(in crate::operator::asof) struct Journal {
    changes: Vec<Change>,
    keys: Vec<Encoding>,
    lease: Option<MemoryReservation>,
}

impl Journal {
    pub fn changes(&self) -> &[Change] {
        &self.changes
    }

    pub fn keys(&self) -> &[Encoding] {
        &self.keys
    }

    pub fn bytes(&self) -> u64 {
        self.lease.as_ref().map_or(0, |lease| lease.size() as u64)
    }

    pub fn is_empty(&self) -> bool {
        self.changes.is_empty() && self.keys.is_empty()
    }

    pub fn prepare(
        &self,
        edits: &[Change],
        live_allocation_after_commit: impl Fn(usize) -> bool,
        mut reserve: impl FnMut(u64) -> Result<MemoryReservation>,
        name: &str,
    ) -> Result<Self> {
        let count = self
            .changes
            .len()
            .checked_add(edits.len())
            .ok_or_else(|| super::super::mismatch("ASOF journal count overflowed"))?;
        let scratch_bytes = (count as u64)
            .checked_mul((size_of::<Change>() + 512) as u64)
            .and_then(|bytes| bytes.checked_add(self.keys.len() as u64 * 512 + 4096))
            .ok_or_else(|| super::super::mismatch("ASOF journal workspace overflowed"))?;
        let _scratch = reserve(scratch_bytes)?;
        let mut pending = self
            .changes
            .iter()
            .cloned()
            .map(|change| (change.identity.clone(), change))
            .collect::<BTreeMap<_, _>>();
        for edit in edits {
            edit.validate()?;
            if let Some(previous) = pending.get_mut(&edit.identity) {
                if previous.after != edit.before {
                    return Err(super::super::mismatch(
                        "ASOF journal predecessor version differs",
                    ));
                }
                if previous.before == edit.after {
                    pending.remove(&edit.identity);
                } else {
                    previous.after = edit.after;
                }
            } else {
                pending.insert(edit.identity.clone(), edit.clone());
            }
        }
        let mut keys = self
            .keys
            .iter()
            .cloned()
            .map(|key| (key.clone(), key))
            .collect::<BTreeMap<_, _>>();
        for edit in edits {
            if let Identity::Right((_, key, _)) = &edit.identity {
                keys.entry(key.clone()).or_insert_with(|| key.clone());
            }
        }
        let mut dirty_keys = Vec::with_capacity(keys.len());
        dirty_keys.extend(keys.into_values());
        let mut changes = Vec::with_capacity(pending.len());
        changes.extend(pending.into_values());
        if changes.is_empty() && dirty_keys.is_empty() {
            return Ok(Self::default());
        }
        let mut retired = BTreeMap::new();
        for encoding in changes
            .iter()
            .flat_map(Change::encodings)
            .chain(dirty_keys.iter())
        {
            if let Some((address, bytes)) = encoding.allocation()
                && !live_allocation_after_commit(address)
            {
                retired.entry(address).or_insert(bytes);
            }
        }
        let bytes = changes.capacity() as u64 * size_of::<Change>() as u64
            + dirty_keys.capacity() as u64 * size_of::<Encoding>() as u64;
        let bytes = retired
            .values()
            .try_fold(bytes, |bytes, retired| checked(name, bytes, *retired))?;
        let lease = reserve(bytes)?;
        Ok(Self {
            changes,
            keys: dirty_keys,
            lease: Some(lease),
        })
    }

    pub fn install(&mut self, prepared: Self) {
        *self = prepared;
    }
}
