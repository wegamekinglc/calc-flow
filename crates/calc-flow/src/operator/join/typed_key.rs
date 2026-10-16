use std::hash::{BuildHasher, Hasher};

use super::{
    borrowed_key::{CanonicalStream, KeyHashState},
    columnar::{COPY_BYTES, COPY_VISITS},
    key_arena::KeyWriter,
};
use crate::Result;

#[cfg(test)]
#[path = "tests/typed_key_tests.rs"]
mod tests;

#[cfg(test)]
#[derive(Clone, Copy, Eq, PartialEq)]
pub(super) enum KeyPhase {
    Hash,
    Comparison,
}
#[cfg(test)]
type WorkHook = std::sync::Arc<dyn Fn(KeyPhase, usize) + Send + Sync>;
#[cfg(test)]
thread_local! { static WORK_HOOK: std::cell::RefCell<Option<WorkHook>> = const { std::cell::RefCell::new(None) }; }
#[cfg(test)]
pub(super) struct Observation(Option<WorkHook>);
#[cfg(test)]
impl Drop for Observation {
    fn drop(&mut self) {
        WORK_HOOK.with(|hook| hook.replace(self.0.take()));
    }
}
#[cfg(test)]
pub(super) fn observe(hook: WorkHook) -> Observation {
    Observation(WORK_HOOK.with(|current| current.replace(Some(hook))))
}
#[cfg(test)]
pub(super) fn note(phase: KeyPhase, bytes: usize) {
    WORK_HOOK.with(|hook| {
        if let Some(hook) = &*hook.borrow() {
            hook(phase, bytes);
        }
    });
}

#[derive(Clone, Copy)]
pub(super) struct TypedKey<'a, 'b> {
    pub(super) columns: &'a [KeyWriter<'b>],
    pub(super) row: usize,
}

/// Synchronous identity for a row whose whole framed key fits one copy quantum.
pub(super) struct SmallKey<'a, 'b> {
    columns: &'a [KeyWriter<'b>],
    row: usize,
    bytes: usize,
}

impl<'a, 'b> TypedKey<'a, 'b> {
    /// why: small keys cooperate once per row instead of once per column/chunk;
    /// rows that cannot be charged in one step keep the chunked path
    pub(super) fn small(self) -> Option<SmallKey<'a, 'b>> {
        let mut bytes = 0_usize;
        for column in self.columns {
            for part in column.parts(self.row).ok()?.slices() {
                bytes = bytes.checked_add(part.len())?;
            }
        }
        (bytes <= COPY_BYTES && self.columns.len() < COPY_VISITS).then_some(SmallKey {
            columns: self.columns,
            row: self.row,
            bytes,
        })
    }
}

impl SmallKey<'_, '_> {
    /// One cooperative charge covering the column visits, the framed key bytes
    /// and the first candidate comparison of the row.
    pub(super) fn charge(&self) -> (usize, usize) {
        (self.columns.len() + 1, self.bytes)
    }

    pub(super) fn hash(&self, state: &KeyHashState) -> Result<u64> {
        let mut hasher = state.build_hasher();
        let mut length = 0_u64;
        let mut stream = CanonicalStream::new(|bytes: &[u8]| {
            hasher.write(bytes);
        });
        for column in self.columns {
            for part in column.parts(self.row)?.slices() {
                length = next_length(length, part.len())?;
                stream.write(part);
            }
        }
        stream.finish();
        hasher.write_u64(length);
        Ok(hasher.finish())
    }

    pub(super) fn equals(&self, bytes: &[u8]) -> Result<bool> {
        let mut remaining = bytes;
        for column in self.columns {
            for part in column.parts(self.row)?.slices() {
                if !remaining.starts_with(part) {
                    return Ok(false);
                }
                remaining = &remaining[part.len()..];
            }
        }
        Ok(remaining.is_empty())
    }
}

pub(super) struct KeyFlow<'c, 'j, C = fn() -> Result<()>> {
    context: Option<&'c crate::StreamOperatorContext<'j>>,
    quantum: super::columnar::Quantum,
    check: C,
}
impl<'c, 'j> KeyFlow<'c, 'j> {
    pub(super) fn new(context: Option<&'c crate::StreamOperatorContext<'j>>) -> Self {
        Self {
            context,
            quantum: super::columnar::Quantum::default(),
            check: || Ok(()),
        }
    }
}
impl<C: Fn() -> Result<()>> KeyFlow<'_, '_, C> {
    pub(super) fn checked(check: C) -> Self {
        Self {
            context: None,
            quantum: super::columnar::Quantum::default(),
            check,
        }
    }
    pub(super) async fn step(&mut self, visits: usize, bytes: usize) -> Result<()> {
        (self.check)()?;
        if let Some(context) = self.context {
            self.quantum.step(context, visits, bytes).await?;
        }
        Ok(())
    }
}

impl TypedKey<'_, '_> {
    pub(super) async fn hash(
        self,
        state: &KeyHashState,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<u64> {
        let mut hasher = state.build_hasher();
        let mut length = 0_u64;
        let mut stream = CanonicalStream::new(|bytes: &[u8]| {
            hasher.write(bytes);
        });
        for column in self.columns {
            flow.step(1, 0).await?;
            for part in column.parts(self.row)?.slices() {
                for chunk in part.chunks(4096) {
                    flow.step(0, chunk.len()).await?;
                    #[cfg(test)]
                    note(KeyPhase::Hash, chunk.len());
                    length = next_length(length, chunk.len())?;
                    stream.write(chunk);
                }
            }
        }
        stream.finish();
        hasher.write_u64(length);
        Ok(hasher.finish())
    }

    pub(super) async fn equals(self, bytes: &[u8], flow: &mut KeyFlow<'_, '_>) -> Result<bool> {
        if let Some(small) = self.small() {
            return small.equals(bytes);
        }
        let mut remaining = bytes;
        for column in self.columns {
            flow.step(1, 0).await?;
            for part in column.parts(self.row)?.slices() {
                for chunk in part.chunks(4096) {
                    flow.step(0, chunk.len()).await?;
                    if !remaining.starts_with(chunk) {
                        return Ok(false);
                    }
                    remaining = &remaining[chunk.len()..];
                }
            }
        }
        Ok(remaining.is_empty())
    }

    pub(super) async fn length(self, flow: &mut KeyFlow<'_, '_>) -> Result<usize> {
        let mut length = 0_usize;
        for column in self.columns {
            flow.step(1, 0).await?;
            for part in column.parts(self.row)?.slices() {
                length = length
                    .checked_add(part.len())
                    .ok_or_else(|| super::charge_overflow("key value length"))?;
            }
        }
        Ok(length)
    }

    pub(super) async fn encode(
        self,
        capacity: usize,
        flow: &mut KeyFlow<'_, '_>,
    ) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(capacity);
        for column in self.columns {
            flow.step(1, 0).await?;
            for part in column.parts(self.row)?.slices() {
                for chunk in part.chunks(4096) {
                    flow.step(0, chunk.len()).await?;
                    bytes.extend_from_slice(chunk);
                }
            }
        }
        #[cfg(test)]
        super::note_join_work(|work| work.key_encodings += 1);
        Ok(bytes)
    }
}

fn next_length(current: u64, bytes: usize) -> Result<u64> {
    current
        .checked_add(u64::try_from(bytes).map_err(|_| super::charge_overflow("key bytes"))?)
        .ok_or_else(|| super::charge_overflow("key bytes"))
}
