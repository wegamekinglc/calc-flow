use std::hash::{BuildHasher, Hasher};
use std::sync::Arc;

use super::{Dictionary, FramedKey, KeyFlow, KeyHashState};
use crate::Result;

pub(super) async fn hash_bytes<C: Fn() -> Result<()>>(
    state: &KeyHashState,
    bytes: &[u8],
    flow: &mut KeyFlow<'_, '_, C>,
) -> Result<u64> {
    let mut hasher = state.build_hasher();
    for chunk in bytes.chunks(4096) {
        flow.step(0, chunk.len()).await?;
        #[cfg(test)]
        super::super::typed_key::note(super::super::typed_key::KeyPhase::Hash, chunk.len());
        for block in chunk.chunks(64) {
            hasher.write(block);
        }
    }
    hasher.write_u64(
        u64::try_from(bytes.len()).map_err(|_| super::super::charge_overflow("key bytes"))?,
    );
    Ok(hasher.finish())
}

async fn equal_bytes<C: Fn() -> Result<()>>(
    left: &[u8],
    right: &[u8],
    flow: &mut KeyFlow<'_, '_, C>,
) -> Result<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (left, right) in left.chunks(4096).zip(right.chunks(4096)) {
        flow.step(0, left.len()).await?;
        if left != right {
            return Ok(false);
        }
    }
    Ok(true)
}

impl Dictionary {
    pub(super) async fn find_bytes<C: Fn() -> Result<()>>(
        &self,
        bytes: &[u8],
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<(u64, Option<u32>)> {
        let hash = hash_bytes(&self.hasher, bytes, flow).await?;
        #[cfg(test)]
        let hash = if self.force_collisions { 0 } else { hash };
        let mut candidate = self
            .entries
            .find(hash, |head| head.hash == hash)
            .map(|head| head.id);
        while let Some(id) = candidate {
            flow.step(1, 0).await?;
            #[cfg(test)]
            super::super::typed_key::note(super::super::typed_key::KeyPhase::Comparison, 0);
            let entry = self.slots[id as usize].as_ref().unwrap();
            if equal_bytes(entry.lease.owner(), bytes, flow).await? {
                return Ok((hash, Some(id)));
            }
            candidate = entry.next;
        }
        Ok((hash, None))
    }

    pub(in crate::operator::join) async fn canonical_bytes<C: Fn() -> Result<()>>(
        &self,
        bytes: &[u8],
        flow: &mut KeyFlow<'_, '_, C>,
    ) -> Result<Option<Arc<FramedKey>>> {
        let (_, id) = self.find_bytes(bytes, flow).await?;
        Ok(id.map(|id| Arc::clone(self.slots[id as usize].as_ref().unwrap().lease.owner())))
    }
}
