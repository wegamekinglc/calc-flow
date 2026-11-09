use super::{Result, add, invalid, size};
use std::{alloc::Layout, sync::atomic::AtomicUsize};

#[derive(Clone, Copy, Default)]
pub(super) struct Cost {
    pub owned: u64,
    pub temporary: u64,
}
impl Cost {
    pub fn owned(bytes: u64) -> Self {
        Self {
            owned: bytes,
            temporary: 0,
        }
    }
    pub fn combine(self, other: Self) -> Result<Self> {
        Ok(Self {
            owned: add(self.owned, other.owned)?,
            temporary: add(self.temporary, other.temporary)?,
        })
    }
    pub fn sequence(self, other: Self) -> Result<Self> {
        Ok(Self {
            owned: add(self.owned, other.owned)?,
            temporary: self.temporary.max(other.temporary),
        })
    }
    pub fn extra(self, bytes: u64) -> Result<Self> {
        Ok(Self {
            temporary: add(self.temporary, bytes)?,
            ..self
        })
    }
}
pub(super) fn times<T>(count: usize) -> Result<u64> {
    let bytes = count.checked_mul(size_of::<T>()).ok_or_else(invalid)?;
    size(bytes)
}
pub(super) fn product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(invalid)
}
pub(super) fn arc<T>() -> Result<u64> {
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(Layout::new::<T>())
        .map_err(|_| invalid())?;
    size(layout.pad_to_align().size())
}
pub(super) fn arc_slice<T>(len: usize) -> Result<u64> {
    let elements = Layout::array::<T>(len).map_err(|_| invalid())?;
    let (layout, _) = Layout::new::<[AtomicUsize; 2]>()
        .extend(elements)
        .map_err(|_| invalid())?;
    size(layout.pad_to_align().size())
}
pub(super) fn bits(len: usize) -> usize {
    len / 8 + usize::from(!len.is_multiple_of(8))
}
pub(super) fn buffer(len: usize) -> Result<u64> {
    Layout::from_size_align(len, datafusion::arrow::alloc::ALIGNMENT).map_err(|_| invalid())?;
    // arrow-buffer 59.3.2 Bytes: pointer + length + three-word Deallocation, pool off.
    add(size(len)?, times::<usize>(7)?)
}
