use std::ops::Range;

use super::super::columnar::Quantum;
use crate::{Result, StreamOperatorContext};

pub(super) enum Values<T, const N: usize> {
    Inline { values: [T; N], len: usize },
    Heap(Vec<T>),
}

impl<T: Copy + Default, const N: usize> Values<T, N> {
    pub(super) fn new(capacity: usize, inline: bool) -> Self {
        if inline {
            Self::Inline {
                values: [T::default(); N],
                len: 0,
            }
        } else {
            Self::Heap(Vec::with_capacity(capacity))
        }
    }
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Inline { len, .. } => *len,
            Self::Heap(values) => values.len(),
        }
    }
    pub(super) fn push(&mut self, value: T) {
        match self {
            Self::Inline { values, len } => {
                values[*len] = value;
                *len += 1;
            }
            Self::Heap(values) => {
                assert!(
                    values.len() < values.capacity(),
                    "prepaid admission capacity"
                );
                values.push(value);
            }
        }
    }
    pub(super) fn get(&self, index: usize) -> T {
        match self {
            Self::Inline { values, len } => {
                assert!(index < *len);
                values[index]
            }
            Self::Heap(values) => values[index],
        }
    }
    fn set(&mut self, index: usize, value: T) {
        match self {
            Self::Inline { values, len } => {
                assert!(index < *len);
                values[index] = value;
            }
            Self::Heap(values) => values[index] = value,
        }
    }
    fn truncate(&mut self, length: usize) {
        match self {
            Self::Inline { len, .. } => {
                assert!(length <= *len);
                *len = length;
            }
            Self::Heap(values) => values.truncate(length),
        }
    }
    #[cfg(test)]
    fn capacity(&self) -> usize {
        match self {
            Self::Inline { values, .. } => values.len(),
            Self::Heap(values) => values.capacity(),
        }
    }
    #[cfg(test)]
    fn allocated_bytes(&self) -> usize {
        self.capacity() * size_of::<T>()
    }
}

pub(super) struct RetainBits {
    words: Values<u64, 1>,
    len: usize,
}
impl RetainBits {
    pub(super) fn new(rows: usize, inline: bool) -> Self {
        Self {
            words: Values::new(rows.div_ceil(64), inline),
            len: 0,
        }
    }
    pub(super) fn push(&mut self, retained: bool) {
        let offset = self.len % 64;
        if offset == 0 {
            self.words.push(0);
        }
        if retained {
            let index = self.len / 64;
            self.words
                .set(index, self.words.get(index) | (1_u64 << offset));
        }
        self.len += 1;
    }
    pub(super) fn get(&self, index: usize) -> bool {
        assert!(index < self.len);
        self.words.get(index / 64) & (1_u64 << (index % 64)) != 0
    }
    pub(super) fn any(&self, range: Range<usize>) -> bool {
        if range.is_empty() {
            return false;
        }
        let first = range.start / 64;
        let last = (range.end - 1) / 64;
        let first_mask = u64::MAX << (range.start % 64);
        let last_mask = u64::MAX >> (63 - ((range.end - 1) % 64));
        if first == last {
            return self.words.get(first) & first_mask & last_mask != 0;
        }
        if self.words.get(first) & first_mask != 0 || self.words.get(last) & last_mask != 0 {
            return true;
        }
        (first + 1..last).any(|index| self.words.get(index) != 0)
    }
    #[cfg(test)]
    pub(super) fn allocated_bytes(&self) -> usize {
        self.words.allocated_bytes()
    }
}

enum IndexValues {
    Inline(Values<u8, 64>),
    Heap(Values<usize, 0>),
}
impl IndexValues {
    fn new(rows: usize, inline: bool) -> Self {
        if inline {
            Self::Inline(Values::new(rows, true))
        } else {
            Self::Heap(Values::new(rows, false))
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::Inline(values) => values.len(),
            Self::Heap(values) => values.len(),
        }
    }
    fn get(&self, index: usize) -> usize {
        match self {
            Self::Inline(values) => usize::from(values.get(index)),
            Self::Heap(values) => values.get(index),
        }
    }
    fn push(&mut self, value: usize) {
        match self {
            Self::Inline(values) => {
                values.push(u8::try_from(value).expect("inline physical position"));
            }
            Self::Heap(values) => values.push(value),
        }
    }
    fn truncate(&mut self, len: usize) {
        match self {
            Self::Inline(values) => values.truncate(len),
            Self::Heap(values) => values.truncate(len),
        }
    }
    #[cfg(test)]
    fn capacity(&self) -> usize {
        match self {
            Self::Inline(values) => values.capacity(),
            Self::Heap(values) => values.capacity(),
        }
    }
}

pub(super) struct SparseIndices {
    values: Option<IndexValues>,
    rows: usize,
    inline: bool,
}
impl SparseIndices {
    pub(super) fn new(rows: usize, inline: bool) -> Self {
        Self {
            values: None,
            rows,
            inline,
        }
    }
    pub(super) fn get(&self, index: usize) -> usize {
        self.values
            .as_ref()
            .expect("installed sparse selection")
            .get(index)
    }
    pub(super) fn push(&mut self, value: usize) {
        self.values
            .as_mut()
            .expect("installed sparse selection")
            .push(value);
        #[cfg(test)]
        super::super::note_join_work(|work| work.sparse_index_writes += 1);
    }
    pub(super) async fn promote(
        &mut self,
        first: usize,
        count: usize,
        context: &StreamOperatorContext<'_>,
        quantum: &mut Quantum,
    ) -> Result<usize> {
        context.check_cancelled()?;
        let values = self
            .values
            .get_or_insert_with(|| IndexValues::new(self.rows, self.inline));
        let mut promotion = Promotion::new(values);
        for start in (0..count).step_by(64) {
            let length = (count - start).min(64);
            quantum
                .step(context, length, length * size_of::<usize>())
                .await?;
            for offset in start..start + length {
                promotion.push(first + offset);
            }
            #[cfg(test)]
            note_promotion(length);
        }
        Ok(promotion.finish())
    }
    #[cfg(test)]
    pub(super) fn capacity(&self) -> usize {
        self.values.as_ref().map_or(0, IndexValues::capacity)
    }
    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.values.as_ref().map_or(0, IndexValues::len)
    }
}

struct Promotion<'a> {
    values: &'a mut IndexValues,
    start: usize,
    complete: bool,
}
impl<'a> Promotion<'a> {
    fn new(values: &'a mut IndexValues) -> Self {
        let start = values.len();
        Self {
            values,
            start,
            complete: false,
        }
    }
    fn push(&mut self, value: usize) {
        self.values.push(value);
        #[cfg(test)]
        super::super::note_join_work(|work| work.sparse_index_writes += 1);
    }
    fn finish(mut self) -> usize {
        self.complete = true;
        self.start
    }
}
impl Drop for Promotion<'_> {
    fn drop(&mut self) {
        if !self.complete {
            self.values.truncate(self.start);
        }
    }
}

#[cfg(test)]
fn note_promotion(count: usize) {
    super::super::note_join_work(|work| work.sparse_prefix_promoted += count);
    super::SPARSE_PROMOTION_OBSERVER.with(|observer| {
        if let Some(observer) = observer.borrow().as_ref() {
            observer.notify_one();
        }
    });
}

pub(super) fn metadata_bytes(rows: usize) -> Option<usize> {
    rows.checked_mul(size_of::<usize>() + size_of::<i64>() + size_of::<u64>())?
        .checked_add(rows.div_ceil(64).checked_mul(size_of::<u64>())?)
}
