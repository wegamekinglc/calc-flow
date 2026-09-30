//! The canonical integer column codec shared by native chunks and v3 indexes.

use super::Encoding;
use crate::operator::asof::AsofJoinSide;
use datafusion::arrow::datatypes::{DataType, Schema};
use std::{borrow::Cow, ops::Range};

pub(in super::super) type SequenceRef<'a> = Cow<'a, Encoding>;

#[derive(Clone)]
enum Storage {
    Canonical(Vec<Encoding>),
    Integer(Vec<u8>),
}

/// Integer columns retain their declared width; generic identities borrow
/// one canonical batch allocation through the existing encoding handles.
pub(in super::super) struct SequenceColumn {
    kind: SequenceKind,
    storage: Storage,
}

impl Clone for SequenceColumn {
    fn clone(&self) -> Self {
        let storage = match &self.storage {
            Storage::Canonical(values) => {
                let mut cloned = Vec::with_capacity(values.capacity());
                cloned.extend_from_slice(values);
                Storage::Canonical(cloned)
            }
            Storage::Integer(values) => {
                let mut cloned = Vec::with_capacity(values.capacity());
                cloned.extend_from_slice(values);
                Storage::Integer(cloned)
            }
        };
        Self {
            kind: self.kind,
            storage,
        }
    }
}

impl SequenceColumn {
    pub const fn empty(kind: SequenceKind) -> Self {
        let storage = match kind {
            SequenceKind::Canonical => Storage::Canonical(Vec::new()),
            _ => Storage::Integer(Vec::new()),
        };
        Self { kind, storage }
    }

    pub fn with_capacity(capacity: usize, kind: SequenceKind) -> Self {
        let storage = match kind.width() {
            None => Storage::Canonical(Vec::with_capacity(capacity)),
            Some(width) => Storage::Integer(Vec::with_capacity(capacity * width)),
        };
        Self { kind, storage }
    }

    pub fn kind(&self) -> SequenceKind {
        self.kind
    }

    pub fn element_bytes(&self) -> usize {
        self.kind.width().unwrap_or(size_of::<Encoding>())
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Canonical(values) => values.len(),
            Storage::Integer(values) => values.len() / self.element_bytes(),
        }
    }

    pub fn capacity(&self) -> usize {
        match &self.storage {
            Storage::Canonical(values) => values.capacity(),
            Storage::Integer(values) => values.capacity() / self.element_bytes(),
        }
    }

    pub fn allocation_bytes(&self) -> usize {
        match &self.storage {
            Storage::Canonical(values) => values.capacity() * size_of::<Encoding>(),
            Storage::Integer(values) => values.capacity(),
        }
    }

    pub fn integer_slice(&self, range: Range<usize>) -> Option<&[u8]> {
        let Storage::Integer(values) = &self.storage else {
            return None;
        };
        let width = self.element_bytes();
        values.get(range.start.checked_mul(width)?..range.end.checked_mul(width)?)
    }

    pub fn from_integer_slice(values: &[u8], capacity: usize, kind: SequenceKind) -> Self {
        let width = kind.width().expect("integer sequence column");
        assert!(
            values.len() % width == 0 && values.len() / width <= capacity,
            "preflighted sequence column"
        );
        let mut bytes = Vec::with_capacity(capacity * width);
        bytes.extend_from_slice(values);
        Self {
            kind,
            storage: Storage::Integer(bytes),
        }
    }

    pub fn get(&self, index: usize) -> Option<SequenceRef<'_>> {
        match &self.storage {
            Storage::Canonical(values) => values.get(index).map(Cow::Borrowed),
            Storage::Integer(values) => {
                let width = self.element_bytes();
                let start = index.checked_mul(width)?;
                Some(Cow::Owned(self.kind.decode_integer(
                    values.get(start..start.checked_add(width)?)?,
                )))
            }
        }
    }

    pub fn iter(
        &self,
    ) -> impl ExactSizeIterator<Item = SequenceRef<'_>> + DoubleEndedIterator + Clone {
        (0..self.len()).map(|index| self.get(index).expect("sequence column row"))
    }

    pub fn binary_search(&self, range: Range<usize>, value: &Encoding) -> Result<usize, usize> {
        let mut lower = range.start;
        let mut upper = range.end;
        while lower < upper {
            let middle = lower + (upper - lower) / 2;
            match self
                .get(middle)
                .expect("sequence search row")
                .as_ref()
                .cmp(value)
            {
                std::cmp::Ordering::Less => lower = middle + 1,
                std::cmp::Ordering::Greater => upper = middle,
                std::cmp::Ordering::Equal => return Ok(middle),
            }
        }
        Err(lower)
    }

    pub fn reserve(&mut self, additional: usize) {
        let required = self.len() + additional;
        let current = self.capacity();
        let width = self.element_bytes();
        match &mut self.storage {
            Storage::Canonical(values) => values.reserve(additional),
            Storage::Integer(values) if current < required => {
                let next = required.max(current * 2).max(4);
                // Avoid Vec<u8>'s byte-based minimum and growth: capacities
                // are row counts in both the budget ledger and v3 manifest.
                values.reserve_exact(next * width - values.len());
            }
            Storage::Integer(_) => (),
        }
    }

    pub fn push(&mut self, value: Encoding) {
        self.reserve(1);
        let raw = self
            .kind
            .width()
            .map(|width| (self.kind.integer_bytes(&value), width));
        match &mut self.storage {
            Storage::Canonical(values) => values.push(value),
            Storage::Integer(values) => {
                let (bytes, width) = raw.expect("integer sequence column");
                values.extend_from_slice(&bytes[..width]);
            }
        }
    }

    pub fn insert(&mut self, index: usize, value: Encoding) {
        self.reserve(1);
        let raw = self
            .kind
            .width()
            .map(|width| (self.kind.integer_bytes(&value), width));
        match &mut self.storage {
            Storage::Canonical(values) => values.insert(index, value),
            Storage::Integer(values) => {
                let (bytes, width) = raw.expect("integer sequence column");
                let start = index * width;
                let len = values.len();
                values.resize(len + width, 0);
                values.copy_within(start..len, start + width);
                values[start..start + width].copy_from_slice(&bytes[..width]);
            }
        }
    }

    pub fn remove(&mut self, index: usize) {
        let width = self.element_bytes();
        match &mut self.storage {
            Storage::Canonical(values) => {
                values.remove(index);
            }
            Storage::Integer(values) => {
                let start = index * width;
                values.copy_within(start + width.., start);
                values.truncate(values.len() - width);
            }
        }
    }

    pub fn take(&mut self, index: usize) -> Encoding {
        let width = self.element_bytes();
        match &mut self.storage {
            Storage::Canonical(values) => {
                std::mem::replace(&mut values[index], Encoding::from_slice(&[]))
            }
            Storage::Integer(values) => {
                let bytes = &mut values[index * width..(index + 1) * width];
                let encoding = self.kind.decode_integer(bytes);
                bytes.fill(0);
                encoding
            }
        }
    }

    pub fn compact(&mut self, head: usize) {
        let width = self.element_bytes();
        match &mut self.storage {
            Storage::Canonical(values) => {
                *values = values.split_off(head).into_boxed_slice().into_vec();
            }
            Storage::Integer(values) => {
                *values = values.split_off(head * width).into_boxed_slice().into_vec();
            }
        }
    }

    pub fn suffix(&self, head: usize) -> Self {
        let storage = match &self.storage {
            Storage::Canonical(values) => Storage::Canonical(values[head..].to_vec()),
            Storage::Integer(values) => {
                Storage::Integer(values[head * self.element_bytes()..].to_vec())
            }
        };
        Self {
            kind: self.kind,
            storage,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(in super::super) enum SequenceKind {
    #[default]
    Canonical,
    Signed(u8),
    Unsigned(u8),
}

impl SequenceKind {
    pub fn from_flag(flag: u8) -> Option<Self> {
        match flag {
            0 => Some(Self::Canonical),
            1..=4 => Some(Self::Signed(1 << (flag - 1))),
            5..=8 => Some(Self::Unsigned(1 << (flag - 5))),
            _ => None,
        }
    }
    pub fn for_side(schema: &Schema, side: &AsofJoinSide) -> Self {
        let [name] = side.sequence_by() else {
            return Self::Canonical;
        };
        let data_type = schema
            .field_with_name(name)
            .expect("validated sequence column")
            .data_type();
        match data_type {
            DataType::Int8 => Self::Signed(1),
            DataType::Int16 => Self::Signed(2),
            DataType::Int32 => Self::Signed(4),
            DataType::Int64 => Self::Signed(8),
            DataType::UInt8 => Self::Unsigned(1),
            DataType::UInt16 => Self::Unsigned(2),
            DataType::UInt32 => Self::Unsigned(4),
            DataType::UInt64 => Self::Unsigned(8),
            _ => Self::Canonical,
        }
    }

    pub fn width(self) -> Option<usize> {
        match self {
            Self::Canonical => None,
            Self::Signed(width) | Self::Unsigned(width) => Some(usize::from(width)),
        }
    }

    pub fn flag(self) -> u8 {
        match self {
            Self::Canonical => 0,
            Self::Signed(width) => {
                1 + u8::try_from(width.trailing_zeros()).expect("one of four integer widths")
            }
            Self::Unsigned(width) => {
                5 + u8::try_from(width.trailing_zeros()).expect("one of four integer widths")
            }
        }
    }

    pub fn integer_bytes(self, encoding: &Encoding) -> [u8; 8] {
        let width = self.width().expect("integer sequence column");
        let mut bytes = [0; 8];
        bytes[..width].copy_from_slice(&encoding.as_slice()[1..]);
        if matches!(self, Self::Signed(_)) {
            bytes[0] ^= 0x80;
        }
        bytes[..width].reverse();
        bytes
    }

    pub fn decode_integer(self, bytes: &[u8]) -> Encoding {
        let mut encoded = [0; 9];
        encoded[0] = 1;
        let width = bytes.len();
        for (target, byte) in encoded[1..=width].iter_mut().zip(bytes.iter().rev()) {
            *target = *byte;
        }
        if matches!(self, Self::Signed(_)) {
            encoded[1] ^= 0x80;
        }
        Encoding::from_slice(&encoded[..=width])
    }

    pub fn storage_bytes(self) -> usize {
        self.width().unwrap_or(16)
    }

    pub fn reference_bytes(self) -> u64 {
        self.storage_bytes() as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::operator::asof::state::RightBucket;

    fn kinds() -> impl Iterator<Item = SequenceKind> {
        [1, 2, 4, 8]
            .into_iter()
            .flat_map(|width| [SequenceKind::Signed(width), SequenceKind::Unsigned(width)])
    }

    #[test]
    fn integer_column_preserves_order_extremes_and_mutation() {
        for kind in kinds() {
            let width = kind.width().unwrap();
            let raw = [
                0_u64,
                1,
                u64::MAX,
                1 << (width * 8 - 1),
                (1 << (width * 8 - 1)) - 1,
            ];
            let mut encodings = raw
                .map(|value| kind.decode_integer(&value.to_le_bytes()[..width]))
                .to_vec();
            encodings.sort_unstable();
            encodings.dedup();
            let mut column = SequenceColumn::empty(kind);
            for value in &encodings {
                column.push(value.clone());
            }
            let expected = || encodings.clone();
            let actual =
                |column: &SequenceColumn| column.iter().map(Cow::into_owned).collect::<Vec<_>>();
            assert_eq!(actual(&column), expected());
            for (index, value) in encodings.iter().enumerate() {
                assert_eq!(column.binary_search(0..column.len(), value), Ok(index));
            }
            column.remove(1);
            column.insert(1, encodings[1].clone());
            assert_eq!(actual(&column), expected());
            assert_eq!(column.take(0), encodings[0]);
            let suffix = column.suffix(1);
            column.compact(1);
            assert_eq!(actual(&column), encodings[1..]);
            assert_eq!(actual(&suffix), encodings[1..]);
            assert_eq!(column.allocation_bytes(), column.len() * width);
        }
    }

    #[test]
    fn right_integer_history_allocates_only_the_declared_width() {
        let count = 4_096;
        for kind in kinds() {
            let width = kind.width().unwrap();
            let mut bucket = RightBucket::with_sequence_kind(kind);
            let sequence = kind.decode_integer(&[0; 8][..width]);
            let allocations = allocation_counter::measure(|| {
                for time in 0..count {
                    bucket.insert((time, sequence.clone()), None);
                }
            });
            let columns = count as u64 * (8 + width as u64);
            assert!(
                allocations.bytes_current >= 0 && allocations.bytes_current as u64 <= columns + 128,
                "{kind:?}: {allocations:?}, expected={columns}"
            );
            assert!(bucket.metadata_bytes() >= allocations.bytes_current as u64);
            assert_eq!(bucket.len(), count as usize);
            assert!(
                bucket
                    .iter()
                    .all(|((_, value), row)| value.as_ref() == &sequence && row.is_none())
            );
        }
    }
}
