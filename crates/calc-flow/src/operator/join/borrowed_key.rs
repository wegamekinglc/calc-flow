use super::{charge_overflow, charge_type_mismatch, key_type_tag, primitive};
use crate::Result;
use datafusion::arrow::{
    array::{Array, ArrayRef, BooleanArray, LargeStringArray, StringArray},
    datatypes::{
        DataType, Int16Type, Int32Type, Int64Type, TimeUnit, TimestampMicrosecondType,
        TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type,
        UInt16Type, UInt32Type, UInt64Type,
    },
};
use std::hash::{BuildHasher, Hasher};

pub(super) type KeyHashState = hashbrown::DefaultHashBuilder;

#[cfg(test)]
#[path = "tests/hash_stream_tests.rs"]
mod tests;

#[derive(Clone, Copy)]
pub(super) struct BorrowedKey<'a> {
    pub(super) columns: &'a [ArrayRef],
    pub(super) row: usize,
    pub(super) indices: &'a [usize],
}

impl BorrowedKey<'_> {
    pub(super) fn hash(&self, state: &KeyHashState) -> Result<u64> {
        #[cfg(test)]
        super::note_join_work(|work| work.borrowed_key_hashes += 1);
        let mut hasher = state.build_hasher();
        let mut length = 0_u64;
        self.visit_hash_blocks(|bytes| {
            hasher.write(bytes);
            length = length.wrapping_add(bytes.len() as u64);
        })?;
        hasher.write_u64(length);
        Ok(hasher.finish())
    }

    pub(super) fn equals(&self, bytes: &[u8]) -> bool {
        #[cfg(test)]
        super::note_join_work(|work| work.borrowed_key_equalities += 1);
        let mut remaining = bytes;
        let mut equal = true;
        self.visit(|part| {
            if remaining.starts_with(part) {
                remaining = &remaining[part.len()..];
            } else {
                equal = false;
            }
        })
        .expect("borrowed hash validates immutable typed key values");
        equal && remaining.is_empty()
    }

    fn visit_hash_blocks(&self, visitor: impl FnMut(&[u8])) -> Result<()> {
        let mut stream = CanonicalStream::new(visitor);
        self.visit(|part| stream.write(part))?;
        stream.finish();
        Ok(())
    }

    pub(in crate::operator::join) fn visit(&self, mut visitor: impl FnMut(&[u8])) -> Result<()> {
        for &index in self.indices {
            #[cfg(test)]
            super::note_join_work(|work| work.key_type_resolutions += 1);
            let array = self.columns[index].as_ref();
            let tag = key_type_tag(array.data_type())?;
            let timezone = timezone(array.data_type());
            let value = cell_value(array, self.row)?;
            let bytes = value.bytes();
            let timezone_len = framed_length(timezone.len(), "key timezone length")?;
            let value_len = framed_length(bytes.len(), "key value length")?;
            visitor(&[tag]);
            visitor(&timezone_len);
            visitor(timezone);
            visitor(&value_len);
            visitor(bytes);
        }
        Ok(())
    }
}

struct CanonicalStream<F> {
    block: [u8; 64],
    used: usize,
    visitor: F,
}

impl<F: FnMut(&[u8])> CanonicalStream<F> {
    fn new(visitor: F) -> Self {
        Self {
            block: [0; 64],
            used: 0,
            visitor,
        }
    }

    fn write(&mut self, mut bytes: &[u8]) {
        while !bytes.is_empty() {
            let length = bytes.len().min(self.block.len() - self.used);
            self.block[self.used..self.used + length].copy_from_slice(&bytes[..length]);
            self.used += length;
            bytes = &bytes[length..];
            if self.used == self.block.len() {
                (self.visitor)(&self.block);
                self.used = 0;
            }
        }
    }

    fn finish(mut self) {
        if self.used != 0 {
            (self.visitor)(&self.block[..self.used]);
        }
    }
}

pub(super) fn framed_hash(state: &KeyHashState, bytes: &[u8]) -> u64 {
    let mut hasher = state.build_hasher();
    for block in bytes.chunks(64) {
        hasher.write(block);
    }
    hasher.write_u64(bytes.len() as u64);
    hasher.finish()
}

fn framed_length(length: usize, field: &str) -> Result<[u8; 4]> {
    u32::try_from(length)
        .map(u32::to_le_bytes)
        .map_err(|_| charge_overflow(field))
}

fn timezone(data_type: &DataType) -> &[u8] {
    match data_type {
        DataType::Timestamp(_, Some(timezone)) => timezone.as_bytes(),
        _ => &[],
    }
}

enum CellValue<'a> {
    Inline { bytes: [u8; 8], length: usize },
    Borrowed(&'a [u8]),
}

impl CellValue<'_> {
    fn inline<const N: usize>(value: [u8; N]) -> Self {
        let mut bytes = [0; 8];
        bytes[..N].copy_from_slice(&value);
        Self::Inline { bytes, length: N }
    }

    fn bytes(&self) -> &[u8] {
        match self {
            Self::Inline { bytes, length } => &bytes[..*length],
            Self::Borrowed(bytes) => bytes,
        }
    }
}

fn cell_value(array: &dyn Array, row: usize) -> Result<CellValue<'_>> {
    match array.data_type() {
        DataType::Boolean => boolean_value(array, row),
        DataType::Int16 | DataType::Int32 | DataType::Int64 => signed_value(array, row),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            unsigned_value(array, row)
        }
        DataType::Utf8 => Ok(CellValue::Borrowed(
            array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| charge_type_mismatch("Utf8 key"))?
                .value(row)
                .as_bytes(),
        )),
        DataType::LargeUtf8 => Ok(CellValue::Borrowed(
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| charge_type_mismatch("LargeUtf8 key"))?
                .value(row)
                .as_bytes(),
        )),
        DataType::Timestamp(unit, _) => timestamp_value(array, row, *unit),
        _ => Err(charge_type_mismatch("native Join key")),
    }
}

fn boolean_value(array: &dyn Array, row: usize) -> Result<CellValue<'_>> {
    let typed = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| charge_type_mismatch("Boolean key"))?;
    Ok(CellValue::inline([u8::from(typed.value(row))]))
}

fn signed_value(array: &dyn Array, row: usize) -> Result<CellValue<'_>> {
    Ok(match array.data_type() {
        DataType::Int16 => {
            CellValue::inline(primitive::<Int16Type>(array)?.value(row).to_le_bytes())
        }
        DataType::Int32 => {
            CellValue::inline(primitive::<Int32Type>(array)?.value(row).to_le_bytes())
        }
        DataType::Int64 => {
            CellValue::inline(primitive::<Int64Type>(array)?.value(row).to_le_bytes())
        }
        _ => unreachable!("native signed key type"),
    })
}

fn unsigned_value(array: &dyn Array, row: usize) -> Result<CellValue<'_>> {
    Ok(match array.data_type() {
        DataType::UInt8 => {
            CellValue::inline(primitive::<UInt8Type>(array)?.value(row).to_le_bytes())
        }
        DataType::UInt16 => {
            CellValue::inline(primitive::<UInt16Type>(array)?.value(row).to_le_bytes())
        }
        DataType::UInt32 => {
            CellValue::inline(primitive::<UInt32Type>(array)?.value(row).to_le_bytes())
        }
        DataType::UInt64 => {
            CellValue::inline(primitive::<UInt64Type>(array)?.value(row).to_le_bytes())
        }
        _ => unreachable!("native unsigned key type"),
    })
}

fn timestamp_value(array: &dyn Array, row: usize, unit: TimeUnit) -> Result<CellValue<'_>> {
    Ok(match unit {
        TimeUnit::Second => CellValue::inline(
            primitive::<TimestampSecondType>(array)?
                .value(row)
                .to_le_bytes(),
        ),
        TimeUnit::Millisecond => CellValue::inline(
            primitive::<TimestampMillisecondType>(array)?
                .value(row)
                .to_le_bytes(),
        ),
        TimeUnit::Microsecond => CellValue::inline(
            primitive::<TimestampMicrosecondType>(array)?
                .value(row)
                .to_le_bytes(),
        ),
        TimeUnit::Nanosecond => CellValue::inline(
            primitive::<TimestampNanosecondType>(array)?
                .value(row)
                .to_le_bytes(),
        ),
    })
}
