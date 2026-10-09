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
use std::{
    collections::hash_map::RandomState,
    hash::{BuildHasher, Hasher},
};

#[derive(Clone, Copy)]
pub(super) struct BorrowedKey<'a> {
    pub(super) columns: &'a [ArrayRef],
    pub(super) row: usize,
    pub(super) indices: &'a [usize],
}

impl BorrowedKey<'_> {
    pub(super) fn hash(&self, state: &RandomState) -> Result<u64> {
        let mut hasher = state.build_hasher();
        self.visit(|bytes| hasher.write(bytes))?;
        Ok(hasher.finish())
    }

    pub(super) fn equals(&self, bytes: &[u8]) -> bool {
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

    fn visit(&self, mut visitor: impl FnMut(&[u8])) -> Result<()> {
        for &index in self.indices {
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

pub(super) fn framed_hash(state: &RandomState, bytes: &[u8]) -> u64 {
    let mut hasher = state.build_hasher();
    hasher.write(bytes);
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
