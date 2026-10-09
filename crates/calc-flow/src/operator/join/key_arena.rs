use super::{AdmittedRow, charge_overflow, key_type_tag, native_dictionary::BASE_BYTES};
use crate::Result;
use datafusion::arrow::{
    array::{Array, BooleanArray, LargeStringArray, PrimitiveArray, StringArray},
    datatypes::{
        ArrowPrimitiveType, DataType, Int16Type, Int32Type, Int64Type, TimeUnit,
        TimestampMicrosecondType, TimestampMillisecondType, TimestampNanosecondType,
        TimestampSecondType, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
};

#[derive(Clone, Copy)]
pub(super) struct ArenaLayout {
    pub(super) charge: usize,
    bytes: usize,
    row_bytes: usize,
}

impl ArenaLayout {
    pub(super) fn measure(rows: &[AdmittedRow], indices: &[usize]) -> Result<Option<Self>> {
        let Some(extra) = rows
            .len()
            .checked_mul(size_of::<u32>() + size_of::<Option<u32>>())
            .and_then(|n| n.checked_add(BASE_BYTES))
        else {
            return Err(super::native_lookup::scratch_error("join"));
        };
        let mut layout = Self {
            charge: extra,
            bytes: 0,
            row_bytes: 0,
        };
        for group in parent_groups(rows) {
            if !layout.measure_group(group, indices)? {
                return Ok(None);
            }
        }
        Ok(Some(layout))
    }

    fn measure_group(&mut self, rows: &[AdmittedRow], indices: &[usize]) -> Result<bool> {
        let mut maximum = 0_usize;
        for &index in indices {
            let Some(column) = KeyColumn::bind(rows[0].record.column(index).as_ref()) else {
                return Ok(false);
            };
            let mut maximum_cell = 0;
            for row in rows {
                if !column.frame_supported(row.record.offset()) {
                    return Ok(false);
                }
                maximum_cell = maximum_cell.max(self.add(&column, row.record.offset())?);
            }
            maximum = maximum
                .checked_add(maximum_cell)
                .ok_or_else(|| super::native_lookup::scratch_error("join"))?;
        }
        self.row_bytes = self.row_bytes.max(maximum);
        Ok(true)
    }

    fn add(&mut self, column: &KeyColumn<'_>, row: usize) -> Result<usize> {
        let value = column.values.len(row);
        let logical = value.checked_add(column.values.prefix() + 1);
        let fee = logical
            .and_then(|n| n.checked_add(column.timezone.len()))
            .and_then(|n| n.checked_add(64))
            .and_then(|n| n.checked_mul(4));
        self.charge = fee
            .and_then(|n| self.charge.checked_add(n))
            .ok_or_else(|| super::native_lookup::scratch_error("join"))?;
        let frame = value
            .checked_add(column.timezone.len())
            .and_then(|n| n.checked_add(9))
            .ok_or_else(|| super::native_lookup::scratch_error("join"))?;
        self.bytes = self
            .bytes
            .checked_add(frame)
            .ok_or_else(|| super::native_lookup::scratch_error("join"))?;
        Ok(frame)
    }

    pub(super) fn fits(self, rows: usize, columns: usize) -> bool {
        self.peak(rows, columns)
            .is_some_and(|peak| peak <= self.charge)
    }

    pub(super) fn peak(self, rows: usize, columns: usize) -> Option<usize> {
        probe_peak(self.bytes, self.row_bytes, rows, columns)
    }
}

fn probe_peak(bytes: usize, row_bytes: usize, rows: usize, columns: usize) -> Option<usize> {
    if rows == 0 {
        return None;
    }
    [
        bytes.checked_add(row_bytes),
        rows.checked_mul(size_of::<u32>()),
        interner_vector_peak(rows),
        rows.checked_mul(size_of::<super::columnar::FramedKey>() + 2 * size_of::<usize>()),
        table_peak(rows),
        columns.checked_mul(size_of::<KeyWriter<'_>>()),
    ]
    .into_iter()
    .try_fold(BASE_BYTES, |total, bytes| total.checked_add(bytes?))
}

fn interner_vector_peak(rows: usize) -> Option<usize> {
    let capacity = rows.checked_next_power_of_two()?.max(4);
    let key = size_of::<std::sync::Arc<super::columnar::FramedKey>>();
    let hash = size_of::<u64>();
    capacity
        .checked_mul(key + hash)?
        .checked_add((capacity / 2).checked_mul(key.max(hash))?)
}

fn table_peak(rows: usize) -> Option<usize> {
    let buckets = rows
        .checked_mul(8)?
        .div_ceil(7)
        .checked_next_power_of_two()?
        .max(4);
    buckets
        .checked_mul(size_of::<u32>() + 1)?
        .checked_add(64)?
        .checked_mul(2)
}

fn parent_groups(rows: &[AdmittedRow]) -> impl Iterator<Item = &[AdmittedRow]> {
    rows.chunk_by(|a, b| std::ptr::eq(a.record.columns(), b.record.columns()))
}

pub(super) fn visit_key_frames(
    rows: &[AdmittedRow],
    indices: &[usize],
    layout: ArenaLayout,
    mut visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut buffer = Vec::with_capacity(layout.row_bytes);
    let mut writers = Vec::with_capacity(indices.len());
    for group in parent_groups(rows) {
        writers.clear();
        for &index in indices {
            let column = KeyColumn::bind(group[0].record.column(index).as_ref())
                .expect("readonly typed sizing validated key column");
            writers.push(KeyWriter::new(column));
        }
        for row in group {
            buffer.clear();
            for writer in &writers {
                writer.write(row.record.offset(), &mut buffer)?;
            }
            debug_assert!(buffer.len() <= layout.row_bytes);
            #[cfg(test)]
            super::note_join_work(|work| work.arena_frames += 1);
            visit(&buffer)?;
        }
    }
    Ok(())
}

struct KeyColumn<'a> {
    array: &'a dyn Array,
    values: KeyValues<'a>,
    tag: u8,
    timezone: &'a [u8],
}

impl<'a> KeyColumn<'a> {
    fn frame_supported(&self, row: usize) -> bool {
        !self.array.is_null(row)
            && u32::try_from(self.timezone.len()).is_ok()
            && u32::try_from(self.values.len(row)).is_ok()
    }

    fn bind(array: &'a dyn Array) -> Option<Self> {
        let values = KeyValues::bind(array)?;
        let tag = key_type_tag(array.data_type()).ok()?;
        let timezone = match array.data_type() {
            DataType::Timestamp(_, Some(timezone)) => timezone.as_bytes(),
            _ => &[],
        };
        #[cfg(test)]
        super::note_join_work(|work| work.key_type_resolutions += 1);
        Some(Self {
            array,
            values,
            tag,
            timezone,
        })
    }
}

struct KeyWriter<'a> {
    values: KeyValues<'a>,
    prefix: Option<[u8; 5]>,
    timezone: &'a [u8],
}

impl<'a> KeyWriter<'a> {
    fn new(column: KeyColumn<'a>) -> Self {
        let prefix = u32::try_from(column.timezone.len()).ok().map(|length| {
            let length = length.to_le_bytes();
            [column.tag, length[0], length[1], length[2], length[3]]
        });
        Self {
            values: column.values,
            prefix,
            timezone: column.timezone,
        }
    }

    fn write(&self, row: usize, bytes: &mut Vec<u8>) -> Result<()> {
        let prefix = self
            .prefix
            .ok_or_else(|| charge_overflow("key timezone length"))?;
        bytes.extend_from_slice(&prefix);
        bytes.extend_from_slice(self.timezone);
        bytes.extend_from_slice(&framed_length(self.values.len(row), "key value length")?);
        self.values.write(row, bytes);
        Ok(())
    }
}

fn framed_length(length: usize, field: &str) -> Result<[u8; 4]> {
    u32::try_from(length)
        .map(u32::to_le_bytes)
        .map_err(|_| charge_overflow(field))
}

enum KeyValues<'a> {
    Boolean(&'a BooleanArray),
    Signed(SignedValues<'a>),
    Unsigned(UnsignedValues<'a>),
    Utf8(&'a StringArray),
    LargeUtf8(&'a LargeStringArray),
}

impl<'a> KeyValues<'a> {
    fn bind(array: &'a dyn Array) -> Option<Self> {
        match array.data_type() {
            DataType::Boolean => Some(Self::Boolean(array.as_any().downcast_ref()?)),
            DataType::Int16 | DataType::Int32 | DataType::Int64 | DataType::Timestamp(..) => {
                SignedValues::bind(array).map(Self::Signed)
            }
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                UnsignedValues::bind(array).map(Self::Unsigned)
            }
            DataType::Utf8 => Some(Self::Utf8(array.as_any().downcast_ref()?)),
            DataType::LargeUtf8 => Some(Self::LargeUtf8(array.as_any().downcast_ref()?)),
            _ => None,
        }
    }

    fn len(&self, row: usize) -> usize {
        match self {
            Self::Boolean(_) => 1,
            Self::Signed(values) => values.width(),
            Self::Unsigned(values) => values.width(),
            Self::Utf8(values) => values.value(row).len(),
            Self::LargeUtf8(values) => values.value(row).len(),
        }
    }

    fn prefix(&self) -> usize {
        match self {
            Self::Utf8(_) => 4,
            Self::LargeUtf8(_) => 8,
            _ => 0,
        }
    }

    fn write(&self, row: usize, bytes: &mut Vec<u8>) {
        match self {
            Self::Boolean(values) => bytes.push(u8::from(values.value(row))),
            Self::Signed(values) => values.write(row, bytes),
            Self::Unsigned(values) => values.write(row, bytes),
            Self::Utf8(values) => bytes.extend_from_slice(values.value(row).as_bytes()),
            Self::LargeUtf8(values) => bytes.extend_from_slice(values.value(row).as_bytes()),
        }
    }
}

enum SignedValues<'a> {
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl<'a> SignedValues<'a> {
    fn bind(array: &'a dyn Array) -> Option<Self> {
        match array.data_type() {
            DataType::Int16 => Some(Self::I16(primitive_values::<Int16Type>(array)?)),
            DataType::Int32 => Some(Self::I32(primitive_values::<Int32Type>(array)?)),
            DataType::Int64 => Some(Self::I64(primitive_values::<Int64Type>(array)?)),
            DataType::Timestamp(unit, _) => Self::timestamp(array, *unit),
            _ => None,
        }
    }

    fn timestamp(array: &'a dyn Array, unit: TimeUnit) -> Option<Self> {
        let values = match unit {
            TimeUnit::Second => primitive_values::<TimestampSecondType>(array)?,
            TimeUnit::Millisecond => primitive_values::<TimestampMillisecondType>(array)?,
            TimeUnit::Microsecond => primitive_values::<TimestampMicrosecondType>(array)?,
            TimeUnit::Nanosecond => primitive_values::<TimestampNanosecondType>(array)?,
        };
        Some(Self::I64(values))
    }

    fn width(&self) -> usize {
        match self {
            Self::I16(_) => 2,
            Self::I32(_) => 4,
            Self::I64(_) => 8,
        }
    }

    fn write(&self, row: usize, bytes: &mut Vec<u8>) {
        match self {
            Self::I16(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
            Self::I32(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
            Self::I64(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
        }
    }
}

enum UnsignedValues<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
    U32(&'a [u32]),
    U64(&'a [u64]),
}

impl<'a> UnsignedValues<'a> {
    fn bind(array: &'a dyn Array) -> Option<Self> {
        match array.data_type() {
            DataType::UInt8 => Some(Self::U8(primitive_values::<UInt8Type>(array)?)),
            DataType::UInt16 => Some(Self::U16(primitive_values::<UInt16Type>(array)?)),
            DataType::UInt32 => Some(Self::U32(primitive_values::<UInt32Type>(array)?)),
            DataType::UInt64 => Some(Self::U64(primitive_values::<UInt64Type>(array)?)),
            _ => None,
        }
    }

    fn width(&self) -> usize {
        match self {
            Self::U8(_) => 1,
            Self::U16(_) => 2,
            Self::U32(_) => 4,
            Self::U64(_) => 8,
        }
    }

    fn write(&self, row: usize, bytes: &mut Vec<u8>) {
        match self {
            Self::U8(values) => bytes.push(values[row]),
            Self::U16(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
            Self::U32(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
            Self::U64(values) => bytes.extend_from_slice(&values[row].to_le_bytes()),
        }
    }
}

fn primitive_values<T: ArrowPrimitiveType>(array: &dyn Array) -> Option<&[T::Native]> {
    Some(array.as_any().downcast_ref::<PrimitiveArray<T>>()?.values())
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    pub(in crate::operator::join) fn assert_empty_arena_sizing_for_wide_schema_allocates_nothing() {
        let columns = vec![0; 1024];
        let mut layout = None;
        let measured = allocation_counter::measure(|| {
            layout = ArenaLayout::measure(&[], &columns).unwrap();
        });
        let layout = layout.unwrap();
        assert_eq!(layout.bytes, 0);
        assert_eq!(layout.charge, BASE_BYTES);
        assert!(!layout.fits(0, columns.len()));
        assert_eq!(measured.bytes_max, 0);
        assert_eq!(measured.bytes_current, 0);
    }
}
