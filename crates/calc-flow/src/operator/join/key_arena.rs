#[cfg(test)]
use super::{AdmittedRows, admitted::AdmittedGroup, native_dictionary::BASE_BYTES};
use super::{borrowed_key::CellValue, charge_overflow, key_type_tag};
use crate::Result;
use datafusion::arrow::{
    array::{Array, BooleanArray, LargeStringArray, PrimitiveArray, StringArray},
    datatypes::{
        ArrowPrimitiveType, DataType, Date32Type, Date64Type, Int8Type, Int16Type, Int32Type,
        Int64Type, TimeUnit, TimestampMicrosecondType, TimestampMillisecondType,
        TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type, UInt32Type,
        UInt64Type,
    },
};

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) struct ArenaLayout {
    pub(super) charge: usize,
    pub(super) single_string: bool,
    bytes: usize,
    row_bytes: usize,
}

#[cfg(test)]
fn err() -> crate::CalcFlowError {
    super::native_lookup::scratch_error("join")
}

#[cfg(test)]
impl ArenaLayout {
    pub(super) fn measure(rows: &dyn AdmittedRows, indices: &[usize]) -> Result<Option<Self>> {
        let Some(extra) = rows
            .len()
            .checked_mul(size_of::<u32>() + size_of::<Option<u32>>())
            .and_then(|n| n.checked_add(BASE_BYTES))
        else {
            return Err(super::native_lookup::scratch_error("join"));
        };
        let mut layout = Self {
            charge: extra,
            single_string: indices.len() == 1 && !rows.is_empty(),
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

    fn measure_group(&mut self, rows: AdmittedGroup<'_>, indices: &[usize]) -> Result<bool> {
        let mut maximum = 0_usize;
        for &index in indices {
            let Some(column) = KeyColumn::bind(rows.row(0).record.column(index).as_ref()) else {
                return Ok(false);
            };
            let Some((sum, largest)) = column_extent(&column, rows) else {
                return Ok(false);
            };
            self.single_string &=
                matches!(column.values, KeyValues::Utf8(_) | KeyValues::LargeUtf8(_));
            let size = self.measure_column(rows.len(), &column, sum, largest)?;
            maximum = maximum.checked_add(size).ok_or_else(err)?;
        }
        self.row_bytes = self.row_bytes.max(maximum);
        Ok(true)
    }

    fn measure_column(
        &mut self,
        count: usize,
        column: &KeyColumn<'_>,
        sum: usize,
        largest: usize,
    ) -> Result<usize> {
        let frame = column.timezone.len().checked_add(9).ok_or_else(err)?;
        self.bytes = count
            .checked_mul(frame)
            .and_then(|bytes| bytes.checked_add(sum))
            .and_then(|bytes| self.bytes.checked_add(bytes))
            .ok_or_else(err)?;
        let charge = column_charge(count, column, sum)?;
        self.charge = self.charge.checked_add(charge).ok_or_else(err)?;
        largest.checked_add(frame).ok_or_else(err)
    }

    pub(super) fn fits(self, rows: usize, columns: usize) -> bool {
        self.peak(rows, columns)
            .is_some_and(|peak| peak <= self.charge)
    }

    pub(super) fn peak(self, rows: usize, columns: usize) -> Option<usize> {
        probe_peak(self.bytes, self.row_bytes, rows, columns)
    }
}

#[cfg(test)]
fn column_extent(column: &KeyColumn<'_>, rows: AdmittedGroup<'_>) -> Option<(usize, usize)> {
    if u32::try_from(column.timezone.len()).is_err() {
        return None;
    }
    let extent = column.values.extent(rows)?;
    u32::try_from(extent.1).ok()?;
    Some(extent)
}

#[cfg(test)]
fn column_charge(count: usize, column: &KeyColumn<'_>, sum: usize) -> Result<usize> {
    let fixed = column
        .timezone
        .len()
        .checked_add(column.values.prefix())
        .and_then(|bytes| bytes.checked_add(65));
    fixed
        .and_then(|fixed| count.checked_mul(fixed))
        .and_then(|bytes| bytes.checked_add(sum))
        .and_then(|bytes| bytes.checked_mul(4))
        .ok_or_else(err)
}

#[cfg(test)]
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

#[cfg(test)]
fn interner_vector_peak(rows: usize) -> Option<usize> {
    let capacity = rows.checked_next_power_of_two()?.max(4);
    let key = size_of::<std::sync::Arc<super::columnar::FramedKey>>();
    let hash = size_of::<u64>();
    capacity
        .checked_mul(key + hash)?
        .checked_add((capacity / 2).checked_mul(key.max(hash))?)
}

#[cfg(test)]
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

#[cfg(test)]
fn parent_groups(rows: &dyn AdmittedRows) -> impl Iterator<Item = AdmittedGroup<'_>> {
    rows.groups()
}

#[cfg(test)]
pub(super) fn visit_string_values(
    rows: &dyn AdmittedRows,
    index: usize,
    mut visit: impl FnMut(u8, &[u8]) -> Result<()>,
) -> Result<()> {
    for group in parent_groups(rows) {
        let column = KeyColumn::bind(group.row(0).record.column(index).as_ref())
            .expect("readonly sizing validated a string key column");
        for row in group.iter() {
            let value = match &column.values {
                KeyValues::Utf8(values) => values.value(row.record.offset()),
                KeyValues::LargeUtf8(values) => values.value(row.record.offset()),
                _ => unreachable!("string layout contains only string parents"),
            };
            visit(column.tag, value.as_bytes())?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(super) fn visit_key_frames(
    rows: &dyn AdmittedRows,
    indices: &[usize],
    layout: ArenaLayout,
    mut visit: impl FnMut(&[u8]) -> Result<()>,
) -> Result<()> {
    let mut buffer = Vec::with_capacity(layout.row_bytes);
    let mut writers = Vec::with_capacity(indices.len());
    for group in parent_groups(rows) {
        writers.clear();
        for &index in indices {
            let column = KeyColumn::bind(group.row(0).record.column(index).as_ref())
                .expect("readonly typed sizing validated key column");
            writers.push(KeyWriter::new(column));
        }
        for row in group.iter() {
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
    values: KeyValues<'a>,
    tag: u8,
    timezone: &'a [u8],
}

impl<'a> KeyColumn<'a> {
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
            values,
            tag,
            timezone,
        })
    }
}

pub(super) struct KeyWriter<'a> {
    values: KeyValues<'a>,
    prefix: Option<[u8; 5]>,
    timezone: &'a [u8],
}

impl<'a> KeyWriter<'a> {
    pub(super) fn bind(array: &'a dyn Array) -> Result<Self> {
        KeyColumn::bind(array)
            .map(Self::new)
            .ok_or_else(|| super::charge_type_mismatch("Join key column"))
    }

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

    #[cfg(test)]
    fn write(&self, row: usize, bytes: &mut Vec<u8>) -> Result<()> {
        self.visit(row, |part| bytes.extend_from_slice(part))
    }

    pub(super) fn parts(&self, row: usize) -> Result<KeyParts<'a>> {
        Ok(KeyParts {
            prefix: self
                .prefix
                .ok_or_else(|| charge_overflow("key timezone length"))?,
            timezone: self.timezone,
            length: framed_length(self.values.len(row), "key value length")?,
            value: self.values.cell(row),
        })
    }

    #[cfg(test)]
    pub(super) fn visit(&self, row: usize, mut visitor: impl FnMut(&[u8])) -> Result<()> {
        for part in self.parts(row)?.slices() {
            visitor(part);
        }
        Ok(())
    }
}

pub(super) struct KeyParts<'a> {
    prefix: [u8; 5],
    timezone: &'a [u8],
    length: [u8; 4],
    value: CellValue<'a>,
}
impl KeyParts<'_> {
    pub(super) fn slices(&self) -> [&[u8]; 4] {
        [
            &self.prefix,
            self.timezone,
            &self.length,
            self.value.bytes(),
        ]
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
            DataType::Boolean => array.as_any().downcast_ref().map(Self::Boolean),
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(..) => SignedValues::bind(array).map(Self::Signed),
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                UnsignedValues::bind(array).map(Self::Unsigned)
            }
            DataType::Utf8 => array.as_any().downcast_ref().map(Self::Utf8),
            DataType::LargeUtf8 => array.as_any().downcast_ref().map(Self::LargeUtf8),
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

    /// Summed and largest value length over the group's rows; `None` when a
    /// length cannot be framed. Fixed-width columns resolve in O(1).
    #[cfg(test)]
    fn extent(&self, rows: AdmittedGroup<'_>) -> Option<(usize, usize)> {
        match self {
            Self::Boolean(_) => Some((rows.len(), 1)),
            Self::Signed(values) => Some((rows.len().checked_mul(values.width())?, values.width())),
            Self::Unsigned(values) => {
                Some((rows.len().checked_mul(values.width())?, values.width()))
            }
            Self::Utf8(_) | Self::LargeUtf8(_) => {
                let mut sum = 0_usize;
                let mut max = 0_usize;
                for row in rows.iter() {
                    let len = self.len(row.record.offset());
                    sum = sum.checked_add(len)?;
                    max = max.max(len);
                }
                Some((sum, max))
            }
        }
    }

    #[cfg(test)]
    fn prefix(&self) -> usize {
        match self {
            Self::Utf8(_) => 4,
            Self::LargeUtf8(_) => 8,
            _ => 0,
        }
    }

    fn cell(&self, row: usize) -> CellValue<'a> {
        match self {
            Self::Boolean(values) => CellValue::inline([u8::from(values.value(row))]),
            Self::Signed(values) => values.cell(row),
            Self::Unsigned(values) => values.cell(row),
            Self::Utf8(values) => CellValue::Borrowed((*values).value(row).as_bytes()),
            Self::LargeUtf8(values) => CellValue::Borrowed((*values).value(row).as_bytes()),
        }
    }
}

enum SignedValues<'a> {
    I8(&'a [i8]),
    I16(&'a [i16]),
    I32(&'a [i32]),
    I64(&'a [i64]),
}

impl<'a> SignedValues<'a> {
    fn bind(array: &'a dyn Array) -> Option<Self> {
        match array.data_type() {
            DataType::Int8 => primitive_values::<Int8Type>(array).map(Self::I8),
            DataType::Int16 => primitive_values::<Int16Type>(array).map(Self::I16),
            DataType::Int32 => primitive_values::<Int32Type>(array).map(Self::I32),
            DataType::Int64 => primitive_values::<Int64Type>(array).map(Self::I64),
            DataType::Timestamp(unit, _) => Self::timestamp(array, *unit),
            DataType::Date32 | DataType::Date64 => Self::date(array),
            _ => None,
        }
    }

    fn date(array: &'a dyn Array) -> Option<Self> {
        match array.data_type() {
            DataType::Date32 => Some(Self::I32(primitive_values::<Date32Type>(array)?)),
            DataType::Date64 => Some(Self::I64(primitive_values::<Date64Type>(array)?)),
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
            Self::I8(_) => 1,
            Self::I16(_) => 2,
            Self::I32(_) => 4,
            Self::I64(_) => 8,
        }
    }

    fn cell(&self, row: usize) -> CellValue<'static> {
        match self {
            Self::I8(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::I16(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::I32(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::I64(values) => CellValue::inline(values[row].to_le_bytes()),
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

    fn cell(&self, row: usize) -> CellValue<'static> {
        match self {
            Self::U8(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::U16(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::U32(values) => CellValue::inline(values[row].to_le_bytes()),
            Self::U64(values) => CellValue::inline(values[row].to_le_bytes()),
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
