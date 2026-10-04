use datafusion::arrow::array::builder::{
    ArrayBuilder, BooleanBuilder, Decimal32Builder, Decimal64Builder, Decimal128Builder,
    Decimal256Builder, Float32Builder, Float64Builder, Int8Builder, Int16Builder, Int32Builder,
    Int64Builder, LargeStringBuilder, NullBuilder, StringBuilder, UInt8Builder, UInt16Builder,
    UInt32Builder, UInt64Builder, make_builder,
};

use super::{
    ArrayRef, CHUNK_ROWS, DataType, Group, IncrementalSql, MemoryReservation,
    NativeStateDescriptor, PaidNativeStateRecords, RecordBatch, Result, STATE_CHUNK_ROWS,
    ScalarValue, StateColumn, checked_bytes, df_error, ensure_reservation, validate_scalar,
};

const STRING_STEP_BYTES: usize = 64 * 1024;

pub(super) struct ExportCursor<'a> {
    native: &'a IncrementalSql,
    slots: Option<Vec<usize>>,
    name: &'a str,
    records: Vec<RecordBatch>,
    arrays: Vec<ArrayRef>,
    builder: Option<ColumnExport>,
    columns: Vec<StateColumn>,
    string_bytes: Vec<usize>,
    census_row: usize,
    census_column: usize,
    record_count: usize,
    charge: usize,
    descriptor: NativeStateDescriptor,
    reservation: MemoryReservation,
    census_complete: bool,
}

impl<'a> ExportCursor<'a> {
    pub(super) fn new(native: &'a IncrementalSql, name: &'a str) -> Result<Self> {
        Self::create(native, false, name)
    }

    pub(super) fn dirty(native: &'a IncrementalSql, name: &'a str) -> Result<Self> {
        Self::create(native, true, name)
    }

    fn create(native: &'a IncrementalSql, dirty: bool, name: &'a str) -> Result<Self> {
        let mut descriptor = native.native_descriptor(name)?;
        let rows = if dirty {
            native.dirty.slots().len()
        } else {
            native.groups.len()
        };
        descriptor.group_count = rows;
        let width = descriptor.wire_schema.fields().len();
        if width == 0 && !native.groups.is_empty() {
            return Err(df_error(name, "native groups have no state columns"));
        }
        let record_count = rows.div_ceil(CHUNK_ROWS).max(1);
        let charge = checked_bytes(
            4096,
            [
                (record_count, size_of::<RecordBatch>()),
                (record_count, checked_bytes(0, [(width, 512)], name)?),
                (usize::from(dirty) * rows, size_of::<usize>()),
            ],
            name,
        )?;
        let reservation = native.reservation.new_empty();
        ensure_reservation(&reservation, charge, name)?;
        let selected_slots = dirty.then(|| {
            let mut slots = native.dirty.slots().to_vec();
            slots.sort_unstable();
            slots
        });
        let slots = record_count
            .checked_mul(width)
            .ok_or_else(|| df_error(name, "native export slot count overflowed"))?;
        let columns =
            (0..descriptor.key_fields.len())
                .map(StateColumn::Key)
                .chain(descriptor.state_fields.iter().enumerate().flat_map(
                    |(aggregate, fields)| {
                        (0..fields.len()).map(move |state| StateColumn::Aggregate(aggregate, state))
                    },
                ))
                .collect();
        Ok(Self {
            native,
            slots: selected_slots,
            name,
            descriptor,
            records: Vec::with_capacity(record_count),
            arrays: Vec::with_capacity(width),
            builder: None,
            columns,
            string_bytes: Vec::with_capacity(slots),
            census_row: 0,
            census_column: 0,
            record_count,
            charge,
            reservation,
            census_complete: false,
        })
    }

    pub(super) fn step(&mut self, check: &mut impl FnMut() -> Result<()>) -> Result<bool> {
        check()?;
        if !self.census_complete {
            self.census_step(check)?;
            return Ok(false);
        }
        if self.arrays.len() == self.columns.len() {
            self.finish_record()?;
            return Ok(self.records.len() == self.record_count);
        }
        if self.builder.is_none() {
            let start = self.records.len() * CHUNK_ROWS;
            let rows = self
                .descriptor
                .group_count
                .saturating_sub(start)
                .min(CHUNK_ROWS);
            let column = self.arrays.len();
            let bytes = self
                .string_bytes
                .get(self.records.len() * self.columns.len() + column)
                .copied()
                .unwrap_or(0);
            self.builder = Some(ColumnExport::new(
                self.descriptor.wire_schema.field(column).data_type(),
                rows,
                bytes,
                self.name,
            )?);
        }
        let start = self.records.len() * CHUNK_ROWS;
        let end = (start + CHUNK_ROWS).min(self.descriptor.group_count);
        let builder = self.builder.as_mut().expect("started export column");
        if builder.step(
            GroupView::new(&self.native.groups, self.slots.as_deref()).slice(start, end),
            self.columns[self.arrays.len()],
            self.name,
            check,
        )? {
            let array = self
                .builder
                .take()
                .expect("finished export column")
                .builder
                .finish();
            #[cfg(test)]
            super::super::super::compact::direct_async_tests::after_array(self.name, &array);
            self.arrays.push(array);
        }
        check()?;
        Ok(false)
    }

    fn census_step(&mut self, check: &mut impl FnMut() -> Result<()>) -> Result<()> {
        check()?;
        for _ in 0..STATE_CHUNK_ROWS {
            if self.census_row == self.descriptor.group_count {
                ensure_reservation(&self.reservation, self.charge, self.name)?;
                self.census_complete = true;
                return Ok(());
            }
            let group =
                GroupView::new(&self.native.groups, self.slots.as_deref()).get(self.census_row);
            if group.values.len() != self.descriptor.key_fields.len()
                || group.states.len() != self.descriptor.state_fields.len()
            {
                return Err(df_error(self.name, "native group field census differs"));
            }
            let column = self.columns[self.census_column];
            if let StateColumn::Aggregate(aggregate, _) = column
                && group.states[aggregate].len() != self.descriptor.state_fields[aggregate].len()
            {
                return Err(df_error(self.name, "native aggregate field census differs"));
            }
            let value = scalar(group, column);
            validate_scalar(
                value,
                self.descriptor.wire_schema.field(self.census_column),
                self.name,
            )?;
            self.charge = checked_bytes(self.charge, [(value.size(), 4), (1, 128)], self.name)?;
            let slot = (self.census_row / CHUNK_ROWS) * self.columns.len() + self.census_column;
            if self.string_bytes.len() == slot {
                self.string_bytes.push(0);
            }
            if let ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) = value {
                self.string_bytes[slot] = self.string_bytes[slot]
                    .checked_add(value.len())
                    .ok_or_else(|| df_error(self.name, "native export string size overflowed"))?;
            }
            self.census_column += 1;
            if self.census_column == self.columns.len() {
                self.census_column = 0;
                self.census_row += 1;
            }
        }
        #[cfg(test)]
        super::super::super::compact::direct_async_tests::after_census(self.name, self.census_row);
        Ok(())
    }

    fn finish_record(&mut self) -> Result<()> {
        let arrays = std::mem::replace(&mut self.arrays, Vec::with_capacity(self.columns.len()));
        let record = RecordBatch::try_new(self.descriptor.wire_schema.clone(), arrays)
            .map_err(|error| df_error(self.name, error))?;
        self.records.push(record);
        Ok(())
    }

    pub(super) fn finish(self) -> PaidNativeStateRecords {
        PaidNativeStateRecords {
            records: self.records,
            descriptor: self.descriptor,
            _reservation: self.reservation,
        }
    }
}

fn scalar(group: &Group, column: StateColumn) -> &ScalarValue {
    match column {
        StateColumn::Key(key) => &group.values[key],
        StateColumn::Aggregate(aggregate, state) => &group.states[aggregate][state],
    }
}

#[derive(Clone, Copy)]
struct GroupView<'a> {
    groups: &'a [Group],
    slots: Option<&'a [usize]>,
}

impl<'a> GroupView<'a> {
    fn new(groups: &'a [Group], slots: Option<&'a [usize]>) -> Self {
        Self { groups, slots }
    }

    fn len(self) -> usize {
        self.slots.map_or(self.groups.len(), <[usize]>::len)
    }

    fn get(self, row: usize) -> &'a Group {
        &self.groups[self.slots.map_or(row, |slots| slots[row])]
    }

    fn slice(self, start: usize, end: usize) -> Self {
        match self.slots {
            Some(slots) => Self::new(self.groups, Some(&slots[start..end])),
            None => Self::new(&self.groups[start..end], None),
        }
    }
}

struct ColumnExport {
    builder: Box<dyn ArrayBuilder>,
    row: usize,
    offset: usize,
}

impl ColumnExport {
    fn new(data_type: &DataType, rows: usize, bytes: usize, name: &str) -> Result<Self> {
        let builder: Box<dyn ArrayBuilder> = match data_type {
            DataType::Utf8 => {
                if bytes > i32::MAX as usize {
                    return Err(df_error(name, "native UTF8 export exceeds offset range"));
                }
                Box::new(StringBuilder::with_capacity(rows, bytes))
            }
            DataType::LargeUtf8 => Box::new(LargeStringBuilder::with_capacity(rows, bytes)),
            DataType::Null
            | DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Decimal32(..)
            | DataType::Decimal64(..)
            | DataType::Decimal128(..)
            | DataType::Decimal256(..) => make_builder(data_type, rows),
            _ => {
                return Err(df_error(
                    name,
                    "native export has an unsupported field type",
                ));
            }
        };
        Ok(Self {
            builder,
            row: 0,
            offset: 0,
        })
    }

    fn step(
        &mut self,
        groups: GroupView<'_>,
        column: StateColumn,
        name: &str,
        check: &mut impl FnMut() -> Result<()>,
    ) -> Result<bool> {
        check()?;
        let mut bytes = 0;
        for _ in 0..STATE_CHUNK_ROWS {
            if self.row == groups.len() {
                return Ok(true);
            }
            let value = scalar(groups.get(self.row), column);
            let copied = match value {
                ScalarValue::Utf8(value) => {
                    self.string::<StringBuilder>(value.as_deref(), STRING_STEP_BYTES - bytes, name)?
                }
                ScalarValue::LargeUtf8(value) => self.string::<LargeStringBuilder>(
                    value.as_deref(),
                    STRING_STEP_BYTES - bytes,
                    name,
                )?,
                _ => {
                    append_fixed(self.builder.as_mut(), value, name)?;
                    self.row += 1;
                    0
                }
            };
            bytes += copied;
            #[cfg(test)]
            super::super::super::compact::direct_async_tests::after_bytes(name, copied);
            if bytes == STRING_STEP_BYTES || self.offset != 0 {
                break;
            }
        }
        Ok(self.row == groups.len())
    }

    fn string<B: StringExport>(
        &mut self,
        value: Option<&str>,
        limit: usize,
        name: &str,
    ) -> Result<usize> {
        let builder = self
            .builder
            .as_any_mut()
            .downcast_mut::<B>()
            .ok_or_else(|| df_error(name, "native string builder differs from field"))?;
        let Some(value) = value else {
            builder.null();
            self.row += 1;
            return Ok(0);
        };
        let mut end = self.offset.saturating_add(limit).min(value.len());
        while !value.is_char_boundary(end) {
            end -= 1;
        }
        let fragment = &value[self.offset..end];
        builder
            .write_str(fragment)
            .map_err(|error| df_error(name, error))?;
        self.offset = end;
        if end == value.len() {
            builder.end();
            self.offset = 0;
            self.row += 1;
        }
        Ok(fragment.len())
    }
}

trait StringExport: std::fmt::Write + 'static {
    fn null(&mut self);
    fn end(&mut self);
}

impl StringExport for StringBuilder {
    fn null(&mut self) {
        self.append_null();
    }
    fn end(&mut self) {
        self.append_value("");
    }
}

impl StringExport for LargeStringBuilder {
    fn null(&mut self) {
        self.append_null();
    }
    fn end(&mut self) {
        self.append_value("");
    }
}

macro_rules! append {
    ($builder:expr, $kind:ty, $value:expr, $name:expr) => {
        $builder
            .as_any_mut()
            .downcast_mut::<$kind>()
            .ok_or_else(|| df_error($name, "native builder differs from field"))?
            .append_option(*$value)
    };
}

fn append_fixed(builder: &mut dyn ArrayBuilder, value: &ScalarValue, name: &str) -> Result<()> {
    if append_signed(builder, value, name)?
        || append_unsigned(builder, value, name)?
        || append_decimal(builder, value, name)?
    {
        return Ok(());
    }
    match value {
        ScalarValue::Null => builder
            .as_any_mut()
            .downcast_mut::<NullBuilder>()
            .ok_or_else(|| df_error(name, "native null builder differs from field"))?
            .append_null(),
        ScalarValue::Boolean(value) => append!(builder, BooleanBuilder, value, name),
        ScalarValue::Float32(value) => append!(builder, Float32Builder, value, name),
        ScalarValue::Float64(value) => append!(builder, Float64Builder, value, name),
        _ => {
            return Err(df_error(
                name,
                "native export scalar has an unsupported type",
            ));
        }
    }
    Ok(())
}

fn append_signed(builder: &mut dyn ArrayBuilder, value: &ScalarValue, name: &str) -> Result<bool> {
    match value {
        ScalarValue::Int8(value) => append!(builder, Int8Builder, value, name),
        ScalarValue::Int16(value) => append!(builder, Int16Builder, value, name),
        ScalarValue::Int32(value) => append!(builder, Int32Builder, value, name),
        ScalarValue::Int64(value) => append!(builder, Int64Builder, value, name),
        _ => return Ok(false),
    }
    Ok(true)
}

fn append_unsigned(
    builder: &mut dyn ArrayBuilder,
    value: &ScalarValue,
    name: &str,
) -> Result<bool> {
    match value {
        ScalarValue::UInt8(value) => append!(builder, UInt8Builder, value, name),
        ScalarValue::UInt16(value) => append!(builder, UInt16Builder, value, name),
        ScalarValue::UInt32(value) => append!(builder, UInt32Builder, value, name),
        ScalarValue::UInt64(value) => append!(builder, UInt64Builder, value, name),
        _ => return Ok(false),
    }
    Ok(true)
}

fn append_decimal(builder: &mut dyn ArrayBuilder, value: &ScalarValue, name: &str) -> Result<bool> {
    match value {
        ScalarValue::Decimal32(value, ..) => append!(builder, Decimal32Builder, value, name),
        ScalarValue::Decimal64(value, ..) => append!(builder, Decimal64Builder, value, name),
        ScalarValue::Decimal128(value, ..) => append!(builder, Decimal128Builder, value, name),
        ScalarValue::Decimal256(value, ..) => append!(builder, Decimal256Builder, value, name),
        _ => return Ok(false),
    }
    Ok(true)
}
