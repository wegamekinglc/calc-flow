//! Borrowed schema validation before Arrow IPC decoding allocates schema owners.

use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    datatypes::{DataType, Field, IntervalUnit, Schema, TimeUnit, UnionFields, UnionMode},
    ipc,
};
use std::{cmp::Ordering, collections::HashMap};

#[derive(Debug, Default, Clone, Copy, Eq, PartialEq)]
pub(super) struct SchemaCounts {
    pub(super) fields: usize,
    pub(super) dictionary_fields: usize,
    pub(super) metadata_pairs: usize,
    pub(super) metadata_string_bytes: usize,
    pub(super) field_name_bytes: usize,
    pub(super) timezone_string_bytes: usize,
    pub(super) union_type_ids: usize,
}
struct Walk<'a> {
    counts: SchemaCounts,
    check: &'a mut dyn FnMut() -> Result<()>,
}
fn format() -> CalcFlowError {
    CalcFlowError::Format {
        message: "rolling IPC schema differs from compiled schema".into(),
    }
}
fn require(valid: bool) -> Result<()> {
    if valid { Ok(()) } else { Err(format()) }
}
fn add(value: &mut usize, amount: usize) -> Result<()> {
    *value = value.checked_add(amount).ok_or_else(format)?;
    Ok(())
}
pub(super) fn validate(
    actual: ipc::Schema<'_>,
    expected: &Schema,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<SchemaCounts> {
    check()?;
    let mut walk = Walk {
        counts: SchemaCounts::default(),
        check,
    };
    header(actual)?;
    schema_fields(actual, expected, &mut walk)?;
    let pairs = actual.custom_metadata();
    metadata(
        pairs.map_or(0, |pairs| pairs.len()),
        pairs.into_iter().flat_map(|pairs| pairs.iter()),
        expected.metadata(),
        &mut walk,
    )?;
    (walk.check)()?;
    Ok(walk.counts)
}
fn schema_fields(actual: ipc::Schema<'_>, expected: &Schema, walk: &mut Walk<'_>) -> Result<()> {
    let fields = actual.fields().ok_or_else(format)?;
    require(fields.len() == expected.fields().len())?;
    for (actual, expected) in fields.iter().zip(expected.fields()) {
        field(actual, expected, walk)?;
    }
    Ok(())
}
fn header(actual: ipc::Schema<'_>) -> Result<()> {
    let endian = if cfg!(target_endian = "little") {
        ipc::Endianness::Little
    } else {
        ipc::Endianness::Big
    };
    require(actual.endianness() == endian)?;
    require(actual.features().is_none())
}
fn field(actual: ipc::Field<'_>, expected: &Field, walk: &mut Walk<'_>) -> Result<()> {
    (walk.check)()?;
    add(&mut walk.counts.fields, 1)?;
    let name = actual.name().ok_or_else(format)?;
    field_identity(actual, expected, name, walk)?;
    field_type(actual, expected, walk)
}
fn field_type(actual: ipc::Field<'_>, expected: &Field, walk: &mut Walk<'_>) -> Result<()> {
    let dtype = value_type(expected.data_type())?;
    datatype(actual, dtype, walk)?;
    children(actual, dtype, walk)?;
    dictionary(actual, expected, walk)
}

fn field_identity(
    actual: ipc::Field<'_>,
    expected: &Field,
    name: &str,
    walk: &mut Walk<'_>,
) -> Result<()> {
    require(compare(name, expected.name(), walk.check)? == Ordering::Equal)?;
    require(actual.nullable() == expected.is_nullable())?;
    add(&mut walk.counts.field_name_bytes, name.len())?;
    let pairs = actual.custom_metadata();
    metadata(
        pairs.map_or(0, |pairs| pairs.len()),
        pairs.into_iter().flat_map(|pairs| pairs.iter()),
        expected.metadata(),
        walk,
    )
}
fn value_type(expected: &DataType) -> Result<&DataType> {
    match expected {
        DataType::Dictionary(_, value) => {
            require(!matches!(value.as_ref(), DataType::Dictionary(..)))?;
            Ok(value)
        }
        value => Ok(value),
    }
}
fn children(actual: ipc::Field<'_>, expected: &DataType, walk: &mut Walk<'_>) -> Result<()> {
    let values = actual.children().ok_or_else(format)?;
    let mut actual = values.iter();
    visit_children(expected, &mut |expected| {
        field(actual.next().ok_or_else(format)?, expected, walk)
    })?;
    require(actual.next().is_none())
}
fn visit_children<'a>(
    dtype: &'a DataType,
    visit: &mut dyn FnMut(&'a Field) -> Result<()>,
) -> Result<()> {
    use DataType as D;
    match dtype {
        D::List(child)
        | D::ListView(child)
        | D::LargeList(child)
        | D::LargeListView(child)
        | D::FixedSizeList(child, _)
        | D::Map(child, _) => visit(child),
        D::Struct(fields) => fields.iter().try_for_each(|field| visit(field)),
        D::Union(fields, _) => fields.iter().try_for_each(|(_, field)| visit(field)),
        D::RunEndEncoded(ends, values) => {
            visit(ends)?;
            visit(values)
        }
        D::Dictionary(_, values) => visit_children(values, visit),
        _ => Ok(()),
    }
}
fn dictionary(actual: ipc::Field<'_>, expected: &Field, walk: &mut Walk<'_>) -> Result<()> {
    match expected.data_type() {
        DataType::Dictionary(index, _) => {
            let actual = actual.dictionary().ok_or_else(format)?;
            dictionary_identity(actual, expected, index, walk)?;
            add(&mut walk.counts.dictionary_fields, 1)
        }
        _ => require(actual.dictionary().is_none()),
    }
}
fn dictionary_identity(
    actual: ipc::DictionaryEncoding<'_>,
    expected: &Field,
    index: &DataType,
    walk: &mut Walk<'_>,
) -> Result<()> {
    let id = i64::try_from(walk.counts.dictionary_fields).map_err(|_| format())?;
    require(actual.id() == id)?;
    require(Some(actual.isOrdered()) == expected.dict_is_ordered())?;
    require(actual.dictionaryKind() == ipc::DictionaryKind::DenseArray)?;
    require(integer(actual.indexType(), index))
}
fn metadata<'a>(
    len: usize,
    actual: impl Iterator<Item = ipc::KeyValue<'a>>,
    expected: &HashMap<String, String>,
    walk: &mut Walk<'_>,
) -> Result<()> {
    require(len == expected.len())?;
    let mut previous = None;
    for pair in actual {
        previous = Some(metadata_pair(pair, previous, expected, walk)?);
    }
    Ok(())
}
fn metadata_pair<'a>(
    pair: ipc::KeyValue<'a>,
    previous: Option<&str>,
    expected: &HashMap<String, String>,
    walk: &mut Walk<'_>,
) -> Result<&'a str> {
    let (key, value) = pair_strings(pair)?;
    metadata_order(previous, key, walk.check)?;
    metadata_value(key, value, expected, walk.check)?;
    add(&mut walk.counts.metadata_pairs, 1)?;
    metadata_bytes(key, value, &mut walk.counts)?;
    Ok(key)
}
fn pair_strings(pair: ipc::KeyValue<'_>) -> Result<(&str, &str)> {
    Ok((
        pair.key().ok_or_else(format)?,
        pair.value().ok_or_else(format)?,
    ))
}

fn metadata_bytes(key: &str, value: &str, counts: &mut SchemaCounts) -> Result<()> {
    add(&mut counts.metadata_string_bytes, key.len())?;
    add(&mut counts.metadata_string_bytes, value.len())
}
fn metadata_order(
    previous: Option<&str>,
    key: &str,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    if let Some(previous) = previous {
        require(compare(previous, key, check)? == Ordering::Less)?;
    }
    Ok(())
}
fn metadata_value(
    key: &str,
    value: &str,
    expected: &HashMap<String, String>,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<()> {
    for (expected_key, expected_value) in expected {
        check()?;
        if compare(key, expected_key, check)? == Ordering::Equal {
            return require(compare(value, expected_value, check)? == Ordering::Equal);
        }
    }
    Err(format())
}
fn compare(left: &str, right: &str, check: &mut dyn FnMut() -> Result<()>) -> Result<Ordering> {
    for (left, right) in left
        .as_bytes()
        .chunks(4096)
        .zip(right.as_bytes().chunks(4096))
    {
        check()?;
        let order = left.cmp(right);
        if order != Ordering::Equal {
            return Ok(order);
        }
    }
    Ok(left.len().cmp(&right.len()))
}
fn datatype(actual: ipc::Field<'_>, expected: &DataType, walk: &mut Walk<'_>) -> Result<()> {
    if let DataType::Timestamp(unit, zone) = expected {
        return timestamp(actual, *unit, zone.as_deref(), walk);
    }
    if let DataType::Union(fields, mode) = expected {
        return union(actual, fields, *mode, walk);
    }
    let simple =
        simple_type(expected).map(|tag| actual.type_type() == tag && actual.type_().is_some());
    require(
        simple
            .or_else(|| numeric(actual, expected))
            .or_else(|| temporal(actual, expected))
            .or_else(|| parameterized(actual, expected))
            .unwrap_or(false),
    )
}
fn simple_type(expected: &DataType) -> Option<ipc::Type> {
    simple_binary(expected)
        .or_else(|| simple_string(expected))
        .or_else(|| simple_list(expected))
}
fn simple_binary(expected: &DataType) -> Option<ipc::Type> {
    match expected {
        DataType::Null => Some(ipc::Type::Null),
        DataType::Boolean => Some(ipc::Type::Bool),
        DataType::Binary => Some(ipc::Type::Binary),
        DataType::LargeBinary => Some(ipc::Type::LargeBinary),
        DataType::BinaryView => Some(ipc::Type::BinaryView),
        _ => None,
    }
}
fn simple_string(expected: &DataType) -> Option<ipc::Type> {
    match expected {
        DataType::Utf8 => Some(ipc::Type::Utf8),
        DataType::LargeUtf8 => Some(ipc::Type::LargeUtf8),
        DataType::Utf8View => Some(ipc::Type::Utf8View),
        DataType::Struct(_) => Some(ipc::Type::Struct_),
        DataType::RunEndEncoded(..) => Some(ipc::Type::RunEndEncoded),
        _ => None,
    }
}
fn simple_list(expected: &DataType) -> Option<ipc::Type> {
    match expected {
        DataType::List(_) => Some(ipc::Type::List),
        DataType::ListView(_) => Some(ipc::Type::ListView),
        DataType::LargeList(_) => Some(ipc::Type::LargeList),
        DataType::LargeListView(_) => Some(ipc::Type::LargeListView),
        _ => None,
    }
}
fn numeric(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    use DataType as D;
    match expected {
        D::Int8 | D::Int16 | D::Int32 | D::Int64 | D::UInt8 | D::UInt16 | D::UInt32 | D::UInt64 => {
            Some(integer(actual.type_as_int(), expected))
        }
        D::Float16 | D::Float32 | D::Float64 => Some(
            actual
                .type_as_floating_point()
                .is_some_and(|value| value.precision() == precision(expected)),
        ),
        D::Decimal32(..) | D::Decimal64(..) | D::Decimal128(..) | D::Decimal256(..) => {
            Some(decimal(actual.type_as_decimal(), expected))
        }
        _ => None,
    }
}
fn integer(actual: Option<ipc::Int<'_>>, expected: &DataType) -> bool {
    integer_width(expected).is_some_and(|width| {
        actual.is_some_and(|value| {
            value.bitWidth() == width && value.is_signed() == expected.is_signed_integer()
        })
    })
}
fn integer_width(expected: &DataType) -> Option<i32> {
    use DataType as D;
    match expected {
        D::Int8 | D::UInt8 => Some(8),
        D::Int16 | D::UInt16 => Some(16),
        D::Int32 | D::UInt32 => Some(32),
        D::Int64 | D::UInt64 => Some(64),
        _ => None,
    }
}
fn precision(expected: &DataType) -> ipc::Precision {
    match expected {
        DataType::Float16 => ipc::Precision::HALF,
        DataType::Float32 => ipc::Precision::SINGLE,
        _ => ipc::Precision::DOUBLE,
    }
}
fn decimal(actual: Option<ipc::Decimal<'_>>, expected: &DataType) -> bool {
    use DataType as D;
    let (width, precision, scale) = match expected {
        D::Decimal32(p, s) => (32, *p, *s),
        D::Decimal64(p, s) => (64, *p, *s),
        D::Decimal128(p, s) => (128, *p, *s),
        D::Decimal256(p, s) => (256, *p, *s),
        _ => return false,
    };
    actual.is_some_and(|value| {
        value.bitWidth() == width
            && value.precision() == i32::from(precision)
            && value.scale() == i32::from(scale)
    })
}
fn temporal(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    use DataType as D;
    match expected {
        D::Date32 => Some(
            actual
                .type_as_date()
                .is_some_and(|value| value.unit() == ipc::DateUnit::DAY),
        ),
        D::Date64 => Some(
            actual
                .type_as_date()
                .is_some_and(|value| value.unit() == ipc::DateUnit::MILLISECOND),
        ),
        D::Time32(unit) | D::Time64(unit) => Some(time(actual.type_as_time(), *unit, expected)),
        D::Duration(unit) => Some(
            actual
                .type_as_duration()
                .is_some_and(|value| value.unit() == time_unit(*unit)),
        ),
        D::Interval(unit) => Some(
            actual
                .type_as_interval()
                .is_some_and(|value| value.unit() == interval_unit(*unit)),
        ),
        _ => None,
    }
}
fn time(actual: Option<ipc::Time<'_>>, unit: TimeUnit, expected: &DataType) -> bool {
    let width = if matches!(expected, DataType::Time32(_)) {
        32
    } else {
        64
    };
    actual.is_some_and(|value| value.unit() == time_unit(unit) && value.bitWidth() == width)
}
fn time_unit(unit: TimeUnit) -> ipc::TimeUnit {
    match unit {
        TimeUnit::Second => ipc::TimeUnit::SECOND,
        TimeUnit::Millisecond => ipc::TimeUnit::MILLISECOND,
        TimeUnit::Microsecond => ipc::TimeUnit::MICROSECOND,
        TimeUnit::Nanosecond => ipc::TimeUnit::NANOSECOND,
    }
}
fn interval_unit(unit: IntervalUnit) -> ipc::IntervalUnit {
    match unit {
        IntervalUnit::YearMonth => ipc::IntervalUnit::YEAR_MONTH,
        IntervalUnit::DayTime => ipc::IntervalUnit::DAY_TIME,
        IntervalUnit::MonthDayNano => ipc::IntervalUnit::MONTH_DAY_NANO,
    }
}
fn parameterized(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::FixedSizeBinary(size) => Some(
            actual
                .type_as_fixed_size_binary()
                .is_some_and(|value| value.byteWidth() == *size),
        ),
        DataType::FixedSizeList(_, size) => Some(
            actual
                .type_as_fixed_size_list()
                .is_some_and(|value| value.listSize() == *size),
        ),
        DataType::Map(_, sorted) => Some(
            actual
                .type_as_map()
                .is_some_and(|value| value.keysSorted() == *sorted),
        ),
        _ => None,
    }
}
fn timestamp(
    actual: ipc::Field<'_>,
    unit: TimeUnit,
    zone: Option<&str>,
    walk: &mut Walk<'_>,
) -> Result<()> {
    let actual = actual.type_as_timestamp().ok_or_else(format)?;
    require(actual.unit() == time_unit(unit))?;
    match (actual.timezone(), zone) {
        (Some(actual), Some(expected)) => {
            require(compare(actual, expected, walk.check)? == Ordering::Equal)?;
            add(&mut walk.counts.timezone_string_bytes, actual.len())
        }
        (None, None) => Ok(()),
        _ => Err(format()),
    }
}
fn union(
    actual: ipc::Field<'_>,
    expected: &UnionFields,
    mode: UnionMode,
    walk: &mut Walk<'_>,
) -> Result<()> {
    let actual = actual.type_as_union().ok_or_else(format)?;
    let mode = match mode {
        UnionMode::Sparse => ipc::UnionMode::Sparse,
        UnionMode::Dense => ipc::UnionMode::Dense,
    };
    require(actual.mode() == mode)?;
    let ids = actual.typeIds().ok_or_else(format)?;
    union_ids(ids.iter(), ids.len(), expected, walk)
}
fn union_ids(
    actual: impl Iterator<Item = i32>,
    length: usize,
    expected: &UnionFields,
    walk: &mut Walk<'_>,
) -> Result<()> {
    require(length == expected.len())?;
    for (actual, (expected, _)) in actual.zip(expected.iter()) {
        (walk.check)()?;
        require(actual == i32::from(expected))?;
    }
    add(&mut walk.counts.union_type_ids, length)
}

pub(super) fn dictionary_value<'a>(
    expected: &'a Schema,
    id: i64,
    check: &mut dyn FnMut() -> Result<()>,
) -> Result<Option<&'a DataType>> {
    check()?;
    let mut lookup = DictionaryLookup {
        target: id,
        next: 0,
        found: None,
        check,
    };
    for field in expected.fields() {
        lookup.field(field)?;
        if lookup.found.is_some() {
            break;
        }
    }
    (lookup.check)()?;
    Ok(lookup.found)
}
struct DictionaryLookup<'a, 'b> {
    target: i64,
    next: usize,
    found: Option<&'a DataType>,
    check: &'b mut dyn FnMut() -> Result<()>,
}
impl<'a> DictionaryLookup<'a, '_> {
    fn field(&mut self, field: &'a Field) -> Result<()> {
        if self.found.is_some() {
            return Ok(());
        }
        (self.check)()?;
        visit_children(field.data_type(), &mut |field| self.field(field))?;
        self.assign(field.data_type())
    }
    fn assign(&mut self, dtype: &'a DataType) -> Result<()> {
        if self.found.is_some() {
            return Ok(());
        }
        if let DataType::Dictionary(_, value) = dtype {
            self.value(value)?;
        }
        Ok(())
    }
    fn value(&mut self, value: &'a DataType) -> Result<()> {
        let id = i64::try_from(self.next).map_err(|_| format())?;
        add(&mut self.next, 1)?;
        if id == self.target {
            self.found = Some(value);
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
