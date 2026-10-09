use std::collections::HashMap;

use datafusion::arrow::{
    datatypes::{DataType, Field, IntervalUnit, Schema, TimeUnit, UnionMode},
    ipc,
};
use flatbuffers::{ForwardsUOffset, Vector};

use super::super::geometry::invalid;
use crate::Result;

pub(super) fn validate(
    actual: ipc::Schema<'_>,
    expected: &Schema,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    if !actual.endianness().equals_to_target_endianness() {
        return Err(invalid("V2 IPC schema endianness differs from this host"));
    }
    metadata(actual.custom_metadata(), expected.metadata(), check)?;
    let fields = Some(
        actual
            .fields()
            .ok_or_else(|| invalid("V2 IPC schema fields are missing"))?,
    );
    if fields.map_or(0, |fields| fields.len()) != expected.fields().len() {
        return Err(invalid(
            "V2 IPC schema field count differs from the input schema",
        ));
    }
    for (actual, expected) in fields.into_iter().flatten().zip(expected.fields()) {
        field(actual, expected, check)?;
    }
    check()
}

fn field(actual: ipc::Field<'_>, expected: &Field, check: &dyn Fn() -> Result<()>) -> Result<()> {
    check()?;
    if actual.name() != Some(expected.name().as_str())
        || actual.nullable() != expected.is_nullable()
    {
        return Err(invalid(
            "V2 IPC field name or nullability differs from the input schema",
        ));
    }
    metadata(actual.custom_metadata(), expected.metadata(), check)?;
    let data_type = dictionary_value(actual, expected.data_type())?;
    if !type_matches(actual, data_type) {
        return Err(invalid("V2 IPC field type differs from the input schema"));
    }
    children(actual, data_type, check)
}

fn dictionary_value<'a>(actual: ipc::Field<'_>, expected: &'a DataType) -> Result<&'a DataType> {
    match (actual.dictionary(), expected) {
        (Some(dictionary), DataType::Dictionary(key, value)) => {
            let index = dictionary
                .indexType()
                .ok_or_else(|| invalid("V2 IPC dictionary index type is missing"))?;
            if !integer_matches(index.bitWidth(), index.is_signed(), key) {
                return Err(invalid(
                    "V2 IPC dictionary index type differs from the input schema",
                ));
            }
            Ok(value)
        }
        (None, DataType::Dictionary(..)) | (Some(_), _) => Err(invalid(
            "V2 IPC dictionary encoding differs from the input schema",
        )),
        (None, expected) => Ok(expected),
    }
}

fn metadata(
    actual: Option<Vector<'_, ForwardsUOffset<ipc::KeyValue<'_>>>>,
    expected: &HashMap<String, String>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    if actual.map_or(0, |entries| entries.len()) != expected.len() {
        return Err(invalid("V2 IPC metadata differs from the input schema"));
    }
    for (key, value) in expected {
        check()?;
        if metadata_matches(actual, key, value, check)? != 1 {
            return Err(invalid("V2 IPC metadata differs from the input schema"));
        }
    }
    Ok(())
}

fn metadata_matches(
    actual: Option<Vector<'_, ForwardsUOffset<ipc::KeyValue<'_>>>>,
    key: &str,
    value: &str,
    check: &dyn Fn() -> Result<()>,
) -> Result<usize> {
    let mut matches = 0;
    for entry in actual.into_iter().flatten() {
        check()?;
        if entry.key() == Some(key) && entry.value() == Some(value) {
            matches += 1;
        }
    }
    Ok(matches)
}

fn type_matches(actual: ipc::Field<'_>, expected: &DataType) -> bool {
    primitive(actual, expected)
        .or_else(|| variable(actual, expected))
        .or_else(|| numeric(actual, expected))
        .or_else(|| temporal(actual, expected))
        .or_else(|| list(actual, expected))
        .or_else(|| aggregate(actual, expected))
        .unwrap_or(false)
}

fn primitive(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Null => Some(actual.type_as_null().is_some()),
        DataType::Boolean => Some(actual.type_as_bool().is_some()),
        expected if integer_width(expected).is_some() => {
            Some(actual.type_as_int().is_some_and(|integer| {
                integer_matches(integer.bitWidth(), integer.is_signed(), expected)
            }))
        }
        _ => None,
    }
}

fn integer_matches(width: i32, signed: bool, expected: &DataType) -> bool {
    integer_width(expected) == Some(width)
        && signed
            == matches!(
                expected,
                DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64
            )
}

fn integer_width(data_type: &DataType) -> Option<i32> {
    match data_type {
        DataType::Int8 | DataType::UInt8 => Some(8),
        DataType::Int16 | DataType::UInt16 => Some(16),
        DataType::Int32 | DataType::UInt32 => Some(32),
        DataType::Int64 | DataType::UInt64 => Some(64),
        _ => None,
    }
}

fn variable(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Binary => Some(actual.type_as_binary().is_some()),
        DataType::LargeBinary => Some(actual.type_as_large_binary().is_some()),
        DataType::Utf8 => Some(actual.type_as_utf_8().is_some()),
        DataType::LargeUtf8 => Some(actual.type_as_large_utf_8().is_some()),
        DataType::BinaryView => Some(actual.type_as_binary_view().is_some()),
        DataType::Utf8View => Some(actual.type_as_utf_8_view().is_some()),
        _ => None,
    }
}

fn numeric(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Float16 => Some(float(actual, ipc::Precision::HALF)),
        DataType::Float32 => Some(float(actual, ipc::Precision::SINGLE)),
        DataType::Float64 => Some(float(actual, ipc::Precision::DOUBLE)),
        DataType::FixedSizeBinary(width) => Some(
            actual
                .type_as_fixed_size_binary()
                .is_some_and(|binary| binary.byteWidth() == *width && *width >= 0),
        ),
        expected => decimal(actual, expected),
    }
}

fn float(actual: ipc::Field<'_>, expected: ipc::Precision) -> bool {
    actual
        .type_as_floating_point()
        .is_some_and(|float| float.precision() == expected)
}

fn decimal(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    let (width, precision, scale) = match expected {
        DataType::Decimal32(precision, scale) => (32, *precision, *scale),
        DataType::Decimal64(precision, scale) => (64, *precision, *scale),
        DataType::Decimal128(precision, scale) => (128, *precision, *scale),
        DataType::Decimal256(precision, scale) => (256, *precision, *scale),
        _ => return None,
    };
    Some(actual.type_as_decimal().is_some_and(|decimal| {
        decimal.bitWidth() == width
            && decimal.precision() == i32::from(precision)
            && decimal.scale() == i32::from(scale)
    }))
}

fn temporal(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Date32 => Some(
            actual
                .type_as_date()
                .is_some_and(|date| date.unit() == ipc::DateUnit::DAY),
        ),
        DataType::Date64 => Some(
            actual
                .type_as_date()
                .is_some_and(|date| date.unit() == ipc::DateUnit::MILLISECOND),
        ),
        DataType::Timestamp(unit, timezone) => {
            Some(actual.type_as_timestamp().is_some_and(|timestamp| {
                timestamp.unit() == time_unit(*unit) && timestamp.timezone() == timezone.as_deref()
            }))
        }
        DataType::Duration(unit) => Some(
            actual
                .type_as_duration()
                .is_some_and(|duration| duration.unit() == time_unit(*unit)),
        ),
        expected => time_or_interval(actual, expected),
    }
}

fn time_or_interval(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Time32(unit) => Some(
            matches!(unit, TimeUnit::Second | TimeUnit::Millisecond) && time(actual, *unit, 32),
        ),
        DataType::Time64(unit) => Some(
            matches!(unit, TimeUnit::Microsecond | TimeUnit::Nanosecond) && time(actual, *unit, 64),
        ),
        DataType::Interval(unit) => Some(actual.type_as_interval().is_some_and(|interval| {
            interval.unit()
                == match unit {
                    IntervalUnit::YearMonth => ipc::IntervalUnit::YEAR_MONTH,
                    IntervalUnit::DayTime => ipc::IntervalUnit::DAY_TIME,
                    IntervalUnit::MonthDayNano => ipc::IntervalUnit::MONTH_DAY_NANO,
                }
        })),
        _ => None,
    }
}

fn time(actual: ipc::Field<'_>, unit: TimeUnit, width: i32) -> bool {
    actual
        .type_as_time()
        .is_some_and(|time| time.bitWidth() == width && time.unit() == time_unit(unit))
}

fn time_unit(unit: TimeUnit) -> ipc::TimeUnit {
    match unit {
        TimeUnit::Second => ipc::TimeUnit::SECOND,
        TimeUnit::Millisecond => ipc::TimeUnit::MILLISECOND,
        TimeUnit::Microsecond => ipc::TimeUnit::MICROSECOND,
        TimeUnit::Nanosecond => ipc::TimeUnit::NANOSECOND,
    }
}

fn list(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::List(_) => Some(actual.type_as_list().is_some()),
        DataType::LargeList(_) => Some(actual.type_as_large_list().is_some()),
        DataType::ListView(_) => Some(actual.type_as_list_view().is_some()),
        DataType::LargeListView(_) => Some(actual.type_as_large_list_view().is_some()),
        DataType::FixedSizeList(_, size) => Some(
            actual
                .type_as_fixed_size_list()
                .is_some_and(|list| list.listSize() == *size && *size >= 0),
        ),
        _ => None,
    }
}

fn aggregate(actual: ipc::Field<'_>, expected: &DataType) -> Option<bool> {
    match expected {
        DataType::Struct(_) => Some(actual.type_as_struct_().is_some()),
        DataType::Map(_, sorted) => Some(
            actual
                .type_as_map()
                .is_some_and(|map| map.keysSorted() == *sorted),
        ),
        DataType::RunEndEncoded(_, _) => Some(actual.type_as_run_end_encoded().is_some()),
        DataType::Union(fields, mode) => Some(actual.type_as_union().is_some_and(|union| {
            let actual_ids = union.typeIds();
            union.mode() == union_mode(*mode)
                && actual_ids.map_or(0, |ids| ids.len()) == fields.len()
                && actual_ids
                    .into_iter()
                    .flatten()
                    .zip(fields.iter())
                    .all(|(actual, (expected, _))| actual == i32::from(expected))
        })),
        _ => None,
    }
}

fn union_mode(mode: UnionMode) -> ipc::UnionMode {
    match mode {
        UnionMode::Sparse => ipc::UnionMode::Sparse,
        UnionMode::Dense => ipc::UnionMode::Dense,
    }
}

fn children(
    actual: ipc::Field<'_>,
    expected: &DataType,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    let actual = actual.children();
    let expected_count = child_count(expected);
    if actual.map_or(0, |children| children.len()) != expected_count {
        return Err(invalid("V2 IPC child count differs from the input schema"));
    }
    for (index, actual) in actual.into_iter().flatten().enumerate() {
        let expected =
            child_field(expected, index).ok_or_else(|| invalid("V2 IPC child type is invalid"))?;
        field(actual, expected, check)?;
    }
    Ok(())
}

fn child_count(data_type: &DataType) -> usize {
    if unary_child(data_type).is_some() {
        return 1;
    }
    match data_type {
        DataType::Struct(fields) => fields.len(),
        DataType::Union(fields, _) => fields.len(),
        DataType::RunEndEncoded(_, _) => 2,
        _ => 0,
    }
}

fn child_field(data_type: &DataType, index: usize) -> Option<&Field> {
    if index == 0
        && let Some(field) = unary_child(data_type)
    {
        return Some(field);
    }
    match data_type {
        DataType::Struct(fields) => fields.get(index).map(AsRef::as_ref),
        DataType::Union(fields, _) => fields.iter().nth(index).map(|(_, field)| field.as_ref()),
        DataType::RunEndEncoded(run_ends, values) => {
            [run_ends.as_ref(), values.as_ref()].get(index).copied()
        }
        _ => None,
    }
}

fn unary_child(data_type: &DataType) -> Option<&Field> {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => Some(field),
        _ => None,
    }
}

pub(super) fn dictionary_type<'a>(
    actual: ipc::Schema<'_>,
    expected: &'a Schema,
    id: i64,
    check: &dyn Fn() -> Result<()>,
) -> Result<&'a DataType> {
    let mut value = None;
    for (actual, expected) in actual
        .fields()
        .expect("validated schema fields")
        .iter()
        .zip(expected.fields())
    {
        visit_dictionary(actual, expected, id, &mut value, check)?;
    }
    value.ok_or_else(|| invalid("V2 IPC dictionary ID is not declared by its schema"))
}

fn visit_dictionary<'a>(
    actual: ipc::Field<'_>,
    expected: &'a Field,
    id: i64,
    value: &mut Option<&'a DataType>,
    check: &dyn Fn() -> Result<()>,
) -> Result<()> {
    check()?;
    let data_type = dictionary_value(actual, expected.data_type())?;
    if actual
        .dictionary()
        .is_some_and(|dictionary| dictionary.id() == id)
    {
        if value.is_some_and(|previous| previous != data_type) {
            return Err(invalid("V2 IPC dictionary ID has inconsistent value types"));
        }
        *value = Some(data_type);
    }
    for (index, actual) in actual.children().into_iter().flatten().enumerate() {
        let expected = child_field(data_type, index).expect("validated child field");
        visit_dictionary(actual, expected, id, value, check)?;
    }
    Ok(())
}
