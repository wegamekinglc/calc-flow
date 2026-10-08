use super::{
    PreparedSides, RestoreBasesConstruction, RestoreBasesDecision, RestoreBasesWork, input,
    inspector,
};
use crate::StateSegment;
use crate::operator::join::{
    StoredRow, StreamJoinState,
    columnar::{self, FramedKey},
};
use crate::time::EventTime;
use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field, FieldRef, Schema},
};
use datafusion::execution::memory_pool::MemoryReservation;

#[derive(Clone, Copy)]
pub(super) struct Bounds {
    pub input: usize,
    pub workspace: usize,
    pub resident: [usize; 2],
}

pub(super) fn required(
    geometry: &input::Geometry,
    schemas: [&Schema; 2],
    keys: [&[usize]; 2],
    name: &str,
) -> Option<Bounds> {
    let registration = super::super::super::inventory::registration_controls()?;
    let resident = [
        columnar::restored::required(schemas[0], registration)?,
        columnar::restored::required(schemas[1], registration)?,
    ];
    let input = input_bytes(geometry, keys, name)?;
    let workspace = workspace_bytes(geometry, schemas, keys, name)?;
    std::alloc::Layout::from_size_align(input, align_of::<usize>()).ok()?;
    std::alloc::Layout::from_size_align(workspace, align_of::<usize>()).ok()?;
    Some(Bounds {
        input,
        workspace,
        resident,
    })
}

fn workspace_bytes(
    geometry: &input::Geometry,
    schemas: [&Schema; 2],
    keys: [&[usize]; 2],
    name: &str,
) -> Option<usize> {
    let fold = side_fold(geometry.rows[0], schemas[0], keys[0])?.checked_add(side_fold(
        geometry.rows[1],
        schemas[1],
        keys[1],
    )?)?;
    let workspace = checked_sum(&[
        reader_workspace(geometry, schemas)?,
        fold,
        tail_workspace(geometry, schemas, keys)?,
        diagnostic_bytes(name)?,
        size_of::<RestoreBasesWork>(),
        size_of::<RestoreBasesDecision>(),
        size_of::<PreparedSides>(),
        size_of::<MemoryReservation>(),
    ])?;
    Some(workspace)
}

fn reader_workspace(geometry: &input::Geometry, schemas: [&Schema; 2]) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let wire = geometry.bytes[0].checked_add(geometry.bytes[1])?;
    checked_sum(&[
        inspector::trace_bytes()?,
        bulk_bytes_peak(wire)?,
        wire.checked_add(rows.checked_mul(2 * 63)?)?,
        decoder_bytes(geometry, schemas)?,
    ])
}

fn tail_workspace(
    geometry: &input::Geometry,
    schemas: [&Schema; 2],
    keys: [&[usize]; 2],
) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let validation_key = bulk_bytes_peak(key_bytes(schemas[0], keys[0])?)?
        .max(bulk_bytes_peak(key_bytes(schemas[1], keys[1])?)?);
    installation_bytes(rows, geometry.rows)?.checked_add(validation_key)
}

fn input_bytes(geometry: &input::Geometry, keys: [&[usize]; 2], name: &str) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let key_count = keys[0].len().checked_add(keys[1].len())?;
    checked_sum(&[
        geometry.bytes[0],
        geometry.bytes[1],
        rows.checked_mul(size_of::<columnar::restored::ResidentLease>())?,
        key_count.checked_mul(size_of::<usize>())?,
        name.len(),
        tree::<String, StateSegment>(2)?,
        "left-base".len() + "right-base".len() + 2 * 64,
        arc::<Vec<u8>>()?.checked_mul(2)?,
        size_of::<input::OwnedInput>(),
        size_of::<RestoreBasesConstruction>(),
        size_of::<RestoreBasesWork>(),
    ])
}

fn decoder_bytes(geometry: &input::Geometry, schemas: [&Schema; 2]) -> Option<usize> {
    let mut sum = 0_usize;
    for (side, schema) in schemas.into_iter().enumerate() {
        sum = sum.checked_add(geometry.rows[side].checked_mul(reader_row_bytes(schema)?)?)?;
    }
    Some(sum)
}

fn reader_schema_bytes(schema: &Schema) -> Option<usize> {
    let fields = schema.fields().len();
    checked_sum(&[
        columnar::schema_inventory(schema)?,
        vector_peak::<Field>(fields)?,
        vector_peak::<FieldRef>(fields)?,
    ])
}

pub(in crate::operator::join::metadata_validation::schema) fn reader_row_bytes(
    schema: &Schema,
) -> Option<usize> {
    let fields = schema.fields().len();
    let arrays = fields.checked_mul(columnar::restored::column_controls()?)?;
    checked_sum(&[
        reader_schema_bytes(schema)?,
        arrays.checked_mul(2)?,
        vector_peak::<ArrayRef>(fields)?.checked_mul(2)?,
    ])
}

pub(in crate::operator::join::metadata_validation::schema) fn side_fold(
    rows: usize,
    schema: &Schema,
    indices: &[usize],
) -> Option<usize> {
    let keys = folded_key_bytes(schema, indices)?;
    checked_sum(&[
        rows.checked_mul(keys)?,
        rows.checked_mul(4 * size_of::<StoredRow>())?,
        tree::<(Vec<u8>, i64, u64), StoredRow>(rows)?,
        tree::<(EventTime, u64), ()>(rows)?,
        // Original key_value_bytes allocates at most one eight-byte scalar temporary.
        8,
    ])
}

fn folded_key_bytes(schema: &Schema, indices: &[usize]) -> Option<usize> {
    let key = key_bytes(schema, indices)?;
    checked_sum(&[bulk_bytes_peak(key)?, key, arc::<FramedKey>()?])
}

pub(in crate::operator::join::metadata_validation::schema) fn key_bytes(
    schema: &Schema,
    indices: &[usize],
) -> Option<usize> {
    indices.iter().try_fold(0_usize, |total, index| {
        let field = schema.fields().get(*index)?;
        let timezone = match field.data_type() {
            DataType::Timestamp(_, timezone) => timezone.as_ref().map_or(0, |text| text.len()),
            _ => 0,
        };
        total
            .checked_add(9)?
            .checked_add(timezone)?
            .checked_add(columnar::restored::key_width(field.data_type())?)
    })
}

pub(in crate::operator::join::metadata_validation::schema) fn bulk_bytes_peak(
    bytes: usize,
) -> Option<usize> {
    if bytes == 0 {
        return Some(0);
    }
    // Bulk growth keeps old capacity below required length; new capacity is at most twice it.
    bytes.checked_mul(3).map(|peak| peak.max(8))
}

pub(in crate::operator::join::metadata_validation::schema) fn installation_bytes(
    rows: usize,
    sides: [usize; 2],
) -> Option<usize> {
    let indexes = expiration_bytes(sides[0])?.checked_add(expiration_bytes(sides[1])?)?;
    checked_sum(&[
        indexes,
        arc::<Vec<StoredRow>>()?.checked_mul(2)?,
        rows.checked_mul(size_of::<StoredRow>())?,
        tree::<&'static str, StateSegment>(2)?,
        "left-base".len() + "right-base".len() + 2 * 64,
        4 * size_of::<(&str, crate::operator::join::SegmentKind)>(),
        8 * size_of::<(&str, &crate::operator::join::SegmentKind, &StateSegment)>(),
        2 * "right-base".len(),
        size_of::<StreamJoinState>(),
    ])
}

fn expiration_bytes(rows: usize) -> Option<usize> {
    type Entry = ((EventTime, u64), (usize, u128));
    tree::<(EventTime, u64), (usize, u128)>(rows)?
        .checked_add(rows.checked_mul(2 * size_of::<Entry>())?)
}

pub(in crate::operator::join::metadata_validation::schema) fn diagnostic_bytes(
    name: &str,
) -> Option<usize> {
    checked_sum(&[
        checkpoint_diagnostic(name)?,
        time_diagnostic()?,
        reason_diagnostic(name)?,
    ])
}

fn checkpoint_diagnostic(name: &str) -> Option<usize> {
    let templates = [
        "stream Join   IPC must contain one row",
        "stream Join   IPC contains extra record batches",
        "stream Join   row identity is invalid",
        "stream Join   row event time or charge is inconsistent",
        "stream Join  restored state charge is inconsistent",
        "stream Join   row count is too large",
        "stream Join   byte charge overflowed",
        "stream Join  restored  state exceeds configured limits",
    ];
    let literal = templates.iter().map(|text| text.len()).max()?;
    let debug_name = name.len().checked_mul(6)?.checked_add(2)?;
    let output = literal.checked_add(5)?.checked_add(debug_name)?;
    formatted_peak(output, literal)
}

fn time_diagnostic() -> Option<usize> {
    let column = "stream_join.right_event_time".len();
    checked_sum(&[
        formatted_peak(column, "stream_join._event_time".len())?,
        column,
        "timestamp value overflows the microsecond event-time range".len(),
    ])
}

fn reason_diagnostic(name: &str) -> Option<usize> {
    let messages = [
        "right event time cannot be represented",
        "retained rows counter overflowed",
        "retained bytes counter overflowed",
        "encoded key counter overflowed",
        "logical payload bytes counter overflowed",
        "state row charge counter overflowed",
        "segments counter overflowed",
    ];
    let message = messages.iter().map(|text| text.len()).max()?;
    checked_sum(&[formatted_peak(message, message)?, message, name.len()])
}

pub(in crate::operator::join::metadata_validation::schema) fn formatted_peak(
    output: usize,
    literals: usize,
) -> Option<usize> {
    let initial = literals.checked_mul(2)?;
    output.checked_mul(3).map(|peak| peak.max(initial))
}

pub(super) fn checked_sum(parts: &[usize]) -> Option<usize> {
    parts
        .iter()
        .try_fold(0_usize, |sum, part| sum.checked_add(*part))
}

pub(in crate::operator::join::metadata_validation::schema) fn vector_peak<T>(
    count: usize,
) -> Option<usize> {
    if count == 0 || size_of::<T>() == 0 {
        return Some(0);
    }
    let minimum = match size_of::<T>() {
        1 => 8,
        2..=1024 => 4,
        _ => 1,
    };
    let capacity = count.max(minimum).checked_next_power_of_two()?;
    capacity
        .checked_add(capacity / 2)?
        .checked_mul(size_of::<T>())
}

pub(in crate::operator::join::metadata_validation::schema) fn arc<T>() -> Option<usize> {
    let header = align_up(2 * size_of::<usize>(), align_of::<T>())?;
    align_up(
        header.checked_add(size_of::<T>())?,
        align_of::<T>().max(align_of::<usize>()),
    )
}

fn align_up(bytes: usize, alignment: usize) -> Option<usize> {
    bytes
        .checked_add(alignment - 1)?
        .checked_div(alignment)?
        .checked_mul(alignment)
}

pub(in crate::operator::join::metadata_validation::schema) fn tree<K, V>(
    count: usize,
) -> Option<usize> {
    let alignment = align_of::<K>()
        .max(align_of::<V>())
        .max(align_of::<usize>());
    let fields = size_of::<usize>() + 2 * size_of::<u16>() + 12 * size_of::<usize>();
    let slots = 11_usize.checked_mul(size_of::<K>().checked_add(size_of::<V>())?)?;
    let node = align_up(
        fields
            .checked_add(slots)?
            .checked_add(6 * (alignment - 1))?,
        alignment,
    )?;
    count.checked_add(1)?.checked_mul(node)
}
