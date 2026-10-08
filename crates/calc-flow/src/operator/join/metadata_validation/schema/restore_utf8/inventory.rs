use super::super::restore_bases::{inspector as scalar_inspector, inventory as base};
use super::{
    PreparedSides, RestoreUtf8Construction, RestoreUtf8Decision, RestoreUtf8Work, frame, input,
    inspector,
};
use crate::StateSegment;
use crate::operator::join::{StoredRow, StreamJoinOperator, columnar};
use crate::time::EventTime;
use datafusion::arrow::datatypes::DataType;
use datafusion::arrow::datatypes::Schema;
use datafusion::execution::memory_pool::MemoryReservation;

#[derive(Clone, Copy)]
pub(super) struct Bounds {
    pub input: usize,
    pub workspace: usize,
}

pub(super) fn required(
    geometry: &frame::Geometry,
    deltas: usize,
    schemas: [&Schema; 2],
    keys: [&[usize]; 2],
    name: &str,
) -> Option<Bounds> {
    let input = input_bytes(geometry, keys, name)?;
    let workspace = workspace_bytes(geometry, deltas, schemas, keys, name)?;
    std::alloc::Layout::from_size_align(input, align_of::<usize>()).ok()?;
    std::alloc::Layout::from_size_align(workspace, align_of::<usize>()).ok()?;
    Some(Bounds { input, workspace })
}

fn input_bytes(geometry: &frame::Geometry, keys: [&[usize]; 2], name: &str) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let key_count = keys[0].len().checked_add(keys[1].len())?;
    sum(&[
        Some(geometry.wire[0]),
        Some(geometry.wire[1]),
        rows.checked_mul(size_of::<columnar::restored::ResidentLease>()),
        key_count.checked_mul(size_of::<usize>()),
        Some(name.len()),
        geometry
            .segments
            .checked_mul(size_of::<(String, Vec<u8>)>()),
        Some(geometry.id_bytes),
        base::tree::<String, StateSegment>(geometry.segments),
        geometry.segments.checked_mul(64),
        base::arc::<Vec<u8>>().and_then(|bytes| bytes.checked_mul(geometry.segments)),
        Some(size_of::<input::OwnedInput>()),
        Some(size_of::<RestoreUtf8Construction>()),
        Some(size_of::<RestoreUtf8Work>()),
    ])
}

fn workspace_bytes(
    geometry: &frame::Geometry,
    deltas: usize,
    schemas: [&Schema; 2],
    _keys: [&[usize]; 2],
    name: &str,
) -> Option<usize> {
    sum(&[
        reader_workspace(geometry, schemas),
        side_containers(geometry.rows[0]),
        side_containers(geometry.rows[1]),
        delta_workspace(geometry),
        tail_workspace(geometry, deltas),
        base::diagnostic_bytes(name),
        delta_diagnostics(name),
        Some(size_of::<RestoreUtf8Work>()),
        Some(size_of::<RestoreUtf8Decision>()),
        Some(size_of::<PreparedSides>()),
        Some(size_of::<MemoryReservation>()),
    ])
}

fn reader_workspace(geometry: &frame::Geometry, schemas: [&Schema; 2]) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let wire = geometry.wire[0].checked_add(geometry.wire[1])?;
    sum(&[
        scalar_inspector::trace_bytes(),
        base::bulk_bytes_peak(wire),
        rows.checked_mul(2 * 63)
            .and_then(|bytes| wire.checked_add(bytes)),
        base::reader_row_bytes(schemas[0]).and_then(|bytes| geometry.rows[0].checked_mul(bytes)),
        base::reader_row_bytes(schemas[1]).and_then(|bytes| geometry.rows[1].checked_mul(bytes)),
    ])
}

fn delta_workspace(geometry: &frame::Geometry) -> Option<usize> {
    sum(&[
        geometry.header_keys.checked_mul(2),
        Some(geometry.seen_keys),
        geometry.longest_key.checked_mul(2),
        base::tree::<(Vec<u8>, i64, u64), ()>(geometry.seen_count),
        base::vector_peak::<(&str, crate::operator::join::SegmentKind)>(geometry.segments),
        base::vector_peak::<(&str, &crate::operator::join::SegmentKind, &StateSegment)>(
            geometry.segments,
        ),
        geometry.segments.checked_mul(size_of::<(
            &str,
            &crate::operator::join::SegmentKind,
            &StateSegment,
        )>()),
        geometry.longest_id.checked_mul(2),
    ])
}

fn tail_workspace(geometry: &frame::Geometry, deltas: usize) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    sum(&[
        base::installation_bytes(rows, geometry.rows),
        base::tree::<(u64, &'static str), StateSegment>(deltas),
        deltas.checked_mul(64),
        base::vector_peak::<(&str, crate::operator::join::SegmentKind)>(geometry.segments),
        base::vector_peak::<(&str, &crate::operator::join::SegmentKind, &StateSegment)>(
            geometry.segments,
        ),
        geometry.longest_id.checked_mul(2),
    ])
}

fn delta_diagnostics(name: &str) -> Option<usize> {
    let templates = [
        "stream Join   delta segment repeats one row identity",
        "stream Join   delta upsert key does not match its record",
        "stream Join   delta segment has trailing bytes",
        "stream Join   delta magic is invalid",
        "stream Join   delta op count is invalid",
        "stream Join   delta key length is invalid",
        "stream Join   delta op tag is invalid",
        "stream Join  segment id is invalid",
    ];
    let literal = templates.iter().map(|text| text.len()).max()?;
    let debug_name = name.len().checked_mul(6)?.checked_add(2)?;
    let output = literal.checked_add(5)?.checked_add(debug_name)?;
    base::formatted_peak(output, literal)
}

fn sum(parts: &[Option<usize>]) -> Option<usize> {
    parts
        .iter()
        .try_fold(0_usize, |sum, part| sum.checked_add((*part)?))
}

pub(super) fn eligible(operator: &StreamJoinOperator) -> bool {
    let schemas = [operator.input_schema(0), operator.input_schema(1)];
    if !schemas.iter().any(|schema| {
        schema
            .fields()
            .iter()
            .any(|field| field.data_type() == &DataType::Utf8)
    }) {
        return false;
    }
    schemas.iter().all(|schema| {
        schema.fields().iter().all(|field| {
            field.data_type() == &DataType::Utf8
                || columnar::restored::width(field.data_type()).is_some()
        })
    }) && keys_eligible(operator)
}

fn keys_eligible(operator: &StreamJoinOperator) -> bool {
    let indices = [
        &operator.compiled.left_key_indices,
        &operator.compiled.right_key_indices,
    ];
    indices.into_iter().enumerate().all(|(side, indices)| {
        indices.iter().all(|index| {
            let data_type = operator.input_schema(side).field(*index).data_type();
            data_type == &DataType::Utf8 || columnar::restored::key_width(data_type).is_some()
        })
    })
}

fn side_containers(rows: usize) -> Option<usize> {
    sum(&[
        rows.checked_mul(4 * size_of::<StoredRow>()),
        base::tree::<(Vec<u8>, i64, u64), StoredRow>(rows),
        base::tree::<(EventTime, u64), ()>(rows),
    ])
}

pub(super) fn dynamic_workspace(
    geometry: &frame::Geometry,
    facts: &[inspector::Facts; 2],
) -> Option<usize> {
    let rows = geometry.rows[0].checked_add(geometry.rows[1])?;
    let keys = facts[0].key_total.checked_add(facts[1].key_total)?;
    let maximum = facts[0].key_max.max(facts[1].key_max);
    let value = facts[0].value_max.max(facts[1].value_max);
    sum(&[
        keys.checked_mul(4),
        rows.checked_mul(base::arc::<columnar::FramedKey>()?),
        base::bulk_bytes_peak(maximum),
        value.checked_mul(2),
    ])
}
