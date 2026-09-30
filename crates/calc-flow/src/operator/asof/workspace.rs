use super::{StreamAsofJoinOperator, checked, reason};
use crate::{Batch, CalcFlowError, Result, StreamingFailureReason};
use datafusion::{
    arrow::{
        array::{ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray},
        datatypes::{DataType, Schema},
        record_batch::RecordBatch,
    },
    common::ScalarValue,
    execution::memory_pool::{MemoryConsumer, MemoryReservation},
};

/// Headroom for one identity row: the two owned encodings, the ordered-map
/// node that will hold them, and row-converter scratch.
const IDENTITY_ROW_BYTES: u64 = 384;
/// Fixed schema-message headroom: stream envelope, version, flags, padding.
const SCHEMA_ENVELOPE_BYTES: u64 = 256;
/// Per-field headroom over `Field::size()`: the field table, type union,
/// nullability, and alignment inside the schema flatbuffer.
const SCHEMA_FIELD_BYTES: u64 = 192;
/// Per-entry headroom for schema custom metadata inside the flatbuffer.
const SCHEMA_METADATA_ENTRY_BYTES: u64 = 64;
/// Field-node and buffer-directory entries for one column in a legacy row
/// record message, plus IPC alignment slack on top of the slice bytes.
#[cfg(test)]
const COLUMN_FRAMING_BYTES: u64 = 48;
/// Legacy per-row record-message envelope: continuation, metadata length, message
/// flatbuffer padding, and the end-of-stream marker.
#[cfg(test)]
const ROW_FRAMING_BYTES: u64 = 96;
/// Framing headroom for one fixed-width identity encoding on top of the
/// aligned slice: the Arrow row format adds one non-null marker byte to the
/// value width.
const FIXED_IDENTITY_FRAMING_BYTES: u64 = 16;
/// Framing headroom for one string identity encoding on top of the 33/32
/// slice scaling: Arrow's blocked string row format (`identity_compare.rs`)
/// adds one marker byte, four 8-byte block sentinels and final-block
/// rounding beyond the scaled bytes.
const STRING_IDENTITY_FRAMING_BYTES: u64 = 64;

/// Allocation-free upper bound on one batch's IPC schema message. A batch
/// encoding writes this schema once, independent of its accepted row count.
pub(super) struct PayloadCharge {
    schema_bytes: u64,
}

/// The fixed part of Arrow's one-row slice charge is identical for every row
/// in a flat array. Only variable-width values need a row-specific length.
pub(super) struct ColumnWorkspace {
    column: ArrayRef,
    fixed: u64,
}

impl ColumnWorkspace {
    pub(super) fn new(column: ArrayRef) -> Result<Self> {
        let fixed = if column.is_empty() {
            0
        } else {
            column_workspace(&column, 0)?
                .checked_sub(variable_length(&column, 0)?)
                .ok_or_else(|| CalcFlowError::Format {
                    message: "ASOF variable value exceeds Arrow slice workspace".into(),
                })?
        };
        Ok(Self { column, fixed })
    }

    pub(super) fn range_bytes(&self, range: std::ops::Range<usize>, name: &str) -> Result<u64> {
        let variable = variable_range_length(&self.column, range.clone())?;
        let bytes = self
            .fixed
            .checked_mul((range.end - range.start) as u64)
            .and_then(|fixed| fixed.checked_add(variable));
        bytes.ok_or_else(|| {
            reason(
                name,
                StreamingFailureReason::AsofCounterOverflow,
                "ASOF column workspace arithmetic overflowed",
            )
        })
    }

    pub(super) fn bytes(&self, row: usize, name: &str) -> Result<u64> {
        self.fixed
            .checked_add(variable_length(&self.column, row)?)
            .ok_or_else(|| {
                reason(
                    name,
                    StreamingFailureReason::AsofCounterOverflow,
                    "ASOF column workspace arithmetic overflowed",
                )
            })
    }
}

fn variable_range_length(column: &ArrayRef, range: std::ops::Range<usize>) -> Result<u64> {
    let length = match column.data_type() {
        DataType::Utf8 => {
            let values = column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("validated Arrow type");
            i64::from(values.value_offsets()[range.end] - values.value_offsets()[range.start])
        }
        DataType::LargeUtf8 => {
            let values = column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("validated Arrow type");
            values.value_offsets()[range.end] - values.value_offsets()[range.start]
        }
        DataType::Binary => {
            let values = column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("validated Arrow type");
            i64::from(values.value_offsets()[range.end] - values.value_offsets()[range.start])
        }
        DataType::LargeBinary => {
            let values = column
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("validated Arrow type");
            values.value_offsets()[range.end] - values.value_offsets()[range.start]
        }
        _ => return Ok(0),
    };
    u64::try_from(length).map_err(|_| CalcFlowError::Format {
        message: "negative ASOF variable-length value".into(),
    })
}

fn variable_length(column: &ArrayRef, row: usize) -> Result<u64> {
    let length = match column.data_type() {
        DataType::Utf8 => column
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("validated Arrow type")
            .value_length(row)
            .into(),
        DataType::LargeUtf8 => column
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("validated Arrow type")
            .value_length(row),
        DataType::Binary => column
            .as_any()
            .downcast_ref::<BinaryArray>()
            .expect("validated Arrow type")
            .value_length(row)
            .into(),
        DataType::LargeBinary => column
            .as_any()
            .downcast_ref::<LargeBinaryArray>()
            .expect("validated Arrow type")
            .value_length(row),
        _ => return Ok(0),
    };
    u64::try_from(length).map_err(|_| CalcFlowError::Format {
        message: "negative ASOF variable-length value".into(),
    })
}

impl StreamAsofJoinOperator {
    pub(super) fn reserve_workspace(&self, bytes: u64) -> Result<MemoryReservation> {
        let bytes = usize::try_from(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF workspace exceeds address domain",
            )
        })?;
        let reservation = MemoryConsumer::new("asof-owned-workspace").register(&self.runtime.pool);
        reservation.try_grow(bytes).map_err(|_| {
            reason(
                &self.name,
                StreamingFailureReason::AsofWorkspaceLimitExceeded,
                "ASOF aggregate workspace exceeds max_state_bytes",
            )
        })?;
        Ok(reservation)
    }

    pub(super) fn input_workspace(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
    ) -> Result<MemoryReservation> {
        let side = input.side(&self.spec);
        let mut bytes = 0;
        for record in batch.table_payload()?.batches() {
            if record.num_rows() == 0 {
                continue;
            }
            let event_times = super::admission::times(record, side);
            let _column_scratch = self.reserve_workspace(record.num_columns() as u64 * 512)?;
            let columns = record
                .columns()
                .iter()
                .cloned()
                .map(ColumnWorkspace::new)
                .collect::<Result<Vec<_>>>()?;
            let mut accepted = 0_u64;
            let mut raw = 0_u64;
            for range in accepted_ranges(event_times.values(), input.watermark) {
                accepted = checked(&self.name, accepted, (range.end - range.start) as u64)?;
                for column in &columns {
                    raw = checked(
                        &self.name,
                        raw,
                        column.range_bytes(range.clone(), &self.name)?,
                    )?;
                }
            }
            if accepted == 0 {
                continue;
            }
            let schema = payload_charge(record.schema().as_ref(), &self.name)?.schema_bytes;
            let estimate = checked(&self.name, schema.saturating_mul(2), raw.saturating_mul(4))?;
            bytes = checked(
                &self.name,
                bytes,
                checked(
                    &self.name,
                    estimate,
                    accepted.saturating_mul(64).saturating_add(256),
                )?,
            )?;
        }
        self.reserve_workspace(bytes)
    }

    pub(super) fn identity_workspace(
        &self,
        batch: &Batch,
        input: super::admission::ValidatedInput,
    ) -> Result<MemoryReservation> {
        let mut bytes = 0;
        for record in batch.table_payload()?.batches() {
            bytes = checked(
                &self.name,
                bytes,
                self.identity_record_workspace(record, input)?,
            )?;
        }
        self.reserve_workspace(bytes)
    }

    fn identity_record_workspace(
        &self,
        record: &RecordBatch,
        input: super::admission::ValidatedInput,
    ) -> Result<u64> {
        if record.num_rows() == 0 {
            return Ok(0);
        }
        let side = input.side(&self.spec);
        let event_times = super::admission::times(record, side);
        let mut ranges = accepted_ranges(event_times.values(), input.watermark).peekable();
        if ranges.peek().is_none() {
            return Ok(0);
        }
        let identity_columns = side.keys().len() + side.sequence_by().len();
        let _column_scratch = self.reserve_workspace(identity_columns as u64 * 512)?;
        let columns = resolve_identity_columns(record, side)?;
        // Converter configuration and Arrow buffer headers remain live while
        // the accepted identities are assembled, including a one-row batch.
        let mut bytes = identity_columns as u64 * 512;
        for range in ranges {
            bytes = checked(
                &self.name,
                bytes,
                identity_range_workspace(&columns, range, &self.name)?,
            )?;
        }
        Ok(bytes)
    }
}

/// Group accepted rows without allocating an index vector. With no ingress
/// watermark, the complete batch is one range and no time values are scanned.
fn accepted_ranges(
    times: &[i64],
    watermark: Option<i64>,
) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
    let mut next = 0;
    std::iter::from_fn(move || {
        let Some(watermark) = watermark else {
            let range = next..times.len();
            next = times.len();
            return (!range.is_empty()).then_some(range);
        };
        next += times[next..]
            .iter()
            .take_while(|time| **time < watermark)
            .count();
        let start = next;
        next += times[next..]
            .iter()
            .take_while(|time| **time >= watermark)
            .count();
        (start < next).then_some(start..next)
    })
}

fn payload_charge(schema: &Schema, name: &str) -> Result<PayloadCharge> {
    let mut bytes = SCHEMA_ENVELOPE_BYTES;
    for (key, value) in schema.metadata() {
        bytes = checked(name, bytes, key.len() as u64 + value.len() as u64)?;
        bytes = checked(name, bytes, SCHEMA_METADATA_ENTRY_BYTES)?;
    }
    for field in schema.fields() {
        bytes = checked(name, bytes, field.size() as u64)?;
        bytes = checked(name, bytes, SCHEMA_FIELD_BYTES)?;
    }
    Ok(PayloadCharge {
        schema_bytes: bytes,
    })
}

/// A deterministic bound for the canonical IPC batch. A one-row schema
/// skeleton supplies fixed framing; the full payload body is measured from
/// logical Arrow lengths without serializing its rows. Restore recomputes the
/// same charge whether or not the encoded segment is materialized.
#[cfg(test)]
pub(super) fn payload_encoded_bound(record: &RecordBatch, name: &str) -> Result<(u64, u64)> {
    let body = payload_ipc_body_bytes(record, name)?;
    let header = ipc_header_bytes(record)?;
    let encoded = checked(name, header, body)?;
    Ok((encoded, body))
}

pub(super) fn payload_header_bytes(
    schema: &datafusion::arrow::datatypes::SchemaRef,
) -> Result<u64> {
    ipc_header_bytes(&RecordBatch::new_empty(schema.clone()))
}

pub(super) fn payload_bound_with_header(
    record: &RecordBatch,
    header: u64,
    name: &str,
) -> Result<(u64, u64)> {
    let body = payload_ipc_body_bytes(record, name)?;
    Ok((checked(name, header, body)?, body))
}

fn ipc_header_bytes(record: &RecordBatch) -> Result<u64> {
    let columns = record
        .schema()
        .fields()
        .iter()
        .map(|field| {
            ScalarValue::new_default(field.data_type())
                .and_then(|value| value.to_array())
                .map_err(|error| CalcFlowError::Format {
                    message: format!("ASOF IPC header skeleton failed: {error}"),
                })
        })
        .collect::<Result<Vec<_>>>()?;
    let skeleton = RecordBatch::try_new(record.schema(), columns)
        .map_err(|error| super::arrow_error(&error))?;
    let encoded = super::codec::encode_batch(&skeleton, usize::MAX, &mut Vec::new())?;
    let body = super::codec::payload_body_bytes(&encoded)?;
    Ok(encoded.len() as u64 - body)
}

fn payload_ipc_body_bytes(record: &RecordBatch, name: &str) -> Result<u64> {
    record.columns().iter().try_fold(0, |total, column| {
        checked(name, total, column_ipc_body_bytes(column, name)?)
    })
}

fn column_ipc_body_bytes(column: &ArrayRef, name: &str) -> Result<u64> {
    let rows = column.len() as u64;
    if matches!(column.data_type(), DataType::Null) {
        return Ok(0);
    }
    let bitmap = ipc_aligned(rows.div_ceil(8), name)?;
    let data = match column.data_type() {
        DataType::Boolean => ipc_aligned(rows.div_ceil(8), name)?,
        DataType::Utf8 | DataType::Binary => variable_ipc_bytes(column, rows, 4, name)?,
        DataType::LargeUtf8 | DataType::LargeBinary => variable_ipc_bytes(column, rows, 8, name)?,
        data_type => fixed_ipc_bytes(data_type, rows, name)?,
    };
    checked(name, bitmap, data)
}

fn variable_ipc_bytes(column: &ArrayRef, rows: u64, offset_width: u64, name: &str) -> Result<u64> {
    let offsets = ipc_multiply(checked(name, rows, 1)?, offset_width, name)?;
    checked(
        name,
        ipc_aligned(offsets, name)?,
        ipc_aligned(value_span(column)?, name)?,
    )
}

fn fixed_ipc_bytes(data_type: &DataType, rows: u64, name: &str) -> Result<u64> {
    let width = match data_type {
        DataType::FixedSizeBinary(width) => {
            u64::try_from(*width).expect("validated flat ASOF type")
        }
        _ => data_type
            .primitive_width()
            .expect("validated flat ASOF type") as u64,
    };
    ipc_aligned(ipc_multiply(rows, width, name)?, name)
}

fn ipc_multiply(left: u64, right: u64, name: &str) -> Result<u64> {
    left.checked_mul(right).ok_or_else(|| {
        reason(
            name,
            StreamingFailureReason::AsofCounterOverflow,
            "ASOF IPC buffer length overflowed",
        )
    })
}

fn value_span(column: &ArrayRef) -> Result<u64> {
    let span = match column.data_type() {
        DataType::Utf8 => span_offsets(
            column
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("string")
                .value_offsets(),
        ),
        DataType::Binary => span_offsets(
            column
                .as_any()
                .downcast_ref::<BinaryArray>()
                .expect("binary")
                .value_offsets(),
        ),
        DataType::LargeUtf8 => span_offsets(
            column
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .expect("large string")
                .value_offsets(),
        ),
        DataType::LargeBinary => span_offsets(
            column
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .expect("large binary")
                .value_offsets(),
        ),
        _ => unreachable!("validated variable ASOF type"),
    };
    u64::try_from(span).map_err(|_| CalcFlowError::Format {
        message: "ASOF variable IPC offsets are invalid".into(),
    })
}

fn span_offsets<T: Copy + Into<i128>>(offsets: &[T]) -> i128 {
    offsets.last().copied().expect("Arrow offsets").into()
        - offsets.first().copied().expect("Arrow offsets").into()
}

fn ipc_aligned(bytes: u64, name: &str) -> Result<u64> {
    checked(name, bytes, 63).map(|value| value & !63)
}

/// Historical upper bound for a legacy single-row IPC encoding. Kept in the
/// focused tests as evidence that version 1 row snapshots were bounded.
#[cfg(test)]
fn row_workspace(
    payload: &PayloadCharge,
    record: &RecordBatch,
    row: usize,
    name: &str,
) -> Result<u64> {
    let mut bytes = payload.schema_bytes;
    for column in record.columns() {
        bytes = checked(name, bytes, aligned(column_workspace(column, row)?))?;
        bytes = checked(name, bytes, COLUMN_FRAMING_BYTES)?;
    }
    checked(name, bytes, ROW_FRAMING_BYTES)
}

/// Identity columns resolved once per record batch: the array reference plus
/// whether the column uses Arrow's blocked string row encoding.
type ResolvedIdentityColumns = Vec<(ColumnWorkspace, bool, bool)>;

fn resolve_identity_columns(
    record: &RecordBatch,
    side: &super::AsofJoinSide,
) -> Result<ResolvedIdentityColumns> {
    side.keys()
        .iter()
        .chain(side.sequence_by())
        .enumerate()
        .map(|(index, field)| {
            let column = record
                .column(record.schema().index_of(field).expect("validated schema"))
                .clone();
            let string = matches!(column.data_type(), DataType::Utf8 | DataType::LargeUtf8);
            Ok((
                ColumnWorkspace::new(column)?,
                string,
                index >= side.keys().len(),
            ))
        })
        .collect()
}

/// Covers batch row bytes and per-row sequence copies. Unique owned key copies
/// grow a separate reservation before allocation. String encodings use 33/32
/// scaling for block sentinels.
#[cfg(test)]
fn identity_row_workspace(
    columns: &ResolvedIdentityColumns,
    row: usize,
    name: &str,
) -> Result<u64> {
    let mut bytes = IDENTITY_ROW_BYTES;
    for (column, string, retained_copy) in columns {
        let slice = aligned(column.bytes(row, name)?);
        let encoded = if *string {
            checked(
                name,
                checked(name, slice, slice / 32)?,
                STRING_IDENTITY_FRAMING_BYTES,
            )?
        } else {
            checked(name, slice, FIXED_IDENTITY_FRAMING_BYTES)?
        };
        bytes = checked(name, bytes, encoded)?;
        if *retained_copy {
            bytes = checked(name, bytes, encoded)?;
        }
    }
    Ok(bytes)
}

fn identity_encoding_workspace(
    column: &ColumnWorkspace,
    row: usize,
    string: bool,
    name: &str,
) -> Result<u64> {
    let slice = aligned(column.bytes(row, name)?);
    if string {
        checked(
            name,
            checked(name, slice, slice / 32)?,
            STRING_IDENTITY_FRAMING_BYTES,
        )
    } else {
        checked(name, slice, FIXED_IDENTITY_FRAMING_BYTES)
    }
}

/// Fixed-width identity columns have identical per-row framing and slice
/// bytes. Variable columns retain the exact per-value alignment charge.
fn identity_range_workspace(
    columns: &ResolvedIdentityColumns,
    range: std::ops::Range<usize>,
    name: &str,
) -> Result<u64> {
    let count = (range.end - range.start) as u64;
    if count == 0 {
        return Ok(0);
    }
    let mut bytes = ipc_multiply(IDENTITY_ROW_BYTES, count, name)?;
    for (column, string, retained_copy) in columns {
        let encoded = identity_column_range_workspace(column, range.clone(), *string, name)?;
        bytes = checked(name, bytes, encoded)?;
        if *retained_copy {
            bytes = checked(name, bytes, encoded)?;
        }
    }
    Ok(bytes)
}

fn identity_column_range_workspace(
    column: &ColumnWorkspace,
    mut range: std::ops::Range<usize>,
    string: bool,
    name: &str,
) -> Result<u64> {
    if string {
        range.try_fold(0, |total, row| {
            checked(
                name,
                total,
                identity_encoding_workspace(column, row, true, name)?,
            )
        })
    } else {
        ipc_multiply(
            identity_encoding_workspace(column, range.start, false, name)?,
            (range.end - range.start) as u64,
            name,
        )
    }
}

/// Aligns a buffer length to the IPC writer's alignment boundary.
fn aligned(bytes: u64) -> u64 {
    bytes.checked_add(63).map_or(bytes, |aligned| aligned & !63)
}

fn column_workspace(column: &ArrayRef, row: usize) -> Result<u64> {
    column
        .to_data()
        .slice(row, 1)
        .get_slice_memory_size()
        .map(|bytes| bytes as u64)
        .map_err(|error| super::arrow_error(&error))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AsofJoinSide;
    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
        datatypes::{DataType, Field, Schema, TimeUnit},
    };
    use std::sync::Arc;

    use super::super::{codec, state};

    fn repro_record(rows: u64) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, false),
        ]));
        let indexes = 0..i32::try_from(rows).unwrap();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(
                        indexes.clone().map(|row| 1_000_000 + i64::from(row)),
                    )
                    .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(0..rows)),
                Arc::new(StringArray::from_iter_values(
                    indexes.clone().map(|row| format!("S{row:03}")),
                )),
                Arc::new(Float64Array::from_iter_values(
                    indexes.map(|row| 100.0 + f64::from(row % 257)),
                )),
            ],
        )
        .unwrap()
    }

    fn repro_side() -> AsofJoinSide {
        AsofJoinSide::new(
            vec!["symbol".into()],
            "event_time".into(),
            vec!["sequence".into()],
            "left".into(),
        )
        .unwrap()
    }

    fn metadata_record() -> RecordBatch {
        let mut fields = repro_record(1)
            .schema()
            .fields()
            .iter()
            .map(|field| field.as_ref().clone())
            .collect::<Vec<_>>();
        fields[3] = Field::new("value", DataType::Utf8, false);
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            std::collections::HashMap::from([("note".to_string(), "m".repeat(4_096))]),
        ));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(std::iter::once(1_000_000))
                        .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(std::iter::once(0))),
                Arc::new(StringArray::from_iter_values(std::iter::once("S000"))),
                Arc::new(StringArray::from_iter_values(std::iter::once(
                    "m".repeat(4_096),
                ))),
            ],
        )
        .unwrap()
    }

    fn wide_string_record() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("value", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(std::iter::once(1_000_000))
                        .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(std::iter::once(0))),
                Arc::new(StringArray::from_iter_values(std::iter::once("S000"))),
                Arc::new(StringArray::from_iter_values(std::iter::once(
                    "x".repeat(48_000),
                ))),
            ],
        )
        .unwrap()
    }

    fn string_key_record(length: usize) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("symbol", DataType::Utf8, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(
                    TimestampMicrosecondArray::from_iter_values(std::iter::once(1_000_000))
                        .with_timezone("UTC"),
                ),
                Arc::new(UInt64Array::from_iter_values(std::iter::once(0))),
                Arc::new(StringArray::from_iter_values(std::iter::once(
                    "x".repeat(length),
                ))),
            ],
        )
        .unwrap()
    }

    #[test]
    fn cached_column_workspace_matches_arrow_slice_charge() {
        use datafusion::arrow::array::{BinaryArray, BooleanArray};

        let columns: Vec<ArrayRef> = vec![
            Arc::new(BooleanArray::from(vec![
                Some(true),
                None,
                Some(false),
                Some(true),
            ])),
            Arc::new(StringArray::from(vec![
                Some("short"),
                None,
                Some(&"x".repeat(48_000)),
                Some(""),
            ])),
            Arc::new(BinaryArray::from(vec![
                Some(b"a".as_slice()),
                None,
                Some(b"many bytes".as_slice()),
                Some(b"".as_slice()),
            ])),
            Arc::new(LargeStringArray::from(vec![
                Some("short"),
                None,
                Some("many bytes"),
                Some(""),
            ])),
            Arc::new(LargeBinaryArray::from(vec![
                Some(b"a".as_slice()),
                None,
                Some(b"many bytes".as_slice()),
                Some(b"".as_slice()),
            ])),
        ];
        for column in columns {
            for sliced in [column.clone(), column.slice(1, 3)] {
                let cached = ColumnWorkspace::new(sliced.clone()).unwrap();
                for row in 0..sliced.len() {
                    assert_eq!(
                        cached.bytes(row, "asof").unwrap(),
                        column_workspace(&sliced, row).unwrap()
                    );
                }
                for start in 0..=sliced.len() {
                    for end in start..=sliced.len() {
                        let slices = (start..end)
                            .map(|row| column_workspace(&sliced, row).unwrap())
                            .sum::<u64>();
                        assert_eq!(cached.range_bytes(start..end, "asof").unwrap(), slices);
                    }
                }
            }
        }
    }

    #[test]
    fn row_workspace_bounds_the_actual_encoded_row_bytes() {
        for record in [repro_record(8), wide_string_record(), metadata_record()] {
            let payload = payload_charge(record.schema().as_ref(), "asof").unwrap();
            let mut scratch = Vec::new();
            for row in 0..record.num_rows() {
                let charge = row_workspace(&payload, &record, row, "asof").unwrap();
                let actual = codec::encode_batch(&record.slice(row, 1), usize::MAX, &mut scratch)
                    .unwrap()
                    .len() as u64;
                assert!(
                    charge >= actual,
                    "row {row}: charge {charge} < actual {actual}"
                );
            }
        }
    }

    #[test]
    fn batch_payload_charge_bounds_ipc_encoding_and_body() {
        use datafusion::arrow::array::{BinaryArray, BooleanArray, NullArray};
        let nullable = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("flag", DataType::Boolean, true),
                Field::new("text", DataType::Utf8, true),
                Field::new("bytes", DataType::Binary, true),
                Field::new("nothing", DataType::Null, true),
            ])),
            vec![
                Arc::new(BooleanArray::from(vec![
                    Some(true),
                    None,
                    Some(false),
                    Some(true),
                ])),
                Arc::new(StringArray::from(vec![
                    Some("a"),
                    None,
                    Some("many"),
                    Some(""),
                ])),
                Arc::new(BinaryArray::from(vec![
                    Some(b"z".as_slice()),
                    None,
                    Some(b"abc".as_slice()),
                    Some(b"".as_slice()),
                ])),
                Arc::new(NullArray::new(4)),
            ],
        )
        .unwrap();
        for record in [
            repro_record(8),
            wide_string_record(),
            metadata_record(),
            repro_record(8).slice(2, 4),
            nullable.clone(),
            nullable.slice(1, 2),
        ] {
            let (encoded_bound, body_bound) = payload_encoded_bound(&record, "asof").unwrap();
            let encoded = codec::encode_batch(&record, usize::MAX, &mut Vec::new()).unwrap();
            let body = codec::payload_body_bytes(&encoded).unwrap();
            assert_eq!(body_bound, body);
            assert_eq!(encoded_bound, encoded.len() as u64);
        }
    }

    #[test]
    fn batch_payload_charge_covers_supported_flat_scalar_types() {
        use datafusion::arrow::datatypes::IntervalUnit;

        let types = [
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::Int16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float16,
            DataType::Float32,
            DataType::Float64,
            DataType::Date32,
            DataType::Date64,
            DataType::Time32(TimeUnit::Second),
            DataType::Time32(TimeUnit::Millisecond),
            DataType::Time64(TimeUnit::Microsecond),
            DataType::Time64(TimeUnit::Nanosecond),
            DataType::Timestamp(TimeUnit::Second, None),
            DataType::Timestamp(TimeUnit::Millisecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
            DataType::Duration(TimeUnit::Second),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Duration(TimeUnit::Microsecond),
            DataType::Duration(TimeUnit::Nanosecond),
            DataType::Interval(IntervalUnit::YearMonth),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Decimal32(8, 2),
            DataType::Decimal64(16, 2),
            DataType::Decimal128(30, 2),
            DataType::Decimal256(70, 2),
            DataType::FixedSizeBinary(8),
            DataType::FixedSizeBinary(0),
            DataType::LargeUtf8,
            DataType::LargeBinary,
        ];
        for data_type in types {
            let value = ScalarValue::new_default(&data_type).unwrap();
            let record = RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "value",
                    data_type.clone(),
                    true,
                )])),
                vec![value.to_array().unwrap()],
            )
            .unwrap();
            let (encoded_bound, body_bound) = payload_encoded_bound(&record, "asof").unwrap();
            let encoded = codec::encode_batch(&record, usize::MAX, &mut Vec::new()).unwrap();
            let body = codec::payload_body_bytes(&encoded).unwrap();
            assert_eq!(body_bound, body, "{data_type:?}");
            assert_eq!(encoded_bound, encoded.len() as u64, "{data_type:?}");
        }
    }

    #[test]
    fn schema_metadata_charge_covers_the_declared_metadata_bytes() {
        let plain = payload_charge(repro_record(1).schema().as_ref(), "asof").unwrap();
        let metadata = payload_charge(metadata_record().schema().as_ref(), "asof").unwrap();
        assert!(
            metadata.schema_bytes >= plain.schema_bytes + 4_096,
            "metadata charge {} does not cover the declared bytes over {}",
            metadata.schema_bytes,
            plain.schema_bytes
        );
    }

    #[test]
    fn identity_row_workspace_bounds_the_actual_identity_encodings() {
        let record = repro_record(8);
        let side = repro_side();
        let columns = resolve_identity_columns(&record, &side).unwrap();
        for row in 0..record.num_rows() {
            let charge = identity_row_workspace(&columns, row, "asof").unwrap();
            let actual = state::encoded_columns(&record, row, side.keys())
                .unwrap()
                .len() as u64
                + state::encoded_columns(&record, row, side.sequence_by())
                    .unwrap()
                    .len() as u64;
            assert!(
                charge >= actual + 64,
                "row {row}: charge {charge} < actual {actual} plus allocations"
            );
        }
    }

    #[test]
    fn admission_ranges_preserve_exact_row_workspace_and_skip_late_rows() {
        assert_eq!(
            accepted_ranges(&[-2, 3, -1, 4, 4], Some(0)).collect::<Vec<_>>(),
            [1..2, 3..5]
        );
        assert_eq!(
            accepted_ranges(&[-2, 3, -1, 4, 4], None).collect::<Vec<_>>(),
            std::iter::once(0..5).collect::<Vec<_>>()
        );
        assert!(accepted_ranges(&[], None).next().is_none());
        let record = repro_record(8);
        let columns = resolve_identity_columns(&record, &repro_side()).unwrap();
        for start in 0..=record.num_rows() {
            for end in start..=record.num_rows() {
                let expected = (start..end)
                    .map(|row| identity_row_workspace(&columns, row, "asof").unwrap())
                    .sum::<u64>();
                assert_eq!(
                    identity_range_workspace(&columns, start..end, "asof").unwrap(),
                    expected
                );
            }
        }
    }

    #[test]
    fn identity_row_workspace_bounds_long_string_keys() {
        // Arrow's blocked string row encoding adds one marker byte plus a
        // sentinel byte per block (~L/32 for length L), so the per-column
        // charge must scale with the key length, not just the slice bytes.
        let side = repro_side();
        for length in (0..=260_usize).chain([1_000, 4_096, 48_000, 100_000]) {
            let record = string_key_record(length);
            let columns = resolve_identity_columns(&record, &side).unwrap();
            let charge = identity_row_workspace(&columns, 0, "asof").unwrap();
            let actual = state::encoded_columns(&record, 0, side.keys())
                .unwrap()
                .len() as u64
                + state::encoded_columns(&record, 0, side.sequence_by())
                    .unwrap()
                    .len() as u64;
            assert!(
                charge >= actual,
                "length {length}: charge {charge} < actual {actual}"
            );
        }
    }

    #[test]
    fn repro_shaped_admissions_charge_close_to_actual_state_bytes() {
        let rows = 10_000_u64;
        let record = repro_record(rows);
        let payload = payload_charge(record.schema().as_ref(), "asof").unwrap();
        let charge = (0..record.num_rows())
            .map(|row| row_workspace(&payload, &record, row, "asof").unwrap())
            .sum::<u64>();
        let actual = codec::encode_batch(&record.slice(0, 1), usize::MAX, &mut Vec::new())
            .unwrap()
            .len() as u64
            * rows;
        assert!(charge < 64 * 1024 * 1024, "charge {charge} exceeds 64 MiB");
        assert!(
            charge < actual.saturating_mul(2),
            "charge {charge} is more than twice the actual {actual}"
        );
    }
}
