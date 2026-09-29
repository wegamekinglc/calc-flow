use super::{StreamAsofJoinOperator, checked, reason};
use crate::{Batch, CalcFlowError, Result, StreamingFailureReason};
use datafusion::{
    arrow::{
        array::{ArrayRef, BinaryArray, LargeBinaryArray, LargeStringArray, StringArray},
        datatypes::{DataType, Schema},
        record_batch::RecordBatch,
    },
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
            for row in 0..record.num_rows() {
                if input.is_late(event_times.value(row)) {
                    continue;
                }
                accepted = checked(&self.name, accepted, 1)?;
                for column in &columns {
                    raw = checked(&self.name, raw, column.bytes(row, &self.name)?)?;
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
        let identity_columns = side.keys().len() + side.sequence_by().len();
        let _column_scratch = self.reserve_workspace(identity_columns as u64 * 512)?;
        let columns = resolve_identity_columns(record, side)?;
        let event_times = super::admission::times(record, side);
        let mut bytes = 0;
        for row in 0..record.num_rows() {
            if input.is_late(event_times.value(row)) {
                continue;
            }
            bytes = checked(
                &self.name,
                bytes,
                identity_row_workspace(&columns, row, &self.name)?,
            )?;
        }
        Ok(bytes)
    }
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
type ResolvedIdentityColumns = Vec<(ColumnWorkspace, bool)>;

fn resolve_identity_columns(
    record: &RecordBatch,
    side: &super::AsofJoinSide,
) -> Result<ResolvedIdentityColumns> {
    side.keys()
        .iter()
        .chain(side.sequence_by())
        .map(|field| {
            let column = record
                .column(record.schema().index_of(field).expect("validated schema"))
                .clone();
            let string = matches!(column.data_type(), DataType::Utf8 | DataType::LargeUtf8);
            Ok((ColumnWorkspace::new(column)?, string))
        })
        .collect()
}

/// Allocation-free upper bound on one row's identity encodings: fixed
/// headroom plus, per column, the aligned slice bytes scaled for the row
/// format's marker framing. String columns are scaled by 33/32 because the
/// blocked encoding adds a sentinel byte per block (~L/32 for length L).
fn identity_row_workspace(
    columns: &ResolvedIdentityColumns,
    row: usize,
    name: &str,
) -> Result<u64> {
    let mut bytes = IDENTITY_ROW_BYTES;
    for (column, string) in columns {
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
    }
    Ok(bytes)
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
