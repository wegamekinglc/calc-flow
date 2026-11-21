use crate::Result;
use crate::operator::asof::arrow_error;
use crate::operator::keyed::GroupId;
use datafusion::arrow::{
    array::{
        Array, ArrayRef, BinaryArray, LargeBinaryArray, LargeBinaryBuilder, LargeStringArray,
        StringArray,
    },
    datatypes::{DataType, Field, Schema, SchemaRef},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub(in crate::operator::asof) struct RowLocator {
    pub(super) run: u64,
    pub(super) row: u32,
}

#[derive(Clone)]
pub(super) enum SequenceColumn {
    Binary(BinaryArray),
    LargeBinary(LargeBinaryArray),
}

pub(super) fn sequence_copy_count(encoded_extent: usize) -> usize {
    if encoded_extent > i32::MAX as usize {
        2
    } else {
        1
    }
}

pub(super) fn sequence_tuple_copy_count(extents: impl IntoIterator<Item = usize>) -> usize {
    sequence_copy_count(extents.into_iter().fold(0, usize::saturating_add))
}

impl SequenceColumn {
    pub(super) fn from_names(record: &RecordBatch, names: &[String]) -> Result<Self> {
        let columns = names
            .iter()
            .map(|name| {
                record
                    .schema()
                    .index_of(name)
                    .map(|index| record.column(index).clone())
                    .map_err(|error| arrow_error(&error))
            })
            .collect::<Result<Vec<ArrayRef>>>()?;
        Self::from_columns(&columns)
    }

    fn from_columns(columns: &[ArrayRef]) -> Result<Self> {
        let fields = columns
            .iter()
            .map(|array| SortField::new(array.data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).map_err(|error| arrow_error(&error))?;
        let rows = converter
            .convert_columns(columns)
            .map_err(|error| arrow_error(&error))?;
        if i32::try_from(rows.size()).is_ok() {
            return rows
                .try_into_binary()
                .map(Self::Binary)
                .map_err(|error| arrow_error(&error));
        }
        let bytes = rows.lengths().sum();
        let mut builder = LargeBinaryBuilder::with_capacity(rows.num_rows(), bytes);
        for row in &rows {
            builder.append_value(row.data());
        }
        Ok(Self::LargeBinary(builder.finish()))
    }

    pub(super) fn value(&self, row: usize) -> &[u8] {
        match self {
            Self::Binary(values) => values.value(row),
            Self::LargeBinary(values) => values.value(row),
        }
    }

    pub(super) fn array(&self) -> ArrayRef {
        match self {
            Self::Binary(values) => Arc::new(values.clone()),
            Self::LargeBinary(values) => Arc::new(values.clone()),
        }
    }

    pub(super) fn from_array(array: &ArrayRef) -> Result<Self> {
        match array.data_type() {
            DataType::Binary => array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .cloned()
                .map(Self::Binary),
            DataType::LargeBinary => array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .cloned()
                .map(Self::LargeBinary),
            _ => None,
        }
        .ok_or_else(|| {
            arrow_error(&datafusion::arrow::error::ArrowError::InvalidArgumentError(
                "ASOF restored sequence column is not binary".into(),
            ))
        })
    }

    pub(super) fn memory_size(&self) -> usize {
        match self {
            Self::Binary(values) => values.get_array_memory_size(),
            Self::LargeBinary(values) => values.get_array_memory_size(),
        }
    }
}

pub(super) fn identity_schema(large_sequence: bool) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("group", DataType::UInt32, false),
        Field::new("time", DataType::Int64, false),
        Field::new(
            "sequence",
            if large_sequence {
                DataType::LargeBinary
            } else {
                DataType::Binary
            },
            false,
        ),
        Field::new("payload_live", DataType::Boolean, false),
        Field::new("identity_live", DataType::Boolean, false),
    ]))
}

fn ipc_schema_charge(schema: &SchemaRef) -> usize {
    let fields = schema.fields().iter().fold(256usize, |total, field| {
        total.saturating_add(field.size()).saturating_add(192)
    });
    schema
        .metadata()
        .iter()
        .fold(fields, |total, (key, value)| {
            total
                .saturating_add(key.len())
                .saturating_add(value.len())
                .saturating_add(64)
        })
}

pub(super) fn logical_array_charge(array: &ArrayRef) -> usize {
    macro_rules! variable {
        ($array_type:ty, $offset_width:expr) => {{
            let values = array
                .as_any()
                .downcast_ref::<$array_type>()
                .expect("typed array");
            let offsets = values.value_offsets();
            let start = values.offset();
            let bytes = offsets[start + values.len()] - offsets[start];
            (values.len() + 1) * $offset_width + usize::try_from(bytes).expect("nonnegative values")
        }};
    }
    let data = match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean => array.len().div_ceil(8),
        DataType::Utf8 => variable!(StringArray, 4),
        DataType::LargeUtf8 => variable!(LargeStringArray, 8),
        DataType::Binary => variable!(BinaryArray, 4),
        DataType::LargeBinary => variable!(LargeBinaryArray, 8),
        DataType::FixedSizeBinary(width) => array
            .len()
            .saturating_mul(usize::try_from(*width).expect("positive fixed binary width")),
        data_type => array.len().saturating_mul(
            data_type
                .primitive_width()
                .expect("validated flat ASOF payload type"),
        ),
    };
    let validity = if array.null_count() == 0 {
        0
    } else {
        array.len().div_ceil(8)
    };
    128usize.saturating_add(data).saturating_add(validity)
}

pub(super) fn logical_array_selection_charge(array: &ArrayRef, accepted: &[bool]) -> usize {
    assert_eq!(array.len(), accepted.len());
    let count = accepted.iter().filter(|accepted| **accepted).count();
    macro_rules! variable {
        ($array_type:ty, $offset_width:expr) => {{
            let values = array
                .as_any()
                .downcast_ref::<$array_type>()
                .expect("typed array");
            let mut bytes = (count + 1) * $offset_width;
            for (row, accepted) in accepted.iter().enumerate() {
                if *accepted {
                    bytes = bytes.saturating_add(
                        usize::try_from(values.value_length(row)).expect("nonnegative value"),
                    );
                }
            }
            bytes
        }};
    }
    let data = match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean => count.div_ceil(8),
        DataType::Utf8 => variable!(StringArray, 4),
        DataType::LargeUtf8 => variable!(LargeStringArray, 8),
        DataType::Binary => variable!(BinaryArray, 4),
        DataType::LargeBinary => variable!(LargeBinaryArray, 8),
        DataType::FixedSizeBinary(width) => {
            count.saturating_mul(usize::try_from(*width).expect("positive fixed binary width"))
        }
        data_type => count.saturating_mul(
            data_type
                .primitive_width()
                .expect("validated flat ASOF payload type"),
        ),
    };
    128usize
        .saturating_add(data)
        .saturating_add(count.div_ceil(8))
}

pub(super) fn logical_array_indices_charge(array: &ArrayRef, indices: &[usize]) -> usize {
    let count = indices.len();
    macro_rules! variable {
        ($array_type:ty, $offset_width:expr) => {{
            let values = array
                .as_any()
                .downcast_ref::<$array_type>()
                .expect("typed array");
            indices
                .iter()
                .fold((count + 1) * $offset_width, |total, row| {
                    total.saturating_add(
                        usize::try_from(values.value_length(*row)).expect("nonnegative value"),
                    )
                })
        }};
    }
    let data = match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean => count.div_ceil(8),
        DataType::Utf8 => variable!(StringArray, 4),
        DataType::LargeUtf8 => variable!(LargeStringArray, 8),
        DataType::Binary => variable!(BinaryArray, 4),
        DataType::LargeBinary => variable!(LargeBinaryArray, 8),
        DataType::FixedSizeBinary(width) => {
            count.saturating_mul(usize::try_from(*width).expect("positive fixed binary width"))
        }
        data_type => count.saturating_mul(
            data_type
                .primitive_width()
                .expect("validated flat ASOF payload type"),
        ),
    };
    128usize
        .saturating_add(data)
        .saturating_add(count.div_ceil(8))
}

pub(super) fn logical_null_charge(data_type: &DataType, rows: usize) -> usize {
    match data_type {
        DataType::Null => 0,
        DataType::Boolean => rows.div_ceil(8),
        DataType::Utf8 | DataType::Binary => (rows + 1) * 4,
        DataType::LargeUtf8 | DataType::LargeBinary => (rows + 1) * 8,
        DataType::FixedSizeBinary(width) => {
            rows.saturating_mul(usize::try_from(*width).expect("positive fixed binary width"))
        }
        data_type => rows.saturating_mul(
            data_type
                .primitive_width()
                .expect("validated flat ASOF payload type"),
        ),
    }
}

pub(in crate::operator::asof) fn logical_record_selection_charge(
    record: &RecordBatch,
    accepted: &[bool],
) -> usize {
    record
        .columns()
        .iter()
        .fold(0usize, |total, column| {
            total.saturating_add(logical_array_selection_charge(column, accepted))
        })
        .saturating_add(ipc_schema_charge(&record.schema()))
        .saturating_add(256)
        .saturating_add(
            accepted
                .iter()
                .filter(|accepted| **accepted)
                .count()
                .saturating_mul(384),
        )
}

fn retained_payload_charge(record: &RecordBatch) -> usize {
    record
        .columns()
        .iter()
        .fold(0usize, |total, column| {
            total.saturating_add(logical_array_charge(column))
        })
        .saturating_add(ipc_schema_charge(&record.schema()))
        .saturating_add(256)
}

fn retained_identity_charge(
    groups: &[GroupId],
    times: &[i64],
    sequences: &SequenceColumn,
) -> usize {
    let rows = times.len();
    groups
        .len()
        .saturating_mul(size_of::<GroupId>())
        .saturating_add(times.len().saturating_mul(size_of::<i64>()))
        .saturating_add(logical_array_charge(&sequences.array()))
        .saturating_add(rows.div_ceil(8).saturating_mul(2))
        .saturating_add(256)
        .saturating_add(ipc_schema_charge(&identity_schema(matches!(
            sequences,
            SequenceColumn::LargeBinary(_)
        ))))
}

pub(super) struct Run {
    pub(super) payload: Option<Arc<RecordBatch>>,
    pub(super) groups: Vec<GroupId>,
    pub(super) times: Vec<i64>,
    pub(super) sequences: SequenceColumn,
    payload_live: Vec<bool>,
    identity_live: Vec<bool>,
    payload_rows: usize,
    identity_rows: usize,
    payload_charge: usize,
    identity_charge: usize,
}

impl Run {
    pub(super) fn new(
        record: RecordBatch,
        groups: Vec<GroupId>,
        times: Vec<i64>,
        sequences: SequenceColumn,
        accepted: &[bool],
    ) -> Self {
        let payload_rows = accepted.iter().filter(|accepted| **accepted).count();
        let payload_charge = if payload_rows == 0 {
            0
        } else {
            retained_payload_charge(&record)
        };
        let identity_charge = retained_identity_charge(&groups, &times, &sequences);
        Self {
            payload: (payload_rows != 0).then(|| Arc::new(record)),
            groups,
            times,
            sequences,
            payload_live: accepted.to_vec(),
            identity_live: accepted.to_vec(),
            payload_rows,
            identity_rows: payload_rows,
            payload_charge,
            identity_charge,
        }
    }

    pub(super) fn locator(run: u64, row: usize) -> RowLocator {
        RowLocator {
            run,
            row: u32::try_from(row).expect("Arrow rows fit the ASOF locator domain"),
        }
    }

    pub(super) fn accepted_locators(&self, run: u64) -> Vec<RowLocator> {
        self.identity_live
            .iter()
            .enumerate()
            .filter(|(_, accepted)| **accepted)
            .map(|(row, _)| Self::locator(run, row))
            .collect()
    }

    pub(super) fn time(&self, locator: RowLocator) -> i64 {
        self.times[locator.row as usize]
    }

    pub(super) fn group(&self, locator: RowLocator) -> GroupId {
        self.groups[locator.row as usize]
    }

    pub(super) fn sequence(&self, locator: RowLocator) -> &[u8] {
        self.sequences.value(locator.row as usize)
    }

    pub(super) fn same_identity(&self, left: RowLocator, right: RowLocator) -> bool {
        self.group(left) == self.group(right)
            && self.time(left) == self.time(right)
            && self.sequence(left) == self.sequence(right)
    }

    pub(super) fn has_payload(&self, locator: RowLocator) -> bool {
        self.payload_live[locator.row as usize]
    }

    pub(super) fn has_identity(&self, locator: RowLocator) -> bool {
        self.identity_live[locator.row as usize]
    }

    pub(super) fn is_live(&self) -> bool {
        self.payload_rows != 0 || self.identity_rows != 0
    }

    pub(super) fn charge(&self) -> usize {
        self.identity_charge
            + if self.payload.is_some() {
                self.payload_charge
            } else {
                0
            }
    }

    pub(super) fn restored(
        payload: Option<Arc<RecordBatch>>,
        groups: Vec<GroupId>,
        times: Vec<i64>,
        sequences: SequenceColumn,
        payload_live: Vec<bool>,
        identity_live: Vec<bool>,
        charges: (usize, usize),
    ) -> Result<Self> {
        let (payload_charge, identity_charge) = charges;
        let actual_payload_charge = payload
            .as_ref()
            .map_or(0, |record| retained_payload_charge(record));
        let actual_identity_charge = retained_identity_charge(&groups, &times, &sequences);
        if charges != (actual_payload_charge, actual_identity_charge) {
            return Err(crate::CalcFlowError::CheckpointMismatch {
                message: format!(
                    "ASOF columnar run charges differ from decoded state: payload stored={payload_charge} actual={actual_payload_charge}, identity stored={identity_charge} actual={actual_identity_charge}"
                ),
            });
        }
        let payload_rows = payload_live.iter().filter(|live| **live).count();
        let identity_rows = identity_live.iter().filter(|live| **live).count();
        Ok(Self {
            payload,
            groups,
            times,
            sequences,
            payload_live,
            identity_live,
            payload_rows,
            identity_rows,
            payload_charge,
            identity_charge,
        })
    }

    pub(super) fn rows(&self) -> usize {
        self.times.len()
    }

    pub(super) fn payload_record(&self) -> Option<&RecordBatch> {
        self.payload.as_deref()
    }

    pub(super) fn payload_liveness(&self) -> &[bool] {
        &self.payload_live
    }

    pub(super) fn identity_liveness(&self) -> &[bool] {
        &self.identity_live
    }

    pub(super) fn groups(&self) -> &[GroupId] {
        &self.groups
    }

    pub(super) fn times(&self) -> &[i64] {
        &self.times
    }

    pub(super) fn sequences(&self) -> &SequenceColumn {
        &self.sequences
    }

    pub(super) fn charges(&self) -> (usize, usize) {
        (self.payload_charge, self.identity_charge)
    }

    pub(super) fn drop_payload(&mut self, locator: RowLocator) -> bool {
        let row = locator.row as usize;
        if !std::mem::take(&mut self.payload_live[row]) {
            return false;
        }
        self.payload_rows -= 1;
        if self.payload_rows == 0 {
            self.payload = None;
        }
        true
    }

    pub(super) fn drop_identity(&mut self, locator: RowLocator) -> bool {
        let row = locator.row as usize;
        if !std::mem::take(&mut self.identity_live[row]) {
            return false;
        }
        self.identity_rows -= 1;
        true
    }
}
