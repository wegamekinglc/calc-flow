use std::{collections::HashMap, sync::Arc};

use super::{
    EntityKey, RingKind, StateTable, ValueKind,
    core::{KeyedSnapshot, Layout7HistorySchema, RingSnapshot, TableSnapshot},
    layout7_keyed::{decode_keys, decode_logical_bytes, read_component, write_component},
};
use crate::{CalcFlowError, Result};
use datafusion::arrow::{
    array::{Array, ArrayRef, BinaryArray, Float64Array, UInt64Array},
    datatypes::{DataType, Field},
};

const MAGIC: &[u8; 8] = b"CFST7H\0\0";
const COMPONENTS: usize = 3;
const HEADER_BYTES: usize = 8 + 4 + COMPONENTS * 8;

pub(super) fn encode(table: &StateTable) -> Option<Result<Vec<u8>>> {
    let schema = table.layout7_history_schema()?;
    Some(encode_history(table, &schema))
}

pub(super) fn decode(table: &StateTable, bytes: &[u8]) -> Option<Result<TableSnapshot>> {
    if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
        return None;
    }
    Some(decode_history(table, bytes))
}

fn encode_history(table: &StateTable, schema: &Layout7HistorySchema) -> Result<Vec<u8>> {
    let snapshot = table.snapshot_for_layout7();
    let ring = &snapshot.rings[0];
    let keyed = write_component(
        std::iter::once(Field::new("entity_key", DataType::Binary, false))
            .chain(
                schema
                    .keyed_names
                    .iter()
                    .map(|name| Field::new(name, DataType::UInt64, false)),
            )
            .chain(std::iter::once(Field::new(
                "ring_capacity",
                DataType::UInt64,
                false,
            )))
            .collect(),
        std::iter::once(Arc::new(BinaryArray::from_iter_values(
            snapshot.keys.iter().map(EntityKey::bytes),
        )) as ArrayRef)
        .chain(snapshot.keyed.iter().map(|column| column.values.clone()))
        .chain(std::iter::once(usize_array(&ring.capacities)?))
        .collect(),
        HashMap::from([(
            "logical_bytes".to_owned(),
            snapshot.logical_bytes.to_string(),
        )]),
    )?;
    let history = write_component(
        vec![Field::new(&schema.history_name, DataType::Float64, true)],
        vec![ring.values.clone()],
        HashMap::new(),
    )?;
    let metadata = write_component(
        vec![Field::new("ring_start", DataType::UInt64, false)],
        vec![usize_array(&ring.offsets)?],
        HashMap::new(),
    )?;
    assemble(&[keyed, history, metadata])
}

fn assemble(components: &[Vec<u8>]) -> Result<Vec<u8>> {
    let mut encoded =
        Vec::with_capacity(HEADER_BYTES + components.iter().map(Vec::len).sum::<usize>());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(
        &u32::try_from(COMPONENTS)
            .expect("three history components")
            .to_le_bytes(),
    );
    for component in components {
        let len = u64::try_from(component.len())
            .map_err(|_| mismatch("rolling history layout-7 component exceeds u64"))?;
        encoded.extend_from_slice(&len.to_le_bytes());
    }
    for component in components {
        encoded.extend_from_slice(component);
    }
    Ok(encoded)
}

fn decode_history(table: &StateTable, bytes: &[u8]) -> Result<TableSnapshot> {
    let schema = table
        .layout7_history_schema()
        .ok_or_else(|| mismatch("rolling history layout-7 schema mismatch"))?;
    let components = decode_components(bytes)?;
    let keyed_record = read_component(&components[0])?;
    let history = read_component(&components[1])?;
    let metadata = read_component(&components[2])?;
    validate_keyed(&keyed_record, &schema.keyed_names)?;
    validate_history(&history, &schema.history_name)?;
    validate_metadata(&metadata)?;
    let keys = decode_keys(&keyed_record)?;
    let keyed = schema
        .keyed_names
        .iter()
        .enumerate()
        .map(|(index, name)| KeyedSnapshot {
            name: name.clone(),
            kind: ValueKind::UInt64,
            values: keyed_record.column(index + 1).clone(),
        })
        .collect();
    let ring = RingSnapshot {
        name: schema.history_name,
        kind: RingKind::Float64,
        retention: schema.retention,
        values: history.column(0).clone(),
        offsets: decode_usize(metadata.column(0))?,
        capacities: decode_usize(keyed_record.column(schema.keyed_names.len() + 1))?,
        numeric: None,
    };
    let logical_bytes = decode_logical_bytes(&keyed_record)?;
    let (snapshot, actual_bytes) = table.layout7_from_snapshot(keys, keyed, vec![ring])?;
    if actual_bytes != logical_bytes {
        return Err(mismatch(&format!(
            "rolling history layout-7 logical size differs from restored pages: encoded {logical_bytes}, restored {actual_bytes}"
        )));
    }
    Ok(snapshot)
}

fn decode_components(bytes: &[u8]) -> Result<Vec<Vec<u8>>> {
    if bytes.len() < HEADER_BYTES {
        return Err(mismatch("rolling history layout-7 header is truncated"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().expect("header length checked"));
    if usize::try_from(count).ok() != Some(COMPONENTS) {
        return Err(mismatch(
            "rolling history layout-7 component count mismatch",
        ));
    }
    let mut components = Vec::with_capacity(COMPONENTS);
    let mut offset = HEADER_BYTES;
    for index in 0..COMPONENTS {
        let start = 8 + 4 + index * 8;
        let len = usize::try_from(u64::from_le_bytes(
            bytes[start..start + 8]
                .try_into()
                .expect("header length checked"),
        ))
        .map_err(|_| mismatch("rolling history layout-7 component exceeds usize"))?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| mismatch("rolling history layout-7 component length overflowed"))?;
        if end > bytes.len() {
            return Err(mismatch("rolling history layout-7 component is truncated"));
        }
        components.push(bytes[offset..end].to_vec());
        offset = end;
    }
    if offset != bytes.len() {
        return Err(mismatch("rolling history layout-7 has trailing bytes"));
    }
    Ok(components)
}

fn validate_keyed(
    record: &datafusion::arrow::record_batch::RecordBatch,
    names: &[String],
) -> Result<()> {
    if record.num_columns() != names.len() + 2
        || record.schema().field(0).name() != "entity_key"
        || record.schema().field(0).data_type() != &DataType::Binary
    {
        return Err(mismatch(
            "rolling history layout-7 keyed inventory mismatch",
        ));
    }
    let schema = record.schema();
    for (index, name) in names.iter().enumerate() {
        let field = schema.field(index + 1);
        let values = record
            .column(index + 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| mismatch("rolling history layout-7 keyed type mismatch"))?;
        if field.name() != name
            || field.data_type() != &DataType::UInt64
            || values.null_count() != 0
        {
            return Err(mismatch(
                "rolling history layout-7 keyed inventory mismatch",
            ));
        }
    }
    let capacity = record
        .column(names.len() + 1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| mismatch("rolling history layout-7 capacity type mismatch"))?;
    if schema.field(names.len() + 1).name() != "ring_capacity"
        || schema.field(names.len() + 1).data_type() != &DataType::UInt64
        || capacity.null_count() != 0
    {
        return Err(mismatch("rolling history layout-7 capacity mismatch"));
    }
    Ok(())
}

fn validate_history(
    record: &datafusion::arrow::record_batch::RecordBatch,
    name: &str,
) -> Result<()> {
    if record.num_columns() != 1
        || record.schema().field(0).name() != name
        || record.schema().field(0).data_type() != &DataType::Float64
        || record
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .is_none()
    {
        return Err(mismatch("rolling history layout-7 history type mismatch"));
    }
    Ok(())
}

fn validate_metadata(record: &datafusion::arrow::record_batch::RecordBatch) -> Result<()> {
    if record.num_columns() != 1 {
        return Err(mismatch("rolling history layout-7 ring metadata mismatch"));
    }
    let schema = record.schema();
    let field = schema.field(0);
    let values = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| mismatch("rolling history layout-7 ring metadata type mismatch"))?;
    if field.name() != "ring_start"
        || field.data_type() != &DataType::UInt64
        || values.null_count() != 0
    {
        return Err(mismatch("rolling history layout-7 ring metadata mismatch"));
    }
    Ok(())
}

fn decode_usize(values: &ArrayRef) -> Result<Vec<usize>> {
    values
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| mismatch("rolling history layout-7 ring metadata type mismatch"))?
        .iter()
        .map(|value| {
            usize::try_from(value.ok_or_else(|| mismatch("rolling history layout-7 null offset"))?)
                .map_err(|_| mismatch("rolling history layout-7 offset exceeds usize"))
        })
        .collect()
}

fn usize_array(values: &[usize]) -> Result<ArrayRef> {
    let values = values
        .iter()
        .map(|value| {
            u64::try_from(*value)
                .map_err(|_| mismatch("rolling history layout-7 offset exceeds u64"))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(UInt64Array::from_iter_values(values)))
}

fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.to_owned(),
    }
}
