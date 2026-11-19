use std::{collections::HashMap, io::Cursor, sync::Arc};

use datafusion::arrow::{
    array::{Array, ArrayRef, BinaryArray, UInt64Array},
    datatypes::{DataType, Field, Schema},
    ipc::{reader::FileReader, writer::FileWriter},
    record_batch::RecordBatch,
};

use super::{
    EntityKey, StateTable, ValueKind,
    core::{KeyedSnapshot, TableSnapshot},
};
use crate::{CalcFlowError, Result};

const MAGIC: &[u8; 8] = b"CFST7K\0\0";
const HEADER_BYTES: usize = 20;

pub(super) fn encode(table: &StateTable) -> Option<Result<Vec<u8>>> {
    let names = table.layout7_keyed_schema()?;
    Some(encode_keyed(table, &names))
}

pub(super) fn decode(table: &StateTable, bytes: &[u8]) -> Option<Result<TableSnapshot>> {
    if bytes.get(..MAGIC.len()) != Some(MAGIC.as_slice()) {
        return None;
    }
    Some(decode_keyed(table, bytes))
}

fn encode_keyed(table: &StateTable, names: &[String]) -> Result<Vec<u8>> {
    let snapshot = table.snapshot_for_layout7();
    let fields = std::iter::once(Field::new("entity_key", DataType::Binary, false))
        .chain(
            names
                .iter()
                .map(|name| Field::new(name, DataType::UInt64, false)),
        )
        .collect::<Vec<_>>();
    let columns = std::iter::once(Arc::new(BinaryArray::from_iter_values(
        snapshot.keys.iter().map(EntityKey::bytes),
    )) as ArrayRef)
    .chain(snapshot.keyed.iter().map(|column| column.values.clone()))
    .collect::<Vec<_>>();
    let metadata = HashMap::from([(
        "logical_bytes".to_owned(),
        snapshot.logical_bytes.to_string(),
    )]);
    let component = write_component(fields, columns, metadata)?;
    let component_len = u64::try_from(component.len())
        .map_err(|_| mismatch("rolling keyed layout-7 component exceeds u64"))?;
    let mut encoded = Vec::with_capacity(HEADER_BYTES + component.len());
    encoded.extend_from_slice(MAGIC);
    encoded.extend_from_slice(&1_u32.to_le_bytes());
    encoded.extend_from_slice(&component_len.to_le_bytes());
    encoded.extend_from_slice(&component);
    Ok(encoded)
}

fn decode_keyed(table: &StateTable, bytes: &[u8]) -> Result<TableSnapshot> {
    let names = table
        .layout7_keyed_schema()
        .ok_or_else(|| mismatch("rolling keyed layout-7 schema mismatch"))?;
    let component = decode_component(bytes)?;
    let record = read_component(&component)?;
    validate_record(&record, &names)?;
    let keys = decode_keys(&record)?;
    let keyed = names
        .iter()
        .enumerate()
        .map(|(index, name)| KeyedSnapshot {
            name: name.clone(),
            kind: ValueKind::UInt64,
            values: record.column(index + 1).clone(),
        })
        .collect();
    let logical_bytes = decode_logical_bytes(&record)?;
    let (snapshot, actual_bytes) = table.layout7_from_snapshot(keys, keyed, Vec::new())?;
    if actual_bytes != logical_bytes {
        return Err(mismatch(&format!(
            "rolling keyed layout-7 logical size differs from restored pages: encoded {logical_bytes}, restored {actual_bytes}"
        )));
    }
    Ok(snapshot)
}

fn decode_component(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < HEADER_BYTES {
        return Err(mismatch("rolling keyed layout-7 header is truncated"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().expect("header length checked"));
    if count != 1 {
        return Err(mismatch(
            "rolling keyed layout-7 component count mismatch; start a new lineage",
        ));
    }
    let component_len = usize::try_from(u64::from_le_bytes(
        bytes[12..20].try_into().expect("header length checked"),
    ))
    .map_err(|_| mismatch("rolling keyed layout-7 component exceeds usize"))?;
    let end = HEADER_BYTES
        .checked_add(component_len)
        .ok_or_else(|| mismatch("rolling keyed layout-7 component length overflowed"))?;
    if end != bytes.len() {
        return Err(mismatch(
            "rolling keyed layout-7 component is truncated or has trailing bytes",
        ));
    }
    Ok(bytes[HEADER_BYTES..].to_vec())
}

fn read_component(bytes: &[u8]) -> Result<RecordBatch> {
    let mut reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|_| mismatch("rolling keyed layout-7 component is invalid"))?;
    let record = reader
        .next()
        .transpose()
        .map_err(|error| mismatch(&error.to_string()))?
        .ok_or_else(|| mismatch("rolling keyed layout-7 component is empty"))?;
    if reader
        .next()
        .transpose()
        .map_err(|error| mismatch(&error.to_string()))?
        .is_some()
    {
        return Err(mismatch(
            "rolling keyed layout-7 component contains multiple batches",
        ));
    }
    Ok(record)
}

fn validate_record(record: &RecordBatch, names: &[String]) -> Result<()> {
    let schema = record.schema();
    let fields = schema.fields();
    if fields.len() != names.len() + 1
        || fields[0].name() != "entity_key"
        || fields[0].data_type() != &DataType::Binary
    {
        return Err(mismatch("rolling keyed layout-7 column inventory mismatch"));
    }
    for (index, name) in names.iter().enumerate() {
        let field = &fields[index + 1];
        let values = record
            .column(index + 1)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| mismatch("rolling keyed layout-7 column type mismatch"))?;
        if field.name() != name
            || field.data_type() != &DataType::UInt64
            || values.null_count() != 0
        {
            return Err(mismatch("rolling keyed layout-7 column inventory mismatch"));
        }
    }
    Ok(())
}

fn decode_keys(record: &RecordBatch) -> Result<Vec<EntityKey>> {
    record
        .column(0)
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| mismatch("rolling keyed layout-7 entity key type mismatch"))?
        .iter()
        .map(|key| key.map(|key| EntityKey::new(key.to_vec())))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| mismatch("rolling keyed layout-7 entity key is null"))
}

fn decode_logical_bytes(record: &RecordBatch) -> Result<usize> {
    let schema = record.schema();
    let metadata = schema.metadata();
    if metadata.len() != 1 {
        return Err(mismatch("rolling keyed layout-7 metadata mismatch"));
    }
    metadata
        .get("logical_bytes")
        .ok_or_else(|| mismatch("rolling keyed layout-7 logical size is absent"))?
        .parse::<usize>()
        .map_err(|_| mismatch("rolling keyed layout-7 logical size is invalid"))
}

fn write_component(
    fields: Vec<Field>,
    columns: Vec<ArrayRef>,
    metadata: HashMap<String, String>,
) -> Result<Vec<u8>> {
    let schema = Schema::new(fields).with_metadata(metadata);
    let record = RecordBatch::try_new(Arc::new(schema), columns)
        .map_err(|error| mismatch(&error.to_string()))?;
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, record.schema().as_ref())
        .map_err(|error| mismatch(&error.to_string()))?;
    writer
        .write(&record)
        .map_err(|error| mismatch(&error.to_string()))?;
    writer
        .finish()
        .map_err(|error| mismatch(&error.to_string()))?;
    drop(writer);
    Ok(bytes)
}

fn mismatch(message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: message.to_owned(),
    }
}
