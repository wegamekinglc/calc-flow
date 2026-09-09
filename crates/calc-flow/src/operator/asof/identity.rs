use crate::{CalcFlowError, Result};
use datafusion::arrow::datatypes::{DataType, Schema};

// State layout 1 pins the non-null Arrow 58 row encoding used by its typed indexes.
pub(super) fn validate(mut bytes: &[u8], schema: &Schema, columns: &[String]) -> Result<()> {
    for column in columns {
        let data_type = schema
            .field_with_name(column)
            .map_err(|error| super::arrow_error(&error))?
            .data_type();
        if matches!(data_type, DataType::Utf8 | DataType::LargeUtf8) {
            string(&mut bytes)?;
        } else {
            primitive(&mut bytes, data_type)?;
        }
    }
    if bytes.is_empty() {
        Ok(())
    } else {
        Err(invalid())
    }
}

fn primitive(bytes: &mut &[u8], data_type: &DataType) -> Result<()> {
    let width = match data_type {
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 => 2,
        DataType::Int32 | DataType::UInt32 | DataType::Date32 => 4,
        DataType::Int64 | DataType::UInt64 | DataType::Date64 | DataType::Timestamp(_, _) => 8,
        _ => return Err(invalid()),
    };
    let encoded = take(bytes, width + 1)?;
    if encoded[0] != 1 || (data_type == &DataType::Boolean && encoded[1] > 1) {
        return Err(invalid());
    }
    Ok(())
}

fn string(bytes: &mut &[u8]) -> Result<()> {
    match take(bytes, 1)?[0] {
        1 => return Ok(()),
        2 => {}
        _ => return Err(invalid()),
    }
    let mut value = Vec::with_capacity(bytes.len());
    let mut block_index = 0;
    loop {
        let width = if block_index < 4 { 8 } else { 32 };
        let block = take(bytes, width + 1)?;
        let length = usize::from(block[width]);
        if length == 255 {
            value.extend_from_slice(&block[..width]);
        } else {
            value.extend_from_slice(string_tail(block, width, length)?);
            std::str::from_utf8(&value).map_err(|_| invalid())?;
            return Ok(());
        }
        block_index += 1;
    }
}

fn string_tail(block: &[u8], width: usize, length: usize) -> Result<&[u8]> {
    if length == 0 || length > width || block[length..width].iter().any(|byte| *byte != 0) {
        return Err(invalid());
    }
    Ok(&block[..length])
}

fn take<'a>(bytes: &mut &'a [u8], length: usize) -> Result<&'a [u8]> {
    if length > bytes.len() {
        return Err(invalid());
    }
    let (prefix, remaining) = bytes.split_at(length);
    *bytes = remaining;
    Ok(prefix)
}
fn invalid() -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: "ASOF identity is not a canonical non-null typed row encoding".into(),
    }
}
