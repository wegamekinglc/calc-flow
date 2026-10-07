use super::Quantum;
use crate::{Result, StreamOperatorContext};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::{collections::HashMap, sync::Arc};

const ARC_HEADER: usize = 2 * size_of::<usize>();
const STRING_FRAGMENT: usize = 2_046;

fn map_inventory(metadata: &HashMap<String, String>) -> Option<usize> {
    // Nonempty caller maps have no proven bound on sparse-bucket iteration.
    metadata.is_empty().then_some(0)
}

fn timezone_inventory(data_type: &DataType) -> Option<usize> {
    match data_type {
        DataType::Timestamp(_, Some(timezone)) => (timezone.len() <= STRING_FRAGMENT)
            .then_some(())
            .and_then(|()| timezone.len().checked_mul(2))
            .and_then(|bytes| bytes.checked_add(ARC_HEADER + align_of::<usize>() - 1)),
        _ => Some(0),
    }
}

fn field_inventory(field: &Field) -> Option<usize> {
    (size_of::<Field>() + ARC_HEADER)
        .checked_add(field.name().len())?
        .checked_add(map_inventory(field.metadata())?)?
        .checked_add(timezone_inventory(field.data_type())?)
}

pub(in crate::operator::join) fn schema_inventory(schema: &Schema) -> Option<usize> {
    if schema.fields().len() > 256 {
        return None;
    }
    let fields = schema
        .fields()
        .len()
        .checked_mul(2 * size_of::<Arc<Field>>())?;
    let control = (size_of::<Schema>() + 2 * ARC_HEADER)
        .checked_add(fields)?
        .checked_add(map_inventory(schema.metadata())?)?;
    schema.fields().iter().try_fold(control, |bytes, field| {
        bytes.checked_add(field_inventory(field)?)
    })
}

async fn copy_text(
    source: &str,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<String> {
    let mut output = String::with_capacity(source.len());
    let mut start = 0;
    while start < source.len() {
        let mut end = (start + STRING_FRAGMENT).min(source.len());
        while !source.is_char_boundary(end) {
            end -= 1;
        }
        quantum.step(context, 8, 2 * (end - start) + 4).await?;
        output.push_str(&source[start..end]);
        start = end;
    }
    Ok(output)
}

async fn copy_type(
    source: &DataType,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<DataType> {
    if let DataType::Timestamp(unit, Some(timezone)) = source {
        let timezone = copy_text(timezone, context, quantum).await?;
        quantum.step(context, 8, 2 * timezone.len()).await?;
        return Ok(DataType::Timestamp(*unit, Some(Arc::from(timezone))));
    }
    Ok(source.clone())
}

pub(super) async fn copy_schema(
    source: &Schema,
    context: &StreamOperatorContext<'_>,
    quantum: &mut Quantum,
) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(source.fields().len());
    for source in source.fields() {
        let name = copy_text(source.name(), context, quantum).await?;
        let data_type = copy_type(source.data_type(), context, quantum).await?;
        quantum.step(context, 16, 0).await?;
        fields.push(Arc::new(Field::new(name, data_type, source.is_nullable())));
    }
    // Fields transfers initialized pointers without inspecting or cloning individual fields.
    quantum
        .step(context, 16, 2 * fields.len() * size_of::<Arc<Field>>())
        .await?;
    Ok(Arc::new(Schema::new(fields)))
}
