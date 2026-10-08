use super::super::copy_boundary;
use crate::runtime::streaming::gather_work::GatherStop;
use crate::{Result, StreamJobContext};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use std::sync::Arc;

pub(super) const MAX_FIELDS: usize = 32;
pub(super) const MAX_TEXT_BYTES: usize = 2_046;

pub(super) struct FieldPlan {
    name: String,
    data_type: DataType,
    timezone: Option<String>,
    nullable: bool,
}

pub(super) fn scalar_type(source: &DataType) -> Option<DataType> {
    match source {
        DataType::Null
        | DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::Date32
        | DataType::Date64 => Some(source.clone()),
        DataType::Timestamp(unit, _) => Some(DataType::Timestamp(*unit, None)),
        _ => None,
    }
}

pub(super) fn timezone(source: &DataType) -> Option<&str> {
    match source {
        DataType::Timestamp(_, timezone) => timezone.as_deref(),
        _ => None,
    }
}

impl FieldPlan {
    fn empty(source: &Field) -> Self {
        Self {
            name: String::with_capacity(source.name().len()),
            data_type: scalar_type(source.data_type()).expect("checked flat descriptor type"),
            timezone: timezone(source.data_type()).map(|text| String::with_capacity(text.len())),
            nullable: source.is_nullable(),
        }
    }

    fn into_field(self) -> Field {
        let data_type = match self.data_type {
            DataType::Timestamp(unit, _) => DataType::Timestamp(unit, self.timezone.map(Arc::from)),
            data_type => data_type,
        };
        Field::new(self.name, data_type, self.nullable)
    }
}

pub(super) async fn copy(
    output: &mut [Vec<FieldPlan>; 2],
    source: [&Schema; 2],
    job: &StreamJobContext,
) -> Result<()> {
    for (plans, schema) in output.iter_mut().zip(source) {
        *plans = Vec::with_capacity(schema.fields().len());
        for field in schema.fields() {
            copy_boundary(job).await?;
            plans.push(FieldPlan::empty(field));
            let plan = plans.last_mut().expect("owned partial field plan");
            copy_boundary(job).await?;
            plan.name.push_str(field.name());
            if let Some(text) = timezone(field.data_type()) {
                copy_boundary(job).await?;
                plan.timezone
                    .as_mut()
                    .expect("owned timezone plan")
                    .push_str(text);
            }
        }
    }
    Ok(())
}

pub(super) fn build(plans: Vec<FieldPlan>, stop: &GatherStop) -> Result<SchemaRef> {
    let mut fields = Vec::with_capacity(plans.len());
    for plan in plans {
        stop.check()?;
        fields.push(Arc::new(plan.into_field()));
    }
    Ok(Arc::new(Schema::new(fields)))
}
