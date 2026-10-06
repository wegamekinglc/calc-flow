use super::{
    Arc, DFSchema, ExecutionProps, PhysicalExpr, RecordBatch, Result, SchemaRef,
    create_physical_expr, df_error, native_expression,
};
use datafusion::{
    arrow::{array::ArrayRef, datatypes::FieldRef},
    logical_expr::Aggregate,
};

#[derive(Debug)]
pub(super) struct GroupKey {
    pub expression: Arc<dyn PhysicalExpr>,
    pub field: FieldRef,
}

pub(super) fn bind(
    aggregate: &Aggregate,
    input: &DFSchema,
    schema: &SchemaRef,
    props: &ExecutionProps,
) -> Option<Vec<GroupKey>> {
    aggregate
        .group_expr
        .iter()
        .enumerate()
        .map(|(index, logical)| {
            native_expression::aggregate_filter_work(logical, aggregate.input.schema())?;
            let expression = create_physical_expr(logical, input, props).ok()?;
            let field = aggregate.schema.as_arrow().fields().get(index)?.clone();
            (super::key_type(field.data_type())
                && expression.data_type(schema).ok()? == *field.data_type())
            .then_some(GroupKey { expression, field })
        })
        .collect()
}

pub(super) fn evaluate(
    keys: &[GroupKey],
    batch: &RecordBatch,
    name: &str,
) -> Result<Vec<ArrayRef>> {
    keys.iter()
        .map(|key| {
            key.expression
                .evaluate(batch)
                .and_then(|value| value.into_array(batch.num_rows()))
                .map_err(|error| df_error(name, error))
        })
        .collect()
}
