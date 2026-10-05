use super::{
    ArrayRef, BooleanArray, DFSchema, ExecutionProps, Expr, NativeKeys, PhysicalExpr, RecordBatch,
    Result, SchemaRef, checked_bytes, create_physical_expr, df_error,
};
use datafusion::{
    arrow::{
        array::Array,
        compute::{and, filter},
        datatypes::DataType,
    },
    logical_expr::{LogicalPlan, Operator},
};
use std::sync::Arc;

#[derive(Clone)]
pub(super) struct InputPredicate {
    expression: Arc<dyn PhysicalExpr>,
    nodes: usize,
}

impl InputPredicate {
    pub fn supported(expression: &Expr, schema: &SchemaRef) -> bool {
        nodes(expression, schema, 0).is_some_and(|nodes| nodes <= 256)
    }

    pub fn new(
        expression: &Expr,
        input: &DFSchema,
        schema: &SchemaRef,
        props: &ExecutionProps,
    ) -> Option<Self> {
        let nodes = nodes(expression, schema, 0)?;
        let expression = create_physical_expr(expression, input, props).ok()?;
        (expression.data_type(schema).ok()? == DataType::Boolean)
            .then_some(Self { expression, nodes })
    }

    pub fn workspace(&self, rows: usize, aggregates: usize, name: &str) -> Result<usize> {
        let width = checked_bytes(64, [(self.nodes, 128), (aggregates, 2)], name)?;
        checked_bytes(4096, [(rows, width), (aggregates, 256)], name)
    }

    pub fn evaluate(&self, chunk: &RecordBatch, name: &str) -> Result<BooleanArray> {
        let array = self
            .expression
            .evaluate(chunk)
            .and_then(|value| value.into_array(chunk.num_rows()))
            .map_err(|error| df_error(name, error))?;
        array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .cloned()
            .ok_or_else(|| df_error(name, "SQL predicate is not Boolean"))
    }
}

pub(super) fn plan_nodes(input: &LogicalPlan, schema: &SchemaRef) -> usize {
    match input {
        LogicalPlan::Filter(filter) => nodes(&filter.predicate, schema, 0).unwrap_or(0),
        _ => 0,
    }
}

fn scalar_type(dtype: &DataType) -> bool {
    dtype.primitive_width().is_some()
        || matches!(
            dtype,
            DataType::Boolean | DataType::Null | DataType::Utf8 | DataType::LargeUtf8
        )
}

fn nodes(expression: &Expr, schema: &SchemaRef, depth: usize) -> Option<usize> {
    if depth > 32 {
        return None;
    }
    let children = match expression {
        Expr::Column(column) => {
            scalar_type(schema.field_with_name(&column.name).ok()?.data_type()).then_some(0)
        }
        Expr::Literal(value, _) => scalar_type(&value.data_type()).then_some(0),
        Expr::BinaryExpr(binary)
            if matches!(
                binary.op,
                Operator::Eq
                    | Operator::NotEq
                    | Operator::Lt
                    | Operator::LtEq
                    | Operator::Gt
                    | Operator::GtEq
                    | Operator::And
                    | Operator::Or
            ) =>
        {
            nodes(&binary.left, schema, depth + 1)?.checked_add(nodes(
                &binary.right,
                schema,
                depth + 1,
            )?)
        }
        Expr::Not(inner)
        | Expr::IsNull(inner)
        | Expr::IsNotNull(inner)
        | Expr::IsTrue(inner)
        | Expr::IsNotTrue(inner)
        | Expr::IsFalse(inner)
        | Expr::IsNotFalse(inner)
        | Expr::IsUnknown(inner)
        | Expr::IsNotUnknown(inner) => nodes(inner, schema, depth + 1),
        Expr::Cast(cast) if scalar_type(cast.field.data_type()) => {
            nodes(&cast.expr, schema, depth + 1)
        }
        Expr::TryCast(cast) if scalar_type(cast.field.data_type()) => {
            nodes(&cast.expr, schema, depth + 1)
        }
        _ => None,
    }?;
    children.checked_add(1)
}

pub(super) fn selected(selection: &BooleanArray, row: usize) -> bool {
    selection.is_valid(row) && selection.value(row)
}

pub(super) fn combine(
    selection: Option<&BooleanArray>,
    filters: &[Option<&BooleanArray>],
    count: usize,
    name: &str,
) -> Result<Vec<Option<BooleanArray>>> {
    let Some(selection) = selection else {
        return Ok(Vec::new());
    };
    (0..count)
        .map(|index| {
            match filters.get(index).copied().flatten() {
                Some(filter) => and(selection, filter).map_err(|error| df_error(name, error)),
                None => Ok(selection.clone()),
            }
            .map(Some)
        })
        .collect()
}

impl NativeKeys {
    pub(super) fn intern_selected(
        &mut self,
        array: ArrayRef,
        selection: Option<&BooleanArray>,
        name: &str,
    ) -> Result<()> {
        let Some(selection) = selection else {
            return self.intern(array, name);
        };
        let array = filter(array.as_ref(), selection).map_err(|error| df_error(name, error))?;
        self.intern(array, name)?;
        let mut selected_indices = self.indices.iter();
        self.indices = selection
            .iter()
            .map(|selected| {
                if selected == Some(true) {
                    *selected_indices.next().expect("selected native key index")
                } else {
                    0
                }
            })
            .collect();
        self.validate_capacity(name)
    }
}
