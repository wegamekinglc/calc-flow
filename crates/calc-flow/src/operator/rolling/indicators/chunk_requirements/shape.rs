use datafusion::arrow::{datatypes::DataType, record_batch::RecordBatch};

use super::{
    Check, Result,
    bytes::{add, invalid},
};

#[derive(Default)]
pub(super) struct Shape {
    pub descriptors: u64,
    pub slice_peak: u64,
    pub metadata: u64,
    pub nodes: usize,
}
pub(super) fn record(record: &RecordBatch, check: Check<'_>) -> Result<Shape> {
    let mut shape = Shape::default();
    for column in record.columns() {
        check()?;
        shape.include(column.as_ref(), check)?;
    }
    Ok(shape)
}
impl Shape {
    fn include(
        &mut self,
        array: &dyn datafusion::arrow::array::Array,
        check: Check<'_>,
    ) -> Result<()> {
        let dtype = array.data_type();
        let profile = super::super::scalar_materialization_plan::null_array_cost(dtype, 0, check)?;
        let logical = super::super::scalar_output_plan::slice_descriptor_requirement(array, check)?;
        self.descriptors = add(self.descriptors, profile.logical_owned.max(logical))?;
        self.slice_peak = self.slice_peak.max(profile.construction_peak);
        self.include_metadata(array, check)?;
        Ok(())
    }
    fn include_metadata(
        &mut self,
        array: &dyn datafusion::arrow::array::Array,
        check: Check<'_>,
    ) -> Result<()> {
        let (nodes, layouts) = tree(array.data_type(), check)?;
        self.nodes = self.nodes.checked_add(nodes).ok_or_else(invalid)?;
        let data = super::super::scalar_output_plan::array_data_workspace(array, check)?;
        self.metadata = self.metadata.max(add(data, layouts)?);
        Ok(())
    }
}
/// Each Arrow `get_slice_memory_size` call holds its own layout `Vec` while it
/// visits children. Summing all node layouts safely covers that nesting.
fn tree(dtype: &DataType, check: Check<'_>) -> Result<(usize, u64)> {
    check()?;
    let (nodes, layouts) = children(dtype, check)?;
    let own = super::super::scalar_materialization_plan::datatype_layout_requirement(dtype)?;
    Ok((
        nodes.checked_add(1).ok_or_else(invalid)?,
        add(own, layouts)?,
    ))
}
fn children(dtype: &DataType, check: Check<'_>) -> Result<(usize, u64)> {
    match dtype {
        DataType::Struct(fields) => fields.iter().try_fold((0, 0), |sum, field| {
            combine(sum, tree(field.data_type(), check)?)
        }),
        DataType::Union(fields, _) => fields.iter().try_fold((0, 0), |sum, (_, field)| {
            combine(sum, tree(field.data_type(), check)?)
        }),
        _ => encoded(dtype, check),
    }
}
fn encoded(dtype: &DataType, check: Check<'_>) -> Result<(usize, u64)> {
    match dtype {
        DataType::Dictionary(_, value) => tree(value, check),
        DataType::RunEndEncoded(ends, values) => combine(
            tree(ends.data_type(), check)?,
            tree(values.data_type(), check)?,
        ),
        _ => list(dtype, check),
    }
}
fn list(dtype: &DataType, check: Check<'_>) -> Result<(usize, u64)> {
    match dtype {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => tree(field.data_type(), check),
        _ => Ok((0, 0)),
    }
}
fn combine(left: (usize, u64), right: (usize, u64)) -> Result<(usize, u64)> {
    Ok((
        left.0.checked_add(right.0).ok_or_else(invalid)?,
        add(left.1, right.1)?,
    ))
}
