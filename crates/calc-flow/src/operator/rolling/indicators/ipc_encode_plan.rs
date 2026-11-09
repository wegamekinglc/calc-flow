//! Pre-writer allocation bounds for the pinned default single-record IPC file.

use super::super::state_codec::FreshStateSchema;
use crate::Result;
use datafusion::arrow::{datatypes::Schema, record_batch::RecordBatch};

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

#[path = "ipc_encode_plan/math.rs"]
mod math;
#[path = "ipc_encode_plan/schema.rs"]
mod schema;
#[path = "ipc_encode_plan/workspace.rs"]
mod workspace;

#[path = "ipc_encode_plan/body.rs"]
mod body;
#[path = "ipc_encode_plan/geometry.rs"]
mod geometry;
#[path = "ipc_encode_plan/inventory.rs"]
mod inventory;

use math::{
    add, aligned, builder, elements, growing, hash_bytes, mul, size, sum, table, vec_elements,
    vector,
};

pub(super) fn planning_workspace(
    schema: &FreshStateSchema,
    record: &RecordBatch,
    check: Check<'_>,
) -> Result<u64> {
    workspace_fresh(schema.as_schema(), record, check)
}

fn workspace_fresh(schema: &Schema, record: &RecordBatch, check: Check<'_>) -> Result<u64> {
    check()?;
    i64::try_from(record.num_rows()).map_err(|_| math::invalid())?;
    if schema.fields().len() != record.num_columns() {
        return Err(math::invalid());
    }
    record.columns().iter().try_fold(0, |peak, column| {
        Ok(peak.max(workspace::array(column.as_ref(), check)?.metadata))
    })
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EncodingPlan {
    pub(super) encoded_quota: u64,
    pub(super) scratch_bytes: u64,
}

/// The caller reserves `planning_workspace` before this metadata-copying phase.
pub(super) fn plan(
    schema: &FreshStateSchema,
    record: &RecordBatch,
    check: Check<'_>,
) -> Result<EncodingPlan> {
    plan_fresh(schema.as_schema(), record, check)
}

fn plan_fresh(schema: &Schema, record: &RecordBatch, check: Check<'_>) -> Result<EncodingPlan> {
    let planner_workspace = workspace_fresh(schema, record, check)?;
    let schema_plan = schema::SchemaPlan::collect(schema, check)?;
    let (inventory, descriptors) = record_inventory(record, check)?;
    let writer = writer_plan(&schema_plan, &inventory, descriptors)?;
    check()?;
    Ok(EncodingPlan {
        encoded_quota: writer.encoded_quota,
        scratch_bytes: writer.scratch_bytes.max(planner_workspace),
    })
}

fn record_inventory(record: &RecordBatch, check: Check<'_>) -> Result<(inventory::Inventory, u64)> {
    let mut inventory = inventory::Inventory::default();
    let mut descriptors = 0;
    for column in record.columns() {
        check()?;
        let counts = workspace::array(column.as_ref(), check)?;
        descriptors = sum([Ok(descriptors), Ok(counts.metadata), Ok(counts.writer)])?;
        let data = column.to_data();
        inventory.column(&data, check)?;
    }
    Ok((inventory, descriptors))
}

struct Headers {
    schema: u64,
    record: u64,
    footer: u64,
}
impl Headers {
    fn new(schema: &schema::SchemaPlan, inventory: &inventory::Inventory) -> Result<Self> {
        Ok(Self {
            schema: sum([Ok(schema.bytes), table(5), Ok(16)])?,
            record: inventory.record.header(false)?,
            footer: sum([
                Ok(schema.bytes),
                vector(1, 24),
                vector(inventory.dictionary_count, 24),
                table(5),
                Ok(16),
            ])?,
        })
    }
    fn file_quota(&self, inventory: &inventory::Inventory) -> Result<u64> {
        let bytes = sum([
            Ok(64),
            frame(self.schema, 0),
            Ok(inventory.dictionary_file),
            frame(self.record, inventory.record.body),
            Ok(self.footer),
            Ok(18),
        ])?;
        growing(bytes, 1)
    }
    fn builders(
        &self,
        schema: &schema::SchemaPlan,
        inventory: &inventory::Inventory,
    ) -> Result<u64> {
        let tables = add(schema.tables, 1)?;
        sum([
            builder(self.schema, tables),
            builder(self.record.max(inventory.max_header), 3),
            builder(self.footer, tables),
        ])
    }
}
fn frame(header: u64, body: u64) -> Result<u64> {
    let framed = aligned(add(header, 8)?)?;
    let metadata = framed.checked_sub(8).ok_or_else(math::invalid)?;
    i32::try_from(metadata).map_err(|_| math::invalid())?;
    sum([Ok(framed), Ok(body)])
}

fn writer_plan(
    schema: &schema::SchemaPlan,
    inventory: &inventory::Inventory,
    descriptors: u64,
) -> Result<EncodingPlan> {
    let headers = Headers::new(schema, inventory)?;
    let scratch_bytes = writer_workspace(schema, inventory, (&headers, descriptors))?;
    Ok(EncodingPlan {
        encoded_quota: headers.file_quota(inventory)?,
        scratch_bytes,
    })
}
fn writer_workspace(
    schema: &schema::SchemaPlan,
    inventory: &inventory::Inventory,
    extra: (&Headers, u64),
) -> Result<u64> {
    let (headers, descriptors) = extra;
    sum([
        headers.builders(schema, inventory),
        Ok(schema.workspace),
        Ok(schema.clone_bytes),
        Ok(descriptors),
        Ok(headers.schema),
        Ok(headers.record),
        Ok(inventory.dictionary_headers),
        encoded_buffers(inventory.record.buffers),
        inventory.record.workspace(),
        Ok(inventory.temporaries),
        dictionary_storage(schema, inventory.dictionary_count),
        Ok(inventory.dictionary_owned),
    ])
}

fn encoded_buffers(count: u64) -> Result<u64> {
    use datafusion::arrow::buffer::Buffer;
    let width = size(size_of::<(Buffer, Vec<u8>, u64)>())?;
    growing(count, width)
}

fn dictionary_storage(schema: &schema::SchemaPlan, dictionaries: u64) -> Result<u64> {
    use datafusion::arrow::ipc::Block;
    sum([
        tracker_storage(dictionaries),
        vec_elements::<i64>(schema.dictionaries),
        elements::<i64>(schema.dictionaries),
        elements::<(Vec<u8>, Vec<u8>)>(schema.fields),
        elements::<(usize, usize)>(dictionaries),
        vec_elements::<Block>(dictionaries),
        vec_elements::<Block>(1),
        flatten_workspace(schema.fields),
    ])
}
fn flatten_workspace(fields: u64) -> Result<u64> {
    use datafusion::arrow::datatypes::Field;
    mul(vec_elements::<&Field>(fields)?, 3)
}
fn tracker_storage(count: u64) -> Result<u64> {
    use datafusion::arrow::array::ArrayData;
    let count = usize::try_from(count).map_err(|_| math::invalid())?;
    mul(hash_bytes(count, size_of::<(i64, ArrayData)>())?, 2)
}

#[cfg(test)]
#[path = "ipc_encode_plan/tests.rs"]
mod tests;
