use std::sync::Arc;

use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field, Schema},
    ipc,
};

use super::super::ipc_preflight::IpcEnvelopePlan;
use super::{
    Check, DecodePlan, Result, add_graph, invalid,
    math::{arc, count, grow, index, slice_arc, sum},
    messages, schema,
    walk::Walk,
};

pub(super) fn append(
    bytes: &[u8],
    expected: &Schema,
    envelope: &IpcEnvelopePlan<'_>,
    cost: &mut DecodePlan,
    check: Check<'_>,
) -> Result<()> {
    let mut inventory = Inventory {
        bytes,
        footer: envelope.footer(),
        cost,
        check,
        id: 0,
    };
    for field in expected.fields() {
        inventory.dtype(field.data_type())?;
    }
    Ok(())
}
struct Inventory<'a, 'c> {
    bytes: &'a [u8],
    footer: ipc::Footer<'a>,
    cost: &'c mut DecodePlan,
    check: Check<'c>,
    id: i64,
}
impl Inventory<'_, '_> {
    fn dtype(&mut self, dtype: &DataType) -> Result<()> {
        (self.check)()?;
        if let DataType::Dictionary(_, value) = dtype {
            messages::visit_children(value, &mut |field| self.dtype(field.data_type()))?;
            dictionary(
                self.bytes,
                self.footer,
                self.id,
                value,
                self.cost,
                self.check,
            )?;
            self.id = self
                .id
                .checked_add(1)
                .ok_or_else(|| invalid("dictionary id"))?;
            return Ok(());
        }
        messages::visit_children(dtype, &mut |field| self.dtype(field.data_type()))
    }
}

fn dictionary(
    bytes: &[u8],
    footer: ipc::Footer<'_>,
    id: i64,
    dtype: &DataType,
    cost: &mut DecodePlan,
    check: Check<'_>,
) -> Result<()> {
    check()?;
    let record = dictionary_record(bytes, footer, id)?;
    let fields = footer
        .schema()
        .expect("verified schema")
        .fields()
        .expect("verified fields");
    let field =
        messages::dictionary_field(fields.iter(), id, check)?.expect("verified dictionary field");
    let mut walk = Walk {
        bytes,
        footer,
        record,
        node: 0,
        variadic: 0,
        check,
    };
    add_graph(cost, walk.array(field, dtype)?)?;
    cost.scratch_bytes = sum([Ok(cost.scratch_bytes), temporary_schema(dtype)])?;
    Ok(())
}
fn dictionary_record<'a>(
    bytes: &'a [u8],
    footer: ipc::Footer<'_>,
    id: i64,
) -> Result<ipc::RecordBatch<'a>> {
    let block = footer
        .dictionaries()
        .expect("verified dictionaries")
        .get(index(count(id)?)?);
    Ok(messages::dictionary(bytes, block)?
        .data()
        .expect("verified data"))
}
fn temporary_schema(dtype: &DataType) -> Result<u64> {
    sum([
        arc::<Schema>(),
        arc::<Field>(),
        grow::<Field>(1),
        slice_arc::<Arc<Field>>(1),
        schema::clone_type(dtype),
        grow::<ArrayRef>(1),
        grow::<Arc<Field>>(1),
    ])
}
