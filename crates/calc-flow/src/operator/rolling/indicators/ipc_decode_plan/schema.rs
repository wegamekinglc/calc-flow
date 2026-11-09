use std::sync::Arc;

use datafusion::arrow::{
    datatypes::{DataType, Field, Schema},
    ipc,
};

use super::{
    Check, Result,
    math::{add, arc, bytes, grow, hash, mul, size, slice_arc, sum},
};

#[derive(Clone, Copy, Default)]
pub(super) struct SchemaCost {
    pub(super) retained: u64,
    pub(super) work: u64,
    pub(super) flattened: u64,
}
impl SchemaCost {
    fn merge(&mut self, other: Self) -> Result<()> {
        self.retained = add(self.retained, other.retained)?;
        self.work = add(self.work, other.work)?;
        self.flattened = add(self.flattened, other.flattened)?;
        Ok(())
    }
}
pub(super) fn inventory(schema: ipc::Schema<'_>, check: Check<'_>) -> Result<SchemaCost> {
    let fields = schema.fields().expect("verified schema fields");
    let mut cost = schema_root(schema, check)?;
    for field in fields {
        cost.merge(field_cost(field, check)?)?;
    }
    Ok(cost)
}
fn schema_root(schema: ipc::Schema<'_>, check: Check<'_>) -> Result<SchemaCost> {
    let mut cost = fields_cost(schema.fields().expect("verified fields").len())?;
    cost.retained = sum([Ok(cost.retained), arc::<Schema>()])?;
    cost.merge(metadata(schema.custom_metadata(), check)?)?;
    Ok(cost)
}
fn append_children(cost: &mut SchemaCost, field: ipc::Field<'_>, check: Check<'_>) -> Result<()> {
    let children = field.children().expect("verified children");
    cost.merge(fields_cost(children.len())?)?;
    for child in children {
        cost.merge(child_cost(child, check)?)?;
    }
    Ok(())
}

fn field_cost(field: ipc::Field<'_>, check: Check<'_>) -> Result<SchemaCost> {
    check()?;
    let mut cost = identity(field, check)?;
    append_children(&mut cost, field, check)?;
    cost.work = sum([Ok(cost.work), grow::<&Field>(cost.flattened)])?;
    Ok(cost)
}
fn identity(field: ipc::Field<'_>, check: Check<'_>) -> Result<SchemaCost> {
    let mut cost = metadata(field.custom_metadata(), check)?;
    cost.retained = sum([
        Ok(cost.retained),
        arc::<Field>(),
        size(field.name().unwrap_or_default().len()),
        type_storage(field),
    ])?;
    cost.flattened = 1;
    Ok(cost)
}
fn child_cost(field: ipc::Field<'_>, check: Check<'_>) -> Result<SchemaCost> {
    let mut cost = field_cost(field, check)?;
    cost.work = sum([Ok(cost.work), grow::<&Field>(cost.flattened)])?;
    Ok(cost)
}
fn fields_cost(len: usize) -> Result<SchemaCost> {
    let len = size(len)?;
    Ok(SchemaCost {
        retained: slice_arc::<(i8, Arc<Field>)>(len)?,
        work: sum([grow::<Field>(len), grow::<(i8, Arc<Field>)>(len)])?,
        flattened: 0,
    })
}
fn type_storage(field: ipc::Field<'_>) -> Result<u64> {
    let dictionary = if field.dictionary().is_some() {
        bytes::<DataType>(2)?
    } else {
        0
    };
    let timezone = field
        .type_as_timestamp()
        .and_then(|timestamp| timestamp.timezone());
    add(
        dictionary,
        timezone.map_or(Ok(0), |value| slice_arc::<u8>(size(value.len())?))?,
    )
}
type Pairs<'a> = flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<ipc::KeyValue<'a>>>;
pub(super) fn metadata(pairs: Option<Pairs<'_>>, check: Check<'_>) -> Result<SchemaCost> {
    let len = pairs.map_or(0, |pairs| pairs.len());
    let table = hash::<(String, String)>(size(len)?)?;
    let mut cost = SchemaCost {
        retained: table,
        work: mul(table, 2)?,
        flattened: 0,
    };
    for pair in pairs.into_iter().flatten() {
        cost.retained = sum([Ok(cost.retained), pair_strings(pair, check)])?;
    }
    Ok(cost)
}
fn pair_strings(pair: ipc::KeyValue<'_>, check: Check<'_>) -> Result<u64> {
    check()?;
    sum([
        size(pair.key().unwrap_or_default().len()),
        size(pair.value().unwrap_or_default().len()),
    ])
}
pub(super) fn clone_type(dtype: &DataType) -> Result<u64> {
    match dtype {
        DataType::Dictionary(key, value) => {
            sum([bytes::<DataType>(2), clone_type(key), clone_type(value)])
        }
        _ => Ok(0),
    }
}
