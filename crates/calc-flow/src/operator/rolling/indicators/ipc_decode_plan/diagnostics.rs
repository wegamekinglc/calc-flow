use datafusion::arrow::ipc;

use super::{
    Check, Result,
    math::{add, grow, mul, scale, size, sum},
};

#[derive(Clone, Copy, Default)]
struct Render {
    length: u64,
    work: u64,
}
impl Render {
    fn merge(&mut self, other: Self) -> Result<()> {
        self.length = add(self.length, other.length)?;
        self.work = add(self.work, other.work)?;
        Ok(())
    }
}

pub(super) fn plan(schema: ipc::Schema<'_>, check: Check<'_>) -> Result<u64> {
    let mut render = metadata(schema.custom_metadata(), check)?;
    for field in schema.fields().expect("verified fields") {
        render.merge(field_render(field, check)?)?;
    }
    // Reader validation errors format at most three types plus fixed/numeric text.
    sum([
        scale(Ok(render.work), 3),
        grow::<u8>(add(mul(render.length, 3)?, 512)?),
    ])
}

fn field_render(field: ipc::Field<'_>, check: Check<'_>) -> Result<Render> {
    check()?;
    let child = children(field, check)?;
    let mut own = metadata(field.custom_metadata(), check)?;
    own.length = sum([
        Ok(own.length),
        escaped(field.name().unwrap_or_default()),
        timezone(field),
        Ok(256),
    ])?;
    let length = add(own.length, child.length)?;
    let work = formatting_workspace(
        own,
        child,
        length,
        field.children().expect("verified children").len(),
    )?;
    Ok(Render { length, work })
}
fn children(field: ipc::Field<'_>, check: Check<'_>) -> Result<Render> {
    let mut result = Render::default();
    for child in field.children().expect("verified children") {
        result.merge(field_render(child, check)?)?;
    }
    Ok(result)
}
fn formatting_workspace(own: Render, child: Render, length: u64, fields: usize) -> Result<u64> {
    // datatype_display: format_field, optional Union wrapper, join, Vec<String>.
    sum([
        Ok(own.work),
        Ok(child.work),
        grow::<u8>(length),
        scale(grow::<u8>(child.length), 2),
        grow::<String>(size(fields)?),
    ])
}
fn timezone(field: ipc::Field<'_>) -> Result<u64> {
    field
        .type_as_timestamp()
        .and_then(|value| value.timezone())
        .map_or(Ok(0), escaped)
}
fn escaped(value: &str) -> Result<u64> {
    scale(size(value.len()), 12)
}

type Pairs<'a> = flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<ipc::KeyValue<'a>>>;
fn metadata(pairs: Option<Pairs<'_>>, check: Check<'_>) -> Result<Render> {
    let mut length = 0;
    for pair in pairs.into_iter().flatten() {
        check()?;
        length = sum([
            Ok(length),
            escaped(pair.key().unwrap_or_default()),
            escaped(pair.value().unwrap_or_default()),
            Ok(8),
        ])?;
    }
    metadata_workspace(length, pairs.map_or(0, |pairs| pairs.len()))
}
fn metadata_workspace(length: u64, pairs: usize) -> Result<Render> {
    // Field Display preformats metadata; datatype FormatMetadata also sorts refs.
    let work = sum([
        scale(grow::<u8>(length), 2),
        grow::<(&String, &String)>(size(pairs)?),
    ])?;
    Ok(Render { length, work })
}
