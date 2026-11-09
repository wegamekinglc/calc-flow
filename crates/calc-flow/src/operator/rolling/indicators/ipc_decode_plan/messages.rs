use datafusion::arrow::{
    datatypes::{DataType, Field},
    ipc,
};

use super::{
    Check, Result, invalid,
    math::{add, count, index},
};

pub(super) fn record<'a>(bytes: &'a [u8], block: &ipc::Block) -> Result<ipc::RecordBatch<'a>> {
    let message = message(bytes, block)?;
    message
        .header_as_record_batch()
        .ok_or_else(|| invalid("record message"))
}

pub(super) fn dictionary<'a>(
    bytes: &'a [u8],
    block: &ipc::Block,
) -> Result<ipc::DictionaryBatch<'a>> {
    message(bytes, block)?
        .header_as_dictionary_batch()
        .ok_or_else(|| invalid("dictionary message"))
}

fn message<'a>(bytes: &'a [u8], block: &ipc::Block) -> Result<ipc::Message<'a>> {
    let offset = count(block.offset())?;
    let len = u64::try_from(block.metaDataLength()).map_err(|_| invalid("metadata length"))?;
    let metadata = message_span(bytes, offset, len)?;
    ipc::root_as_message(metadata).map_err(|_| invalid("metadata flatbuffer"))
}

fn message_span(bytes: &[u8], offset: u64, len: u64) -> Result<&[u8]> {
    let start = index(add(offset, 8)?)?;
    let end = index(add(offset, len)?)?;
    bytes
        .get(start..end)
        .ok_or_else(|| invalid("metadata span"))
}

pub(super) fn visit_children<'a>(
    dtype: &'a DataType,
    visit: &mut dyn FnMut(&'a Field) -> Result<()>,
) -> Result<()> {
    use DataType as D;
    match dtype {
        D::Struct(fields) => fields.iter().try_for_each(|field| visit(field)),
        D::Union(fields, _) => fields.iter().try_for_each(|(_, field)| visit(field)),
        D::RunEndEncoded(ends, values) => {
            visit(ends)?;
            visit(values)
        }
        dtype => visit_single(dtype, visit),
    }
}
fn visit_single<'a>(
    dtype: &'a DataType,
    visit: &mut dyn FnMut(&'a Field) -> Result<()>,
) -> Result<()> {
    use DataType as D;
    match dtype {
        D::List(field)
        | D::LargeList(field)
        | D::ListView(field)
        | D::LargeListView(field)
        | D::Map(field, _)
        | D::FixedSizeList(field, _) => visit(field),
        _ => Ok(()),
    }
}

pub(super) fn dictionary_field<'a>(
    fields: impl Iterator<Item = ipc::Field<'a>>,
    id: i64,
    check: Check<'_>,
) -> Result<Option<ipc::Field<'a>>> {
    for field in fields {
        check()?;
        if field
            .dictionary()
            .is_some_and(|dictionary| dictionary.id() == id)
        {
            return Ok(Some(field));
        }
        if let Some(found) = dictionary_field(field.children().into_iter().flatten(), id, check)? {
            return Ok(Some(found));
        }
    }
    Ok(None)
}
