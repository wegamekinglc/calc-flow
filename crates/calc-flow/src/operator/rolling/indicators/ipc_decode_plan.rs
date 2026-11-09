//! Pre-admission bounds for the pinned, default uncompressed Arrow `FileReader`.

use datafusion::arrow::{
    array::ArrayRef,
    datatypes::{Field, Schema},
    ipc,
};

use super::ipc_preflight::{IpcEnvelopePlan, preflight};
use crate::{CalcFlowError, Result};

#[path = "ipc_decode_plan/arrays.rs"]
mod arrays;
#[path = "ipc_decode_plan/diagnostics.rs"]
mod diagnostics;
#[path = "ipc_decode_plan/dictionaries.rs"]
mod dictionaries;
#[path = "ipc_decode_plan/math.rs"]
mod math;
#[path = "ipc_decode_plan/messages.rs"]
mod messages;
#[path = "ipc_decode_plan/schema.rs"]
mod schema;
#[path = "ipc_decode_plan/walk.rs"]
mod walk;

use math::{add, buffer, grow, hash, index, scale, size, sum};
type Check<'a> = &'a mut dyn FnMut() -> Result<()>;

#[derive(Debug)]
pub(crate) struct DecodePlan {
    /// Retained payload/descriptors; excludes private immutable Buffer owner headers.
    pub(crate) reader_bytes: u64,
    /// Additional construction peak, including new private Buffer owner headers.
    pub(crate) scratch_bytes: u64,
}

pub(crate) fn plan(
    bytes: &[u8],
    expected: &Schema,
    envelope: &IpcEnvelopePlan<'_>,
    check: Check<'_>,
) -> Result<DecodePlan> {
    check()?;
    provenance(bytes, envelope)?;
    let verified = verify(bytes, expected, envelope, check)?;
    let mut cost = initial(&verified, check)?;
    dictionaries::append(bytes, expected, &verified, &mut cost, check)?;
    let graph = record_columns(bytes, expected, verified.footer(), check)?;
    finish(cost, graph, verified.variadic_count)
}
fn verify<'a>(
    bytes: &'a [u8],
    expected: &Schema,
    envelope: &IpcEnvelopePlan<'_>,
    check: Check<'_>,
) -> Result<IpcEnvelopePlan<'a>> {
    preflight(bytes, expected, size(envelope.record_rows)?, check)
}
fn initial(envelope: &IpcEnvelopePlan<'_>, check: Check<'_>) -> Result<DecodePlan> {
    let footer = envelope.footer();
    let schema = schema::inventory(footer.schema().expect("verified schema"), check)?;
    let metadata = schema::metadata(footer.custom_metadata(), check)?;
    let mut cost = base(envelope, schema, metadata)?;
    cost.scratch_bytes = sum([
        Ok(cost.scratch_bytes),
        diagnostics::plan(footer.schema().expect("verified schema"), check),
    ])?;
    Ok(cost)
}
fn finish(mut cost: DecodePlan, graph: arrays::Graph, variadic: usize) -> Result<DecodePlan> {
    add_graph(&mut cost, graph)?;
    cost.scratch_bytes = sum([Ok(cost.scratch_bytes), grow::<i64>(size(variadic)?)])?;
    Ok(cost)
}

fn provenance(bytes: &[u8], envelope: &IpcEnvelopePlan<'_>) -> Result<()> {
    let footer = footer_source(bytes, envelope)?;
    let original = envelope.footer()._tab.buf();
    if original.as_ptr() != footer.as_ptr() || original.len() != footer.len() {
        return Err(invalid("envelope source identity"));
    }
    Ok(())
}
fn footer_source<'a>(bytes: &'a [u8], envelope: &IpcEnvelopePlan<'_>) -> Result<&'a [u8]> {
    let start = sum([
        size(envelope.record_block.offset),
        size(envelope.record_block.length),
        Ok(8),
    ])?;
    let end = sum([Ok(start), size(envelope.footer_bytes)])?;
    if add(end, 10)? != size(bytes.len())? {
        return Err(invalid("envelope source length"));
    }
    borrowed_slice(bytes, start, end)
}

fn borrowed_slice(bytes: &[u8], start: u64, end: u64) -> Result<&[u8]> {
    bytes
        .get(index(start)?..index(end)?)
        .ok_or_else(|| invalid("footer span"))
}

fn base(
    envelope: &IpcEnvelopePlan<'_>,
    schema: schema::SchemaCost,
    metadata: schema::SchemaCost,
) -> Result<DecodePlan> {
    let dictionary_map = hash::<(i64, ArrayRef)>(size(envelope.dictionary_count)?)?;
    Ok(DecodePlan {
        reader_bytes: sum([
            backing(envelope),
            Ok(schema.retained),
            Ok(metadata.retained),
            Ok(dictionary_map),
            record_blocks(),
        ])?,
        scratch_bytes: sum([
            size(envelope.footer_bytes),
            construction_headers(envelope),
            Ok(schema.work),
            Ok(metadata.work),
            scale(Ok(dictionary_map), 2),
            grow::<&Field>(schema.flattened),
        ])?,
    })
}
fn record_blocks() -> Result<u64> {
    grow::<ipc::Block>(1)
}

fn backing(envelope: &IpcEnvelopePlan<'_>) -> Result<u64> {
    sum([
        size(envelope.dictionary_block_bytes),
        size(envelope.record_block.length),
    ])
}
fn construction_headers(envelope: &IpcEnvelopePlan<'_>) -> Result<u64> {
    let blocks = add(size(envelope.dictionary_count)?, 1)?;
    scale(buffer(0), blocks)
}

fn record_columns(
    bytes: &[u8],
    expected: &Schema,
    footer: ipc::Footer<'_>,
    check: Check<'_>,
) -> Result<arrays::Graph> {
    let block = footer.recordBatches().expect("verified record").get(0);
    let record = messages::record(bytes, block)?;
    let mut walk = walk::Walk {
        bytes,
        footer,
        record,
        node: 0,
        variadic: 0,
        check,
    };
    let mut graph = arrays::Graph::default();
    let fields = footer
        .schema()
        .expect("verified schema")
        .fields()
        .expect("verified fields");
    for (field, expected) in fields.iter().zip(expected.fields()) {
        graph.merge(walk.array(field, expected.data_type())?)?;
    }
    column_vector(graph, expected.fields().len())
}
fn column_vector(mut graph: arrays::Graph, len: usize) -> Result<arrays::Graph> {
    let columns = grow::<ArrayRef>(size(len)?)?;
    graph.retained = add(graph.retained, columns)?;
    graph.work = add(graph.work, columns)?;
    Ok(graph)
}

fn add_graph(cost: &mut DecodePlan, graph: arrays::Graph) -> Result<()> {
    cost.reader_bytes = add(cost.reader_bytes, graph.retained)?;
    cost.scratch_bytes = add(cost.scratch_bytes, graph.work)?;
    Ok(())
}
fn invalid(reason: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("rolling IPC decoding allocation plan: {reason}"),
    }
}

#[cfg(test)]
#[path = "ipc_decode_plan/tests.rs"]
mod tests;
