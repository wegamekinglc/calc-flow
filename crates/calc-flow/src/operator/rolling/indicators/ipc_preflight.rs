//! Borrowed preflight for the canonical uncompressed rolling IPC file.

use datafusion::arrow::{datatypes::Schema as ArrowSchema, ipc};

use crate::{CalcFlowError, Result};

#[path = "ipc_preflight/layout.rs"]
mod layout;
#[path = "ipc_preflight/schema.rs"]
mod schema;

type Check<'a> = &'a mut dyn FnMut() -> Result<()>;
const ALIGNMENT: usize = 64;

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Span {
    pub(crate) offset: usize,
    pub(crate) length: usize,
}
impl Span {
    fn end(self) -> Result<usize> {
        add(self.offset, self.length)
    }
    pub(crate) fn bytes(self, source: &[u8]) -> Result<&[u8]> {
        source
            .get(self.offset..self.end()?)
            .ok_or_else(|| invalid("buffer bounds"))
    }
}

pub(crate) struct IpcEnvelopePlan<'a> {
    pub(crate) record_rows: usize,
    #[cfg(test)]
    pub(crate) record_body: Span,
    pub(crate) record_block: Span,
    #[cfg(test)]
    pub(crate) dictionary_body_bytes: usize,
    pub(crate) dictionary_block_bytes: usize,
    #[cfg(test)]
    pub(crate) metadata_bytes: usize,
    pub(crate) footer_bytes: usize,
    #[cfg(test)]
    pub(crate) field_count: usize,
    pub(crate) dictionary_count: usize,
    #[cfg(test)]
    pub(crate) metadata_pairs: usize,
    #[cfg(test)]
    pub(crate) metadata_string_bytes: usize,
    #[cfg(test)]
    pub(crate) field_name_bytes: usize,
    #[cfg(test)]
    pub(crate) timezone_string_bytes: usize,
    #[cfg(test)]
    pub(crate) union_type_ids: usize,
    #[cfg(test)]
    pub(crate) node_count: usize,
    #[cfg(test)]
    pub(crate) buffer_count: usize,
    pub(crate) variadic_count: usize,
    #[cfg(test)]
    pub(crate) entity_rows: usize,
    #[cfg(test)]
    pub(crate) history_rows: usize,
    #[cfg(test)]
    pub(crate) pending_rows: usize,
    #[cfg(test)]
    pub(crate) summary_rows: usize,
    #[cfg(test)]
    pub(crate) kinds: Span,
    #[cfg(test)]
    pub(crate) entity_ids: Span,
    #[cfg(test)]
    pub(crate) positions: Span,
    #[cfg(test)]
    pub(crate) position_validity: Span,
    footer: ipc::Footer<'a>,
}
impl<'a> IpcEnvelopePlan<'a> {
    pub(crate) const fn footer(&self) -> ipc::Footer<'a> {
        self.footer
    }

    #[cfg(test)]
    pub(crate) fn dictionary_bodies(&self) -> impl Iterator<Item = Span> + '_ {
        self.footer()
            .dictionaries()
            .into_iter()
            .flatten()
            .map(|block| Span {
                offset: usize::try_from(block.offset()).expect("checked block offset")
                    + usize::try_from(block.metaDataLength()).expect("checked metadata length"),
                length: usize::try_from(block.bodyLength()).expect("checked body length"),
            })
    }
}

pub(crate) fn preflight<'a>(
    bytes: &'a [u8],
    expected: &ArrowSchema,
    max_record_rows: u64,
    check: Check<'_>,
) -> Result<IpcEnvelopePlan<'a>> {
    let mut envelope = Envelope::new(bytes, expected, max_record_rows, check)?;
    envelope.dictionaries()?;
    envelope.record()
}

struct Envelope<'a, 's, 'c> {
    bytes: &'a [u8],
    expected: &'s ArrowSchema,
    check: Check<'c>,
    max_rows: u64,
    footer: ipc::Footer<'a>,
    footer_span: Span,
    schema_counts: schema::SchemaCounts,
    next: usize,
    metadata_bytes: usize,
    dictionary_body_bytes: usize,
    dictionary_block_bytes: usize,
    totals: layout::Counts,
}
impl<'a, 's, 'c> Envelope<'a, 's, 'c> {
    fn new(
        bytes: &'a [u8],
        expected: &'s ArrowSchema,
        max_rows: u64,
        check: Check<'c>,
    ) -> Result<Self> {
        check()?;
        let (footer, footer_span) = footer(bytes)?;
        let schema_counts = schema_inventory(footer, expected, check)?;
        let next = schema_header(bytes, expected, check)?;
        let metadata_bytes = add(footer_span.length, next - ALIGNMENT)?;
        Ok(Self {
            bytes,
            expected,
            check,
            max_rows,
            footer,
            footer_span,
            schema_counts,
            next,
            metadata_bytes,
            dictionary_body_bytes: 0,
            dictionary_block_bytes: 0,
            totals: layout::Counts::default(),
        })
    }
    fn dictionaries(&mut self) -> Result<()> {
        for (index, block) in self.footer.dictionaries().into_iter().flatten().enumerate() {
            (self.check)()?;
            let message = block_message(self.bytes, block, self.next, self.footer_span.offset)?;
            self.dictionary(index, message)?;
        }
        Ok(())
    }
    fn dictionary(&mut self, index: usize, message: BlockMessage<'a>) -> Result<()> {
        let dictionary = message
            .message
            .header_as_dictionary_batch()
            .ok_or_else(|| invalid("dictionary message"))?;
        validate_dictionary(dictionary, index)?;
        let data_type = schema::dictionary_value(self.expected, dictionary.id(), self.check)?
            .ok_or_else(|| invalid("dictionary schema ID"))?;
        let record = dictionary
            .data()
            .ok_or_else(|| invalid("dictionary record"))?;
        let counts = layout::dictionary(self.bytes, record, message.body, data_type, self.check)?;
        self.append_dictionary(message, counts)
    }
    fn append_dictionary(
        &mut self,
        message: BlockMessage<'_>,
        counts: layout::Counts,
    ) -> Result<()> {
        self.totals.merge(counts)?;
        self.dictionary_body_bytes = add(self.dictionary_body_bytes, message.body.length)?;
        self.dictionary_block_bytes = add(self.dictionary_block_bytes, message.block.length)?;
        self.metadata_bytes = add(self.metadata_bytes, message.metadata.length)?;
        self.next = message.block.end()?;
        Ok(())
    }
    fn record(mut self) -> Result<IpcEnvelopePlan<'a>> {
        let block = self
            .footer
            .recordBatches()
            .expect("checked record inventory")
            .get(0);
        let message = block_message(self.bytes, block, self.next, self.footer_span.offset)?;
        let record = message
            .message
            .header_as_record_batch()
            .ok_or_else(|| invalid("record message"))?;
        let rows = record_rows(record.length(), self.max_rows)?;
        let indexed = layout::record(self.bytes, record, message.body, self.expected, self.check)?;
        self.finish_record(&message, indexed.counts)?;
        Ok(self.into_plan(
            message,
            rows,
            #[cfg(test)]
            &indexed,
        ))
    }
    fn finish_record(&mut self, message: &BlockMessage<'_>, counts: layout::Counts) -> Result<()> {
        self.totals.merge(counts)?;
        self.metadata_bytes = add(self.metadata_bytes, message.metadata.length)?;
        let eos = Span {
            offset: message.block.end()?,
            length: 8,
        };
        validate_eos(self.bytes, eos, self.footer_span.offset)?;
        (self.check)()
    }
    fn into_plan(
        self,
        message: BlockMessage<'_>,
        rows: usize,
        #[cfg(test)] indexed: &layout::Indexed,
    ) -> IpcEnvelopePlan<'a> {
        IpcEnvelopePlan {
            record_rows: rows,
            #[cfg(test)]
            record_body: message.body,
            record_block: message.block,
            #[cfg(test)]
            dictionary_body_bytes: self.dictionary_body_bytes,
            dictionary_block_bytes: self.dictionary_block_bytes,
            #[cfg(test)]
            metadata_bytes: self.metadata_bytes,
            footer_bytes: self.footer_span.length,
            #[cfg(test)]
            field_count: self.schema_counts.fields,
            dictionary_count: self.schema_counts.dictionary_fields,
            #[cfg(test)]
            metadata_pairs: self.schema_counts.metadata_pairs,
            #[cfg(test)]
            metadata_string_bytes: self.schema_counts.metadata_string_bytes,
            #[cfg(test)]
            field_name_bytes: self.schema_counts.field_name_bytes,
            #[cfg(test)]
            timezone_string_bytes: self.schema_counts.timezone_string_bytes,
            #[cfg(test)]
            union_type_ids: self.schema_counts.union_type_ids,
            #[cfg(test)]
            node_count: self.totals.nodes,
            #[cfg(test)]
            buffer_count: self.totals.buffers,
            variadic_count: self.totals.variadic,
            #[cfg(test)]
            entity_rows: indexed.entities,
            #[cfg(test)]
            history_rows: indexed.histories,
            #[cfg(test)]
            pending_rows: indexed.pending,
            #[cfg(test)]
            summary_rows: indexed.summaries,
            #[cfg(test)]
            kinds: indexed.kinds,
            #[cfg(test)]
            entity_ids: indexed.entity_ids,
            #[cfg(test)]
            positions: indexed.positions,
            #[cfg(test)]
            position_validity: indexed.position_validity,
            footer: self.footer,
        }
    }
}

fn schema_inventory(
    footer: ipc::Footer<'_>,
    expected: &ArrowSchema,
    check: Check<'_>,
) -> Result<schema::SchemaCounts> {
    let actual = footer.schema().ok_or_else(|| invalid("footer schema"))?;
    let counts = schema::validate(actual, expected, check)?;
    let records = footer
        .recordBatches()
        .ok_or_else(|| invalid("record inventory"))?;
    if records.len() != 1 {
        return Err(invalid("requires exactly one record"));
    }
    if footer.dictionaries().map_or(0, |values| values.len()) != counts.dictionary_fields {
        return Err(invalid("dictionary inventory"));
    }
    Ok(counts)
}
fn validate_dictionary(dictionary: ipc::DictionaryBatch<'_>, index: usize) -> Result<()> {
    if dictionary.isDelta() || usize::try_from(dictionary.id()).ok() != Some(index) {
        return Err(invalid("dictionary ID or delta"));
    }
    Ok(())
}
fn record_rows(rows: i64, maximum: u64) -> Result<usize> {
    let rows = size(rows, "record rows")?;
    if rows == 0 || u64::try_from(rows).map_or(true, |rows| rows > maximum) {
        return Err(invalid("record rows exceed the selected limit"));
    }
    Ok(rows)
}
fn validate_eos(bytes: &[u8], eos: Span, footer_offset: usize) -> Result<()> {
    if eos.end()? != footer_offset || eos.bytes(bytes)? != [255, 255, 255, 255, 0, 0, 0, 0] {
        return Err(invalid("footer end-of-stream boundary"));
    }
    Ok(())
}

fn footer(bytes: &[u8]) -> Result<(ipc::Footer<'_>, Span)> {
    let span = footer_span(bytes)?;
    validate_header(bytes)?;
    let footer =
        ipc::root_as_footer(span.bytes(bytes)?).map_err(|_| invalid("footer flatbuffer"))?;
    validate_footer(footer)?;
    Ok((footer, span))
}

fn footer_span(bytes: &[u8]) -> Result<Span> {
    if bytes.len() < ALIGNMENT + 10 || bytes.get(bytes.len() - 6..) != Some(b"ARROW1") {
        return Err(invalid("footer magic or length"));
    }
    let length = i32::from_le_bytes(
        bytes[bytes.len() - 10..bytes.len() - 6]
            .try_into()
            .expect("four footer bytes"),
    );
    let length = usize::try_from(length).map_err(|_| invalid("footer length"))?;
    let offset = bytes
        .len()
        .checked_sub(10)
        .and_then(|end| end.checked_sub(length))
        .ok_or_else(|| invalid("footer bounds"))?;
    if offset < ALIGNMENT + 8 {
        return Err(invalid("footer bounds"));
    }
    Ok(Span { offset, length })
}

fn validate_header(bytes: &[u8]) -> Result<()> {
    if bytes.get(..6) != Some(b"ARROW1") || bytes[6..ALIGNMENT].iter().any(|byte| *byte != 0) {
        return Err(invalid("header magic or alignment"));
    }
    Ok(())
}

fn validate_footer(footer: ipc::Footer<'_>) -> Result<()> {
    if footer.version() != ipc::MetadataVersion::V5
        || footer
            .custom_metadata()
            .is_some_and(|metadata| !metadata.is_empty())
    {
        return Err(invalid("footer version or custom metadata"));
    }
    Ok(())
}

fn schema_header(bytes: &[u8], expected: &ArrowSchema, check: Check<'_>) -> Result<usize> {
    let (message, metadata) = schema_message(bytes)?;
    validate_schema_message(message, expected, check)?;
    aligned_metadata_end(metadata)
}
fn schema_message(bytes: &[u8]) -> Result<(ipc::Message<'_>, Span)> {
    let length = message_length(bytes, ALIGNMENT)?;
    let metadata = Span {
        offset: add(ALIGNMENT, 8)?,
        length,
    };
    let message = ipc::root_as_message(metadata.bytes(bytes)?)
        .map_err(|_| invalid("header schema message"))?;
    validate_message(message)?;
    Ok((message, metadata))
}
fn validate_schema_message(
    message: ipc::Message<'_>,
    expected: &ArrowSchema,
    check: Check<'_>,
) -> Result<()> {
    if message.bodyLength() != 0 {
        return Err(invalid("header schema body"));
    }
    let actual = message
        .header_as_schema()
        .ok_or_else(|| invalid("header schema"))?;
    schema::validate(actual, expected, check).map(|_| ())
}
fn aligned_metadata_end(metadata: Span) -> Result<usize> {
    let next = metadata.end()?;
    if !next.is_multiple_of(ALIGNMENT) {
        return Err(invalid("header schema alignment"));
    }
    Ok(next)
}

#[derive(Clone, Copy)]
struct BlockMessage<'a> {
    message: ipc::Message<'a>,
    metadata: Span,
    block: Span,
    body: Span,
}
fn block_message<'a>(
    bytes: &'a [u8],
    block: &ipc::Block,
    expected_offset: usize,
    footer_offset: usize,
) -> Result<BlockMessage<'a>> {
    let layout = block_layout(block, expected_offset, footer_offset)?;
    let message = parse_block_message(bytes, layout.metadata)?;
    validate_body_length(message, block.bodyLength())?;
    Ok(BlockMessage {
        message,
        metadata: layout.metadata,
        block: layout.block,
        body: layout.body,
    })
}
struct BlockLayout {
    block: Span,
    metadata: Span,
    body: Span,
}
fn block_layout(block: &ipc::Block, expected: usize, footer_offset: usize) -> Result<BlockLayout> {
    let (offset, metadata_length, body_length) = block_dimensions(block)?;
    validate_block_alignment(offset, expected, metadata_length, body_length)?;
    let span = Span {
        offset,
        length: add(metadata_length, body_length)?,
    };
    validate_block_bounds(span, footer_offset)?;
    let metadata = Span {
        offset,
        length: metadata_length,
    };
    let body = Span {
        offset: metadata.end()?,
        length: body_length,
    };
    Ok(BlockLayout {
        block: span,
        metadata,
        body,
    })
}
fn block_dimensions(block: &ipc::Block) -> Result<(usize, usize, usize)> {
    let offset = size(block.offset(), "block offset")?;
    let metadata =
        usize::try_from(block.metaDataLength()).map_err(|_| invalid("block metadata length"))?;
    let body = size(block.bodyLength(), "block body length")?;
    Ok((offset, metadata, body))
}
fn validate_block_alignment(
    offset: usize,
    expected: usize,
    metadata: usize,
    body: usize,
) -> Result<()> {
    if offset != expected
        || !offset.is_multiple_of(ALIGNMENT)
        || !metadata.is_multiple_of(ALIGNMENT)
        || !body.is_multiple_of(ALIGNMENT)
        || metadata < 8
    {
        return Err(invalid("block offset, length or alignment"));
    }
    Ok(())
}
fn validate_block_bounds(span: Span, footer_offset: usize) -> Result<()> {
    if span.end()? > footer_offset {
        return Err(invalid("block bounds"));
    }
    Ok(())
}
fn metadata_payload(bytes: &[u8], metadata: Span) -> Result<Span> {
    let length = message_length(bytes, metadata.offset)?;
    if add(length, 8)? != metadata.length {
        return Err(invalid("block metadata prefix length"));
    }
    Ok(Span {
        offset: add(metadata.offset, 8)?,
        length,
    })
}
fn parse_block_message(bytes: &[u8], metadata: Span) -> Result<ipc::Message<'_>> {
    let payload = metadata_payload(bytes, metadata)?;
    let message = ipc::root_as_message(payload.bytes(bytes)?)
        .map_err(|_| invalid("block message flatbuffer"))?;
    validate_message(message)?;
    Ok(message)
}
fn validate_body_length(message: ipc::Message<'_>, expected: i64) -> Result<()> {
    if message.bodyLength() != expected {
        return Err(invalid("block body length differs from message"));
    }
    Ok(())
}

fn message_length(bytes: &[u8], offset: usize) -> Result<usize> {
    let prefix = Span { offset, length: 8 }.bytes(bytes)?;
    if prefix[..4] != [255; 4] {
        return Err(invalid("message continuation prefix"));
    }
    let length = i32::from_le_bytes(prefix[4..].try_into().expect("four prefix bytes"));
    usize::try_from(length).map_err(|_| invalid("message length"))
}
fn validate_message(message: ipc::Message<'_>) -> Result<()> {
    if message.version() != ipc::MetadataVersion::V5
        || message
            .custom_metadata()
            .is_some_and(|metadata| !metadata.is_empty())
    {
        return Err(invalid("message version or metadata"));
    }
    Ok(())
}
fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("size overflow"))
}
fn multiply(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right)
        .ok_or_else(|| invalid("size overflow"))
}
fn size(value: i64, field: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid(field))
}
fn aligned(value: usize) -> Result<usize> {
    Ok(add(value, ALIGNMENT - 1)? / ALIGNMENT * ALIGNMENT)
}
fn invalid(field: &str) -> CalcFlowError {
    CalcFlowError::Format {
        message: format!("rolling state IPC {field} is invalid"),
    }
}

#[cfg(test)]
#[path = "ipc_preflight/tests.rs"]
mod tests;
