pub(super) mod accounting;
mod concat;
mod framing;
mod nodes;
mod reader;
mod schema;

pub(super) use framing::VerifierCredit;
pub(super) use reader::{Admission, decode};

use datafusion::arrow::{datatypes::Schema, ipc};

use super::geometry::invalid;
use crate::Result;

pub(super) struct Plan<'a> {
    schema: ipc::Schema<'a>,
    pub(super) dictionary_messages: usize,
    pub(super) body_bytes: usize,
    pub(super) nodes: usize,
    pub(super) buffers: usize,
    pub(super) buffer_bytes: usize,
    pub(super) variadic_counts: usize,
}

pub(super) fn inspect<'a>(
    bytes: &'a [u8],
    expected: &Schema,
    rows: usize,
    verifier: &VerifierCredit,
    check: &dyn Fn() -> Result<()>,
) -> Result<Plan<'a>> {
    let mut cursor = framing::Cursor::new(bytes);
    let schema = first_schema(&mut cursor, expected, verifier, check)?;
    let mut plan = Plan {
        schema,
        dictionary_messages: 0,
        body_bytes: 0,
        nodes: 0,
        buffers: 0,
        buffer_bytes: 0,
        variadic_counts: 0,
    };
    let mut record_seen = false;
    while let Some(message) = cursor.next(verifier, check)? {
        if record_seen {
            return Err(invalid("V2 IPC contains a message after its record batch"));
        }
        plan.message(&message, expected, rows, &mut record_seen, check)?;
    }
    if !record_seen || rows == 0 {
        return Err(invalid("V2 IPC must contain one nonempty record batch"));
    }
    Ok(plan)
}

fn first_schema<'a>(
    cursor: &mut framing::Cursor<'a>,
    expected: &Schema,
    verifier: &VerifierCredit,
    check: &dyn Fn() -> Result<()>,
) -> Result<ipc::Schema<'a>> {
    let first = cursor
        .next(verifier, check)?
        .ok_or_else(|| invalid("V2 IPC schema message is missing"))?;
    let schema = first
        .metadata
        .header_as_schema()
        .ok_or_else(|| invalid("V2 IPC first message must be a schema"))?;
    if !first.body.is_empty() {
        return Err(invalid("V2 IPC schema message must not have a body"));
    }
    schema::validate(schema, expected, check)?;
    Ok(schema)
}

impl Plan<'_> {
    fn message(
        &mut self,
        message: &framing::Message<'_>,
        expected: &Schema,
        rows: usize,
        record_seen: &mut bool,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let facts = match message.metadata.header_type() {
            ipc::MessageHeader::DictionaryBatch => {
                self.dictionary_messages = add(self.dictionary_messages, 1)?;
                self.dictionary_facts(message, expected, check)?
            }
            ipc::MessageHeader::RecordBatch => {
                *record_seen = true;
                let batch = message
                    .metadata
                    .header_as_record_batch()
                    .ok_or_else(|| invalid("V2 IPC record message is missing its header"))?;
                nodes::record(batch, message.body, expected, rows, check)?
            }
            _ => return Err(invalid("V2 IPC contains an unsupported message")),
        };
        self.add_facts(message.body.len(), &facts)
    }

    fn dictionary_facts(
        &self,
        message: &framing::Message<'_>,
        expected: &Schema,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<nodes::Facts> {
        let dictionary = message
            .metadata
            .header_as_dictionary_batch()
            .ok_or_else(|| invalid("V2 IPC dictionary message is missing its header"))?;
        let value = schema::dictionary_type(self.schema, expected, dictionary.id(), check)?;
        let batch = dictionary
            .data()
            .ok_or_else(|| invalid("V2 IPC dictionary message is missing its record batch"))?;
        nodes::dictionary(batch, message.body, value, check)
    }

    fn add_facts(&mut self, body_bytes: usize, facts: &nodes::Facts) -> Result<()> {
        self.body_bytes = add(self.body_bytes, body_bytes)?;
        self.nodes = add(self.nodes, facts.nodes)?;
        self.buffers = add(self.buffers, facts.buffers)?;
        self.buffer_bytes = add(self.buffer_bytes, facts.buffer_bytes)?;
        self.variadic_counts = add(self.variadic_counts, facts.variadic_counts)?;
        Ok(())
    }
}

fn add(left: usize, right: usize) -> Result<usize> {
    left.checked_add(right)
        .ok_or_else(|| invalid("V2 IPC constructor count overflow"))
}
