use datafusion::arrow::{
    datatypes::{DataType, Field, Fields, Schema, UnionFields, UnionMode},
    ipc,
};

use super::super::geometry::invalid;
use crate::Result;

#[derive(Default)]
pub(super) struct Facts {
    pub(super) nodes: usize,
    pub(super) buffers: usize,
    pub(super) buffer_bytes: usize,
    pub(super) variadic_counts: usize,
}

pub(super) fn record(
    batch: ipc::RecordBatch<'_>,
    body: &[u8],
    schema: &Schema,
    rows: usize,
    check: &dyn Fn() -> Result<()>,
) -> Result<Facts> {
    let mut cursor = Cursor::new(batch, body, rows, check)?;
    for field in schema.fields() {
        let length = cursor.field(field)?;
        if length != rows {
            return Err(invalid(
                "V2 IPC column length differs from its record batch",
            ));
        }
    }
    cursor.finish()
}

pub(super) fn dictionary(
    batch: ipc::RecordBatch<'_>,
    body: &[u8],
    data_type: &DataType,
    check: &dyn Fn() -> Result<()>,
) -> Result<Facts> {
    let rows = count(batch.length())?;
    let mut cursor = Cursor::new(batch, body, rows, check)?;
    let length = cursor.array(data_type)?;
    if length != rows {
        return Err(invalid("V2 IPC dictionary length differs from its batch"));
    }
    cursor.finish()
}

struct Cursor<'a, 'c> {
    batch: ipc::RecordBatch<'a>,
    body: &'a [u8],
    node: usize,
    buffer: usize,
    variadic: usize,
    facts: Facts,
    check: &'c dyn Fn() -> Result<()>,
}

impl<'a, 'c> Cursor<'a, 'c> {
    fn new(
        batch: ipc::RecordBatch<'a>,
        body: &'a [u8],
        rows: usize,
        check: &'c dyn Fn() -> Result<()>,
    ) -> Result<Self> {
        check()?;
        if batch.compression().is_some() || count(batch.length())? != rows {
            return Err(invalid("V2 IPC batch length or compression is invalid"));
        }
        if batch.nodes().is_none() || batch.buffers().is_none() {
            return Err(invalid("V2 IPC nodes or buffers are missing"));
        }
        Ok(Self {
            batch,
            body,
            node: 0,
            buffer: 0,
            variadic: 0,
            facts: Facts::default(),
            check,
        })
    }

    fn field(&mut self, field: &Field) -> Result<usize> {
        (self.check)()?;
        let nulls = self
            .batch
            .nodes()
            .and_then(|nodes| (self.node < nodes.len()).then(|| nodes.get(self.node)))
            .ok_or_else(|| invalid("V2 IPC field node is missing"))?
            .null_count();
        if !field.is_nullable() && nulls != 0 {
            return Err(invalid("V2 IPC non-nullable field contains nulls"));
        }
        self.array(field.data_type())
    }

    fn array(&mut self, data_type: &DataType) -> Result<usize> {
        let (length, nulls) = self.next_node()?;
        if self.primitive(data_type, length, nulls)? {
            return Ok(length);
        }
        if self.leaf(data_type, length, nulls)? {
            return Ok(length);
        }
        self.nested(data_type, length, nulls)?;
        Ok(length)
    }

    fn primitive(&mut self, data_type: &DataType, length: usize, nulls: usize) -> Result<bool> {
        let Some(width) = data_type.primitive_width() else {
            return Ok(false);
        };
        self.validity(length, nulls)?;
        self.at_least(product(length, width)?)?;
        Ok(true)
    }

    fn leaf(&mut self, data_type: &DataType, length: usize, nulls: usize) -> Result<bool> {
        match data_type {
            DataType::Null => Self::null(length, nulls),
            DataType::Boolean => self.boolean(length, nulls),
            DataType::FixedSizeBinary(width) => self.fixed_binary(*width, length, nulls),
            DataType::Dictionary(key, _) => self.dictionary_keys(key, length, nulls),
            _ => return self.variable(data_type, length, nulls),
        }?;
        Ok(true)
    }

    fn null(length: usize, nulls: usize) -> Result<()> {
        if nulls != length {
            return Err(invalid("V2 IPC Null node has a non-null value"));
        }
        Ok(())
    }

    fn boolean(&mut self, length: usize, nulls: usize) -> Result<()> {
        self.validity(length, nulls)?;
        self.at_least(bits(length)?)
    }

    fn fixed_binary(&mut self, width: i32, length: usize, nulls: usize) -> Result<()> {
        self.validity(length, nulls)?;
        self.at_least(product(
            length,
            usize::try_from(width)
                .map_err(|_| invalid("V2 IPC fixed-size binary width is negative"))?,
        )?)
    }

    fn dictionary_keys(&mut self, key: &DataType, length: usize, nulls: usize) -> Result<()> {
        self.validity(length, nulls)?;
        let width = key
            .primitive_width()
            .ok_or_else(|| invalid("V2 IPC dictionary key width is invalid"))?;
        self.at_least(product(length, width)?)
    }

    fn variable(&mut self, data_type: &DataType, length: usize, nulls: usize) -> Result<bool> {
        match data_type {
            DataType::Utf8 | DataType::Binary => self.bytes(length, nulls, 4),
            DataType::LargeUtf8 | DataType::LargeBinary => self.bytes(length, nulls, 8),
            DataType::Utf8View | DataType::BinaryView => self.view(length, nulls),
            _ => return Ok(false),
        }?;
        Ok(true)
    }

    fn bytes(&mut self, length: usize, nulls: usize, width: usize) -> Result<()> {
        self.validity(length, nulls)?;
        self.at_least(offsets(length, width)?)?;
        self.next_buffer()?;
        Ok(())
    }

    fn view(&mut self, length: usize, nulls: usize) -> Result<()> {
        let count = self.next_variadic()?;
        self.validity(length, nulls)?;
        self.at_least(product(length, 16)?)?;
        for _ in 0..count {
            self.next_buffer()?;
        }
        Ok(())
    }

    fn nested(&mut self, data_type: &DataType, length: usize, nulls: usize) -> Result<()> {
        if self.list(data_type, length, nulls)? {
            return Ok(());
        }
        match data_type {
            DataType::Struct(fields) => self.structure(fields, length, nulls),
            DataType::Union(fields, mode) => self.union_fields(fields, *mode, length, nulls),
            DataType::RunEndEncoded(run_ends, values) => self.run_end_fields(run_ends, values),
            _ => Err(invalid("V2 IPC array type has no decoded layout")),
        }
    }

    fn structure(&mut self, fields: &Fields, length: usize, nulls: usize) -> Result<()> {
        self.validity(length, nulls)?;
        for field in fields {
            if self.field(field)? < length {
                return Err(invalid("V2 IPC struct child is shorter than its parent"));
            }
        }
        Ok(())
    }

    fn union_fields(
        &mut self,
        fields: &UnionFields,
        mode: UnionMode,
        length: usize,
        nulls: usize,
    ) -> Result<()> {
        self.union(length, nulls, mode)?;
        for (_, field) in fields.iter() {
            let child_length = self.field(field)?;
            if mode == UnionMode::Sparse && child_length < length {
                return Err(invalid("V2 IPC sparse union child is too short"));
            }
        }
        Ok(())
    }

    fn run_end_fields(&mut self, run_ends: &Field, values: &Field) -> Result<()> {
        self.field(run_ends)?;
        self.field(values)?;
        Ok(())
    }

    fn list(&mut self, data_type: &DataType, length: usize, nulls: usize) -> Result<bool> {
        let Some((field, width, view)) = list_layout(data_type) else {
            return Ok(false);
        };
        self.validity(length, nulls)?;
        if let Some(width) = width {
            self.list_buffers(length, width, view)?;
        }
        self.field(field)?;
        Ok(true)
    }

    fn list_buffers(&mut self, length: usize, width: usize, view: bool) -> Result<()> {
        self.at_least(list_first_buffer(length, width, view)?)?;
        if view {
            self.at_least(product(length, width)?)?;
        }
        Ok(())
    }

    fn union(&mut self, length: usize, nulls: usize, mode: UnionMode) -> Result<()> {
        if nulls != 0 {
            return Err(invalid("V2 IPC V5 union node cannot contain nulls"));
        }
        self.at_least(length)?;
        if mode == UnionMode::Dense {
            self.at_least(product(length, 4)?)?;
        }
        Ok(())
    }

    fn next_node(&mut self) -> Result<(usize, usize)> {
        (self.check)()?;
        let nodes = self.batch.nodes().expect("validated nodes");
        if self.node >= nodes.len() {
            return Err(invalid("V2 IPC field node is missing"));
        }
        let node = nodes.get(self.node);
        self.node += 1;
        let length = count(node.length())?;
        let nulls = count(node.null_count())?;
        if nulls > length {
            return Err(invalid("V2 IPC null count exceeds its node length"));
        }
        self.facts.nodes = self.node;
        Ok((length, nulls))
    }

    fn validity(&mut self, length: usize, nulls: usize) -> Result<()> {
        let buffer = self.next_buffer()?;
        if nulls > 0 && buffer.len() < bits(length)? {
            return Err(invalid("V2 IPC validity bitmap is shorter than its node"));
        }
        Ok(())
    }

    fn at_least(&mut self, length: usize) -> Result<()> {
        if self.next_buffer()?.len() < length {
            return Err(invalid("V2 IPC values or offsets buffer is too short"));
        }
        Ok(())
    }

    fn next_buffer(&mut self) -> Result<&'a [u8]> {
        (self.check)()?;
        let buffers = self.batch.buffers().expect("validated buffers");
        if self.buffer >= buffers.len() {
            return Err(invalid("V2 IPC buffer is missing"));
        }
        let buffer = buffers.get(self.buffer);
        self.buffer += 1;
        let offset = count(buffer.offset())?;
        let length = count(buffer.length())?;
        let bytes = buffer_slice(self.body, offset, length)?;
        self.facts.buffers = self.buffer;
        self.facts.buffer_bytes = self
            .facts
            .buffer_bytes
            .checked_add(length)
            .ok_or_else(|| invalid("V2 IPC buffer byte count overflow"))?;
        Ok(bytes)
    }

    fn next_variadic(&mut self) -> Result<usize> {
        let counts = self.batch.variadicBufferCounts();
        let value = counts
            .filter(|counts| self.variadic < counts.len())
            .map(|counts| counts.get(self.variadic))
            .ok_or_else(|| invalid("V2 IPC variadic buffer count is missing"))?;
        self.variadic += 1;
        self.facts.variadic_counts = self.variadic;
        let value = count(value)?;
        let remaining = self.batch.buffers().expect("validated buffers").len() - self.buffer;
        if value.checked_add(2).is_none_or(|count| count > remaining) {
            return Err(invalid(
                "V2 IPC variadic count exceeds the remaining buffers",
            ));
        }
        Ok(value)
    }

    fn finish(self) -> Result<Facts> {
        (self.check)()?;
        let nodes = self.batch.nodes().expect("validated nodes").len();
        let buffers = self.batch.buffers().expect("validated buffers").len();
        let variadic = self
            .batch
            .variadicBufferCounts()
            .map_or(0, |counts| counts.len());
        if self.node != nodes || self.buffer != buffers || self.variadic != variadic {
            return Err(invalid(
                "V2 IPC contains unused nodes, buffers or variadic counts",
            ));
        }
        Ok(self.facts)
    }
}

fn list_layout(data_type: &DataType) -> Option<(&Field, Option<usize>, bool)> {
    match data_type {
        DataType::List(field) | DataType::Map(field, _) => Some((field, Some(4), false)),
        DataType::LargeList(field) => Some((field, Some(8), false)),
        DataType::ListView(field) => Some((field, Some(4), true)),
        DataType::LargeListView(field) => Some((field, Some(8), true)),
        DataType::FixedSizeList(field, _) => Some((field, None, false)),
        _ => None,
    }
}

fn list_first_buffer(length: usize, width: usize, view: bool) -> Result<usize> {
    if view {
        product(length, width)
    } else {
        offsets(length, width)
    }
}

fn buffer_slice(body: &[u8], offset: usize, length: usize) -> Result<&[u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| invalid("V2 IPC buffer range overflow"))?;
    if offset % 8 != 0 {
        return Err(invalid(
            "V2 IPC buffer offset is not aligned to eight bytes",
        ));
    }
    body.get(offset..end)
        .ok_or_else(|| invalid("V2 IPC buffer extends outside its body"))
}

fn count(value: i64) -> Result<usize> {
    usize::try_from(value).map_err(|_| invalid("V2 IPC count or range is negative or too large"))
}

fn product(length: usize, width: usize) -> Result<usize> {
    length
        .checked_mul(width)
        .ok_or_else(|| invalid("V2 IPC values length overflow"))
}

fn offsets(length: usize, width: usize) -> Result<usize> {
    product(
        length
            .checked_add(1)
            .ok_or_else(|| invalid("V2 IPC offsets length overflow"))?,
        width,
    )
}

fn bits(length: usize) -> Result<usize> {
    length
        .checked_add(7)
        .map(|length| length / 8)
        .ok_or_else(|| invalid("V2 IPC bitmap length overflow"))
}
