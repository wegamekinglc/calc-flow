use datafusion::arrow::{
    datatypes::{DataType, Schema, UnionMode},
    ipc,
};

use super::{Check, Span, add, aligned, invalid, multiply, size};
use crate::Result;

#[derive(Clone, Copy, Default)]
pub(super) struct Counts {
    pub(super) nodes: usize,
    pub(super) buffers: usize,
    pub(super) variadic: usize,
}
impl Counts {
    pub(super) fn merge(&mut self, other: Self) -> Result<()> {
        self.nodes = add(self.nodes, other.nodes)?;
        self.buffers = add(self.buffers, other.buffers)?;
        self.variadic = add(self.variadic, other.variadic)?;
        Ok(())
    }
}
#[derive(Clone, Copy)]
struct IndexArrays {
    kinds: ArrayInfo,
    entities: ArrayInfo,
    #[cfg(test)]
    positions: ArrayInfo,
}

pub(super) struct Indexed {
    pub(super) counts: Counts,
    #[cfg(test)]
    pub(super) entities: usize,
    #[cfg(test)]
    pub(super) histories: usize,
    #[cfg(test)]
    pub(super) pending: usize,
    #[cfg(test)]
    pub(super) summaries: usize,
    #[cfg(test)]
    pub(super) kinds: Span,
    #[cfg(test)]
    pub(super) entity_ids: Span,
    #[cfg(test)]
    pub(super) positions: Span,
    #[cfg(test)]
    pub(super) position_validity: Span,
}

#[derive(Clone, Copy, Default)]
struct ArrayInfo {
    length: usize,
    nulls: usize,
    validity: Span,
    values: Span,
}
impl ArrayInfo {
    fn valid(self, source: &[u8], index: usize) -> Result<bool> {
        if self.nulls == 0 {
            return Ok(true);
        }
        let bitmap = self.validity.bytes(source)?;
        let byte = bitmap
            .get(index / 8)
            .ok_or_else(|| invalid("validity bounds"))?;
        Ok(byte & (1 << (index % 8)) != 0)
    }
}

struct Cursor<'a, 'c> {
    source: &'a [u8],
    body: Span,
    record: ipc::RecordBatch<'a>,
    node: usize,
    buffer: usize,
    variadic: usize,
    buffer_end: usize,
    steps: usize,
    check: Check<'c>,
}
impl<'a, 'c> Cursor<'a, 'c> {
    fn new(
        source: &'a [u8],
        record: ipc::RecordBatch<'a>,
        body: Span,
        check: Check<'c>,
    ) -> Result<Self> {
        check()?;
        if record.compression().is_some() {
            return Err(invalid("compression"));
        }
        record.nodes().ok_or_else(|| invalid("field nodes"))?;
        record.buffers().ok_or_else(|| invalid("field buffers"))?;
        body.bytes(source)?;
        Ok(Self {
            source,
            body,
            record,
            node: 0,
            buffer: 0,
            variadic: 0,
            buffer_end: 0,
            steps: 0,
            check,
        })
    }
    fn tick(&mut self) -> Result<()> {
        self.steps = add(self.steps, 1)?;
        if self.steps.is_multiple_of(4096) {
            (self.check)()?;
        }
        Ok(())
    }
    fn finish(self) -> Result<Counts> {
        (self.check)()?;
        if self.node != self.record.nodes().expect("checked nodes").len()
            || self.buffer != self.record.buffers().expect("checked buffers").len()
            || self.variadic
                != self
                    .record
                    .variadicBufferCounts()
                    .map_or(0, |values| values.len())
            || aligned(self.buffer_end)? != self.body.length
        {
            return Err(invalid(
                "unused nodes, buffers, variadic counts or body bytes",
            ));
        }
        Ok(Counts {
            nodes: self.node,
            buffers: self.buffer,
            variadic: self.variadic,
        })
    }
    fn next_node(&mut self, expected: Option<usize>) -> Result<ArrayInfo> {
        self.tick()?;
        let nodes = self.record.nodes().expect("checked nodes");
        if self.node >= nodes.len() {
            return Err(invalid("missing field node"));
        }
        let node = nodes.get(self.node);
        self.node = add(self.node, 1)?;
        node_info(node, expected)
    }
    fn next_buffer(&mut self, exact: Option<usize>) -> Result<Span> {
        let (offset, length) = self.buffer_descriptor()?;
        self.buffer_extent(offset, length, exact)
    }
    fn buffer_descriptor(&mut self) -> Result<(usize, usize)> {
        self.tick()?;
        let buffers = self.record.buffers().expect("checked buffers");
        if self.buffer >= buffers.len() {
            return Err(invalid("missing buffer"));
        }
        let buffer = buffers.get(self.buffer);
        self.buffer = add(self.buffer, 1)?;
        let offset = size(buffer.offset(), "buffer offset")?;
        let length = size(buffer.length(), "buffer length")?;
        Ok((offset, length))
    }
    fn buffer_extent(
        &mut self,
        offset: usize,
        length: usize,
        exact: Option<usize>,
    ) -> Result<Span> {
        if offset != aligned(self.buffer_end)? || exact.is_some_and(|expected| expected != length) {
            return Err(invalid("buffer length, overlap or alignment"));
        }
        let end = add(offset, length)?;
        if end > self.body.length {
            return Err(invalid("buffer bounds"));
        }
        self.buffer_end = end;
        Ok(Span {
            offset: add(self.body.offset, offset)?,
            length,
        })
    }
    fn validity(&mut self, info: &mut ArrayInfo) -> Result<()> {
        let length = add(info.length, 7)? / 8;
        info.validity = self.next_buffer(Some(length))?;
        let valid = self.validity_bits(*info)?;
        if info.length - valid != info.nulls {
            return Err(invalid("validity null count"));
        }
        Ok(())
    }
    fn validity_bits(&mut self, info: ArrayInfo) -> Result<usize> {
        let bitmap = info.validity.bytes(self.source)?;
        let mut valid = 0;
        for (index, byte) in bitmap.iter().copied().enumerate() {
            self.tick()?;
            valid = add(valid, valid_bits(byte, info.length - index * 8))?;
        }
        Ok(valid)
    }
    fn value(&mut self, data_type: &DataType, expected: Option<usize>) -> Result<ArrayInfo> {
        let mut info = self.next_node(expected)?;
        if !matches!(
            data_type,
            DataType::Null | DataType::Union(_, _) | DataType::RunEndEncoded(_, _)
        ) {
            self.validity(&mut info)?;
        }
        self.storage(data_type, &mut info)?;
        Ok(info)
    }
    fn storage(&mut self, data_type: &DataType, info: &mut ArrayInfo) -> Result<()> {
        if let Some(width) = fixed_width(data_type)? {
            return self.fixed_values(info, width);
        }
        self.variable(data_type, info)
    }
    fn fixed_values(&mut self, info: &mut ArrayInfo, width: usize) -> Result<()> {
        info.values = self.next_buffer(Some(multiply(info.length, width)?))?;
        Ok(())
    }
    fn variable(&mut self, data_type: &DataType, info: &mut ArrayInfo) -> Result<()> {
        match data_type {
            DataType::Null => null_values(info),
            DataType::Boolean => self.boolean_values(info),
            DataType::Binary | DataType::Utf8 => self.bytes(info, 4),
            DataType::LargeBinary | DataType::LargeUtf8 => self.bytes(info, 8),
            DataType::BinaryView | DataType::Utf8View => self.byte_views(info),
            _ => self.nested(data_type, info),
        }
    }
    fn boolean_values(&mut self, info: &mut ArrayInfo) -> Result<()> {
        info.values = self.next_buffer(Some(add(info.length, 7)? / 8))?;
        Ok(())
    }
    fn bytes(&mut self, info: &mut ArrayInfo, width: usize) -> Result<()> {
        let offsets = self.next_buffer(Some(multiply(add(info.length, 1)?, width)?))?;
        info.values = self.next_buffer(None)?;
        let last = self.offsets(offsets, width, info.length)?;
        if last != info.values.length {
            return Err(invalid("byte offsets or values length"));
        }
        Ok(())
    }
    fn offsets(&mut self, offsets: Span, width: usize, length: usize) -> Result<usize> {
        let bytes = offsets.bytes(self.source)?;
        let mut previous = 0;
        for index in 0..=length {
            self.tick()?;
            let value = offset(bytes, index, width)?;
            offset_order(index, value, previous)?;
            previous = value;
        }
        Ok(previous)
    }
    fn nested(&mut self, data_type: &DataType, info: &mut ArrayInfo) -> Result<()> {
        match data_type {
            DataType::List(field) | DataType::Map(field, _) => {
                self.list(info, field.data_type(), 4)
            }
            DataType::LargeList(field) => self.list(info, field.data_type(), 8),
            DataType::ListView(field) => self.list_view(info, field.data_type(), 4),
            DataType::LargeListView(field) => self.list_view(info, field.data_type(), 8),
            _ => self.children(data_type, info),
        }
    }
    fn children(&mut self, data_type: &DataType, info: &mut ArrayInfo) -> Result<()> {
        match data_type {
            DataType::FixedSizeList(field, size) => self.fixed_list(info, field.data_type(), *size),
            DataType::Struct(fields) => self.struct_values(fields, info.length),
            _ => self.encoded(data_type, info),
        }
    }
    fn fixed_list(&mut self, info: &ArrayInfo, child: &DataType, size: i32) -> Result<()> {
        let size = usize::try_from(size).map_err(|_| invalid("fixed list size"))?;
        self.value(child, Some(multiply(info.length, size)?))?;
        Ok(())
    }
    fn struct_values(
        &mut self,
        fields: &datafusion::arrow::datatypes::Fields,
        length: usize,
    ) -> Result<()> {
        for field in fields {
            self.value(field.data_type(), Some(length))?;
        }
        Ok(())
    }
    fn list(&mut self, info: &mut ArrayInfo, child: &DataType, width: usize) -> Result<()> {
        let offsets = self.next_buffer(Some(multiply(add(info.length, 1)?, width)?))?;
        let length = self.offsets(offsets, width, info.length)?;
        self.value(child, Some(length))?;
        Ok(())
    }
    fn list_view(&mut self, info: &mut ArrayInfo, child: &DataType, width: usize) -> Result<()> {
        let (offsets, sizes) = self.view_spans(info.length, width)?;
        let child = self.value(child, None)?;
        self.list_view_rows(*info, offsets, sizes, child.length, width)
    }
    fn view_spans(&mut self, length: usize, width: usize) -> Result<(Span, Span)> {
        let offsets = self.next_buffer(Some(multiply(length, width)?))?;
        let sizes = self.next_buffer(Some(multiply(length, width)?))?;
        Ok((offsets, sizes))
    }
    fn list_view_rows(
        &mut self,
        info: ArrayInfo,
        offsets: Span,
        sizes: Span,
        length: usize,
        width: usize,
    ) -> Result<()> {
        for index in 0..info.length {
            self.tick()?;
            if info.valid(self.source, index)? {
                self.list_view_end(index, offsets, sizes, length, width)?;
            }
        }
        Ok(())
    }
    fn list_view_end(
        &self,
        index: usize,
        offsets: Span,
        sizes: Span,
        length: usize,
        width: usize,
    ) -> Result<()> {
        let end = add(
            offset(offsets.bytes(self.source)?, index, width)?,
            offset(sizes.bytes(self.source)?, index, width)?,
        )?;
        if end > length {
            return Err(invalid("list view bounds"));
        }
        Ok(())
    }
    fn byte_views(&mut self, info: &mut ArrayInfo) -> Result<()> {
        let count = self.variadic_count()?;
        info.values = self.next_buffer(Some(multiply(info.length, 16)?))?;
        let first = self.buffer;
        if count > self.record.buffers().expect("checked buffers").len() - first {
            return Err(invalid("variadic buffer count"));
        }
        for _ in 0..count {
            self.next_buffer(None)?;
        }
        self.validate_views(*info, first, count)
    }
    fn variadic_count(&mut self) -> Result<usize> {
        let counts = self
            .record
            .variadicBufferCounts()
            .ok_or_else(|| invalid("missing variadic count"))?;
        if self.variadic >= counts.len() {
            return Err(invalid("missing variadic count"));
        }
        let count = size(counts.get(self.variadic), "variadic count")?;
        self.variadic = add(self.variadic, 1)?;
        Ok(count)
    }
    fn validate_views(&mut self, info: ArrayInfo, first: usize, count: usize) -> Result<()> {
        let views = info.values.bytes(self.source)?;
        for index in 0..info.length {
            self.tick()?;
            if !info.valid(self.source, index)? {
                continue;
            }
            self.byte_view(views, index, first, count)?;
        }
        Ok(())
    }
    fn byte_view(&self, views: &[u8], index: usize, first: usize, count: usize) -> Result<()> {
        let view = &views[multiply(index, 16)?..multiply(add(index, 1)?, 16)?];
        let length = word(view, 0)?;
        if length <= 12 {
            return Ok(());
        }
        self.external_view(view, first, count, length)
    }
    fn external_view(&self, view: &[u8], first: usize, count: usize, length: usize) -> Result<()> {
        let buffer = word(view, 8)?;
        let end = add(word(view, 12)?, length)?;
        if buffer >= count {
            return Err(invalid("byte view buffer ID"));
        }
        let length = self.view_buffer_length(first, buffer)?;
        if end > length {
            return Err(invalid("byte view bounds"));
        }
        Ok(())
    }
    fn view_buffer_length(&self, first: usize, buffer: usize) -> Result<usize> {
        size(
            self.record
                .buffers()
                .expect("checked buffers")
                .get(add(first, buffer)?)
                .length(),
            "byte view buffer length",
        )
    }
    fn encoded(&mut self, data_type: &DataType, info: &mut ArrayInfo) -> Result<()> {
        match data_type {
            DataType::Dictionary(index, _) => self.dictionary_values(info, index),
            DataType::Union(fields, mode) => self.union(info, fields, *mode),
            DataType::RunEndEncoded(ends, values) => {
                self.run_end(info, ends.data_type(), values.data_type())
            }
            _ => Err(invalid("array type")),
        }
    }
    fn dictionary_values(&mut self, info: &mut ArrayInfo, index: &DataType) -> Result<()> {
        let width = fixed_width(index)?.ok_or_else(|| invalid("dictionary index width"))?;
        info.values = self.next_buffer(Some(multiply(info.length, width)?))?;
        Ok(())
    }
    fn union(
        &mut self,
        info: &mut ArrayInfo,
        fields: &datafusion::arrow::datatypes::UnionFields,
        mode: UnionMode,
    ) -> Result<()> {
        let offsets = self.union_buffers(info, mode)?;
        let ids = info.values.bytes(self.source)?;
        self.union_ids(ids, fields)?;
        self.union_children(fields, mode, info.length, ids, offsets)
    }
    fn union_buffers(&mut self, info: &mut ArrayInfo, mode: UnionMode) -> Result<Option<Span>> {
        if info.nulls != 0 {
            return Err(invalid("union null count"));
        }
        info.values = self.next_buffer(Some(info.length))?;
        if mode == UnionMode::Dense {
            Ok(Some(self.next_buffer(Some(multiply(info.length, 4)?))?))
        } else {
            Ok(None)
        }
    }
    fn union_ids(
        &mut self,
        ids: &[u8],
        fields: &datafusion::arrow::datatypes::UnionFields,
    ) -> Result<()> {
        for id in ids {
            self.tick()?;
            if !fields
                .iter()
                .any(|(expected, _)| u8::try_from(expected).ok() == Some(*id))
            {
                return Err(invalid("union type ID"));
            }
        }
        Ok(())
    }
    fn union_children(
        &mut self,
        fields: &datafusion::arrow::datatypes::UnionFields,
        mode: UnionMode,
        length: usize,
        ids: &[u8],
        offsets: Option<Span>,
    ) -> Result<()> {
        for (id, field) in fields.iter() {
            let expected = (mode == UnionMode::Sparse).then_some(length);
            let child = self.value(field.data_type(), expected)?;
            if let Some(offsets) = offsets {
                self.union_offsets(ids, id, offsets, child.length)?;
            }
        }
        Ok(())
    }
    fn union_offsets(&mut self, ids: &[u8], id: i8, offsets: Span, length: usize) -> Result<()> {
        let offsets = offsets.bytes(self.source)?;
        for (index, actual) in ids.iter().copied().enumerate() {
            self.tick()?;
            if u8::try_from(id).ok() == Some(actual) && offset(offsets, index, 4)? >= length {
                return Err(invalid("union child offset"));
            }
        }
        Ok(())
    }
    fn run_end(
        &mut self,
        info: &mut ArrayInfo,
        end_type: &DataType,
        value_type: &DataType,
    ) -> Result<()> {
        let (ends, values) = self.run_children(info, end_type, value_type)?;
        let width = fixed_width(end_type)?.ok_or_else(|| invalid("run-end width"))?;
        let bytes = ends.values.bytes(self.source)?;
        let previous = self.run_order(bytes, width, values.length)?;
        if previous < info.length {
            return Err(invalid("run-end logical length"));
        }
        Ok(())
    }
    fn run_children(
        &mut self,
        info: &ArrayInfo,
        end_type: &DataType,
        value_type: &DataType,
    ) -> Result<(ArrayInfo, ArrayInfo)> {
        if info.nulls != 0 {
            return Err(invalid("run-end null count"));
        }
        let ends = self.value(end_type, None)?;
        let values = self.value(value_type, Some(ends.length))?;
        if ends.nulls != 0 {
            return Err(invalid("run-end null values"));
        }
        Ok((ends, values))
    }
    fn run_order(&mut self, bytes: &[u8], width: usize, length: usize) -> Result<usize> {
        let mut previous = 0;
        for index in 0..length {
            self.tick()?;
            let value = offset(bytes, index, width)?;
            if value <= previous {
                return Err(invalid("run-end ordering"));
            }
            previous = value;
        }
        Ok(previous)
    }
}

pub(super) fn dictionary(
    source: &[u8],
    record: ipc::RecordBatch<'_>,
    body: Span,
    data_type: &DataType,
    check: Check<'_>,
) -> Result<Counts> {
    let mut cursor = Cursor::new(source, record, body, check)?;
    cursor.value(data_type, Some(size(record.length(), "dictionary rows")?))?;
    cursor.finish()
}

pub(super) fn record(
    source: &[u8],
    record: ipc::RecordBatch<'_>,
    body: Span,
    schema: &Schema,
    check: Check<'_>,
) -> Result<Indexed> {
    let rows = size(record.length(), "record rows")?;
    index_schema(schema)?;
    let mut cursor = Cursor::new(source, record, body, check)?;
    let arrays = index_arrays(&mut cursor, schema, rows)?;
    additional_columns(&mut cursor, schema, rows)?;
    let counts = index_rows(&mut cursor, arrays.kinds, arrays.entities)?;
    indexed(
        cursor,
        counts,
        #[cfg(test)]
        arrays,
    )
}
fn index_schema(schema: &Schema) -> Result<()> {
    let fields = schema.fields();
    if fields.len() < 3
        || fields[0].data_type() != &DataType::UInt8
        || fields[1].data_type() != &DataType::UInt64
        || fields[2].data_type() != &DataType::UInt64
    {
        return Err(invalid("index schema"));
    }
    Ok(())
}
fn index_arrays(cursor: &mut Cursor<'_, '_>, schema: &Schema, rows: usize) -> Result<IndexArrays> {
    let fields = schema.fields();
    let kinds = cursor.value(fields[0].data_type(), Some(rows))?;
    let entities = cursor.value(fields[1].data_type(), Some(rows))?;
    #[cfg(test)]
    let positions = cursor.value(fields[2].data_type(), Some(rows))?;
    #[cfg(not(test))]
    cursor.value(fields[2].data_type(), Some(rows))?;
    if kinds.nulls != 0 || entities.nulls != 0 {
        return Err(invalid("null kind or entity ID"));
    }
    Ok(IndexArrays {
        kinds,
        entities,
        #[cfg(test)]
        positions,
    })
}
fn additional_columns(cursor: &mut Cursor<'_, '_>, schema: &Schema, rows: usize) -> Result<()> {
    for field in schema.fields().iter().skip(3) {
        cursor.value(field.data_type(), Some(rows))?;
    }
    Ok(())
}
fn indexed(
    cursor: Cursor<'_, '_>,
    counts: [usize; 6],
    #[cfg(test)] arrays: IndexArrays,
) -> Result<Indexed> {
    #[cfg(not(test))]
    summary_count(&counts)?;
    Ok(Indexed {
        #[cfg(test)]
        entities: counts[0],
        #[cfg(test)]
        histories: counts[1],
        #[cfg(test)]
        pending: counts[2],
        #[cfg(test)]
        summaries: summary_count(&counts)?,
        #[cfg(test)]
        kinds: arrays.kinds.values,
        #[cfg(test)]
        entity_ids: arrays.entities.values,
        #[cfg(test)]
        positions: arrays.positions.values,
        #[cfg(test)]
        position_validity: arrays.positions.validity,
        counts: cursor.finish()?,
    })
}
fn summary_count(counts: &[usize; 6]) -> Result<usize> {
    add(add(counts[3], counts[4])?, counts[5])
}
fn index_rows(cursor: &mut Cursor<'_, '_>, kinds: ArrayInfo, ids: ArrayInfo) -> Result<[usize; 6]> {
    let tags = kinds.values.bytes(cursor.source)?;
    let entities = ids.values.bytes(cursor.source)?;
    scan_index(cursor, tags, entities)
}
fn scan_index(cursor: &mut Cursor<'_, '_>, tags: &[u8], entities: &[u8]) -> Result<[usize; 6]> {
    let mut counts = [0; 6];
    let mut previous = 0;
    for (index, kind) in tags.iter().copied().enumerate() {
        cursor.tick()?;
        kind_order(kind, previous)?;
        let id = offset(entities, index, 8)?;
        entity_index(kind, id, &mut counts)?;
        previous = kind;
    }
    if counts[0] == 0 {
        return Err(invalid("missing entity dictionary"));
    }
    Ok(counts)
}
fn kind_order(kind: u8, previous: u8) -> Result<()> {
    if kind > 5 || kind < previous {
        return Err(invalid("state kind order"));
    }
    Ok(())
}
fn entity_index(kind: u8, id: usize, counts: &mut [usize; 6]) -> Result<()> {
    if kind == 0 {
        if id != counts[0] {
            return Err(invalid("entity dictionary ID"));
        }
    } else if id >= counts[0] {
        return Err(invalid("unknown entity ID"));
    }
    counts[usize::from(kind)] = add(counts[usize::from(kind)], 1)?;
    Ok(())
}
fn node_info(node: &ipc::FieldNode, expected: Option<usize>) -> Result<ArrayInfo> {
    let length = size(node.length(), "field node length")?;
    let nulls = size(node.null_count(), "field node null count")?;
    if nulls > length || expected.is_some_and(|expected| expected != length) {
        return Err(invalid("field node rows or null count"));
    }
    Ok(ArrayInfo {
        length,
        nulls,
        ..Default::default()
    })
}
fn valid_bits(byte: u8, remaining: usize) -> usize {
    let bits = if remaining < 8 {
        byte & ((1 << remaining) - 1)
    } else {
        byte
    };
    usize::try_from(bits.count_ones()).expect("byte popcount fits usize")
}
fn null_values(info: &ArrayInfo) -> Result<()> {
    if info.nulls != info.length {
        return Err(invalid("null array null count"));
    }
    Ok(())
}
fn offset_order(index: usize, value: usize, previous: usize) -> Result<()> {
    if (index == 0 && value != 0) || value < previous {
        return Err(invalid("noncanonical offsets"));
    }
    Ok(())
}

fn fixed_width(data_type: &DataType) -> Result<Option<usize>> {
    match data_type {
        DataType::FixedSizeBinary(width) => usize::try_from(*width)
            .map(Some)
            .map_err(|_| invalid("fixed binary width")),
        _ => Ok(data_type.primitive_width()),
    }
}

fn offset(bytes: &[u8], index: usize, width: usize) -> Result<usize> {
    let start = multiply(index, width)?;
    let value = bytes
        .get(start..add(start, width)?)
        .ok_or_else(|| invalid("offset bounds"))?;
    let signed = match width {
        2 => i64::from(i16::from_le_bytes(
            value.try_into().expect("two checked bytes"),
        )),
        4 => i64::from(i32::from_le_bytes(
            value.try_into().expect("four checked bytes"),
        )),
        8 => i64::from_le_bytes(value.try_into().expect("eight checked bytes")),
        _ => return Err(invalid("offset width")),
    };
    size(signed, "negative or overflowing offset")
}
fn word(bytes: &[u8], start: usize) -> Result<usize> {
    let value = bytes
        .get(start..add(start, 4)?)
        .ok_or_else(|| invalid("view bounds"))?;
    usize::try_from(u32::from_le_bytes(
        value.try_into().expect("four checked view bytes"),
    ))
    .map_err(|_| invalid("view size"))
}

#[cfg(test)]
#[path = "layout_tests.rs"]
mod tests;
