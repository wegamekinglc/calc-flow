use super::super::super::math::{
    add, bitmap, buffer_headers, bytes, mul, scale, size, sum, vector,
};
use super::{At, Check, Cost, Result, invalid, merge, source};
use datafusion::arrow::{array::Array, datatypes::DataType};

#[derive(Default)]
pub(super) struct Facts {
    pub(super) values: u64,
    pub(super) rows: u64,
    pub(super) references: u64,
    pub(super) mask: u64,
    nullable: bool,
    metadata: u64,
}
pub(super) fn facts(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<Facts> {
    let mut facts = Facts::default();
    for index in 0..count {
        check()?;
        if let Some(piece) = at(index)? {
            facts.source(piece, check)?;
        }
    }
    Ok(facts)
}
impl Facts {
    fn source(
        &mut self,
        piece: super::super::super::concatenated::Piece<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        self.rows = add(self.rows, size(piece.len)?)?;
        if let Some(values) = source::values(piece)? {
            self.append(values, piece.len, check)?;
        }
        Ok(())
    }
    fn append(&mut self, array: &dyn Array, keys: usize, check: Check<'_>) -> Result<()> {
        let len = size(array.len())?;
        self.values = add(self.values, len)?;
        self.references = add(self.references, size(keys.min(array.len()))?)?;
        self.mask = self.mask.max(bitmap(len)?);
        self.nullable |= array.null_count() != 0;
        self.metadata = self
            .metadata
            .max(super::super::super::workspace::array_data_workspace(
                array, check,
            )?);
        Ok(())
    }
}
pub(super) fn plan(
    key: &DataType,
    value: &DataType,
    count: usize,
    rows: u64,
    merged: &merge::Merged,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let facts = facts(count, at, check)?;
    let width = key_width(key)?;
    let owned = owned(value, rows, width, merged, facts.nullable)?;
    Ok(Cost {
        owned,
        temporary: scratch(count, width, merged, &facts, owned)?,
    })
}
fn key_width(key: &DataType) -> Result<u64> {
    size(
        key.primitive_width()
            .ok_or_else(|| invalid("normal dictionary key width"))?,
    )
}
fn owned(
    value: &DataType,
    rows: u64,
    width: u64,
    merged: &merge::Merged,
    nullable: bool,
) -> Result<u64> {
    let data = add(layout(rows, width)?, value_bytes(value, merged)?)?;
    let mask = if nullable { bitmap(merged.count)? } else { 0 };
    add(data, mask)
}
fn value_bytes(dtype: &DataType, merged: &merge::Merged) -> Result<u64> {
    if let Some(width) = dtype.primitive_width() {
        return layout(merged.count, size(width)?);
    }
    byte_bytes(dtype, merged)
}
fn byte_bytes(dtype: &DataType, merged: &merge::Merged) -> Result<u64> {
    let width = byte_width(dtype)?;
    if merged.bytes > byte_limit(width) {
        return Err(invalid("normal merged byte offset"));
    }
    add(
        layout(merged.bytes, 1)?,
        layout(add(merged.count, 1)?, width)?,
    )
}
fn byte_width(dtype: &DataType) -> Result<u64> {
    match dtype {
        DataType::Utf8 | DataType::Binary => Ok(4),
        DataType::LargeUtf8 | DataType::LargeBinary => Ok(8),
        _ => Err(invalid("normal merged bytes")),
    }
}
fn byte_limit(width: u64) -> u64 {
    if width == 4 {
        u64::try_from(i32::MAX).expect("positive limit")
    } else {
        u64::try_from(i64::MAX).expect("positive limit")
    }
}
fn layout(count: u64, width: u64) -> Result<u64> {
    let bytes = mul(count, width)?;
    let n = usize::try_from(bytes).map_err(|_| invalid("normal merged allocation size"))?;
    let align =
        usize::try_from(width).map_err(|_| invalid("normal merged allocation alignment"))?;
    std::alloc::Layout::from_size_align(n, align)
        .map_err(|_| invalid("normal merged allocation layout"))?;
    Ok(bytes)
}
fn scratch(
    count: usize,
    width: u64,
    merged: &merge::Merged,
    facts: &Facts,
    owned: u64,
) -> Result<u64> {
    let vectors = vectors(count, merged.references)?;
    sum([
        Ok(owned),
        Ok(vectors),
        mul(facts.values, width),
        Ok(facts.mask),
        bitmap(merged.count),
        buffer_headers(5),
        mul(facts.metadata, 2),
        simulation(facts.references, facts.mask),
    ])
}
fn vectors(count: usize, references: u64) -> Result<u64> {
    sum([
        vector::<&dyn Array>(count),
        scale(vector::<Vec<u64>>(count), 2),
        bytes::<(usize, Option<&[u8]>)>(references),
        bytes::<(usize, usize)>(references),
        bytes::<merge::Bucket<'_>>(merge::buckets(references)?),
    ])
}
pub(super) fn simulation(references: u64, mask: u64) -> Result<u64> {
    add(
        bytes::<merge::Bucket<'_>>(merge::buckets(references)?)?,
        mask,
    )
}
