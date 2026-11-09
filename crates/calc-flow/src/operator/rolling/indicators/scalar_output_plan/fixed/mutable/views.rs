use super::super::super::producer::view_index as reference_index;
use super::buffers::{Buffers, Growing};
use super::{
    Check, Result, Sources, invalid,
    math::{add, arc_slice, bitmap, mul, scale, size, sum, vector},
};
use datafusion::arrow::{
    array::{ByteView, GenericByteViewArray},
    buffer::Buffer,
    datatypes::{BinaryViewType, ByteViewType, DataType, StringViewType},
};

pub(super) fn plan(
    dtype: &DataType,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Option<Buffers>> {
    match dtype {
        DataType::Utf8View => scan::<StringViewType>(capacity, sources, check).map(Some),
        DataType::BinaryView => scan::<BinaryViewType>(capacity, sources, check).map(Some),
        _ => Ok(None),
    }
}
struct Cursor {
    views: Growing,
    mask: Growing,
    rows: u64,
    blocks: u64,
    pinned_offset: u32,
}
impl Cursor {
    fn new(capacity: u64) -> Result<Self> {
        Ok(Self {
            views: Growing::new(mul(capacity, 16)?, 0)?,
            mask: Growing::mask(capacity)?,
            rows: 0,
            blocks: 0,
            pinned_offset: 0,
        })
    }
    fn input<T: ByteViewType>(
        &mut self,
        index: usize,
        sources: &mut dyn Sources<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        if let Some(value) = sources.scalar(index)? {
            return self.scalar(value);
        }
        self.array::<T>(index, sources, check)
    }
    fn array<T: ByteViewType>(
        &mut self,
        index: usize,
        sources: &mut dyn Sources<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        let blocks = block_count::<T>(sources.source(index)?)?;
        self.pinned_offset = super::super::views::advance(self.pinned_offset, blocks)?;
        sources.visit(
            index,
            &mut |piece, check| self.push::<T>(piece, check),
            check,
        )?;
        self.blocks = add(self.blocks, blocks)?;
        Ok(())
    }
    fn scalar(&mut self, value: &datafusion::common::ScalarValue) -> Result<()> {
        let len = super::super::super::leaf::byte_length(value)?;
        super::super::super::math::scalar_view_length(len)?;
        let blocks = u64::from(len > 12);
        self.pinned_offset = super::super::views::advance(self.pinned_offset, blocks)?;
        if blocks != 0 {
            reference_index(self.blocks, 0)?;
        }
        self.blocks = add(self.blocks, blocks)?;
        self.append(1)
    }
    fn push<T: ByteViewType>(
        &mut self,
        piece: super::super::super::concatenated::Piece<'_>,
        check: Check<'_>,
    ) -> Result<()> {
        referenced::<T>(piece, self.blocks, check)?;
        self.append(piece.len)
    }
    fn append(&mut self, len: usize) -> Result<()> {
        self.rows = add(self.rows, size(len)?)?;
        self.views.append(mul(size(len)?, 16)?, 0)?;
        let end = add(self.rows, 7)? / 8;
        self.mask.append(end - self.mask.len, 0)
    }
    fn finish(self) -> Result<Buffers> {
        let blocks = usize::try_from(self.blocks).map_err(|_| invalid("mutable view buffers"))?;
        let descriptors = arc_slice::<Buffer>(blocks)?;
        let output = sum([
            Ok(self.views.capacity),
            Ok(self.mask.capacity),
            Ok(descriptors),
        ])?;
        Ok(Buffers {
            output,
            scratch: add(output, self.workspace(blocks)?)?,
            count: 2,
        })
    }
    fn workspace(&self, blocks: usize) -> Result<u64> {
        let slots = blocks
            .checked_add(1)
            .ok_or_else(|| invalid("mutable view buffers"))?;
        sum([
            vector::<Buffer>(slots),
            Ok(self.views.old),
            Ok(self.mask.old),
            scale(bitmap(self.rows), 2),
        ])
    }
}
fn scan<T: ByteViewType>(
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Buffers> {
    let mut cursor = Cursor::new(capacity)?;
    for index in 0..sources.count() {
        check()?;
        cursor.input::<T>(index, sources, check)?;
    }
    cursor.finish()
}
fn typed<T: ByteViewType>(
    array: &dyn datafusion::arrow::array::Array,
) -> Result<&GenericByteViewArray<T>> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("mutable view array"))
}
fn block_count<T: ByteViewType>(
    piece: super::super::super::concatenated::Piece<'_>,
) -> Result<u64> {
    match piece.array {
        Some(array) => size(typed::<T>(array)?.data_buffers().len()),
        None => Ok(0),
    }
}
fn referenced<T: ByteViewType>(
    piece: super::super::super::concatenated::Piece<'_>,
    preceding: u64,
    check: Check<'_>,
) -> Result<()> {
    let Some(array) = piece.array else {
        return Ok(());
    };
    let views = typed::<T>(array)?.views();
    for value in &views[piece.start..piece.start + piece.len] {
        check()?;
        let view = ByteView::from(*value);
        if view.length > 12 {
            reference_index(preceding, view.buffer_index)?;
        }
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    #[test]
    fn test_copied_long_view_index_checks_true_preceding_buffer_boundaries() {
        assert_eq!(super::reference_index(0, u32::MAX).unwrap(), u32::MAX);
        assert_eq!(
            super::reference_index(u64::from(u32::MAX), 0).unwrap(),
            u32::MAX
        );
        assert!(super::reference_index(u64::from(u32::MAX), 1).is_err());
        assert!(super::reference_index(u64::from(u32::MAX) + 1, 0).is_err());
        assert!(super::reference_index(u64::MAX, 1).is_err());
    }
}
