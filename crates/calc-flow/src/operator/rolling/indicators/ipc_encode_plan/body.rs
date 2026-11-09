use super::{
    Check,
    geometry::{self, Range, RunSlice, Window},
    math::{add, aligned, growing, invalid, mul, size, sum, table, vector},
};
use crate::Result;
use datafusion::arrow::{array::ArrayData, buffer::Buffer, datatypes::DataType};

#[derive(Default)]
pub(super) struct Message {
    pub(super) nodes: u64,
    pub(super) buffers: u64,
    pub(super) variadic: u64,
    pub(super) body: u64,
    pub(super) normalized: u64,
}
impl Message {
    pub(super) fn header(&self, dictionary: bool) -> Result<u64> {
        sum([
            vector(self.nodes, 16),
            vector(self.buffers, 16),
            vector(self.variadic, 8),
            table(6),
            table(5),
            Ok(16),
            if dictionary { table(3) } else { Ok(0) },
        ])
    }
    pub(super) fn workspace(&self) -> Result<u64> {
        sum([
            Ok(self.normalized),
            growing(self.nodes, 16),
            growing(self.buffers, 16),
            growing(self.variadic, 8),
        ])
    }
    fn buffer(&mut self, len: u64) -> Result<()> {
        self.buffers = add(self.buffers, 1)?;
        self.body = sum([Ok(self.body), aligned(len)])?;
        Ok(())
    }
    fn allocated(&mut self, bytes: u64) -> Result<()> {
        let owner = size(size_of::<(Buffer, Vec<u8>, usize, usize)>())?;
        self.normalized = sum([Ok(self.normalized), Ok(bytes), Ok(owner)])?;
        Ok(())
    }
    fn allocate_aligned(&mut self, bytes: u64) -> Result<()> {
        self.allocated(aligned(bytes)?)
    }
    fn bitmap(&mut self, data: &ArrayData, window: Window) -> Result<()> {
        if let Some(bytes) = geometry::bitmap_copy(data, window)? {
            self.allocated(bytes)?;
        }
        self.buffer(geometry::bits(window.len)?)
    }
    pub(super) fn visit(
        &mut self,
        data: &ArrayData,
        window: Window,
        check: Check<'_>,
    ) -> Result<()> {
        check()?;
        i64::try_from(window.len).map_err(|_| invalid())?;
        self.nodes = add(self.nodes, 1)?;
        if !matches!(
            data.data_type(),
            DataType::Null | DataType::Union(_, _) | DataType::RunEndEncoded(_, _)
        ) {
            self.bitmap(data, window)?;
        }
        self.payload(data, window, check)
    }
    fn payload(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        match data.data_type() {
            DataType::Binary | DataType::Utf8 => self.bytes(data, window, 4),
            DataType::LargeBinary | DataType::LargeUtf8 => self.bytes(data, window, 8),
            DataType::List(_) | DataType::Map(_, _) => self.list(data, window, 4, check),
            DataType::LargeList(_) => self.list(data, window, 8, check),
            _ => self.composite(data, window, check),
        }
    }
    fn bytes(&mut self, data: &ArrayData, window: Window, width: usize) -> Result<()> {
        let range = self.offsets(data, window, width)?;
        if window.len == 0 {
            self.allocated(0)?;
        }
        self.buffer(size(range.len)?)
    }
    fn offsets(&mut self, data: &ArrayData, window: Window, width: usize) -> Result<Range> {
        let range = geometry::range(data, window, width)?;
        let bytes = offset_bytes(window, width)?;
        self.buffer(bytes)?;
        if window.len == 0 || range.start != 0 {
            self.allocate_aligned(bytes)?;
        }
        Ok(range)
    }
    fn list(
        &mut self,
        data: &ArrayData,
        window: Window,
        width: usize,
        check: Check<'_>,
    ) -> Result<()> {
        let range = self.offsets(data, window, width)?;
        let child = data.child_data().first().ok_or_else(invalid)?;
        self.visit(child, Window::slice(child, range.start, range.len)?, check)
    }
    fn composite(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        match data.data_type() {
            DataType::FixedSizeList(_, width) => self.fixed_list(data, window, *width, check),
            DataType::ListView(_) => self.list_view(data, window, 4, check),
            DataType::LargeListView(_) => self.list_view(data, window, 8, check),
            DataType::RunEndEncoded(_, _) => self.run(data, window, check),
            _ => self.other(data, window, check),
        }
    }
    fn fixed_list(
        &mut self,
        data: &ArrayData,
        window: Window,
        width: i32,
        check: Check<'_>,
    ) -> Result<()> {
        let width = usize::try_from(width).map_err(|_| invalid())?;
        let child = data.child_data().first().ok_or_else(invalid)?;
        let start = window.offset.checked_mul(width).ok_or_else(invalid)?;
        let len = window.len.checked_mul(width).ok_or_else(invalid)?;
        self.visit(child, Window::slice(child, start, len)?, check)
    }
    fn list_view(
        &mut self,
        data: &ArrayData,
        window: Window,
        width: u64,
        check: Check<'_>,
    ) -> Result<()> {
        let bytes = mul(size(window.len)?, width)?;
        self.buffer_pair(bytes, window.len == 0)?;
        let child = data.child_data().first().ok_or_else(invalid)?;
        self.visit(child, geometry::view_child(child, window)?, check)
    }
    fn buffer_pair(&mut self, bytes: u64, empty: bool) -> Result<()> {
        self.buffer(bytes)?;
        self.buffer(bytes)?;
        if empty {
            self.allocated(0)?;
            self.allocated(0)?;
        }
        Ok(())
    }
    fn other(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        if matches!(data.data_type(), DataType::Utf8View | DataType::BinaryView) {
            return self.view(data, window, check);
        }
        if let Some(width) = geometry::fixed_width(data.data_type()) {
            return self.fixed(window.len, width);
        }
        if matches!(data.data_type(), DataType::Boolean) {
            return self.boolean(window);
        }
        self.children(data, window, check)
    }
    fn view(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        self.variadic = add(self.variadic, 1)?;
        self.fixed(window.len, 16)?;
        self.raw_buffers(&data.buffers()[1..], check)
    }
    fn fixed(&mut self, len: usize, width: usize) -> Result<()> {
        self.buffer(mul(size(len)?, size(width)?)?)
    }
    fn boolean(&mut self, window: Window) -> Result<()> {
        if !window.offset.is_multiple_of(8) {
            self.allocated(geometry::bit_copy(window.len)?)?;
        }
        self.buffer(geometry::bits(window.len)?)
    }
    fn raw_buffers(&mut self, buffers: &[Buffer], check: Check<'_>) -> Result<()> {
        for buffer in buffers {
            check()?;
            self.buffer(size(buffer.len())?)?;
        }
        Ok(())
    }
    fn children(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        self.raw_buffers(data.buffers(), check)?;
        for child in data.child_data() {
            self.visit(child, geometry::child_window(data, child, window)?, check)?;
        }
        Ok(())
    }
    fn run(&mut self, data: &ArrayData, window: Window, check: Check<'_>) -> Result<()> {
        match geometry::run_slice(data, window, check)? {
            Some(slice) => self.sliced_run(data, slice, check),
            None => self.children(data, window, check),
        }
    }
    fn sliced_run(&mut self, data: &ArrayData, slice: RunSlice, check: Check<'_>) -> Result<()> {
        let bytes = mul(size(slice.count)?, size(slice.width)?)?;
        self.run_ends(slice.count, bytes)?;
        let values = data.child_data().get(1).ok_or_else(invalid)?;
        self.visit(
            values,
            Window::slice(values, slice.first, slice.count)?,
            check,
        )
    }
    fn run_ends(&mut self, len: usize, bytes: u64) -> Result<()> {
        self.nodes = add(self.nodes, 1)?;
        self.generated_bitmap(len)?;
        self.allocate_aligned(bytes)?;
        self.buffer(bytes)
    }
    fn generated_bitmap(&mut self, len: usize) -> Result<()> {
        let bytes = geometry::bits(len)?;
        self.allocate_aligned(bytes)?;
        self.buffer(bytes)
    }
}
fn offset_bytes(window: Window, width: usize) -> Result<u64> {
    mul(add(size(window.len)?, 1)?, size(width)?)
}
