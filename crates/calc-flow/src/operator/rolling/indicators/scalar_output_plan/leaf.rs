use super::{
    Check, OutputPlan, Result, invalid,
    math::{
        add, align, arc, arc_slice, bitmap, buffer_headers, bytes, grow, mul, retained,
        scalar_view_length, scale, size, sum,
    },
};
use datafusion::{
    arrow::{
        array::{
            BooleanArray, FixedSizeBinaryArray, Int64Array, LargeStringArray, NullArray,
            StringViewArray,
        },
        buffer::Buffer,
        datatypes::DataType as D,
    },
    common::ScalarValue,
};

pub(super) fn plan<'a>(
    dtype: &D,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<OutputPlan> {
    if !supported(dtype) {
        return Err(invalid("scalar datatype"));
    }
    if rows == 0 {
        return empty(dtype, check);
    }
    let stats = scan(dtype, rows, cell, check)?;
    finish(dtype, size(rows)?, &stats)
}
fn scan<'a>(
    dtype: &D,
    rows: usize,
    cell: &mut dyn FnMut(usize) -> Option<&'a ScalarValue>,
    check: Check<'_>,
) -> Result<Stats> {
    let mut stats = Stats::default();
    for row in 0..rows {
        check()?;
        stats.observe(dtype, cell(row))?;
    }
    Ok(stats)
}

fn supported(dtype: &D) -> bool {
    dtype.is_primitive()
        || matches!(
            dtype,
            D::Null
                | D::Boolean
                | D::Utf8
                | D::LargeUtf8
                | D::Binary
                | D::LargeBinary
                | D::Utf8View
                | D::BinaryView
                | D::FixedSizeBinary(_)
        )
}
#[derive(Default)]
struct Stats {
    copied: u64,
    nulls: bool,
    view: View,
}
impl Stats {
    fn observe(&mut self, dtype: &D, value: Option<&ScalarValue>) -> Result<()> {
        let Some(value) = value else {
            self.nulls = true;
            return Ok(());
        };
        validate(dtype, value)?;
        self.nulls |= value.is_null();
        let len = byte_length(value)?;
        self.copied = add(self.copied, len)?;
        if matches!(dtype, D::Utf8View | D::BinaryView) {
            self.view.append(len)?;
        }
        Ok(())
    }
}
fn validate(dtype: &D, value: &ScalarValue) -> Result<()> {
    if matches!(
        value,
        ScalarValue::List(_)
            | ScalarValue::LargeList(_)
            | ScalarValue::ListView(_)
            | ScalarValue::LargeListView(_)
            | ScalarValue::FixedSizeList(_)
            | ScalarValue::Map(_)
            | ScalarValue::Struct(_)
            | ScalarValue::Dictionary(..)
            | ScalarValue::RunEndEncoded(..)
            | ScalarValue::Union(..)
    ) {
        return Err(invalid("scalar datatype mismatch"));
    }
    if &value.data_type() != dtype {
        return Err(invalid("scalar datatype mismatch"));
    }
    Ok(())
}
pub(super) fn byte_length(value: &ScalarValue) -> Result<u64> {
    match value {
        ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) | ScalarValue::Utf8View(value) => {
            size(value.as_ref().map_or(0, String::len))
        }
        ScalarValue::Binary(value)
        | ScalarValue::LargeBinary(value)
        | ScalarValue::BinaryView(value)
        | ScalarValue::FixedSizeBinary(_, value) => size(value.as_ref().map_or(0, Vec::len)),
        _ => Ok(0),
    }
}
#[derive(Default)]
pub(super) struct View {
    used: u64,
    capacity: u64,
    completed: u64,
    blocks: u64,
    next: u64,
}
impl View {
    pub(super) fn append(&mut self, len: u64) -> Result<()> {
        if len <= 12 {
            return Ok(());
        }
        scalar_view_length(len)?;
        if add(self.used, len)? > self.capacity {
            self.allocate(len)?;
        }
        self.used = add(self.used, len)?;
        scalar_view_length(self.used)
    }
    fn allocate(&mut self, len: u64) -> Result<()> {
        let blocks = add(self.blocks, 1)?;
        u32::try_from(blocks).map_err(|_| invalid("fresh view block count"))?;
        self.completed = add(self.completed, self.capacity)?;
        self.next = mul(self.next.max(8192), 2)?.min(2 * 1024 * 1024);
        self.capacity = len.max(self.next);
        self.blocks = blocks;
        self.used = 0;
        Ok(())
    }
}
fn finish(dtype: &D, n: u64, stats: &Stats) -> Result<OutputPlan> {
    let data = data(dtype, n, stats)?;
    let mask = if stats.nulls && !matches!(dtype, D::Null) {
        scale(bitmap(n), 2)?
    } else {
        0
    };
    let output = sum([owner(dtype), Ok(data), Ok(mask)])?;
    let scratch = sum([
        bytes::<ScalarValue>(n),
        Ok(stats.copied),
        Ok(data),
        Ok(mask),
        buffer_headers(3),
        view_workspace(dtype, &stats.view),
    ])?;
    Ok(OutputPlan {
        output_bytes: output,
        scratch_bytes: scratch,
    })
}
fn data(dtype: &D, n: u64, stats: &Stats) -> Result<u64> {
    match dtype {
        D::Null => Ok(0),
        D::Boolean => scale(bitmap(n), 2),
        D::Utf8 | D::Binary => binary_data(n, stats.copied, 4),
        D::LargeUtf8 | D::LargeBinary => binary_data(n, stats.copied, 8),
        D::Utf8View | D::BinaryView => view_data(n, &stats.view),
        _ => numeric_data(dtype, n),
    }
}
fn numeric_data(dtype: &D, n: u64) -> Result<u64> {
    if let D::FixedSizeBinary(width) = dtype {
        return scale(
            align(mul(
                n,
                u64::try_from(*width).map_err(|_| invalid("binary width"))?,
            )?),
            2,
        );
    }
    retained(
        n,
        size(
            dtype
                .primitive_width()
                .ok_or_else(|| invalid("primitive width"))?,
        )?,
    )
}
fn binary_data(n: u64, len: u64, width: u64) -> Result<u64> {
    let limit = if width == 4 {
        u64::try_from(i32::MAX).expect("positive offset limit")
    } else {
        u64::try_from(i64::MAX).expect("positive offset limit")
    };
    if len > limit {
        return Err(invalid("binary offset overflow"));
    }
    sum([retained(add(n, 1)?, width), mul(len.max(1024), 2)])
}
fn view_data(n: u64, view: &View) -> Result<u64> {
    let blocks = usize::try_from(view.blocks).map_err(|_| invalid("view buffers"))?;
    sum([
        retained(n, 16),
        Ok(view.completed),
        Ok(view.capacity),
        arc_slice::<Buffer>(blocks),
    ])
}
fn view_workspace(dtype: &D, view: &View) -> Result<u64> {
    if !matches!(dtype, D::Utf8View | D::BinaryView) {
        return Ok(512);
    }
    sum([
        grow(view.blocks, size(size_of::<Buffer>())?),
        buffer_headers(view.blocks),
        Ok(512),
    ])
}
fn empty(dtype: &D, check: Check<'_>) -> Result<OutputPlan> {
    let payload = match dtype {
        D::Utf8 | D::Binary | D::LargeUtf8 | D::LargeBinary => Ok(64),
        D::Utf8View | D::BinaryView => arc_slice::<Buffer>(0),
        _ => Ok(0),
    };
    Ok(OutputPlan {
        output_bytes: sum([owner(dtype), payload])?,
        scratch_bytes: super::super::scalar_materialization_plan::null_array_requirement(
            dtype, 0, check,
        )?,
    })
}

fn owner(dtype: &D) -> Result<u64> {
    match dtype {
        D::Null => arc::<NullArray>(),
        D::Boolean => arc::<BooleanArray>(),
        D::Utf8 | D::Binary | D::LargeUtf8 | D::LargeBinary => arc::<LargeStringArray>(),
        D::Utf8View | D::BinaryView => arc::<StringViewArray>(),
        D::FixedSizeBinary(_) => arc::<FixedSizeBinaryArray>(),
        _ => arc::<Int64Array>(),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_fresh_view_slab_count_matches_the_inclusive_pinned_limit() {
        let maximum = u64::from(u32::MAX);
        let mut last = super::View {
            blocks: maximum - 1,
            used: 16_384,
            capacity: 16_384,
            ..super::View::default()
        };
        last.append(13).unwrap();
        assert_eq!(last.blocks, maximum);
        last.append(13).unwrap();
        assert_eq!(last.blocks, maximum);
        let mut overflow = super::View {
            blocks: maximum,
            used: 16_384,
            capacity: 16_384,
            ..super::View::default()
        };
        assert!(overflow.append(13).is_err());
    }
    #[test]
    fn test_fresh_view_block_limit_tracks_consumed_block_not_total_column_bytes() {
        let mut view = super::View::default();
        let maximum = u64::from(u32::MAX);
        view.append(maximum - 1).unwrap();
        assert_eq!(view.used, maximum - 1);
        view.append(13).unwrap();
        assert_eq!(view.used, 13);
        assert_eq!(view.blocks, 2);
        assert!(view.append(maximum).is_err());
    }
}
