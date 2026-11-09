use super::{
    Check, Cost, Result, Sources,
    math::{add, bitmap, bytes, mul, scale, size, sum},
    node,
};
use super::{
    buffers::{Buffers, Growing},
    sources::{Child, Kind},
};
use datafusion::arrow::datatypes::{UnionFields, UnionMode};

pub(super) fn children(
    fields: &UnionFields,
    mode: UnionMode,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    let mut total = Cost::default();
    for (index, (id, field)) in fields.iter().enumerate() {
        let kind = Kind::Union {
            id,
            sparse: mode == UnionMode::Sparse,
            first: index == 0,
        };
        let mut child = Child {
            parent: sources,
            kind,
        };
        let cost = child_cost(field.data_type(), mode, capacity, &mut child, check)?;
        total = total.add(cost)?;
    }
    Ok(total)
}
fn child_cost(
    dtype: &datafusion::arrow::datatypes::DataType,
    mode: UnionMode,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Cost> {
    if mode == UnionMode::Dense {
        super::super::super::producer::dense_length(copied_rows(sources, check)?)?;
    }
    let mut cost = node(dtype, capacity, sources, check)?;
    cost.descriptors = add(
        cost.descriptors,
        bytes::<datafusion::arrow::array::MutableArrayData<'_>>(1)?,
    )?;
    Ok(cost)
}
fn copied_rows(sources: &mut dyn Sources<'_>, check: Check<'_>) -> Result<u64> {
    let mut rows = 0;
    for index in 0..sources.count() {
        check()?;
        sources.visit(
            index,
            &mut |piece, _| {
                rows = add(rows, size(piece.len)?)?;
                Ok(())
            },
            check,
        )?;
    }
    Ok(rows)
}
pub(super) fn buffers(
    mode: UnionMode,
    capacity: u64,
    sources: &mut dyn Sources<'_>,
    check: Check<'_>,
) -> Result<Buffers> {
    let mut cursor = Cursor::new(mode, capacity)?;
    for index in 0..sources.count() {
        check()?;
        sources.visit(
            index,
            &mut |piece, check| cursor.push(piece.len, check),
            check,
        )?;
    }
    cursor.finish()
}
struct Cursor {
    ids: Growing,
    offsets: Growing,
    mask: Growing,
    rows: u64,
    dense: bool,
}
impl Cursor {
    fn new(mode: UnionMode, capacity: u64) -> Result<Self> {
        let dense = mode == UnionMode::Dense;
        let offsets = if dense { mul(capacity, 4)? } else { 0 };
        Ok(Self {
            ids: Growing::new(capacity, 0)?,
            offsets: Growing::new(offsets, 0)?,
            mask: Growing::mask(capacity)?,
            rows: 0,
            dense,
        })
    }
    fn push(&mut self, len: usize, check: Check<'_>) -> Result<()> {
        self.rows = add(self.rows, size(len)?)?;
        self.ids.append(size(len)?, 0)?;
        let end = add(self.rows, 7)? / 8;
        self.mask.append(end - self.mask.len, 0)?;
        self.extend_offsets(len, check)
    }
    fn extend_offsets(&mut self, len: usize, check: Check<'_>) -> Result<()> {
        if self.dense {
            for _ in 0..len {
                check()?;
                self.offsets.append(4, 0)?;
            }
        }
        Ok(())
    }
    fn finish(self) -> Result<Buffers> {
        let output = add(self.ids.capacity, self.offsets.capacity)?;
        Ok(Buffers {
            output,
            scratch: add(output, self.workspace()?)?,
            count: 3,
        })
    }
    fn workspace(&self) -> Result<u64> {
        sum([
            Ok(self.ids.old),
            Ok(self.offsets.old),
            Ok(self.mask.capacity),
            Ok(self.mask.old),
            scale(bitmap(self.rows), 2),
        ])
    }
}
