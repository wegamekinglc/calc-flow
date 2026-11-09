use super::super::super::{concatenated::Piece, invalid};
use super::super::{Cell, array};
use super::{Check, Result};
use datafusion::arrow::array::UnionArray;
use datafusion::arrow::array::{
    Array, FixedSizeListArray, GenericListArray, GenericListViewArray, MapArray, OffsetSizeTrait,
    StructArray,
};
use datafusion::arrow::array::{RunArray, downcast_run_array};
use datafusion::common::ScalarValue;

#[path = "sources/fresh.rs"]
mod fresh;
pub(super) use fresh::Fresh;

pub(super) type Emit<'a, 'b> = dyn FnMut(Piece<'a>, Check<'_>) -> Result<()> + 'b;
pub(super) trait Sources<'a> {
    fn count(&self) -> usize;
    fn scalar(&mut self, _index: usize) -> Result<Option<&'a ScalarValue>> {
        Ok(None)
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>>;
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()>;
}
pub(super) struct Root<'a, 'b> {
    pub(super) count: usize,
    pub(super) width: usize,
    pub(super) repair: bool,
    pub(super) cell: &'b mut Cell<'a, 'b>,
}
impl<'a> Sources<'a> for Root<'a, '_> {
    fn count(&self) -> usize {
        self.count
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>> {
        let Some(array) = array((self.cell)(index))? else {
            return Ok(Piece::null(self.width));
        };
        if self.repair && array.is_null(0) {
            return Ok(Piece::null(self.width));
        }
        Ok(Piece {
            array: Some(array.values().as_ref()),
            start: 0,
            len: self.width,
        })
    }
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
        check()?;
        emit(self.source(index)?, check)
    }
}
#[derive(Clone, Copy)]
pub(super) enum Kind {
    List32,
    List64,
    View32,
    View64,
    Fixed(usize),
    Struct(usize),
    Map,
    Run,
    Union { id: i8, sparse: bool, first: bool },
}
pub(super) struct Child<'a, 'b> {
    pub(super) parent: &'b mut dyn Sources<'a>,
    pub(super) kind: Kind,
}
impl<'a> Sources<'a> for Child<'a, '_> {
    fn count(&self) -> usize {
        self.parent.count()
    }
    fn scalar(&mut self, index: usize) -> Result<Option<&'a ScalarValue>> {
        let Some(value) = self.parent.scalar(index)? else {
            return Ok(None);
        };
        fresh::child_scalar(value, self.kind)
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>> {
        if let Some(value) = self.parent.scalar(index)? {
            return fresh::child_piece(value, self.kind);
        }
        let parent = self.parent.source(index)?;
        full(parent, self.kind)
    }
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
        if self.parent.scalar(index)?.is_some() {
            check()?;
            return emit(self.source(index)?, check);
        }
        let kind = self.kind;
        self.parent.visit(
            index,
            &mut |piece, check| selected(piece, kind, emit, check),
            check,
        )
    }
}
fn typed<T: Array + 'static>(array: &dyn Array) -> Result<&T> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("mutable child array"))
}
fn full(piece: Piece<'_>, kind: Kind) -> Result<Piece<'_>> {
    let Some(array) = piece.array else {
        return virtual_child(piece, kind);
    };
    let array = full_array(array, kind)?;
    Ok(Piece {
        array: Some(array),
        start: 0,
        len: array.len(),
    })
}
fn full_array(array: &dyn Array, kind: Kind) -> Result<&dyn Array> {
    match kind {
        Kind::List32 => Ok(typed::<GenericListArray<i32>>(array)?.values().as_ref()),
        Kind::List64 => Ok(typed::<GenericListArray<i64>>(array)?.values().as_ref()),
        Kind::View32 => Ok(typed::<GenericListViewArray<i32>>(array)?.values().as_ref()),
        _ => full_other(array, kind),
    }
}

fn full_other(array: &dyn Array, kind: Kind) -> Result<&dyn Array> {
    match kind {
        Kind::View64 => Ok(typed::<GenericListViewArray<i64>>(array)?.values().as_ref()),
        Kind::Fixed(_) => Ok(typed::<FixedSizeListArray>(array)?.values().as_ref()),
        _ => full_structural(array, kind),
    }
}
fn full_structural(array: &dyn Array, kind: Kind) -> Result<&dyn Array> {
    match kind {
        Kind::Struct(index) => Ok(typed::<StructArray>(array)?.column(index).as_ref()),
        Kind::Map => Ok(typed::<MapArray>(array)?.entries()),
        _ => full_encoded(array, kind),
    }
}
fn full_encoded(array: &dyn Array, kind: Kind) -> Result<&dyn Array> {
    match kind {
        Kind::Run => run_values(array),
        Kind::Union { id, .. } => Ok(typed::<UnionArray>(array)?.child(id).as_ref()),
        _ => Err(invalid("mutable child kind")),
    }
}
fn virtual_child(piece: Piece<'_>, kind: Kind) -> Result<Piece<'_>> {
    let len = match kind {
        Kind::Fixed(width) => piece
            .len
            .checked_mul(width)
            .ok_or_else(|| invalid("fixed child length"))?,
        Kind::Struct(_) => piece.len,
        Kind::Run => usize::from(piece.len != 0),
        Kind::Union { sparse, first, .. } => union_null_len(piece.len, sparse, first),
        _ => 0,
    };
    Ok(Piece::null(len))
}
fn union_null_len(len: usize, sparse: bool, first: bool) -> usize {
    if sparse || first { len } else { 0 }
}
fn selected<'a>(
    piece: Piece<'a>,
    kind: Kind,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    let Some(array) = piece.array else {
        return emit(virtual_child(piece, kind)?, check);
    };
    match kind {
        Kind::List32 => list::<i32>(piece, emit, check),
        Kind::List64 => list::<i64>(piece, emit, check),
        Kind::View32 => view::<i32>(piece, emit, check),
        Kind::View64 => view::<i64>(piece, emit, check),
        _ => selected_other(piece, array, kind, emit, check),
    }
}
fn selected_other<'a>(
    piece: Piece<'a>,
    array: &'a dyn Array,
    kind: Kind,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    match kind {
        Kind::Union { id, sparse, .. } => union_selected(piece, id, sparse, emit, check),
        Kind::Run => run_selected(piece, emit, check),
        _ => selected_structural(piece, array, kind, emit, check),
    }
}
fn selected_structural<'a>(
    piece: Piece<'a>,
    array: &'a dyn Array,
    kind: Kind,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    if matches!(kind, Kind::Map) {
        return selected_map(piece, array, emit, check);
    }
    let (start, len) = if let Kind::Fixed(width) = kind {
        (piece.start.checked_mul(width), piece.len.checked_mul(width))
    } else {
        (Some(piece.start), Some(piece.len))
    };
    let mut child = full(piece, kind)?;
    child.start = start.ok_or_else(|| invalid("child range"))?;
    child.len = len.ok_or_else(|| invalid("child range"))?;
    emit(child, check)
}
fn selected_map<'a>(
    piece: Piece<'a>,
    array: &'a dyn Array,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    let offsets = typed::<MapArray>(array)?.value_offsets();
    let start = usize::try_from(offsets[piece.start]).map_err(|_| invalid("map offset"))?;
    let end =
        usize::try_from(offsets[piece.start + piece.len]).map_err(|_| invalid("map offset"))?;
    let mut child = full(piece, Kind::Map)?;
    child.start = start;
    child.len = end.checked_sub(start).ok_or_else(|| invalid("map range"))?;
    emit(child, check)
}

fn list<'a, O: OffsetSizeTrait>(
    piece: Piece<'a>,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    let array = typed::<GenericListArray<O>>(piece.array.ok_or_else(|| invalid("list source"))?)?;
    let offsets = array.value_offsets();
    let start = offsets[piece.start].as_usize();
    let end = offsets[piece.start + piece.len].as_usize();
    emit(
        Piece {
            array: Some(array.values().as_ref()),
            start,
            len: end
                .checked_sub(start)
                .ok_or_else(|| invalid("list range"))?,
        },
        check,
    )
}
fn view<'a, O: OffsetSizeTrait>(
    piece: Piece<'a>,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    let array =
        typed::<GenericListViewArray<O>>(piece.array.ok_or_else(|| invalid("list view source"))?)?;
    for index in piece.start..piece.start + piece.len {
        check()?;
        let len = array.value_sizes()[index].as_usize();
        if len != 0 {
            emit(
                Piece {
                    array: Some(array.values().as_ref()),
                    start: array.value_offsets()[index].as_usize(),
                    len,
                },
                check,
            )?;
        }
    }
    Ok(())
}

fn run_values(array: &dyn Array) -> Result<&dyn Array> {
    downcast_run_array!(array=>Ok(array.values().as_ref()),_=>Err(invalid("mutable run array")))
}
fn run_selected<'a>(piece: Piece<'a>, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
    let array = piece.array.ok_or_else(|| invalid("mutable run source"))?;
    downcast_run_array!(array=>selected_run(array,piece.start,piece.len,emit,check),_=>Err(invalid("mutable run type")))
}
fn selected_run<'a, R: datafusion::arrow::datatypes::RunEndIndexType>(
    array: &'a RunArray<R>,
    start: usize,
    len: usize,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    let end = start
        .checked_add(len)
        .filter(|end| *end <= array.len())
        .ok_or_else(|| invalid("mutable run range"))?;
    let first = array.get_physical_index(start);
    let last = array.get_physical_index(end - 1);
    emit(
        Piece {
            array: Some(array.values().as_ref()),
            start: first,
            len: last - first + 1,
        },
        check,
    )
}

fn union_selected<'a>(
    piece: Piece<'a>,
    id: i8,
    sparse: bool,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    let array = typed::<UnionArray>(piece.array.ok_or_else(|| invalid("mutable union source"))?)?;
    let child = array.child(id).as_ref();
    if sparse {
        return emit(
            Piece {
                array: Some(child),
                start: piece.start,
                len: piece.len,
            },
            check,
        );
    }
    dense_selected(array, id, piece, emit, check)
}
fn dense_selected<'a>(
    array: &'a UnionArray,
    id: i8,
    piece: Piece<'a>,
    emit: &mut Emit<'a, '_>,
    check: Check<'_>,
) -> Result<()> {
    for index in piece.start..piece.start + piece.len {
        check()?;
        if array.type_id(index) == id {
            emit(
                Piece {
                    array: Some(array.child(id).as_ref()),
                    start: array.value_offset(index),
                    len: 1,
                },
                check,
            )?;
        }
    }
    Ok(())
}
