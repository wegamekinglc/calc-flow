use super::{Check, Emit, Result, invalid};
use datafusion::arrow::{
    array::ArrayData,
    datatypes::{ArrowNativeType, DataType, UnionMode},
};
pub(super) fn list<T: ArrowNativeType>(
    array: &ArrayData,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    let offsets = array.buffer::<T>(0);
    let first = offsets[start].as_usize();
    let last = offsets[start + len].as_usize();
    emit(
        first,
        last.checked_sub(first)
            .ok_or_else(|| invalid("view guard list range"))?,
        check,
    )
}
pub(super) fn view<T: ArrowNativeType>(
    array: &ArrayData,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    let offsets = array.buffer::<T>(0);
    let sizes = array.buffer::<T>(1);
    for index in start..start + len {
        check()?;
        if sizes[index].as_usize() != 0 {
            emit(offsets[index].as_usize(), sizes[index].as_usize(), check)?;
        }
    }
    Ok(())
}
pub(super) fn fixed(
    array: &ArrayData,
    width: i32,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    let width = usize::try_from(width).map_err(|_| invalid("view guard fixed width"))?;
    let first = start
        .checked_add(array.offset())
        .and_then(|n| n.checked_mul(width))
        .ok_or_else(|| invalid("view guard fixed start"))?;
    let len = len
        .checked_mul(width)
        .ok_or_else(|| invalid("view guard fixed length"))?;
    emit(first, len, check)
}
pub(super) fn union(
    array: &ArrayData,
    id: i8,
    mode: UnionMode,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    if mode == UnionMode::Sparse {
        return emit(start, len, check);
    }
    let ids = array.buffer::<i8>(0);
    let offsets = array.buffer::<i32>(1);
    for index in start..start + len {
        check()?;
        if ids[index] == id {
            let offset =
                usize::try_from(offsets[index]).map_err(|_| invalid("view guard union offset"))?;
            emit(offset, 1, check)?;
        }
    }
    Ok(())
}
pub(super) fn run(
    array: &ArrayData,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    if len == 0 {
        return Ok(());
    }
    let ends = &array.child_data()[0];
    match ends.data_type() {
        DataType::Int16 => physical::<i16>(array, start, len, emit, check),
        DataType::Int32 => physical::<i32>(array, start, len, emit, check),
        DataType::Int64 => physical::<i64>(array, start, len, emit, check),
        _ => Err(invalid("view guard run end type")),
    }
}
fn physical<T: ArrowNativeType>(
    array: &ArrayData,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    let start = start
        .checked_add(array.offset())
        .ok_or_else(|| invalid("view guard run start"))?;
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid("view guard run end"))?;
    let ends = array.child_data()[0].buffer::<T>(0);
    let first = ends.partition_point(|value| value.as_usize() <= start);
    let last = ends.partition_point(|value| value.as_usize() < end);
    if first > last || last >= array.child_data()[1].len() {
        return Err(invalid("view guard run range"));
    }
    emit(first, last - first + 1, check)
}
