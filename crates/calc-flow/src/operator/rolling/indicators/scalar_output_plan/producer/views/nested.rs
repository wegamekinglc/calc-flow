use super::{At, Check, Copies, Emit, Result, ranges, walk as recurse};
use datafusion::arrow::datatypes::DataType;
pub(super) fn walk(
    count: usize,
    at: &At<'_, '_>,
    copies: &Copies<'_>,
    check: Check<'_>,
) -> Result<()> {
    match at(0).data_type() {
        DataType::List(_) | DataType::Map(..) => child(count, at, copies, 0, Kind::List32, check),
        DataType::LargeList(_) => child(count, at, copies, 0, Kind::List64, check),
        DataType::ListView(_) => child(count, at, copies, 0, Kind::View32, check),
        DataType::LargeListView(_) => child(count, at, copies, 0, Kind::View64, check),
        _ => other(count, at, copies, check),
    }
}
fn other(count: usize, at: &At<'_, '_>, copies: &Copies<'_>, check: Check<'_>) -> Result<()> {
    match at(0).data_type() {
        DataType::FixedSizeList(_, width) => {
            child(count, at, copies, 0, Kind::Fixed(*width), check)
        }
        DataType::RunEndEncoded(_, _) => child(count, at, copies, 1, Kind::Run, check),
        _ => Ok(()),
    }
}
#[derive(Clone, Copy)]
enum Kind {
    List32,
    List64,
    View32,
    View64,
    Fixed(i32),
    Run,
}
fn child(
    count: usize,
    at: &At<'_, '_>,
    copies: &Copies<'_>,
    field: usize,
    kind: Kind,
    check: Check<'_>,
) -> Result<()> {
    let child = |index| &at(index).child_data()[field];
    let selected = |index, emit: &mut Emit<'_>, check: Check<'_>| {
        copies(
            index,
            &mut |start, len, check| select(at(index), kind, start, len, emit, check),
            check,
        )
    };
    recurse(count, &child, &selected, check)
}
fn select(
    array: &datafusion::arrow::array::ArrayData,
    kind: Kind,
    start: usize,
    len: usize,
    emit: &mut Emit<'_>,
    check: Check<'_>,
) -> Result<()> {
    match kind {
        Kind::List32 => ranges::list::<i32>(array, start, len, emit, check),
        Kind::List64 => ranges::list::<i64>(array, start, len, emit, check),
        Kind::View32 => ranges::view::<i32>(array, start, len, emit, check),
        Kind::View64 => ranges::view::<i64>(array, start, len, emit, check),
        Kind::Fixed(width) => ranges::fixed(array, width, start, len, emit, check),
        Kind::Run => ranges::run(array, start, len, emit, check),
    }
}
