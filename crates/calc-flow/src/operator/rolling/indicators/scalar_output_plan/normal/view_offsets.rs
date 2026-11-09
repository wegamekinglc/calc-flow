use super::super::{
    concatenated, invalid,
    math::{add, size},
};
use super::{At, Check, Result};
use datafusion::arrow::{
    array::{GenericListViewArray, OffsetSizeTrait},
    datatypes::ArrowNativeType,
};

pub(in crate::operator::rolling::indicators::scalar_output_plan) fn shift<O: ArrowNativeType>(
    global: u64,
    local: u64,
) -> Result<O> {
    let value = usize::try_from(add(global, local)?).map_err(|_| invalid("normal view offset"))?;
    O::from_usize(value).ok_or_else(|| invalid("normal view offset"))
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn native<O: ArrowNativeType>(
    global: u64,
    local: O,
) -> Result<O> {
    shift::<O>(global, size(local.as_usize())?)
}
pub(in crate::operator::rolling::indicators::scalar_output_plan) fn advance<O: ArrowNativeType>(
    global: u64,
    len: u64,
    consumed: bool,
) -> Result<u64> {
    if !consumed {
        return Ok(global);
    }
    let value = add(global, len)?;
    shift::<O>(value, 0)?;
    Ok(value)
}
pub(super) fn validate<O: OffsetSizeTrait>(
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    let last = last(count, at, check)?;
    let mut global = 0;
    for index in 0..count {
        check()?;
        if let Some(piece) = at(index)? {
            global = next::<O>(piece, global, last.is_some_and(|last| index < last), check)?;
        }
    }
    Ok(())
}
fn next<O: OffsetSizeTrait>(
    piece: concatenated::Piece<'_>,
    global: u64,
    consumed: bool,
    check: Check<'_>,
) -> Result<u64> {
    offsets::<O>(piece, global, check)?;
    let len = concatenated::view_values::<O>(piece)?.len;
    advance::<O>(global, size(len)?, consumed)
}
fn last(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<Option<usize>> {
    let mut last = None;
    for index in 0..count {
        check()?;
        if at(index)?.is_some_and(|piece| piece.len != 0) {
            last = Some(index);
        }
    }
    Ok(last)
}
fn offsets<O: OffsetSizeTrait>(
    piece: concatenated::Piece<'_>,
    global: u64,
    check: Check<'_>,
) -> Result<()> {
    let Some(array) = piece.array else {
        shift::<O>(global, 0)?;
        return Ok(());
    };
    let array = typed::<O>(array)?;
    for offset in &array.value_offsets()[piece.start..piece.start + piece.len] {
        check()?;
        native::<O>(global, *offset)?;
    }
    Ok(())
}

fn typed<O: OffsetSizeTrait>(
    array: &dyn datafusion::arrow::array::Array,
) -> Result<&GenericListViewArray<O>> {
    array
        .as_any()
        .downcast_ref()
        .ok_or_else(|| invalid("normal view representation"))
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_normal_view_offsets_skip_only_unused_global_advancement() {
        let maximum = u64::try_from(i32::MAX).unwrap();
        assert_eq!(super::shift::<i32>(maximum, 0).unwrap(), i32::MAX);
        assert!(super::shift::<i32>(maximum, 1).is_err());
        assert_eq!(super::advance::<i32>(maximum, 1, false).unwrap(), maximum);
        assert!(super::advance::<i32>(maximum, 1, true).is_err());
        assert_eq!(super::advance::<i32>(0, maximum + 1, false).unwrap(), 0);
        assert!(super::advance::<i32>(0, maximum + 1, true).is_err());
        assert_eq!(super::advance::<i32>(7, u64::MAX, false).unwrap(), 7);
    }
}
