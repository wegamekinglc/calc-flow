use super::super::{
    concatenated, invalid,
    math::{add, size},
};
use super::{At, Check, Result};
use datafusion::arrow::{array::OffsetSizeTrait, datatypes::ArrowNativeType};

pub(super) fn list<O: OffsetSizeTrait>(
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    let mut len = 0;
    for index in 0..count {
        check()?;
        len = add(len, length(at(index)?, concatenated::list_piece::<O>)?)?;
    }
    limit::<O>(len)
}
pub(super) fn map(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<()> {
    let mut len = 0;
    for index in 0..count {
        check()?;
        len = add(len, length(at(index)?, concatenated::map_piece)?)?;
    }
    limit::<i32>(len)
}
fn length<'a>(
    piece: Option<concatenated::Piece<'a>>,
    select: impl FnOnce(concatenated::Piece<'a>) -> Result<concatenated::Piece<'a>>,
) -> Result<u64> {
    match piece {
        Some(piece) => size(select(piece)?.len),
        None => Ok(0),
    }
}
fn limit<O: ArrowNativeType>(len: u64) -> Result<()> {
    let len = usize::try_from(len).map_err(|_| invalid("normal output length"))?;
    O::from_usize(len).ok_or_else(|| invalid("normal output offset"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_native_offset_and_array_length_boundaries_without_large_buffers() {
        let maximum = u64::try_from(usize::MAX)
            .unwrap()
            .min(u64::try_from(i64::MAX).unwrap());
        assert!(super::limit::<i64>(maximum).is_ok());
        assert!(super::limit::<i64>(maximum + 1).is_err());
        assert!(super::limit::<i32>(u64::try_from(i32::MAX).unwrap()).is_ok());
        assert!(super::limit::<i32>(u64::try_from(i32::MAX).unwrap() + 1).is_err());
    }
}
