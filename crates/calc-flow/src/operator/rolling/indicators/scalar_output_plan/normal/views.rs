use super::super::{
    concatenated::Piece,
    invalid,
    math::{add, size},
};
use super::{At, Check, Result};
use datafusion::arrow::{
    array::{ByteView, GenericByteViewArray},
    datatypes::ByteViewType,
};

pub(super) fn validate<T: ByteViewType>(
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<()> {
    let mut preceding = 0;
    for index in 0..count {
        check()?;
        if let Some(piece) = at(index)? {
            preceding = source::<T>(piece, preceding, check)?;
        }
    }
    Ok(())
}
fn source<T: ByteViewType>(piece: Piece<'_>, preceding: u64, check: Check<'_>) -> Result<u64> {
    let Some(array) = piece.array else {
        return Ok(preceding);
    };
    let array = array
        .as_any()
        .downcast_ref::<GenericByteViewArray<T>>()
        .ok_or_else(|| invalid("normal byte view representation"))?;
    referenced(
        preceding,
        &array.views()[piece.start..piece.start + piece.len],
        check,
    )?;
    add(preceding, size(array.data_buffers().len())?)
}
fn referenced(preceding: u64, views: &[u128], check: Check<'_>) -> Result<()> {
    for value in views {
        check()?;
        let view = ByteView::from(*value);
        if view.length > 12 {
            super::super::producer::view_index(preceding, view.buffer_index)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_normal_copied_view_references_use_true_prefix_without_unused_cast_rejection() {
        use datafusion::arrow::array::ByteView;
        let long = ByteView {
            length: 16,
            prefix: 0,
            buffer_index: 0,
            offset: 0,
        }
        .as_u128();
        let inline = ByteView {
            length: 1,
            prefix: 0,
            buffer_index: 0,
            offset: 0,
        }
        .as_u128();
        let maximum = u64::from(u32::MAX);
        assert!(super::referenced(maximum, &[long], &mut || Ok(())).is_ok());
        assert!(super::referenced(maximum + 1, &[inline], &mut || Ok(())).is_ok());
        assert!(super::referenced(maximum + 1, &[long], &mut || Ok(())).is_err());
        let indexed = ByteView {
            length: 16,
            prefix: 0,
            buffer_index: 1,
            offset: 0,
        }
        .as_u128();
        assert!(super::referenced(maximum, &[indexed], &mut || Ok(())).is_err());
    }
}
