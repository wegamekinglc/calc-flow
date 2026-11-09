use super::{Check, Emit, Kind, Piece, Result, ScalarValue, Sources, invalid};
use crate::operator::rolling::indicators::scalar_output_plan::single;

pub(in crate::operator::rolling::indicators::scalar_output_plan::fixed::mutable) struct Fresh<
    'a,
    'b,
> {
    pub(in crate::operator::rolling::indicators::scalar_output_plan::fixed::mutable) count: usize,
    pub(in crate::operator::rolling::indicators::scalar_output_plan::fixed::mutable) cell:
        &'b mut (dyn FnMut(usize) -> Option<&'a ScalarValue> + 'b),
}
impl<'a> Sources<'a> for Fresh<'a, '_> {
    fn count(&self) -> usize {
        self.count
    }
    fn scalar(&mut self, index: usize) -> Result<Option<&'a ScalarValue>> {
        Ok((self.cell)(index).filter(|value| single::array(value).is_none()))
    }
    fn source(&mut self, index: usize) -> Result<Piece<'a>> {
        piece((self.cell)(index), 1)
    }
    fn visit(&mut self, index: usize, emit: &mut Emit<'a, '_>, check: Check<'_>) -> Result<()> {
        check()?;
        emit(self.source(index)?, check)
    }
}
pub(super) fn child_scalar(value: &ScalarValue, kind: Kind) -> Result<Option<&ScalarValue>> {
    Ok(child(value, kind)?.filter(|value| single::array(value).is_none()))
}
pub(super) fn child_piece(value: &ScalarValue, kind: Kind) -> Result<Piece<'_>> {
    if let Some(value) = child(value, kind)? {
        return piece(Some(value), 1);
    }
    let len = match kind {
        Kind::Union { sparse, first, .. } => {
            usize::from(sparse || first && matches!(value, ScalarValue::Union(None, ..)))
        }
        _ => 0,
    };
    Ok(Piece::null(len))
}
fn piece(value: Option<&ScalarValue>, len: usize) -> Result<Piece<'_>> {
    let Some(array) = value.and_then(single::array) else {
        return Ok(Piece::null(len));
    };
    if array.len() != 1 {
        return Err(invalid("fresh child scalar length"));
    }
    Ok(Piece {
        array: Some(array),
        start: 0,
        len: array.len(),
    })
}
fn child(value: &ScalarValue, kind: Kind) -> Result<Option<&ScalarValue>> {
    match (value, kind) {
        (ScalarValue::Union(value, ..), Kind::Union { id, .. }) => Ok(value
            .as_ref()
            .filter(|(active, _)| *active == id)
            .map(|(_, value)| value.as_ref())),
        (ScalarValue::RunEndEncoded(_, _, value), Kind::Run) => Ok(Some(value.as_ref())),
        _ => Err(invalid("fresh child scalar kind")),
    }
}
