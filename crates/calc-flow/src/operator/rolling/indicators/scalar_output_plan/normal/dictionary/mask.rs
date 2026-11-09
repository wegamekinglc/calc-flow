use super::super::super::{
    concatenated::Piece,
    math::{add, size},
};
use super::{Check, Result, invalid};
use datafusion::arrow::{
    array::{Array, AsArray, PrimitiveArray, downcast_integer_array},
    datatypes::{ArrowNativeType, ArrowPrimitiveType},
};

pub(super) struct Mask {
    words: Vec<u64>,
}
impl Mask {
    fn new(values: usize) -> Result<Self> {
        let words = usize::try_from(add(size(values)?, 63)? / 64)
            .map_err(|_| invalid("dictionary mask length"))?;
        std::alloc::Layout::array::<u64>(words).map_err(|_| invalid("dictionary mask layout"))?;
        Ok(Self {
            words: vec![0; words],
        })
    }
    fn insert(&mut self, index: usize) -> Result<()> {
        let word = self
            .words
            .get_mut(index / 64)
            .ok_or_else(|| invalid("dictionary reference"))?;
        *word |= 1 << (index % 64);
        Ok(())
    }
    pub(super) fn len(&self, check: Check<'_>) -> Result<u64> {
        self.words.iter().try_fold(0, |sum, word| {
            check()?;
            add(sum, u64::from(word.count_ones()))
        })
    }
    pub(super) fn visit(
        &self,
        emit: &mut dyn FnMut(usize) -> Result<()>,
        check: Check<'_>,
    ) -> Result<()> {
        for (index, word) in self.words.iter().enumerate() {
            check()?;
            bits(index, *word, emit, check)?;
        }
        Ok(())
    }
}
fn bits(
    index: usize,
    mut word: u64,
    emit: &mut dyn FnMut(usize) -> Result<()>,
    check: Check<'_>,
) -> Result<()> {
    while word != 0 {
        check()?;
        emit(index * 64 + word.trailing_zeros() as usize)?;
        check()?;
        word &= word - 1;
    }
    Ok(())
}
pub(super) fn select(piece: Piece<'_>, check: Check<'_>) -> Result<Mask> {
    let Some(array) = piece.array else {
        return Mask::new(0);
    };
    let dictionary = array
        .as_any_dictionary_opt()
        .ok_or_else(|| invalid("normal dictionary mask"))?;
    let mut mask = Mask::new(dictionary.values().len())?;
    let keys = dictionary.keys();
    downcast_integer_array!(keys => fill(keys, piece, &mut mask, check), _ => Err(invalid("normal dictionary key type")))?;
    Ok(mask)
}
fn fill<T: ArrowPrimitiveType>(
    keys: &PrimitiveArray<T>,
    piece: Piece<'_>,
    mask: &mut Mask,
    check: Check<'_>,
) -> Result<()> {
    for row in piece.start..piece.start + piece.len {
        check()?;
        if keys.is_valid(row) {
            mask.insert(keys.value(row).as_usize())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_unreferenced_mask_word_scan_checks_cancellation() {
        let mask = super::Mask::new(1024).unwrap();
        let result = mask.visit(&mut |_| panic!("zero reference words"), &mut || {
            Err(super::invalid("cancelled"))
        });
        assert!(result.is_err());
    }
    #[test]
    fn test_reference_count_scan_checks_cancellation_and_counts_each_bit() {
        let mut mask = super::Mask::new(128).unwrap();
        mask.insert(63).unwrap();
        mask.insert(127).unwrap();
        assert_eq!(mask.len(&mut || Ok(())).unwrap(), 2);
        assert!(mask.len(&mut || Err(super::invalid("cancelled"))).is_err());
    }
}
