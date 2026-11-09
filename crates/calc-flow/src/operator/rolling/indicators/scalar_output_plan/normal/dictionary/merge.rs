use super::super::super::math::add;
use super::{At, Check, Result, invalid, mask, source};
use datafusion::arrow::datatypes::DataType;

pub(super) type Bucket<'a> = Option<(Option<&'a [u8]>, u64)>;
#[derive(Default)]
pub(super) struct Merged {
    pub(super) references: u64,
    pub(super) count: u64,
    pub(super) bytes: u64,
}
pub(super) fn buckets(references: u64) -> Result<u64> {
    add(references, 129)?
        .checked_next_power_of_two()
        .ok_or_else(|| invalid("dictionary interner capacity"))
}
pub(super) fn run(
    key: &DataType,
    count: usize,
    at: &mut At<'_>,
    check: Check<'_>,
) -> Result<Merged> {
    let references = references(count, at, check)?;
    let mut table = Table::new(key, references)?;
    for index in 0..count {
        check()?;
        table.source(at(index)?, check)?;
    }
    let merged = std::mem::take(&mut table.merged);
    drop(table);
    Ok(merged)
}
fn references(count: usize, at: &mut At<'_>, check: Check<'_>) -> Result<u64> {
    let mut total = 0;
    for index in 0..count {
        check()?;
        total = add(total, count_references(at(index)?, check)?)?;
    }
    Ok(total)
}
fn count_references(
    piece: Option<super::super::super::concatenated::Piece<'_>>,
    check: Check<'_>,
) -> Result<u64> {
    match piece {
        Some(piece) => mask::select(piece, check)?.len(check),
        None => Ok(0),
    }
}
struct Table<'a> {
    entries: Vec<Bucket<'a>>,
    shift: u32,
    state: ahash::RandomState,
    limit: u64,
    merged: Merged,
}
impl<'a> Table<'a> {
    fn new(key: &DataType, references: u64) -> Result<Self> {
        let n = usize::try_from(buckets(references)?)
            .map_err(|_| invalid("dictionary interner length"))?;
        std::alloc::Layout::array::<Bucket<'_>>(n)
            .map_err(|_| invalid("dictionary interner layout"))?;
        Ok(Self {
            entries: (0..n).map(|_| None).collect(),
            shift: u64::BITS - n.trailing_zeros(),
            state: ahash::RandomState::with_seeds(0, 0, 0, 0),
            limit: super::super::super::producer::key_limit(key)?,
            merged: Merged {
                references,
                ..Merged::default()
            },
        })
    }
    fn source(
        &mut self,
        piece: Option<super::super::super::concatenated::Piece<'a>>,
        check: Check<'_>,
    ) -> Result<()> {
        if let Some(piece) = piece {
            self.append(piece, check)?;
        }
        Ok(())
    }
    fn append(
        &mut self,
        piece: super::super::super::concatenated::Piece<'a>,
        check: Check<'_>,
    ) -> Result<()> {
        let Some(values) = source::values(piece)? else {
            return Ok(());
        };
        let mask = mask::select(piece, check)?;
        mask.visit(&mut |index| self.intern(values, index), check)
    }
    fn intern(
        &mut self,
        values: &'a dyn datafusion::arrow::array::Array,
        index: usize,
    ) -> Result<()> {
        let value = source::raw(values, index)?;
        let bucket = usize::try_from(self.state.hash_one(value) >> self.shift)
            .map_err(|_| invalid("dictionary hash bucket"))?;
        if self.entries[bucket]
            .as_ref()
            .is_some_and(|(existing, _)| *existing == value)
        {
            return Ok(());
        }
        let next = self.emit(values, index)?;
        self.entries[bucket] = Some((value, next));
        Ok(())
    }
    fn emit(&mut self, values: &dyn datafusion::arrow::array::Array, index: usize) -> Result<u64> {
        if self.merged.count > self.limit {
            return Err(invalid("normal merged dictionary key"));
        }
        let next = self.merged.count;
        self.merged.count = add(next, 1)?;
        self.merged.bytes = add(self.merged.bytes, source::raw_length(values, index)?)?;
        Ok(next)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_interner_capacity_uses_pinned_strict_power_and_checked_overflow() {
        assert_eq!(super::buckets(0).unwrap(), 256);
        assert_eq!(super::buckets(127).unwrap(), 256);
        assert_eq!(super::buckets(128).unwrap(), 512);
        assert!(super::buckets(u64::MAX).is_err());
    }
    #[test]
    fn test_interner_cancellation_releases_prepaid_metadata_before_return() {
        use datafusion::arrow::{
            array::{DictionaryArray, Float64Array, Int8Array},
            datatypes::{DataType, Int8Type},
        };
        use std::sync::Arc;
        let arrays = [1.0, 2.0].map(|value| {
            DictionaryArray::<Int8Type>::try_new(
                Int8Array::from(vec![0]),
                Arc::new(Float64Array::from(vec![value])),
            )
            .unwrap()
        });
        let mut checks = 0;
        let allocation = allocation_counter::measure(|| {
            let mut at = |index: usize| {
                Ok(Some(super::super::super::super::concatenated::Piece {
                    array: Some(&arrays[index] as &dyn datafusion::arrow::array::Array),
                    start: 0,
                    len: 1,
                }))
            };
            let result = super::run(&DataType::Int8, 2, &mut at, &mut || {
                checks += 1;
                if checks == 11 {
                    Err(super::invalid("cancelled"))
                } else {
                    Ok(())
                }
            });
            assert!(result.is_err());
        });
        assert_eq!(checks, 11);
        let minimum = u64::try_from(256 * size_of::<super::Bucket<'_>>()).unwrap();
        assert!(allocation.bytes_max >= minimum);
        assert_eq!(allocation.bytes_current, 0);
    }
}
