use super::*;
use std::{cell::Cell, sync::Arc};

thread_local! {
    static CONVERTED: Cell<usize> = const { Cell::new(0) };
}

pub(super) fn note_converted(rows: usize) {
    CONVERTED.with(|converted| converted.set(converted.get() + rows));
}

pub(in crate::operator::asof) fn take_converted() -> usize {
    CONVERTED.with(|converted| converted.replace(0))
}

impl Positions {
    pub fn legacy_mut(&mut self) -> &mut Vec<(usize, usize)> {
        assert!(self.prefix.is_none(), "legacy test fixture positions");
        &mut self.pairs
    }

    pub fn capacity(&self) -> usize {
        self.prefix.as_ref().map_or(0, |prefix| {
            prefix.rows.inner().capacity() / size_of::<u64>()
        }) + self.pairs.capacity()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl From<Vec<(usize, usize)>> for Positions {
    fn from(pairs: Vec<(usize, usize)>) -> Self {
        Self {
            prefix: None,
            pairs,
        }
    }
}

impl FromIterator<(usize, usize)> for Positions {
    fn from_iter<T: IntoIterator<Item = (usize, usize)>>(iter: T) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl PartialEq for Positions {
    fn eq(&self, other: &Self) -> bool {
        self.view().iter().eq(other.view().iter())
    }
}

impl PartialEq<Vec<(usize, usize)>> for Positions {
    fn eq(&self, other: &Vec<(usize, usize)>) -> bool {
        self.view().iter().eq(other.iter().copied())
    }
}

#[test]
fn compact_indices_preserve_offsets_wider_than_u32_without_large_arrays() {
    let wide = u64::from(u32::MAX) + 7;
    let Ok(row) = usize::try_from(wide) else {
        return;
    };
    let mut builder = Builder::new(1);
    builder.push(Some((0, row)), "asof").unwrap();
    let positions = builder.finish();
    assert_eq!(positions, vec![(1, row)]);
    let view = positions.view();
    let indices = view.prefix.unwrap().indices();
    assert_eq!(indices.value(0), wide);
}

#[test]
fn promotion_checks_stop_after_at_most_sixty_four_converted_positions() {
    let cancellation = crate::CancellationToken::new();
    let job = crate::StreamJobContext::new(
        923,
        "bounded-position-promotion",
        crate::JsonMap::new(),
        None,
        cancellation.clone(),
    );
    CONVERTED.with(|converted| converted.set(0));
    let stop = GatherStop::from_job(&job).with_check_hook(Arc::new(move |atomic_only| {
        assert!(
            !atomic_only,
            "promotion must check deadlines and job cancellation"
        );
        if CONVERTED.with(Cell::get) >= 64 {
            cancellation.cancel();
        }
    }));
    let mut builder = Builder::new(4_096);
    builder.set_stop(stop);
    builder.push(None, "asof").unwrap();
    for row in 1..4_095 {
        builder.push(Some((0, row - 1)), "asof").unwrap();
    }
    let result = builder.push(Some((1, 0)), "asof");
    assert!(matches!(
        result,
        Err(crate::CalcFlowError::Cancelled { .. })
    ));
    assert_eq!(CONVERTED.with(Cell::get), 64);
    assert_eq!(builder.compact.as_ref().unwrap().rows.len(), 4_095);
    assert!(
        builder.pairs.is_empty(),
        "cancelled promotion published partial tuples"
    );
}
