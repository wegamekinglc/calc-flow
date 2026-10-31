use crate::{Result, runtime::streaming::gather_work::GatherStop};
use datafusion::arrow::{
    array::{UInt64Array, builder::BooleanBufferBuilder},
    buffer::{BooleanBuffer, NullBuffer, ScalarBuffer},
};
use std::ops::Range;

#[derive(Debug, Default)]
pub(in crate::operator::asof) struct Positions {
    prefix: Option<Prefix>,
    pairs: Vec<(usize, usize)>,
}

#[derive(Debug)]
struct Prefix {
    source: Option<usize>,
    rows: ScalarBuffer<u64>,
    validity: Option<BooleanBuffer>,
}

pub(super) struct Builder {
    compact: Option<CompactBuilder>,
    pairs: Vec<(usize, usize)>,
    capacity: usize,
    stop: Option<GatherStop>,
}

struct CompactBuilder {
    source: Option<usize>,
    rows: Vec<u64>,
    validity: BooleanBufferBuilder,
    nulls: usize,
}

#[derive(Clone)]
pub(in crate::operator::asof) struct View<'a> {
    pub prefix: Option<PrefixView<'a>>,
    pub pairs: &'a [(usize, usize)],
}

#[derive(Clone)]
pub(in crate::operator::asof) struct PrefixView<'a> {
    pub source: Option<usize>,
    rows: &'a ScalarBuffer<u64>,
    validity: Option<&'a BooleanBuffer>,
    range: Range<usize>,
}

impl Builder {
    pub fn new(capacity: usize) -> Self {
        Self {
            compact: Some(CompactBuilder {
                source: None,
                rows: Vec::with_capacity(capacity),
                validity: BooleanBufferBuilder::new(capacity),
                nulls: 0,
            }),
            pairs: Vec::new(),
            capacity,
            stop: None,
        }
    }

    pub fn legacy(capacity: usize) -> Self {
        Self {
            compact: None,
            pairs: Vec::with_capacity(capacity),
            capacity,
            stop: None,
        }
    }

    pub fn set_stop(&mut self, stop: GatherStop) {
        self.stop = Some(stop);
    }

    pub fn is_compact(&self) -> bool {
        self.compact.is_some()
    }

    pub fn requires_promotion(&self, source: Option<usize>) -> bool {
        self.compact
            .as_ref()
            .is_some_and(|compact| !compact.accepts(source))
    }

    pub fn push(&mut self, right: Option<(usize, usize)>, name: &str) -> Result<()> {
        if let Some(compact) = &mut self.compact
            && compact.accepts(right.map(|(source, _)| source))
        {
            return compact.push(right, name);
        }
        self.promote()?;
        self.pairs
            .push(right.map_or((0, 0), |(source, row)| (source + 1, row)));
        Ok(())
    }

    fn promote(&mut self) -> Result<()> {
        let Some(compact) = &self.compact else {
            return Ok(());
        };
        self.check_stop()?;
        let mut positions = Vec::with_capacity(self.capacity);
        for start in (0..compact.rows.len()).step_by(64) {
            self.check_stop()?;
            let end = (start + 64).min(compact.rows.len());
            positions.extend((start..end).map(|index| compact.position(index)));
            #[cfg(test)]
            test_support::note_converted(end - start);
            self.check_stop()?;
        }
        self.check_stop()?;
        self.pairs = positions;
        self.compact = None;
        Ok(())
    }

    fn check_stop(&self) -> Result<()> {
        self.stop.as_ref().map_or(Ok(()), GatherStop::check)
    }

    pub fn finish(self) -> Positions {
        Positions {
            prefix: self.compact.map(CompactBuilder::finish),
            pairs: self.pairs,
        }
    }
}

impl CompactBuilder {
    fn accepts(&self, source: Option<usize>) -> bool {
        self.source
            .is_none_or(|current| source.is_none_or(|source| source == current))
    }

    fn push(&mut self, right: Option<(usize, usize)>, name: &str) -> Result<()> {
        let row =
            u64::try_from(right.map_or(0, |(_, row)| row)).map_err(|_| super::overflow(name))?;
        self.rows.push(row);
        self.validity.append(right.is_some());
        self.nulls += usize::from(right.is_none());
        self.source = self.source.or(right.map(|(source, _)| source));
        Ok(())
    }

    fn finish(self) -> Prefix {
        Prefix {
            source: self.source,
            rows: self.rows.into(),
            validity: (self.nulls != 0).then(|| self.validity.build()),
        }
    }

    fn position(&self, index: usize) -> (usize, usize) {
        if self.nulls != 0 && !self.validity.get_bit(index) {
            (0, 0)
        } else {
            (
                self.source.expect("non-null ASOF prefix source") + 1,
                usize::try_from(self.rows[index]).expect("validated ASOF source row"),
            )
        }
    }
}

impl Positions {
    pub fn len(&self) -> usize {
        self.prefix.as_ref().map_or(0, |prefix| prefix.rows.len()) + self.pairs.len()
    }

    pub fn view(&self) -> View<'_> {
        View {
            prefix: self.prefix.as_ref().map(|prefix| PrefixView {
                source: prefix.source,
                rows: &prefix.rows,
                validity: prefix.validity.as_ref(),
                range: 0..prefix.rows.len(),
            }),
            pairs: &self.pairs,
        }
    }
}

impl View<'_> {
    pub fn slice(self, range: Range<usize>) -> Self {
        let prefix_len = self.prefix.as_ref().map_or(0, PrefixView::len);
        let prefix = self.prefix.and_then(|prefix| {
            let start = range.start.min(prefix_len);
            let end = range.end.min(prefix_len);
            (start < end).then(|| prefix.slice(start..end))
        });
        Self {
            prefix,
            pairs: &self.pairs
                [range.start.saturating_sub(prefix_len)..range.end.saturating_sub(prefix_len)],
        }
    }

    pub fn len(&self) -> usize {
        self.prefix.as_ref().map_or(0, PrefixView::len) + self.pairs.len()
    }

    pub fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.prefix
            .iter()
            .flat_map(PrefixView::iter)
            .chain(self.pairs.iter().copied())
    }
}

impl PrefixView<'_> {
    pub fn len(&self) -> usize {
        self.range.len()
    }

    fn slice(mut self, range: Range<usize>) -> Self {
        self.range = self.range.start + range.start..self.range.start + range.end;
        self
    }

    pub fn indices(&self) -> UInt64Array {
        let validity = self
            .validity
            .map(|validity| NullBuffer::new(validity.slice(self.range.start, self.len())));
        UInt64Array::new(self.rows.slice(self.range.start, self.len()), validity)
    }

    fn iter(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.range.clone().map(|index| {
            if self.validity.is_some_and(|validity| !validity.value(index)) {
                (0, 0)
            } else {
                (
                    self.source.expect("non-null ASOF prefix source") + 1,
                    usize::try_from(self.rows[index]).expect("validated ASOF source row"),
                )
            }
        })
    }
}

#[cfg(test)]
#[path = "positions/test_support.rs"]
pub(super) mod test_support;
