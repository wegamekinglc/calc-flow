use super::{Result, ScalarValue};
use std::{borrow::Cow, ops::Deref};

pub(super) enum GroupAggregates {
    Generic {
        states: Vec<Vec<ScalarValue>>,
        results: Vec<ScalarValue>,
    },
}

pub(super) enum StateView<'a> {
    Borrowed(&'a [ScalarValue]),
}

impl Deref for StateView<'_> {
    type Target = [ScalarValue];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(values) => values,
        }
    }
}

impl AsRef<[ScalarValue]> for StateView<'_> {
    fn as_ref(&self) -> &[ScalarValue] {
        self
    }
}

impl GroupAggregates {
    pub(super) fn len(&self) -> usize {
        match self {
            Self::Generic { states, .. } => states.len(),
        }
    }

    pub(super) fn state(&self, index: usize) -> StateView<'_> {
        match self {
            Self::Generic { states, .. } => StateView::Borrowed(&states[index]),
        }
    }

    #[cfg(test)]
    pub(super) fn states(&self) -> impl ExactSizeIterator<Item = StateView<'_>> {
        (0..self.len()).map(|index| self.state(index))
    }

    pub(super) fn state_scalar(&self, aggregate: usize, field: usize) -> Cow<'_, ScalarValue> {
        match self {
            Self::Generic { states, .. } => Cow::Borrowed(&states[aggregate][field]),
        }
    }

    pub(super) fn result(&self, index: usize) -> Cow<'_, ScalarValue> {
        match self {
            Self::Generic { results, .. } => Cow::Borrowed(&results[index]),
        }
    }

    pub(super) fn results(&self) -> impl ExactSizeIterator<Item = Cow<'_, ScalarValue>> {
        let count = match self {
            Self::Generic { results, .. } => results.len(),
        };
        (0..count).map(|index| self.result(index))
    }

    pub(super) fn generic_states(&self) -> Option<&[Vec<ScalarValue>]> {
        match self {
            Self::Generic { states, .. } => Some(states),
        }
    }

    pub(super) fn replace(
        &mut self,
        index: usize,
        state: Vec<ScalarValue>,
        result: ScalarValue,
        _name: &str,
    ) -> Result<()> {
        let Self::Generic { states, results } = self;
        states[index] = state;
        results[index] = result;
        Ok(())
    }

    pub(super) fn restored(
        states: Vec<Vec<ScalarValue>>,
        results: Vec<ScalarValue>,
        _name: &str,
    ) -> Result<Self> {
        Ok(Self::Generic { states, results })
    }
}
