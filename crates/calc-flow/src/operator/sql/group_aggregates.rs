use super::ScalarValue;
use std::{borrow::Cow, ops::Deref};

pub(super) struct GroupAggregates {
    pub(super) states: Vec<Vec<ScalarValue>>,
    pub(super) results: Vec<ScalarValue>,
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
        self.states.len()
    }

    pub(super) fn state(&self, index: usize) -> StateView<'_> {
        StateView::Borrowed(&self.states[index])
    }

    pub(super) fn state_scalar(&self, aggregate: usize, field: usize) -> Cow<'_, ScalarValue> {
        Cow::Borrowed(&self.states[aggregate][field])
    }

    pub(super) fn result(&self, index: usize) -> Cow<'_, ScalarValue> {
        Cow::Borrowed(&self.results[index])
    }

    pub(super) fn results(&self) -> impl ExactSizeIterator<Item = Cow<'_, ScalarValue>> {
        (0..self.results.len()).map(|index| self.result(index))
    }

    pub(super) fn generic_states(&self) -> &[Vec<ScalarValue>] {
        &self.states
    }

    pub(super) fn replace(&mut self, index: usize, state: Vec<ScalarValue>, result: ScalarValue) {
        self.states[index] = state;
        self.results[index] = result;
    }

    pub(super) fn restored(states: Vec<Vec<ScalarValue>>, results: Vec<ScalarValue>) -> Self {
        Self { states, results }
    }
}
