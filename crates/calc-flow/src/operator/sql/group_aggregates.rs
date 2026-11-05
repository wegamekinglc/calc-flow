use super::{
    AggregateFunctionExpr, Arc, DataType, GroupKey, Result, ScalarValue, df_error, grouped_sum,
};
use std::{borrow::Cow, ops::Deref};

#[derive(Clone, Copy)]
pub(super) enum InlineSum {
    Int64(Option<i64>),
    UInt64(Option<u64>),
}

impl InlineSum {
    fn null(dtype: &DataType) -> Option<Self> {
        match dtype {
            DataType::Int64 => Some(Self::Int64(None)),
            DataType::UInt64 => Some(Self::UInt64(None)),
            _ => None,
        }
    }
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Int64(value) => ScalarValue::Int64(value),
            Self::UInt64(value) => ScalarValue::UInt64(value),
        }
    }

    fn from_scalar(value: &ScalarValue, name: &str) -> Result<Self> {
        match value {
            ScalarValue::Int64(value) => Ok(Self::Int64(*value)),
            ScalarValue::UInt64(value) => Ok(Self::UInt64(*value)),
            _ => Err(df_error(name, "inline SUM state type differs")),
        }
    }
}

pub(super) fn bind_inline_sum(
    keys: &[GroupKey],
    aggregates: &[Arc<AggregateFunctionExpr>],
) -> Option<InlineSum> {
    let [expression] = aggregates else {
        return None;
    };
    if keys.is_empty() || !grouped_sum::integer_sum(expression) {
        return None;
    }
    InlineSum::null(expression.field().data_type())
}

pub(super) enum GroupAggregates {
    IntegerSum(InlineSum),
    Generic {
        states: Vec<Vec<ScalarValue>>,
        results: Vec<ScalarValue>,
    },
}

pub(super) enum StateView<'a> {
    Borrowed(&'a [ScalarValue]),
    Inline([ScalarValue; 1]),
}

impl Deref for StateView<'_> {
    type Target = [ScalarValue];

    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(values) => values,
            Self::Inline(values) => values,
        }
    }
}

impl AsRef<[ScalarValue]> for StateView<'_> {
    fn as_ref(&self) -> &[ScalarValue] {
        self
    }
}

impl GroupAggregates {
    pub(super) fn is_inline(&self) -> bool {
        matches!(self, Self::IntegerSum(_))
    }

    pub(super) fn len(&self) -> usize {
        match self {
            Self::IntegerSum(_) => 1,
            Self::Generic { states, .. } => states.len(),
        }
    }

    pub(super) fn state(&self, index: usize) -> StateView<'_> {
        match self {
            Self::IntegerSum(value) => {
                assert_eq!(index, 0, "inline SUM aggregate index");
                StateView::Inline([value.scalar()])
            }
            Self::Generic { states, .. } => StateView::Borrowed(&states[index]),
        }
    }

    #[cfg(test)]
    pub(super) fn states(&self) -> impl ExactSizeIterator<Item = StateView<'_>> {
        (0..self.len()).map(|index| self.state(index))
    }

    pub(super) fn state_scalar(&self, aggregate: usize, field: usize) -> Cow<'_, ScalarValue> {
        match self {
            Self::IntegerSum(value) => {
                assert_eq!((aggregate, field), (0, 0), "inline SUM state index");
                Cow::Owned(value.scalar())
            }
            Self::Generic { states, .. } => Cow::Borrowed(&states[aggregate][field]),
        }
    }

    pub(super) fn result(&self, index: usize) -> Cow<'_, ScalarValue> {
        match self {
            Self::IntegerSum(value) => {
                assert_eq!(index, 0, "inline SUM result index");
                Cow::Owned(value.scalar())
            }
            Self::Generic { results, .. } => Cow::Borrowed(&results[index]),
        }
    }

    pub(super) fn results(&self) -> impl ExactSizeIterator<Item = Cow<'_, ScalarValue>> {
        let count = match self {
            Self::IntegerSum(_) => 1,
            Self::Generic { results, .. } => results.len(),
        };
        (0..count).map(|index| self.result(index))
    }

    pub(super) fn generic_states(&self) -> Option<&[Vec<ScalarValue>]> {
        match self {
            Self::Generic { states, .. } => Some(states),
            Self::IntegerSum(_) => None,
        }
    }

    pub(super) fn replace_inline(&mut self, value: &ScalarValue, name: &str) -> Result<()> {
        let Self::IntegerSum(previous) = self else {
            return Err(df_error(name, "inline update targets generic aggregate"));
        };
        let next = InlineSum::from_scalar(value, name)?;
        if std::mem::discriminant(previous) != std::mem::discriminant(&next) {
            return Err(df_error(name, "inline SUM state type differs"));
        }
        *previous = next;
        Ok(())
    }

    pub(super) fn replace(
        &mut self,
        index: usize,
        state: Vec<ScalarValue>,
        result: ScalarValue,
        name: &str,
    ) -> Result<()> {
        let Self::Generic { states, results } = self else {
            return Err(df_error(name, "generic update targets inline SUM"));
        };
        states[index] = state;
        results[index] = result;
        Ok(())
    }

    pub(super) fn restored(
        inline: bool,
        states: Vec<Vec<ScalarValue>>,
        results: Vec<ScalarValue>,
        name: &str,
    ) -> Result<Self> {
        if !inline {
            return Ok(Self::Generic { states, results });
        }
        let ([state], [result]) = (states.as_slice(), results.as_slice()) else {
            return Err(df_error(
                name,
                "inline SUM restored aggregate count differs",
            ));
        };
        let [value] = state.as_slice() else {
            return Err(df_error(name, "inline SUM restored field count differs"));
        };
        if value != result {
            return Err(df_error(name, "inline SUM restored result differs"));
        }
        Ok(Self::IntegerSum(InlineSum::from_scalar(value, name)?))
    }
}
