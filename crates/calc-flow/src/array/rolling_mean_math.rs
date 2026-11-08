//! Shared floating-point transitions for array and table rolling means.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArithmeticError {
    CountOverflow,
    FiniteCountOverflow,
    RemovalWithoutAdd,
    NonFloatingSum,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FloatSummary {
    pub(crate) valid_count: u64,
    pub(crate) sum: f64,
    pub(crate) mean: f64,
    pub(crate) m2: f64,
    pub(crate) nan_count: u64,
    pub(crate) pos_inf: u64,
    pub(crate) neg_inf: u64,
}

pub(crate) struct SummaryMut<'a> {
    pub(crate) valid_count: &'a mut u64,
    pub(crate) sum: Option<&'a mut f64>,
    pub(crate) mean: &'a mut f64,
    pub(crate) m2: &'a mut f64,
    pub(crate) nan_count: &'a mut u64,
    pub(crate) pos_inf: &'a mut u64,
    pub(crate) neg_inf: &'a mut u64,
}

impl FloatSummary {
    #[inline]
    pub(crate) fn as_mut(&mut self) -> SummaryMut<'_> {
        SummaryMut {
            valid_count: &mut self.valid_count,
            sum: Some(&mut self.sum),
            mean: &mut self.mean,
            m2: &mut self.m2,
            nan_count: &mut self.nan_count,
            pos_inf: &mut self.pos_inf,
            neg_inf: &mut self.neg_inf,
        }
    }

    #[inline]
    pub(crate) fn is_non_finite(&self) -> bool {
        numerical_risk(Some(self.sum), self.mean, self.m2)
    }

    #[inline]
    pub(crate) fn read_mean(&self) -> f64 {
        read_mean(self.nan_count, self.pos_inf, self.neg_inf, self.mean)
    }
}

impl SummaryMut<'_> {
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen mean and variance output type is Float64"
    )]
    #[inline]
    pub(crate) fn add(self, sample: f64) -> Result<(), ArithmeticError> {
        *self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        if sample.is_nan() {
            *self.nan_count = self.nan_count.saturating_add(1);
            return Ok(());
        }
        let total = self.sum.ok_or(ArithmeticError::NonFloatingSum)?;
        *total += sample;
        if sample.is_infinite() {
            update_infinity_count(self.pos_inf, self.neg_inf, sample);
        }
        let count = *self.valid_count as f64;
        let delta = sample - *self.mean;
        *self.mean += delta / count;
        *self.m2 += delta * (sample - *self.mean);
        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen mean and variance output type is Float64"
    )]
    #[inline]
    pub(crate) fn remove(self, sample: f64) -> Result<(), ArithmeticError> {
        *self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or(ArithmeticError::RemovalWithoutAdd)?;
        if sample.is_nan() {
            return Ok(());
        }
        let total = self.sum.ok_or(ArithmeticError::NonFloatingSum)?;
        *total -= sample;
        if sample.is_infinite() {
            if sample > 0.0 {
                *self.pos_inf = self.pos_inf.saturating_sub(1);
            } else {
                *self.neg_inf = self.neg_inf.saturating_sub(1);
            }
        }
        if *self.valid_count == 0 {
            *self.mean = 0.0;
            *self.m2 = 0.0;
        } else {
            let count = *self.valid_count as f64;
            let delta = sample - *self.mean;
            *self.mean -= delta / count;
            *self.m2 -= delta * (sample - *self.mean);
        }
        Ok(())
    }
}

fn update_infinity_count(positive: &mut u64, negative: &mut u64, sample: f64) {
    if sample > 0.0 {
        *positive = positive.saturating_add(1);
    } else {
        *negative = negative.saturating_add(1);
    }
}

#[inline]
pub(crate) fn numerical_risk(sum: Option<f64>, mean: f64, m2: f64) -> bool {
    sum.is_some_and(|value| !value.is_finite()) || !mean.is_finite() || !m2.is_finite()
}

#[inline]
pub(crate) fn read_mean(nan_count: u64, pos_inf: u64, neg_inf: u64, mean: f64) -> f64 {
    if nan_count > 0 {
        return f64::NAN;
    }
    match (pos_inf > 0, neg_inf > 0) {
        (true, true) => f64::NAN,
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => mean,
    }
}

pub(crate) fn rebase_due(count: u64, retained: usize, numerical_risk: bool) -> bool {
    if numerical_risk {
        return true;
    }
    if retained < 2 {
        return false;
    }
    let cadence = u64::try_from(retained.max(64)).unwrap_or(u64::MAX);
    count.is_multiple_of(cadence)
}

#[derive(Default)]
pub(crate) struct CompensatedSum {
    pub(crate) value: f64,
    correction: f64,
}

impl CompensatedSum {
    pub(crate) fn add(&mut self, sample: f64) {
        let adjusted = sample - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

struct StableFold {
    summary: FloatSummary,
    origin: Option<f64>,
    finite_count: u64,
    offsets: CompensatedSum,
    offset_squares: CompensatedSum,
}

impl StableFold {
    fn new() -> Self {
        Self {
            summary: FloatSummary::default(),
            origin: None,
            finite_count: 0,
            offsets: CompensatedSum::default(),
            offset_squares: CompensatedSum::default(),
        }
    }

    fn push(&mut self, sample: f64) -> Result<(), ArithmeticError> {
        self.summary.valid_count = self
            .summary
            .valid_count
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        if sample.is_nan() {
            self.summary.nan_count = self.summary.nan_count.saturating_add(1);
            return Ok(());
        }
        if sample.is_infinite() {
            update_infinity_count(&mut self.summary.pos_inf, &mut self.summary.neg_inf, sample);
            return Ok(());
        }
        let origin = *self.origin.get_or_insert(sample);
        let offset = sample - origin;
        self.offsets.add(offset);
        self.offset_squares.add(offset * offset);
        self.finite_count = self
            .finite_count
            .checked_add(1)
            .ok_or(ArithmeticError::FiniteCountOverflow)?;
        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the stable_v2 preview retains the frozen Float64 output type"
    )]
    fn finish(mut self) -> FloatSummary {
        let origin = self.origin.unwrap_or(0.0);
        let finite_total = origin * self.finite_count as f64 + self.offsets.value;
        self.summary.sum = match (self.summary.pos_inf > 0, self.summary.neg_inf > 0) {
            (true, true) => f64::NAN,
            (true, false) => f64::INFINITY,
            (false, true) => f64::NEG_INFINITY,
            (false, false) => finite_total,
        };
        if self.finite_count > 0 {
            let count = self.finite_count as f64;
            self.summary.mean = origin + self.offsets.value / count;
            self.summary.m2 =
                self.offset_squares.value - self.offsets.value * self.offsets.value / count;
        }
        self.summary
    }
}

pub(crate) fn stable_fold(
    values: impl Iterator<Item = f64>,
) -> Result<FloatSummary, ArithmeticError> {
    stable_fold_controlled(values.map(Some), &mut || Ok(()), std::convert::identity)
}

pub(crate) fn stable_fold_controlled<E>(
    values: impl Iterator<Item = Option<f64>>,
    check: &mut dyn FnMut() -> Result<(), E>,
    arithmetic: impl Fn(ArithmeticError) -> E,
) -> Result<FloatSummary, E> {
    check()?;
    let mut fold = StableFold::new();
    for (index, sample) in values.enumerate() {
        if index.is_multiple_of(4096) {
            check()?;
        }
        if let Some(sample) = sample {
            fold.push(sample).map_err(&arithmetic)?;
        }
    }
    check()?;
    Ok(fold.finish())
}

pub(crate) const fn stable_scratch_bytes() -> usize {
    size_of::<StableFold>() + size_of::<FloatSummary>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_transitions_preserve_counts_and_west_state() {
        let mut state = FloatSummary::default();
        state.as_mut().add(1.0).unwrap();
        state.as_mut().add(3.0).unwrap();
        assert_eq!(state.valid_count, 2);
        assert_eq!(state.mean.to_bits(), 2.0_f64.to_bits());
        assert_eq!(state.m2.to_bits(), 2.0_f64.to_bits());
        state.as_mut().remove(1.0).unwrap();
        assert_eq!(state.valid_count, 1);
        assert_eq!(state.read_mean().to_bits(), 3.0_f64.to_bits());
        assert_eq!(state.m2.to_bits(), 0.0_f64.to_bits());
        state.as_mut().remove(3.0).unwrap();
        assert_eq!(state.valid_count, 0);
        assert_eq!(state.mean.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn test_count_errors_and_nan_compatibility_keep_existing_order() {
        let mut state = FloatSummary {
            valid_count: u64::MAX,
            ..FloatSummary::default()
        };
        assert_eq!(state.as_mut().add(1.0), Err(ArithmeticError::CountOverflow));
        assert_eq!(state.sum.to_bits(), 0.0_f64.to_bits());
        let mut state = FloatSummary::default();
        assert_eq!(
            state.as_mut().remove(1.0),
            Err(ArithmeticError::RemovalWithoutAdd)
        );
        let mut view = state.as_mut();
        view.sum = None;
        view.add(f64::NAN).unwrap();
        assert_eq!(state.valid_count, 1);
        assert_eq!(state.nan_count, 1);
        state.as_mut().remove(f64::NAN).unwrap();
        assert_eq!(state.valid_count, 0);
        assert_eq!(state.nan_count, 1);
        assert!(state.read_mean().is_nan());
        let mut view = state.as_mut();
        view.sum = None;
        assert_eq!(view.add(1.0), Err(ArithmeticError::NonFloatingSum));
        assert_eq!(state.valid_count, 1);
    }

    #[test]
    fn test_stable_fold_and_rebase_cadence_preserve_existing_contract() {
        let state = stable_fold([1.0e12, 1.0e12 + 0.25, 1.0e12 + 0.5].into_iter()).unwrap();
        assert_eq!(state.mean.to_bits(), (1.0e12_f64 + 0.25).to_bits());
        assert_eq!(state.m2.to_bits(), 0.125_f64.to_bits());
        let state = stable_fold([f64::INFINITY, f64::NEG_INFINITY].into_iter()).unwrap();
        assert_eq!(state.valid_count, 2);
        assert!(state.read_mean().is_nan());
        assert!(rebase_due(1, 1, true));
        assert!(!rebase_due(1, 1, false));
        assert!(rebase_due(64, 2, false));
        assert!(!rebase_due(65, 2, false));
    }
}
