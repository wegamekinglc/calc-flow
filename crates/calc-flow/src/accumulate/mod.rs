//! Shared compensated arithmetic for rolling windows.
//!
//! Families own only floating-point sums. Window containers retain sample
//! validity and history, and call [`NumericAccumulator::finish_rows`] or
//! [`NumericAccumulator::finish_duration`] once after each finalized row.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArithmeticError {
    CountOverflow,
    FiniteCountOverflow,
    RemovalWithoutAdd,
    NonFloatingSum,
}

/// Knuth's branch-free `TwoSum`. The correction is kept separate from `sum`.
#[inline]
pub(crate) fn two_sum(left: f64, right: f64) -> (f64, f64) {
    let sum = left + right;
    let right_virtual = sum - left;
    let left_virtual = sum - right_virtual;
    let right_error = right - right_virtual;
    let left_error = left - left_virtual;
    (sum, left_error + right_error)
}

#[derive(Clone, Copy, Debug, Default)]
struct CompensatedSum {
    value: f64,
    error: f64,
}

impl CompensatedSum {
    #[inline]
    fn add(&mut self, operand: f64) {
        let (sum, error) = two_sum(self.value, operand);
        self.value = sum;
        self.error += error;
    }

    #[inline]
    fn pair(&mut self, entering: f64, leaving: f64) {
        let (delta, delta_error) = two_sum(entering, -leaving);
        let (sum, error) = two_sum(self.value, delta);
        self.value = sum;
        self.error += delta_error + error;
    }

    #[inline]
    fn read(self) -> f64 {
        read_compensated(self.value, self.error)
    }

    #[inline]
    fn is_non_finite(self) -> bool {
        !self.value.is_finite() || !self.error.is_finite()
    }
}

#[inline]
fn read_compensated(sum: f64, error: f64) -> f64 {
    if sum.is_finite() { sum + error } else { sum }
}

/// Compensated sum of finite valid samples in one window group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SumFamily {
    sum: CompensatedSum,
    enabled: bool,
}

impl Default for SumFamily {
    fn default() -> Self {
        Self {
            sum: CompensatedSum::default(),
            enabled: true,
        }
    }
}

impl SumFamily {
    #[inline]
    pub(crate) fn add(&mut self, sample: f64) {
        if self.enabled {
            self.sum.add(sample);
        }
    }

    #[inline]
    pub(crate) fn remove(&mut self, sample: f64) {
        self.add(-sample);
    }

    #[inline]
    pub(crate) fn pair(&mut self, entering: f64, leaving: f64) {
        if self.enabled {
            self.sum.pair(entering, leaving);
        }
    }

    pub(crate) fn rebuild(&mut self, samples: impl Iterator<Item = f64>) {
        self.sum = CompensatedSum::default();
        if self.enabled {
            samples.for_each(|sample| self.sum.add(sample));
        }
    }

    #[inline]
    pub(crate) fn read(&self) -> f64 {
        self.sum.read()
    }

    #[inline]
    fn is_non_finite(&self) -> bool {
        self.enabled && self.sum.is_non_finite()
    }

    pub(crate) fn raw_parts(&self) -> (f64, f64) {
        (self.sum.value, self.sum.error)
    }

    fn set_raw_parts(&mut self, value: f64, error: f64) {
        self.sum = CompensatedSum { value, error };
    }

    fn reset(&mut self) {
        self.sum = CompensatedSum::default();
    }
}

/// Shifted compensated `Σ(x-k)` and `Σ(x-k)²` for dispersion readouts.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DispersionFamily {
    shift: Option<f64>,
    offsets: CompensatedSum,
    squares: CompensatedSum,
    enabled: bool,
}

impl Default for DispersionFamily {
    fn default() -> Self {
        Self {
            shift: None,
            offsets: CompensatedSum::default(),
            squares: CompensatedSum::default(),
            enabled: true,
        }
    }
}

impl DispersionFamily {
    fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    pub(crate) fn add(&mut self, sample: f64) {
        if !self.enabled {
            return;
        }
        let shift = *self.shift.get_or_insert(sample);
        let offset = sample - shift;
        self.offsets.add(offset);
        self.squares.add(offset * offset);
    }

    pub(crate) fn remove(&mut self, sample: f64) {
        if !self.enabled {
            return;
        }
        let Some(shift) = self.shift else {
            return;
        };
        let offset = sample - shift;
        self.offsets.add(-offset);
        self.squares.add(-(offset * offset));
    }

    pub(crate) fn pair(&mut self, entering: f64, leaving: f64) {
        if !self.enabled {
            return;
        }
        let Some(shift) = self.shift else {
            return;
        };
        let entering_offset = entering - shift;
        let leaving_offset = leaving - shift;
        self.offsets.pair(entering_offset, leaving_offset);
        self.squares.pair(
            entering_offset * entering_offset,
            leaving_offset * leaving_offset,
        );
    }

    /// Refolds oldest-to-newest after choosing the newest finite sample as shift.
    pub(crate) fn rebuild(&mut self, samples: impl Iterator<Item = f64> + Clone) {
        if !self.enabled {
            self.reset();
            return;
        }
        let newest = samples.clone().last();
        self.reset();
        self.shift = newest;
        samples.for_each(|sample| self.add(sample));
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "rolling dispersion reads Float64 by contract"
    )]
    pub(crate) fn moment(&self, count: u64) -> f64 {
        let offsets = self.offsets.read();
        let squares = self.squares.read();
        let centered = squares - (offsets * offsets) / count as f64;
        if centered < 0.0 { 0.0 } else { centered }
    }

    #[inline]
    fn is_non_finite(&self) -> bool {
        self.enabled && (self.offsets.is_non_finite() || self.squares.is_non_finite())
    }

    #[cfg(test)]
    pub(crate) const fn shift(&self) -> Option<f64> {
        self.shift
    }

    fn reset(&mut self) {
        self.shift = None;
        self.offsets = CompensatedSum::default();
        self.squares = CompensatedSum::default();
    }
}

/// Shifted compensated co-moments over pairwise-valid finite samples.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PairFamily {
    shifts: Option<(f64, f64)>,
    left_offsets: CompensatedSum,
    right_offsets: CompensatedSum,
    cross_products: CompensatedSum,
    left_squares: CompensatedSum,
    right_squares: CompensatedSum,
}

impl PairFamily {
    pub(crate) fn add(&mut self, left: f64, right: f64) {
        let (left_shift, right_shift) = *self.shifts.get_or_insert((left, right));
        let left_offset = left - left_shift;
        let right_offset = right - right_shift;
        self.left_offsets.add(left_offset);
        self.right_offsets.add(right_offset);
        self.cross_products.add(left_offset * right_offset);
        self.left_squares.add(left_offset * left_offset);
        self.right_squares.add(right_offset * right_offset);
    }

    pub(crate) fn remove(&mut self, left: f64, right: f64) {
        let Some((left_shift, right_shift)) = self.shifts else {
            return;
        };
        let left_offset = left - left_shift;
        let right_offset = right - right_shift;
        self.left_offsets.add(-left_offset);
        self.right_offsets.add(-right_offset);
        self.cross_products.add(-(left_offset * right_offset));
        self.left_squares.add(-(left_offset * left_offset));
        self.right_squares.add(-(right_offset * right_offset));
    }

    pub(crate) fn pair(&mut self, entering: (f64, f64), leaving: (f64, f64)) {
        let Some((left_shift, right_shift)) = self.shifts else {
            return;
        };
        let entering_left = entering.0 - left_shift;
        let entering_right = entering.1 - right_shift;
        let leaving_left = leaving.0 - left_shift;
        let leaving_right = leaving.1 - right_shift;
        self.left_offsets.pair(entering_left, leaving_left);
        self.right_offsets.pair(entering_right, leaving_right);
        self.cross_products
            .pair(entering_left * entering_right, leaving_left * leaving_right);
        self.left_squares
            .pair(entering_left * entering_left, leaving_left * leaving_left);
        self.right_squares.pair(
            entering_right * entering_right,
            leaving_right * leaving_right,
        );
    }

    pub(crate) fn rebuild(&mut self, samples: impl Iterator<Item = (f64, f64)> + Clone) {
        let newest = samples.clone().last();
        self.reset();
        self.shifts = newest;
        samples.for_each(|(left, right)| self.add(left, right));
    }

    pub(crate) fn covariance(&self, count: u64, ddof: u8) -> f64 {
        let left = self.left_offsets.read();
        let right = self.right_offsets.read();
        let cross = self.cross_products.read();
        (cross - (left * right) / count_float(count)) / count_float(count - u64::from(ddof))
    }

    pub(crate) fn correlation(&self, count: u64) -> Option<f64> {
        let left = self.moment(true, count);
        let right = self.moment(false, count);
        if left == 0.0 || right == 0.0 {
            return None;
        }
        let left_offset = self.left_offsets.read();
        let right_offset = self.right_offsets.read();
        let cross = self.cross_products.read();
        let co_moment = cross - (left_offset * right_offset) / count_float(count);
        Some(co_moment / (left.sqrt() * right.sqrt()))
    }

    fn moment(&self, left: bool, count: u64) -> f64 {
        let (offsets, squares) = if left {
            (self.left_offsets, self.left_squares)
        } else {
            (self.right_offsets, self.right_squares)
        };
        let offsets = offsets.read();
        let centered = squares.read() - (offsets * offsets) / count_float(count);
        if centered < 0.0 { 0.0 } else { centered }
    }

    pub(crate) fn is_non_finite(&self) -> bool {
        self.left_offsets.is_non_finite()
            || self.right_offsets.is_non_finite()
            || self.cross_products.is_non_finite()
            || self.left_squares.is_non_finite()
            || self.right_squares.is_non_finite()
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Raw layout-7 fields for bit-identical numeric checkpoint continuation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericAccumulatorLayout7 {
    pub(crate) valid_count: u64,
    pub(crate) finite_count: u64,
    pub(crate) nan_count: u64,
    pub(crate) pos_inf: u64,
    pub(crate) neg_inf: u64,
    pub(crate) sum_value: f64,
    pub(crate) sum_error: f64,
    pub(crate) sum_enabled: bool,
    pub(crate) shift: Option<f64>,
    pub(crate) offsets_value: f64,
    pub(crate) offsets_error: f64,
    pub(crate) squares_value: f64,
    pub(crate) squares_error: f64,
    pub(crate) dispersion_enabled: bool,
    pub(crate) since_rebuild: u32,
    pub(crate) n_prev: u64,
}

/// Shared validity, compensated families, and rebuild cadence for a scalar group.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NumericAccumulator {
    pub(crate) valid_count: u64,
    pub(crate) finite_count: u64,
    pub(crate) nan_count: u64,
    pub(crate) pos_inf: u64,
    pub(crate) neg_inf: u64,
    sum: SumFamily,
    dispersion: DispersionFamily,
    since_rebuild: u32,
    n_prev: u64,
}

impl Default for NumericAccumulator {
    fn default() -> Self {
        Self {
            valid_count: 0,
            finite_count: 0,
            nan_count: 0,
            pos_inf: 0,
            neg_inf: 0,
            sum: SumFamily::default(),
            dispersion: DispersionFamily::default(),
            since_rebuild: 0,
            n_prev: 64,
        }
    }
}

impl NumericAccumulator {
    pub(crate) fn mean_only() -> Self {
        Self {
            dispersion: DispersionFamily::disabled(),
            ..Self::default()
        }
    }

    /// Applies one Rows-frame transition, pairing finite leave/enter operands
    /// whenever the post-eviction group still has a finite shift.
    pub(crate) fn rows(
        &mut self,
        leaving: Option<f64>,
        entering: Option<f64>,
    ) -> Result<(), ArithmeticError> {
        if leaving.is_some_and(f64::is_finite)
            && entering.is_some_and(f64::is_finite)
            && self.finite_count > 1
        {
            return self.paired_rows(leaving.unwrap(), entering.unwrap());
        }
        self.unpaired_rows(leaving, entering)
    }

    fn paired_rows(&mut self, leaving: f64, entering: f64) -> Result<(), ArithmeticError> {
        self.evict_counters(leaving)?;
        self.insert_counters(entering)?;
        self.sum.pair(entering, leaving);
        self.dispersion.pair(entering, leaving);
        Ok(())
    }

    fn unpaired_rows(
        &mut self,
        leaving: Option<f64>,
        entering: Option<f64>,
    ) -> Result<(), ArithmeticError> {
        if let Some(sample) = leaving {
            self.evict(sample)?;
        }
        self.reset_if_empty();
        if let Some(sample) = entering {
            self.insert(sample)?;
        }
        Ok(())
    }

    pub(crate) fn insert(&mut self, sample: f64) -> Result<(), ArithmeticError> {
        self.insert_counters(sample)?;
        if sample.is_finite() {
            self.sum.add(sample);
            self.dispersion.add(sample);
        }
        Ok(())
    }

    pub(crate) fn insert_count_only(&mut self) -> Result<(), ArithmeticError> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        Ok(())
    }

    pub(crate) fn evict(&mut self, sample: f64) -> Result<(), ArithmeticError> {
        self.evict_counters(sample)?;
        if sample.is_finite() {
            self.sum.remove(sample);
            self.dispersion.remove(sample);
        }
        Ok(())
    }

    pub(crate) fn evict_count_only(&mut self) -> Result<(), ArithmeticError> {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or(ArithmeticError::RemovalWithoutAdd)?;
        Ok(())
    }

    pub(crate) fn reset_if_empty(&mut self) {
        if self.finite_count == 0 {
            self.sum.reset();
            self.dispersion.reset();
            self.since_rebuild = 0;
            self.n_prev = 64;
        }
    }

    pub(crate) fn finish_rows(
        &mut self,
        window: usize,
        history: impl Iterator<Item = Option<f64>> + Clone,
    ) -> Result<(), ArithmeticError> {
        let retained = history.clone().count();
        self.since_rebuild = self
            .since_rebuild
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        let cadence = u64::try_from(window.max(64)).unwrap_or(u64::MAX);
        self.repair(cadence, retained, finite_samples(history));
        Ok(())
    }

    pub(crate) fn finish_duration(
        &mut self,
        history: impl Iterator<Item = Option<f64>> + Clone,
    ) -> Result<(), ArithmeticError> {
        let retained = history.clone().count();
        self.since_rebuild = self
            .since_rebuild
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        self.repair(self.n_prev.max(64), retained, finite_samples(history));
        Ok(())
    }

    fn repair(&mut self, cadence: u64, retained: usize, finite: impl Iterator<Item = f64> + Clone) {
        if u64::from(self.since_rebuild) >= cadence {
            self.sum.rebuild(finite.clone());
            self.dispersion.rebuild(finite);
            self.since_rebuild = 0;
            self.n_prev = u64::try_from(retained).unwrap_or(u64::MAX);
            return;
        }
        if self.sum.is_non_finite() {
            self.sum.rebuild(finite.clone());
        }
        if self.dispersion.is_non_finite() {
            self.dispersion.rebuild(finite);
        }
    }

    pub(crate) fn sum(&self) -> f64 {
        if self.nan_count > 0 {
            return f64::NAN;
        }
        let finite = self.sum.read();
        match (self.pos_inf > 0, self.neg_inf > 0) {
            (true, true) => f64::NAN,
            (true, false) => finite + f64::INFINITY,
            (false, true) => finite + f64::NEG_INFINITY,
            (false, false) => finite,
        }
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "rolling mean reads Float64 by contract"
    )]
    pub(crate) fn mean(&self) -> f64 {
        self.sum() / self.valid_count as f64
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "rolling variance reads Float64 by contract"
    )]
    pub(crate) fn variance(&self, ddof: u8) -> f64 {
        if self.nan_count > 0 || self.pos_inf > 0 || self.neg_inf > 0 {
            return f64::NAN;
        }
        self.dispersion.moment(self.valid_count) / (self.valid_count - u64::from(ddof)) as f64
    }

    pub(crate) fn stddev(&self, ddof: u8) -> f64 {
        self.variance(ddof).sqrt()
    }

    pub(crate) fn layout7(self) -> NumericAccumulatorLayout7 {
        NumericAccumulatorLayout7 {
            valid_count: self.valid_count,
            finite_count: self.finite_count,
            nan_count: self.nan_count,
            pos_inf: self.pos_inf,
            neg_inf: self.neg_inf,
            sum_value: self.sum.sum.value,
            sum_error: self.sum.sum.error,
            sum_enabled: self.sum.enabled,
            shift: self.dispersion.shift,
            offsets_value: self.dispersion.offsets.value,
            offsets_error: self.dispersion.offsets.error,
            squares_value: self.dispersion.squares.value,
            squares_error: self.dispersion.squares.error,
            dispersion_enabled: self.dispersion.enabled,
            since_rebuild: self.since_rebuild,
            n_prev: self.n_prev,
        }
    }

    pub(crate) fn from_layout7(parts: NumericAccumulatorLayout7) -> Self {
        Self {
            valid_count: parts.valid_count,
            finite_count: parts.finite_count,
            nan_count: parts.nan_count,
            pos_inf: parts.pos_inf,
            neg_inf: parts.neg_inf,
            sum: SumFamily {
                sum: CompensatedSum {
                    value: parts.sum_value,
                    error: parts.sum_error,
                },
                enabled: parts.sum_enabled,
            },
            dispersion: DispersionFamily {
                shift: parts.shift,
                offsets: CompensatedSum {
                    value: parts.offsets_value,
                    error: parts.offsets_error,
                },
                squares: CompensatedSum {
                    value: parts.squares_value,
                    error: parts.squares_error,
                },
                enabled: parts.dispersion_enabled,
            },
            since_rebuild: parts.since_rebuild,
            n_prev: parts.n_prev,
        }
    }

    pub(crate) fn raw_sum_parts(&self) -> (f64, f64) {
        self.sum.raw_parts()
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "restore consumes the frozen counter and sum fields independently"
    )]
    pub(crate) fn restore_mean_parts(
        &mut self,
        valid: u64,
        finite: u64,
        nan: u64,
        positive: u64,
        negative: u64,
        sum: f64,
        error: f64,
    ) {
        self.valid_count = valid;
        self.finite_count = finite;
        self.nan_count = nan;
        self.pos_inf = positive;
        self.neg_inf = negative;
        self.sum.set_raw_parts(sum, error);
        self.since_rebuild = 0;
        self.n_prev = 64;
    }

    fn insert_counters(&mut self, sample: f64) -> Result<(), ArithmeticError> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or(ArithmeticError::CountOverflow)?;
        if sample.is_nan() {
            self.nan_count = self.nan_count.saturating_add(1);
        } else if sample.is_infinite() {
            update_infinity(&mut self.pos_inf, &mut self.neg_inf, sample, true);
        } else {
            self.finite_count = self
                .finite_count
                .checked_add(1)
                .ok_or(ArithmeticError::FiniteCountOverflow)?;
        }
        Ok(())
    }

    fn evict_counters(&mut self, sample: f64) -> Result<(), ArithmeticError> {
        self.valid_count = self
            .valid_count
            .checked_sub(1)
            .ok_or(ArithmeticError::RemovalWithoutAdd)?;
        if sample.is_nan() {
            // Sticky AVG NaN is partition state, not window membership.
        } else if sample.is_infinite() {
            update_infinity(&mut self.pos_inf, &mut self.neg_inf, sample, false);
        } else {
            self.finite_count = self
                .finite_count
                .checked_sub(1)
                .ok_or(ArithmeticError::RemovalWithoutAdd)?;
        }
        Ok(())
    }
}

fn finite_samples(
    history: impl Iterator<Item = Option<f64>> + Clone,
) -> impl Iterator<Item = f64> + Clone {
    history.filter_map(|sample| sample.filter(|value| value.is_finite()))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "rolling pair readouts use the frozen Float64 contract"
)]
fn count_float(count: u64) -> f64 {
    count as f64
}

fn update_infinity(positive: &mut u64, negative: &mut u64, sample: f64, add: bool) {
    let count = if sample > 0.0 { positive } else { negative };
    *count = if add {
        count.saturating_add(1)
    } else {
        count.saturating_sub(1)
    };
}

#[cfg(test)]
mod tests {
    use super::{DispersionFamily, NumericAccumulator, PairFamily, SumFamily, two_sum};

    fn advance(
        accumulator: &mut NumericAccumulator,
        history: &mut Vec<Option<f64>>,
        window: usize,
        sample: Option<f64>,
    ) {
        let leaving = (history.len() == window)
            .then(|| history.first().copied().flatten())
            .flatten();
        accumulator.rows(leaving, sample).unwrap();
        if history.len() == window {
            history.remove(0);
        }
        history.push(sample);
        accumulator
            .finish_rows(window, history.iter().copied())
            .unwrap();
    }

    fn mean_bits(samples: &[Option<f64>], chunks: &[usize]) -> Vec<u64> {
        let mut accumulator = NumericAccumulator::default();
        let mut history = Vec::new();
        let mut outputs = Vec::new();
        let mut position = 0;
        for &chunk in chunks {
            let end = position + chunk;
            for &sample in &samples[position..end] {
                advance(&mut accumulator, &mut history, 3, sample);
                outputs.push(accumulator.mean().to_bits());
            }
            position = end;
        }
        assert_eq!(position, samples.len());
        outputs
    }

    #[test]
    fn two_sum_retains_the_low_operand_without_rewriting_the_high_sum() {
        let (sum, error) = two_sum(1.0e16, 1.0);
        assert_eq!(sum.to_bits(), 1.0e16_f64.to_bits());
        assert_eq!(error.to_bits(), 1.0_f64.to_bits());
    }

    #[test]
    fn compensated_rows_are_identical_across_batch_splits() {
        let samples = [
            Some(1.0e12),
            None,
            Some(1.0e12 + 0.25),
            Some(f64::NAN),
            Some(-1.0e12),
            Some(1.0e12 + 0.5),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(3.0),
        ];
        let whole = mean_bits(&samples, &[samples.len()]);
        for splits in [vec![1; samples.len()], vec![2, 1, 4, 2], vec![5, 4]] {
            assert_eq!(mean_bits(&samples, &splits), whole);
        }
    }

    #[test]
    fn near_overflow_sum_is_compensated_and_mean_divides_only_at_readout() {
        let mut accumulator = NumericAccumulator::default();
        let mut history = Vec::new();
        advance(&mut accumulator, &mut history, 2, Some(-1.0e308));
        advance(&mut accumulator, &mut history, 2, Some(1.0e308));
        assert_eq!(accumulator.sum().to_bits(), 0.0_f64.to_bits());
        assert_eq!(accumulator.mean().to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn overflow_and_mixed_infinities_follow_the_classification_contract() {
        let mut accumulator = NumericAccumulator::default();
        let mut history = Vec::new();
        advance(&mut accumulator, &mut history, 2, Some(1.0e308));
        advance(&mut accumulator, &mut history, 2, Some(1.0e308));
        assert_eq!(accumulator.sum().to_bits(), f64::INFINITY.to_bits());
        assert_eq!(accumulator.mean().to_bits(), f64::INFINITY.to_bits());
        advance(&mut accumulator, &mut history, 2, Some(f64::INFINITY));
        advance(&mut accumulator, &mut history, 2, Some(f64::NEG_INFINITY));
        assert!(accumulator.sum().is_nan());
        assert!(accumulator.mean().is_nan());
        assert!(accumulator.variance(0).is_nan());
    }

    #[test]
    fn dispersion_overflow_rebuild_does_not_change_mean_bits_or_clamp_nan() {
        let samples = [1.0e200, -1.0e200, 1.0e200];
        let mut full = NumericAccumulator::default();
        let mut mean_only = NumericAccumulator::mean_only();
        let mut full_history = Vec::new();
        let mut mean_history = Vec::new();
        for sample in samples.into_iter().map(Some) {
            advance(&mut full, &mut full_history, 3, sample);
            advance(&mut mean_only, &mut mean_history, 3, sample);
            assert_eq!(full.mean().to_bits(), mean_only.mean().to_bits());
        }
        assert!(full.variance(0).is_nan());

        let mut positive = NumericAccumulator::default();
        let mut history = Vec::new();
        advance(&mut positive, &mut history, 2, Some(1.0e200));
        advance(&mut positive, &mut history, 2, Some(1.0e200));
        assert_eq!(positive.variance(0).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn family_rebuild_uses_the_newest_shift_and_constant_windows_read_zero() {
        let mut dispersion = DispersionFamily::default();
        dispersion.rebuild([3.0, 5.0, 5.0].into_iter());
        assert_eq!(dispersion.shift(), Some(5.0));
        assert_eq!(dispersion.moment(2).to_bits(), 2.0_f64.to_bits());
        dispersion.rebuild([7.0, 7.0, 7.0].into_iter());
        assert_eq!(dispersion.moment(3).to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn pair_readouts_preserve_nan_and_constant_null_classification() {
        let mut pair = PairFamily::default();
        pair.rebuild([(1.0, 2.0), (1.0, 4.0)].into_iter());
        assert!(pair.correlation(2).is_none());
        pair.rebuild([(1.0e200, 1.0e200), (-1.0e200, -1.0e200)].into_iter());
        assert!(pair.covariance(2, 0).is_nan());
        assert!(pair.correlation(2).unwrap().is_nan());
    }

    #[test]
    fn sum_family_pairs_entering_and_leaving_operands() {
        let mut paired = SumFamily::default();
        paired.add(1.0e16);
        paired.add(1.0);
        paired.add(-1.0e16);
        let before = paired.read();
        paired.pair(1.0, 1.0);
        assert_eq!(paired.read().to_bits(), before.to_bits());
    }
}
