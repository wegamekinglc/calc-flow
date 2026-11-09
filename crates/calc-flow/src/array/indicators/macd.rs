use crate::{
    Result,
    array::{invalid, rolling::next_ewma},
};

/// Unadjusted MACD spans and independent fast/slow and signal warm-up counts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[must_use]
pub struct MacdOptions {
    fast_span: u64,
    slow_span: u64,
    signal_span: u64,
    min_periods: u64,
    signal_min_periods: u64,
}

impl Default for MacdOptions {
    fn default() -> Self {
        Self {
            fast_span: 12,
            slow_span: 26,
            signal_span: 9,
            min_periods: 1,
            signal_min_periods: 1,
        }
    }
}
impl MacdOptions {
    /// Creates positive spans with `fast_span < slow_span`, and both minima one.
    ///
    /// # Errors
    /// Returns `InvalidArgument` for zero spans or reversed/equal fast/slow spans.
    pub fn new(fast_span: u64, slow_span: u64, signal_span: u64) -> Result<Self> {
        positive(fast_span, "options.fast_span")?;
        positive(slow_span, "options.slow_span")?;
        positive(signal_span, "options.signal_span")?;
        if fast_span >= slow_span {
            return Err(invalid("options.fast_span", "must be less than slow_span"));
        }
        Ok(Self {
            fast_span,
            slow_span,
            signal_span,
            min_periods: 1,
            signal_min_periods: 1,
        })
    }
    /// Sets the positive warm-up count for both price recurrences, without a span cap.
    ///
    /// # Errors
    /// Returns `InvalidArgument` when `min_periods` is zero.
    pub fn with_min_periods(self, min_periods: u64) -> Result<Self> {
        positive(min_periods, "options.min_periods")?;
        Ok(Self {
            min_periods,
            ..self
        })
    }
    /// Sets the positive signal warm-up count, without a span or other minimum cap.
    ///
    /// # Errors
    /// Returns `InvalidArgument` when `signal_min_periods` is zero.
    pub fn with_signal_min_periods(self, signal_min_periods: u64) -> Result<Self> {
        positive(signal_min_periods, "options.signal_min_periods")?;
        Ok(Self {
            signal_min_periods,
            ..self
        })
    }
    /// Returns the fast exponential span.
    pub const fn fast_span(&self) -> u64 {
        self.fast_span
    }
    /// Returns the slow exponential span.
    pub const fn slow_span(&self) -> u64 {
        self.slow_span
    }
    /// Returns the signal exponential span.
    pub const fn signal_span(&self) -> u64 {
        self.signal_span
    }
    /// Returns the valid-price count needed by both fast/slow recurrences.
    pub const fn min_periods(&self) -> u64 {
        self.min_periods
    }
    /// Returns the valid-line count needed by the signal recurrence.
    pub const fn signal_min_periods(&self) -> u64 {
        self.signal_min_periods
    }
}
fn positive(value: u64, field: &str) -> Result<()> {
    if value == 0 {
        return Err(invalid(field, "must be a positive u64"));
    }
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct MacdParameters {
    options: MacdOptions,
    alpha: [f64; 3],
}
impl MacdParameters {
    pub(crate) fn new(options: MacdOptions) -> Self {
        Self {
            options,
            alpha: [
                alpha(options.fast_span),
                alpha(options.slow_span),
                alpha(options.signal_span),
            ],
        }
    }
}
#[allow(
    clippy::cast_precision_loss,
    reason = "preserves the existing EWMA Float64 span conversion"
)]
fn alpha(span: u64) -> f64 {
    2.0 / (span as f64 + 1.0)
}

#[derive(Clone, Copy, Default)]
pub(crate) struct MacdKernel {
    pub(crate) positions: u64,
    pub(crate) fast_count: u64,
    pub(crate) fast_value: f64,
    pub(crate) slow_count: u64,
    pub(crate) slow_value: f64,
    pub(crate) signal_count: u64,
    pub(crate) signal_value: f64,
}
impl MacdKernel {
    pub(crate) fn push(
        &mut self,
        price: Option<f64>,
        parameters: &MacdParameters,
    ) -> Result<[Option<f64>; 3]> {
        let positions = super::positions(self.positions)?;
        let price = price.filter(|value| !value.is_nan());
        let fast = transition(
            (self.fast_count, self.fast_value),
            price,
            parameters.alpha[0],
        )?;
        let slow = transition(
            (self.slow_count, self.slow_value),
            price,
            parameters.alpha[1],
        )?;
        let line = readout(fast, parameters.options.min_periods)
            .zip(readout(slow, parameters.options.min_periods))
            .map(|(fast, slow)| fast - slow);
        let next_signal = transition(
            (self.signal_count, self.signal_value),
            line.filter(|value| !value.is_nan()),
            parameters.alpha[2],
        )?;
        let signal = readout(next_signal, parameters.options.signal_min_periods);
        let histogram = line.zip(signal).map(|(line, signal)| line - signal);
        *self = Self {
            positions,
            fast_count: fast.0,
            fast_value: fast.1,
            slow_count: slow.0,
            slow_value: slow.1,
            signal_count: next_signal.0,
            signal_value: next_signal.1,
        };
        Ok([line, signal, histogram])
    }
    pub(crate) fn validate(&self, options: MacdOptions) -> Result<()> {
        self.validate_counts(options.min_periods)?;
        zero(self.fast_count, self.fast_value, "state.fast_value")?;
        zero(self.slow_count, self.slow_value, "state.slow_value")?;
        zero(self.signal_count, self.signal_value, "state.signal_value")
    }
    fn validate_counts(&self, minimum: u64) -> Result<()> {
        if self.fast_count != self.slow_count || self.fast_count > self.positions {
            return Err(invalid(
                "state.fast_count",
                "fast/slow counts must agree and not exceed positions",
            ));
        }
        if self.signal_count > self.positions {
            return Err(invalid("state.signal_count", "must not exceed positions"));
        }
        self.validate_signal_count(minimum)
    }
    fn validate_signal_count(&self, minimum: u64) -> Result<()> {
        if self.fast_count < minimum && self.signal_count != 0 {
            return Err(invalid(
                "state.signal_count",
                "must be zero before price warm-up",
            ));
        }
        if self.signal_count > self.positions.saturating_sub(minimum - 1) {
            return Err(invalid(
                "state.signal_count",
                "exceeds positions eligible for a ready line",
            ));
        }
        Ok(())
    }
}
fn transition(previous: (u64, f64), sample: Option<f64>, alpha: f64) -> Result<(u64, f64)> {
    next_ewma(previous.0, previous.1, sample, alpha, false).ok_or_else(|| {
        invalid(
            "input.rows",
            "MACD valid-sample count would exceed u64::MAX",
        )
    })
}
fn readout(value: (u64, f64), minimum: u64) -> Option<f64> {
    (value.0 >= minimum).then_some(value.1)
}
fn zero(count: u64, value: f64, field: &str) -> Result<()> {
    if count == 0 && value.to_bits() != 0 {
        return Err(invalid(
            field,
            "zero-count MACD values require positive zero bits",
        ));
    }
    Ok(())
}
