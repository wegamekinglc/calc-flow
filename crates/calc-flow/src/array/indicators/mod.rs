//! Shared finite-only indicator transitions; see the native array guide.

mod atr;
mod rsi;

pub(crate) use atr::{AtrKernel, samples as validate_atr_samples};
pub(crate) use rsi::{RsiKernel, checked_sample as validate_rsi_sample};

use crate::Result;

/// Required warm-up period for the native relative strength index.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RsiOptions {
    period: u64,
}

impl RsiOptions {
    /// Creates options with a positive number of price differences.
    ///
    /// # Errors
    /// Returns `InvalidArgument` when `period` is zero.
    pub fn new(period: u64) -> Result<Self> {
        positive(period)?;
        Ok(Self { period })
    }

    /// Returns the seed and Wilder recurrence period.
    pub const fn period(&self) -> u64 {
        self.period
    }
}

/// Required warm-up period for the native average true range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AtrOptions {
    period: u64,
}

impl AtrOptions {
    /// Creates options with a positive number of true-range samples.
    ///
    /// # Errors
    /// Returns `InvalidArgument` when `period` is zero.
    pub fn new(period: u64) -> Result<Self> {
        positive(period)?;
        Ok(Self { period })
    }

    /// Returns the seed and Wilder recurrence period.
    pub const fn period(&self) -> u64 {
        self.period
    }
}

fn positive(period: u64) -> Result<()> {
    if period == 0 {
        return Err(super::invalid("options.period", "must be a positive u64"));
    }
    Ok(())
}

fn finite(value: f64, field: &str, operation: &str) -> Result<f64> {
    if !value.is_finite() {
        return Err(super::invalid(
            field,
            format!("{operation} is not representable as finite float64"),
        ));
    }
    Ok(value)
}

fn sample(value: Option<f64>, field: &str) -> Result<Option<f64>> {
    match value {
        Some(value) if value.is_infinite() => Err(super::invalid(
            field,
            "RSI/ATR requires finite non-missing samples",
        )),
        Some(value) if !value.is_nan() => Ok(Some(value)),
        _ => Ok(None),
    }
}

fn nonnegative(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the numerical contract specifies rounded u64-to-f64 denominators"
)]
fn next_mean(previous: f64, sample: f64, count: u64, field: &str) -> Result<f64> {
    if count == 1 {
        return Ok(nonnegative(sample));
    }
    let difference = finite(
        sample - previous,
        field,
        "RSI/ATR seed/recurrence difference",
    )?;
    let increment = finite(
        difference / count as f64,
        field,
        "RSI/ATR seed/recurrence increment",
    )?;
    finite(previous + increment, field, "RSI/ATR seed/recurrence mean").map(nonnegative)
}

fn positions(previous: u64) -> Result<u64> {
    previous.checked_add(1).ok_or_else(|| {
        super::invalid(
            "input.rows",
            "processed position count would exceed u64::MAX",
        )
    })
}

#[cfg(test)]
mod tests;

fn validate_counts(positions: u64, previous: bool, count: u64, period: u64) -> Result<()> {
    positive(period)?;
    if count > period || count > positions.saturating_sub(1) {
        return Err(super::invalid(
            "state.seed_count",
            "exceeds period or processed positions",
        ));
    }
    validate_presence(positions, previous, count)
}

fn validate_presence(positions: u64, previous: bool, count: u64) -> Result<()> {
    if previous && positions == 0 {
        return Err(super::invalid(
            "state.has_previous",
            "requires a processed position",
        ));
    }
    if !previous && count != 0 {
        return Err(super::invalid(
            "state.seed_count",
            "absent previous requires zero seed count",
        ));
    }
    Ok(())
}

fn validate_mean(value: f64) -> Result<()> {
    if !value.is_finite() || value < 0.0 {
        return Err(super::invalid(
            "state.mean",
            "must be finite and nonnegative",
        ));
    }
    Ok(())
}

fn validate_zeros(values: &[f64]) -> Result<()> {
    if values.iter().any(|value| value.to_bits() != 0) {
        return Err(super::invalid(
            "state.reset",
            "inactive fields must have positive zero bits",
        ));
    }
    Ok(())
}
