use super::{finite, next_mean, positions, sample};
use crate::Result;

#[derive(Clone, Copy, Default)]
pub(crate) struct RsiKernel {
    pub(crate) positions: u64,
    pub(crate) has_previous: bool,
    pub(crate) seed_count: u64,
    pub(crate) previous_price: f64,
    pub(crate) mean_gain: f64,
    pub(crate) mean_loss: f64,
}

pub(crate) fn checked_sample(value: Option<f64>, field: &str) -> Result<Option<f64>> {
    sample(value, field)
}

impl RsiKernel {
    pub(crate) fn push(
        &mut self,
        value: Option<f64>,
        period: u64,
        field: &str,
    ) -> Result<Option<f64>> {
        let positions = positions(self.positions)?;
        let value = super::validate_rsi_sample(value, field)?;
        let Some(price) = value else {
            *self = Self {
                positions,
                ..Self::default()
            };
            return Ok(None);
        };
        let output = self.advance(price, period, field)?;
        self.previous_price = price;
        self.has_previous = true;
        self.positions = positions;
        Ok(output)
    }

    fn advance(&mut self, price: f64, period: u64, field: &str) -> Result<Option<f64>> {
        if !self.has_previous {
            return Ok(None);
        }
        let delta = finite(
            price - self.previous_price,
            field,
            "RSI price difference from previous price",
        )?;
        let count = self.seed_count.saturating_add(1).min(period);
        let gain = if delta > 0.0 { delta } else { 0.0 };
        let loss = if delta < 0.0 { -delta } else { 0.0 };
        let mean_gain = next_mean(self.mean_gain, gain, count, field)?;
        let mean_loss = next_mean(self.mean_loss, loss, count, field)?;
        self.mean_gain = mean_gain;
        self.mean_loss = mean_loss;
        self.seed_count = count;
        Ok((count == period).then(|| strength(mean_gain, mean_loss)))
    }
}

fn strength(gain: f64, loss: f64) -> f64 {
    if gain == 0.0 && loss == 0.0 {
        50.0
    } else if loss == 0.0 {
        100.0
    } else if gain == 0.0 {
        0.0
    } else if gain >= loss {
        let ratio = loss / gain;
        100.0 / (1.0 + ratio)
    } else {
        let ratio = gain / loss;
        100.0 * (ratio / (1.0 + ratio))
    }
}

impl RsiKernel {
    pub(crate) fn validate(&self, period: u64) -> Result<()> {
        super::validate_counts(self.positions, self.has_previous, self.seed_count, period)?;
        finite(
            self.previous_price,
            "state.previous_price",
            "RSI previous price",
        )?;
        super::validate_mean(self.mean_gain)?;
        super::validate_mean(self.mean_loss)?;
        self.validate_inactive()
    }

    fn validate_inactive(&self) -> Result<()> {
        if !self.has_previous {
            super::validate_zeros(&[self.previous_price])?;
        }
        if self.seed_count == 0 {
            super::validate_zeros(&[self.mean_gain, self.mean_loss])?;
        }
        Ok(())
    }
}
