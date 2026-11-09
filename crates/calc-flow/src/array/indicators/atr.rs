use super::{finite, next_mean, nonnegative, positions, sample};
use crate::Result;

#[derive(Clone, Copy, Default)]
pub(crate) struct AtrKernel {
    pub(crate) positions: u64,
    pub(crate) has_previous: bool,
    pub(crate) seed_count: u64,
    pub(crate) previous_close: f64,
    pub(crate) mean_tr: f64,
}

pub(crate) fn samples(values: [Option<f64>; 3], fields: [&str; 3]) -> Result<Option<[f64; 4]>> {
    let high = sample(values[0], fields[0])?;
    let low = sample(values[1], fields[1])?;
    let close = sample(values[2], fields[2])?;
    let range = domain(high, low, fields)?;
    Ok(match (high, low, close, range) {
        (Some(high), Some(low), Some(close), Some(range)) => Some([high, low, close, range]),
        _ => None,
    })
}

fn domain(high: Option<f64>, low: Option<f64>, fields: [&str; 3]) -> Result<Option<f64>> {
    let (Some(high), Some(low)) = (high, low) else {
        return Ok(None);
    };
    if high < low {
        return Err(super::super::invalid(
            fields[0],
            format!("must be greater than or equal to {}", fields[1]),
        ));
    }
    finite(high - low, fields[0], "ATR high - low").map(Some)
}

impl AtrKernel {
    pub(crate) fn push(
        &mut self,
        values: [Option<f64>; 3],
        period: u64,
        fields: [&str; 3],
    ) -> Result<Option<f64>> {
        let positions = positions(self.positions)?;
        let Some([high, low, close, range]) = super::validate_atr_samples(values, fields)? else {
            *self = Self {
                positions,
                ..Self::default()
            };
            return Ok(None);
        };
        let output = self.advance(high, low, range, period, fields)?;
        self.previous_close = close;
        self.has_previous = true;
        self.positions = positions;
        Ok(output)
    }

    fn advance(
        &mut self,
        high: f64,
        low: f64,
        range: f64,
        period: u64,
        fields: [&str; 3],
    ) -> Result<Option<f64>> {
        if !self.has_previous {
            return Ok(None);
        }
        let high_gap = finite(
            high - self.previous_close,
            fields[0],
            "ATR high - previous_close",
        )?;
        let low_gap = finite(
            low - self.previous_close,
            fields[1],
            "ATR low - previous_close",
        )?;
        let tr = nonnegative(range.max(high_gap.abs()).max(low_gap.abs()));
        let count = if self.seed_count < period {
            self.seed_count + 1
        } else {
            period
        };
        self.mean_tr = next_mean(self.mean_tr, tr, count, fields[0])?;
        self.seed_count = count;
        Ok((count == period).then_some(self.mean_tr))
    }
}

impl AtrKernel {
    pub(crate) fn validate(&self, period: u64) -> Result<()> {
        super::validate_counts(self.positions, self.has_previous, self.seed_count, period)?;
        finite(
            self.previous_close,
            "state.previous_close",
            "ATR previous close",
        )?;
        super::validate_mean(self.mean_tr)?;
        self.validate_inactive()
    }

    fn validate_inactive(&self) -> Result<()> {
        if !self.has_previous {
            super::validate_zeros(&[self.previous_close])?;
        }
        if self.seed_count == 0 {
            super::validate_zeros(&[self.mean_tr])?;
        }
        Ok(())
    }
}
