//! Constant summaries for unbounded entity prefixes. Floating state is encoded
//! as IEEE bits so checkpointing preserves infinities, NaNs and signed zero.

use datafusion::{arrow::datatypes::DataType, common::ScalarValue};
use serde::{Deserialize, Serialize};

use super::{
    CompiledAggregate, Statistic, SumClass, compare_samples, float_sample, internal_error,
    operator_error, signed_sample, state_format, typed_null, unsigned_sample,
};
use crate::{CalcFlowError, Result};

fn inconsistent_summary() -> CalcFlowError {
    state_format("cumulative summary is inconsistent with its group".to_owned())
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", content = "value", deny_unknown_fields)]
enum NumericValue {
    Signed(i64),
    SignedTotal([u8; 16]),
    Unsigned(u64),
    UnsignedTotal([u8; 16]),
    Float(u64),
}

impl NumericValue {
    fn from_scalar(value: &ScalarValue) -> Self {
        match SumClass::from_input(&value.data_type()) {
            SumClass::Signed => Self::Signed(signed_sample(value)),
            SumClass::Unsigned => Self::Unsigned(unsigned_sample(value)),
            _ => Self::Float(float_sample(value).to_bits()),
        }
    }

    fn add(previous: Option<&Self>, next: Self, node: &str) -> Result<Self> {
        Ok(match (previous, next) {
            (previous, NumericValue::Signed(next)) => {
                let total = match previous {
                    Some(NumericValue::SignedTotal(bytes)) => i128::from_le_bytes(*bytes),
                    None => 0,
                    _ => return Err(internal_error("cumulative sum type changed")),
                };
                NumericValue::SignedTotal(
                    total
                        .checked_add(i128::from(next))
                        .ok_or_else(|| operator_error(node, "cumulative integer sum overflowed"))?
                        .to_le_bytes(),
                )
            }
            (previous, NumericValue::Unsigned(next)) => {
                let total = match previous {
                    Some(NumericValue::UnsignedTotal(bytes)) => u128::from_le_bytes(*bytes),
                    None => 0,
                    _ => return Err(internal_error("cumulative sum type changed")),
                };
                NumericValue::UnsignedTotal(
                    total
                        .checked_add(u128::from(next))
                        .ok_or_else(|| operator_error(node, "cumulative integer sum overflowed"))?
                        .to_le_bytes(),
                )
            }
            (Some(NumericValue::Float(total)), NumericValue::Float(next)) => {
                NumericValue::Float((f64::from_bits(*total) + f64::from_bits(next)).to_bits())
            }
            (None, NumericValue::Float(next)) => {
                NumericValue::Float((0.0 + f64::from_bits(next)).to_bits())
            }
            (None, next) => next,
            _ => return Err(internal_error("cumulative sum type changed")),
        })
    }

    fn scalar(&self, node: &str) -> Result<ScalarValue> {
        Ok(match self {
            Self::Signed(value) => ScalarValue::Int64(Some(*value)),
            Self::Unsigned(value) => ScalarValue::UInt64(Some(*value)),
            Self::Float(bits) => ScalarValue::Float64(Some(f64::from_bits(*bits))),
            Self::SignedTotal(bytes) => ScalarValue::Int64(Some(
                i64::try_from(i128::from_le_bytes(*bytes))
                    .map_err(|_| operator_error(node, "cumulative integer sum overflowed"))?,
            )),
            Self::UnsignedTotal(bytes) => ScalarValue::UInt64(Some(
                u64::try_from(u128::from_le_bytes(*bytes))
                    .map_err(|_| operator_error(node, "cumulative integer sum overflowed"))?,
            )),
        })
    }

    fn matches_input_class(&self, input_type: &DataType) -> bool {
        matches!(
            (SumClass::from_input(input_type), self),
            (SumClass::Signed, Self::Signed(_) | Self::SignedTotal(_))
                | (
                    SumClass::Unsigned,
                    Self::Unsigned(_) | Self::UnsignedTotal(_)
                )
                | (SumClass::Float, Self::Float(_))
        )
    }

    fn validate_group(&self, statistic: Statistic, input_type: &DataType) -> Result<()> {
        let total = matches!(self, Self::SignedTotal(_) | Self::UnsignedTotal(_));
        if !self.matches_input_class(input_type)
            || ((statistic == Statistic::Sum
                && !matches!(SumClass::from_input(input_type), SumClass::Float))
                != total)
        {
            return Err(inconsistent_summary());
        }
        if matches!(statistic, Statistic::Min | Statistic::Max) {
            self.validate_extremum(input_type)?;
        }
        Ok(())
    }

    fn validate_extremum(&self, input_type: &DataType) -> Result<()> {
        let scalar = self
            .scalar("cumulative")
            .map_err(|_| inconsistent_summary())?
            .cast_to(input_type)
            .map_err(|_| inconsistent_summary())?;
        if !super::is_valid_sample(&scalar) {
            return Err(inconsistent_summary());
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct CumulativeAccumulator {
    valid_count: u64,
    value: Option<NumericValue>,
    origin_bits: u64,
    mean_bits: u64,
    m2_bits: u64,
    has_infinity: bool,
}

impl CumulativeAccumulator {
    pub(super) fn add(
        &mut self,
        value: &ScalarValue,
        statistic: Statistic,
        node: &str,
    ) -> Result<()> {
        let count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node, "cumulative valid count overflowed"))?;
        match statistic {
            Statistic::Sum => {
                let next = NumericValue::from_scalar(value);
                self.value = Some(NumericValue::add(self.value.as_ref(), next, node)?);
            }
            Statistic::Min | Statistic::Max => self.add_extremum(value, statistic, node)?,
            Statistic::Variance => self.add_variance(float_sample(value), count),
            _ => return Err(internal_error("unsupported cumulative statistic")),
        }
        self.valid_count = count;
        Ok(())
    }

    fn add_extremum(
        &mut self,
        value: &ScalarValue,
        statistic: Statistic,
        node: &str,
    ) -> Result<()> {
        let replace = if let Some(previous) = &self.value {
            let previous = previous
                .scalar(node)?
                .cast_to(&value.data_type())
                .map_err(|error| {
                    internal_error(format!("cumulative extremum type mismatch: {error}"))
                })?;
            let order = compare_samples(value, &previous);
            if statistic == Statistic::Min {
                order.is_lt()
            } else {
                order.is_gt()
            }
        } else {
            true
        };
        if replace {
            self.value = Some(NumericValue::from_scalar(value));
        }
        Ok(())
    }

    fn add_variance(&mut self, sample: f64, count: u64) {
        if sample.is_infinite() {
            self.has_infinity = true;
        } else if !self.has_infinity {
            if self.valid_count == 0 {
                self.origin_bits = sample.to_bits();
            }
            let (next_mean, next_m2) = self.next_variance(sample, count);
            // Once M2 overflows, its classification is permanent. Keep
            // a finite mean so the durable summary remains canonical.
            if next_mean.is_finite() {
                self.mean_bits = next_mean.to_bits();
            }
            self.m2_bits = next_m2.to_bits();
        }
    }

    #[allow(clippy::cast_precision_loss, reason = "variance is a Float64 readout")]
    fn next_variance(&self, sample: f64, count: u64) -> (f64, f64) {
        let shifted = sample - f64::from_bits(self.origin_bits);
        let mean = f64::from_bits(self.mean_bits);
        let delta = shifted - mean;
        let next_mean = if self.valid_count == 0 {
            0.0
        } else if delta.is_finite() {
            mean + delta / count as f64
        } else {
            let weight = 1.0 / count as f64;
            mean * (1.0 - weight) + shifted * weight
        };
        let next_m2 = if self.valid_count == 0 {
            0.0
        } else if delta.is_infinite() {
            f64::INFINITY
        } else {
            f64::from_bits(self.m2_bits) + delta * (shifted - next_mean)
        };
        (next_mean, next_m2)
    }

    #[allow(clippy::cast_precision_loss, reason = "variance is a Float64 readout")]
    pub(super) fn read(
        &self,
        output: &CompiledAggregate,
        data_type: &DataType,
        node: &str,
    ) -> Result<ScalarValue> {
        if self.valid_count < output.min_periods {
            return Ok(typed_null(data_type));
        }
        if output.statistic == Statistic::Variance {
            if self.valid_count <= u64::from(output.ddof) {
                return Ok(ScalarValue::Float64(None));
            }
            let variance = if self.has_infinity {
                f64::NAN
            } else {
                f64::from_bits(self.m2_bits).max(0.0)
                    / (self.valid_count - u64::from(output.ddof)) as f64
            };
            return Ok(ScalarValue::Float64(Some(variance)));
        }
        self.value
            .as_ref()
            .ok_or_else(|| internal_error("cumulative value is missing"))?
            .scalar(node)?
            .cast_to(data_type)
            .map_err(|error| internal_error(format!("cumulative value type mismatch: {error}")))
    }

    pub(super) fn encode(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|error| state_format(error.to_string()))
    }

    pub(super) fn decode(
        bytes: &[u8],
        statistic: Statistic,
        input_type: &DataType,
    ) -> Result<Self> {
        let state: Self = serde_json::from_slice(bytes)
            .map_err(|error| state_format(format!("invalid cumulative summary: {error}")))?;
        if state.valid_count == 0 {
            return if state == Self::default() {
                Ok(state)
            } else {
                Err(inconsistent_summary())
            };
        }
        if statistic == Statistic::Variance {
            state.validate_variance()?;
        } else {
            state.validate_value(statistic, input_type)?;
        }
        Ok(state)
    }

    fn validate_variance(&self) -> Result<()> {
        if self.value.is_some()
            || !f64::from_bits(self.origin_bits).is_finite()
            || !f64::from_bits(self.mean_bits).is_finite()
            || f64::from_bits(self.m2_bits).is_nan()
            || f64::from_bits(self.m2_bits) < 0.0
        {
            return Err(inconsistent_summary());
        }
        Ok(())
    }

    fn validate_value(&self, statistic: Statistic, input_type: &DataType) -> Result<()> {
        if self.origin_bits != 0 || self.mean_bits != 0 || self.m2_bits != 0 || self.has_infinity {
            return Err(inconsistent_summary());
        }
        let value = self.value.as_ref().ok_or_else(inconsistent_summary)?;
        value.validate_group(statistic, input_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readout(statistic: Statistic, ddof: u8) -> CompiledAggregate {
        CompiledAggregate {
            group: 0,
            statistic,
            min_periods: 1,
            ddof,
        }
    }

    #[test]
    fn test_cumulative_integer_sum_is_exact_and_overflow_is_loud() {
        for (first, last, dtype) in [
            (
                ScalarValue::Int64(Some(i64::MAX - 1)),
                ScalarValue::Int64(Some(1)),
                DataType::Int64,
            ),
            (
                ScalarValue::UInt64(Some(u64::MAX - 1)),
                ScalarValue::UInt64(Some(1)),
                DataType::UInt64,
            ),
        ] {
            let mut state = CumulativeAccumulator::default();
            state.add(&first, Statistic::Sum, "test").unwrap();
            state.add(&last, Statistic::Sum, "test").unwrap();
            let restored =
                CumulativeAccumulator::decode(&state.encode().unwrap(), Statistic::Sum, &dtype)
                    .unwrap();
            assert_eq!(state, restored);
            assert_eq!(
                state
                    .read(&readout(Statistic::Sum, 0), &dtype, "test")
                    .unwrap(),
                match dtype {
                    DataType::Int64 => ScalarValue::Int64(Some(i64::MAX)),
                    _ => ScalarValue::UInt64(Some(u64::MAX)),
                }
            );
            state.add(&last, Statistic::Sum, "test").unwrap();
            assert!(
                state
                    .read(&readout(Statistic::Sum, 0), &dtype, "test")
                    .is_err()
            );
        }
    }

    #[test]
    fn test_cumulative_variance_infinity_and_extreme_finite_values() {
        for values in [
            [1e308, -1e308],
            [f64::INFINITY, 1.0],
            [f64::NEG_INFINITY, f64::INFINITY],
        ] {
            let mut state = CumulativeAccumulator::default();
            state
                .add(
                    &ScalarValue::Float64(Some(values[0])),
                    Statistic::Variance,
                    "test",
                )
                .unwrap();
            assert_eq!(
                state
                    .read(&readout(Statistic::Variance, 1), &DataType::Float64, "test")
                    .unwrap(),
                ScalarValue::Float64(None)
            );
            state
                .add(
                    &ScalarValue::Float64(Some(values[1])),
                    Statistic::Variance,
                    "test",
                )
                .unwrap();
            let restored = CumulativeAccumulator::decode(
                &state.encode().unwrap(),
                Statistic::Variance,
                &DataType::Float64,
            )
            .unwrap();
            assert_eq!(state, restored);
            let ScalarValue::Float64(Some(value)) = state
                .read(&readout(Statistic::Variance, 1), &DataType::Float64, "test")
                .unwrap()
            else {
                panic!("non-null variance");
            };
            if values[0].is_infinite() {
                assert!(value.is_nan());
            } else {
                assert_eq!(value, f64::INFINITY);
            }
        }
    }

    #[test]
    fn test_cumulative_ieee_values_roundtrip_exactly() {
        for statistic in [Statistic::Sum, Statistic::Min, Statistic::Max] {
            let mut state = CumulativeAccumulator::default();
            for sample in [0.0, -0.0] {
                state
                    .add(&ScalarValue::Float64(Some(sample)), statistic, "test")
                    .unwrap();
            }
            let restored = CumulativeAccumulator::decode(
                &state.encode().unwrap(),
                statistic,
                &DataType::Float64,
            )
            .unwrap();
            assert_eq!(state, restored);
            let ScalarValue::Float64(Some(value)) = restored
                .read(&readout(statistic, 0), &DataType::Float64, "test")
                .unwrap()
            else {
                panic!("non-null float");
            };
            assert_eq!(
                value.to_bits(),
                if statistic == Statistic::Min {
                    (-0.0_f64).to_bits()
                } else {
                    0
                }
            );
            for sample in [f64::INFINITY, f64::NEG_INFINITY] {
                state
                    .add(&ScalarValue::Float64(Some(sample)), statistic, "test")
                    .unwrap();
            }
            let restored = CumulativeAccumulator::decode(
                &state.encode().unwrap(),
                statistic,
                &DataType::Float64,
            )
            .unwrap();
            assert_eq!(state, restored);
        }
    }

    #[test]
    fn test_cumulative_summaries_reject_missing_and_inconsistent_payloads() {
        let empty = CumulativeAccumulator::default().encode().unwrap();
        for statistic in [
            Statistic::Sum,
            Statistic::Min,
            Statistic::Max,
            Statistic::Variance,
        ] {
            assert!(CumulativeAccumulator::decode(&empty, statistic, &DataType::Float64).is_ok());
        }
        let mut value: serde_json::Value = serde_json::from_slice(&empty).unwrap();
        value["valid_count"] = serde_json::json!(1);
        assert!(
            CumulativeAccumulator::decode(
                &serde_json::to_vec(&value).unwrap(),
                Statistic::Sum,
                &DataType::Float64
            )
            .is_err()
        );
        value["m2_bits"] = serde_json::json!((-1.0_f64).to_bits());
        assert!(
            CumulativeAccumulator::decode(
                &serde_json::to_vec(&value).unwrap(),
                Statistic::Variance,
                &DataType::Float64
            )
            .is_err()
        );
    }
}
