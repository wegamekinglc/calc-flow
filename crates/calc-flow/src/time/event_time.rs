use datafusion::arrow::datatypes::{DataType, TimeUnit};
use serde::{Deserialize, Serialize};

use crate::{CalcFlowError, Result};

/// A UTC instant counted in microseconds since the Unix epoch
/// (1970-01-01T00:00:00Z), per spec D1.1.
///
/// The inner value is never exposed as a bare `i64` in serialized form other
/// than the exact microsecond count (D1.2); ordering is total across pre- and
/// post-epoch values.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize, Deserialize)]
pub struct EventTime(i64);

impl EventTime {
    /// Wraps an exact microsecond count.
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Returns the exact microsecond count (D1.2).
    pub const fn as_micros(self) -> i64 {
        self.0
    }

    /// Imports one Arrow timestamp value with a checked conversion (D1.3).
    ///
    /// Finer-than-microsecond input floors toward negative infinity (D1.5);
    /// coarser input multiplies with an overflow check. The column must be an
    /// Arrow timestamp that is timezone-naive (interpreted as UTC) or carries
    /// the explicit timezone `"UTC"` (D1.6).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming `column` when the
    /// data type is not a timestamp, the timezone is neither naive nor
    /// `"UTC"`, or the checked multiplication overflows.
    pub fn import_timestamp(value: i64, data_type: &DataType, column: &str) -> Result<Self> {
        let DataType::Timestamp(unit, timezone) = data_type else {
            return Err(CalcFlowError::InvalidArgument {
                field: column.into(),
                message: format!("expected an Arrow timestamp column, found {data_type}"),
            });
        };
        if let Some(timezone) = timezone
            && timezone.as_ref() != "UTC"
        {
            return Err(CalcFlowError::InvalidArgument {
                field: column.into(),
                message: format!(
                    "event-time columns must be timezone-naive (UTC) or \"UTC\", found {timezone:?}"
                ),
            });
        }
        match unit {
            TimeUnit::Second => value
                .checked_mul(1_000_000)
                .map(Self::from_micros)
                .ok_or_else(|| overflow(column)),
            TimeUnit::Millisecond => value
                .checked_mul(1_000)
                .map(Self::from_micros)
                .ok_or_else(|| overflow(column)),
            TimeUnit::Microsecond => Ok(Self::from_micros(value)),
            TimeUnit::Nanosecond => Ok(Self::from_micros(value.div_euclid(1_000))),
        }
    }

    /// Exports this instant to one Arrow timestamp unit (D1.4).
    ///
    /// Producing nanoseconds is checked; producing any coarser unit floors
    /// toward negative infinity (D1.5).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the microsecond value
    /// overflows the nanosecond range.
    pub fn export_timestamp(self, unit: TimeUnit) -> Result<i64> {
        match unit {
            TimeUnit::Second => Ok(self.0.div_euclid(1_000_000)),
            TimeUnit::Millisecond => Ok(self.0.div_euclid(1_000)),
            TimeUnit::Microsecond => Ok(self.0),
            TimeUnit::Nanosecond => {
                self.0
                    .checked_mul(1_000)
                    .ok_or_else(|| CalcFlowError::InvalidArgument {
                        field: "event_time".into(),
                        message: "microsecond value overflows the nanosecond range".into(),
                    })
            }
        }
    }
}

fn overflow(column: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: column.into(),
        message: "timestamp value overflows the microsecond event-time range".into(),
    }
}
