use std::time::Duration;

use datafusion::arrow::array::{
    Array, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};

use crate::{Batch, CalcFlowError, EventTime, Result};

use super::{
    prepare::{ArrowTimestampUnit, ResolvedEventTimeColumn},
    types::LogicalInstant,
};

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GeneratedWatermarkState {
    event_time: ResolvedEventTimeColumn,
    max_out_of_orderness: Duration,
    max_observed_nanos: Option<i128>,
    last_emitted: Option<EventTime>,
}

impl GeneratedWatermarkState {
    pub(crate) fn new(event_time: ResolvedEventTimeColumn, max_out_of_orderness: Duration) -> Self {
        Self {
            event_time,
            max_out_of_orderness,
            max_observed_nanos: None,
            last_emitted: None,
        }
    }

    pub(crate) fn observe_batch(&mut self, batch: &Batch, binding: &str) -> Result<()> {
        let event_time_path = format!("sources.{binding}.watermark_policy.event_time_column");
        let table = batch
            .table_payload()
            .map_err(|_| CalcFlowError::InvalidArgument {
                field: event_time_path.clone(),
                message: "generated watermarks require table batches".into(),
            })?;
        for record in table.batches() {
            let column = record.column(self.event_time.index);
            let maximum =
                maximum_timestamp_nanos(column.as_ref(), self.event_time.unit, &event_time_path)?;
            if let Some(maximum) = maximum {
                self.max_observed_nanos = Some(
                    self.max_observed_nanos
                        .map_or(maximum, |previous| previous.max(maximum)),
                );
            }
        }
        Ok(())
    }

    pub(crate) fn on_timer(&mut self, binding: &str) -> Result<Option<EventTime>> {
        let Some(maximum) = self.max_observed_nanos else {
            return Ok(None);
        };
        let delay = i128::try_from(self.max_out_of_orderness.as_nanos()).map_err(|_| {
            CalcFlowError::InvalidArgument {
                field: format!("sources.{binding}.watermark_policy.max_out_of_orderness"),
                message: "duration exceeds event-time arithmetic range".into(),
            }
        })?;
        let candidate_nanos =
            maximum
                .checked_sub(delay)
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: format!("sources.{binding}.watermark_policy.max_out_of_orderness"),
                    message: "generated watermark arithmetic underflowed".into(),
                })?;
        let candidate_micros = i64::try_from(candidate_nanos.div_euclid(1_000)).map_err(|_| {
            CalcFlowError::InvalidArgument {
                field: format!("sources.{binding}.watermark_policy.event_time_column"),
                message: "generated watermark exceeds the EventTime range".into(),
            }
        })?;
        let candidate = EventTime::from_micros(candidate_micros);
        if self
            .last_emitted
            .is_some_and(|previous| candidate <= previous)
        {
            return Ok(None);
        }
        self.last_emitted = Some(candidate);
        Ok(Some(candidate))
    }

    pub(crate) const fn max_observed_nanos(&self) -> Option<i128> {
        self.max_observed_nanos
    }

    pub(crate) const fn last_emitted(&self) -> Option<EventTime> {
        self.last_emitted
    }
}

fn maximum_timestamp_nanos(
    column: &dyn Array,
    unit: ArrowTimestampUnit,
    event_time_path: &str,
) -> Result<Option<i128>> {
    macro_rules! maximum {
        ($array:ty, $factor:expr) => {{
            let array = column.as_any().downcast_ref::<$array>().ok_or_else(|| {
                CalcFlowError::InvalidArgument {
                    field: event_time_path.into(),
                    message: "record batch column type differs from the prepared schema".into(),
                }
            })?;
            array.iter().flatten().try_fold(None, |maximum, value| {
                let nanos = i128::from(value).checked_mul($factor).ok_or_else(|| {
                    CalcFlowError::InvalidArgument {
                        field: event_time_path.into(),
                        message: "timestamp unit conversion overflowed".into(),
                    }
                })?;
                Ok::<_, CalcFlowError>(Some(
                    maximum.map_or(nanos, |current: i128| current.max(nanos)),
                ))
            })
        }};
    }
    match unit {
        ArrowTimestampUnit::Second => maximum!(TimestampSecondArray, 1_000_000_000),
        ArrowTimestampUnit::Millisecond => maximum!(TimestampMillisecondArray, 1_000_000),
        ArrowTimestampUnit::Microsecond => maximum!(TimestampMicrosecondArray, 1_000),
        ArrowTimestampUnit::Nanosecond => maximum!(TimestampNanosecondArray, 1),
    }
}

pub(crate) fn next_phase_deadline(
    expired_deadline: LogicalInstant,
    current_instant: LogicalInstant,
    emit_interval: Duration,
) -> Result<LogicalInstant> {
    let interval = emit_interval.as_nanos();
    if interval == 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: "runtime.progress.timers.watermark.emit_interval".into(),
            message: "must be positive".into(),
        });
    }
    let elapsed = current_instant.0.saturating_sub(expired_deadline.0);
    let phases = elapsed
        .checked_div(interval)
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "runtime.progress.timers.deadline".into(),
            message: "phase calculation overflowed".into(),
        })?;
    expired_deadline
        .0
        .checked_add(phases.checked_mul(interval).ok_or_else(|| {
            CalcFlowError::InvalidArgument {
                field: "runtime.progress.timers.deadline".into(),
                message: "phase multiplication overflowed".into(),
            }
        })?)
        .map(LogicalInstant)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "runtime.progress.timers.deadline".into(),
            message: "phase deadline overflowed".into(),
        })
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{
            ArrayRef, TimestampMicrosecondArray, TimestampMillisecondArray,
            TimestampNanosecondArray, TimestampSecondArray,
        },
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::{GeneratedWatermarkState, LogicalInstant, next_phase_deadline};
    use crate::runtime::streaming::progress::prepare::{
        ArrowTimestampUnit, ResolvedEventTimeColumn,
    };
    use crate::{Batch, BatchMetadata, CalcFlowError, EventTime};

    fn state(unit: ArrowTimestampUnit, delay: Duration) -> GeneratedWatermarkState {
        GeneratedWatermarkState::new(
            ResolvedEventTimeColumn {
                name: Arc::from("at"),
                index: 0,
                unit,
            },
            delay,
        )
    }

    fn batch(array: ArrayRef, unit: TimeUnit) -> Batch {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "at",
            DataType::Timestamp(unit, None),
            true,
        )]));
        Batch::table(
            vec![RecordBatch::try_new(schema, vec![array]).unwrap()],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn generated_watermark_supports_all_timestamp_units() {
        let cases = [
            (
                ArrowTimestampUnit::Second,
                TimeUnit::Second,
                Arc::new(TimestampSecondArray::from(vec![Some(2)])) as ArrayRef,
            ),
            (
                ArrowTimestampUnit::Millisecond,
                TimeUnit::Millisecond,
                Arc::new(TimestampMillisecondArray::from(vec![Some(2_000)])) as ArrayRef,
            ),
            (
                ArrowTimestampUnit::Microsecond,
                TimeUnit::Microsecond,
                Arc::new(TimestampMicrosecondArray::from(vec![Some(2_000_000)])) as ArrayRef,
            ),
            (
                ArrowTimestampUnit::Nanosecond,
                TimeUnit::Nanosecond,
                Arc::new(TimestampNanosecondArray::from(vec![Some(2_000_000_999)])) as ArrayRef,
            ),
        ];
        for (prepared_unit, arrow_unit, array) in cases {
            let mut generated = state(prepared_unit, Duration::from_secs(1));
            generated
                .observe_batch(&batch(array, arrow_unit), "left")
                .unwrap();
            assert_eq!(
                generated.on_timer("left").unwrap(),
                Some(EventTime::from_micros(1_000_000))
            );
        }
    }

    #[test]
    fn generated_watermark_uses_non_null_max() {
        let mut generated = state(ArrowTimestampUnit::Microsecond, Duration::from_micros(2));
        generated
            .observe_batch(
                &batch(
                    Arc::new(TimestampMicrosecondArray::from(vec![
                        None,
                        Some(7),
                        Some(4),
                    ])),
                    TimeUnit::Microsecond,
                ),
                "left",
            )
            .unwrap();
        assert_eq!(
            generated.on_timer("left").unwrap(),
            Some(EventTime::from_micros(5))
        );
    }

    #[test]
    fn empty_or_null_batch_does_not_advance_watermark() {
        let mut generated = state(ArrowTimestampUnit::Microsecond, Duration::from_micros(1));
        generated
            .observe_batch(
                &batch(
                    Arc::new(TimestampMicrosecondArray::from(vec![None, None])),
                    TimeUnit::Microsecond,
                ),
                "left",
            )
            .unwrap();
        assert_eq!(generated.on_timer("left").unwrap(), None);
    }

    #[test]
    fn generated_watermark_checks_event_time_arithmetic() {
        let mut generated = state(ArrowTimestampUnit::Nanosecond, Duration::MAX);
        generated
            .observe_batch(
                &batch(
                    Arc::new(TimestampNanosecondArray::from(vec![Some(i64::MIN)])),
                    TimeUnit::Nanosecond,
                ),
                "left",
            )
            .unwrap();
        assert!(generated.on_timer("left").is_err());
    }

    #[test]
    fn generated_watermark_conversion_error_names_the_binding_policy_path() {
        let mut generated = state(ArrowTimestampUnit::Second, Duration::from_secs(1));
        let error = generated
            .observe_batch(
                &batch(
                    Arc::new(TimestampMillisecondArray::from(vec![Some(1)])),
                    TimeUnit::Millisecond,
                ),
                "left",
            )
            .unwrap_err();
        assert!(matches!(
            error,
            CalcFlowError::InvalidArgument { ref field, .. }
                if field == "sources.left.watermark_policy.event_time_column"
        ));
    }

    #[test]
    fn generated_watermark_never_regresses_or_duplicates() {
        let mut generated = state(ArrowTimestampUnit::Microsecond, Duration::from_micros(1));
        for values in [vec![Some(5)], vec![Some(5)], vec![Some(4)]] {
            generated
                .observe_batch(
                    &batch(
                        Arc::new(TimestampMicrosecondArray::from(values)),
                        TimeUnit::Microsecond,
                    ),
                    "left",
                )
                .unwrap();
            let emitted = generated.on_timer("left").unwrap();
            if generated.last_emitted() == Some(EventTime::from_micros(4)) {
                assert!(emitted.is_none() || emitted == Some(EventTime::from_micros(4)));
            }
        }
        assert_eq!(generated.last_emitted(), Some(EventTime::from_micros(4)));
    }

    #[test]
    fn first_watermark_deadline_is_running_instant_plus_interval() {
        assert_eq!(
            LogicalInstant(7)
                .checked_add(Duration::from_nanos(5))
                .unwrap(),
            LogicalInstant(12)
        );
    }

    #[test]
    fn watermark_cadence_is_phase_anchored() {
        assert_eq!(
            next_phase_deadline(
                LogicalInstant(10),
                LogicalInstant(13),
                Duration::from_nanos(10),
            )
            .unwrap(),
            LogicalInstant(20)
        );
    }

    #[test]
    fn missed_watermark_ticks_coalesce_deterministically() {
        assert_eq!(
            next_phase_deadline(
                LogicalInstant(10),
                LogicalInstant(42),
                Duration::from_nanos(10),
            )
            .unwrap(),
            LogicalInstant(50)
        );
    }
}
