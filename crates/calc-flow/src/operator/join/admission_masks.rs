#[cfg(test)]
use super::note_join_work;
use super::{
    BatchEventTimes, DropKind, IngressProgress, JoinTimeBounds, RowAdmission, SidePlan, columnar,
    counter_overflow, operator_reason,
};
use crate::{EventTime, Result, StreamOperatorContext};
use datafusion::arrow::{
    array::Array, datatypes::TimeUnit, record_batch::RecordBatch,
    util::bit_chunk_iterator::BitChunks,
};

pub(super) struct AdmissionMaskContext<'a> {
    pub(super) record: &'a RecordBatch,
    pub(super) plan: &'a SidePlan,
    pub(super) side_progress: Option<IngressProgress>,
    pub(super) opposite: Option<IngressProgress>,
    pub(super) bounds: JoinTimeBounds,
}

pub(super) struct AdmissionMasks {
    times: [i64; 64],
    null_time: u64,
    conversion_error: u64,
    null_key: u64,
    late: u64,
    retain: u64,
    watermark: i128,
}

impl AdmissionMasks {
    pub(super) async fn new(
        config: &AdmissionMaskContext<'_>,
        times: &BatchEventTimes<'_>,
        start: usize,
        length: usize,
        context: &StreamOperatorContext<'_>,
        quantum: &mut columnar::Quantum,
    ) -> Result<Self> {
        debug_assert!(length <= 64);
        let watermark = config
            .side_progress
            .and_then(IngressProgress::watermark)
            .map_or(i128::MIN, |time| i128::from(time.as_micros()));
        let threshold = retain_threshold(config);
        quantum
            .step(context, mask_visits(times.unit, watermark, threshold), 512)
            .await?;
        let present = low_bits(length);
        let valid_time = valid_bits(times.array, start, length);
        let mut valid_key = present;
        for &column in &config.plan.key_indices {
            quantum.step(context, 1, 0).await?;
            valid_key &= valid_bits(config.record.column(column).as_ref(), start, length);
        }
        let (times, conversion_error) =
            normalize_times(&times.values[start..start + length], times.unit, valid_time);
        let (late, retain) =
            temporal_masks(&times, valid_time & !conversion_error, watermark, threshold);
        #[cfg(test)]
        note_join_work(|work| work.admission_mask_blocks += 1);
        Ok(Self {
            times,
            null_time: present & !valid_time,
            conversion_error,
            null_key: present & !valid_key,
            late,
            retain,
            watermark,
        })
    }

    pub(super) fn all_admitted(&self, length: usize) -> bool {
        (self.null_time | self.conversion_error | self.null_key | self.late) & low_bits(length) == 0
    }

    pub(super) fn admitted_time(&self, offset: usize) -> EventTime {
        EventTime::from_micros(self.times[offset])
    }

    pub(super) fn at(&self, offset: usize, name: &str, ingress: &str) -> Result<RowAdmission> {
        let bit = 1_u64 << offset;
        if self.null_time & bit != 0 {
            return Ok(RowAdmission::Dropped(DropKind::NullEventTime));
        }
        if self.conversion_error & bit != 0 {
            return Err(operator_reason(
                name,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{ingress} event time cannot be represented"),
            ));
        }
        if self.null_key & bit != 0 {
            return Ok(RowAdmission::Dropped(DropKind::NullKey));
        }
        if self.late & bit != 0 {
            let lateness = u64::try_from(self.watermark - i128::from(self.times[offset]))
                .map_err(|_| counter_overflow(name, "lateness"))?;
            return Ok(RowAdmission::Dropped(DropKind::Late(lateness)));
        }
        Ok(RowAdmission::Admitted(EventTime::from_micros(
            self.times[offset],
        )))
    }

    pub(super) fn retain(&self, offset: usize) -> bool {
        self.retain & (1_u64 << offset) != 0
    }
}

fn low_bits(length: usize) -> u64 {
    if length == 64 {
        u64::MAX
    } else {
        (1_u64 << length) - 1
    }
}

fn valid_bits(array: &dyn Array, start: usize, length: usize) -> u64 {
    array.nulls().map_or_else(
        || low_bits(length),
        |nulls| {
            let bits = nulls.inner();
            BitChunks::new(bits.values(), bits.offset() + start, length)
                .iter_padded()
                .next()
                .unwrap_or(0)
        },
    )
}

fn normalize_times(values: &[i64], unit: TimeUnit, mut valid: u64) -> ([i64; 64], u64) {
    let mut times = [0; 64];
    if unit == TimeUnit::Microsecond {
        times[..values.len()].copy_from_slice(values);
        return (times, 0);
    }
    let mut errors = 0;
    while valid != 0 {
        #[cfg(test)]
        note_join_work(|work| work.normalized_time_visits += 1);
        let offset = valid.trailing_zeros() as usize;
        match normalize_time(values[offset], unit) {
            Some(time) => times[offset] = time,
            None => errors |= 1_u64 << offset,
        }
        valid &= valid - 1;
    }
    (times, errors)
}

fn normalize_time(value: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => value.checked_mul(1_000_000),
        TimeUnit::Millisecond => value.checked_mul(1_000),
        TimeUnit::Microsecond => Some(value),
        TimeUnit::Nanosecond => Some(value.div_euclid(1_000)),
    }
}

fn retain_threshold(config: &AdmissionMaskContext<'_>) -> i128 {
    let Some(opposite) = config.opposite else {
        return i128::MIN;
    };
    if opposite.state() == crate::IngressState::Ended {
        return i128::MAX;
    }
    let extension = if config.plan.incoming_is_left {
        config.bounds.after_micros
    } else {
        config.bounds.before_micros
    };
    opposite.watermark().map_or(i128::MIN, |time| {
        i128::from(time.as_micros()) - i128::from(extension)
    })
}

fn temporal_masks(
    times: &[i64; 64],
    mut valid: u64,
    watermark: i128,
    threshold: i128,
) -> (u64, u64) {
    if let Some(retain) = constant_retention(watermark, threshold) {
        return (0, if retain { valid } else { 0 });
    }
    let mut late = 0;
    let mut retain = 0;
    while valid != 0 {
        #[cfg(test)]
        note_join_work(|work| work.temporal_mask_visits += 1);
        let offset = valid.trailing_zeros() as usize;
        let time = i128::from(times[offset]);
        late |= u64::from(time < watermark) << offset;
        retain |= u64::from(time >= threshold) << offset;
        valid &= valid - 1;
    }
    (late, retain)
}

fn constant_retention(watermark: i128, threshold: i128) -> Option<bool> {
    if watermark != i128::MIN {
        return None;
    }
    match threshold {
        i128::MIN => Some(true),
        i128::MAX => Some(false),
        _ => None,
    }
}

fn mask_visits(unit: TimeUnit, watermark: i128, threshold: i128) -> usize {
    if unit == TimeUnit::Microsecond && constant_retention(watermark, threshold).is_some() {
        1
    } else {
        64
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::super::{join_work, reset_join_work};
    use super::*;

    pub(in crate::operator::join) fn assert_microsecond_copy_skips_scalar_normalization() {
        for length in [0, 1, 63, 64] {
            let values = (0..length)
                .map(|row| match row % 3 {
                    0 => i64::MIN,
                    1 => i64::MAX,
                    _ => -1,
                })
                .collect::<Vec<_>>();
            let valid = low_bits(length) & 0xaaaa_aaaa_aaaa_aaaa;
            reset_join_work();
            let (times, errors) = normalize_times(&values, TimeUnit::Microsecond, valid);
            assert_eq!(errors, 0);
            for row in 0..length {
                if valid & (1_u64 << row) != 0 {
                    assert_eq!(times[row], values[row]);
                }
            }
            assert!(times[length..].iter().all(|&value| value == 0));
            assert_eq!(join_work().normalized_time_visits, 0);
        }
    }

    pub(in crate::operator::join) fn assert_constant_temporal_masks_skip_scalar_rows() {
        let mut times = [0; 64];
        times[0] = i64::MIN;
        times[63] = i64::MAX;
        for valid in [0, 1, 0xaaaa_aaaa_aaaa_aaaa, u64::MAX] {
            for (threshold, retained) in [(i128::MIN, valid), (i128::MAX, 0)] {
                reset_join_work();
                assert_eq!(
                    temporal_masks(&times, valid, i128::MIN, threshold),
                    (0, retained)
                );
                assert_eq!(join_work().temporal_mask_visits, 0);
            }
        }
    }

    pub(in crate::operator::join) fn assert_vectorized_masks_equal_scalar_units_and_finite_boundaries()
     {
        let values = [i64::MIN, -1_001, -1, 0, 1, 1_001, i64::MAX];
        for unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            for valid in [0, 0x55, 0x7f] {
                let (times, errors) = normalize_times(&values, unit, valid);
                for (row, &value) in values.iter().enumerate() {
                    if valid & (1 << row) != 0 {
                        assert_eq!(
                            errors & (1 << row) != 0,
                            normalize_time(value, unit).is_none()
                        );
                    }
                }
                for watermark in [i128::MIN, i128::from(i64::MIN), 0, i128::from(i64::MAX)] {
                    for threshold in [
                        i128::MIN,
                        i128::from(i64::MIN),
                        0,
                        i128::from(i64::MAX),
                        i128::MAX,
                    ] {
                        let present = valid & !errors;
                        let expected = (0..values.len())
                            .filter(|&row| present & (1 << row) != 0)
                            .fold((0, 0), |(late, retain), row| {
                                let time = i128::from(normalize_time(values[row], unit).unwrap());
                                (
                                    late | (u64::from(time < watermark) << row),
                                    retain | (u64::from(time >= threshold) << row),
                                )
                            });
                        assert_eq!(
                            temporal_masks(&times, present, watermark, threshold),
                            expected
                        );
                    }
                }
            }
        }
    }
}
