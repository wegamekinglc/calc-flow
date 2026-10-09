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
        let present = low_bits(length);
        let valid_time = valid_bits(times.array, start, length);
        let mut valid_key = present;
        for &column in &config.plan.key_indices {
            quantum.step(context, 1, 0).await?;
            valid_key &= valid_bits(config.record.column(column).as_ref(), start, length);
        }
        let (times, conversion_error) =
            normalize_times(&times.values[start..start + length], times.unit, valid_time);
        let watermark = config
            .side_progress
            .and_then(IngressProgress::watermark)
            .map_or(i128::MIN, |time| i128::from(time.as_micros()));
        let (late, retain) = temporal_masks(
            &times,
            valid_time & !conversion_error,
            watermark,
            retain_threshold(config),
        );
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
    let mut errors = 0;
    while valid != 0 {
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
    let mut late = 0;
    let mut retain = 0;
    while valid != 0 {
        let offset = valid.trailing_zeros() as usize;
        let time = i128::from(times[offset]);
        late |= u64::from(time < watermark) << offset;
        retain |= u64::from(time >= threshold) << offset;
        valid &= valid - 1;
    }
    (late, retain)
}
