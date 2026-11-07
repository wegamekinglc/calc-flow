//! Certified rolling decay; numerical argument lives in the P2 specification.

use std::{collections::VecDeque, mem::size_of, sync::Arc};

use datafusion::execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};

use crate::operator::rolling_metrics::DecayWork;

const ETA: f64 = f64::from_bits(1);
const UNIT_ROUNDOFF: f64 = 1.0 / 9_007_199_254_740_992.0;
const INTERNAL_TOLERANCE: f64 = 1.0 / 1_099_511_627_776.0;
const EXACT_INTEGER: u128 = 9_007_199_254_740_992;
const REGISTRATION_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Default)]
struct Summary {
    sum: f64,
    weighted: f64,
    sum_error: f64,
    weighted_error: f64,
    absolute_sum: f64,
}

impl Summary {
    fn remove(&mut self, value: f64) {
        self.weighted -= self.sum;
        self.weighted_error = upper_add(
            upper_add(self.weighted_error, self.sum_error),
            rounding_radius(self.weighted),
        );
        self.sum -= value;
        self.sum_error = upper_add(self.sum_error, rounding_radius(self.sum));
        self.absolute_sum = (self.absolute_sum - value.abs()).next_up().max(0.0);
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the certificate separately proves ordinal conversion exactness"
    )]
    fn add(&mut self, value: f64, ordinal: usize) {
        let product = ordinal as f64 * value;
        self.weighted += product;
        self.weighted_error = upper_add(
            upper_add(self.weighted_error, rounding_radius(product)),
            rounding_radius(self.weighted),
        );
        self.sum += value;
        self.sum_error = upper_add(self.sum_error, rounding_radius(self.sum));
        self.absolute_sum = upper_add(self.absolute_sum, value.abs());
    }

    fn finite(self) -> bool {
        [
            self.sum,
            self.weighted,
            self.sum_error,
            self.weighted_error,
            self.absolute_sum,
        ]
        .into_iter()
        .all(f64::is_finite)
            && self.sum_error >= 0.0
            && self.weighted_error >= 0.0
            && self.absolute_sum >= 0.0
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    valid: usize,
    positive: usize,
    negative: usize,
    infinite: usize,
}

impl Counts {
    fn add(&mut self, value: f64) {
        self.valid += 1;
        self.positive += usize::from(value > 0.0);
        self.negative += usize::from(value < 0.0);
        self.infinite += usize::from(value.is_infinite());
    }

    fn remove(&mut self, value: f64) {
        self.valid -= 1;
        self.positive -= usize::from(value > 0.0);
        self.negative -= usize::from(value < 0.0);
        self.infinite -= usize::from(value.is_infinite());
    }

    const fn one_sign(self) -> bool {
        self.positive == 0 || self.negative == 0
    }
}

#[derive(Debug)]
pub(super) struct OwnedDecay {
    window: usize,
    rows: VecDeque<Option<f64>>,
    counts: Counts,
    summary: Summary,
    since_seed: usize,
    pub(super) latest: Option<f64>,
    pool: Arc<dyn MemoryPool>,
    credit: MemoryReservation,
    work: DecayWork,
    #[cfg(test)]
    scanned: usize,
}

impl OwnedDecay {
    pub(super) fn try_new(window: usize, rows: usize, pool: Arc<dyn MemoryPool>) -> Option<Self> {
        let capacity = rows.min(window);
        let credit = MemoryConsumer::new("rolling decay cache").register(&pool);
        let peak = allocation_bytes(capacity.checked_mul(2)?)?;
        credit.try_grow(peak).ok()?;
        let rows = VecDeque::with_capacity(capacity);
        let actual = allocation_bytes(rows.capacity())?;
        if actual > peak {
            return None;
        }
        credit.shrink(peak - actual);
        Some(Self {
            window,
            rows,
            counts: Counts::default(),
            summary: Summary::default(),
            since_seed: 0,
            latest: None,
            pool,
            credit,
            work: DecayWork::default(),
            #[cfg(test)]
            scanned: 0,
        })
    }

    pub(super) fn try_clone_for_batch(&self, additional: usize) -> Option<Self> {
        let needed = self.rows.len().saturating_add(additional).min(self.window);
        let mut cloned = Self::try_new(self.window, needed, Arc::clone(&self.pool))?;
        cloned.rows.extend(self.rows.iter().copied());
        cloned
            .work
            .add(RollingWork::DecayCacheCopiedSamples, self.rows.len());
        cloned.counts = self.counts;
        cloned.summary = self.summary;
        cloned.since_seed = self.since_seed;
        cloned.latest = self.latest;
        #[cfg(test)]
        {
            cloned.scanned = 0;
        }
        Some(cloned)
    }

    pub(super) fn valid_count(&self) -> u64 {
        u64::try_from(self.counts.valid).unwrap_or(u64::MAX)
    }

    pub(super) fn bytes(&self) -> usize {
        self.credit.size()
    }

    pub(super) fn seed(&mut self, values: impl IntoIterator<Item = Option<f64>>) {
        for value in values {
            if self.rows.len() == self.window {
                self.rows.pop_front();
            }
            self.rows.push_back(value);
        }
        self.work
            .add(RollingWork::DecayInitialSeedSamples, self.rows.len());
        self.counts = Counts::default();
        self.work
            .add(RollingWork::DecayInitialSeedSamples, self.rows.len());
        for value in self.rows.iter().flatten() {
            self.counts.add(*value);
        }
        self.reseed(RollingWork::DecayInitialSeedSamples);
    }

    pub(super) fn advance(&mut self, value: Option<f64>) {
        self.work.add(RollingWork::DecayTransitions, 1);
        if self.rows.len() == self.window
            && let Some(leaving) = self.rows.pop_front().flatten()
        {
            self.counts.remove(leaving);
            self.summary.remove(leaving);
        }
        debug_assert!(self.rows.len() < self.rows.capacity());
        self.rows.push_back(value);
        if let Some(value) = value {
            self.counts.add(value);
            self.summary.add(value, self.counts.valid);
        }
        self.since_seed += 1;
        if self.counts.valid == 0 {
            self.work.add(RollingWork::DecayShortcutRows, 1);
            self.summary = Summary::default();
            self.latest = None;
            self.since_seed = 0;
        } else if self.counts.positive == 0 && self.counts.negative == 0 {
            self.work.add(RollingWork::DecayShortcutRows, 1);
            self.summary = Summary::default();
            self.latest = Some(0.0);
            self.since_seed = 0;
        } else if self.counts.infinite > 0 {
            self.work.add(RollingWork::DecayGuardFallbackRows, 1);
            self.reseed(RollingWork::DecayGuardReseedSamples);
        } else if self.since_seed >= self.window {
            self.work.add(RollingWork::DecayScheduledReseedRows, 1);
            self.reseed(RollingWork::DecayScheduledReseedSamples);
        } else if let Some(value) = self.certified() {
            self.work.add(RollingWork::DecayCertifiedRows, 1);
            self.latest = Some(value);
        } else {
            self.work.add(RollingWork::DecayGuardFallbackRows, 1);
            self.reseed(RollingWork::DecayGuardReseedSamples);
        }
    }

    fn reseed(&mut self, samples: RollingWork) {
        self.work.add(samples, self.rows.len());
        let mut summary = Summary::default();
        let mut ordinal = 0;
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for value in self.rows.iter().flatten() {
            ordinal += 1;
            summary.add(*value, ordinal);
            let weight = weight(self.window - self.counts.valid + ordinal);
            numerator += weight * *value;
            denominator += weight;
        }
        self.summary = summary;
        self.latest = (self.counts.valid > 0).then_some(numerator / denominator);
        self.since_seed = 0;
        #[cfg(test)]
        {
            self.scanned += self.rows.len();
        }
    }

    fn certified(&self) -> Option<f64> {
        if !self.summary.finite() {
            return None;
        }
        let denominator = exact_denominator(self.window, self.counts.valid)?;
        let (numerator, error) = self.numerator();
        let value = numerator / denominator;
        if !value.is_finite() || !error.is_finite() {
            return None;
        }
        let absolute = if self.counts.one_sign() {
            upper_add(numerator.abs(), error)
        } else {
            upper_mul(weight(self.window), self.summary.absolute_sum)
        };
        let reference = reference_error(self.counts.valid, absolute, denominator)?;
        let candidate = upper_add(upper_div(error, denominator), rounding_radius(value));
        let bound = upper_add(candidate, reference);
        let scale = (value.abs() - bound).next_down().max(0.0);
        let tolerance = (INTERNAL_TOLERANCE + (INTERNAL_TOLERANCE * scale).next_down()).next_down();
        (bound < value.abs() && bound <= (tolerance / 4.0).next_down()).then_some(value)
    }

    fn numerator(&self) -> (f64, f64) {
        let Summary {
            sum,
            weighted,
            sum_error,
            weighted_error,
            ..
        } = self.summary;
        let shift = self.window - self.counts.valid;
        if shift == 0 {
            return (weighted, weighted_error);
        }
        let factor = weight(shift);
        let product = factor * sum;
        let numerator = weighted + product;
        let product_error = upper_add(upper_mul(factor, sum_error), rounding_radius(product));
        (
            numerator,
            upper_add(
                upper_add(weighted_error, product_error),
                rounding_radius(numerator),
            ),
        )
    }
}

fn allocation_bytes(capacity: usize) -> Option<usize> {
    size_of::<OwnedDecay>()
        .checked_add(2 * size_of::<usize>())?
        .checked_add(REGISTRATION_BYTES)?
        .checked_add(capacity.checked_mul(size_of::<Option<f64>>())?)
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the old Float64 scan defines weight conversion for all supported frames"
)]
fn weight(value: usize) -> f64 {
    value as f64
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the integer total and weights are checked at most 2^53"
)]
fn exact_denominator(window: usize, count: usize) -> Option<f64> {
    if window as u128 > EXACT_INTEGER || count == 0 || count > window {
        return None;
    }
    let mut left = count as u128;
    let mut right = (window as u128)
        .checked_mul(2)?
        .checked_sub(left)?
        .checked_add(1)?;
    if left.is_multiple_of(2) {
        left /= 2;
    } else {
        right /= 2;
    }
    let total = left.checked_mul(right)?;
    (total <= EXACT_INTEGER).then_some(total as f64)
}

fn reference_error(count: usize, absolute: f64, denominator: f64) -> Option<f64> {
    let mu = upper_mul(weight(count.checked_add(1)?), UNIT_ROUNDOFF);
    let lower = (1.0 - mu).next_down();
    if lower <= 0.0 {
        return None;
    }
    let gamma = upper_div(mu, lower);
    let sum_error = upper_add(
        upper_mul(gamma, absolute),
        upper_div(upper_mul(weight(count.checked_mul(2)?), ETA), lower),
    );
    let sum_bound = upper_add(absolute, sum_error);
    if !sum_bound.is_finite() {
        return None;
    }
    let quotient_bound = upper_div(sum_bound, denominator);
    let division = upper_add(upper_mul(2.0 * UNIT_ROUNDOFF, quotient_bound), ETA);
    Some(upper_add(upper_div(sum_error, denominator), division))
}

fn upper_add(left: f64, right: f64) -> f64 {
    (left + right).next_up()
}
fn upper_mul(left: f64, right: f64) -> f64 {
    (left * right).next_up()
}
fn upper_div(left: f64, right: f64) -> f64 {
    (left / right).next_up()
}

fn rounding_radius(value: f64) -> f64 {
    if !value.is_finite() {
        return f64::INFINITY;
    }
    let exponent = i32::try_from((value.to_bits() >> 52) & 0x7ff)
        .expect("binary64 exponent is eleven bits")
        - 1023;
    let power = (exponent - 53).max(-1074);
    if power < -1022 {
        f64::from_bits(1_u64 << u32::try_from(power + 1074).expect("subnormal exponent is bounded"))
    } else {
        f64::from_bits(u64::try_from(power + 1023).expect("normal exponent is bounded") << 52)
    }
}

use std::collections::BTreeMap;

use super::super::{
    CompiledEvaluation, CompiledFrame, CompiledRollingOutput, CompiledWindowGroup, KeyValue,
    RollingHistories, ScalarValue,
};
use super::{
    DerivedBuilder, PreparedTypedFill, RollingKernelPlan, ScanKind, TypedGroupInput,
    TypedGroupPlan, TypedOutputKind, TypedOutputPlan, TypedWindowState, internal_error,
};
use crate::{
    Result,
    operator::rolling_metrics::{RollingMetricsRecorder, RollingStage, RollingWork},
};
use datafusion::arrow::{
    array::{Array, Float64Array, UInt64Array},
    datatypes::DataType,
};

const CACHE_BYTES: usize = 64 << 20;

pub(super) fn cache_pool() -> Arc<dyn MemoryPool> {
    Arc::new(datafusion::execution::memory_pool::GreedyMemoryPool::new(
        CACHE_BYTES,
    ))
}

pub(super) fn legacy_fallback_reason(
    schema: &datafusion::arrow::datatypes::Schema,
    outputs: &[CompiledRollingOutput],
    groups: &[CompiledWindowGroup],
) -> Option<String> {
    if !outputs.iter().any(|output| matches!(output.evaluation, CompiledEvaluation::Scan(scan) if scan.kind == ScanKind::Decay)) {
        return None;
    }
    if let Err(reason) = super::compile_typed_groups(schema, groups) {
        return Some(reason);
    }
    for output in outputs {
        let CompiledEvaluation::Scan(scan) = output.evaluation else {
            continue;
        };
        if output.input_type != DataType::Float64 {
            return Some("typed scan requires Float64 input".into());
        }
        let CompiledFrame::Rows(rows) = scan.frame else {
            return Some("typed scan requires a row frame".into());
        };
        if usize::try_from(rows).is_err() {
            return Some("scan row window does not fit the platform usize".into());
        }
        if !matches!(
            scan.kind,
            ScanKind::Argmax | ScanKind::Argmin | ScanKind::UniqueCount
        ) {
            return Some("typed scan requires argmax, argmin, or unique_count".into());
        }
    }
    None
}

#[derive(Clone)]
pub(super) struct ReferenceInput {
    pub(super) source: Float64Array,
    values: Float64Array,
    counts: UInt64Array,
}

struct ReferenceBuffers {
    values: Vec<Option<f64>>,
    counts: Vec<u64>,
}

impl ReferenceBuffers {
    fn new(rows: usize) -> Self {
        Self {
            values: vec![None; rows],
            counts: vec![0; rows],
        }
    }
}

pub(super) fn prepare_inputs(
    plan: &RollingKernelPlan,
    fill: &mut PreparedTypedFill,
    input: &datafusion::arrow::record_batch::RecordBatch,
    histories: Option<&RollingHistories>,
    node: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<()> {
    let Some(pool) = &plan.decay_pool else {
        return Ok(());
    };
    let mut work = DecayWork::default();
    let result = (|| {
        let mut entity_rows = BTreeMap::<usize, Vec<usize>>::new();
        for (row, entity) in fill.stream.entity_ids.iter().enumerate() {
            entity_rows.entry(*entity).or_default().push(row);
        }
        let mut references = BTreeMap::<usize, ReferenceBuffers>::new();
        for (entity_id, indices) in entity_rows {
            let entity = Arc::make_mut(&mut fill.stream.state.states[entity_id]);
            for (ordinal, group) in entity.groups.iter_mut().enumerate() {
                let TypedWindowState::Decay(cache) = group else {
                    continue;
                };
                let TypedGroupPlan::Scan {
                    input_index,
                    window,
                    kind: ScanKind::Decay,
                } = plan.groups[ordinal]
                else {
                    return Err(internal_error("typed decay cache has no matching plan"));
                };
                let previous = cache.clone();
                let state_stage =
                    observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
                let history = if previous.is_none() {
                    Some(history_values(
                        plan,
                        input,
                        histories,
                        indices[0],
                        input_index,
                        window,
                        entity.transition_count,
                        node,
                        observer,
                        &mut work,
                    )?)
                } else {
                    None
                };
                let owned = if let Some(previous) = previous.as_ref() {
                    previous.try_clone_for_batch(indices.len())
                } else {
                    let values = history.as_ref().expect("reference history was read");
                    let mut owned = OwnedDecay::try_new(
                        window,
                        values.len().saturating_add(indices.len()),
                        Arc::clone(pool),
                    );
                    if let Some(owned) = &mut owned {
                        owned.seed(values.iter().copied());
                    }
                    owned
                };
                *cache = owned.map(|mut owned| {
                    work.merge(std::mem::take(&mut owned.work));
                    Arc::new(owned)
                });
                drop(state_stage);
                let needs_output = plan.outputs.iter().any(|output| {
                    output.group == ordinal && output.kind == TypedOutputKind::Scan(ScanKind::Decay)
                });
                if cache.is_none() && needs_output {
                    let _stage =
                        observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
                    let history = match previous.as_ref() {
                        Some(previous) => {
                            work.add(
                                RollingWork::DecayReferenceHistorySamples,
                                previous.rows.len(),
                            );
                            previous.rows.clone()
                        }
                        None => history.expect("reference history was read"),
                    };
                    let TypedGroupInput::Single(values) = &fill.columns[ordinal] else {
                        return Err(internal_error("typed decay source is not Float64"));
                    };
                    let target = references
                        .entry(ordinal)
                        .or_insert_with(|| ReferenceBuffers::new(input.num_rows()));
                    fill_reference(target, history, values, &indices, window, &mut work);
                }
            }
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
        finish_references(fill, references)
    })();
    if let Some(recorder) = observer {
        recorder.add_decay(&work);
    }
    result
}

fn finish_references(
    fill: &mut PreparedTypedFill,
    references: BTreeMap<usize, ReferenceBuffers>,
) -> Result<()> {
    for (ordinal, buffers) in references {
        let source = match &fill.columns[ordinal] {
            TypedGroupInput::Single(values) => values.clone(),
            _ => return Err(internal_error("typed decay source is not Float64")),
        };
        fill.columns[ordinal] = TypedGroupInput::DecayReference(Box::new(ReferenceInput {
            source,
            values: Float64Array::from(buffers.values),
            counts: UInt64Array::from(buffers.counts),
        }));
    }
    Ok(())
}

pub(super) fn record_work(
    states: &mut [Arc<super::TypedEntityState>],
    entity_ids: &[usize],
    rows: &super::EntityRows,
    observer: Option<&RollingMetricsRecorder>,
) {
    let mut work = DecayWork::default();
    for &first in &rows.first_rows {
        let entity = Arc::get_mut(&mut states[entity_ids[first]])
            .expect("prepared decay entity is privately owned");
        for group in &mut entity.groups {
            if let TypedWindowState::Decay(Some(cache)) = group {
                let cache = Arc::get_mut(cache).expect("prepared decay cache is privately owned");
                work.merge(std::mem::take(&mut cache.work));
            }
        }
    }
    if let Some(recorder) = observer {
        recorder.add_decay(&work);
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "the fallback binds one entity's read-only history to its exact source plan"
)]
fn history_values(
    plan: &RollingKernelPlan,
    input: &datafusion::arrow::record_batch::RecordBatch,
    histories: Option<&RollingHistories>,
    first: usize,
    input_index: usize,
    window: usize,
    transitions: u64,
    node: &str,
    observer: Option<&RollingMetricsRecorder>,
    work: &mut DecayWork,
) -> Result<VecDeque<Option<f64>>> {
    let mut values = VecDeque::new();
    let Some(histories) = histories else {
        if transitions > 0 {
            return Err(internal_error(
                "reference decay requires authoritative retained history",
            ));
        }
        return Ok(values);
    };
    let key = plan
        .partition_columns
        .iter()
        .map(|&index| {
            let value =
                ScalarValue::try_from_array(input.column(index), first).map_err(|error| {
                    internal_error(format!("decay entity key read failed: {error}"))
                })?;
            if let Some(recorder) = observer {
                recorder.add(RollingWork::ScalarValueConversions, 1);
            }
            KeyValue::from_nullable_scalar(&value, node)
        })
        .collect::<Result<Vec<_>>>()?;
    let Some(history) = histories.by_entity.get(&key) else {
        return Ok(values);
    };
    for row in &history.rows {
        let ScalarValue::Float64(value) = &row[input_index] else {
            return Err(internal_error("decay history is not Float64"));
        };
        work.add(RollingWork::DecayReferenceHistorySamples, 1);
        push_history(&mut values, value.filter(|value| !value.is_nan()), window);
    }
    for record in &history.columnar.records {
        let source = record
            .column(input_index)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| internal_error("columnar decay history is not Float64"))?;
        for value in source {
            work.add(RollingWork::DecayReferenceHistorySamples, 1);
            push_history(&mut values, value.filter(|value| !value.is_nan()), window);
        }
    }
    Ok(values)
}

fn push_history(values: &mut VecDeque<Option<f64>>, value: Option<f64>, window: usize) {
    if values.len() == window {
        values.pop_front();
    }
    values.push_back(value);
}

fn fill_reference(
    target: &mut ReferenceBuffers,
    mut history: VecDeque<Option<f64>>,
    source: &Float64Array,
    indices: &[usize],
    window: usize,
    work: &mut DecayWork,
) {
    for &row in indices {
        work.add(RollingWork::DecayTransitions, 1);
        work.add(RollingWork::DecayQuotaReferenceRows, 1);
        push_history(
            &mut history,
            super::valid_float64(source, row, false),
            window,
        );
        work.add(RollingWork::DecayReferenceScanSamples, history.len());
        let valid = history.iter().flatten().count();
        work.add(RollingWork::DecayReferenceScanSamples, history.len());
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        for (ordinal, value) in history.iter().flatten().enumerate() {
            let weight = weight(window - valid + ordinal + 1);
            numerator += weight * *value;
            denominator += weight;
        }
        target.counts[row] = u64::try_from(valid).unwrap_or(u64::MAX);
        target.values[row] = (valid > 0).then_some(numerator / denominator);
    }
}

pub(super) fn append_reference(
    builder: &mut DerivedBuilder,
    output: &TypedOutputPlan,
    input: &TypedGroupInput,
    row: usize,
) -> Result<()> {
    let (DerivedBuilder::Float(builder, _), TypedGroupInput::DecayReference(reference)) =
        (builder, input)
    else {
        return Err(internal_error(
            "reference decay output has no prepared Float64 input",
        ));
    };
    if reference.counts.value(row) < output.min_periods || reference.values.is_null(row) {
        builder.append_null();
    } else {
        builder.append_value(reference.values.value(row));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};

    use super::*;

    fn pool(bytes: usize) -> Arc<dyn MemoryPool> {
        Arc::new(GreedyMemoryPool::new(bytes))
    }

    fn close(left: f64, right: f64) {
        if right.is_nan() {
            assert!(left.is_nan());
        } else if right.is_infinite() || right == 0.0 {
            assert_eq!(left.to_bits(), right.to_bits());
        } else {
            assert!((left - right).abs() <= 1e-12 + 1e-12 * left.abs().max(right.abs()));
        }
    }

    #[test]
    fn p2_decay_certificate_repairs_magnitude_eviction_and_overflow() {
        for (window, values, expected) in [
            (
                2,
                vec![Some(1e20), Some(1.0), Some(1.0)],
                vec![1e20, 1e20 / 3.0, 1.0],
            ),
            (2, vec![Some(1e308)], vec![f64::INFINITY]),
            (
                3,
                vec![Some(1.0), None, Some(4.0)],
                vec![1.0, 1.0, 14.0 / 5.0],
            ),
            (
                2,
                vec![
                    Some(f64::INFINITY),
                    Some(f64::NEG_INFINITY),
                    Some(2.0),
                    Some(3.0),
                ],
                vec![f64::INFINITY, f64::NAN, f64::NEG_INFINITY, 8.0 / 3.0],
            ),
        ] {
            let mut state = OwnedDecay::try_new(window, window, pool(1 << 20)).unwrap();
            for (value, expected) in values.into_iter().zip(expected) {
                state.advance(value);
                close(state.latest.unwrap(), expected);
            }
        }
    }

    #[test]
    fn p2_decay_cache_credit_tracks_allocation_owners_and_deep_copies() {
        let limit = pool(1 << 20);
        let state = Arc::new(OwnedDecay::try_new(64, 64, Arc::clone(&limit)).unwrap());
        let once = limit.reserved();
        assert!(once >= 64 * size_of::<Option<f64>>());
        let shallow = Arc::clone(&state);
        assert_eq!(limit.reserved(), once);
        let deep = state.try_clone_for_batch(0).unwrap();
        assert!(limit.reserved() > once);
        drop(deep);
        assert_eq!(limit.reserved(), once);
        drop(shallow);
        drop(state);
        assert_eq!(limit.reserved(), 0);
        assert!(OwnedDecay::try_new(64, 64, pool(1)).is_none());
    }

    #[test]
    fn p2_decay_rounding_radius_handles_ieee_boundaries() {
        assert_eq!(rounding_radius(0.0), f64::from_bits(1));
        assert_eq!(rounding_radius(f64::from_bits(1)), f64::from_bits(1));
        assert_eq!(rounding_radius(f64::MIN_POSITIVE), f64::from_bits(1));
        assert_eq!(rounding_radius(1.0), 2.0_f64.powi(-53));
        assert_eq!(rounding_radius(f64::MAX), 2.0_f64.powi(970));
    }

    #[test]
    fn p2_extra_decay_normal_work_and_recovered_candidates_remain_certified() {
        for window in [20, 256, 1000] {
            let mut state = OwnedDecay::try_new(window, window, pool(1 << 20)).unwrap();
            let mut recovered = None;
            for index in 0..(4 * window + 17) {
                let value = (index % 23 != 0)
                    .then(|| 100.0 + (f64::from(u32::try_from(index).unwrap()) * 0.17).sin());
                state.advance(value);
                if let Some(recovered) = &mut recovered {
                    OwnedDecay::advance(recovered, value);
                    match (state.latest, recovered.latest) {
                        (Some(left), Some(right)) => close(left, right),
                        values => assert_eq!(values, (None, None)),
                    }
                }
                if index == 2 * window + 3 {
                    let mut rebuilt = OwnedDecay::try_new(window, window, pool(1 << 20)).unwrap();
                    rebuilt.seed(state.rows.iter().copied());
                    recovered = Some(rebuilt);
                }
                let valid = state.rows.iter().flatten().count();
                if valid > 0 {
                    let mut sum = 0.0;
                    let mut denominator = 0.0;
                    for (ordinal, value) in state.rows.iter().flatten().enumerate() {
                        let weight = weight(window - valid + ordinal + 1);
                        sum += weight * *value;
                        denominator += weight;
                    }
                    close(state.latest.unwrap(), sum / denominator);
                } else {
                    assert!(state.latest.is_none());
                }
            }
            assert!(
                state.scanned < 8 * (4 * window + 17),
                "window={window}, samples={}",
                state.scanned
            );
        }
    }

    #[test]
    fn p2_extra_decay_preserves_subnormal_and_signed_zero_readouts() {
        let values = [
            Some(-0.0),
            Some(0.0),
            None,
            Some(f64::from_bits(1)),
            Some(-f64::from_bits(1)),
            Some(f64::MIN_POSITIVE),
            Some(-f64::MIN_POSITIVE),
            Some(1e308),
            Some(-1e308),
            None,
        ];
        for window in [1, 2, 3] {
            let mut state = OwnedDecay::try_new(window, window, pool(1 << 20)).unwrap();
            for value in values {
                state.advance(value);
                let valid = state.rows.iter().flatten().count();
                if valid == 0 {
                    assert!(state.latest.is_none());
                    continue;
                }
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for (ordinal, value) in state.rows.iter().flatten().enumerate() {
                    let factor = weight(window - valid + ordinal + 1);
                    numerator += factor * *value;
                    denominator += factor;
                }
                close(state.latest.unwrap(), numerator / denominator);
            }
        }
    }
}
