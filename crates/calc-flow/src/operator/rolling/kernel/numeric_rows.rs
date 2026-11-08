//! Optional shared Rows storage; arithmetic and durable state stay per window.

use std::{
    collections::VecDeque,
    fmt,
    mem::size_of,
    sync::{Arc, OnceLock},
};

use datafusion::execution::memory_pool::{
    GreedyMemoryPool, MemoryConsumer, MemoryPool, MemoryReservation,
};

use super::{
    Float64NumericState, RollingMetricsRecorder, RollingNumericalProfile, SumClass, TimedSample,
    TypedEntityState, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedWindowState,
    WindowAccumulator, add_float64, internal_error, remove_float64, stable_v2_float64_accumulator,
    stable_v2_rebase_due, valid_float64,
};
use crate::Result;

const CACHE_LIMIT: usize = 64 << 20;
const REGISTRATION_BYTES: usize = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SharedRowsShape {
    windows: [usize; 2],
    widest: usize,
}

impl SharedRowsShape {
    pub(super) fn compile(groups: &[TypedGroupPlan]) -> Option<Self> {
        let [
            TypedGroupPlan::Numeric {
                input_index: left,
                frame: TypedFrame::Rows(first),
            },
            TypedGroupPlan::Numeric {
                input_index: right,
                frame: TypedFrame::Rows(second),
            },
        ] = groups
        else {
            return None;
        };
        (left == right && first != second).then_some(Self {
            windows: [*first, *second],
            widest: usize::from(second > first),
        })
    }
}

pub(super) fn prepare_typed_entities(
    shape: Option<SharedRowsShape>,
    states: &mut [Arc<TypedEntityState>],
    entity_ids: &[usize],
    observer: Option<&RollingMetricsRecorder>,
) {
    if shape.is_some() {
        drop(super::mean::touched_entities(
            states, entity_ids, shape, observer,
        ));
    }
}

#[cfg(test)]
thread_local! {
    static TEST_POOL: std::cell::RefCell<Option<Arc<dyn MemoryPool>>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(super) struct TestPoolScope {
    previous: Option<Arc<dyn MemoryPool>>,
}

#[cfg(test)]
impl Drop for TestPoolScope {
    fn drop(&mut self) {
        TEST_POOL.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}

#[cfg(test)]
pub(super) fn test_pool_scope(pool: Arc<dyn MemoryPool>) -> TestPoolScope {
    let previous = TEST_POOL.with(|slot| slot.replace(Some(pool)));
    TestPoolScope { previous }
}

fn cache_pool() -> Arc<dyn MemoryPool> {
    static POOL: OnceLock<Arc<dyn MemoryPool>> = OnceLock::new();
    #[cfg(test)]
    if let Some(pool) = TEST_POOL.with(|slot| slot.borrow().clone()) {
        return pool;
    }
    Arc::clone(POOL.get_or_init(|| Arc::new(GreedyMemoryPool::new(CACHE_LIMIT))))
}

pub(super) struct RowsTape {
    rows: VecDeque<Option<f64>>,
    window: usize,
    pool: Arc<dyn MemoryPool>,
    credit: MemoryReservation,
}

impl fmt::Debug for RowsTape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RowsTape")
            .field("rows", &self.rows)
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

fn allocation_bytes(capacity: usize) -> Option<usize> {
    capacity
        .checked_mul(size_of::<Option<f64>>())?
        .checked_add(size_of::<RowsTape>())?
        .checked_add(2 * size_of::<usize>())?
        .checked_add(REGISTRATION_BYTES)
}

impl RowsTape {
    fn try_new(window: usize, pool: Arc<dyn MemoryPool>) -> Option<Self> {
        let capacity = window.checked_add(1)?;
        let credit = MemoryConsumer::new("rolling shared numeric rows").register(&pool);
        Self::try_with_credit(window, capacity, pool, credit)
    }

    fn try_with_credit(
        window: usize,
        capacity: usize,
        pool: Arc<dyn MemoryPool>,
        credit: MemoryReservation,
    ) -> Option<Self> {
        let reserved = allocation_bytes(capacity.checked_mul(2)?)?;
        credit.try_grow(reserved).ok()?;
        let rows = VecDeque::with_capacity(capacity);
        let actual = allocation_bytes(rows.capacity())?;
        if actual > reserved {
            return None;
        }
        credit.shrink(reserved - actual);
        Some(Self {
            rows,
            window,
            pool,
            credit,
        })
    }

    fn try_clone_for_batch(&self) -> Option<Arc<Self>> {
        let mut copied = Self::try_with_credit(
            self.window,
            self.window.checked_add(1)?,
            Arc::clone(&self.pool),
            self.credit.new_empty(),
        )?;
        copied.rows.extend(self.rows.iter().copied());
        Some(Arc::new(copied))
    }

    fn window_values(&self, window: usize) -> impl Iterator<Item = f64> + Clone + '_ {
        self.rows
            .iter()
            .skip(self.rows.len().saturating_sub(window))
            .filter_map(|value| *value)
    }

    pub(super) fn bytes(&self) -> usize {
        self.credit.size()
    }
}

impl TypedEntityState {
    pub(super) fn prepare_shared_rows(&mut self, shape: Option<SharedRowsShape>) {
        if self.shared_rows.is_some() {
            self.prepare_shared_rows_copy();
        } else if self.transition_count == 0
            && let Some(shape) = shape
        {
            self.try_share_rows(shape, cache_pool());
        }
    }

    fn try_share_rows(&mut self, shape: SharedRowsShape, pool: Arc<dyn MemoryPool>) {
        if !self.matches_shared_shape(shape) {
            return;
        }
        let Some(mut tape) = RowsTape::try_new(shape.windows[shape.widest], pool) else {
            return;
        };
        let TypedWindowState::Numeric(widest) = &self.groups[shape.widest] else {
            return;
        };
        tape.rows
            .extend(widest.samples.values.iter().map(|sample| sample.value));
        for group in &mut self.groups {
            if let TypedWindowState::Numeric(state) = group {
                state.samples.values = VecDeque::new();
            }
        }
        self.shared_rows = Some(Arc::new(tape));
    }

    fn matches_shared_shape(&self, shape: SharedRowsShape) -> bool {
        self.groups.len() == 2
            && self.groups.iter().zip(shape.windows).all(|(group, width)| {
                matches!(group, TypedWindowState::Numeric(state)
                if matches!(state.samples.frame, TypedFrame::Rows(actual) if actual == width)
                    && state.samples.len() <= width)
            })
    }

    fn prepare_shared_rows_copy(&mut self) {
        let Some(tape) = &mut self.shared_rows else {
            return;
        };
        if Arc::get_mut(tape).is_some() {
            return;
        }
        if let Some(copied) = tape.try_clone_for_batch() {
            self.shared_rows = Some(copied);
        } else {
            self.materialize_shared_rows();
        }
    }

    fn materialize_shared_rows(&mut self) {
        let Some(tape) = self.shared_rows.take() else {
            return;
        };
        for group in self.groups.iter_mut().take(2) {
            if let TypedWindowState::Numeric(state) = group
                && let TypedFrame::Rows(width) = state.samples.frame
            {
                let mut values = VecDeque::with_capacity(width.saturating_add(1));
                values.extend(
                    tape.rows
                        .iter()
                        .skip(tape.rows.len().saturating_sub(width))
                        .map(|&value| TimedSample {
                            event_time: 0,
                            value,
                        }),
                );
                state.samples.values = values;
            }
        }
    }

    pub(super) fn shared_rows_match_groups(&self, groups: &[TypedGroupPlan]) -> bool {
        let Some(tape) = &self.shared_rows else {
            return true;
        };
        let Some(shape) = SharedRowsShape::compile(groups) else {
            return false;
        };
        tape.window == shape.windows[shape.widest]
            && tape.rows.len() <= tape.window
            && self.groups.iter().all(|group| {
                matches!(group,
                TypedWindowState::Numeric(state) if state.samples.values.is_empty())
            })
    }

    pub(super) fn prepare_lane_shared_rows(&mut self, columns: &[TypedGroupInput]) {
        if self.shared_rows.is_none() {
            return;
        }
        let compatible = columns.len() == 2
            && self
                .groups
                .iter()
                .take(2)
                .zip(columns)
                .all(|(state, input)| {
                    matches!(
                        (state, input),
                        (TypedWindowState::Numeric(_), TypedGroupInput::Single(_))
                    )
                });
        if compatible {
            self.prepare_shared_rows_copy();
        } else {
            self.materialize_shared_rows();
        }
    }

    pub(super) fn rebase_shared_rows(&mut self, node: &str) -> Result<()> {
        let tape = self
            .shared_rows
            .as_ref()
            .ok_or_else(|| internal_error("shared numeric rows are missing"))?;
        for group in self.groups.iter_mut().take(2) {
            let TypedWindowState::Numeric(state) = group else {
                return Err(internal_error(
                    "typed rolling group state does not match its input plan",
                ));
            };
            let TypedFrame::Rows(width) = state.samples.frame else {
                return Err(internal_error("shared numeric group is not a Rows frame"));
            };
            state.accumulator = stable_v2_float64_accumulator(tape.window_values(width), node)?;
        }
        Ok(())
    }

    pub(super) fn push_shared_row(&mut self, sample: Option<f64>) {
        let tape = self
            .shared_rows
            .as_mut()
            .and_then(Arc::get_mut)
            .expect("shared rows were made private before the row loop");
        debug_assert!(tape.rows.len() < tape.rows.capacity());
        tape.rows.push_back(sample);
    }

    pub(super) fn trim_shared_rows(&mut self) {
        let tape = self
            .shared_rows
            .as_mut()
            .and_then(Arc::get_mut)
            .expect("shared rows were made private before the row loop");
        if tape.rows.len() > tape.window {
            tape.rows.pop_front();
        }
    }

    pub(super) fn update_shared_group(
        &mut self,
        ordinal: usize,
        sample: Option<f64>,
        profile: RollingNumericalProfile,
        node: &str,
    ) -> Result<()> {
        let tape = self
            .shared_rows
            .as_ref()
            .ok_or_else(|| internal_error("shared numeric rows are missing"))?;
        let Some(TypedWindowState::Numeric(state)) = self.groups.get_mut(ordinal) else {
            return Err(internal_error(
                "typed rolling group state does not match its input plan",
            ));
        };
        let TypedFrame::Rows(width) = state.samples.frame else {
            return Err(internal_error("shared numeric group is not a Rows frame"));
        };
        update_shared_numeric(
            state,
            tape,
            width,
            sample,
            profile,
            self.transition_count,
            node,
        )
    }
}

pub(super) fn update_shared_groups(
    entity: &mut TypedEntityState,
    inputs: &[TypedGroupInput],
    row: usize,
    profile: RollingNumericalProfile,
    nan_as_value: bool,
    node: &str,
) -> Result<()> {
    let Some(TypedGroupInput::Single(first)) = inputs.first() else {
        return Err(internal_error(
            "typed rolling group state does not match its input plan",
        ));
    };
    let sample = valid_float64(first, row, nan_as_value);
    entity.push_shared_row(sample);
    for ordinal in 0..entity.groups.len().min(inputs.len()) {
        entity.update_shared_group(ordinal, sample, profile, node)?;
    }
    entity.trim_shared_rows();
    Ok(())
}

fn update_shared_numeric(
    state: &mut Float64NumericState,
    tape: &RowsTape,
    width: usize,
    sample: Option<f64>,
    profile: RollingNumericalProfile,
    transitions: u64,
    node: &str,
) -> Result<()> {
    if let Some(value) = sample {
        add_float64(&mut state.accumulator, value, node)?;
    }
    let leaving = tape
        .rows
        .len()
        .checked_sub(width.saturating_add(1))
        .and_then(|index| tape.rows[index]);
    if let Some(value) = leaving {
        remove_float64(&mut state.accumulator, value)?;
    }
    repair_shared_numeric(
        &mut state.accumulator,
        tape,
        width,
        profile,
        transitions,
        node,
    )
}

fn repair_shared_numeric(
    accumulator: &mut WindowAccumulator,
    tape: &RowsTape,
    width: usize,
    profile: RollingNumericalProfile,
    transitions: u64,
    node: &str,
) -> Result<()> {
    if accumulator.nan_count > 0 {
        return Ok(());
    }
    let count = width.min(tape.rows.len());
    if profile == RollingNumericalProfile::StableV2Preview
        && stable_v2_rebase_due(transitions, count, accumulator.is_non_finite())
    {
        *accumulator = stable_v2_float64_accumulator(tape.window_values(width), node)?;
    } else if accumulator.is_non_finite() {
        *accumulator = WindowAccumulator::new(SumClass::Float);
        for sample in tape.window_values(width) {
            add_float64(accumulator, sample, node)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state_memory_pool::OwnedStatePool;

    fn groups(windows: [usize; 2]) -> Vec<TypedGroupPlan> {
        windows
            .into_iter()
            .map(|window| TypedGroupPlan::Numeric {
                input_index: 1,
                frame: TypedFrame::Rows(window),
            })
            .collect()
    }

    fn accumulator_bits(value: &WindowAccumulator) -> [u64; 8] {
        let Some(super::super::SumState::Float(sum)) = value.sum else {
            panic!("float summary required");
        };
        [
            value.valid_count,
            sum.to_bits(),
            value.mean.to_bits(),
            value.m2.to_bits(),
            value.nan_count,
            value.pos_inf,
            value.neg_inf,
            u64::from(value.expired_through == i128::MIN),
        ]
    }

    fn numeric(entity: &TypedEntityState, ordinal: usize) -> &Float64NumericState {
        let TypedWindowState::Numeric(state) = &entity.groups[ordinal] else {
            panic!("numeric required");
        };
        state
    }

    fn advance(
        entity: &mut TypedEntityState,
        value: Option<f64>,
        profile: RollingNumericalProfile,
    ) {
        entity.transition_count += 1;
        entity.push_shared_row(value);
        for ordinal in 0..2 {
            entity
                .update_shared_group(ordinal, value, profile, "r")
                .unwrap();
        }
        entity.trim_shared_rows();
    }

    #[test]
    fn shared_numeric_rows_match_every_independent_summary_bit() {
        let special = [
            None,
            Some(-0.0),
            Some(0.0),
            Some(1e155),
            Some(1e155),
            Some(1e155),
            Some(-1e155),
            Some(1e308),
            Some(1e308),
            Some(-1e308),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(f64::from_bits(0x7ff8_0000_0000_1234)),
            Some(1.0),
            Some(-1.0),
            Some(1e-300),
        ];
        for profile in [
            RollingNumericalProfile::StableV1,
            RollingNumericalProfile::StableV2Preview,
        ] {
            for windows in [[2, 4], [5, 20], [64, 80]] {
                for nan_as_value in [false, true] {
                    let plans = groups(windows);
                    let mut shared = TypedEntityState::new(&plans, 300);
                    shared.try_share_rows(
                        SharedRowsShape::compile(&plans).unwrap(),
                        Arc::new(GreedyMemoryPool::new(CACHE_LIMIT)),
                    );
                    let mut independent = TypedEntityState::new(&plans, 300);
                    for row in 0..300 {
                        let value = special[row % special.len()]
                            .filter(|value| nan_as_value || !value.is_nan());
                        advance(&mut shared, value, profile);
                        independent.transition_count += 1;
                        for ordinal in 0..2 {
                            let TypedWindowState::Numeric(state) = &mut independent.groups[ordinal]
                            else {
                                unreachable!();
                            };
                            state
                                .update(
                                    i64::try_from(row).unwrap(),
                                    value,
                                    profile,
                                    independent.transition_count,
                                    "r",
                                )
                                .unwrap();
                            assert_eq!(
                                accumulator_bits(&numeric(&shared, ordinal).accumulator),
                                accumulator_bits(&state.accumulator),
                                "{profile:?} windows={windows:?} row={row} group={ordinal} nan={nan_as_value}"
                            );
                            let actual = shared
                                .shared_rows
                                .as_ref()
                                .unwrap()
                                .rows
                                .iter()
                                .skip(
                                    shared
                                        .shared_rows
                                        .as_ref()
                                        .unwrap()
                                        .rows
                                        .len()
                                        .saturating_sub(windows[ordinal]),
                                )
                                .map(|value| value.map(f64::to_bits))
                                .collect::<Vec<_>>();
                            let expected = state
                                .samples
                                .values
                                .iter()
                                .map(|sample| sample.value.map(f64::to_bits))
                                .collect::<Vec<_>>();
                            assert_eq!(actual, expected);
                            assert!(numeric(&shared, ordinal).samples.values.is_empty());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn shared_rows_restore_rebase_uses_each_borrowed_window_without_copying_tape() {
        let plans = groups([5, 80]);
        let mut shared = TypedEntityState::new(&plans, 160);
        shared.try_share_rows(
            SharedRowsShape::compile(&plans).unwrap(),
            Arc::new(GreedyMemoryPool::new(CACHE_LIMIT)),
        );
        let mut independent = TypedEntityState::new(&plans, 160);
        for row in 0..160_u32 {
            let value = Some(1e12 + f64::from(row) * 0.3);
            advance(&mut shared, value, RollingNumericalProfile::StableV2Preview);
            independent.transition_count += 1;
            for group in &mut independent.groups {
                let TypedWindowState::Numeric(state) = group else {
                    unreachable!();
                };
                state
                    .update(
                        i64::from(row),
                        value,
                        RollingNumericalProfile::StableV2Preview,
                        independent.transition_count,
                        "r",
                    )
                    .unwrap();
            }
        }
        let old = Arc::clone(shared.shared_rows.as_ref().unwrap());
        let credit = old.pool.reserved();
        let mut restored = shared.clone();
        restored.force_stable_v2_rebase("r").unwrap();
        independent.force_stable_v2_rebase("r").unwrap();
        for ordinal in 0..2 {
            assert_eq!(
                accumulator_bits(&numeric(&restored, ordinal).accumulator),
                accumulator_bits(&numeric(&independent, ordinal).accumulator)
            );
        }
        assert!(Arc::ptr_eq(restored.shared_rows.as_ref().unwrap(), &old));
        assert_eq!(old.pool.reserved(), credit);
    }

    #[test]
    fn shared_rows_copy_once_and_keep_each_allocation_credit_until_last_alias() {
        let pool = Arc::new(GreedyMemoryPool::new(CACHE_LIMIT));
        let plans = groups([5, 20]);
        let shape = SharedRowsShape::compile(&plans).unwrap();
        let mut resident = TypedEntityState::new(&plans, 32);
        resident.try_share_rows(shape, pool.clone());
        for value in 1..=32 {
            advance(
                &mut resident,
                Some(f64::from(value)),
                RollingNumericalProfile::StableV1,
            );
        }
        let old = Arc::clone(resident.shared_rows.as_ref().unwrap());
        let one_credit = old.bytes();
        assert_eq!(pool.reserved(), one_credit);
        assert_eq!(one_credit, allocation_bytes(old.rows.capacity()).unwrap());
        let mut candidate = resident.clone();
        let copied_allocations =
            allocation_counter::measure(|| candidate.prepare_shared_rows(Some(shape)));
        assert_eq!(copied_allocations.count_total, 2, "{copied_allocations:?}");
        assert!(std::ptr::eq(
            candidate.shared_rows.as_ref().unwrap().credit.consumer(),
            old.credit.consumer()
        ));
        assert_eq!(pool.reserved(), 2 * one_credit);
        assert!(!Arc::ptr_eq(candidate.shared_rows.as_ref().unwrap(), &old));
        assert_eq!(candidate.shared_rows.as_ref().unwrap().rows.len(), 20);
        let copied = Arc::clone(candidate.shared_rows.as_ref().unwrap());
        let capacity = copied.rows.capacity();
        drop(copied);
        let no_allocations = allocation_counter::measure(|| {
            advance(
                &mut candidate,
                Some(33.0),
                RollingNumericalProfile::StableV1,
            );
        });
        assert_eq!(no_allocations.count_total, 0, "{no_allocations:?}");
        assert_eq!(
            candidate.shared_rows.as_ref().unwrap().rows.capacity(),
            capacity
        );
        assert_eq!(old.rows.back(), Some(&Some(32.0)));
        drop(candidate);
        assert_eq!(pool.reserved(), one_credit);
        drop(resident);
        assert_eq!(pool.reserved(), one_credit);
        drop(old);
        assert_eq!(pool.reserved(), 0);
    }

    #[test]
    fn shared_rows_credit_refusal_materializes_suffix_without_replaying_summary() {
        let pool = Arc::new(OwnedStatePool::new(
            Arc::new(GreedyMemoryPool::new(CACHE_LIMIT)),
            "shared test",
            "shared test quota",
        ));
        let plans = groups([2, 4]);
        let shape = SharedRowsShape::compile(&plans).unwrap();
        let mut resident = TypedEntityState::new(&plans, 8);
        resident.try_share_rows(shape, pool.clone());
        for value in [1e155, 1e155, 1e155, -1e155, -0.0, 0.0] {
            advance(
                &mut resident,
                Some(value),
                RollingNumericalProfile::StableV1,
            );
        }
        let prior = [
            accumulator_bits(&numeric(&resident, 0).accumulator),
            accumulator_bits(&numeric(&resident, 1).accumulator),
        ];
        let credit = pool.reserved();
        pool.set_limit(Some(u64::try_from(credit).unwrap()))
            .unwrap();
        let mut candidate = resident.clone();
        candidate.prepare_shared_rows(Some(shape));
        assert!(candidate.shared_rows.is_none());
        assert!(resident.shared_rows.is_some());
        assert_eq!(pool.reserved(), credit);
        for (ordinal, expected) in prior.iter().enumerate() {
            assert_eq!(
                accumulator_bits(&numeric(&candidate, ordinal).accumulator),
                *expected
            );
            assert_eq!(
                numeric(&candidate, ordinal).samples.len(),
                shape.windows[ordinal]
            );
        }
        advance(&mut resident, Some(1.0), RollingNumericalProfile::StableV1);
        candidate.transition_count += 1;
        for ordinal in 0..2 {
            let TypedWindowState::Numeric(state) = &mut candidate.groups[ordinal] else {
                unreachable!();
            };
            state
                .update(
                    7,
                    Some(1.0),
                    RollingNumericalProfile::StableV1,
                    candidate.transition_count,
                    "r",
                )
                .unwrap();
            assert_eq!(
                accumulator_bits(&state.accumulator),
                accumulator_bits(&numeric(&resident, ordinal).accumulator)
            );
        }
        drop(candidate);
        drop(resident);
        assert_eq!(pool.reserved(), 0);
        let mut refused = TypedEntityState::new(&plans, 8);
        refused.try_share_rows(shape, Arc::new(GreedyMemoryPool::new(0)));
        assert!(refused.shared_rows.is_none());
        assert_eq!(refused.transition_count, 0);
        assert_eq!(SharedRowsShape::compile(&groups([4, 4])), None);
        assert!(
            RowsTape::try_new(usize::MAX, Arc::new(GreedyMemoryPool::new(CACHE_LIMIT))).is_none()
        );
    }
}
