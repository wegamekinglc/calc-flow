use std::{collections::BTreeMap, mem::size_of, sync::Arc, time::Instant};

use datafusion::arrow::{
    array::{Array, ArrayRef, Float64Array},
    buffer::NullBuffer,
    record_batch::RecordBatch,
};

use super::{
    OutputStorage, Result, RollingKernelExecution, RollingKernelMetrics, RollingKernelPlan,
    RollingKernelState, RollingMetricsRecorder, RollingStage, RollingWork, StateCharge, Statistic,
    StreamKernelUpdate, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, internal_error, nanos, operator_error,
};
use crate::accumulate::NumericAccumulator;

const SHORT_WINDOW: usize = 5;
const LONG_WINDOW: usize = 20;
const KEYED_PAGE_ROWS: usize = 1_024;

#[derive(Clone, Debug)]
struct DualSmaEntity {
    slots: [Option<f64>; LONG_WINDOW],
    head: usize,
    len: usize,
    accumulators: [NumericAccumulator; 2],
    transitions: u64,
}

impl Default for DualSmaEntity {
    fn default() -> Self {
        Self {
            slots: [None; LONG_WINDOW],
            head: 0,
            len: 0,
            accumulators: [
                NumericAccumulator::mean_only(),
                NumericAccumulator::mean_only(),
            ],
            transitions: 0,
        }
    }
}

impl DualSmaEntity {
    fn position(&self, offset: usize) -> usize {
        (self.head + offset) % LONG_WINDOW
    }

    fn push(&mut self, value: Option<f64>) -> Result<()> {
        let previous_head = self.head;
        let previous_len = self.len;
        let index = self.position(previous_len);
        let evicted = if previous_len == LONG_WINDOW {
            self.slots[index]
        } else {
            None
        };
        let leaving_short = if previous_len >= SHORT_WINDOW {
            self.slots[(previous_head + previous_len - SHORT_WINDOW) % LONG_WINDOW]
        } else {
            None
        };
        if evicted.is_some() {
            self.head = (previous_head + 1) % LONG_WINDOW;
        } else {
            self.len += 1;
        }
        self.slots[index] = value;
        for (index, window, leaving) in
            [(0, SHORT_WINDOW, leaving_short), (1, LONG_WINDOW, evicted)]
        {
            let accumulator = &mut self.accumulators[index];
            accumulator.rows(leaving, value).map_err(|error| {
                internal_error(format!("dual SMA arithmetic failed: {error:?}"))
            })?;
            let history = DualSmaHistory {
                slots: &self.slots,
                head: self.head,
                len: self.len,
                offset: self.len.saturating_sub(window),
            };
            accumulator.finish_rows(window, history).map_err(|error| {
                internal_error(format!("dual SMA arithmetic failed: {error:?}"))
            })?;
        }
        Ok(())
    }
}

#[derive(Clone)]
struct DualSmaHistory<'a> {
    slots: &'a [Option<f64>; LONG_WINDOW],
    head: usize,
    len: usize,
    offset: usize,
}

impl Iterator for DualSmaHistory<'_> {
    type Item = Option<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset == self.len {
            return None;
        }
        let value = self.slots[(self.head + self.offset) % LONG_WINDOW];
        self.offset += 1;
        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.offset;
        (remaining, Some(remaining))
    }

    fn count(self) -> usize {
        self.len - self.offset
    }
}

#[derive(Clone, Debug, Default)]
pub(in crate::operator::rolling) struct DualSmaState {
    entities: BTreeMap<Vec<u8>, DualSmaEntity>,
    order: Vec<Vec<u8>>,
}

impl DualSmaState {
    fn entity_mut(&mut self, key: &[u8]) -> &mut DualSmaEntity {
        if !self.entities.contains_key(key) {
            self.order.push(key.to_vec());
        }
        self.entities.entry(key.to_vec()).or_default()
    }

    pub(super) fn charge(&self, node_id: &str) -> Result<StateCharge> {
        let mut charge = StateCharge::default();
        let counter_bytes = self.order.len().div_ceil(KEYED_PAGE_ROWS) * KEYED_PAGE_ROWS * 8;
        charge = charge.checked_add(
            StateCharge {
                rows: 0,
                bytes: u64::try_from(counter_bytes)
                    .map_err(|_| operator_error(node_id, "dual SMA counter charge overflowed"))?,
            },
            node_id,
        )?;
        let ring_bytes = 20 * size_of::<f64>()
            + 20_usize.div_ceil(8)
            + 5 * size_of::<usize>()
            + 2 * size_of::<NumericAccumulator>();
        for key in &self.order {
            let entity = &self.entities[key];
            charge = charge.checked_add(
                StateCharge {
                    rows: u64::try_from(entity.len)
                        .map_err(|_| operator_error(node_id, "dual SMA row charge overflowed"))?,
                    bytes: u64::try_from(key.len() + ring_bytes)
                        .map_err(|_| operator_error(node_id, "dual SMA byte charge overflowed"))?,
                },
                node_id,
            )?;
        }
        Ok(charge)
    }
}

pub(super) struct DualSmaUpdate {
    previous: DualSmaState,
    next: DualSmaState,
    created: bool,
}

impl DualSmaUpdate {
    pub(super) fn charge_change(&self, node_id: &str) -> Result<(StateCharge, StateCharge)> {
        Ok((self.previous.charge(node_id)?, self.next.charge(node_id)?))
    }

    pub(super) fn rollback(self, base: &mut RollingKernelState) {
        if self.created {
            base.dual_sma = None;
        } else {
            base.dual_sma = Some(self.previous);
        }
    }

    pub(super) fn commit(self, base: &mut RollingKernelState) {
        base.dual_sma = Some(self.next);
    }

    pub(super) fn entity_count(&self) -> usize {
        self.next.order.len()
    }
}

struct DualSmaOutput {
    values: Vec<f64>,
    valid: Vec<bool>,
    output: TypedOutputPlan,
}

impl DualSmaOutput {
    fn new(output: &TypedOutputPlan, rows: usize) -> Option<Self> {
        if output.storage != OutputStorage::Float64
            || !matches!(output.kind, TypedOutputKind::Statistic(Statistic::Mean))
        {
            return None;
        }
        Some(Self {
            values: vec![0.0; rows],
            valid: vec![true; rows],
            output: *output,
        })
    }

    fn append(&mut self, row: usize, accumulators: &[NumericAccumulator]) {
        let accumulator = &accumulators[self.output.group];
        let value =
            if accumulator.valid_count < self.output.min_periods || accumulator.valid_count == 0 {
                None
            } else {
                Some(accumulator.mean())
            };
        self.valid[row] = value.is_some();
        self.values[row] = value.unwrap_or(0.0);
    }

    fn finish(self) -> ArrayRef {
        Arc::new(Float64Array::new(
            self.values.into(),
            Some(NullBuffer::from(self.valid)),
        ))
    }
}

impl RollingKernelPlan {
    fn dual_sma_eligible(&self, state: &RollingKernelState) -> bool {
        if state.dual_sma.is_none() && (state.slab.is_some() || !state.states.is_empty()) {
            return false;
        }
        let [short, long] = self.groups.as_slice() else {
            return false;
        };
        if !matches!(
            (short, long),
            (
                TypedGroupPlan::Numeric {
                    input_index: short_index,
                    frame: TypedFrame::Rows(SHORT_WINDOW),
                    ..
                },
                TypedGroupPlan::Numeric {
                    input_index: long_index,
                    frame: TypedFrame::Rows(LONG_WINDOW),
                    ..
                }
            ) if short_index == long_index
        ) {
            return false;
        }
        self.outputs.len() == 2
            && self.outputs.iter().enumerate().all(|(group, output)| {
                output.group == group
                    && output.min_periods == 1
                    && output.storage == OutputStorage::Float64
                    && matches!(output.kind, TypedOutputKind::Statistic(Statistic::Mean))
            })
    }

    pub(in crate::operator::rolling) fn prepare_dual_sma(
        &self,
        state: &mut RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Option<StreamKernelUpdate>> {
        if !self.dual_sma_eligible(state) {
            return Ok(None);
        }
        let inputs = self.typed_inputs(input, node_id)?;
        let Some(TypedGroupInput::Single(values)) = inputs.first() else {
            return Err(internal_error("dual SMA input must be one Float64 column"));
        };
        let last_identity = self.last_ordered_identity(input, node_id)?;
        let previous = state.dual_sma.clone().unwrap_or_default();
        let mut next = previous.clone();
        let created = state.dual_sma.is_none();
        let started = Instant::now();
        let routing = self.encoded_entity_routing(input, next.order.len(), node_id)?;
        let entity_encode_ns = nanos(started.elapsed());
        if let Some(recorder) = observer {
            recorder.add(RollingWork::ResolvedRows, routing.entity_ids.len());
            recorder.add(RollingWork::TouchedEntities, routing.keys.len());
        }
        let Some(mut outputs) = self
            .outputs
            .iter()
            .map(|output| DualSmaOutput::new(output, input.num_rows()))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let mut runs = routing
            .rows
            .counts
            .iter()
            .map(|&count| Vec::with_capacity(count))
            .collect::<Vec<_>>();
        for (row, &local) in routing.entity_ids.iter().enumerate() {
            runs[local].push(row);
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
        for (local, key) in routing.keys.iter().enumerate() {
            let entity = next.entity_mut(key);
            entity.transitions = entity
                .transitions
                .checked_add(
                    u64::try_from(runs[local].len()).map_err(|_| {
                        operator_error(node_id, "dual SMA transition count overflowed")
                    })?,
                )
                .ok_or_else(|| operator_error(node_id, "dual SMA transition count overflowed"))?;
            for &row in &runs[local] {
                let raw = values.value(row);
                let value =
                    (values.is_valid(row) && (self.nan_as_value || !raw.is_nan())).then_some(raw);
                entity.push(value)?;
                for output in &mut outputs {
                    output.append(row, &entity.accumulators);
                }
            }
        }
        if let Some(recorder) = observer {
            recorder.add(RollingWork::NumericRows, input.num_rows());
            recorder.add(RollingWork::NumericSerialCalls, 1);
        }
        let columns = outputs.into_iter().map(DualSmaOutput::finish).collect();
        let state_bytes = next.charge(node_id)?.bytes;
        let local = RollingKernelState {
            kernel_fingerprint: Some(self.fingerprint.clone()),
            last_identity,
            ..RollingKernelState::default()
        };
        Ok(Some(StreamKernelUpdate {
            execution: RollingKernelExecution {
                columns,
                entity_ids: routing.entity_ids.clone(),
                entity_rows: routing.rows.clone(),
                metrics: RollingKernelMetrics {
                    entity_encode_ns,
                    entities: routing.keys.len(),
                    input_rows: input.num_rows(),
                    output_rows: input.num_rows(),
                    state_bytes: usize::try_from(state_bytes).unwrap_or(usize::MAX),
                    ..RollingKernelMetrics::default()
                },
                state: local,
            },
            in_place: None,
            slab: None,
            dual_sma: Some(DualSmaUpdate {
                previous,
                next,
                created,
            }),
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{Float64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::super::{
        KernelComplexity, KernelSelection, OutputStorage, ROLLING_KERNEL_PLAN_VERSION,
        RollingKernelPlan, RollingKernelState, Statistic, TypedFrame, TypedGroupPlan,
        TypedOutputKind, TypedOutputPlan,
    };

    fn plan() -> RollingKernelPlan {
        RollingKernelPlan {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version: 7,
            selection: KernelSelection::OrderedPrimitive,
            complexity: KernelComplexity::AmortizedConstant,
            event_time_index: 0,
            order_columns: vec![0, 1],
            partition_columns: vec![1],
            sequence_columns: Vec::new(),
            nan_as_value: false,
            shared_rows: None,
            groups: vec![
                TypedGroupPlan::Numeric {
                    input_index: 2,
                    frame: TypedFrame::Rows(5),
                    dispersion: false,
                },
                TypedGroupPlan::Numeric {
                    input_index: 2,
                    frame: TypedFrame::Rows(20),
                    dispersion: false,
                },
            ],
            outputs: vec![
                TypedOutputPlan {
                    group: 0,
                    kind: TypedOutputKind::Statistic(Statistic::Mean),
                    storage: OutputStorage::Float64,
                    min_periods: 1,
                    ddof: 0,
                },
                TypedOutputPlan {
                    group: 1,
                    kind: TypedOutputKind::Statistic(Statistic::Mean),
                    storage: OutputStorage::Float64,
                    min_periods: 1,
                    ddof: 0,
                },
            ],
            fallback_reason: None,
            estimated_state_bytes_per_entity: 0,
            fingerprint: "dual-sma-test".to_owned(),
            checkpoint_fingerprint: None,
            decay_pool: None,
        }
    }

    fn input(times: &[i64], values: &[Option<f64>]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new(
                "time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                false,
            ),
            Field::new("key", DataType::Utf8, false),
            Field::new("value", DataType::Float64, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(TimestampMicrosecondArray::from_iter_values(
                    times.iter().copied(),
                )),
                Arc::new(StringArray::from(vec!["a"; times.len()])),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    fn means(columns: &[super::super::ArrayRef], index: usize) -> Vec<f64> {
        columns[index]
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn assert_mean_bits(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "row {index}: actual {actual:?}, expected {expected:?}"
            );
        }
    }

    #[test]
    fn dual_sma_matches_outputs_charge_and_rollback() {
        let plan = plan();
        let mut state = RollingKernelState::default();
        let empty = state.canonical_state_for_test();
        let first = input(&[1, 2, 3], &[Some(1.0), Some(2.0), Some(3.0)]);
        let mut update = plan
            .prepare_dual_sma(&mut state, &first, "r", None)
            .unwrap()
            .unwrap();
        let columns = update.take_columns();
        assert_mean_bits(&means(&columns, 0), &[1.0, 1.5, 2.0]);
        assert_mean_bits(&means(&columns, 1), &[1.0, 1.5, 2.0]);
        let mut reference = crate::state::StateTable::new();
        reference.add_keyed_column("transition_count", crate::state::ValueKind::UInt64);
        let column = reference.add_numeric_rows_column("value", 20);
        reference.add_numeric_window(column, 5);
        reference.set_numeric_dispersion(column, 0, false);
        reference.set_numeric_dispersion(column, 1, false);
        let canonical_key = plan.encoded_entity_routing(&first, 0, "r").unwrap().keys[0].clone();
        reference.begin().unwrap();
        let entity = reference
            .resolve_or_append(crate::state::EntityKey::new(canonical_key))
            .unwrap();
        for value in [1.0, 2.0, 3.0] {
            reference
                .push_numeric_rows(column, entity, Some(value))
                .unwrap();
        }
        reference.commit().unwrap();
        let (old, new) = update.charge_change(&state, "r").unwrap();
        assert_eq!(old.rows, 0);
        assert_eq!(old.bytes, 0);
        assert_eq!(new.rows, 3);
        assert_eq!(
            usize::try_from(new.bytes).unwrap(),
            reference.logical_bytes()
        );
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), empty);

        let update = plan
            .prepare_dual_sma(&mut state, &first, "r", None)
            .unwrap()
            .unwrap();
        update.commit(&mut state);
        let first_state = state.canonical_state_for_test();
        let second = input(&[4, 5], &[Some(5.0), Some(7.0)]);
        let mut update = plan
            .prepare_dual_sma(&mut state, &second, "r", None)
            .unwrap()
            .unwrap();
        let columns = update.take_columns();
        assert_mean_bits(&means(&columns, 0), &[2.75, 3.6]);
        assert_mean_bits(&means(&columns, 1), &[2.75, 3.6]);
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), first_state);

        let update = plan
            .prepare_dual_sma(&mut state, &second, "r", None)
            .unwrap()
            .unwrap();
        update.commit(&mut state);
        let second_state = state.canonical_state_for_test();
        let third = input(
            &[6, 7, 8, 9, 10, 11],
            &[
                Some(9.0),
                Some(11.0),
                Some(13.0),
                Some(15.0),
                Some(17.0),
                Some(21.0),
            ],
        );
        let mut update = plan
            .prepare_dual_sma(&mut state, &third, "r", None)
            .unwrap()
            .unwrap();
        let columns = update.take_columns();
        assert_mean_bits(&means(&columns, 0), &[5.2, 7.0, 9.0, 11.0, 13.0, 15.4]);
        assert_mean_bits(
            &means(&columns, 1),
            &[4.5, 38.0 / 7.0, 6.375, 66.0 / 9.0, 8.3, 104.0 / 11.0],
        );
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), second_state);
    }
}
