//! State-table ownership for typed rolling stream transitions.

mod indicator;

use std::{sync::Arc, time::Instant};

use datafusion::arrow::array::{ArrayRef, Float64Array, Float64Builder, UInt64Builder};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;

use super::{
    DerivedBuilder, EntityRouting, LagValue, OutputStorage, Result, RollingKernelExecution,
    RollingKernelMetrics, RollingKernelPlan, RollingKernelState, RollingMetricsRecorder,
    RollingStage, RollingWork, StateCharge, Statistic, StreamKernelUpdate, TypedEwmaState,
    TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind, TypedOutputPlan,
    TypedWindowState, direct_entity_column, direct_entity_routing, internal_error, nanos,
    operator_error,
};
use crate::{
    accumulate::NumericAccumulator,
    state::{EntityId, EntityKey, StateTable, ValueKind},
};

pub(super) struct SlabUpdate {
    previous_charge: StateCharge,
    created: bool,
}

impl SlabUpdate {
    pub(in crate::operator::rolling) fn charge_change(
        &self,
        base: &RollingKernelState,
        node_id: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let new = charge(
            base.slab
                .as_ref()
                .ok_or_else(|| internal_error("rolling slab is absent"))?,
            node_id,
        )?;
        Ok((self.previous_charge, new))
    }

    pub(in crate::operator::rolling) fn rollback(self, base: &mut RollingKernelState) {
        base.slab
            .as_mut()
            .expect("rolling slab is present")
            .rollback()
            .expect("an active rolling slab transaction rolls back");
        if self.created {
            base.slab = None;
        }
    }

    #[allow(
        clippy::unused_self,
        reason = "consuming the update enforces that each slab transaction commits once"
    )]
    pub(in crate::operator::rolling) fn commit(self, base: &mut RollingKernelState) {
        base.slab
            .as_mut()
            .expect("rolling slab is present")
            .commit()
            .expect("an active rolling slab transaction commits");
    }
}

fn charge(table: &StateTable, node_id: &str) -> Result<StateCharge> {
    Ok(StateCharge {
        rows: u64::try_from(table.logical_rows())
            .map_err(|_| operator_error(node_id, "rolling slab row charge overflowed"))?,
        bytes: u64::try_from(table.logical_bytes())
            .map_err(|_| operator_error(node_id, "rolling slab byte charge overflowed"))?,
    })
}

fn batch_ewma_seed(
    groups: &[TypedGroupPlan],
    entities: &Vec<EntityId>,
    rows: usize,
) -> Option<TypedEwmaState> {
    let [
        TypedGroupPlan::Ewma {
            alpha, cumulative, ..
        },
    ] = groups
    else {
        return None;
    };
    // Keep both the resolved IDs and recurrence states within the former per-row ID budget.
    let remaining = rows
        .saturating_sub(entities.capacity())
        .saturating_mul(size_of::<EntityId>());
    (entities.len() <= remaining / size_of::<TypedEwmaState>())
        .then(|| TypedEwmaState::new(*alpha, *cumulative))
}

enum SlabBuilder {
    Float(Float64Builder),
    Count(UInt64Builder),
    Derived(DerivedBuilder),
}

impl SlabBuilder {
    fn new(output: &TypedOutputPlan, rows: usize) -> Result<Self> {
        match output.storage {
            OutputStorage::Float64 => Ok(Self::Float(Float64Builder::with_capacity(rows))),
            OutputStorage::Count => Ok(Self::Count(UInt64Builder::with_capacity(rows))),
            OutputStorage::Int8
            | OutputStorage::Int16
            | OutputStorage::Int32
            | OutputStorage::Int64
            | OutputStorage::UInt8
            | OutputStorage::UInt16
            | OutputStorage::UInt32
            | OutputStorage::UInt64
            | OutputStorage::Float32
            | OutputStorage::TimestampMicrosecond
            | OutputStorage::TimestampMicrosecondUtc
            | OutputStorage::Date32
            | OutputStorage::Date64
            | OutputStorage::Time64Microsecond
            | OutputStorage::DurationMicrosecond => Ok(Self::Derived(
                DerivedBuilder::new_with_inputs(*output, rows, &[])?,
            )),
            _ => Err(internal_error("rolling slab output storage mismatch")),
        }
    }

    fn append_row(
        &mut self,
        groups: &[TypedGroupPlan],
        output: &TypedOutputPlan,
        readouts: &indicator::RowReadouts<'_>,
        node_id: &str,
    ) -> Result<()> {
        if matches!(groups[output.group], TypedGroupPlan::Cumulative { .. }) {
            let summary = readouts
                .cumulative
                .as_ref()
                .expect("cumulative state is present")[output.group]
                .as_ref()
                .expect("cumulative group was updated");
            return self.append_cumulative(summary, output, node_id);
        }
        if matches!(
            groups[output.group],
            TypedGroupPlan::Numeric { .. } | TypedGroupPlan::Count { .. }
        ) {
            let numeric = TypedOutputPlan {
                group: indicator::numeric_ordinal(groups, output.group),
                ..*output
            };
            return self.append(readouts.accumulators, &numeric);
        }
        if let Some(values) = readouts.lag_values.as_ref() {
            let (reference, current) = values[output.group];
            return match output.kind {
                TypedOutputKind::Lag => self.append_lag(reference),
                TypedOutputKind::Delta => self.append_delta(current, reference, node_id),
                _ => Err(internal_error("rolling lag output kind mismatch")),
            };
        }
        let value = match output.kind {
            TypedOutputKind::Macd(component) => readouts.values[output.group][component],
            TypedOutputKind::Rsi | TypedOutputKind::Atr | TypedOutputKind::Lag => {
                readouts.values[output.group][0]
            }
            TypedOutputKind::Delta => match (
                readouts.values[output.group][1],
                readouts.values[output.group][0],
            ) {
                (Some(current), Some(reference)) => Some(current - reference),
                _ => None,
            },
            TypedOutputKind::Ewma
                if readouts
                    .valid_counts
                    .as_ref()
                    .expect("EWMA counts are present")[output.group]
                    >= output.min_periods =>
            {
                readouts.values[output.group][0]
            }
            TypedOutputKind::Ewma => None,
            _ => return Err(internal_error("rolling slab output kind mismatch")),
        };
        match self {
            Self::Float(builder) | Self::Derived(DerivedBuilder::Float(builder, _)) => {
                builder.append_option(value);
                Ok(())
            }
            Self::Count(_) | Self::Derived(_) => {
                Err(internal_error("rolling slab output storage mismatch"))
            }
        }
    }

    fn append_lag(&mut self, value: LagValue) -> Result<()> {
        match (self, value) {
            (Self::Derived(DerivedBuilder::Signed(builder, _)), LagValue::Signed(value)) => {
                builder.append_option(value);
                Ok(())
            }
            (Self::Derived(DerivedBuilder::Unsigned(builder, _)), LagValue::Unsigned(value)) => {
                builder.append_option(value);
                Ok(())
            }
            _ => Err(internal_error("rolling lag builder type mismatch")),
        }
    }

    fn append_delta(
        &mut self,
        current: LagValue,
        reference: LagValue,
        node_id: &str,
    ) -> Result<()> {
        let value = match (current, reference) {
            (LagValue::Signed(Some(current)), LagValue::Signed(Some(reference))) => current
                .checked_sub(reference)
                .map(|difference| LagValue::Signed(Some(difference)))
                .ok_or_else(|| {
                    operator_error(node_id, "rolling delta integer subtraction overflowed")
                })?,
            (LagValue::Unsigned(Some(current)), LagValue::Unsigned(Some(reference))) => current
                .checked_sub(reference)
                .map(|difference| LagValue::Unsigned(Some(difference)))
                .ok_or_else(|| {
                    operator_error(node_id, "rolling delta integer subtraction overflowed")
                })?,
            (LagValue::Signed(current), LagValue::Signed(reference)) => LagValue::Signed(
                current
                    .zip(reference)
                    .map(|(current, reference)| current - reference),
            ),
            (LagValue::Unsigned(current), LagValue::Unsigned(reference)) => LagValue::Unsigned(
                current
                    .zip(reference)
                    .map(|(current, reference)| current - reference),
            ),
            _ => return Err(internal_error("rolling delta input type mismatch")),
        };
        self.append_lag(value)
    }

    fn append_cumulative(
        &mut self,
        summary: &TypedWindowState,
        output: &TypedOutputPlan,
        node_id: &str,
    ) -> Result<()> {
        if let Self::Derived(builder) = self {
            return builder.append(
                std::slice::from_ref(summary),
                &TypedOutputPlan {
                    group: 0,
                    ..*output
                },
                node_id,
            );
        }
        let (Self::Float(builder), TypedWindowState::Cumulative(summary)) = (self, summary) else {
            return Err(internal_error("cumulative slab builder type mismatch"));
        };
        let value = if summary.accumulator.valid_count() < output.min_periods {
            None
        } else if matches!(output.kind, TypedOutputKind::Statistic(Statistic::Variance)) {
            summary.accumulator.variance(output.ddof)
        } else {
            let ScalarValue::Float64(value) = summary.accumulator.scalar(node_id)? else {
                return Err(internal_error("cumulative slab output type mismatch"));
            };
            value
        };
        builder.append_option(value);
        Ok(())
    }

    fn append(
        &mut self,
        accumulators: &[NumericAccumulator],
        output: &TypedOutputPlan,
    ) -> Result<()> {
        let accumulator = accumulators
            .get(output.group)
            .ok_or_else(|| internal_error("rolling slab output group is absent"))?;
        if accumulator.valid_count < output.min_periods {
            self.append_null();
            return Ok(());
        }
        match self {
            Self::Count(builder) => match output.kind {
                TypedOutputKind::Statistic(Statistic::Count) => {
                    builder.append_value(accumulator.valid_count);
                    Ok(())
                }
                _ => Err(internal_error("rolling slab output kind mismatch")),
            },
            Self::Float(builder) => Self::append_float(builder, accumulator, output),
            Self::Derived(_) => Err(internal_error("rolling slab output kind mismatch")),
        }
    }

    fn append_null(&mut self) {
        match self {
            Self::Float(builder) => builder.append_null(),
            Self::Count(builder) => builder.append_null(),
            Self::Derived(builder) => builder.append_null(),
        }
    }

    fn append_float(
        builder: &mut Float64Builder,
        accumulator: &NumericAccumulator,
        output: &TypedOutputPlan,
    ) -> Result<()> {
        match output.kind {
            TypedOutputKind::Statistic(Statistic::Sum) => {
                builder.append_value(accumulator.sum());
            }
            TypedOutputKind::Statistic(Statistic::Mean) => {
                if accumulator.valid_count > 0 {
                    builder.append_value(accumulator.mean());
                } else {
                    builder.append_null();
                }
            }
            TypedOutputKind::Statistic(Statistic::Variance) => {
                builder.append_value(accumulator.variance(output.ddof));
            }
            TypedOutputKind::Statistic(Statistic::Stddev) => {
                builder.append_value(accumulator.stddev(output.ddof));
            }
            _ => return Err(internal_error("rolling slab output kind mismatch")),
        }
        Ok(())
    }

    fn finish(self) -> Result<ArrayRef> {
        match self {
            Self::Float(mut builder) => Ok(Arc::new(builder.finish())),
            Self::Count(mut builder) => Ok(Arc::new(builder.finish())),
            Self::Derived(builder) => builder.finish(),
        }
    }
}

struct SlabInput {
    inputs: Vec<TypedGroupInput>,
    last_identity: Option<Vec<u8>>,
}

impl RollingKernelPlan {
    pub(in crate::operator::rolling) fn supports_slab(&self) -> bool {
        indicator::supports_plan(&self.groups, &self.outputs)
    }

    pub(in crate::operator::rolling) fn can_use_slab(&self, state: &RollingKernelState) -> bool {
        self.supports_slab() && (state.slab.is_some() || state.states.is_empty())
    }

    fn ensure_slab(&self, state: &mut RollingKernelState) {
        if state.slab.is_some() {
            return;
        }
        let mut table = StateTable::new();
        table.add_keyed_column("transition_count", ValueKind::UInt64);
        let mut windows = self.groups.iter().filter_map(|group| match group {
            TypedGroupPlan::Numeric {
                frame: TypedFrame::Rows(window),
                dispersion,
                ..
            } => Some((*window, *dispersion)),
            TypedGroupPlan::Count {
                frame: TypedFrame::Rows(window),
                ..
            } => Some((*window, false)),
            _ => None,
        });
        if let Some((first_window, first_dispersion)) = windows.next() {
            let column = table.add_numeric_rows_column("numeric", first_window);
            table.set_numeric_dispersion(column, 0, first_dispersion);
            for (window, dispersion) in windows {
                let group = table.add_numeric_window(column, window);
                table.set_numeric_dispersion(column, group, dispersion);
            }
        }
        if let Some(window) = self
            .groups
            .iter()
            .filter_map(|group| match group {
                TypedGroupPlan::Lag { periods, .. } => Some(*periods),
                _ => None,
            })
            .max()
        {
            let storage = self
                .groups
                .iter()
                .find_map(|group| match group {
                    TypedGroupPlan::Lag { storage, .. } => Some(*storage),
                    _ => None,
                })
                .unwrap_or(OutputStorage::Float64);
            if storage.is_signed() {
                table.add_i64_rows_column("lag", window.max(1));
            } else if storage.is_unsigned() {
                table.add_u64_rows_column("lag", window.max(1));
            } else {
                table.add_f64_rows_column("lag", window.max(1));
            }
        }
        indicator::add_columns(&mut table, &self.groups);
        state.slab = Some(table);
    }

    pub(in crate::operator::rolling) fn encode_slab_state(
        state: &RollingKernelState,
    ) -> Result<Vec<u8>> {
        let table = state
            .slab
            .as_ref()
            .ok_or_else(|| internal_error("rolling layout-7 state is absent"))?;
        crate::state::encode_layout7(table)
    }

    pub(in crate::operator::rolling) fn decode_slab_state(
        &self,
        bytes: &[u8],
        last_identity: Option<Vec<u8>>,
    ) -> Result<RollingKernelState> {
        let mut state = RollingKernelState::default();
        self.ensure_slab(&mut state);
        let table = state.slab.as_mut().expect("slab initialized");
        let snapshot = crate::state::decode_layout7(table, bytes)?;
        table.restore(snapshot)?;
        state.kernel_fingerprint = Some(self.fingerprint.clone());
        state.last_identity = last_identity;
        Ok(state)
    }

    pub(in crate::operator::rolling) fn slab_charge(
        state: &RollingKernelState,
        node_id: &str,
    ) -> Result<Option<StateCharge>> {
        state
            .slab
            .as_ref()
            .map(|table| charge(table, node_id))
            .transpose()
    }

    pub(in crate::operator::rolling) fn execution_charge(
        state: &RollingKernelState,
        node_id: &str,
    ) -> Result<StateCharge> {
        let mut total = if let Some(charge) = Self::slab_charge(state, node_id)? {
            charge
        } else {
            let mut total = StateCharge::default();
            for (key, &index) in &state.entities {
                total =
                    total.checked_add(state.states[index].charge(key.len(), node_id)?, node_id)?;
            }
            total
        };
        let entities = state
            .slab
            .as_ref()
            .map_or(state.states.len(), StateTable::entity_count);
        total.rows =
            total
                .rows
                .max(u64::try_from(entities).map_err(|_| {
                    operator_error(node_id, "rolling entity row charge overflowed")
                })?);
        Ok(total)
    }

    pub(in crate::operator::rolling) fn slab_last_identity(
        state: &RollingKernelState,
    ) -> Option<Vec<u8>> {
        state.slab.as_ref()?;
        state.last_identity.clone()
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_history_for_test(
        state: &RollingKernelState,
    ) -> Option<Vec<Option<f64>>> {
        state.slab.as_ref()?.first_f64_history(0)
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_histories_for_test(
        state: &RollingKernelState,
    ) -> Option<Vec<Vec<Option<f64>>>> {
        Some(state.slab.as_ref()?.all_f64_histories(0))
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_transition_counts_for_test(
        state: &RollingKernelState,
    ) -> Option<Vec<u64>> {
        Some(state.slab.as_ref()?.all_u64(0))
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_window_count_for_test(
        state: &RollingKernelState,
    ) -> Option<usize> {
        Some(state.slab.as_ref()?.numeric_window_count(0))
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_history_address_for_test(
        state: &RollingKernelState,
        offset: usize,
    ) -> Option<usize> {
        state.slab.as_ref()?.first_f64_slot_address(0, offset)
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_transition_count_for_test(
        state: &RollingKernelState,
    ) -> Option<u64> {
        state.slab.as_ref()?.first_u64(0)
    }

    #[cfg(test)]
    pub(in crate::operator::rolling) fn slab_slot_addresses_for_test(
        state: &RollingKernelState,
    ) -> Option<Vec<usize>> {
        Some(state.slab.as_ref()?.ring_slot_addresses(0))
    }

    fn slab_input(&self, input: &RecordBatch, node_id: &str) -> Result<SlabInput> {
        Ok(SlabInput {
            inputs: self.typed_inputs(input, node_id)?,
            last_identity: self.last_ordered_identity(input, node_id)?,
        })
    }

    fn route_slab(
        &self,
        state: &RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<(EntityRouting, u64)> {
        let started = Instant::now();
        let residents = state.slab.as_ref().map_or(0, StateTable::entity_count);
        let routing = if input.num_rows() > 1
            && let Some(keys) = direct_entity_column(self, input)
        {
            direct_entity_routing(keys, residents, node_id)?
        } else {
            self.encoded_entity_routing(input, residents, node_id)?
        };
        if let Some(recorder) = observer {
            recorder.add(RollingWork::ResolvedRows, routing.entity_ids.len());
            recorder.add(RollingWork::TouchedEntities, routing.keys.len());
        }
        Ok((routing, nanos(started.elapsed())))
    }

    fn prepare_slab_entities(
        table: &mut StateTable,
        routing: &EntityRouting,
        node_id: &str,
        track_positions: bool,
    ) -> Result<(Vec<EntityId>, Vec<u64>)> {
        let mut resolved = Vec::with_capacity(routing.keys.len());
        for key in &routing.keys {
            resolved.push(table.resolve_or_append(EntityKey::new(key.clone()))?);
        }
        let mut positions = Vec::with_capacity(if track_positions { resolved.len() } else { 0 });
        for (&entity, row) in resolved.iter().zip(&routing.rows) {
            let rows = u64::try_from(row.count)
                .map_err(|_| operator_error(node_id, "rolling entity row count overflowed"))?;
            let previous = table.u64(0, entity)?;
            if track_positions {
                positions.push(previous);
            }
            let transitions = previous.checked_add(rows).ok_or_else(|| {
                operator_error(node_id, "rolling entity transition count overflowed")
            })?;
            table.set_u64(0, entity, transitions)?;
        }
        Ok((resolved, positions))
    }

    fn append_slab_outputs(
        builders: &mut [SlabBuilder],
        outputs: &[TypedOutputPlan],
        accumulators: &[NumericAccumulator],
    ) -> Result<()> {
        for (builder, output) in builders.iter_mut().zip(outputs) {
            builder.append(accumulators, output)?;
        }
        Ok(())
    }

    fn write_numeric_slab_ticks(
        &self,
        table: &mut StateTable,
        resolved: &[EntityId],
        locals: &[usize],
        values: &Float64Array,
        builders: &mut [SlabBuilder],
    ) -> Result<bool> {
        match (builders, self.outputs.as_slice()) {
            (
                [SlabBuilder::Float(builder)],
                [
                    TypedOutputPlan {
                        group: 0,
                        kind: TypedOutputKind::Statistic(Statistic::Mean),
                        min_periods,
                        ..
                    },
                ],
            ) => {
                let minimum = (*min_periods).max(1);
                table.write_numeric_full_tick(
                    0,
                    resolved,
                    locals,
                    values,
                    self.nan_as_value,
                    |_, accumulators| {
                        let accumulator = accumulators
                            .first()
                            .ok_or_else(|| internal_error("rolling slab output group is absent"))?;
                        builder.append_option(
                            (accumulator.valid_count >= minimum).then(|| accumulator.mean()),
                        );
                        Ok(())
                    },
                )
            }
            (builders, outputs) => table.write_numeric_full_tick(
                0,
                resolved,
                locals,
                values,
                self.nan_as_value,
                |_, accumulators| Self::append_slab_outputs(builders, outputs, accumulators),
            ),
        }
    }

    fn append_slab_row(
        &self,
        table: &mut StateTable,
        position: indicator::EntityRow<'_>,
        inputs: &[TypedGroupInput],
        row: usize,
        builders: &mut [SlabBuilder],
    ) -> Result<()> {
        let readouts = indicator::update(
            table,
            position,
            &self.groups,
            inputs,
            row,
            self.nan_as_value,
        )?;
        for (builder, output) in builders.iter_mut().zip(&self.outputs) {
            builder.append_row(&self.groups, output, &readouts, position.node_id)?;
        }
        Ok(())
    }

    fn fill_slab_rows(
        &self,
        table: &mut StateTable,
        inputs: &[TypedGroupInput],
        routing: &EntityRouting,
        input_rows: usize,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Vec<ArrayRef>> {
        let track_positions = self.groups.iter().any(|group| {
            matches!(
                group,
                TypedGroupPlan::Rsi { .. } | TypedGroupPlan::Atr { .. }
            )
        });
        let (resolved, mut positions) =
            Self::prepare_slab_entities(table, routing, node_id, track_positions)?;
        let mut builders = self
            .outputs
            .iter()
            .map(|output| SlabBuilder::new(output, input_rows))
            .collect::<Result<Vec<_>>>()?;
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
        if let Some(seed) = batch_ewma_seed(&self.groups, &resolved, input_rows) {
            let [TypedGroupInput::Single(values)] = inputs else {
                return Err(internal_error("rolling EWMA slab input mismatch"));
            };
            indicator::fill_ewma(
                table,
                &resolved,
                &routing.entity_ids,
                values,
                seed,
                node_id,
                |state| {
                    for (builder, output) in builders.iter_mut().zip(&self.outputs) {
                        let SlabBuilder::Float(builder) = builder else {
                            return Err(internal_error("rolling EWMA slab output type mismatch"));
                        };
                        builder.append_option(
                            (state.valid_count >= output.min_periods).then_some(state.value),
                        );
                    }
                    Ok(())
                },
            )?;
        } else {
            let full_tick = if indicator::numeric_only(&self.groups) {
                let Some(TypedGroupInput::Single(values)) = inputs.first() else {
                    return Err(internal_error("rolling numeric slab input mismatch"));
                };
                self.write_numeric_slab_ticks(
                    table,
                    &resolved,
                    &routing.entity_ids,
                    values,
                    &mut builders,
                )?
            } else {
                false
            };
            if !full_tick {
                for (row, &local) in routing.entity_ids.iter().enumerate() {
                    let ordered_position = if track_positions { positions[local] } else { 0 };
                    if track_positions {
                        positions[local] += 1;
                    }
                    self.append_slab_row(
                        table,
                        indicator::EntityRow {
                            entity: resolved[local],
                            first: routing.rows[local].first == row,
                            ordered_position,
                            node_id,
                        },
                        inputs,
                        row,
                        &mut builders,
                    )?;
                }
            }
        }
        if let Some(recorder) = observer {
            recorder.add(RollingWork::NumericRows, input_rows);
            recorder.add(RollingWork::NumericSerialCalls, 1);
        }
        builders.into_iter().map(SlabBuilder::finish).collect()
    }

    fn finish_slab_columns(
        state: &mut RollingKernelState,
        created: bool,
        result: Result<Vec<ArrayRef>>,
    ) -> Result<Vec<ArrayRef>> {
        match result {
            Ok(columns) => Ok(columns),
            Err(error) => {
                state.slab.as_mut().expect("slab initialized").rollback()?;
                if created {
                    state.slab = None;
                }
                Err(error)
            }
        }
    }

    pub(in crate::operator::rolling) fn prepare_slab_ordered_stream(
        &self,
        state: &mut RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<StreamKernelUpdate> {
        self.validate_state(state, node_id)?;
        let created = state.slab.is_none();
        self.ensure_slab(state);
        let batch = self.slab_input(input, node_id)?;
        let (routing, entity_encode_ns) = self.route_slab(state, input, node_id, observer)?;
        let previous_charge = charge(state.slab.as_ref().expect("slab initialized"), node_id)?;
        state.slab.as_mut().expect("slab initialized").begin()?;
        let entities = routing.keys.len();
        let result = self.fill_slab_rows(
            state.slab.as_mut().expect("slab initialized"),
            &batch.inputs,
            &routing,
            input.num_rows(),
            node_id,
            observer,
        );
        let columns = Self::finish_slab_columns(state, created, result)?;
        // Routing reserves for resident entities; delivery only retains the touched subset.
        let mut entity_rows = routing.rows;
        entity_rows.shrink_to_fit();
        let local = RollingKernelState {
            kernel_fingerprint: Some(self.fingerprint.clone()),
            last_identity: batch.last_identity,
            ..RollingKernelState::default()
        };
        Ok(StreamKernelUpdate {
            execution: RollingKernelExecution {
                columns,
                entity_ids: routing.entity_ids,
                entity_rows,
                metrics: RollingKernelMetrics {
                    entity_encode_ns,
                    entities,
                    input_rows: input.num_rows(),
                    output_rows: input.num_rows(),
                    state_bytes: state.slab.as_ref().map_or(0, StateTable::logical_bytes),
                    ..RollingKernelMetrics::default()
                },
                state: local,
            },
            in_place: None,
            slab: Some(SlabUpdate {
                previous_charge,
                created,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{Array, Float64Array, StringArray, TimestampMicrosecondArray, UInt64Array},
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
            groups: vec![TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(2),
                dispersion: true,
            }],
            outputs: [
                (Statistic::Mean, OutputStorage::Float64),
                (Statistic::Sum, OutputStorage::Float64),
                (Statistic::Count, OutputStorage::Count),
                (Statistic::Variance, OutputStorage::Float64),
            ]
            .into_iter()
            .map(|(statistic, storage)| TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Statistic(statistic),
                storage,
                min_periods: 1,
                ddof: 0,
            })
            .collect(),
            fallback_reason: None,
            estimated_state_bytes_per_entity: 0,
            fingerprint: "slab-test".to_owned(),
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
                Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
                    "a",
                    times.len(),
                ))),
                Arc::new(Float64Array::from(values.to_vec())),
            ],
        )
        .unwrap()
    }

    #[test]
    fn partial_numeric_tick_uses_the_same_allocation_path_as_complete_ticks() {
        let mut plan = plan();
        plan.groups = vec![TypedGroupPlan::Numeric {
            input_index: 2,
            frame: TypedFrame::Rows(20),
            dispersion: false,
        }];
        plan.outputs.truncate(1);
        let entities = 64;
        let names: Vec<_> = (0..entities)
            .map(|index| format!("symbol-{index}"))
            .collect();
        let measure = |rows: i32| {
            let times: Vec<_> = (0..rows).map(i64::from).collect();
            let values: Vec<_> = (0..rows).map(|row| Some(f64::from(row))).collect();
            let fixture = input(&times, &values);
            let keys = names.iter().map(String::as_str).cycle().take(times.len());
            let mut columns = fixture.columns().to_vec();
            columns[1] = Arc::new(StringArray::from_iter_values(keys));
            let fixture = RecordBatch::try_new(fixture.schema(), columns).unwrap();
            let mut state = RollingKernelState::default();
            let mut update = None;
            let allocation = allocation_counter::measure(|| {
                update = Some(
                    plan.prepare_slab_ordered_stream(&mut state, &fixture, "r", None)
                        .unwrap(),
                );
            });
            let mut update = update.unwrap();
            let output = update.take_columns();
            let means = output[0].as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(means.len(), times.len());
            assert_eq!(means.null_count(), 0);
            for row in 0..rows {
                let oldest = row - (row / entities).min(19) * entities;
                let expected = f64::from(oldest + row) / 2.0;
                assert_eq!(
                    means.value(usize::try_from(row).unwrap()).to_bits(),
                    expected.to_bits()
                );
            }
            update.rollback(&mut state);
            assert!(state.slab.is_none());
            allocation
        };
        let complete = measure(9_984);
        let partial = measure(10_000);
        eprintln!("complete {complete:?}; partial {partial:?}");
        assert!(complete.count_total < 1_024);
        assert!(partial.count_total <= complete.count_total + 64);
    }

    #[test]
    fn test_numeric_slab_reuses_local_entity_indices_for_repeated_ticks() {
        let mut plan = plan();
        plan.outputs.truncate(1);
        let measure = |rows: usize| {
            let times = (0..i64::try_from(rows).unwrap()).collect::<Vec<_>>();
            let fixture = input(&times, &vec![Some(2.0); rows]);
            let mut columns = fixture.columns().to_vec();
            columns[1] = Arc::new(StringArray::from_iter_values(
                ["a", "b", "c", "d"].into_iter().cycle().take(rows),
            ));
            let fixture = RecordBatch::try_new(fixture.schema(), columns).unwrap();
            let mut state = RollingKernelState::default();
            plan.ensure_slab(&mut state);
            let inputs = plan.slab_input(&fixture, "r").unwrap();
            let (routing, _) = plan.route_slab(&state, &fixture, "r", None).unwrap();
            let table = state.slab.as_mut().unwrap();
            table.begin().unwrap();
            let mut result = None;
            let allocation = allocation_counter::measure(|| {
                result = Some(
                    plan.fill_slab_rows(table, &inputs.inputs, &routing, rows, "r", None)
                        .unwrap(),
                );
            });
            let result = result.unwrap();
            let means = result[0].as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(means.len(), rows);
            assert!(means.iter().all(|x| x == Some(2.0)));
            table.rollback().unwrap();
            allocation.bytes_total
        };
        let short = measure(4_096);
        let long = measure(32_768);
        // After routing, only the Float64 result and its validity grow with the row count.
        assert!(
            long - short <= (32_768 - 4_096) * 9,
            "four resident keys required {short} and {long} bytes after routing"
        );
    }

    fn average_plan() -> RollingKernelPlan {
        let mut plan = plan();
        plan.groups = vec![TypedGroupPlan::Ewma {
            input_index: 2,
            alpha: 0.0,
            cumulative: true,
        }];
        plan.outputs = [1, 3]
            .map(|min_periods| TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Ewma,
                storage: OutputStorage::Float64,
                min_periods,
                ddof: 0,
            })
            .into();
        plan
    }

    #[test]
    fn cumulative_mean_allocations_scale_with_entities_and_validate_each_readout() {
        let plan = average_plan();
        let measure = |rows: i32| {
            let times = (0..rows).map(i64::from).collect::<Vec<_>>();
            let values = (0..rows)
                .map(|row| match row % 16 {
                    6 => None,
                    9 => Some(f64::NAN),
                    _ => Some(f64::from(row)),
                })
                .collect::<Vec<_>>();
            let record = input(&times, &values);
            let mut columns = record.columns().to_vec();
            columns[1] = Arc::new(StringArray::from_iter_values(
                ["a", "b", "c", "d"].into_iter().cycle().take(times.len()),
            ));
            let record = RecordBatch::try_new(record.schema(), columns).unwrap();
            let mut state = RollingKernelState::default();
            let mut update = None;
            let counts = allocation_counter::measure(|| {
                update = Some(
                    plan.prepare_slab_ordered_stream(&mut state, &record, "r", None)
                        .unwrap(),
                );
            });
            let mut update = update.unwrap();
            let columns = update.take_columns();
            for (column, minimum) in columns.iter().zip([1, 3]) {
                let means = column.as_any().downcast_ref::<Float64Array>().unwrap();
                let mut groups = [(0_u32, 0.0); 4];
                for (row, &value) in values.iter().enumerate() {
                    let (count, sum) = &mut groups[row % groups.len()];
                    if let Some(value) = value.filter(|value| !value.is_nan()) {
                        *count += 1;
                        *sum += value;
                    }
                    assert_eq!(means.is_valid(row), *count >= minimum);
                    if *count >= minimum {
                        assert!((means.value(row) - *sum / f64::from(*count)).abs() < 1e-10);
                    }
                }
            }
            update.rollback(&mut state);
            assert!(state.slab.is_none());
            counts.count_total
        };
        let short = measure(256);
        let long = measure(4_096);
        eprintln!("four entities, 256/4,096 rows: {short}/{long} allocations");
        assert!(
            long <= short + 32,
            "same four entities required {short} allocations for 256 rows and {long} for 4096"
        );
    }

    #[test]
    fn average_scratch_admission_includes_live_entity_ids() {
        let mut table = super::StateTable::new();
        table.begin().unwrap();
        let first = table
            .resolve_or_append(super::EntityKey::new(vec![0]))
            .unwrap();
        let second = table
            .resolve_or_append(super::EntityKey::new(vec![1]))
            .unwrap();
        let mut resolved = Vec::with_capacity(8);
        resolved.extend([first, second]);
        let minimum_rows = resolved.capacity()
            + (resolved.len() * size_of::<super::TypedEwmaState>())
                .div_ceil(size_of::<super::EntityId>());
        let average = average_plan();
        assert!(super::batch_ewma_seed(&average.groups, &resolved, minimum_rows - 1).is_none());
        assert!(super::batch_ewma_seed(&plan().groups, &resolved, minimum_rows).is_none());
        assert!(super::batch_ewma_seed(&average.groups, &resolved, minimum_rows).is_some());
    }

    #[test]
    fn cumulative_mean_overflow_mid_batch_preserves_committed_state() {
        let plan = average_plan();
        let mut state = RollingKernelState::default();
        let record = input(&[1, 2], &[Some(1.0), Some(2.0)]);
        let mut columns = record.columns().to_vec();
        columns[1] = Arc::new(StringArray::from(vec!["a", "b"]));
        let seed = RecordBatch::try_new(record.schema(), columns).unwrap();
        plan.prepare_slab_ordered_stream(&mut state, &seed, "r", None)
            .unwrap()
            .commit(&mut state);
        let (mut routing, _) = plan.route_slab(&state, &seed, "r", None).unwrap();
        let table = state.slab.as_mut().unwrap();
        table.begin().unwrap();
        let b = table
            .resolve_or_append(super::EntityKey::new(routing.keys.pop().unwrap()))
            .unwrap();
        table.set_u64(1, b, u64::MAX).unwrap();
        table.commit().unwrap();
        let committed = RollingKernelPlan::encode_slab_state(&state).unwrap();

        let mut state = plan
            .decode_slab_state(&committed, RollingKernelPlan::slab_last_identity(&state))
            .unwrap();
        let times = (3..19).collect::<Vec<_>>();
        let record = input(&times, &[Some(5.0); 16]);
        let mut columns = record.columns().to_vec();
        columns[1] = Arc::new(StringArray::from_iter_values(
            (0..times.len()).map(|row| if row == 3 { "b" } else { "a" }),
        ));
        let batch = RecordBatch::try_new(record.schema(), columns).unwrap();
        let Err(error) = plan.prepare_slab_ordered_stream(&mut state, &batch, "r", None) else {
            panic!("the second entity has exhausted its valid sample count");
        };
        assert!(error.to_string().contains("EWMA sample count overflowed"));
        assert_eq!(
            RollingKernelPlan::encode_slab_state(&state).unwrap(),
            committed
        );
    }

    #[test]
    fn kernel_slab_commit_and_rollback_preserve_aggregates_and_history() {
        let plan = plan();
        let mut state = RollingKernelState::default();
        let empty = state.canonical_state_for_test();
        let first = input(&[1, 2], &[Some(1.0), Some(3.0)]);
        let mut update = plan
            .prepare_slab_ordered_stream(&mut state, &first, "r", None)
            .unwrap();
        let columns = update.take_columns();
        let mean = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(mean.value(0).to_bits(), 1.0_f64.to_bits());
        assert_eq!(mean.value(1).to_bits(), 2.0_f64.to_bits());
        let sum = columns[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(sum.value(1).to_bits(), 4.0_f64.to_bits());
        let count = columns[2].as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(count.value(1), 2);
        let variance = columns[3].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(variance.value(1).to_bits(), 1.0_f64.to_bits());
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), empty);

        let update = plan
            .prepare_slab_ordered_stream(&mut state, &first, "r", None)
            .unwrap();
        update.commit(&mut state);
        let committed = state.canonical_state_for_test();
        let second = input(&[3], &[Some(5.0)]);
        let mut update = plan
            .prepare_slab_ordered_stream(&mut state, &second, "r", None)
            .unwrap();
        let columns = update.take_columns();
        let mean = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(mean.value(0).to_bits(), 4.0_f64.to_bits());
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), committed);
    }

    #[test]
    fn sparse_slab_update_retains_only_touched_entity_statistics() {
        const RESIDENTS: usize = 128;
        let plan = plan();
        let mut state = RollingKernelState::default();
        let seed = input(&[1; RESIDENTS], &[Some(1.0); RESIDENTS]);
        let mut columns = seed.columns().to_vec();
        columns[1] = Arc::new(StringArray::from_iter_values(
            (0..RESIDENTS).map(|id| format!("key-{id:03}")),
        ));
        let seed = RecordBatch::try_new(seed.schema(), columns).unwrap();
        plan.prepare_slab_ordered_stream(&mut state, &seed, "r", None)
            .unwrap()
            .commit(&mut state);
        let before = state.canonical_state_for_test();

        let times = (2..).take(RESIDENTS).collect::<Vec<_>>();
        let sparse = input(&times, &[Some(3.0); RESIDENTS]);
        let mut columns = sparse.columns().to_vec();
        columns[1] = Arc::new(StringArray::from_iter_values(std::iter::repeat_n(
            "key-064", RESIDENTS,
        )));
        let sparse = RecordBatch::try_new(sparse.schema(), columns).unwrap();
        let mut update = plan
            .prepare_slab_ordered_stream(&mut state, &sparse, "r", None)
            .unwrap();
        let rows = update.entity_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!((rows[0].count, rows[0].first), (RESIDENTS, 0));
        let capacity = rows.capacity();
        assert_eq!(update.entity_ids(), [0; RESIDENTS]);
        let output = update.take_columns();
        let sums = output[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(sums.value(0).to_bits(), 4.0_f64.to_bits());
        assert!(
            sums.values()[1..]
                .iter()
                .all(|value| value.to_bits() == 6.0_f64.to_bits())
        );
        update.rollback(&mut state);
        assert_eq!(state.canonical_state_for_test(), before);
        assert_eq!(
            capacity, 1,
            "prepared capacities after touching one resident"
        );
    }

    #[test]
    fn multi_window_slab_shares_history_and_restores_every_accumulator() {
        let mut plan = plan();
        plan.groups = vec![
            TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(2),
                dispersion: true,
            },
            TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(4),
                dispersion: true,
            },
        ];
        plan.outputs = vec![
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
        ];
        let mut state = RollingKernelState::default();
        let first = input(&[1, 2, 3, 4], &[Some(1.0), Some(3.0), Some(5.0), Some(7.0)]);
        let update = plan
            .prepare_slab_ordered_stream(&mut state, &first, "r", None)
            .unwrap();
        update.commit(&mut state);
        assert_eq!(
            RollingKernelPlan::slab_history_for_test(&state)
                .unwrap()
                .len(),
            4,
            "dual windows must retain one shared history"
        );

        let encoded = RollingKernelPlan::encode_slab_state(&state).unwrap();
        let mut restored = plan
            .decode_slab_state(&encoded, RollingKernelPlan::slab_last_identity(&state))
            .unwrap();
        assert_eq!(
            restored.canonical_state_for_test(),
            state.canonical_state_for_test()
        );
        let second = input(&[5], &[Some(9.0)]);
        let mut uninterrupted = state.clone();
        for current in [&mut state, &mut restored] {
            let update = plan
                .prepare_slab_ordered_stream(current, &second, "r", None)
                .unwrap();
            update.commit(current);
        }
        assert_eq!(
            restored.canonical_state_for_test(),
            state.canonical_state_for_test()
        );
        let update = plan
            .prepare_slab_ordered_stream(&mut uninterrupted, &second, "r", None)
            .unwrap();
        let columns = update.execution.columns;
        let fast = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        let slow = columns[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(fast.value(0).to_bits(), 8.0_f64.to_bits());
        assert_eq!(slow.value(0).to_bits(), 6.0_f64.to_bits());
    }

    #[test]
    fn narrow_window_rebuild_uses_only_its_own_history() {
        let mut plan = plan();
        plan.groups = vec![
            TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(2),
                dispersion: true,
            },
            TypedGroupPlan::Numeric {
                input_index: 2,
                frame: TypedFrame::Rows(4),
                dispersion: true,
            },
        ];
        plan.outputs = [0, 1]
            .into_iter()
            .map(|group| TypedOutputPlan {
                group,
                kind: TypedOutputKind::Statistic(Statistic::Mean),
                storage: OutputStorage::Float64,
                min_periods: 1,
                ddof: 0,
            })
            .collect();
        let times = (1..=64).collect::<Vec<_>>();
        let values = (1..=64)
            .map(|value| Some(f64::from(value)))
            .collect::<Vec<_>>();
        let mut state = RollingKernelState::default();
        let update = plan
            .prepare_slab_ordered_stream(&mut state, &input(&times, &values), "r", None)
            .unwrap();
        let columns = update.execution.columns;
        let narrow = columns[0].as_any().downcast_ref::<Float64Array>().unwrap();
        let wide = columns[1].as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(narrow.value(63).to_bits(), 63.5_f64.to_bits());
        assert_eq!(wide.value(63).to_bits(), 62.5_f64.to_bits());
    }

    #[test]
    fn slab_encodes_each_distinct_utf8_key_once_on_cold_and_warm_batches() {
        let plan = plan();
        let mut state = RollingKernelState::default();
        for start in [0, 6] {
            let times = (start..start + 6).collect::<Vec<_>>();
            let seed = input(&times, &[Some(1.0); 6]);
            let mut columns = seed.columns().to_vec();
            columns[1] = Arc::new(StringArray::from(vec!["é", "a", "", "a", "é", ""]));
            let batch = RecordBatch::try_new(seed.schema(), columns).unwrap();
            let expected = plan.encoded_entity_routing(&batch, 3, "r").unwrap();
            super::super::take_direct_entity_key_encodings();
            let (routing, _) = plan.route_slab(&state, &batch, "r", None).unwrap();
            assert_eq!(super::super::take_direct_entity_key_encodings(), 3);
            assert_eq!(routing.keys, expected.keys);
            assert_eq!(routing.rows, expected.rows);
            assert_eq!(routing.entity_ids, expected.entity_ids);
            plan.prepare_slab_ordered_stream(&mut state, &batch, "r", None)
                .unwrap()
                .commit(&mut state);
        }
    }

    #[test]
    fn warm_slab_routes_uniform_utf8_without_encoding_every_row() {
        const ROWS: usize = 64_000;
        let plan = plan();
        let mut state = RollingKernelState::default();
        plan.prepare_slab_ordered_stream(&mut state, &input(&[0], &[Some(1.0)]), "r", None)
            .unwrap()
            .commit(&mut state);
        let fixture = input(&vec![1; ROWS], &vec![Some(1.0); ROWS]);
        let measure = |batch: &RecordBatch| {
            let expected = plan.encoded_entity_routing(batch, 16, "r").unwrap();
            let mut actual = None;
            let usage = allocation_counter::measure(|| {
                actual = Some(plan.route_slab(&state, batch, "r", None).unwrap().0);
            });
            let actual = actual.unwrap();
            assert_eq!(actual.keys, expected.keys);
            assert_eq!(actual.rows, expected.rows);
            assert_eq!(actual.entity_ids, expected.entity_ids);
            usage
        };
        let uniform = measure(&fixture);
        let mut columns = fixture.columns().to_vec();
        columns[1] = Arc::new(StringArray::from_iter_values(
            (0..ROWS).map(|row| if row == ROWS / 2 { "é" } else { "a" }),
        ));
        let mixed = measure(&RecordBatch::try_new(fixture.schema(), columns).unwrap());
        eprintln!("uniform {uniform:?}; mixed {mixed:?}");
        assert!(uniform.bytes_total <= u64::try_from(ROWS * size_of::<usize>() + 4096).unwrap());
    }

    #[test]
    fn restored_slab_resolves_a_new_uniform_key_before_continuing_the_original() {
        let plan = plan();
        let mut state = RollingKernelState::default();
        plan.prepare_slab_ordered_stream(&mut state, &input(&[0], &[Some(3.0)]), "r", None)
            .unwrap()
            .commit(&mut state);
        let encoded = RollingKernelPlan::encode_slab_state(&state).unwrap();
        let mut restored = plan
            .decode_slab_state(&encoded, RollingKernelPlan::slab_last_identity(&state))
            .unwrap();
        for (start, key, expected) in [(1, "é", 9.0_f64), (3, "a", 6.0_f64)] {
            let fixture = input(&[start, start + 1], &[Some(9.0), Some(11.0)]);
            let mut columns = fixture.columns().to_vec();
            columns[1] = Arc::new(StringArray::from(vec![key, key]));
            let batch = RecordBatch::try_new(fixture.schema(), columns).unwrap();
            let mut update = plan
                .prepare_slab_ordered_stream(&mut restored, &batch, "r", None)
                .unwrap();
            let output = update.take_columns();
            let means = output[0].as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(means.value(0).to_bits(), expected.to_bits());
            assert_eq!(means.value(1).to_bits(), 10.0_f64.to_bits());
            update.commit(&mut restored);
        }
    }
}
