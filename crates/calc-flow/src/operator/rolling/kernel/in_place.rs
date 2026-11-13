use std::{sync::Arc, time::Instant};

use datafusion::arrow::{
    array::{ArrayRef, StringArray, TimestampMicrosecondArray},
    record_batch::RecordBatch,
};

use super::{
    Array, DerivedBuilder, EntityRouting, Float64NumericState, RecurrenceSeed,
    RollingKernelExecution, RollingKernelMetrics, RollingKernelPlan, RollingKernelState,
    RollingMetricsRecorder, RollingStage, RollingWork, StateCharge, StreamKernelUpdate,
    TimedSample, TypedEntityState, TypedFrame, TypedGroupInput, TypedRowInputs, TypedWindowState,
    WindowAccumulator, add_float64, dense_entity_ids, encode_arrays, internal_error, nanos,
    operator_error, remove_float64, valid_float64,
};
use crate::Result;

type DirectRouting<'a> = (EntityRouting, Vec<Option<(usize, usize)>>, Vec<&'a str>);

pub(super) struct InPlaceUpdate {
    residents: Vec<usize>,
    key_bytes: Vec<usize>,
    new_keys: Vec<(usize, Vec<u8>)>,
    new_direct_keys: Vec<(usize, String)>,
    undo: Vec<EntityUndo>,
    previous_charge: StateCharge,
}

struct EntityUndo {
    previous: Option<TypedEntityUndo>,
    is_new: bool,
}

struct TypedEntityUndo {
    transition_count: u64,
    groups: NumericGroupsUndo,
}

enum NumericGroupsUndo {
    One(Float64NumericUndo),
    Many(Vec<Float64NumericUndo>),
}

impl NumericGroupsUndo {
    fn len(&self) -> usize {
        match self {
            Self::One(_) => 1,
            Self::Many(groups) => groups.len(),
        }
    }

    fn as_mut_slice(&mut self) -> &mut [Float64NumericUndo] {
        match self {
            Self::One(group) => std::slice::from_mut(group),
            Self::Many(groups) => groups.as_mut_slice(),
        }
    }
}

struct Float64NumericUndo {
    accumulator: WindowAccumulator,
    original_remaining: usize,
    original_capacity: usize,
    evicted: EvictedSamples,
}

enum EvictedSamples {
    None,
    One(TimedSample<f64>),
    Many(Vec<TimedSample<f64>>),
}

impl EvictedSamples {
    fn push(&mut self, sample: TimedSample<f64>) {
        match self {
            Self::None => *self = Self::One(sample),
            Self::One(first) => {
                *self = Self::Many(vec![*first, sample]);
            }
            Self::Many(samples) => samples.push(sample),
        }
    }

    fn restore(self, values: &mut std::collections::VecDeque<TimedSample<f64>>) {
        match self {
            Self::None => {}
            Self::One(sample) => values.push_front(sample),
            Self::Many(samples) => {
                for sample in samples.into_iter().rev() {
                    values.push_front(sample);
                }
            }
        }
    }
}

impl Float64NumericUndo {
    fn new(state: &Float64NumericState) -> Self {
        Self {
            accumulator: state.accumulator,
            original_remaining: state.samples.len(),
            original_capacity: state.samples.values.capacity(),
            evicted: EvictedSamples::None,
        }
    }

    fn rollback(self, state: &mut Float64NumericState) {
        state.accumulator = self.accumulator;
        state.samples.values.truncate(self.original_remaining);
        self.evicted.restore(&mut state.samples.values);
        state.samples.values.shrink_to(self.original_capacity);
    }
}

impl TypedEntityUndo {
    fn new(entity: &TypedEntityState) -> Result<Self> {
        fn snapshot(group: &TypedWindowState) -> Result<Float64NumericUndo> {
            match group {
                TypedWindowState::Numeric(state) => Ok(Float64NumericUndo::new(state)),
                _ => Err(internal_error(
                    "in-place rolling state does not match its numeric plan",
                )),
            }
        }

        let groups = match entity.groups.as_slice() {
            [group] => NumericGroupsUndo::One(snapshot(group)?),
            groups => {
                NumericGroupsUndo::Many(groups.iter().map(snapshot).collect::<Result<Vec<_>>>()?)
            }
        };
        Ok(Self {
            transition_count: entity.transition_count,
            groups,
        })
    }

    fn rollback(self, entity: &mut TypedEntityState) {
        entity.transition_count = self.transition_count;
        match self.groups {
            NumericGroupsUndo::One(undo) => {
                if let TypedWindowState::Numeric(state) = &mut entity.groups[0] {
                    undo.rollback(state);
                }
            }
            NumericGroupsUndo::Many(undos) => {
                for (group, undo) in entity.groups.iter_mut().zip(undos) {
                    if let TypedWindowState::Numeric(state) = group {
                        undo.rollback(state);
                    }
                }
            }
        }
    }
}

struct ResidentInspection {
    residents: Vec<usize>,
    key_bytes: Vec<usize>,
    undo: Vec<EntityUndo>,
    previous_charge: StateCharge,
}

enum ResidentMatch {
    Existing {
        resident: usize,
        key_bytes: usize,
        charge: StateCharge,
        undo: TypedEntityUndo,
    },
    New,
}

fn inspect_resident(
    state: &mut RollingKernelState,
    key: &[u8],
    known: Option<(usize, usize)>,
    direct_name: Option<&str>,
    node_id: &str,
) -> Result<Option<ResidentMatch>> {
    let existing = known.or_else(|| {
        state
            .entities
            .get(key)
            .map(|&resident| (resident, key.len()))
    });
    let Some((resident, key_bytes)) = existing else {
        return Ok(Some(ResidentMatch::New));
    };
    let entity = &state.states[resident];
    if Arc::strong_count(entity) != 1 {
        return Ok(None);
    }
    let charge = entity.charge(key_bytes, node_id)?;
    if known.is_none()
        && let Some(name) = direct_name
    {
        state
            .direct_entities
            .insert(name.to_owned(), (resident, key_bytes));
    }
    Ok(Some(ResidentMatch::Existing {
        resident,
        key_bytes,
        charge,
        undo: TypedEntityUndo::new(entity)?,
    }))
}

fn inspect_residents(
    state: &mut RollingKernelState,
    keys: &[Vec<u8>],
    known_residents: &[Option<(usize, usize)>],
    direct_names: &[&str],
    node_id: &str,
) -> Result<Option<ResidentInspection>> {
    let mut previous_charge = StateCharge::default();
    let mut residents = Vec::with_capacity(keys.len());
    let mut key_bytes = Vec::with_capacity(keys.len());
    let mut undo = Vec::with_capacity(keys.len());
    for (local, key) in keys.iter().enumerate() {
        let known = known_residents.get(local).copied().flatten();
        match inspect_resident(state, key, known, direct_names.get(local).copied(), node_id)? {
            Some(ResidentMatch::Existing {
                resident,
                key_bytes: encoded_bytes,
                charge,
                undo: previous,
            }) => {
                previous_charge = previous_charge.checked_add(charge, node_id)?;
                residents.push(resident);
                key_bytes.push(encoded_bytes);
                undo.push(EntityUndo {
                    previous: Some(previous),
                    is_new: false,
                });
            }
            Some(ResidentMatch::New) => {
                residents.push(usize::MAX);
                key_bytes.push(key.len());
                undo.push(EntityUndo {
                    previous: None,
                    is_new: true,
                });
            }
            None => return Ok(None),
        }
    }
    Ok(Some(ResidentInspection {
        residents,
        key_bytes,
        undo,
        previous_charge,
    }))
}

impl InPlaceUpdate {
    fn resolve(
        plan: &RollingKernelPlan,
        state: &mut RollingKernelState,
        routing: EntityRouting,
        known_residents: &[Option<(usize, usize)>],
        direct_names: &[&str],
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Option<(RollingKernelState, Self)>> {
        let EntityRouting {
            keys,
            rows,
            entity_ids,
        } = routing;
        if let Some(recorder) = observer {
            recorder.add(RollingWork::ResolvedRows, entity_ids.len());
            recorder.add(RollingWork::TouchedEntities, keys.len());
        }
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::StatePreparation));
        let Some(ResidentInspection {
            mut residents,
            key_bytes,
            mut undo,
            previous_charge,
        }) = inspect_residents(state, &keys, known_residents, direct_names, node_id)?
        else {
            return Ok(None);
        };
        let mut new_keys = Vec::new();
        let mut new_direct_keys = Vec::new();
        for (local, key) in keys.into_iter().enumerate() {
            if !undo[local].is_new {
                continue;
            }
            let resident = state.states.len();
            state.entities.insert(key.clone(), resident);
            state.states.push(Arc::new(TypedEntityState::new(
                &plan.groups,
                rows.counts[local],
            )));
            residents[local] = resident;
            undo[local].previous = Some(
                TypedEntityUndo::new(&state.states[resident])
                    .expect("a new numeric entity matches its plan"),
            );
            if let Some(name) = direct_names.get(local) {
                let name = (*name).to_owned();
                state
                    .direct_entities
                    .insert(name.clone(), (resident, key_bytes[local]));
                new_direct_keys.push((resident, name));
            }
            new_keys.push((resident, key));
        }
        Ok(Some((
            RollingKernelState {
                kernel_fingerprint: Some(plan.fingerprint.clone()),
                ..RollingKernelState::default()
            },
            Self {
                residents,
                key_bytes,
                new_keys,
                new_direct_keys,
                undo,
                previous_charge,
            },
        )))
    }

    pub(super) fn recurrence_seeds(&self) -> Vec<Vec<Option<RecurrenceSeed>>> {
        let groups = self
            .undo
            .first()
            .and_then(|entity| entity.previous.as_ref())
            .map_or(0, |state| state.groups.len());
        vec![vec![None; groups]; self.residents.len()]
    }

    pub(super) fn charge_change(
        &self,
        base: &RollingKernelState,
        node_id: &str,
    ) -> Result<(StateCharge, StateCharge)> {
        let mut new = StateCharge::default();
        for (&resident, &key_bytes) in self.residents.iter().zip(&self.key_bytes) {
            let next = base.states[resident].charge(key_bytes, node_id)?;
            new = new.checked_add(next, node_id)?;
        }
        Ok((self.previous_charge, new))
    }

    pub(super) fn rollback(&mut self, base: &mut RollingKernelState) {
        for local_index in (0..self.residents.len()).rev() {
            let resident = self.residents[local_index];
            let undo = &mut self.undo[local_index];
            if !undo.is_new {
                undo.previous
                    .take()
                    .expect("resident numeric entity has an undo snapshot")
                    .rollback(
                        Arc::get_mut(&mut base.states[resident])
                            .expect("in-place rolling entities remain exclusive"),
                    );
                continue;
            }
            let (key_resident, key) = self
                .new_keys
                .pop()
                .expect("in-place rolling new entity has a resident key");
            debug_assert_eq!(key_resident, resident);
            base.entities.remove(key.as_slice());
            if let Some((direct_resident, name)) = self.new_direct_keys.pop() {
                debug_assert_eq!(direct_resident, resident);
                base.direct_entities.remove(&name);
            }
            debug_assert_eq!(resident, base.states.len() - 1);
            base.states.pop();
        }
    }
}

fn route_direct_entities<'a>(
    state: &RollingKernelState,
    entities: &'a StringArray,
    node_id: &str,
) -> Result<DirectRouting<'a>> {
    let values = (0..entities.len()).map(|row| entities.value(row));
    let (names, rows, entity_ids) = dense_entity_ids(values, state.states.len().max(16));
    let mut known = Vec::with_capacity(names.len());
    let mut keys = vec![Vec::new(); names.len()];
    let mut misses = Vec::new();
    for (local, name) in names.iter().copied().enumerate() {
        let resident = state.direct_entities.get(name).copied();
        known.push(resident);
        if resident.is_none() {
            misses.push((local, name));
        }
    }
    if !misses.is_empty() {
        let distinct: ArrayRef = Arc::new(StringArray::from_iter_values(
            misses.iter().map(|(_, name)| *name),
        ));
        let encoded = encode_arrays(&[distinct], node_id)?;
        for (row, (local, _)) in misses.iter().enumerate() {
            keys[*local] = encoded.row(row).data().to_vec();
        }
        #[cfg(test)]
        super::record_direct_entity_key_encodings(misses.len());
    }
    Ok((
        EntityRouting {
            keys,
            rows,
            entity_ids,
        },
        known,
        names,
    ))
}

impl RollingKernelPlan {
    pub(in crate::operator::rolling) fn supports_inplace_ordered_stream(
        &self,
        input: &RecordBatch,
    ) -> bool {
        input.num_rows() < 64_000
            && self.supports_deferred_history()
            && self.shared_rows.is_none()
            && super::mean::MeanKernel::compile(self).is_none()
    }

    fn route_in_place<'a>(
        &self,
        state: &RollingKernelState,
        input: &'a RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<DirectRouting<'a>> {
        let _stage = observer.map(|recorder| recorder.stage(RollingStage::EntityResolution));
        match super::direct_entity_column(self, input) {
            Some(entities) => route_direct_entities(state, entities, node_id),
            None => Ok((
                self.encoded_entity_routing(input, state.states.len().max(16), node_id)?,
                Vec::new(),
                Vec::new(),
            )),
        }
    }

    fn ordered_event_times(
        &self,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<TimestampMicrosecondArray> {
        Ok(input
            .column(self.event_time_index)
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| {
                operator_error(
                    node_id,
                    "rolling event-time value is not a microsecond timestamp",
                )
            })?
            .clone())
    }

    pub(in crate::operator::rolling) fn prepare_ordered_stream_in_place(
        &self,
        state: &mut RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<Option<StreamKernelUpdate>> {
        self.validate_state(state, node_id)?;
        let columns = self.typed_inputs(input, node_id)?;
        let event_times = self.ordered_event_times(input, node_id)?;
        let last_identity = self.last_ordered_identity(input, node_id)?;
        let started = Instant::now();
        let (routing, known_residents, direct_names) =
            self.route_in_place(state, input, node_id, observer)?;
        let entity_rows = routing.rows.clone();
        let entity_ids = routing.entity_ids.clone();
        let Some((mut local, mut in_place)) = InPlaceUpdate::resolve(
            self,
            state,
            routing,
            &known_residents,
            &direct_names,
            node_id,
            observer,
        )?
        else {
            return Ok(None);
        };
        local.last_identity = last_identity;
        let entity_encode_ns = nanos(started.elapsed());
        let inputs = TypedRowInputs {
            columns: &columns,
            event_times: &event_times,
            entity_ids: &entity_ids,
        };
        let mut builders = self
            .outputs
            .iter()
            .map(|output| DerivedBuilder::new(*output, input.num_rows()))
            .collect::<Vec<_>>();
        let kernel_start = Instant::now();
        let numeric_result = {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::NumericUpdate));
            fill_rows(
                self,
                inputs,
                state,
                &mut in_place,
                &mut builders,
                node_id,
                observer,
            )
        };
        let kernel_ns = nanos(kernel_start.elapsed());
        let output_start = Instant::now();
        let result = numeric_result.and_then(|()| {
            let _stage = observer.map(|recorder| recorder.stage(RollingStage::ArrowOutput));
            builders
                .into_iter()
                .map(DerivedBuilder::finish)
                .collect::<Result<Vec<_>>>()
        });
        let output_build_ns = nanos(output_start.elapsed());
        let state_bytes = in_place
            .residents
            .iter()
            .map(|&resident| state.states[resident].estimated_bytes())
            .sum();
        let update = StreamKernelUpdate {
            execution: RollingKernelExecution {
                columns: Vec::new(),
                entity_ids,
                entity_rows,
                metrics: RollingKernelMetrics {
                    entity_encode_ns,
                    kernel_ns,
                    output_build_ns,
                    entities: in_place.residents.len(),
                    input_rows: input.num_rows(),
                    output_rows: input.num_rows(),
                    state_bytes,
                    ..RollingKernelMetrics::default()
                },
                state: local,
            },
            in_place: Some(in_place),
        };
        complete_in_place_update(update, state, result)
    }
}

fn complete_in_place_update(
    mut update: StreamKernelUpdate,
    state: &mut RollingKernelState,
    result: Result<Vec<ArrayRef>>,
) -> Result<Option<StreamKernelUpdate>> {
    match result {
        Ok(columns) => {
            update.execution.columns = columns;
            Ok(Some(update))
        }
        Err(error) => {
            update.rollback(state);
            Err(error)
        }
    }
}

fn update_entity_groups_reversible(
    plan: &RollingKernelPlan,
    inputs: TypedRowInputs<'_>,
    row_index: usize,
    entity: &mut TypedEntityState,
    previous: &mut TypedEntityUndo,
    node_id: &str,
) -> Result<()> {
    for ((group, input), undo) in entity
        .groups
        .iter_mut()
        .zip(inputs.columns)
        .zip(previous.groups.as_mut_slice())
    {
        let (TypedWindowState::Numeric(group), TypedGroupInput::Single(input)) = (group, input)
        else {
            return Err(internal_error(
                "in-place rolling group state does not match its input plan",
            ));
        };
        group.update_reversible(
            inputs.event_times.value(row_index),
            valid_float64(input, row_index, plan.nan_as_value),
            plan.numerical_profile,
            entity.transition_count,
            undo,
            node_id,
        )?;
    }
    Ok(())
}

struct InPlaceRowContext<'a> {
    plan: &'a RollingKernelPlan,
    inputs: TypedRowInputs<'a>,
    builders: &'a mut [DerivedBuilder],
    node_id: &'a str,
}

fn process_row(
    context: &mut InPlaceRowContext<'_>,
    row_index: usize,
    local: usize,
    state: &mut RollingKernelState,
    update: &mut InPlaceUpdate,
    processed: &mut usize,
) -> Result<()> {
    let resident = update.residents[local];
    let entity = Arc::get_mut(&mut state.states[resident])
        .ok_or_else(|| internal_error("in-place rolling entity lost exclusive ownership"))?;
    entity.transition_count = entity.transition_count.checked_add(1).ok_or_else(|| {
        operator_error(
            context.node_id,
            "rolling entity transition count overflowed",
        )
    })?;
    *processed += 1;
    let previous = update.undo[local]
        .previous
        .as_mut()
        .expect("resident numeric entity has an undo snapshot");
    update_entity_groups_reversible(
        context.plan,
        context.inputs,
        row_index,
        entity,
        previous,
        context.node_id,
    )?;
    super::append_typed_outputs::<false>(
        &context.plan.outputs,
        context.builders,
        entity,
        context.inputs.columns,
        row_index,
        context.node_id,
    )
}

fn fill_rows(
    plan: &RollingKernelPlan,
    inputs: TypedRowInputs<'_>,
    state: &mut RollingKernelState,
    update: &mut InPlaceUpdate,
    builders: &mut [DerivedBuilder],
    node_id: &str,
    observer: Option<&RollingMetricsRecorder>,
) -> Result<()> {
    let mut processed = 0;
    let mut context = InPlaceRowContext {
        plan,
        inputs,
        builders,
        node_id,
    };
    let result = (|| {
        for (row_index, &local) in inputs.entity_ids.iter().enumerate() {
            process_row(
                &mut context,
                row_index,
                local,
                state,
                update,
                &mut processed,
            )?;
        }
        Ok(())
    })();
    if let Some(recorder) = observer {
        recorder.add(RollingWork::NumericRows, processed);
    }
    result
}

impl Float64NumericState {
    fn update_reversible(
        &mut self,
        event_time: i64,
        sample: Option<f64>,
        numerical_profile: super::RollingNumericalProfile,
        transition_count: u64,
        undo: &mut Float64NumericUndo,
        node_id: &str,
    ) -> Result<()> {
        if let Some(sample) = sample {
            add_float64(&mut self.accumulator, sample, node_id)?;
        }
        self.samples.push(event_time, sample);
        let accumulator = &mut self.accumulator;
        self.samples
            .expire_reversible(event_time, undo, |value| remove_float64(accumulator, value))?;
        self.repair_accumulator(numerical_profile, transition_count, node_id)
    }
}

impl super::SlidingSampleQueue<f64> {
    fn expire_reversible(
        &mut self,
        event_time: i64,
        undo: &mut Float64NumericUndo,
        mut on_remove: impl FnMut(f64) -> Result<()>,
    ) -> Result<()> {
        loop {
            let expired = match self.frame {
                TypedFrame::Rows(window) => self.values.len() > window,
                TypedFrame::Duration(micros) => {
                    let bound = i128::from(event_time) - i128::from(micros);
                    self.values
                        .front()
                        .is_some_and(|sample| i128::from(sample.event_time) <= bound)
                }
            };
            if !expired {
                return Ok(());
            }
            let sample = self
                .values
                .pop_front()
                .expect("an expiring rolling queue has a front sample");
            if undo.original_remaining > 0 {
                undo.original_remaining -= 1;
                undo.evicted.push(sample);
            }
            if let Some(value) = sample.value {
                on_remove(value)?;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RollingNumericalProfile;

    fn warm_state() -> Float64NumericState {
        let mut state = Float64NumericState::new(TypedFrame::Duration(6), 1);
        state
            .update(1, Some(1.0), RollingNumericalProfile::StableV1, 1, "r")
            .unwrap();
        state
    }

    #[test]
    fn single_group_undo_snapshot_does_not_allocate() {
        let entity = TypedEntityState {
            groups: vec![TypedWindowState::Numeric(warm_state())],
            shared_rows: None,
            transition_count: 1,
        };
        let measured = allocation_counter::measure(|| {
            TypedEntityUndo::new(&entity).unwrap();
        });
        assert_eq!(measured.count_total, 0, "{measured:?}");
    }

    #[test]
    fn single_eviction_undo_does_not_allocate() {
        let mut state = warm_state();
        let mut undo = Float64NumericUndo::new(&state);
        let measured = allocation_counter::measure(|| {
            state
                .update_reversible(
                    10,
                    Some(10.0),
                    RollingNumericalProfile::StableV1,
                    2,
                    &mut undo,
                    "r",
                )
                .unwrap();
        });
        assert_eq!(measured.count_total, 0, "{measured:?}");
    }
}
