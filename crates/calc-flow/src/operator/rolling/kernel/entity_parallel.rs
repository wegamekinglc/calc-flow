//! Owned entity work for the deliberately narrow native ordered route.

use std::{
    any::Any,
    cmp::Reverse,
    collections::BinaryHeap,
    fmt, mem,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
};

use super::{
    Array, ArrayRef, DataType, DerivedBuilder, Float64Array, Float64Builder, KernelSelection,
    OutputStorage, PreparedTypedFill, RecordBatch, RollingKernelExecution, RollingKernelPlan,
    RollingKernelState, RollingMetricsRecorder, RollingNumericalProfile, Statistic,
    StreamKernelUpdate, TimestampMicrosecondArray, TypedEntityState, TypedFloatReadout,
    TypedFloatReadoutKind, TypedFrame, TypedGroupInput, TypedGroupPlan, TypedOutputKind,
    TypedOutputPlan, TypedWindowState, float_mean, internal_error, operator_error,
    read_typed_float, typed_output_mismatch, valid_float64,
};
use crate::{CalcFlowError, Result};

const ROWS_PER_LANE: usize = 16_000;

#[cfg(test)]
thread_local! {
    pub(super) static SCHEDULE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static ENTITY_COUNT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum NumericPhase {
    Transition,
    Group,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct NumericSite {
    pub row: u32,
    pub phase: NumericPhase,
    pub ordinal: usize,
}

#[derive(Debug)]
pub(crate) struct TaggedNumericError {
    pub site: NumericSite,
    pub error: CalcFlowError,
}

#[derive(Debug)]
pub(crate) enum NumericLaneStop {
    Ordinary(TaggedNumericError),
    Cancelled,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct ActualNumericWork {
    pub numeric_rows: u64,
    pub overflowed: bool,
}

#[derive(Debug, Default)]
pub(crate) struct LaneProgress {
    pub work: ActualNumericWork,
    pub active_site: Option<NumericSite>,
}

pub(crate) struct LocatedPanic {
    pub site: Option<NumericSite>,
    pub payload: Box<dyn Any + Send>,
}

impl fmt::Debug for LocatedPanic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("LocatedPanic")
            .field("site", &self.site)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) enum NumericLaneOutcome {
    Complete(NumericLaneResult),
    OrdinaryError(TaggedNumericError),
    Cancelled,
    Panicked(LocatedPanic),
    /// The merge guard already transferred this outcome; this is not a stop.
    Consumed,
}

#[derive(Debug)]
pub(crate) struct NumericLaneExit {
    pub outcome: NumericLaneOutcome,
    pub work: ActualNumericWork,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LaneRow {
    pub(super) original_row: u32,
    pub(super) lane_entity: u32,
}

pub(super) struct OwnedLaneEntity {
    pub(super) local_id: usize,
    pub(super) state: Arc<TypedEntityState>,
}

struct PreparedNumericInput {
    entity_ids: Vec<usize>,
    assignments: Vec<EntityAssignment>,
    columns: Vec<TypedGroupInput>,
    event_times: TimestampMicrosecondArray,
    outputs: Vec<TypedOutputPlan>,
    node_id: String,
}

pub(crate) struct NumericLaneRequest {
    input: Arc<PreparedNumericInput>,
    row_count: usize,
    pub(super) rows: Vec<LaneRow>,
    pub(super) entities: Vec<OwnedLaneEntity>,
    #[cfg(test)]
    pub(super) test_hook: Option<Arc<dyn Fn(NumericTestPoint) + Send + Sync>>,
}

impl fmt::Debug for NumericLaneRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericLaneRequest")
            .field("rows", &self.row_count)
            .field("entities", &self.entities.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
impl NumericLaneRequest {
    pub(crate) fn input_weak_for_test(&self) -> std::sync::Weak<dyn Any + Send + Sync> {
        let input: Arc<dyn Any + Send + Sync> = self.input.clone();
        Arc::downgrade(&input)
    }

    pub(super) fn replace_inputs_and_outputs_for_test(
        &mut self,
        columns: Vec<TypedGroupInput>,
        outputs: Vec<TypedOutputPlan>,
    ) {
        let input = Arc::get_mut(&mut self.input).expect("the test owns the only input reference");
        input.columns = columns;
        input.outputs = outputs;
    }
}

pub(crate) struct NumericLaneResult {
    input: Option<Arc<PreparedNumericInput>>,
    pub(super) entities: Vec<OwnedLaneEntity>,
    pub(super) rows: Vec<LaneRow>,
    columns: Vec<Float64Array>,
}

#[cfg(test)]
impl NumericLaneResult {
    pub(super) fn columns_for_test(&mut self) -> &mut Vec<Float64Array> {
        &mut self.columns
    }
}

impl fmt::Debug for NumericLaneResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NumericLaneResult")
            .field("rows", &self.rows.len())
            .field("entities", &self.entities.len())
            .field("columns", &self.columns.len())
            .finish_non_exhaustive()
    }
}

pub(crate) struct NumericJoinSeed {
    pub(super) execution: RollingKernelExecution,
    output_count: usize,
    #[cfg(test)]
    panic_before_output_finish: bool,
}

#[cfg(test)]
impl NumericJoinSeed {
    pub(super) fn set_output_count_for_test(&mut self, count: usize) {
        self.output_count = count;
    }

    pub(crate) fn panic_before_output_finish_for_test(&mut self) {
        self.panic_before_output_finish = true;
    }
}

/// Declared additional owned capacities, distinct from retained state or RSS.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ScratchPlan {
    bytes: Option<usize>,
    lanes: usize,
}

impl ScratchPlan {
    pub(crate) const fn accounted_bytes(self) -> Option<usize> {
        self.bytes
    }

    pub(crate) const fn lanes(self) -> usize {
        self.lanes
    }

    #[cfg(test)]
    pub(crate) fn large_test_plan() -> Self {
        Self::for_dimensions(1_000_000, 64, 2, 1)
    }

    #[cfg(test)]
    pub(super) fn for_dimensions(
        rows: usize,
        entities: usize,
        outputs: usize,
        name_bytes: usize,
    ) -> Self {
        Self::for_lanes(rows, entities, outputs, name_bytes, 2)
    }

    fn for_lanes(
        rows: usize,
        entities: usize,
        outputs: usize,
        name_bytes: usize,
        lanes: usize,
    ) -> Self {
        let bytes = Self::owned_capacity(rows, entities, outputs, name_bytes, lanes);
        Self { bytes, lanes }
    }

    fn owned_capacity(
        rows: usize,
        entities: usize,
        outputs: usize,
        name_bytes: usize,
        lanes: usize,
    ) -> Option<usize> {
        let routing = Self::routing_capacity(rows, entities)?;
        let output = Self::output_capacity(rows, outputs, lanes)?;
        let fixed_metadata = Self::fixed_capacity(lanes)?;
        routing
            .checked_add(output)?
            .checked_add(fixed_metadata)?
            .checked_add(Self::name_capacity(name_bytes, lanes)?)
    }

    fn fixed_capacity(lanes: usize) -> Option<usize> {
        let controls = lanes.checked_mul(4096 + COMPILED_LANE_READOUT_METADATA_BYTES / 2)?;
        controls
            .checked_add(NULL_FREE_MERGE_METADATA_BYTES)?
            .checked_add(size_of::<NumericResults>())?
            .checked_add(size_of::<PreparedNumericInput>())
    }

    fn name_capacity(name_bytes: usize, lanes: usize) -> Option<usize> {
        name_bytes.checked_mul(lanes.checked_mul(2)?.checked_add(1)?)
    }

    fn routing_capacity(rows: usize, entities: usize) -> Option<usize> {
        let schedules = rows.checked_mul(size_of::<LaneRow>())?;
        // Charge the lane slot and a conservative Arc owner allowance; the
        // prepared entity payload is shared with the serial fallback.
        let moved = entities.checked_mul(
            size_of::<OwnedLaneEntity>() + size_of::<TypedEntityState>() + 2 * size_of::<usize>(),
        )?;
        let assignment = entities.checked_mul(size_of::<EntityAssignment>())?;
        schedules.checked_add(moved)?.checked_add(assignment)
    }

    fn validity_capacity(rows: usize, outputs: usize, lanes: usize) -> Option<usize> {
        rows.checked_add(7)?
            .checked_div(8)?
            .checked_add(lanes.checked_mul(128)?)?
            .checked_mul(outputs)
    }

    fn output_capacity(rows: usize, outputs: usize, lanes: usize) -> Option<usize> {
        let values = rows.checked_mul(outputs)?.checked_mul(size_of::<f64>())?;
        let validity = Self::validity_capacity(rows, outputs, lanes)?;
        let metadata = outputs.checked_mul(lanes)?.checked_mul(4096)?;
        values.checked_add(validity)?.checked_add(metadata)
    }
}

#[derive(Clone, Copy, Default)]
struct EntityAssignment {
    rows: usize,
    lane: usize,
    local: usize,
}

pub(in crate::operator::rolling) struct PreparedOrderedKernel<'a> {
    plan: &'a RollingKernelPlan,
    node_id: &'a str,
    pub(super) fill: PreparedTypedFill,
    parallel_shape: bool,
}

impl RollingKernelPlan {
    /// The sole route-off control point; managed owner initialization is outside it.
    pub(in crate::operator::rolling) fn supports_entity_parallel(
        &self,
        input: &RecordBatch,
    ) -> bool {
        self.parallel_dimensions(input)
            && self.groups.iter().all(|group| {
                matches!(group,
                TypedGroupPlan::Numeric { input_index, frame: TypedFrame::Rows(5 | 20) }
                if input.column(*input_index).data_type() == &DataType::Float64)
            })
            && self
                .outputs
                .iter()
                .all(|output| self.parallel_output(output))
    }

    fn parallel_dimensions(&self, input: &RecordBatch) -> bool {
        input.num_rows() >= 64_000
            && u32::try_from(input.num_rows()).is_ok()
            && self.selection == KernelSelection::OrderedPrimitive
            && self.numerical_profile == RollingNumericalProfile::StableV1
            && !self.nan_as_value
            && (1..=2).contains(&self.groups.len())
            && (1..=2).contains(&self.outputs.len())
    }

    fn parallel_output(&self, output: &TypedOutputPlan) -> bool {
        output.storage == OutputStorage::Float64
            && match output.kind {
                TypedOutputKind::Statistic(Statistic::Mean) => output.group < self.groups.len(),
                TypedOutputKind::Difference { left, right } => {
                    [left, right].iter().all(|readout| {
                        readout.kind == TypedFloatReadoutKind::Mean
                            && readout.group < self.groups.len()
                    })
                }
                _ => false,
            }
    }

    pub(in crate::operator::rolling) fn prepare_ordered_stream_inputs<'a>(
        &'a self,
        state: &RollingKernelState,
        input: &RecordBatch,
        node_id: &'a str,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<PreparedOrderedKernel<'a>> {
        let stream = self.prepare_ordered_stream_state(state, input, node_id, observer)?;
        let fill = self.prepare_typed_fill(input, stream, node_id, observer)?;
        Ok(PreparedOrderedKernel {
            plan: self,
            node_id,
            fill,
            parallel_shape: self.supports_entity_parallel(input),
        })
    }
}

impl PreparedOrderedKernel<'_> {
    #[cfg(test)]
    pub(in crate::operator::rolling) fn scratch_plan(&self) -> Option<ScratchPlan> {
        self.scratch_plan_for_cpu(2)
    }

    pub(in crate::operator::rolling) fn scratch_plan_for_cpu(
        &self,
        cpus: usize,
    ) -> Option<ScratchPlan> {
        let plan = self.plan;
        let states = &self.fill.stream.state.states;
        if !self.parallel_shape
            || states.len() < 16
            || !states
                .iter()
                .all(|entity| parallel_entity_state(entity, &plan.groups))
            || !self
                .fill
                .columns
                .iter()
                .all(|column| matches!(column, TypedGroupInput::Single(_)))
        {
            return None;
        }
        let lanes = cpus
            .min(self.fill.row_count / ROWS_PER_LANE)
            .min(states.len());
        if lanes < 2 {
            return None;
        }
        let scratch = ScratchPlan::for_lanes(
            self.fill.row_count,
            states.len(),
            plan.outputs.len(),
            self.node_id.len(),
            lanes,
        );
        scratch.accounted_bytes().map(|_| scratch)
    }

    pub(in crate::operator::rolling) fn finish_serial(
        self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<StreamKernelUpdate> {
        self.plan
            .finish_typed_fill(self.fill, self.node_id, observer)
            .map(|execution| StreamKernelUpdate { execution })
    }

    /// Call only while the job's scratch reservation is held. All rejection
    /// happens before moving any state, retaining the same prepared serial value.
    #[allow(
        clippy::result_large_err,
        reason = "rejection returns the same owned preparation without a new heap allocation"
    )]
    #[cfg(test)]
    pub(in crate::operator::rolling) fn split_two(
        self,
    ) -> std::result::Result<(NumericJoinSeed, [NumericLaneRequest; 2]), Self> {
        self.split_lanes(2)
            .map(|(seed, requests)| (seed, requests.try_into().expect("two-lane fixture")))
    }

    #[allow(
        clippy::result_large_err,
        reason = "rejection reuses the existing owned serial preparation"
    )]
    pub(in crate::operator::rolling) fn split_lanes(
        self,
        cpus: usize,
    ) -> std::result::Result<(NumericJoinSeed, Vec<NumericLaneRequest>), Self> {
        let Some(scratch) = self.scratch_plan_for_cpu(cpus) else {
            return Err(self);
        };
        let lanes = scratch.lanes();
        let mut assignments =
            vec![EntityAssignment::default(); self.fill.stream.state.states.len()];
        for (assignment, &count) in assignments
            .iter_mut()
            .zip(&self.fill.stream.entity_rows.counts)
        {
            #[cfg(test)]
            ENTITY_COUNT_VISITS.set(ENTITY_COUNT_VISITS.get() + 1);
            assignment.rows = count;
        }
        let mut loads = vec![0_usize; lanes];
        let mut entities = vec![0_usize; lanes];
        let mut available = (0..lanes)
            .map(|lane| Reverse((0_usize, lane)))
            .collect::<BinaryHeap<_>>();
        for assignment in &mut assignments {
            let Reverse((_, lane)) = available.pop().expect("nonempty lane plan");
            assignment.lane = lane;
            assignment.local = entities[lane];
            entities[lane] += 1;
            let Some(load) = loads[lane].checked_add(assignment.rows) else {
                return Err(self);
            };
            loads[lane] = load;
            available.push(Reverse((loads[lane], lane)));
        }
        if !balanced_lane_loads(&loads, self.fill.row_count) {
            return Err(self);
        }
        let PreparedTypedFill {
            columns,
            event_times,
            row_count,
            mut stream,
        } = self.fill;
        let mut lane_entities = entities
            .iter()
            .map(|&count| Vec::with_capacity(count))
            .collect::<Vec<_>>();
        for (local_id, state) in stream.state.states.drain(..).enumerate() {
            lane_entities[assignments[local_id].lane].push(OwnedLaneEntity { local_id, state });
        }
        let mut schedules = loads
            .iter()
            .map(|&rows| Vec::with_capacity(rows))
            .collect::<Vec<_>>();
        for (row, &entity) in stream.entity_ids.iter().enumerate() {
            let assignment = assignments[entity];
            schedules[assignment.lane].push(LaneRow {
                original_row: u32::try_from(row).expect("qualified rows fit u32"),
                lane_entity: u32::try_from(assignment.local).expect("qualified entities fit u32"),
            });
        }
        let shared = Arc::new(PreparedNumericInput {
            entity_ids: stream.entity_ids,
            assignments,
            columns,
            event_times,
            outputs: self.plan.outputs.clone(),
            node_id: self.node_id.to_owned(),
        });
        let requests = (0..lanes)
            .map(|lane| NumericLaneRequest {
                input: Arc::clone(&shared),
                row_count: loads[lane],
                rows: mem::take(&mut schedules[lane]),
                entities: mem::take(&mut lane_entities[lane]),
                #[cfg(test)]
                test_hook: None,
            })
            .collect();
        if stream.last_identity.is_some() {
            stream.state.last_identity = stream.last_identity;
        }
        stream.metrics.input_rows = row_count;
        stream.metrics.output_rows = row_count;
        let seed = NumericJoinSeed {
            execution: RollingKernelExecution {
                columns: Vec::new(),
                entity_ids: Vec::new(),
                entity_rows: stream.entity_rows,
                metrics: stream.metrics,
                state: stream.state,
            },
            output_count: self.plan.outputs.len(),
            #[cfg(test)]
            panic_before_output_finish: false,
        };
        Ok((seed, requests))
    }
}

fn balanced_lane_loads(loads: &[usize], row_count: usize) -> bool {
    let max_load = row_count / 8 * 5 + row_count % 8 * 5 / 8;
    !loads.iter().any(|&rows| rows == 0 || rows > max_load)
        && loads
            .iter()
            .try_fold(0_usize, |sum, &rows| sum.checked_add(rows))
            == Some(row_count)
}

fn parallel_entity_state(entity: &TypedEntityState, groups: &[TypedGroupPlan]) -> bool {
    entity.groups.len() == groups.len()
        && entity
            .groups
            .iter()
            .zip(groups)
            .all(|(state, group)| match (state, group) {
                (
                    TypedWindowState::Numeric(state),
                    TypedGroupPlan::Numeric {
                        frame: TypedFrame::Rows(window @ (5 | 20)),
                        ..
                    },
                ) => {
                    matches!(state.samples.frame, TypedFrame::Rows(actual) if actual == *window)
                        && state.samples.len() <= *window
                }
                _ => false,
            })
}

#[cfg(test)]
#[derive(Clone, Copy)]
pub(super) enum NumericTestPoint {
    BeforeGroup(NumericSite),
    AfterGroup(NumericSite),
    BeforeOutput(NumericSite),
    Finish,
}

fn check_cancelled(
    token: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<(), NumericLaneStop> {
    progress.active_site = None;
    if token.load(Ordering::SeqCst) {
        Err(NumericLaneStop::Cancelled)
    } else {
        Ok(())
    }
}

pub(crate) fn run_numeric_lane(
    request: NumericLaneRequest,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    check_cancelled(cancellation, progress)?;
    let input = &request.input;
    let builders = input
        .outputs
        .iter()
        .map(|output| DerivedBuilder::new(*output, request.row_count))
        .collect::<Vec<_>>();
    let compiled_output = match input.outputs.as_slice() {
        [output]
            if output.storage == OutputStorage::Float64
                && input
                    .columns
                    .iter()
                    .all(|column| matches!(column, TypedGroupInput::Single(_))) =>
        {
            Some(*output)
        }
        _ => None,
    };
    match (input.columns.len(), compiled_output) {
        (1, Some(output)) => {
            run_compiled_numeric_lane::<1>(request, builders, output, cancellation, progress)
        }
        (2, Some(output)) => {
            run_compiled_numeric_lane::<2>(request, builders, output, cancellation, progress)
        }
        _ => run_prepared_numeric_lane::<0>(
            request,
            builders,
            cancellation,
            progress,
            DerivedBuilder::append,
        ),
    }
}

// Keep both the entry descriptor and its static closure capture in the allowance.
pub(super) const COMPILED_LANE_READOUT_METADATA_BYTES: usize = 2
    * (size_of::<Option<TypedOutputPlan>>()
        + if size_of::<TypedOutputPlan>() > size_of::<[TypedFloatReadout; 2]>() {
            size_of::<TypedOutputPlan>()
        } else {
            size_of::<[TypedFloatReadout; 2]>()
        }
        + size_of::<[usize; 2]>()
        + size_of::<[(&Float64Array, &TypedGroupInput); 2]>());

#[derive(Clone, Copy)]
struct LaneMeanReadout {
    group: usize,
    min_periods: u64,
}

// The Mean-only capture remains covered by the original, conservative allowance.
const _: () = assert!(size_of::<[LaneMeanReadout; 2]>() <= size_of::<[TypedFloatReadout; 2]>());

fn run_compiled_numeric_lane<const GROUPS: usize>(
    request: NumericLaneRequest,
    builders: Vec<DerivedBuilder>,
    output: TypedOutputPlan,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    match output.kind {
        TypedOutputKind::Statistic(Statistic::Mean) => run_prepared_numeric_lane::<GROUPS>(
            request,
            builders,
            cancellation,
            progress,
            move |builder, states, _, node_id| {
                append_numeric_lane_mean(builder, states, &output, node_id)
            },
        ),
        TypedOutputKind::Difference { left, right }
            if left.kind == TypedFloatReadoutKind::Mean
                && right.kind == TypedFloatReadoutKind::Mean =>
        {
            let left = LaneMeanReadout {
                group: left.group,
                min_periods: left.min_periods,
            };
            let right = LaneMeanReadout {
                group: right.group,
                min_periods: right.min_periods,
            };
            run_prepared_numeric_lane::<GROUPS>(
                request,
                builders,
                cancellation,
                progress,
                move |builder, states, _, _| {
                    append_numeric_lane_difference(builder, states, left, right)
                },
            )
        }
        _ => run_prepared_numeric_lane::<0>(
            request,
            builders,
            cancellation,
            progress,
            DerivedBuilder::append,
        ),
    }
}

fn append_numeric_lane_mean(
    builder: &mut DerivedBuilder,
    states: &[TypedWindowState],
    output: &TypedOutputPlan,
    node_id: &str,
) -> Result<()> {
    let state = states
        .get(output.group)
        .ok_or_else(|| internal_error("typed rolling output group is out of bounds"))?;
    let TypedWindowState::Numeric(state) = state else {
        return builder.append(states, output, node_id);
    };
    if state.accumulator.valid_count < output.min_periods {
        builder.append_null();
        return Ok(());
    }
    let DerivedBuilder::Float(values, _) = builder else {
        return Err(typed_output_mismatch());
    };
    values.append_value(float_mean(&state.accumulator));
    Ok(())
}

fn append_numeric_lane_difference(
    builder: &mut DerivedBuilder,
    states: &[TypedWindowState],
    left: LaneMeanReadout,
    right: LaneMeanReadout,
) -> Result<()> {
    let DerivedBuilder::Float(values, _) = builder else {
        return Err(typed_output_mismatch());
    };
    match (
        read_numeric_lane_mean(states, left)?,
        read_numeric_lane_mean(states, right)?,
    ) {
        (Some(left), Some(right)) => values.append_value(left - right),
        _ => values.append_null(),
    }
    Ok(())
}

fn read_numeric_lane_mean(
    states: &[TypedWindowState],
    readout: LaneMeanReadout,
) -> Result<Option<f64>> {
    let state = states
        .get(readout.group)
        .ok_or_else(|| internal_error("typed fused readout group is out of bounds"))?;
    let TypedWindowState::Numeric(state) = state else {
        return read_typed_float(
            states,
            TypedFloatReadout {
                group: readout.group,
                kind: TypedFloatReadoutKind::Mean,
                min_periods: readout.min_periods,
                ddof: 0,
            },
        );
    };
    if state.accumulator.valid_count < readout.min_periods {
        return Ok(None);
    }
    Ok(Some(float_mean(&state.accumulator)))
}

#[cfg(test)]
pub(super) fn append_numeric_lane_difference_for_test(
    builder: &mut DerivedBuilder,
    states: &[TypedWindowState],
    left: TypedFloatReadout,
    right: TypedFloatReadout,
) -> Result<()> {
    assert_eq!(left.kind, TypedFloatReadoutKind::Mean);
    assert_eq!(right.kind, TypedFloatReadoutKind::Mean);
    append_numeric_lane_difference(
        builder,
        states,
        LaneMeanReadout {
            group: left.group,
            min_periods: left.min_periods,
        },
        LaneMeanReadout {
            group: right.group,
            min_periods: right.min_periods,
        },
    )
}

#[cfg(test)]
pub(super) fn append_numeric_lane_mean_for_test(
    builder: &mut DerivedBuilder,
    states: &[TypedWindowState],
    output: &TypedOutputPlan,
    node_id: &str,
) -> Result<()> {
    append_numeric_lane_mean(builder, states, output, node_id)
}

#[cfg(test)]
pub(super) fn run_generic_numeric_lane_for_test(
    request: NumericLaneRequest,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    check_cancelled(cancellation, progress)?;
    let builders = request
        .input
        .outputs
        .iter()
        .map(|output| DerivedBuilder::new(*output, request.row_count))
        .collect();
    run_prepared_numeric_lane::<0>(
        request,
        builders,
        cancellation,
        progress,
        DerivedBuilder::append,
    )
}

struct NumericLaneContext<'a, const GROUPS: usize> {
    input: &'a PreparedNumericInput,
    fixed_columns: [(&'a Float64Array, &'a TypedGroupInput); GROUPS],
    group_count: usize,
    output_count: usize,
    cancellation: &'a AtomicBool,
    #[cfg(test)]
    test_hook: Option<&'a Arc<dyn Fn(NumericTestPoint) + Send + Sync>>,
}

// Borrowed helpers fit the existing two-lane fixed allowance without new heap storage.
const _: () =
    assert!(2 * size_of::<NumericLaneContext<'_, 2>>() <= COMPILED_LANE_READOUT_METADATA_BYTES);

impl<'a, const GROUPS: usize> NumericLaneContext<'a, GROUPS> {
    fn new(
        input: &'a PreparedNumericInput,
        builders: &[DerivedBuilder],
        cancellation: &'a AtomicBool,
        #[cfg(test)] test_hook: Option<&'a Arc<dyn Fn(NumericTestPoint) + Send + Sync>>,
    ) -> Self {
        let fixed_columns = std::array::from_fn(|ordinal| {
            let column = &input.columns[ordinal];
            let TypedGroupInput::Single(values) = column else {
                unreachable!("the compiled lane has Single inputs");
            };
            (values, column)
        });
        Self {
            input,
            fixed_columns,
            group_count: if GROUPS == 0 {
                input.columns.len()
            } else {
                GROUPS
            },
            output_count: if GROUPS == 0 {
                builders.len().min(input.outputs.len())
            } else {
                1
            },
            cancellation,
            #[cfg(test)]
            test_hook,
        }
    }

    fn check_row(
        &self,
        ordinal: usize,
        progress: &mut LaneProgress,
    ) -> std::result::Result<(), NumericLaneStop> {
        if ordinal.is_multiple_of(256) {
            check_cancelled(self.cancellation, progress)?;
        }
        Ok(())
    }

    fn column(&self, ordinal: usize) -> &TypedGroupInput {
        if GROUPS == 0 {
            &self.input.columns[ordinal]
        } else {
            self.fixed_columns[ordinal].1
        }
    }

    #[cfg(test)]
    fn observe(&self, point: NumericTestPoint) {
        if let Some(hook) = self.test_hook {
            hook(point);
        }
    }

    fn update_entity(
        &self,
        entity: &mut TypedEntityState,
        row: LaneRow,
        progress: &mut LaneProgress,
    ) -> std::result::Result<(), NumericLaneStop> {
        let site = NumericSite {
            row: row.original_row,
            phase: NumericPhase::Transition,
            ordinal: 0,
        };
        progress.active_site = Some(site);
        entity.transition_count = entity.transition_count.checked_add(1).ok_or_else(|| {
            NumericLaneStop::Ordinary(TaggedNumericError {
                site,
                error: operator_error(
                    &self.input.node_id,
                    "rolling entity transition count overflowed",
                ),
            })
        })?;
        if let Some(count) = progress.work.numeric_rows.checked_add(1) {
            progress.work.numeric_rows = count;
        } else {
            progress.work.overflowed = true;
        }
        self.update_groups(entity, row, progress)
    }

    fn update_groups(
        &self,
        entity: &mut TypedEntityState,
        row: LaneRow,
        progress: &mut LaneProgress,
    ) -> std::result::Result<(), NumericLaneStop> {
        for ordinal in 0..self.group_count {
            let Some(state) = entity.groups.get_mut(ordinal) else {
                break;
            };
            let column = self.column(ordinal);
            check_cancelled(self.cancellation, progress)?;
            let site = NumericSite {
                row: row.original_row,
                phase: NumericPhase::Group,
                ordinal,
            };
            progress.active_site = Some(site);
            #[cfg(test)]
            self.observe(NumericTestPoint::BeforeGroup(site));
            self.update_group_state(state, column, site, entity.transition_count)
                .map_err(|error| NumericLaneStop::Ordinary(TaggedNumericError { site, error }))?;
            #[cfg(test)]
            self.observe(NumericTestPoint::AfterGroup(site));
            check_cancelled(self.cancellation, progress)?;
        }
        Ok(())
    }

    fn update_group_state(
        &self,
        state: &mut TypedWindowState,
        column: &TypedGroupInput,
        site: NumericSite,
        transition_count: u64,
    ) -> Result<()> {
        // Eligibility proved the Rows5/20 queue bound. Expiration precedes repair.
        let event_time = self.input.event_times.value(site.row as usize);
        match state {
            TypedWindowState::Numeric(state) if GROUPS != 0 => state.update(
                event_time,
                valid_float64(self.fixed_columns[site.ordinal].0, site.row as usize, false),
                RollingNumericalProfile::StableV1,
                transition_count,
                &self.input.node_id,
            ),
            state => state.update(
                event_time,
                column,
                site.row as usize,
                RollingNumericalProfile::StableV1,
                false,
                transition_count,
                &self.input.node_id,
            ),
        }
    }

    fn append_outputs(
        &self,
        builders: &mut [DerivedBuilder],
        entity: &TypedEntityState,
        row: LaneRow,
        progress: &mut LaneProgress,
        append: &mut impl FnMut(
            &mut DerivedBuilder,
            &[TypedWindowState],
            &TypedOutputPlan,
            &str,
        ) -> Result<()>,
    ) -> std::result::Result<(), NumericLaneStop> {
        for (ordinal, builder) in builders.iter_mut().enumerate().take(self.output_count) {
            let output = &self.input.outputs[ordinal];
            let site = NumericSite {
                row: row.original_row,
                phase: NumericPhase::Output,
                ordinal,
            };
            progress.active_site = Some(site);
            #[cfg(test)]
            self.observe(NumericTestPoint::BeforeOutput(site));
            append(builder, &entity.groups, output, &self.input.node_id)
                .map_err(|error| NumericLaneStop::Ordinary(TaggedNumericError { site, error }))?;
        }
        Ok(())
    }
}

fn run_prepared_numeric_lane<const GROUPS: usize>(
    mut request: NumericLaneRequest,
    mut builders: Vec<DerivedBuilder>,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
    mut append: impl FnMut(
        &mut DerivedBuilder,
        &[TypedWindowState],
        &TypedOutputPlan,
        &str,
    ) -> Result<()>,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    // Borrow only the immutable input and hook, leaving owned row/state buffers mutable.
    let context = NumericLaneContext::<GROUPS>::new(
        &request.input,
        &builders,
        cancellation,
        #[cfg(test)]
        request.test_hook.as_ref(),
    );
    progress.active_site = None;
    // Prepared lane entities are already unique; borrow each once per lane.
    let mut entities = request
        .entities
        .iter_mut()
        .map(|entity| Arc::make_mut(&mut entity.state))
        .collect::<Vec<_>>();
    for (ordinal, row) in request.rows.iter().copied().enumerate() {
        #[cfg(test)]
        SCHEDULE_VISITS.set(SCHEDULE_VISITS.get() + 1);
        context.check_row(ordinal, progress)?;
        let entity = &mut *entities[usize::try_from(row.lane_entity).expect("u32 fits platform")];
        context.update_entity(entity, row, progress)?;
        context.append_outputs(&mut builders, entity, row, progress, &mut append)?;
        progress.active_site = None;
    }
    drop(entities);
    check_cancelled(cancellation, progress)?;
    #[cfg(test)]
    context.observe(NumericTestPoint::Finish);
    let columns = finish_numeric_columns(builders);
    Ok(NumericLaneResult {
        input: Some(request.input),
        entities: request.entities,
        rows: request.rows,
        columns,
    })
}

fn finish_numeric_columns(builders: Vec<DerivedBuilder>) -> Vec<Float64Array> {
    builders
        .into_iter()
        .map(|builder| match builder {
            DerivedBuilder::Float(mut values, OutputStorage::Float64) => values.finish(),
            _ => unreachable!("qualified lane output is Float64"),
        })
        .collect()
}

pub(super) fn finish_null_free_numeric_columns<const COLUMNS: usize>(
    values: [Vec<f64>; COLUMNS],
) -> Vec<ArrayRef> {
    values
        .into_iter()
        .map(|values| Arc::new(Float64Array::from(values)) as ArrayRef)
        .collect()
}

enum NumericOutputBuffers {
    One([Vec<f64>; 1]),
    Two([Vec<f64>; 2]),
    Nullable(Vec<Float64Builder>),
}

// Fixed output headers and the largest pair of logical column-view arrays.
pub(super) const NULL_FREE_MERGE_METADATA_BYTES: usize =
    size_of::<NumericOutputBuffers>() + size_of::<[[&[f64]; 2]; 2]>();

impl NumericOutputBuffers {
    fn gather(
        entity_ids: &[usize],
        assignments: &[EntityAssignment],
        results: &[NumericLaneResult],
        count: usize,
    ) -> Result<Self> {
        let rows = entity_ids.len();
        let null_free = matches!(count, 1 | 2)
            && results
                .iter()
                .all(|result| result.columns.iter().all(|column| column.null_count() == 0));
        match count {
            1 if null_free => gather_null_free_numeric_columns_impl(
                entity_ids,
                assignments,
                results,
                [Vec::with_capacity(rows)],
            )
            .map(Self::One),
            2 if null_free => gather_null_free_numeric_columns_impl(
                entity_ids,
                assignments,
                results,
                std::array::from_fn(|_| Vec::with_capacity(rows)),
            )
            .map(Self::Two),
            _ => {
                let mut builders = (0..count)
                    .map(|_| Float64Builder::with_capacity(rows))
                    .collect::<Vec<_>>();
                merge_numeric_rows(entity_ids, assignments, results, |lane, offset| {
                    for (builder, column) in builders.iter_mut().zip(&results[lane].columns) {
                        if column.is_null(offset) {
                            builder.append_null();
                        } else {
                            builder.append_value(column.value(offset));
                        }
                    }
                })?;
                Ok(Self::Nullable(builders))
            }
        }
    }

    fn finish(self) -> Vec<ArrayRef> {
        match self {
            Self::One(values) => finish_null_free_numeric_columns(values),
            Self::Two(values) => finish_null_free_numeric_columns(values),
            Self::Nullable(builders) => builders
                .into_iter()
                .map(|mut builder| Arc::new(builder.finish()) as ArrayRef)
                .collect(),
        }
    }
}

#[cfg(test)]
pub(super) fn gather_null_free_numeric_columns<const COLUMNS: usize>(
    entity_ids: &[usize],
    results: &[NumericLaneResult],
    values: [Vec<f64>; COLUMNS],
) -> Result<[Vec<f64>; COLUMNS]> {
    let assignments = &results[0]
        .input
        .as_ref()
        .expect("test input remains owned")
        .assignments;
    gather_null_free_numeric_columns_impl(entity_ids, assignments, results, values)
}

fn gather_null_free_numeric_columns_impl<const COLUMNS: usize>(
    entity_ids: &[usize],
    assignments: &[EntityAssignment],
    results: &[NumericLaneResult],
    mut values: [Vec<f64>; COLUMNS],
) -> Result<[Vec<f64>; COLUMNS]> {
    merge_numeric_rows(entity_ids, assignments, results, |lane, offset| {
        for (column, values) in values.iter_mut().enumerate() {
            values.push(results[lane].columns[column].value(offset));
        }
    })?;
    Ok(values)
}

fn merge_numeric_rows(
    entity_ids: &[usize],
    assignments: &[EntityAssignment],
    results: &[NumericLaneResult],
    mut append: impl FnMut(usize, usize),
) -> Result<()> {
    let mut inline = [0_usize; 2];
    let mut dynamic = Vec::new();
    let offsets = if results.len() <= inline.len() {
        &mut inline[..results.len()]
    } else {
        dynamic.resize(results.len(), 0_usize);
        dynamic.as_mut_slice()
    };
    for (global, expected_entity) in entity_ids.iter().enumerate() {
        let lane = assignments
            .get(*expected_entity)
            .map(|assignment| assignment.lane)
            .filter(|&lane| lane < results.len())
            .ok_or_else(|| {
                internal_error("numeric lane row coverage or entity identity changed")
            })?;
        let result = &results[lane];
        let row = result
            .rows
            .get(offsets[lane])
            .ok_or_else(|| internal_error("numeric lane row coverage is incomplete"))?;
        if row.original_row as usize != global
            || result
                .entities
                .get(row.lane_entity as usize)
                .map(|entity| entity.local_id)
                != Some(*expected_entity)
        {
            return Err(internal_error(
                "numeric lane row coverage or entity identity changed",
            ));
        }
        append(lane, offsets[lane]);
        offsets[lane] += 1;
    }
    if results
        .iter()
        .zip(offsets.iter())
        .any(|(result, &consumed)| result.rows.len() != consumed)
    {
        return Err(internal_error("numeric lane row coverage has extra rows"));
    }
    Ok(())
}

fn take_numeric_input(results: &mut [NumericLaneResult]) -> Result<PreparedNumericInput> {
    let mut input = None;
    for result in results {
        let current = result
            .input
            .take()
            .ok_or_else(|| internal_error("numeric lane input ownership is missing"))?;
        if let Some(first) = &input {
            if !Arc::ptr_eq(first, &current) {
                return Err(internal_error("numeric lane inputs differ"));
            }
        } else {
            input = Some(current);
        }
    }
    let first = input.ok_or_else(|| internal_error("numeric lane input ownership is missing"))?;
    Arc::try_unwrap(first).map_err(|_| internal_error("numeric lane input ownership is shared"))
}

#[allow(
    clippy::large_enum_variant,
    reason = "the two-lane case preserves its existing allocation-free merge metadata"
)]
enum NumericResults {
    Two([NumericLaneResult; 2]),
    Many(Vec<NumericLaneResult>),
}

impl NumericResults {
    fn as_mut_slice(&mut self) -> &mut [NumericLaneResult] {
        match self {
            Self::Two(results) => results,
            Self::Many(results) => results,
        }
    }
}

fn take_numeric_results(exits: &mut [NumericLaneExit], run_id: &str) -> Result<NumericResults> {
    // Unlocated panics are task failures, never assigned the last active row.
    let selected = exits
        .iter()
        .position(|exit| {
            matches!(
                &exit.outcome,
                NumericLaneOutcome::Panicked(LocatedPanic { site: None, .. })
            )
        })
        .or_else(|| {
            exits
                .iter()
                .enumerate()
                .filter_map(|(index, exit)| match &exit.outcome {
                    NumericLaneOutcome::OrdinaryError(error) => Some((error.site, index)),
                    NumericLaneOutcome::Panicked(LocatedPanic {
                        site: Some(site), ..
                    }) => Some((*site, index)),
                    _ => None,
                })
                .min()
                .map(|(_, index)| index)
        });
    if let Some(index) = selected {
        match mem::replace(&mut exits[index].outcome, NumericLaneOutcome::Consumed) {
            NumericLaneOutcome::OrdinaryError(error) => return Err(error.error),
            NumericLaneOutcome::Panicked(panic) => std::panic::resume_unwind(panic.payload),
            _ => unreachable!("selected numeric failure"),
        }
    }
    if exits
        .iter()
        .any(|exit| matches!(exit.outcome, NumericLaneOutcome::Cancelled))
    {
        return Err(CalcFlowError::Cancelled {
            run_id: run_id.to_owned(),
        });
    }
    let take = |exit: &mut NumericLaneExit| match mem::replace(
        &mut exit.outcome,
        NumericLaneOutcome::Consumed,
    ) {
        NumericLaneOutcome::Complete(result) => result,
        _ => unreachable!("all non-complete outcomes handled above"),
    };
    if exits.len() == 2 {
        Ok(NumericResults::Two(std::array::from_fn(|lane| {
            take(&mut exits[lane])
        })))
    } else {
        Ok(NumericResults::Many(exits.iter_mut().map(take).collect()))
    }
}

/// Used only by `JoinedNumericPair`'s concrete merge operation, with its scratch
/// permit alive. Kept separate from that runtime owner for direct numerical tests.
pub(crate) fn merge_numeric_results(
    mut seed: NumericJoinSeed,
    exits: &mut [NumericLaneExit],
    run_id: &str,
) -> Result<StreamKernelUpdate> {
    let mut results = take_numeric_results(exits, run_id)?;
    let results = results.as_mut_slice();
    let input = take_numeric_input(results)?;
    let count = seed.output_count;
    validate_numeric_shape(results, count)?;
    let buffers =
        NumericOutputBuffers::gather(&input.entity_ids, &input.assignments, results, count)?;
    merge_numeric_entities(&mut seed.execution.state, &input.assignments, results)?;
    seed.execution.entity_ids = input.entity_ids;
    #[cfg(test)]
    if seed.panic_before_output_finish {
        std::panic::resume_unwind(Box::new(673_u32));
    }
    seed.execution.columns = buffers.finish();
    seed.execution.metrics.entities = seed.execution.state.states.len();
    seed.execution.metrics.state_bytes = seed
        .execution
        .state
        .states
        .iter()
        .map(|entity| entity.estimated_bytes())
        .sum();
    // Lane buffers/schedules and moved containers drop here, before the caller's
    // JoinedNumericPair releases its permit; only original serial state/output escape.
    Ok(StreamKernelUpdate {
        execution: seed.execution,
    })
}

fn validate_numeric_shape(results: &[NumericLaneResult], count: usize) -> Result<()> {
    if results.iter().any(|result| {
        result.columns.len() != count
            || result.columns.iter().any(|column| {
                column.data_type() != &DataType::Float64 || column.len() != result.rows.len()
            })
    }) {
        return Err(internal_error("numeric lane output shape changed"));
    }
    Ok(())
}

fn merge_numeric_entities(
    state: &mut RollingKernelState,
    assignments: &[EntityAssignment],
    results: &[NumericLaneResult],
) -> Result<()> {
    if results
        .iter()
        .map(|result| result.entities.len())
        .sum::<usize>()
        != assignments.len()
    {
        return Err(internal_error("numeric lane entity slots changed"));
    }
    for (local_id, assignment) in assignments.iter().enumerate() {
        let entity = results
            .get(assignment.lane)
            .and_then(|result| result.entities.get(assignment.local))
            .filter(|entity| entity.local_id == local_id)
            .ok_or_else(|| internal_error("numeric lane entity slots changed"))?;
        if local_id != state.states.len() {
            return Err(internal_error("numeric lane entity slots changed"));
        }
        state.states.push(Arc::clone(&entity.state));
    }
    if state.states.len() != state.entities.len() {
        return Err(internal_error("numeric lane entity slots are incomplete"));
    }
    Ok(())
}
