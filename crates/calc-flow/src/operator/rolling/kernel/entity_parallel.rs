//! Owned entity work for independently executable typed rolling transitions.

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
    RollingKernelState, RollingMetricsRecorder, RollingNumericalProfile, RollingWork,
    RollingWorkBatch, Statistic, StreamKernelUpdate, TimestampMicrosecondArray, TypedEntityState,
    TypedFloatReadout, TypedFloatReadoutKind, TypedFrame, TypedGroupInput, TypedGroupPlan,
    TypedOutputKind, TypedOutputPlan, TypedWindowState, append_numeric_float64, internal_error,
    operator_error, read_typed_float, typed_output_mismatch, valid_float64, with_float64_builder,
};
use crate::{CalcFlowError, Result};

mod route;
use route::GlobalRowCursor;
pub(super) use route::{LaneRoute, RouteCursor, RouteStep, RoutingLayout};

const ROWS_PER_LANE: usize = 16_000;

#[cfg(test)]
thread_local! {
    pub(super) static SCHEDULE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static ENTITY_COUNT_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static ROUTE_WORD_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static MERGE_VALIDATION_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(in crate::operator::rolling) static QUALIFICATION_ENTITY_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    pub(super) static ROUTE_ENTITY_DECODE_VISITS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
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
pub(crate) struct LaneRow {
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
    routes: Vec<Arc<LaneRoute>>,
    columns: Vec<TypedGroupInput>,
    event_times: TimestampMicrosecondArray,
    outputs: Vec<TypedOutputPlan>,
    numeric_groups: Option<NumericGroups>,
    numerical_profile: RollingNumericalProfile,
    nan_as_value: bool,
    node_id: String,
}

struct NumericGroups;

impl NumericGroups {
    fn compile(plan: &RollingKernelPlan) -> Option<Self> {
        plan.groups
            .iter()
            .all(|group| matches!(group, TypedGroupPlan::Numeric { .. }))
            .then_some(Self)
    }
}

pub(crate) struct NumericLaneRequest {
    input: Arc<PreparedNumericInput>,
    row_count: usize,
    pub(super) rows: Arc<LaneRoute>,
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

    pub(super) fn use_explicit_route_for_test(&mut self) {
        mutable_route_for_test(&mut self.rows, &self.input);
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
    pub(super) rows: Arc<LaneRoute>,
    columns: Vec<Float64Array>,
}

#[cfg(test)]
impl NumericLaneResult {
    pub(super) fn columns_for_test(&mut self) -> &mut Vec<Float64Array> {
        &mut self.columns
    }

    pub(super) fn rows_for_test(&self) -> Vec<LaneRow> {
        let input = self.input.as_ref().expect("test input remains owned");
        let mut cursor = self.rows.cursor(&input.entity_ids, &input.assignments);
        std::iter::from_fn(|| cursor.next_row())
            .map(|row| row.expect("valid fixture route"))
            .collect()
    }

    pub(super) fn rows_mut_for_test(&mut self) -> &mut Vec<LaneRow> {
        mutable_route_for_test(
            &mut self.rows,
            self.input.as_ref().expect("test input remains owned"),
        )
    }
}

#[cfg(test)]
fn mutable_route_for_test<'a>(
    route: &'a mut Arc<LaneRoute>,
    input: &PreparedNumericInput,
) -> &'a mut Vec<LaneRow> {
    let replacement = (!route.is_explicit()).then(|| {
        let mut cursor = route.cursor(&input.entity_ids, &input.assignments);
        std::iter::from_fn(|| cursor.next_row())
            .map(|row| row.expect("valid fixture route"))
            .collect()
    });
    let route = Arc::make_mut(route);
    match replacement {
        Some(rows) => route.replace_explicit(rows),
        None => route.explicit_mut().expect("explicit route"),
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
    routing: RoutingLayout,
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
        Self::for_dimensions(1_000_000, 64, 3, 1)
    }

    #[cfg(test)]
    pub(super) fn for_dimensions(
        rows: usize,
        entities: usize,
        outputs: usize,
        name_bytes: usize,
    ) -> Self {
        Self::for_lanes(rows, entities, outputs, 0, name_bytes, 2)
    }

    fn for_lanes(
        rows: usize,
        entities: usize,
        outputs: usize,
        groups: usize,
        name_bytes: usize,
        lanes: usize,
    ) -> Self {
        let routing = RoutingLayout::choose(rows, lanes);
        let bytes = routing
            .and_then(|_| Self::owned_capacity(rows, entities, outputs, groups, name_bytes, lanes));
        Self {
            bytes,
            lanes,
            routing: routing.unwrap_or(RoutingLayout::Explicit),
        }
    }

    fn owned_capacity(
        rows: usize,
        entities: usize,
        outputs: usize,
        groups: usize,
        name_bytes: usize,
        lanes: usize,
    ) -> Option<usize> {
        let routing = Self::routing_capacity(rows, entities, lanes)?;
        let output = Self::output_capacity(rows, outputs, lanes)?;
        let fixed_metadata = Self::fixed_capacity(lanes, groups, outputs)?;
        routing
            .checked_add(output)?
            .checked_add(fixed_metadata)?
            .checked_add(Self::name_capacity(name_bytes, lanes)?)
    }

    fn fixed_capacity(lanes: usize, groups: usize, outputs: usize) -> Option<usize> {
        let controls = lanes.checked_mul(4096 + COMPILED_LANE_READOUT_METADATA_BYTES / 2)?;
        let descriptors = Self::descriptor_capacity(groups, outputs)?;
        let lane_headers = Self::lane_header_capacity(lanes)?;
        let merge_headers = Self::merge_header_capacity(outputs)?;
        let fixed_headers = NULL_FREE_MERGE_METADATA_BYTES
            + size_of::<NumericResults>()
            + size_of::<LaneRoutes>()
            + size_of::<PreparedNumericInput>();
        controls
            .checked_add(descriptors)?
            .checked_add(lane_headers)?
            .checked_add(merge_headers)?
            .checked_add(fixed_headers)
    }

    fn descriptor_capacity(groups: usize, outputs: usize) -> Option<usize> {
        groups
            .checked_mul(size_of::<TypedGroupInput>())?
            .checked_add(outputs.checked_mul(size_of::<TypedOutputPlan>())?)
    }

    fn lane_header_capacity(lanes: usize) -> Option<usize> {
        lanes.checked_mul(
            size_of::<NumericLaneRequest>()
                + size_of::<NumericLaneResult>()
                + size_of::<Vec<&mut TypedEntityState>>()
                + 2 * size_of::<LaneRoute>()
                + size_of::<Arc<LaneRoute>>()
                + 2 * size_of::<usize>()
                + 2 * size_of::<RouteCursor<'_>>(),
        )
    }

    fn merge_header_capacity(outputs: usize) -> Option<usize> {
        outputs
            .checked_mul(size_of::<Vec<f64>>() + size_of::<&[f64]>())?
            .checked_add(size_of::<Vec<&[f64]>>())
    }

    fn name_capacity(name_bytes: usize, lanes: usize) -> Option<usize> {
        name_bytes.checked_mul(lanes.checked_mul(2)?.checked_add(1)?)
    }

    fn routing_capacity(rows: usize, entities: usize, lanes: usize) -> Option<usize> {
        let schedules = RoutingLayout::choose(rows, lanes)?.payload_bytes(rows, lanes)?;
        // Charge the lane slot and a conservative Arc owner allowance; the
        // prepared entity payload is shared with the serial fallback.
        let moved = entities.checked_mul(
            size_of::<OwnedLaneEntity>()
                + size_of::<TypedEntityState>()
                + 3 * size_of::<usize>()
                + size_of::<&mut TypedEntityState>(),
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
pub(crate) struct EntityAssignment {
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

pub(in crate::operator::rolling) struct EligibleOrderedKernel<'a> {
    prepared: PreparedOrderedKernel<'a>,
    scratch: ScratchPlan,
}

impl RollingKernelPlan {
    /// The sole route-off control point; managed owner initialization is outside it.
    pub(in crate::operator::rolling) fn supports_entity_parallel(
        &self,
        input: &RecordBatch,
    ) -> bool {
        self.entity_parallel_fallback(input).is_none()
    }

    pub(in crate::operator::rolling) fn entity_parallel_fallback(
        &self,
        input: &RecordBatch,
    ) -> Option<RollingWork> {
        if input.num_rows() < 64_000 {
            return Some(RollingWork::ParallelFallbackSmallBatch);
        }
        let supported = self.parallel_dimensions(input)
            && self
                .groups
                .iter()
                .all(|group| !matches!(group, TypedGroupPlan::Scan { .. }))
            && self
                .outputs
                .iter()
                .all(|output| self.parallel_output(output));
        (!supported).then_some(RollingWork::ParallelFallbackUnsupportedPlan)
    }

    fn parallel_dimensions(&self, input: &RecordBatch) -> bool {
        u32::try_from(input.num_rows()).is_ok()
            && self.selection == KernelSelection::OrderedPrimitive
            && !self.groups.is_empty()
            && !self.outputs.is_empty()
    }

    fn parallel_output(&self, output: &TypedOutputPlan) -> bool {
        output.storage == OutputStorage::Float64
            && match output.kind {
                TypedOutputKind::Difference { left, right } => [left, right]
                    .iter()
                    .all(|readout| readout.group < self.groups.len()),
                TypedOutputKind::Scan(_) => false,
                _ => output.group < self.groups.len(),
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
        let mut fill = self.prepare_typed_fill(input, stream, node_id, observer)?;
        if self.shared_rows.is_some() {
            for entity in &mut fill.stream.state.states {
                Arc::make_mut(entity).prepare_shared_rows(self.shared_rows);
            }
        }
        Ok(PreparedOrderedKernel {
            plan: self,
            node_id,
            fill,
            parallel_shape: self.supports_entity_parallel(input),
        })
    }
}

impl<'a> PreparedOrderedKernel<'a> {
    #[cfg(test)]
    pub(in crate::operator::rolling) fn scratch_plan(&self) -> Option<ScratchPlan> {
        self.scratch_plan_for_cpu(2).ok()
    }

    pub(in crate::operator::rolling) fn scratch_plan_for_cpu(
        &self,
        cpus: usize,
    ) -> std::result::Result<ScratchPlan, RollingWork> {
        let plan = self.plan;
        let states = &self.fill.stream.state.states;
        if !self.parallel_shape
            || states.len() < 2
            || !states
                .iter()
                .all(|entity| parallel_entity_state(entity, &plan.groups))
        {
            return Err(RollingWork::ParallelFallbackPreparedIneligible);
        }
        let lanes = cpus
            .min(self.fill.row_count / ROWS_PER_LANE)
            .min(states.len());
        if lanes < 2 {
            return Err(RollingWork::ParallelFallbackPreparedIneligible);
        }
        let scratch = ScratchPlan::for_lanes(
            self.fill.row_count,
            states.len(),
            plan.outputs.len(),
            plan.groups.len(),
            self.node_id.len(),
            lanes,
        );
        scratch
            .accounted_bytes()
            .map(|_| scratch)
            .ok_or(RollingWork::ParallelFallbackScratchUnrepresentable)
    }

    #[allow(
        clippy::result_large_err,
        reason = "rejection preserves the same owned preparation without another allocation"
    )]
    pub(in crate::operator::rolling) fn qualify_parallel(
        self,
        cpus: usize,
    ) -> std::result::Result<EligibleOrderedKernel<'a>, (Self, RollingWork)> {
        match self.scratch_plan_for_cpu(cpus) {
            Ok(scratch) => Ok(EligibleOrderedKernel {
                prepared: self,
                scratch,
            }),
            Err(reason) => Err((self, reason)),
        }
    }

    pub(in crate::operator::rolling) fn finish_serial(
        self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<StreamKernelUpdate> {
        let mut work = RollingWorkBatch::new(observer);
        work.add(RollingWork::NumericSerialCalls, 1);
        self.plan
            .finish_typed_fill(self.fill, self.node_id, observer)
            .map(|execution| StreamKernelUpdate {
                execution,
                in_place: None,
            })
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
    #[cfg(test)]
    pub(in crate::operator::rolling) fn split_lanes(
        self,
        cpus: usize,
    ) -> std::result::Result<(NumericJoinSeed, Vec<NumericLaneRequest>), Self> {
        let eligible = match self.qualify_parallel(cpus) {
            Ok(eligible) => eligible,
            Err((prepared, _)) => return Err(prepared),
        };
        eligible.split()
    }
}

impl<'a> EligibleOrderedKernel<'a> {
    pub(in crate::operator::rolling) const fn scratch(&self) -> ScratchPlan {
        self.scratch
    }

    pub(in crate::operator::rolling) fn finish_serial(
        self,
        observer: Option<&RollingMetricsRecorder>,
    ) -> Result<StreamKernelUpdate> {
        self.prepared.finish_serial(observer)
    }

    #[allow(
        clippy::result_large_err,
        reason = "partition rejection returns the same funded serial preparation"
    )]
    pub(in crate::operator::rolling) fn split(
        self,
    ) -> std::result::Result<(NumericJoinSeed, Vec<NumericLaneRequest>), PreparedOrderedKernel<'a>>
    {
        let Self { prepared, scratch } = self;
        let lanes = scratch.lanes();
        let Some(LaneRoutes {
            assignments,
            loads,
            entities,
        }) = lane_routes(&prepared.fill, prepared.plan.outputs.len(), lanes)
        else {
            return Err(prepared);
        };
        let Some(routes) = prepare_routes(&prepared.fill, &assignments, &loads, scratch.routing)
        else {
            return Err(prepared);
        };
        let PreparedTypedFill {
            columns,
            event_times,
            row_count,
            mut stream,
        } = prepared.fill;
        let mut lane_entities = entities
            .iter()
            .map(|&count| Vec::with_capacity(count))
            .collect::<Vec<_>>();
        for (local_id, state) in stream.state.states.drain(..).enumerate() {
            lane_entities[assignments[local_id].lane].push(OwnedLaneEntity { local_id, state });
        }
        let shared = Arc::new(PreparedNumericInput {
            entity_ids: stream.entity_ids,
            assignments,
            routes,
            columns,
            event_times,
            outputs: prepared.plan.outputs.clone(),
            numeric_groups: NumericGroups::compile(prepared.plan),
            numerical_profile: prepared.plan.numerical_profile,
            nan_as_value: prepared.plan.nan_as_value,
            node_id: prepared.node_id.to_owned(),
        });
        let requests = (0..lanes)
            .map(|lane| NumericLaneRequest {
                input: Arc::clone(&shared),
                row_count: loads[lane],
                rows: Arc::clone(&shared.routes[lane]),
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
            output_count: prepared.plan.outputs.len(),
            #[cfg(test)]
            panic_before_output_finish: false,
        };
        Ok((seed, requests))
    }
}

fn prepare_routes(
    fill: &PreparedTypedFill,
    assignments: &[EntityAssignment],
    loads: &[usize],
    layout: RoutingLayout,
) -> Option<Vec<Arc<LaneRoute>>> {
    let mut routes = loads
        .iter()
        .enumerate()
        .map(|(lane, &rows)| LaneRoute::new(layout, lane, fill.row_count, rows))
        .collect::<Vec<_>>();
    for (row, &entity) in fill.stream.entity_ids.iter().enumerate() {
        append_routed_row(&mut routes, assignments, entity, row)?;
    }
    if routes
        .iter()
        .enumerate()
        .any(|(lane, route)| !route.complete(lane, fill.row_count))
    {
        return None;
    }
    let mut anchored = Vec::with_capacity(routes.len());
    for route in routes {
        anchored.push(Arc::new(route));
    }
    Some(anchored)
}

fn append_routed_row(
    routes: &mut [LaneRoute],
    assignments: &[EntityAssignment],
    entity: usize,
    row: usize,
) -> Option<()> {
    let assignment = assignments.get(entity)?;
    routes.get_mut(assignment.lane)?.push(LaneRow {
        original_row: u32::try_from(row).ok()?,
        lane_entity: u32::try_from(assignment.local).ok()?,
    })
}

struct LaneRoutes {
    assignments: Vec<EntityAssignment>,
    loads: Vec<usize>,
    entities: Vec<usize>,
}

fn lane_routes(fill: &PreparedTypedFill, outputs: usize, lanes: usize) -> Option<LaneRoutes> {
    let mut assignments = vec![EntityAssignment::default(); fill.stream.state.states.len()];
    for (assignment, &count) in assignments.iter_mut().zip(&fill.stream.entity_rows.counts) {
        #[cfg(test)]
        ENTITY_COUNT_VISITS.set(ENTITY_COUNT_VISITS.get() + 1);
        assignment.rows = count;
    }
    let (weights, total_work) = entity_work_weights(fill, outputs)?;
    let mut work_loads = vec![0_usize; lanes];
    let mut loads = vec![0_usize; lanes];
    let mut entities = vec![0_usize; lanes];
    let mut available = (0..lanes)
        .map(|lane| Reverse((0_usize, lane)))
        .collect::<BinaryHeap<_>>();
    for (assignment, weight) in assignments.iter_mut().zip(weights) {
        let Reverse((_, lane)) = available.pop().expect("nonempty lane plan");
        assignment.lane = lane;
        assignment.local = entities[lane];
        entities[lane] += 1;
        let load = loads[lane].checked_add(assignment.rows)?;
        loads[lane] = load;
        let work = work_loads[lane].checked_add(weight)?;
        work_loads[lane] = work;
        available.push(Reverse((work, lane)));
    }
    if !lane_rows_cover_input(&loads, fill.row_count)
        || !balanced_lane_loads(&work_loads, total_work)
    {
        return None;
    }
    Some(LaneRoutes {
        assignments,
        loads,
        entities,
    })
}

fn lane_rows_cover_input(loads: &[usize], rows: usize) -> bool {
    !loads.contains(&0)
        && loads
            .iter()
            .try_fold(0_usize, |sum, &rows| sum.checked_add(rows))
            == Some(rows)
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
    #[cfg(test)]
    QUALIFICATION_ENTITY_VISITS.set(QUALIFICATION_ENTITY_VISITS.get() + 1);
    entity.groups.len() == groups.len()
        && entity.shared_rows_match_groups(groups)
        && entity
            .groups
            .iter()
            .zip(groups)
            .all(|(state, group)| match (state, group) {
                (TypedWindowState::Numeric(state), TypedGroupPlan::Numeric { frame, .. }) => {
                    parallel_frame(state.samples.frame, *frame, state.samples.len())
                }
                (
                    TypedWindowState::Exact(state),
                    TypedGroupPlan::Signed { frame, .. } | TypedGroupPlan::Unsigned { frame, .. },
                ) => parallel_frame(state.samples.frame, *frame, state.samples.len()),
                (
                    TypedWindowState::Extrema(state),
                    TypedGroupPlan::Extrema {
                        frame, descending, ..
                    },
                ) => {
                    state.descending == *descending
                        && parallel_frame(state.frame, *frame, state.values.len())
                }
                (TypedWindowState::Pair(state), TypedGroupPlan::Pair { frame, .. }) => {
                    parallel_frame(state.samples.frame, *frame, state.samples.len())
                }
                (
                    TypedWindowState::Ewma(state),
                    TypedGroupPlan::Ewma {
                        alpha, cumulative, ..
                    },
                ) => state.alpha.to_bits() == alpha.to_bits() && state.cumulative == *cumulative,
                (TypedWindowState::Cumulative(_), TypedGroupPlan::Cumulative { .. }) => true,
                _ => false,
            })
}

fn parallel_frame(actual: TypedFrame, expected: TypedFrame, retained: usize) -> bool {
    match (actual, expected) {
        (TypedFrame::Rows(actual), TypedFrame::Rows(expected)) => {
            actual == expected && retained <= expected
        }
        (TypedFrame::Duration(actual), TypedFrame::Duration(expected)) => actual == expected,
        _ => false,
    }
}

// Scheduling weight is a work heuristic, not a bound on profile-specific refolds.
fn entity_work_weights(fill: &PreparedTypedFill, outputs: usize) -> Option<(Vec<usize>, usize)> {
    if fill.stream.state.states.len() != fill.stream.entity_rows.counts.len() {
        return None;
    }
    fill.stream
        .state
        .states
        .iter()
        .zip(&fill.stream.entity_rows.counts)
        .map(|(entity, &rows)| {
            let steps = entity.groups.len().checked_add(outputs)?;
            let retained = entity.groups.iter().try_fold(0_usize, |work, state| {
                work.checked_add(retained_group_work(state)?)
            })?;
            let shared = entity
                .shared_rows
                .as_ref()
                .map_or(0, |tape| tape.bytes().div_ceil(size_of::<Option<f64>>()));
            rows.checked_mul(steps)?
                .checked_add(retained)?
                .checked_add(shared)
        })
        .collect::<Option<Vec<_>>>()
        .and_then(|weights| {
            weights
                .iter()
                .try_fold(0_usize, |sum, &work| sum.checked_add(work))
                .map(|total| (weights, total))
        })
}

fn retained_group_work(state: &TypedWindowState) -> Option<usize> {
    match state {
        TypedWindowState::Numeric(state) => Some(state.samples.len()),
        TypedWindowState::Exact(state) => Some(state.samples.len()),
        TypedWindowState::Extrema(state) => state.values.len().checked_add(state.candidates.len()),
        TypedWindowState::Pair(state) => state.samples.len().checked_mul(2),
        TypedWindowState::Ewma(_) | TypedWindowState::Cumulative(_) => Some(1),
        TypedWindowState::Decay(_) | TypedWindowState::Scan(_) => None,
    }
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
    if input.numeric_groups.is_some()
        && input
            .outputs
            .iter()
            .all(|output| output.storage == OutputStorage::Float64)
        && input
            .columns
            .iter()
            .all(|column| matches!(column, TypedGroupInput::Single(_)))
    {
        let builders = input
            .outputs
            .iter()
            .map(|_| Float64Builder::with_capacity(request.row_count))
            .collect();
        return run_numeric_lane_with_builders(request, builders, cancellation, progress);
    }
    let builders = input
        .outputs
        .iter()
        .map(|output| DerivedBuilder::new(*output, request.row_count))
        .collect();
    run_numeric_lane_with_builders(request, builders, cancellation, progress)
}

fn run_numeric_lane_with_builders<B: NumericLaneBuilder>(
    request: NumericLaneRequest,
    builders: Vec<B>,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    let input = &request.input;
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
            run_compiled_numeric_lane::<1, _>(request, builders, output, cancellation, progress)
        }
        (2, Some(output)) => {
            run_compiled_numeric_lane::<2, _>(request, builders, output, cancellation, progress)
        }
        _ => {
            run_prepared_numeric_lane::<0, _>(request, builders, cancellation, progress, B::append)
        }
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

fn run_compiled_numeric_lane<const GROUPS: usize, B: NumericLaneBuilder>(
    request: NumericLaneRequest,
    builders: Vec<B>,
    output: TypedOutputPlan,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    match output.kind {
        TypedOutputKind::Statistic(Statistic::Mean) => run_prepared_numeric_lane::<GROUPS, _>(
            request,
            builders,
            cancellation,
            progress,
            move |builder, states, _, node_id| builder.append_mean(states, &output, node_id),
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
            run_prepared_numeric_lane::<GROUPS, _>(
                request,
                builders,
                cancellation,
                progress,
                move |builder, states, _, _| builder.append_mean_difference(states, left, right),
            )
        }
        _ => {
            run_prepared_numeric_lane::<0, _>(request, builders, cancellation, progress, B::append)
        }
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
    let value = super::mean::read_numeric_mean(state, output.min_periods);
    if value.is_none() {
        builder.append_null();
        return Ok(());
    }
    let DerivedBuilder::Float(values, _) = builder else {
        return Err(typed_output_mismatch());
    };
    values.append_option(value);
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
    Ok(super::mean::read_numeric_mean(state, readout.min_periods))
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
    run_prepared_numeric_lane::<0, _>(
        request,
        builders,
        cancellation,
        progress,
        DerivedBuilder::append,
    )
}

trait NumericLaneBuilder: Sized {
    fn append(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()>;
    fn append_mean(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()>;
    fn append_mean_difference(
        &mut self,
        states: &[TypedWindowState],
        left: LaneMeanReadout,
        right: LaneMeanReadout,
    ) -> Result<()>;
    fn finish_column(self) -> Float64Array;
}

impl NumericLaneBuilder for DerivedBuilder {
    fn append(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()> {
        DerivedBuilder::append(self, states, output, node)
    }
    fn append_mean(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()> {
        append_numeric_lane_mean(self, states, output, node)
    }
    fn append_mean_difference(
        &mut self,
        states: &[TypedWindowState],
        left: LaneMeanReadout,
        right: LaneMeanReadout,
    ) -> Result<()> {
        append_numeric_lane_difference(self, states, left, right)
    }
    fn finish_column(self) -> Float64Array {
        match self {
            Self::Float(mut values, OutputStorage::Float64) => values.finish(),
            _ => unreachable!("qualified lane output is Float64"),
        }
    }
}

impl NumericLaneBuilder for Float64Builder {
    fn append(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()> {
        append_numeric_float64(self, states, output, node)
    }
    fn append_mean(
        &mut self,
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node: &str,
    ) -> Result<()> {
        if let Some(TypedWindowState::Numeric(state)) = states.get(output.group) {
            self.append_option(super::mean::read_numeric_mean(state, output.min_periods));
            return Ok(());
        }
        with_float64_builder(self, |builder| {
            append_numeric_lane_mean(builder, states, output, node)
        })
    }
    fn append_mean_difference(
        &mut self,
        states: &[TypedWindowState],
        left: LaneMeanReadout,
        right: LaneMeanReadout,
    ) -> Result<()> {
        let left = read_numeric_lane_mean(states, left)?;
        let right = read_numeric_lane_mean(states, right)?;
        self.append_option(left.zip(right).map(|(left, right)| left - right));
        Ok(())
    }
    fn finish_column(mut self) -> Float64Array {
        Float64Builder::finish(&mut self)
    }
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
        builder_count: usize,
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
                builder_count.min(input.outputs.len())
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

    fn numeric_values<'b>(
        &'b self,
        column: &'b TypedGroupInput,
        ordinal: usize,
    ) -> Option<&'b Float64Array> {
        if GROUPS != 0 {
            return Some(self.fixed_columns[ordinal].0);
        }
        match column {
            TypedGroupInput::Single(values) => Some(values),
            _ => None,
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
        if entity.shared_rows.is_some() {
            let TypedGroupInput::Single(values) = self.column(0) else {
                unreachable!("shared lane rows have Single inputs");
            };
            entity.push_shared_row(valid_float64(
                values,
                row.original_row as usize,
                self.input.nan_as_value,
            ));
        }
        for ordinal in 0..self.group_count.min(entity.groups.len()) {
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
            self.update_group_state(entity, column, site)
                .map_err(|error| NumericLaneStop::Ordinary(TaggedNumericError { site, error }))?;
            #[cfg(test)]
            self.observe(NumericTestPoint::AfterGroup(site));
            check_cancelled(self.cancellation, progress)?;
        }
        if entity.shared_rows.is_some() {
            entity.trim_shared_rows();
        }
        Ok(())
    }

    fn update_group_state(
        &self,
        entity: &mut TypedEntityState,
        column: &TypedGroupInput,
        site: NumericSite,
    ) -> Result<()> {
        if entity.shared_rows.is_some() {
            let TypedGroupInput::Single(values) = self.column(0) else {
                unreachable!("shared lane rows have Single inputs");
            };
            return entity.update_shared_group(
                site.ordinal,
                valid_float64(values, site.row as usize, self.input.nan_as_value),
                self.input.numerical_profile,
                &self.input.node_id,
            );
        }
        let transition_count = entity.transition_count;
        let state = &mut entity.groups[site.ordinal];
        // Expiration and numerical repair use the serial transition unchanged.
        let event_time = self.input.event_times.value(site.row as usize);
        match (state, self.numeric_values(column, site.ordinal)) {
            (TypedWindowState::Numeric(state), Some(values)) => super::mean::update_numeric_group(
                state,
                event_time,
                valid_float64(values, site.row as usize, self.input.nan_as_value),
                self.input.numerical_profile,
                transition_count,
                &self.input.node_id,
            ),
            (state, _) => state.update(
                event_time,
                column,
                site.row as usize,
                self.input.numerical_profile,
                self.input.nan_as_value,
                transition_count,
                &self.input.node_id,
            ),
        }
    }

    fn append_outputs<B>(
        &self,
        builders: &mut [B],
        entity: &TypedEntityState,
        row: LaneRow,
        progress: &mut LaneProgress,
        append: &mut impl FnMut(&mut B, &[TypedWindowState], &TypedOutputPlan, &str) -> Result<()>,
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

fn run_prepared_numeric_lane<const GROUPS: usize, B: NumericLaneBuilder>(
    mut request: NumericLaneRequest,
    mut builders: Vec<B>,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
    mut append: impl FnMut(&mut B, &[TypedWindowState], &TypedOutputPlan, &str) -> Result<()>,
) -> std::result::Result<NumericLaneResult, NumericLaneStop> {
    // Borrow only the immutable input and hook, leaving owned row/state buffers mutable.
    let context = NumericLaneContext::<GROUPS>::new(
        &request.input,
        builders.len(),
        cancellation,
        #[cfg(test)]
        request.test_hook.as_ref(),
    );
    progress.active_site = None;
    // Prepared lane entities are already unique; borrow each once per lane.
    let mut entities = request
        .entities
        .iter_mut()
        .map(|entity| {
            let state = Arc::make_mut(&mut entity.state);
            state.prepare_lane_shared_rows(&request.input.columns);
            state
        })
        .collect::<Vec<_>>();
    assert!(
        request.rows.is_explicit()
            || request
                .input
                .routes
                .get(request.rows.lane())
                .is_some_and(|route| Arc::ptr_eq(route, &request.rows)),
        "numeric lane bitmap is not producer-qualified"
    );
    let mut cursor = request
        .rows
        .cursor(&request.input.entity_ids, &request.input.assignments);
    let mut ordinal = 0;
    while let Some(row) = next_lane_row(&mut cursor, cancellation, progress)? {
        #[cfg(test)]
        SCHEDULE_VISITS.set(SCHEDULE_VISITS.get() + 1);
        context.check_row(ordinal, progress)?;
        let entity = &mut *entities[usize::try_from(row.lane_entity).expect("u32 fits platform")];
        context.update_entity(entity, row, progress)?;
        context.append_outputs(&mut builders, entity, row, progress, &mut append)?;
        progress.active_site = None;
        ordinal += 1;
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

fn next_lane_row(
    cursor: &mut RouteCursor<'_>,
    cancellation: &AtomicBool,
    progress: &mut LaneProgress,
) -> std::result::Result<Option<LaneRow>, NumericLaneStop> {
    while let Some(step) = cursor.next_step() {
        match step {
            RouteStep::Poll => check_cancelled(cancellation, progress)?,
            RouteStep::Row(row) => return Ok(Some(row)),
            RouteStep::Invalid => panic!("producer-qualified numeric lane bitmap changed"),
        }
    }
    Ok(None)
}

fn finish_numeric_columns<B: NumericLaneBuilder>(builders: Vec<B>) -> Vec<Float64Array> {
    builders
        .into_iter()
        .map(NumericLaneBuilder::finish_column)
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
    Many(Vec<Vec<f64>>),
    Nullable(Vec<Float64Builder>),
}

#[derive(Clone, Copy)]
struct NumericOutputShape<'a> {
    results: &'a [NumericLaneResult],
    count: usize,
}

struct TrustedNumericRoutes<'a> {
    input: &'a PreparedNumericInput,
    shape: NumericOutputShape<'a>,
}

// Two producer lanes already fund a second full cursor per lane during merge.
const _: () = assert!(
    2 * size_of::<NumericOutputShape<'_>>()
        + size_of::<TrustedNumericRoutes<'_>>()
        + size_of::<GlobalRowCursor<'_>>()
        <= 2 * size_of::<RouteCursor<'_>>()
);

// Fixed output headers and the largest pair of logical column-view arrays.
pub(super) const NULL_FREE_MERGE_METADATA_BYTES: usize =
    size_of::<NumericOutputBuffers>() + size_of::<[[&[f64]; 2]; 2]>();

impl NumericOutputBuffers {
    fn gather(input: &PreparedNumericInput, shape: NumericOutputShape<'_>) -> Result<Self> {
        let NumericOutputShape { results, count } = shape;
        let null_free = count > 0
            && results
                .iter()
                .all(|result| result.columns.iter().all(|column| column.null_count() == 0));
        if null_free && let Some(proof) = trusted_numeric_routes(input, shape) {
            return Ok(match count {
                1 => Self::One(gather_trusted_numeric_columns(&proof)),
                2 => Self::Two(gather_trusted_numeric_columns(&proof)),
                _ => Self::Many(gather_trusted_numeric_columns_many(&proof)),
            });
        }
        Self::gather_validated(input, results, count, null_free)
    }

    fn gather_validated(
        input: &PreparedNumericInput,
        results: &[NumericLaneResult],
        count: usize,
        null_free: bool,
    ) -> Result<Self> {
        let entity_ids = &input.entity_ids;
        let assignments = &input.assignments;
        let rows = entity_ids.len();
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
            _ => Self::gather_nullable(input, results, count),
        }
    }

    fn gather_nullable(
        input: &PreparedNumericInput,
        results: &[NumericLaneResult],
        count: usize,
    ) -> Result<Self> {
        let mut builders = (0..count)
            .map(|_| Float64Builder::with_capacity(input.entity_ids.len()))
            .collect::<Vec<_>>();
        merge_numeric_rows(
            &input.entity_ids,
            &input.assignments,
            results,
            |lane, offset| {
                for (builder, column) in builders.iter_mut().zip(&results[lane].columns) {
                    if column.is_null(offset) {
                        builder.append_null();
                    } else {
                        builder.append_value(column.value(offset));
                    }
                }
            },
        )?;
        Ok(Self::Nullable(builders))
    }

    fn finish(self) -> Vec<ArrayRef> {
        match self {
            Self::One(values) => finish_null_free_numeric_columns(values),
            Self::Two(values) => finish_null_free_numeric_columns(values),
            Self::Many(values) => values
                .into_iter()
                .map(|values| Arc::new(Float64Array::from(values)) as ArrayRef)
                .collect(),
            Self::Nullable(builders) => builders
                .into_iter()
                .map(|mut builder| Arc::new(builder.finish()) as ArrayRef)
                .collect(),
        }
    }
}

fn trusted_numeric_routes<'a>(
    input: &'a PreparedNumericInput,
    shape: NumericOutputShape<'a>,
) -> Option<TrustedNumericRoutes<'a>> {
    (shape.results.len() >= 2
        && anchored_numeric_routes(input, shape.results)
        && numeric_route_totals_match(input, shape.results)
        && numeric_route_entities_match(input, shape.results))
    .then_some(TrustedNumericRoutes { input, shape })
}

fn anchored_numeric_routes(input: &PreparedNumericInput, results: &[NumericLaneResult]) -> bool {
    input.routes.len() == results.len()
        && input
            .routes
            .iter()
            .zip(results)
            .enumerate()
            .all(|(lane, (route, result))| {
                Arc::ptr_eq(route, &result.rows) && route.complete(lane, input.entity_ids.len())
            })
}

fn numeric_route_totals_match(input: &PreparedNumericInput, results: &[NumericLaneResult]) -> bool {
    results
        .iter()
        .try_fold((0_usize, 0_usize), |(rows, entities), result| {
            Some((
                rows.checked_add(result.rows.len())?,
                entities.checked_add(result.entities.len())?,
            ))
        })
        == Some((input.entity_ids.len(), input.assignments.len()))
}

fn numeric_route_entities_match(
    input: &PreparedNumericInput,
    results: &[NumericLaneResult],
) -> bool {
    input
        .assignments
        .iter()
        .enumerate()
        .all(|(id, assignment)| {
            results
                .get(assignment.lane)
                .and_then(|result| result.entities.get(assignment.local))
                .is_some_and(|entity| entity.local_id == id)
        })
}

fn gather_trusted_numeric_columns<const COLUMNS: usize>(
    proof: &TrustedNumericRoutes<'_>,
) -> [Vec<f64>; COLUMNS] {
    let input = proof.input;
    let results = proof.shape.results;
    let mut values = std::array::from_fn(|_| vec![0.0; input.entity_ids.len()]);
    for (lane, result) in results.iter().enumerate() {
        let columns: [&[f64]; COLUMNS] =
            std::array::from_fn(|column| result.columns[column].values().as_ref());
        let mut cursor = GlobalRowCursor::new(proof, lane);
        let mut offset = 0;
        while let Some(row) = cursor.next_row() {
            for (values, column) in values.iter_mut().zip(columns) {
                values[row] = column[offset];
            }
            offset += 1;
        }
    }
    values
}

fn gather_trusted_numeric_columns_many(proof: &TrustedNumericRoutes<'_>) -> Vec<Vec<f64>> {
    let input = proof.input;
    let NumericOutputShape { results, count } = proof.shape;
    let mut values = (0..count)
        .map(|_| vec![0.0; input.entity_ids.len()])
        .collect::<Vec<_>>();
    for (lane, result) in results.iter().enumerate() {
        let columns = result
            .columns
            .iter()
            .map(|column| column.values().as_ref())
            .collect::<Vec<_>>();
        let mut cursor = GlobalRowCursor::new(proof, lane);
        let mut offset = 0;
        while let Some(row) = cursor.next_row() {
            for (values, column) in values.iter_mut().zip(&columns) {
                values[row] = column[offset];
            }
            offset += 1;
        }
    }
    values
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
    append: impl FnMut(usize, usize),
) -> Result<()> {
    if let [left, right] = results {
        let mut cursors = [
            left.rows.cursor(entity_ids, assignments),
            right.rows.cursor(entity_ids, assignments),
        ];
        return merge_cursor_rows(entity_ids, assignments, results, &mut cursors, append);
    }
    let mut cursors = results
        .iter()
        .map(|result| result.rows.cursor(entity_ids, assignments))
        .collect::<Vec<_>>();
    merge_cursor_rows(entity_ids, assignments, results, &mut cursors, append)
}

fn merge_cursor_rows(
    entity_ids: &[usize],
    assignments: &[EntityAssignment],
    results: &[NumericLaneResult],
    cursors: &mut [RouteCursor<'_>],
    mut append: impl FnMut(usize, usize),
) -> Result<()> {
    for (global, expected_entity) in entity_ids.iter().enumerate() {
        #[cfg(test)]
        MERGE_VALIDATION_VISITS.set(MERGE_VALIDATION_VISITS.get() + 1);
        let lane = assignments
            .get(*expected_entity)
            .map(|assignment| assignment.lane)
            .filter(|&lane| lane < results.len())
            .ok_or_else(|| {
                internal_error("numeric lane row coverage or entity identity changed")
            })?;
        let offset =
            validate_cursor_row(&mut cursors[lane], &results[lane], global, *expected_entity)?;
        append(lane, offset);
    }
    if cursors.iter_mut().zip(results).any(|(cursor, result)| {
        cursor.consumed() != result.rows.len() || cursor.next_row().is_some()
    }) {
        return Err(internal_error("numeric lane row coverage has extra rows"));
    }
    Ok(())
}

fn validate_cursor_row(
    cursor: &mut RouteCursor<'_>,
    result: &NumericLaneResult,
    global: usize,
    expected_entity: usize,
) -> Result<usize> {
    let offset = cursor.consumed();
    if offset >= result.rows.len() {
        return Err(internal_error("numeric lane row coverage is incomplete"));
    }
    let row = cursor
        .next_row()
        .ok_or_else(|| internal_error("numeric lane row coverage is incomplete"))?
        .map_err(|()| internal_error("numeric lane row coverage or entity identity changed"))?;
    if row.original_row as usize != global
        || result
            .entities
            .get(row.lane_entity as usize)
            .map(|entity| entity.local_id)
            != Some(expected_entity)
    {
        return Err(internal_error(
            "numeric lane row coverage or entity identity changed",
        ));
    }
    Ok(offset)
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
    let shape = validate_numeric_shape(results, count)?;
    let buffers = NumericOutputBuffers::gather(&input, shape)?;
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
        in_place: None,
    })
}

fn validate_numeric_shape(
    results: &[NumericLaneResult],
    count: usize,
) -> Result<NumericOutputShape<'_>> {
    if results.iter().any(|result| {
        result.columns.len() != count
            || result.columns.iter().any(|column| {
                column.data_type() != &DataType::Float64 || column.len() != result.rows.len()
            })
    }) {
        return Err(internal_error("numeric lane output shape changed"));
    }
    Ok(NumericOutputShape { results, count })
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
