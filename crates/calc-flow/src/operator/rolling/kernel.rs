//! Typed, allocation-bounded rolling kernels selected from the semantic plan.
//!
//! This module deliberately owns no watermark or checkpoint policy. Callers
//! may invoke a typed kernel only after the finality layer has established the
//! rows that are safe to process.

use std::{
    collections::{HashMap, VecDeque},
    mem::size_of,
    time::{Duration, Instant},
};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Float64Builder, TimestampMicrosecondArray, UInt64Builder,
    },
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use sha2::{Digest, Sha256};

use super::{
    CompiledEvaluation, CompiledFrame, CompiledKeyColumn, CompiledRollingOutput,
    CompiledWindowGroup, Statistic, SumClass, SumState, WindowAccumulator, internal_error,
    operator_error,
};
use crate::Result;

const ROLLING_KERNEL_PLAN_VERSION: u32 = 1;
const STABLE_NUMERICAL_PROFILE: &str = "stable_v1";

/// Selected executable family for one immutable rolling kernel plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelSelection {
    /// Columnar `Float64` numeric aggregates over bounded row frames.
    OrderedFloat64Rows,
    /// The semantic shape requires the general row kernel.
    General,
}

/// Per-row transition complexity advertised by the selected kernel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelComplexity {
    AmortizedConstant,
    General,
}

/// Immutable execution plan derived from a validated rolling declaration.
#[derive(Clone, Debug)]
pub(super) struct RollingKernelPlan {
    version: u32,
    state_layout_version: u32,
    numerical_profile: &'static str,
    selection: KernelSelection,
    complexity: KernelComplexity,
    order_columns: Vec<usize>,
    partition_columns: Vec<usize>,
    sequence_columns: Vec<usize>,
    groups: Vec<Float64GroupPlan>,
    outputs: Vec<Float64OutputPlan>,
    fallback_reason: Option<String>,
    estimated_state_bytes_per_entity: usize,
    fingerprint: String,
}

#[derive(Clone, Copy, Debug)]
struct Float64GroupPlan {
    input_index: usize,
    frame: TypedFrame,
}

#[derive(Clone, Copy, Debug)]
enum TypedFrame {
    Rows(usize),
    Duration(u64),
}

impl TypedFrame {
    const fn estimated_samples(self) -> usize {
        match self {
            Self::Rows(rows) => rows,
            Self::Duration(_) => 0,
        }
    }
}

impl TryFrom<CompiledFrame> for TypedFrame {
    type Error = String;

    fn try_from(frame: CompiledFrame) -> std::result::Result<Self, Self::Error> {
        match frame {
            CompiledFrame::Rows(rows) => usize::try_from(rows)
                .map(Self::Rows)
                .map_err(|_| "row window does not fit the platform usize".to_owned()),
            CompiledFrame::Duration(micros) => Ok(Self::Duration(micros)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Float64OutputPlan {
    group: usize,
    statistic: Statistic,
    min_periods: u64,
    ddof: u8,
}

/// Stage-level facts from one typed kernel invocation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(super) struct RollingKernelMetrics {
    pub input_validation_ns: u64,
    pub order_proof_ns: u64,
    pub sort_permutation_ns: u64,
    pub gather_ns: u64,
    pub entity_encode_ns: u64,
    pub kernel_ns: u64,
    pub output_build_ns: u64,
    pub order_proof_rows: usize,
    pub entities: usize,
    pub input_rows: usize,
    pub output_rows: usize,
    pub state_bytes: usize,
    pub scalar_value_conversions: usize,
    pub sort_count: usize,
}

/// Typed columns and execution facts produced without rebuilding input rows.
#[derive(Debug)]
pub(super) struct RollingKernelExecution {
    pub columns: Vec<ArrayRef>,
    pub metrics: RollingKernelMetrics,
    pub state: RollingKernelState,
}

/// Opaque, cloneable transition state for one typed kernel plan.
#[derive(Clone, Debug, Default)]
pub(super) struct RollingKernelState {
    kernel_fingerprint: Option<String>,
    entities: HashMap<Vec<u8>, usize>,
    states: Vec<Float64EntityState>,
    last_identity: Option<Vec<u8>>,
}

impl RollingKernelPlan {
    pub(super) fn compile(
        input_schema: &Schema,
        state_layout_version: u32,
        event_time_index: usize,
        partition_columns: &[CompiledKeyColumn],
        sequence_columns: &[CompiledKeyColumn],
        outputs: &[CompiledRollingOutput],
        window_groups: &[CompiledWindowGroup],
    ) -> Self {
        let order_columns = std::iter::once(event_time_index)
            .chain(partition_columns.iter().map(|column| column.index))
            .chain(sequence_columns.iter().map(|column| column.index))
            .collect::<Vec<_>>();
        let partition_columns = partition_columns
            .iter()
            .map(|column| column.index)
            .collect::<Vec<_>>();
        let sequence_columns = sequence_columns
            .iter()
            .map(|column| column.index)
            .collect::<Vec<_>>();
        let compiled = compile_float64_rows(input_schema, outputs, window_groups);
        let (selection, complexity, groups, typed_outputs, fallback_reason) = match compiled {
            Ok((groups, typed_outputs)) => (
                KernelSelection::OrderedFloat64Rows,
                KernelComplexity::AmortizedConstant,
                groups,
                typed_outputs,
                None,
            ),
            Err(reason) => (
                KernelSelection::General,
                KernelComplexity::General,
                Vec::new(),
                Vec::new(),
                Some(reason),
            ),
        };
        let estimated_state_bytes_per_entity = groups.iter().fold(0_usize, |total, group| {
            total.saturating_add(
                size_of::<Float64WindowState>().saturating_add(
                    group
                        .frame
                        .estimated_samples()
                        .saturating_mul(size_of::<TimedFloat64Sample>()),
                ),
            )
        });
        let fingerprint = kernel_fingerprint(&KernelFingerprintInput {
            input_schema,
            state_layout_version,
            selection,
            complexity,
            order_columns: &order_columns,
            partition_columns: &partition_columns,
            groups: &groups,
            outputs: &typed_outputs,
            fallback_reason: fallback_reason.as_deref(),
        });
        Self {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version,
            numerical_profile: STABLE_NUMERICAL_PROFILE,
            selection,
            complexity,
            order_columns,
            partition_columns,
            sequence_columns,
            groups,
            outputs: typed_outputs,
            fallback_reason,
            estimated_state_bytes_per_entity,
            fingerprint,
        }
    }

    pub(super) const fn selection(&self) -> KernelSelection {
        self.selection
    }

    pub(super) fn fallback_reason(&self) -> Option<&str> {
        self.fallback_reason.as_deref()
    }

    pub(super) fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub(super) const fn numerical_profile(&self) -> &'static str {
        self.numerical_profile
    }

    pub(super) const fn version(&self) -> u32 {
        self.version
    }

    pub(super) const fn state_layout_version(&self) -> u32 {
        self.state_layout_version
    }

    pub(super) const fn complexity(&self) -> KernelComplexity {
        self.complexity
    }

    pub(super) const fn estimated_state_bytes_per_entity(&self) -> usize {
        self.estimated_state_bytes_per_entity
    }

    pub(super) const fn supports_typed_transition(&self) -> bool {
        matches!(self.selection, KernelSelection::OrderedFloat64Rows)
    }

    /// Executes one already-final, canonical-order candidate batch.
    ///
    /// `Ok(None)` means ordering was not proven and the caller must use the
    /// general sort-capable path. Semantic or Arrow validation failures remain
    /// errors and never silently fall back.
    pub(super) fn open_and_fill(
        &self,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<Option<RollingKernelExecution>> {
        self.update_and_fill(&RollingKernelState::default(), input, node_id)
    }

    /// Applies one canonical micro-batch to a scratch clone and returns the
    /// next state only after every transition and output succeeds.
    // Validation, ordering, entity routing, and scratch-state construction
    // deliberately remain one atomic preparation boundary.
    // #lizard forgives
    pub(super) fn update_and_fill(
        &self,
        state: &RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<Option<RollingKernelExecution>> {
        if self.selection != KernelSelection::OrderedFloat64Rows {
            return Ok(None);
        }
        self.validate_state(state, node_id)?;

        let validation_start = Instant::now();
        self.validate_required_values(input, node_id)?;
        let input_validation_ns = nanos(validation_start.elapsed());

        let order_start = Instant::now();
        let order_rows = encode_rows(input, &self.order_columns, node_id)?;
        if !self.canonical_order_is_proven(input, &order_rows, state, node_id)? {
            return Ok(None);
        }
        let last_identity = input
            .num_rows()
            .checked_sub(1)
            .map(|row_index| order_rows.row(row_index).data().to_vec());
        let order_proof_ns = nanos(order_start.elapsed());

        let entity_start = Instant::now();
        let entity_rows = encode_rows(input, &self.partition_columns, node_id)?;
        let entity_keys = encoded_keys(&entity_rows, input.num_rows());
        let mut next_state = state.clone();
        next_state
            .kernel_fingerprint
            .get_or_insert_with(|| self.fingerprint.clone());
        let entity_ids = next_state.resolve_entities(&entity_keys, &self.groups);
        let entity_encode_ns = nanos(entity_start.elapsed());

        self.fill_float64(
            input,
            &entity_ids,
            next_state,
            last_identity,
            node_id,
            RollingKernelMetrics {
                input_validation_ns,
                order_proof_ns,
                entity_encode_ns,
                order_proof_rows: input.num_rows(),
                input_rows: input.num_rows(),
                output_rows: input.num_rows(),
                ..RollingKernelMetrics::default()
            },
        )
        .map(Some)
    }

    fn validate_state(&self, state: &RollingKernelState, node_id: &str) -> Result<()> {
        if state
            .kernel_fingerprint
            .as_deref()
            .is_some_and(|fingerprint| fingerprint != self.fingerprint)
        {
            return Err(operator_error(
                node_id,
                "typed rolling state belongs to a different kernel plan",
            ));
        }
        Ok(())
    }

    // Cross-batch and within-batch ordering are one canonical identity proof.
    // #lizard forgives
    fn canonical_order_is_proven(
        &self,
        input: &RecordBatch,
        order_rows: &datafusion::arrow::row::Rows,
        state: &RollingKernelState,
        node_id: &str,
    ) -> Result<bool> {
        if input.num_rows() > 0
            && let Some(previous) = state.last_identity.as_deref()
        {
            let current = order_rows.row(0).data();
            if previous == current {
                return Err(duplicate_identity_error(
                    input,
                    self.order_columns[0],
                    0,
                    node_id,
                )?);
            }
            if previous > current {
                return Ok(false);
            }
        }
        for row_index in 1..input.num_rows() {
            let previous = order_rows.row(row_index - 1);
            let current = order_rows.row(row_index);
            if previous == current {
                return Err(duplicate_identity_error(
                    input,
                    self.order_columns[0],
                    row_index,
                    node_id,
                )?);
            }
            if previous > current {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn fill_float64(
        &self,
        input: &RecordBatch,
        entity_ids: &[usize],
        mut state: RollingKernelState,
        last_identity: Option<Vec<u8>>,
        node_id: &str,
        mut metrics: RollingKernelMetrics,
    ) -> Result<RollingKernelExecution> {
        let inputs = self.float64_inputs(input, node_id)?;
        let event_times = input
            .column(self.order_columns[0])
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| {
                operator_error(
                    node_id,
                    "rolling event-time value is not a microsecond timestamp",
                )
            })?;
        let mut builders = self
            .outputs
            .iter()
            .map(|output| DerivedBuilder::new(output.statistic, input.num_rows()))
            .collect::<Vec<_>>();

        let kernel_start = Instant::now();
        fill_float64_rows(
            &self.outputs,
            &inputs,
            event_times,
            &mut state.states,
            &mut builders,
            entity_ids,
            node_id,
        )?;
        let kernel_ns = nanos(kernel_start.elapsed());

        let output_start = Instant::now();
        let columns = builders
            .into_iter()
            .map(DerivedBuilder::finish)
            .collect::<Vec<_>>();
        let output_build_ns = nanos(output_start.elapsed());
        let state_bytes = state
            .states
            .iter()
            .map(Float64EntityState::estimated_bytes)
            .sum();
        if last_identity.is_some() {
            state.last_identity = last_identity;
        }
        metrics.kernel_ns = kernel_ns;
        metrics.output_build_ns = output_build_ns;
        metrics.entities = state.states.len();
        metrics.state_bytes = state_bytes;
        Ok(RollingKernelExecution {
            columns,
            metrics,
            state,
        })
    }

    fn validate_required_values(&self, input: &RecordBatch, node_id: &str) -> Result<()> {
        let event_time_index = self.order_columns[0];
        if input.column(event_time_index).null_count() > 0 {
            return Err(operator_error(
                node_id,
                "rolling event-time value is null or not a microsecond timestamp",
            ));
        }
        if self
            .sequence_columns
            .iter()
            .any(|&index| input.column(index).null_count() > 0)
        {
            return Err(operator_error(
                node_id,
                "rolling sequence key value is null",
            ));
        }
        Ok(())
    }

    fn float64_inputs<'a>(
        &self,
        input: &'a RecordBatch,
        node_id: &str,
    ) -> Result<Vec<&'a Float64Array>> {
        self.groups
            .iter()
            .map(|group| {
                input.columns()[group.input_index]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| operator_error(node_id, "typed rolling input is not Float64"))
            })
            .collect()
    }
}

impl RollingKernelState {
    fn resolve_entities(&mut self, keys: &[Vec<u8>], groups: &[Float64GroupPlan]) -> Vec<usize> {
        let mut counts = HashMap::<&[u8], usize>::new();
        for key in keys {
            let count = counts.entry(key.as_slice()).or_default();
            *count = count.saturating_add(1);
        }
        keys.iter()
            .map(|key| {
                if let Some(&entity_id) = self.entities.get(key.as_slice()) {
                    return entity_id;
                }
                let entity_id = self.states.len();
                let row_count = counts[key.as_slice()];
                self.entities.insert(key.clone(), entity_id);
                self.states.push(Float64EntityState::new(groups, row_count));
                entity_id
            })
            .collect()
    }
}

fn fill_float64_rows(
    outputs: &[Float64OutputPlan],
    inputs: &[&Float64Array],
    event_times: &TimestampMicrosecondArray,
    states: &mut [Float64EntityState],
    builders: &mut [DerivedBuilder],
    entity_ids: &[usize],
    node_id: &str,
) -> Result<()> {
    for (row_index, &entity_id) in entity_ids.iter().enumerate() {
        let entity = &mut states[entity_id];
        update_float64_groups(
            inputs,
            event_times.value(row_index),
            entity,
            row_index,
            node_id,
        )?;
        append_float64_outputs(outputs, builders, entity)?;
    }
    Ok(())
}

fn update_float64_groups(
    inputs: &[&Float64Array],
    event_time: i64,
    entity: &mut Float64EntityState,
    row_index: usize,
    node_id: &str,
) -> Result<()> {
    for (group_index, input) in inputs.iter().enumerate() {
        let sample = if input.is_null(row_index) || input.value(row_index).is_nan() {
            None
        } else {
            Some(input.value(row_index))
        };
        entity.groups[group_index].update(event_time, sample, node_id)?;
    }
    Ok(())
}

fn append_float64_outputs(
    outputs: &[Float64OutputPlan],
    builders: &mut [DerivedBuilder],
    entity: &Float64EntityState,
) -> Result<()> {
    for (builder, output) in builders.iter_mut().zip(outputs) {
        builder.append(&entity.groups[output.group].accumulator, output)?;
    }
    Ok(())
}

fn compile_float64_rows(
    input_schema: &Schema,
    outputs: &[CompiledRollingOutput],
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<(Vec<Float64GroupPlan>, Vec<Float64OutputPlan>), String> {
    let groups = compile_float64_groups(input_schema, window_groups)?;
    let typed_outputs = compile_float64_outputs(outputs, groups.len())?;
    if groups.is_empty() || typed_outputs.is_empty() {
        return Err("requires at least one Float64 row aggregate".into());
    }
    Ok((groups, typed_outputs))
}

fn compile_float64_groups(
    input_schema: &Schema,
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<Vec<Float64GroupPlan>, String> {
    let mut groups = Vec::with_capacity(window_groups.len());
    for group in window_groups {
        let CompiledWindowGroup::Numeric {
            input_index,
            frame,
            sum_class: SumClass::Float,
        } = group
        else {
            return Err("requires a non-Float64, duration, extrema, pair, or EWMA group".into());
        };
        if input_schema.field(*input_index).data_type() != &DataType::Float64 {
            return Err("requires every aggregate input to be Float64".into());
        }
        let frame = TypedFrame::try_from(*frame)?;
        groups.push(Float64GroupPlan {
            input_index: *input_index,
            frame,
        });
    }
    Ok(groups)
}

fn compile_float64_outputs(
    outputs: &[CompiledRollingOutput],
    group_count: usize,
) -> std::result::Result<Vec<Float64OutputPlan>, String> {
    let mut typed_outputs = Vec::with_capacity(outputs.len());
    for output in outputs {
        let CompiledEvaluation::Aggregate(aggregate) = &output.evaluation else {
            return Err("requires aggregate-only outputs".into());
        };
        if !matches!(
            aggregate.statistic,
            Statistic::Count
                | Statistic::Sum
                | Statistic::Mean
                | Statistic::Variance
                | Statistic::Stddev
        ) {
            return Err("requires count, sum, mean, variance, or stddev outputs".into());
        }
        if aggregate.group >= group_count {
            return Err("aggregate group index is outside the typed plan".into());
        }
        typed_outputs.push(Float64OutputPlan {
            group: aggregate.group,
            statistic: aggregate.statistic,
            min_periods: aggregate.min_periods,
            ddof: aggregate.ddof,
        });
    }
    Ok(typed_outputs)
}

fn duplicate_identity_error(
    input: &RecordBatch,
    event_time_index: usize,
    row_index: usize,
    node_id: &str,
) -> Result<crate::CalcFlowError> {
    let event_time = input
        .column(event_time_index)
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .ok_or_else(|| {
            operator_error(
                node_id,
                "rolling event-time value is not a microsecond timestamp",
            )
        })?
        .value(row_index);
    Ok(operator_error(
        node_id,
        &format!("duplicate row identity at event_time_micros={event_time}"),
    ))
}

struct KernelFingerprintInput<'a> {
    input_schema: &'a Schema,
    state_layout_version: u32,
    selection: KernelSelection,
    complexity: KernelComplexity,
    order_columns: &'a [usize],
    partition_columns: &'a [usize],
    groups: &'a [Float64GroupPlan],
    outputs: &'a [Float64OutputPlan],
    fallback_reason: Option<&'a str>,
}

fn kernel_fingerprint(input: &KernelFingerprintInput<'_>) -> String {
    let schema = input
        .input_schema
        .fields()
        .iter()
        .enumerate()
        .map(|(index, field)| {
            format!(
                "{index}:{}:{:?}:{}",
                field.name(),
                field.data_type(),
                field.is_nullable()
            )
        })
        .collect::<Vec<_>>();
    let descriptor = format!(
        "version={ROLLING_KERNEL_PLAN_VERSION}|state={}|numeric={STABLE_NUMERICAL_PROFILE}|selection={:?}|complexity={:?}|schema={schema:?}|order={:?}|partition={:?}|groups={:?}|outputs={:?}|fallback={:?}",
        input.state_layout_version,
        input.selection,
        input.complexity,
        input.order_columns,
        input.partition_columns,
        input.groups,
        input.outputs,
        input.fallback_reason,
    );
    hex::encode(Sha256::digest(descriptor.as_bytes()))
}

fn encode_rows(
    input: &RecordBatch,
    indices: &[usize],
    node_id: &str,
) -> Result<datafusion::arrow::row::Rows> {
    let arrays = indices
        .iter()
        .map(|&index| input.column(index).clone())
        .collect::<Vec<_>>();
    let fields = arrays
        .iter()
        .map(|array| SortField::new(array.data_type().clone()))
        .collect::<Vec<_>>();
    RowConverter::new(fields)
        .and_then(|converter| converter.convert_columns(&arrays))
        .map_err(|error| {
            operator_error(
                node_id,
                &format!("typed rolling key encoding failed: {error}"),
            )
        })
}

fn encoded_keys(entity_rows: &datafusion::arrow::row::Rows, row_count: usize) -> Vec<Vec<u8>> {
    (0..row_count)
        .map(|row_index| entity_rows.row(row_index).data().to_vec())
        .collect()
}

#[derive(Clone, Debug)]
struct Float64EntityState {
    groups: Vec<Float64WindowState>,
}

impl Float64EntityState {
    fn new(groups: &[Float64GroupPlan], entity_rows: usize) -> Self {
        Self {
            groups: groups
                .iter()
                .map(|group| Float64WindowState::with_frame(group.frame, entity_rows))
                .collect(),
        }
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .groups
                .iter()
                .map(Float64WindowState::estimated_bytes)
                .sum::<usize>()
    }
}

#[derive(Clone, Debug)]
struct Float64WindowState {
    frame: TypedFrame,
    values: VecDeque<TimedFloat64Sample>,
    accumulator: WindowAccumulator,
}

#[derive(Clone, Copy, Debug)]
struct TimedFloat64Sample {
    event_time: i64,
    value: Option<f64>,
}

impl Float64WindowState {
    fn with_frame(frame: TypedFrame, entity_rows: usize) -> Self {
        let capacity = frame.estimated_samples().min(entity_rows);
        Self {
            frame,
            values: VecDeque::with_capacity(capacity),
            accumulator: WindowAccumulator::new(SumClass::Float),
        }
    }

    fn update(&mut self, event_time: i64, sample: Option<f64>, node_id: &str) -> Result<()> {
        if let Some(sample) = sample {
            add_float64(&mut self.accumulator, sample, node_id)?;
        }
        self.values.push_back(TimedFloat64Sample {
            event_time,
            value: sample,
        });
        self.expire(event_time)?;
        if self.accumulator.is_non_finite() {
            refold_float64(&mut self.accumulator, &self.values, node_id)?;
        }
        Ok(())
    }

    fn expire(&mut self, event_time: i64) -> Result<()> {
        match self.frame {
            TypedFrame::Rows(window) => {
                if self.values.len() > window {
                    self.remove_front()?;
                }
            }
            TypedFrame::Duration(micros) => {
                let bound = i128::from(event_time) - i128::from(micros);
                while self
                    .values
                    .front()
                    .is_some_and(|sample| i128::from(sample.event_time) <= bound)
                {
                    self.remove_front()?;
                }
            }
        }
        Ok(())
    }

    fn remove_front(&mut self) -> Result<()> {
        if let Some(Some(expiring)) = self.values.pop_front().map(|sample| sample.value) {
            remove_float64(&mut self.accumulator, expiring)?;
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<Option<f64>>()
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen mean and variance output type is Float64"
)]
fn add_float64(accumulator: &mut WindowAccumulator, sample: f64, node_id: &str) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_add(1)
        .ok_or_else(|| operator_error(node_id, "rolling valid sample count overflowed"))?;
    let Some(SumState::Float(total)) = &mut accumulator.sum else {
        return Err(internal_error(
            "typed Float64 rolling group has a non-floating sum state",
        ));
    };
    *total += sample;
    if sample.is_infinite() {
        if sample > 0.0 {
            accumulator.pos_inf = accumulator.pos_inf.saturating_add(1);
        } else {
            accumulator.neg_inf = accumulator.neg_inf.saturating_add(1);
        }
    }
    let count = accumulator.valid_count as f64;
    let delta = sample - accumulator.mean;
    accumulator.mean += delta / count;
    accumulator.m2 += delta * (sample - accumulator.mean);
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen mean and variance output type is Float64"
)]
fn remove_float64(accumulator: &mut WindowAccumulator, sample: f64) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_sub(1)
        .ok_or_else(|| internal_error("rolling removal without a matching add"))?;
    let Some(SumState::Float(total)) = &mut accumulator.sum else {
        return Err(internal_error(
            "typed Float64 rolling group has a non-floating sum state",
        ));
    };
    *total -= sample;
    if sample.is_infinite() {
        if sample > 0.0 {
            accumulator.pos_inf = accumulator.pos_inf.saturating_sub(1);
        } else {
            accumulator.neg_inf = accumulator.neg_inf.saturating_sub(1);
        }
    }
    if accumulator.valid_count == 0 {
        accumulator.mean = 0.0;
        accumulator.m2 = 0.0;
    } else {
        let count = accumulator.valid_count as f64;
        let delta = sample - accumulator.mean;
        accumulator.mean -= delta / count;
        accumulator.m2 -= delta * (sample - accumulator.mean);
    }
    Ok(())
}

fn refold_float64(
    accumulator: &mut WindowAccumulator,
    values: &VecDeque<TimedFloat64Sample>,
    node_id: &str,
) -> Result<()> {
    *accumulator = WindowAccumulator::new(SumClass::Float);
    for sample in values.iter().filter_map(|sample| sample.value) {
        add_float64(accumulator, sample, node_id)?;
    }
    Ok(())
}

enum DerivedBuilder {
    Count(UInt64Builder),
    Float(Float64Builder),
}

impl DerivedBuilder {
    fn new(statistic: Statistic, capacity: usize) -> Self {
        match statistic {
            Statistic::Count => Self::Count(UInt64Builder::with_capacity(capacity)),
            Statistic::Sum | Statistic::Mean | Statistic::Variance | Statistic::Stddev => {
                Self::Float(Float64Builder::with_capacity(capacity))
            }
            Statistic::Min | Statistic::Max => {
                unreachable!("extrema are excluded from the typed Float64 plan")
            }
        }
    }

    fn append(
        &mut self,
        accumulator: &WindowAccumulator,
        output: &Float64OutputPlan,
    ) -> Result<()> {
        if accumulator.valid_count < output.min_periods {
            self.append_null();
            return Ok(());
        }
        match (self, output.statistic) {
            (Self::Count(builder), Statistic::Count) => {
                builder.append_value(accumulator.valid_count);
            }
            (Self::Float(builder), statistic) => {
                append_float(builder, accumulator, statistic, output.ddof)?;
            }
            _ => {
                return Err(internal_error(
                    "typed rolling output builder does not match its statistic",
                ));
            }
        }
        Ok(())
    }

    fn append_null(&mut self) {
        match self {
            Self::Count(builder) => builder.append_null(),
            Self::Float(builder) => builder.append_null(),
        }
    }

    fn finish(self) -> ArrayRef {
        match self {
            Self::Count(mut builder) => std::sync::Arc::new(builder.finish()),
            Self::Float(mut builder) => std::sync::Arc::new(builder.finish()),
        }
    }
}

fn append_float(
    builder: &mut Float64Builder,
    accumulator: &WindowAccumulator,
    statistic: Statistic,
    ddof: u8,
) -> Result<()> {
    match statistic {
        Statistic::Sum => append_sum(builder, accumulator),
        Statistic::Mean => {
            builder.append_value(float_mean(accumulator));
            Ok(())
        }
        Statistic::Variance | Statistic::Stddev => {
            append_dispersion(builder, accumulator, statistic, ddof);
            Ok(())
        }
        _ => Err(internal_error(
            "typed rolling float builder received a non-floating statistic",
        )),
    }
}

fn append_sum(builder: &mut Float64Builder, accumulator: &WindowAccumulator) -> Result<()> {
    let Some(SumState::Float(total)) = accumulator.sum else {
        return Err(internal_error(
            "typed Float64 sum has a non-floating accumulator",
        ));
    };
    builder.append_value(total);
    Ok(())
}

fn float_mean(accumulator: &WindowAccumulator) -> f64 {
    match (accumulator.pos_inf > 0, accumulator.neg_inf > 0) {
        (true, true) => f64::NAN,
        (true, false) => f64::INFINITY,
        (false, true) => f64::NEG_INFINITY,
        (false, false) => accumulator.mean,
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen variance output type is Float64"
)]
fn append_dispersion(
    builder: &mut Float64Builder,
    accumulator: &WindowAccumulator,
    statistic: Statistic,
    ddof: u8,
) {
    let divisor = accumulator.valid_count - u64::from(ddof);
    if divisor == 0 {
        builder.append_null();
        return;
    }
    if accumulator.pos_inf > 0 || accumulator.neg_inf > 0 {
        builder.append_value(f64::NAN);
        return;
    }
    let variance = accumulator.m2.max(0.0) / divisor as f64;
    builder.append_value(if statistic == Statistic::Stddev {
        variance.sqrt()
    } else {
        variance
    });
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
