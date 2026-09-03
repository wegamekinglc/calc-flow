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
    CompiledEvaluation, CompiledKeyColumn, CompiledRollingOutput, CompiledWindowGroup, Statistic,
    SumClass, SumState, WindowAccumulator, internal_error, operator_error,
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
    window: usize,
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
                size_of::<Float64WindowState>()
                    .saturating_add(group.window.saturating_mul(size_of::<Option<f64>>())),
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
        if self.selection != KernelSelection::OrderedFloat64Rows {
            return Ok(None);
        }

        let validation_start = Instant::now();
        self.validate_required_values(input, node_id)?;
        let input_validation_ns = nanos(validation_start.elapsed());

        let order_start = Instant::now();
        let order_rows = encode_rows(input, &self.order_columns, node_id)?;
        for row_index in 1..input.num_rows() {
            let previous = order_rows.row(row_index - 1);
            let current = order_rows.row(row_index);
            if previous == current {
                let event_time = input
                    .column(self.order_columns[0])
                    .as_any()
                    .downcast_ref::<TimestampMicrosecondArray>()
                    .ok_or_else(|| {
                        operator_error(
                            node_id,
                            "rolling event-time value is not a microsecond timestamp",
                        )
                    })?
                    .value(row_index);
                return Err(operator_error(
                    node_id,
                    &format!("duplicate row identity at event_time_micros={event_time}"),
                ));
            }
            if previous > current {
                return Ok(None);
            }
        }
        let order_proof_ns = nanos(order_start.elapsed());

        let entity_start = Instant::now();
        let entity_rows = encode_rows(input, &self.partition_columns, node_id)?;
        let (entity_ids, entity_counts) = dense_entity_ids(&entity_rows, input.num_rows());
        let entity_encode_ns = nanos(entity_start.elapsed());

        let inputs = self.float64_inputs(input, node_id)?;
        let mut states = entity_counts
            .iter()
            .map(|count| Float64EntityState::new(&self.groups, *count))
            .collect::<Vec<_>>();
        let mut builders = self
            .outputs
            .iter()
            .map(|output| DerivedBuilder::new(output.statistic, input.num_rows()))
            .collect::<Vec<_>>();

        let kernel_start = Instant::now();
        for (row_index, &entity_id) in entity_ids.iter().enumerate() {
            let entity = &mut states[entity_id];
            for (group_index, group) in self.groups.iter().enumerate() {
                let input = inputs[group_index];
                let sample = if input.is_null(row_index) || input.value(row_index).is_nan() {
                    None
                } else {
                    Some(input.value(row_index))
                };
                entity.groups[group_index].update(sample, node_id)?;
                debug_assert_eq!(entity.groups[group_index].window, group.window);
            }
            for (builder, output) in builders.iter_mut().zip(&self.outputs) {
                builder.append(&entity.groups[output.group].accumulator, output)?;
            }
        }
        let kernel_ns = nanos(kernel_start.elapsed());

        let output_start = Instant::now();
        let columns = builders
            .into_iter()
            .map(DerivedBuilder::finish)
            .collect::<Vec<_>>();
        let output_build_ns = nanos(output_start.elapsed());
        let state_bytes = states.iter().map(Float64EntityState::estimated_bytes).sum();
        Ok(Some(RollingKernelExecution {
            columns,
            metrics: RollingKernelMetrics {
                input_validation_ns,
                order_proof_ns,
                entity_encode_ns,
                kernel_ns,
                output_build_ns,
                order_proof_rows: input.num_rows(),
                entities: states.len(),
                input_rows: input.num_rows(),
                output_rows: input.num_rows(),
                state_bytes,
                ..RollingKernelMetrics::default()
            },
        }))
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

fn compile_float64_rows(
    input_schema: &Schema,
    outputs: &[CompiledRollingOutput],
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<(Vec<Float64GroupPlan>, Vec<Float64OutputPlan>), String> {
    let mut groups = Vec::with_capacity(window_groups.len());
    for group in window_groups {
        let CompiledWindowGroup::Numeric {
            input_index,
            frame: super::CompiledFrame::Rows(rows),
            sum_class: SumClass::Float,
        } = group
        else {
            return Err("requires a non-Float64, duration, extrema, pair, or EWMA group".into());
        };
        if input_schema.field(*input_index).data_type() != &DataType::Float64 {
            return Err("requires every aggregate input to be Float64".into());
        }
        let window = usize::try_from(*rows)
            .map_err(|_| "row window does not fit the platform usize".to_owned())?;
        groups.push(Float64GroupPlan {
            input_index: *input_index,
            window,
        });
    }
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
        if aggregate.group >= groups.len() {
            return Err("aggregate group index is outside the typed plan".into());
        }
        typed_outputs.push(Float64OutputPlan {
            group: aggregate.group,
            statistic: aggregate.statistic,
            min_periods: aggregate.min_periods,
            ddof: aggregate.ddof,
        });
    }
    if groups.is_empty() || typed_outputs.is_empty() {
        return Err("requires at least one Float64 row aggregate".into());
    }
    Ok((groups, typed_outputs))
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

fn dense_entity_ids(
    entity_rows: &datafusion::arrow::row::Rows,
    row_count: usize,
) -> (Vec<usize>, Vec<usize>) {
    let mut entities = HashMap::<Vec<u8>, usize>::new();
    let mut entity_ids = Vec::with_capacity(row_count);
    let mut entity_counts = Vec::<usize>::new();
    for row_index in 0..row_count {
        let key = entity_rows.row(row_index).data();
        let entity_id = if let Some(&entity_id) = entities.get(key) {
            entity_id
        } else {
            let entity_id = entity_counts.len();
            entities.insert(key.to_vec(), entity_id);
            entity_counts.push(0);
            entity_id
        };
        entity_counts[entity_id] = entity_counts[entity_id].saturating_add(1);
        entity_ids.push(entity_id);
    }
    (entity_ids, entity_counts)
}

#[derive(Debug)]
struct Float64EntityState {
    groups: Vec<Float64WindowState>,
}

impl Float64EntityState {
    fn new(groups: &[Float64GroupPlan], entity_rows: usize) -> Self {
        Self {
            groups: groups
                .iter()
                .map(|group| Float64WindowState::new(group.window, entity_rows))
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

#[derive(Debug)]
struct Float64WindowState {
    window: usize,
    values: VecDeque<Option<f64>>,
    accumulator: WindowAccumulator,
}

impl Float64WindowState {
    fn new(window: usize, entity_rows: usize) -> Self {
        Self {
            window,
            values: VecDeque::with_capacity(window.min(entity_rows)),
            accumulator: WindowAccumulator::new(SumClass::Float),
        }
    }

    fn update(&mut self, sample: Option<f64>, node_id: &str) -> Result<()> {
        if let Some(sample) = sample {
            add_float64(&mut self.accumulator, sample, node_id)?;
        }
        self.values.push_back(sample);
        if self.values.len() > self.window
            && let Some(Some(expiring)) = self.values.pop_front()
        {
            remove_float64(&mut self.accumulator, expiring)?;
        }
        if self.accumulator.is_non_finite() {
            refold_float64(&mut self.accumulator, &self.values, node_id)?;
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
    values: &VecDeque<Option<f64>>,
    node_id: &str,
) -> Result<()> {
    *accumulator = WindowAccumulator::new(SumClass::Float);
    for sample in values.iter().flatten() {
        add_float64(accumulator, *sample, node_id)?;
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
            (Self::Float(builder), Statistic::Sum) => {
                let Some(SumState::Float(total)) = accumulator.sum else {
                    return Err(internal_error(
                        "typed Float64 sum has a non-floating accumulator",
                    ));
                };
                builder.append_value(total);
            }
            (Self::Float(builder), Statistic::Mean) => {
                builder.append_value(match (accumulator.pos_inf > 0, accumulator.neg_inf > 0) {
                    (true, true) => f64::NAN,
                    (true, false) => f64::INFINITY,
                    (false, true) => f64::NEG_INFINITY,
                    (false, false) => accumulator.mean,
                });
            }
            (Self::Float(builder), Statistic::Variance | Statistic::Stddev) => {
                let divisor = accumulator.valid_count - u64::from(output.ddof);
                if divisor == 0 {
                    builder.append_null();
                    return Ok(());
                }
                if accumulator.pos_inf > 0 || accumulator.neg_inf > 0 {
                    builder.append_value(f64::NAN);
                    return Ok(());
                }
                let m2 = if accumulator.m2 < 0.0 {
                    0.0
                } else {
                    accumulator.m2
                };
                #[allow(
                    clippy::cast_precision_loss,
                    reason = "the frozen variance output type is Float64"
                )]
                let variance = m2 / divisor as f64;
                builder.append_value(if output.statistic == Statistic::Stddev {
                    variance.sqrt()
                } else {
                    variance
                });
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

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}
