//! Typed, allocation-bounded rolling kernels selected from the semantic plan.
//!
//! This module deliberately owns no watermark or checkpoint policy. Callers
//! may invoke a typed kernel only after the finality layer has established the
//! rows that are safe to process.

use std::{
    cmp::Ordering,
    collections::{HashMap, VecDeque},
    mem::size_of,
    time::{Duration, Instant},
};

use datafusion::arrow::{
    array::{
        Array, ArrayRef, Float64Array, Float64Builder, Int64Array, Int64Builder,
        TimestampMicrosecondArray, UInt64Array, UInt64Builder,
    },
    compute::cast,
    datatypes::{DataType, Schema},
    record_batch::RecordBatch,
    row::{RowConverter, SortField},
};
use sha2::{Digest, Sha256};

use super::{
    CompiledEvaluation, CompiledFrame, CompiledKeyColumn, CompiledRollingOutput,
    CompiledWindowGroup, PairAccumulator, Statistic, SumClass, SumState, WindowAccumulator,
    internal_error, operator_error,
};
use crate::Result;

const ROLLING_KERNEL_PLAN_VERSION: u32 = 1;
const STABLE_NUMERICAL_PROFILE: &str = "stable_v1";

/// Selected executable family for one immutable rolling kernel plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelSelection {
    /// Columnar `Float64` numeric, extrema, and pair aggregates.
    OrderedFloat64,
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
enum Float64GroupPlan {
    Numeric {
        input_index: usize,
        frame: TypedFrame,
    },
    Signed {
        input_index: usize,
        frame: TypedFrame,
    },
    Unsigned {
        input_index: usize,
        frame: TypedFrame,
    },
    Extrema {
        input_index: usize,
        frame: TypedFrame,
        descending: bool,
        storage: OutputStorage,
    },
    Pair {
        left_index: usize,
        right_index: usize,
        frame: TypedFrame,
    },
}

impl Float64GroupPlan {
    const fn frame(self) -> TypedFrame {
        match self {
            Self::Numeric { frame, .. }
            | Self::Signed { frame, .. }
            | Self::Unsigned { frame, .. }
            | Self::Extrema { frame, .. }
            | Self::Pair { frame, .. } => frame,
        }
    }
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
    kind: Float64OutputKind,
    storage: OutputStorage,
    min_periods: u64,
    ddof: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Float64OutputKind {
    Statistic(Statistic),
    Covariance,
    Correlation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum OutputStorage {
    Count,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
}

impl OutputStorage {
    const fn is_signed(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    const fn is_unsigned(self) -> bool {
        matches!(
            self,
            Self::UInt8 | Self::UInt16 | Self::UInt32 | Self::UInt64
        )
    }

    const fn is_float(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }

    const fn data_type(self) -> DataType {
        match self {
            Self::Count | Self::UInt64 => DataType::UInt64,
            Self::Int8 => DataType::Int8,
            Self::Int16 => DataType::Int16,
            Self::Int32 => DataType::Int32,
            Self::Int64 => DataType::Int64,
            Self::UInt8 => DataType::UInt8,
            Self::UInt16 => DataType::UInt16,
            Self::UInt32 => DataType::UInt32,
            Self::Float32 => DataType::Float32,
            Self::Float64 => DataType::Float64,
        }
    }
}

enum Float64GroupInput {
    Single(Float64Array),
    Signed(Int64Array),
    Unsigned(UInt64Array),
    Pair(Float64Array, Float64Array),
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
                KernelSelection::OrderedFloat64,
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
                        .frame()
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
        matches!(self.selection, KernelSelection::OrderedFloat64)
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
        if self.selection != KernelSelection::OrderedFloat64 {
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
            .map(|output| DerivedBuilder::new(*output, input.num_rows()))
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
            .collect::<Result<Vec<_>>>()?;
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

    fn float64_inputs(&self, input: &RecordBatch, node_id: &str) -> Result<Vec<Float64GroupInput>> {
        self.groups
            .iter()
            .map(|group| float64_group_input(group, input, node_id))
            .collect()
    }
}

fn float64_group_input(
    group: &Float64GroupPlan,
    input: &RecordBatch,
    node_id: &str,
) -> Result<Float64GroupInput> {
    match *group {
        Float64GroupPlan::Numeric { input_index, .. } => Ok(Float64GroupInput::Single(
            cast_float64(input.column(input_index), node_id)?,
        )),
        Float64GroupPlan::Extrema {
            input_index,
            storage,
            ..
        } => {
            if storage.is_signed() {
                cast_signed(input.column(input_index), node_id).map(Float64GroupInput::Signed)
            } else if storage.is_unsigned() {
                cast_unsigned(input.column(input_index), node_id).map(Float64GroupInput::Unsigned)
            } else if storage.is_float() {
                cast_float64(input.column(input_index), node_id).map(Float64GroupInput::Single)
            } else {
                Err(internal_error(
                    "typed extrema group has count output storage",
                ))
            }
        }
        Float64GroupPlan::Signed { input_index, .. } => {
            cast_signed(input.column(input_index), node_id).map(Float64GroupInput::Signed)
        }
        Float64GroupPlan::Unsigned { input_index, .. } => {
            cast_unsigned(input.column(input_index), node_id).map(Float64GroupInput::Unsigned)
        }
        Float64GroupPlan::Pair {
            left_index,
            right_index,
            ..
        } => Ok(Float64GroupInput::Pair(
            cast_float64(input.column(left_index), node_id)?,
            cast_float64(input.column(right_index), node_id)?,
        )),
    }
}

fn cast_signed(array: &ArrayRef, node_id: &str) -> Result<Int64Array> {
    cast_primitive(array, &DataType::Int64, node_id)?
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| internal_error("signed rolling cast did not produce Int64"))
}

fn cast_unsigned(array: &ArrayRef, node_id: &str) -> Result<UInt64Array> {
    cast_primitive(array, &DataType::UInt64, node_id)?
        .as_any()
        .downcast_ref::<UInt64Array>()
        .cloned()
        .ok_or_else(|| internal_error("unsigned rolling cast did not produce UInt64"))
}

fn cast_float64(array: &ArrayRef, node_id: &str) -> Result<Float64Array> {
    cast_primitive(array, &DataType::Float64, node_id)?
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| internal_error("floating rolling cast did not produce Float64"))
}

fn cast_primitive(array: &ArrayRef, target: &DataType, node_id: &str) -> Result<ArrayRef> {
    if array.data_type() == target {
        return Ok(array.clone());
    }
    cast(array, target).map_err(|error| {
        operator_error(
            node_id,
            &format!("typed rolling input cast to {target} failed: {error}"),
        )
    })
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
    inputs: &[Float64GroupInput],
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
        append_float64_outputs(outputs, builders, entity, node_id)?;
    }
    Ok(())
}

fn update_float64_groups(
    inputs: &[Float64GroupInput],
    event_time: i64,
    entity: &mut Float64EntityState,
    row_index: usize,
    node_id: &str,
) -> Result<()> {
    for (state, input) in entity.groups.iter_mut().zip(inputs) {
        state.update(event_time, input, row_index, node_id)?;
    }
    Ok(())
}

fn append_float64_outputs(
    outputs: &[Float64OutputPlan],
    builders: &mut [DerivedBuilder],
    entity: &Float64EntityState,
    node_id: &str,
) -> Result<()> {
    for (builder, output) in builders.iter_mut().zip(outputs) {
        builder.append(&entity.groups[output.group], output, node_id)?;
    }
    Ok(())
}

fn compile_float64_rows(
    input_schema: &Schema,
    outputs: &[CompiledRollingOutput],
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<(Vec<Float64GroupPlan>, Vec<Float64OutputPlan>), String> {
    let groups = compile_float64_groups(input_schema, window_groups)?;
    let typed_outputs = compile_float64_outputs(outputs, &groups)?;
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
        groups.push(compile_float64_group(input_schema, group)?);
    }
    Ok(groups)
}

fn compile_float64_group(
    input_schema: &Schema,
    group: &CompiledWindowGroup,
) -> std::result::Result<Float64GroupPlan, String> {
    match group {
        CompiledWindowGroup::Numeric {
            input_index,
            frame,
            sum_class,
        } => compile_numeric_group(input_schema, *input_index, *frame, *sum_class),
        CompiledWindowGroup::Extrema {
            input_index,
            frame,
            descending,
        } => compile_single_group(input_schema, *input_index, *frame, Some(*descending)),
        CompiledWindowGroup::Pair {
            left_index,
            right_index,
            frame,
        } => compile_pair_group(input_schema, *left_index, *right_index, *frame),
        CompiledWindowGroup::Ewma { .. } => {
            Err("requires non-EWMA numeric, extrema, or pair window groups".into())
        }
    }
}

fn compile_numeric_group(
    input_schema: &Schema,
    input_index: usize,
    frame: CompiledFrame,
    sum_class: SumClass,
) -> std::result::Result<Float64GroupPlan, String> {
    let frame = TypedFrame::try_from(frame)?;
    match sum_class {
        SumClass::Float
            if matches!(
                input_schema.field(input_index).data_type(),
                DataType::Float32 | DataType::Float64
            ) =>
        {
            Ok(Float64GroupPlan::Numeric { input_index, frame })
        }
        SumClass::Signed => Ok(Float64GroupPlan::Signed { input_index, frame }),
        SumClass::Unsigned => Ok(Float64GroupPlan::Unsigned { input_index, frame }),
        _ => Err("requires primitive numeric window groups".into()),
    }
}

fn compile_single_group(
    input_schema: &Schema,
    input_index: usize,
    frame: CompiledFrame,
    descending: Option<bool>,
) -> std::result::Result<Float64GroupPlan, String> {
    let frame = TypedFrame::try_from(frame)?;
    Ok(if let Some(descending) = descending {
        Float64GroupPlan::Extrema {
            input_index,
            frame,
            descending,
            storage: extrema_storage(input_schema.field(input_index).data_type())?,
        }
    } else {
        require_float64(input_schema, input_index)?;
        Float64GroupPlan::Numeric { input_index, frame }
    })
}

fn extrema_storage(data_type: &DataType) -> std::result::Result<OutputStorage, String> {
    match data_type {
        DataType::Int8 => Ok(OutputStorage::Int8),
        DataType::Int16 => Ok(OutputStorage::Int16),
        DataType::Int32 => Ok(OutputStorage::Int32),
        DataType::Int64 => Ok(OutputStorage::Int64),
        DataType::UInt8 => Ok(OutputStorage::UInt8),
        DataType::UInt16 => Ok(OutputStorage::UInt16),
        DataType::UInt32 => Ok(OutputStorage::UInt32),
        DataType::UInt64 => Ok(OutputStorage::UInt64),
        DataType::Float32 => Ok(OutputStorage::Float32),
        DataType::Float64 => Ok(OutputStorage::Float64),
        _ => Err("requires primitive numeric extrema input".into()),
    }
}

fn compile_pair_group(
    input_schema: &Schema,
    left_index: usize,
    right_index: usize,
    frame: CompiledFrame,
) -> std::result::Result<Float64GroupPlan, String> {
    require_numeric_type(input_schema, left_index)?;
    require_numeric_type(input_schema, right_index)?;
    Ok(Float64GroupPlan::Pair {
        left_index,
        right_index,
        frame: TypedFrame::try_from(frame)?,
    })
}

fn require_numeric_type(input_schema: &Schema, index: usize) -> std::result::Result<(), String> {
    if matches!(
        input_schema.field(index).data_type(),
        DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
    ) {
        Ok(())
    } else {
        Err("requires primitive numeric pair inputs".into())
    }
}

fn require_float64(input_schema: &Schema, index: usize) -> std::result::Result<(), String> {
    if input_schema.field(index).data_type() != &DataType::Float64 {
        return Err("requires every aggregate input to be Float64".into());
    }
    Ok(())
}

fn compile_float64_outputs(
    outputs: &[CompiledRollingOutput],
    groups: &[Float64GroupPlan],
) -> std::result::Result<Vec<Float64OutputPlan>, String> {
    let mut typed_outputs = Vec::with_capacity(outputs.len());
    for output in outputs {
        let plan = compile_float64_output(output, groups)?;
        if plan.group >= groups.len() {
            return Err("aggregate group index is outside the typed plan".into());
        }
        typed_outputs.push(plan);
    }
    Ok(typed_outputs)
}

fn compile_float64_output(
    output: &CompiledRollingOutput,
    groups: &[Float64GroupPlan],
) -> std::result::Result<Float64OutputPlan, String> {
    match &output.evaluation {
        CompiledEvaluation::Aggregate(aggregate) => Ok(Float64OutputPlan {
            group: aggregate.group,
            kind: Float64OutputKind::Statistic(aggregate.statistic),
            storage: output_storage(aggregate, groups)?,
            min_periods: aggregate.min_periods,
            ddof: aggregate.ddof,
        }),
        CompiledEvaluation::Pair(pair) => Ok(Float64OutputPlan {
            group: pair.group,
            kind: if pair.correlation {
                Float64OutputKind::Correlation
            } else {
                Float64OutputKind::Covariance
            },
            storage: OutputStorage::Float64,
            min_periods: pair.min_periods,
            ddof: pair.ddof,
        }),
        _ => Err("requires aggregate-only outputs".into()),
    }
}

fn output_storage(
    aggregate: &super::CompiledAggregate,
    groups: &[Float64GroupPlan],
) -> std::result::Result<OutputStorage, String> {
    if aggregate.statistic == Statistic::Count {
        return Ok(OutputStorage::Count);
    }
    if matches!(aggregate.statistic, Statistic::Min | Statistic::Max) {
        return match groups.get(aggregate.group) {
            Some(Float64GroupPlan::Extrema { storage, .. }) => Ok(*storage),
            _ => Err("typed rolling extrema output does not reference an extrema group".into()),
        };
    }
    if aggregate.statistic != Statistic::Sum {
        return Ok(OutputStorage::Float64);
    }
    match groups.get(aggregate.group) {
        Some(Float64GroupPlan::Signed { .. }) => Ok(OutputStorage::Int64),
        Some(Float64GroupPlan::Unsigned { .. }) => Ok(OutputStorage::UInt64),
        Some(Float64GroupPlan::Numeric { .. }) => Ok(OutputStorage::Float64),
        _ => Err("typed rolling sum does not reference a numeric group".into()),
    }
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
                .map(|group| Float64WindowState::new(*group, entity_rows))
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
enum Float64WindowState {
    Numeric(Float64NumericState),
    Exact(ExactNumericState),
    Extrema(Float64ExtremaState),
    Pair(Float64PairState),
}

impl Float64WindowState {
    fn new(group: Float64GroupPlan, entity_rows: usize) -> Self {
        match group {
            Float64GroupPlan::Numeric { frame, .. } => {
                Self::Numeric(Float64NumericState::new(frame, entity_rows))
            }
            Float64GroupPlan::Signed { frame, .. } => {
                Self::Exact(ExactNumericState::new(frame, SumClass::Signed, entity_rows))
            }
            Float64GroupPlan::Unsigned { frame, .. } => Self::Exact(ExactNumericState::new(
                frame,
                SumClass::Unsigned,
                entity_rows,
            )),
            Float64GroupPlan::Extrema {
                frame, descending, ..
            } => Self::Extrema(Float64ExtremaState::new(frame, descending, entity_rows)),
            Float64GroupPlan::Pair { frame, .. } => {
                Self::Pair(Float64PairState::new(frame, entity_rows))
            }
        }
    }

    fn update(
        &mut self,
        event_time: i64,
        input: &Float64GroupInput,
        row_index: usize,
        node_id: &str,
    ) -> Result<()> {
        match (self, input) {
            (Self::Numeric(state), Float64GroupInput::Single(input)) => {
                state.update(event_time, valid_float64(input, row_index), node_id)
            }
            (Self::Exact(state), Float64GroupInput::Signed(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExactValue::Signed(input.value(row_index))),
                node_id,
            ),
            (Self::Exact(state), Float64GroupInput::Unsigned(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExactValue::Unsigned(input.value(row_index))),
                node_id,
            ),
            (Self::Extrema(state), Float64GroupInput::Single(input)) => state.update(
                event_time,
                valid_float64(input, row_index).map(ExtremaValue::Float),
                node_id,
            ),
            (Self::Extrema(state), Float64GroupInput::Signed(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExtremaValue::Signed(input.value(row_index))),
                node_id,
            ),
            (Self::Extrema(state), Float64GroupInput::Unsigned(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExtremaValue::Unsigned(input.value(row_index))),
                node_id,
            ),
            (Self::Pair(state), Float64GroupInput::Pair(left, right)) => state.update(
                event_time,
                valid_float64_pair(left, right, row_index),
                node_id,
            ),
            _ => Err(internal_error(
                "typed rolling group state does not match its input plan",
            )),
        }
    }

    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Numeric(state) => state.estimated_bytes(),
            Self::Exact(state) => state.estimated_bytes(),
            Self::Extrema(state) => state.estimated_bytes(),
            Self::Pair(state) => state.estimated_bytes(),
        }
    }
}

fn valid_float64(input: &Float64Array, row_index: usize) -> Option<f64> {
    if input.is_null(row_index) || input.value(row_index).is_nan() {
        None
    } else {
        Some(input.value(row_index))
    }
}

fn valid_float64_pair(
    left: &Float64Array,
    right: &Float64Array,
    row_index: usize,
) -> Option<(f64, f64)> {
    valid_float64(left, row_index).zip(valid_float64(right, row_index))
}

#[derive(Clone, Debug)]
struct Float64NumericState {
    frame: TypedFrame,
    values: VecDeque<TimedFloat64Sample>,
    accumulator: WindowAccumulator,
}

#[derive(Clone, Copy, Debug)]
struct TimedFloat64Sample {
    event_time: i64,
    value: Option<f64>,
}

#[derive(Clone, Debug)]
struct ExactNumericState {
    frame: TypedFrame,
    values: VecDeque<TimedExactSample>,
    accumulator: WindowAccumulator,
}

#[derive(Clone, Copy, Debug)]
struct TimedExactSample {
    event_time: i64,
    value: Option<ExactValue>,
}

#[derive(Clone, Copy, Debug)]
enum ExactValue {
    Signed(i64),
    Unsigned(u64),
}

impl ExactNumericState {
    fn new(frame: TypedFrame, sum_class: SumClass, entity_rows: usize) -> Self {
        let capacity = frame.estimated_samples().min(entity_rows);
        Self {
            frame,
            values: VecDeque::with_capacity(capacity),
            accumulator: WindowAccumulator::new(sum_class),
        }
    }

    fn update(&mut self, event_time: i64, value: Option<ExactValue>, node_id: &str) -> Result<()> {
        if let Some(value) = value {
            add_exact(&mut self.accumulator, value, node_id)?;
        }
        self.values
            .push_back(TimedExactSample { event_time, value });
        self.expire(event_time)
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
        if let Some(Some(value)) = self.values.pop_front().map(|sample| sample.value) {
            remove_exact(&mut self.accumulator, value)?;
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<TimedExactSample>()
    }
}

impl Float64NumericState {
    fn new(frame: TypedFrame, entity_rows: usize) -> Self {
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

#[derive(Clone, Debug)]
struct Float64ExtremaState {
    frame: TypedFrame,
    descending: bool,
    values: VecDeque<TimedExtremaSample>,
    candidates: VecDeque<(u64, ExtremaValue)>,
    valid_count: u64,
    next_ordinal: u64,
}

#[derive(Clone, Copy, Debug)]
struct TimedExtremaSample {
    ordinal: u64,
    event_time: i64,
    value: Option<ExtremaValue>,
}

#[derive(Clone, Copy, Debug)]
enum ExtremaValue {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

impl ExtremaValue {
    fn total_cmp(self, other: Self) -> Result<Ordering> {
        match (self, other) {
            (Self::Signed(left), Self::Signed(right)) => Ok(left.cmp(&right)),
            (Self::Unsigned(left), Self::Unsigned(right)) => Ok(left.cmp(&right)),
            (Self::Float(left), Self::Float(right)) => Ok(left.total_cmp(&right)),
            _ => Err(internal_error(
                "typed rolling extrema candidates have mismatched storage",
            )),
        }
    }
}

impl Float64ExtremaState {
    fn new(frame: TypedFrame, descending: bool, entity_rows: usize) -> Self {
        let capacity = frame.estimated_samples().min(entity_rows);
        Self {
            frame,
            descending,
            values: VecDeque::with_capacity(capacity),
            candidates: VecDeque::with_capacity(capacity),
            valid_count: 0,
            next_ordinal: 0,
        }
    }

    fn update(
        &mut self,
        event_time: i64,
        value: Option<ExtremaValue>,
        node_id: &str,
    ) -> Result<()> {
        let ordinal = self.next_ordinal;
        self.next_ordinal = self
            .next_ordinal
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling extrema ordinal overflowed"))?;
        if let Some(value) = value {
            self.add_candidate(ordinal, value, node_id)?;
        }
        self.values.push_back(TimedExtremaSample {
            ordinal,
            event_time,
            value,
        });
        self.expire(event_time)
    }

    fn add_candidate(&mut self, ordinal: u64, value: ExtremaValue, node_id: &str) -> Result<()> {
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling extrema count overflowed"))?;
        loop {
            let Some((_, back)) = self.candidates.back() else {
                break;
            };
            let ordering = back.total_cmp(value)?;
            let should_remove = if self.descending {
                ordering != Ordering::Greater
            } else {
                ordering != Ordering::Less
            };
            if !should_remove {
                break;
            }
            self.candidates.pop_back();
        }
        self.candidates.push_back((ordinal, value));
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
        let Some(expiring) = self.values.pop_front() else {
            return Ok(());
        };
        if expiring.value.is_some() {
            self.valid_count = self
                .valid_count
                .checked_sub(1)
                .ok_or_else(|| internal_error("rolling extrema removal without an add"))?;
        }
        if self
            .candidates
            .front()
            .is_some_and(|(ordinal, _)| *ordinal == expiring.ordinal)
        {
            self.candidates.pop_front();
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + self.values.capacity() * size_of::<TimedExtremaSample>()
            + self.candidates.capacity() * size_of::<(u64, ExtremaValue)>()
    }
}

#[derive(Clone, Debug)]
struct Float64PairState {
    frame: TypedFrame,
    values: VecDeque<TimedPairSample>,
    accumulator: PairAccumulator,
}

#[derive(Clone, Copy, Debug)]
struct TimedPairSample {
    event_time: i64,
    value: Option<(f64, f64)>,
}

impl Float64PairState {
    fn new(frame: TypedFrame, entity_rows: usize) -> Self {
        let capacity = frame.estimated_samples().min(entity_rows);
        Self {
            frame,
            values: VecDeque::with_capacity(capacity),
            accumulator: PairAccumulator::default(),
        }
    }

    fn update(&mut self, event_time: i64, value: Option<(f64, f64)>, node_id: &str) -> Result<()> {
        if let Some((left, right)) = value {
            add_pair_float64(&mut self.accumulator, left, right, node_id)?;
        }
        self.values.push_back(TimedPairSample { event_time, value });
        self.expire(event_time)?;
        if self.accumulator.is_non_finite() {
            refold_pair_float64(&mut self.accumulator, &self.values, node_id)?;
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
        if let Some(Some((left, right))) = self.values.pop_front().map(|sample| sample.value) {
            remove_pair_float64(&mut self.accumulator, left, right)?;
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>() + self.values.capacity() * size_of::<TimedPairSample>()
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

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen integer mean and variance output type is Float64"
)]
fn add_exact(accumulator: &mut WindowAccumulator, value: ExactValue, node_id: &str) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_add(1)
        .ok_or_else(|| operator_error(node_id, "rolling valid sample count overflowed"))?;
    add_exact_sum(accumulator, value, node_id)?;
    let sample = exact_as_f64(value);
    let count = accumulator.valid_count as f64;
    let delta = sample - accumulator.mean;
    accumulator.mean += delta / count;
    accumulator.m2 += delta * (sample - accumulator.mean);
    Ok(())
}

fn add_exact_sum(
    accumulator: &mut WindowAccumulator,
    value: ExactValue,
    node_id: &str,
) -> Result<()> {
    match (&mut accumulator.sum, value) {
        (Some(SumState::Signed(total)), ExactValue::Signed(value)) => {
            *total = total
                .checked_add(i128::from(value))
                .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
        }
        (Some(SumState::Unsigned(total)), ExactValue::Unsigned(value)) => {
            *total = total
                .checked_add(u128::from(value))
                .ok_or_else(|| operator_error(node_id, "rolling integer sum overflowed"))?;
        }
        _ => {
            return Err(internal_error(
                "typed exact rolling value does not match its sum class",
            ));
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen integer mean and variance output type is Float64"
)]
fn remove_exact(accumulator: &mut WindowAccumulator, value: ExactValue) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_sub(1)
        .ok_or_else(|| internal_error("rolling removal without a matching add"))?;
    remove_exact_sum(accumulator, value)?;
    if accumulator.valid_count == 0 {
        accumulator.mean = 0.0;
        accumulator.m2 = 0.0;
        return Ok(());
    }
    let sample = exact_as_f64(value);
    let count = accumulator.valid_count as f64;
    let delta = sample - accumulator.mean;
    accumulator.mean -= delta / count;
    accumulator.m2 -= delta * (sample - accumulator.mean);
    Ok(())
}

fn remove_exact_sum(accumulator: &mut WindowAccumulator, value: ExactValue) -> Result<()> {
    match (&mut accumulator.sum, value) {
        (Some(SumState::Signed(total)), ExactValue::Signed(value)) => {
            *total = total
                .checked_sub(i128::from(value))
                .ok_or_else(|| internal_error("rolling signed sum removal overflowed"))?;
        }
        (Some(SumState::Unsigned(total)), ExactValue::Unsigned(value)) => {
            *total = total
                .checked_sub(u128::from(value))
                .ok_or_else(|| internal_error("rolling unsigned sum removal overflowed"))?;
        }
        _ => {
            return Err(internal_error(
                "typed exact rolling value does not match its sum class",
            ));
        }
    }
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen integer mean and variance output type is Float64"
)]
fn exact_as_f64(value: ExactValue) -> f64 {
    match value {
        ExactValue::Signed(value) => value as f64,
        ExactValue::Unsigned(value) => value as f64,
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen covariance/correlation output type is Float64"
)]
fn add_pair_float64(
    accumulator: &mut PairAccumulator,
    left: f64,
    right: f64,
    node_id: &str,
) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_add(1)
        .ok_or_else(|| operator_error(node_id, "rolling pair sample count overflowed"))?;
    update_pair_infinity_counts(accumulator, left, right, true);
    let count = accumulator.valid_count as f64;
    let delta_x = left - accumulator.mean_x;
    accumulator.mean_x += delta_x / count;
    let delta_y = right - accumulator.mean_y;
    accumulator.mean_y += delta_y / count;
    accumulator.co_moment += delta_x * (right - accumulator.mean_y);
    accumulator.m2_x += delta_x * (left - accumulator.mean_x);
    accumulator.m2_y += delta_y * (right - accumulator.mean_y);
    Ok(())
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen covariance/correlation output type is Float64"
)]
fn remove_pair_float64(accumulator: &mut PairAccumulator, left: f64, right: f64) -> Result<()> {
    accumulator.valid_count = accumulator
        .valid_count
        .checked_sub(1)
        .ok_or_else(|| internal_error("rolling pair removal without a matching add"))?;
    update_pair_infinity_counts(accumulator, left, right, false);
    if accumulator.valid_count == 0 {
        reset_pair_values(accumulator);
        return Ok(());
    }
    let count = accumulator.valid_count as f64;
    let delta_x = left - accumulator.mean_x;
    accumulator.mean_x -= delta_x / count;
    let delta_y = right - accumulator.mean_y;
    accumulator.mean_y -= delta_y / count;
    accumulator.co_moment -= delta_x * (right - accumulator.mean_y);
    accumulator.m2_x -= delta_x * (left - accumulator.mean_x);
    accumulator.m2_y -= delta_y * (right - accumulator.mean_y);
    Ok(())
}

fn update_pair_infinity_counts(
    accumulator: &mut PairAccumulator,
    left: f64,
    right: f64,
    add: bool,
) {
    update_infinity_count(
        &mut accumulator.pos_inf_x,
        &mut accumulator.neg_inf_x,
        left,
        add,
    );
    update_infinity_count(
        &mut accumulator.pos_inf_y,
        &mut accumulator.neg_inf_y,
        right,
        add,
    );
}

fn update_infinity_count(positive: &mut u64, negative: &mut u64, value: f64, add: bool) {
    if !value.is_infinite() {
        return;
    }
    let count = if value > 0.0 { positive } else { negative };
    *count = if add {
        count.saturating_add(1)
    } else {
        count.saturating_sub(1)
    };
}

fn reset_pair_values(accumulator: &mut PairAccumulator) {
    accumulator.mean_x = 0.0;
    accumulator.mean_y = 0.0;
    accumulator.co_moment = 0.0;
    accumulator.m2_x = 0.0;
    accumulator.m2_y = 0.0;
}

fn refold_pair_float64(
    accumulator: &mut PairAccumulator,
    values: &VecDeque<TimedPairSample>,
    node_id: &str,
) -> Result<()> {
    *accumulator = PairAccumulator::default();
    for (left, right) in values.iter().filter_map(|sample| sample.value) {
        add_pair_float64(accumulator, left, right, node_id)?;
    }
    Ok(())
}

enum DerivedBuilder {
    Count(UInt64Builder),
    Signed(Int64Builder, OutputStorage),
    Unsigned(UInt64Builder, OutputStorage),
    Float(Float64Builder, OutputStorage),
}

impl DerivedBuilder {
    fn new(output: Float64OutputPlan, capacity: usize) -> Self {
        let storage = output.storage;
        if storage == OutputStorage::Count {
            Self::Count(UInt64Builder::with_capacity(capacity))
        } else if storage.is_signed() {
            Self::Signed(Int64Builder::with_capacity(capacity), storage)
        } else if storage.is_unsigned() {
            Self::Unsigned(UInt64Builder::with_capacity(capacity), storage)
        } else {
            Self::Float(Float64Builder::with_capacity(capacity), storage)
        }
    }

    fn append(
        &mut self,
        state: &Float64WindowState,
        output: &Float64OutputPlan,
        node_id: &str,
    ) -> Result<()> {
        if float64_valid_count(state) < output.min_periods {
            self.append_null();
            return Ok(());
        }
        match state {
            Float64WindowState::Numeric(state) => self.append_numeric(state, output),
            Float64WindowState::Exact(state) => self.append_exact(state, output, node_id),
            Float64WindowState::Extrema(state) => self.append_extrema(state, output),
            Float64WindowState::Pair(state) => self.append_pair(state, output),
        }
    }

    fn append_numeric(
        &mut self,
        state: &Float64NumericState,
        output: &Float64OutputPlan,
    ) -> Result<()> {
        match (self, output.kind) {
            (Self::Count(builder), Float64OutputKind::Statistic(Statistic::Count)) => {
                builder.append_value(state.accumulator.valid_count);
                Ok(())
            }
            (Self::Float(builder, _), Float64OutputKind::Statistic(statistic)) => {
                append_float(builder, &state.accumulator, statistic, output.ddof)
            }
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_exact(
        &mut self,
        state: &ExactNumericState,
        output: &Float64OutputPlan,
        node_id: &str,
    ) -> Result<()> {
        match (self, output.kind) {
            (Self::Signed(builder, _), Float64OutputKind::Statistic(Statistic::Sum)) => {
                append_signed_sum(builder, &state.accumulator, node_id)
            }
            (Self::Unsigned(builder, _), Float64OutputKind::Statistic(Statistic::Sum)) => {
                append_unsigned_sum(builder, &state.accumulator, node_id)
            }
            (Self::Count(builder), Float64OutputKind::Statistic(Statistic::Count)) => {
                builder.append_value(state.accumulator.valid_count);
                Ok(())
            }
            (Self::Float(builder, _), Float64OutputKind::Statistic(statistic)) => {
                append_exact_float(builder, &state.accumulator, statistic, output.ddof)
            }
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_extrema(
        &mut self,
        state: &Float64ExtremaState,
        output: &Float64OutputPlan,
    ) -> Result<()> {
        if !matches!(
            output.kind,
            Float64OutputKind::Statistic(Statistic::Min | Statistic::Max)
        ) {
            return Err(typed_output_mismatch());
        }
        match self {
            Self::Float(builder, _) => append_float_extrema(builder, state),
            Self::Signed(builder, _) => append_signed_extrema(builder, state),
            Self::Unsigned(builder, _) => append_unsigned_extrema(builder, state),
            Self::Count(_) => Err(typed_output_mismatch()),
        }
    }

    fn append_pair(&mut self, state: &Float64PairState, output: &Float64OutputPlan) -> Result<()> {
        match (self, output.kind) {
            (
                Self::Float(builder, _),
                Float64OutputKind::Covariance | Float64OutputKind::Correlation,
            ) => append_pair(builder, &state.accumulator, output),
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_null(&mut self) {
        match self {
            Self::Count(builder) | Self::Unsigned(builder, _) => builder.append_null(),
            Self::Signed(builder, _) => builder.append_null(),
            Self::Float(builder, _) => builder.append_null(),
        }
    }

    fn finish(self) -> Result<ArrayRef> {
        match self {
            Self::Count(mut builder) => Ok(std::sync::Arc::new(builder.finish())),
            Self::Signed(mut builder, storage) => {
                finish_with_storage(std::sync::Arc::new(builder.finish()), storage)
            }
            Self::Unsigned(mut builder, storage) => {
                finish_with_storage(std::sync::Arc::new(builder.finish()), storage)
            }
            Self::Float(mut builder, storage) => {
                finish_with_storage(std::sync::Arc::new(builder.finish()), storage)
            }
        }
    }
}

fn typed_output_mismatch() -> crate::CalcFlowError {
    internal_error("typed rolling output builder does not match its statistic")
}

fn finish_with_storage(array: ArrayRef, storage: OutputStorage) -> Result<ArrayRef> {
    let data_type = storage.data_type();
    if array.data_type() == &data_type {
        return Ok(array);
    }
    cast(&array, &data_type).map_err(|error| {
        internal_error(&format!(
            "typed rolling output cast to {data_type} failed: {error}"
        ))
    })
}

fn append_float_extrema(builder: &mut Float64Builder, state: &Float64ExtremaState) -> Result<()> {
    match state.candidates.front().map(|(_, value)| value) {
        Some(ExtremaValue::Float(value)) => builder.append_value(*value),
        None => builder.append_null(),
        _ => {
            return Err(internal_error(
                "typed Float64 extrema has the wrong candidate storage",
            ));
        }
    }
    Ok(())
}

fn append_signed_extrema(builder: &mut Int64Builder, state: &Float64ExtremaState) -> Result<()> {
    match state.candidates.front().map(|(_, value)| value) {
        Some(ExtremaValue::Signed(value)) => builder.append_value(*value),
        None => builder.append_null(),
        _ => {
            return Err(internal_error(
                "typed signed extrema has the wrong candidate storage",
            ));
        }
    }
    Ok(())
}

fn append_unsigned_extrema(builder: &mut UInt64Builder, state: &Float64ExtremaState) -> Result<()> {
    match state.candidates.front().map(|(_, value)| value) {
        Some(ExtremaValue::Unsigned(value)) => builder.append_value(*value),
        None => builder.append_null(),
        _ => {
            return Err(internal_error(
                "typed unsigned extrema has the wrong candidate storage",
            ));
        }
    }
    Ok(())
}

fn float64_valid_count(state: &Float64WindowState) -> u64 {
    match state {
        Float64WindowState::Numeric(state) => state.accumulator.valid_count,
        Float64WindowState::Exact(state) => state.accumulator.valid_count,
        Float64WindowState::Extrema(state) => state.valid_count,
        Float64WindowState::Pair(state) => state.accumulator.valid_count,
    }
}

fn append_signed_sum(
    builder: &mut Int64Builder,
    accumulator: &WindowAccumulator,
    node_id: &str,
) -> Result<()> {
    let Some(SumState::Signed(total)) = accumulator.sum else {
        return Err(internal_error("typed signed sum has the wrong accumulator"));
    };
    builder.append_value(
        i64::try_from(total)
            .map_err(|_| operator_error(node_id, "rolling integer sum overflowed"))?,
    );
    Ok(())
}

fn append_unsigned_sum(
    builder: &mut UInt64Builder,
    accumulator: &WindowAccumulator,
    node_id: &str,
) -> Result<()> {
    let Some(SumState::Unsigned(total)) = accumulator.sum else {
        return Err(internal_error(
            "typed unsigned sum has the wrong accumulator",
        ));
    };
    builder.append_value(
        u64::try_from(total)
            .map_err(|_| operator_error(node_id, "rolling integer sum overflowed"))?,
    );
    Ok(())
}

fn append_exact_float(
    builder: &mut Float64Builder,
    accumulator: &WindowAccumulator,
    statistic: Statistic,
    ddof: u8,
) -> Result<()> {
    match statistic {
        Statistic::Mean => {
            builder.append_value(accumulator.mean);
            Ok(())
        }
        Statistic::Variance | Statistic::Stddev => {
            append_dispersion(builder, accumulator, statistic, ddof);
            Ok(())
        }
        _ => Err(internal_error(
            "typed exact rolling output requires mean, variance, or stddev",
        )),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen pair output type is Float64"
)]
fn append_pair(
    builder: &mut Float64Builder,
    accumulator: &PairAccumulator,
    output: &Float64OutputPlan,
) -> Result<()> {
    let divisor = accumulator.valid_count - u64::from(output.ddof);
    if divisor == 0 {
        builder.append_null();
        return Ok(());
    }
    if accumulator.holds_infinity() {
        builder.append_value(f64::NAN);
        return Ok(());
    }
    match output.kind {
        Float64OutputKind::Covariance => {
            builder.append_value(accumulator.co_moment / divisor as f64);
        }
        Float64OutputKind::Correlation => {
            append_correlation(builder, accumulator);
        }
        Float64OutputKind::Statistic(_) => {
            return Err(internal_error(
                "typed pair state received a scalar statistic",
            ));
        }
    }
    Ok(())
}

fn append_correlation(builder: &mut Float64Builder, accumulator: &PairAccumulator) {
    let m2_x = accumulator.m2_x.max(0.0);
    let m2_y = accumulator.m2_y.max(0.0);
    if m2_x == 0.0 || m2_y == 0.0 {
        builder.append_null();
    } else {
        builder.append_value(accumulator.co_moment / (m2_x.sqrt() * m2_y.sqrt()));
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
