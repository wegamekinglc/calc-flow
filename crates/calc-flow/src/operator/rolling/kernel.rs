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
    CompiledAggregate, CompiledDifference, CompiledEvaluation, CompiledEwma, CompiledFloatReadout,
    CompiledFrame, CompiledKeyColumn, CompiledPairAggregate, CompiledRollingOutput,
    CompiledWindowGroup, PairAccumulator, RollingNumericalProfile, Statistic, SumClass, SumState,
    WindowAccumulator,
    generated_kernel_manifest::{
        GeneratedComplexity, GeneratedTransition, generated_kernel_capability,
    },
    internal_error, operator_error,
};
use crate::Result;

const ROLLING_KERNEL_PLAN_VERSION: u32 = 1;

/// Selected executable family for one immutable rolling kernel plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum KernelSelection {
    /// Columnar primitive numeric aggregates and EWMA recurrence.
    OrderedPrimitive,
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
    numerical_profile: RollingNumericalProfile,
    selection: KernelSelection,
    complexity: KernelComplexity,
    event_time_index: usize,
    order_columns: Vec<usize>,
    partition_columns: Vec<usize>,
    sequence_columns: Vec<usize>,
    nan_as_value: bool,
    groups: Vec<TypedGroupPlan>,
    outputs: Vec<TypedOutputPlan>,
    fallback_reason: Option<String>,
    estimated_state_bytes_per_entity: usize,
    fingerprint: String,
}

#[derive(Clone, Copy, Debug)]
enum TypedGroupPlan {
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
    Ewma {
        input_index: usize,
        alpha: f64,
    },
}

impl TypedGroupPlan {
    const fn frame(self) -> Option<TypedFrame> {
        match self {
            Self::Numeric { frame, .. }
            | Self::Signed { frame, .. }
            | Self::Unsigned { frame, .. }
            | Self::Extrema { frame, .. }
            | Self::Pair { frame, .. } => Some(frame),
            Self::Ewma { .. } => None,
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
struct TypedOutputPlan {
    group: usize,
    kind: TypedOutputKind,
    storage: OutputStorage,
    min_periods: u64,
    ddof: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TypedOutputKind {
    Statistic(Statistic),
    Covariance,
    Correlation,
    Ewma,
    Difference {
        left: TypedFloatReadout,
        right: TypedFloatReadout,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TypedFloatReadout {
    group: usize,
    kind: TypedFloatReadoutKind,
    min_periods: u64,
    ddof: u8,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TypedFloatReadoutKind {
    Mean,
    Variance,
    Stddev,
    Ewma,
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

enum TypedGroupInput {
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

struct OrderProof {
    last_identity: Option<Vec<u8>>,
    elapsed_ns: u64,
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
    states: Vec<TypedEntityState>,
    last_identity: Option<Vec<u8>>,
}

impl RollingKernelPlan {
    #[allow(
        clippy::too_many_arguments,
        reason = "the compiler boundary keeps every rolling semantic explicit"
    )]
    pub(super) fn compile(
        input_schema: &Schema,
        state_layout_version: u32,
        numerical_profile: RollingNumericalProfile,
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
        Self::compile_with_order(
            input_schema,
            state_layout_version,
            numerical_profile,
            event_time_index,
            order_columns,
            partition_columns,
            sequence_columns,
            outputs,
            window_groups,
        )
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the compiler boundary keeps every ordering semantic explicit"
    )]
    pub(super) fn compile_with_order(
        input_schema: &Schema,
        state_layout_version: u32,
        numerical_profile: RollingNumericalProfile,
        event_time_index: usize,
        order_columns: Vec<usize>,
        partition_columns: Vec<usize>,
        sequence_columns: Vec<usize>,
        outputs: &[CompiledRollingOutput],
        window_groups: &[CompiledWindowGroup],
    ) -> Self {
        let compiled = compile_typed_plan(input_schema, outputs, window_groups);
        let allow_order_peers = order_columns.first() != Some(&event_time_index);
        let (selection, complexity, groups, typed_outputs, fallback_reason) = match compiled {
            Ok((groups, typed_outputs)) => (
                KernelSelection::OrderedPrimitive,
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
                size_of::<TypedWindowState>().saturating_add(
                    group
                        .frame()
                        .map_or(0, TypedFrame::estimated_samples)
                        .saturating_mul(size_of::<TimedSample<f64>>()),
                ),
            )
        });
        let fingerprint = kernel_fingerprint(&KernelFingerprintInput {
            input_schema,
            state_layout_version,
            numerical_profile,
            selection,
            complexity,
            event_time_index,
            order_columns: &order_columns,
            partition_columns: &partition_columns,
            allow_order_peers,
            groups: &groups,
            outputs: &typed_outputs,
            fallback_reason: fallback_reason.as_deref(),
        });
        Self {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version,
            numerical_profile,
            selection,
            complexity,
            event_time_index,
            order_columns,
            partition_columns,
            sequence_columns,
            nan_as_value: false,
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

    pub(super) fn with_nan_as_value(mut self) -> Self {
        self.nan_as_value = true;
        self.fingerprint = hex::encode(Sha256::digest(
            format!("{}|nan=value", self.fingerprint).as_bytes(),
        ));
        self
    }

    pub(super) const fn numerical_profile(&self) -> &'static str {
        self.numerical_profile.name()
    }

    pub(super) const fn numerical_profile_kind(&self) -> RollingNumericalProfile {
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
        matches!(self.selection, KernelSelection::OrderedPrimitive)
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

    /// Installs durable per-entity transition counts and EWMA recurrence values
    /// into an otherwise reconstructed typed state. This is used only after
    /// checkpoint restore; ordinary live transitions keep both values directly.
    pub(super) fn seed_restored_state(
        &self,
        state: &RollingKernelState,
        entities: &RecordBatch,
        transition_counts: &[u64],
        seeds: &[Vec<Option<(u64, f64)>>],
        node_id: &str,
    ) -> Result<RollingKernelState> {
        self.validate_state(state, node_id)?;
        if entities.num_rows() != seeds.len() || seeds.len() != transition_counts.len() {
            return Err(internal_error(
                "typed rolling restore entity metadata counts differ",
            ));
        }
        let entity_rows = encode_rows(entities, &self.partition_columns, node_id)?;
        let entity_keys = encoded_keys(&entity_rows, entities.num_rows());
        let mut seeded = state.clone();
        seeded
            .kernel_fingerprint
            .get_or_insert_with(|| self.fingerprint.clone());
        let entity_ids = seeded.resolve_entities(&entity_keys, &self.groups);
        for (row_index, values) in seeds.iter().enumerate() {
            let entity = &mut seeded.states[entity_ids[row_index]];
            self.seed_restored_entity(entity, transition_counts[row_index], values, node_id)?;
        }
        Ok(seeded)
    }

    fn seed_restored_entity(
        &self,
        entity: &mut TypedEntityState,
        transition_count: u64,
        values: &[Option<(u64, f64)>],
        node_id: &str,
    ) -> Result<()> {
        if self.numerical_profile == RollingNumericalProfile::StableV2Preview {
            entity.force_stable_v2_rebase(node_id)?;
        }
        entity.transition_count = transition_count;
        self.seed_ewma_entity(entity, values)
    }

    fn seed_ewma_entity(
        &self,
        entity: &mut TypedEntityState,
        seeds: &[Option<(u64, f64)>],
    ) -> Result<()> {
        if seeds.len() != self.groups.len() {
            return Err(internal_error(
                "typed EWMA restore seed width differs from the kernel plan",
            ));
        }
        for ((group, state), seed) in self.groups.iter().zip(&mut entity.groups).zip(seeds) {
            let TypedGroupPlan::Ewma { alpha, .. } = group else {
                if seed.is_some() {
                    return Err(internal_error(
                        "typed EWMA restore populated a non-EWMA group",
                    ));
                }
                continue;
            };
            let TypedWindowState::Ewma(state) = state else {
                return Err(internal_error("typed EWMA restore state mismatch"));
            };
            *state = TypedEwmaState::new(*alpha);
            if let Some((valid_count, value)) = seed {
                if *valid_count == 0 {
                    return Err(internal_error(
                        "typed EWMA restore populated a zero sample count",
                    ));
                }
                state.valid_count = *valid_count;
                state.value = *value;
            }
        }
        Ok(())
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
        self.update_and_fill_with_prior_order(state, input, true, node_id)
    }

    /// Applies one finalized stream micro-batch.
    ///
    /// Watermark delivery can split equal event-time peers across envelopes,
    /// so only the current finalized batch has a global canonical-order proof.
    /// Per-entity transition order remains guaranteed by the rolling buffer.
    pub(super) fn update_stream_and_fill(
        &self,
        state: &RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<Option<RollingKernelExecution>> {
        self.update_and_fill_with_prior_order(state, input, false, node_id)
    }

    fn update_and_fill_with_prior_order(
        &self,
        state: &RollingKernelState,
        input: &RecordBatch,
        compare_prior_order: bool,
        node_id: &str,
    ) -> Result<Option<RollingKernelExecution>> {
        if self.selection != KernelSelection::OrderedPrimitive {
            return Ok(None);
        }
        self.validate_state(state, node_id)?;

        let input_validation_ns = self.validate_input_timed(input, node_id)?;
        let Some(order_proof) =
            self.prove_input_order(input, compare_prior_order.then_some(state), node_id)?
        else {
            return Ok(None);
        };
        let (next_state, entity_ids, entity_encode_ns) =
            self.resolve_batch_entities(state, input, node_id)?;

        self.fill_typed(
            input,
            &entity_ids,
            next_state,
            order_proof.last_identity,
            node_id,
            RollingKernelMetrics {
                input_validation_ns,
                order_proof_ns: order_proof.elapsed_ns,
                entity_encode_ns,
                order_proof_rows: input.num_rows(),
                input_rows: input.num_rows(),
                output_rows: input.num_rows(),
                ..RollingKernelMetrics::default()
            },
        )
        .map(Some)
    }

    fn validate_input_timed(&self, input: &RecordBatch, node_id: &str) -> Result<u64> {
        let started = Instant::now();
        self.validate_required_values(input, node_id)?;
        Ok(nanos(started.elapsed()))
    }

    fn prove_input_order(
        &self,
        input: &RecordBatch,
        prior_state: Option<&RollingKernelState>,
        node_id: &str,
    ) -> Result<Option<OrderProof>> {
        let started = Instant::now();
        let rows = encode_rows(input, &self.order_columns, node_id)?;
        if !self.canonical_order_is_proven(input, &rows, prior_state, node_id)? {
            return Ok(None);
        }
        let last_identity = input
            .num_rows()
            .checked_sub(1)
            .map(|row_index| rows.row(row_index).data().to_vec());
        Ok(Some(OrderProof {
            last_identity,
            elapsed_ns: nanos(started.elapsed()),
        }))
    }

    fn resolve_batch_entities(
        &self,
        state: &RollingKernelState,
        input: &RecordBatch,
        node_id: &str,
    ) -> Result<(RollingKernelState, Vec<usize>, u64)> {
        let started = Instant::now();
        let rows = encode_rows(input, &self.partition_columns, node_id)?;
        let keys = encoded_keys(&rows, input.num_rows());
        let mut next_state = state.clone();
        next_state
            .kernel_fingerprint
            .get_or_insert_with(|| self.fingerprint.clone());
        let entity_ids = next_state.resolve_entities(&keys, &self.groups);
        Ok((next_state, entity_ids, nanos(started.elapsed())))
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

    // Within-batch ordering is always proven. Callers that receive a globally
    // ordered sequence can additionally include the prior batch boundary.
    // #lizard forgives
    fn canonical_order_is_proven(
        &self,
        input: &RecordBatch,
        order_rows: &datafusion::arrow::row::Rows,
        prior_state: Option<&RollingKernelState>,
        node_id: &str,
    ) -> Result<bool> {
        if input.num_rows() > 0
            && let Some(previous) = prior_state.and_then(|state| state.last_identity.as_deref())
        {
            let current = order_rows.row(0).data();
            if previous == current && !self.allows_order_peers() {
                return Err(duplicate_identity_error(
                    input,
                    self.event_time_index,
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
            if previous == current && !self.allows_order_peers() {
                return Err(duplicate_identity_error(
                    input,
                    self.event_time_index,
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

    fn allows_order_peers(&self) -> bool {
        self.order_columns.first() != Some(&self.event_time_index)
    }

    fn fill_typed(
        &self,
        input: &RecordBatch,
        entity_ids: &[usize],
        mut state: RollingKernelState,
        last_identity: Option<Vec<u8>>,
        node_id: &str,
        mut metrics: RollingKernelMetrics,
    ) -> Result<RollingKernelExecution> {
        let inputs = self.typed_inputs(input, node_id)?;
        let event_times = input
            .column(self.event_time_index)
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
        fill_typed_rows(
            self,
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
            .map(TypedEntityState::estimated_bytes)
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
        let event_time_index = self.event_time_index;
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

    fn typed_inputs(&self, input: &RecordBatch, node_id: &str) -> Result<Vec<TypedGroupInput>> {
        self.groups
            .iter()
            .map(|group| typed_group_input(group, input, node_id))
            .collect()
    }
}

fn typed_group_input(
    group: &TypedGroupPlan,
    input: &RecordBatch,
    node_id: &str,
) -> Result<TypedGroupInput> {
    match *group {
        TypedGroupPlan::Numeric { input_index, .. } => Ok(TypedGroupInput::Single(cast_float64(
            input.column(input_index),
            node_id,
        )?)),
        TypedGroupPlan::Extrema {
            input_index,
            storage,
            ..
        } => extrema_group_input(input.column(input_index), storage, node_id),
        TypedGroupPlan::Signed { input_index, .. } => {
            cast_signed(input.column(input_index), node_id).map(TypedGroupInput::Signed)
        }
        TypedGroupPlan::Unsigned { input_index, .. } => {
            cast_unsigned(input.column(input_index), node_id).map(TypedGroupInput::Unsigned)
        }
        TypedGroupPlan::Pair {
            left_index,
            right_index,
            ..
        } => Ok(TypedGroupInput::Pair(
            cast_float64(input.column(left_index), node_id)?,
            cast_float64(input.column(right_index), node_id)?,
        )),
        TypedGroupPlan::Ewma { input_index, .. } => Ok(TypedGroupInput::Single(cast_float64(
            input.column(input_index),
            node_id,
        )?)),
    }
}

fn extrema_group_input(
    input: &ArrayRef,
    storage: OutputStorage,
    node_id: &str,
) -> Result<TypedGroupInput> {
    if storage.is_signed() {
        cast_signed(input, node_id).map(TypedGroupInput::Signed)
    } else if storage.is_unsigned() {
        cast_unsigned(input, node_id).map(TypedGroupInput::Unsigned)
    } else if storage.is_float() {
        cast_float64(input, node_id).map(TypedGroupInput::Single)
    } else {
        Err(internal_error(
            "typed extrema group has count output storage",
        ))
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
    fn resolve_entities(&mut self, keys: &[Vec<u8>], groups: &[TypedGroupPlan]) -> Vec<usize> {
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
                self.states.push(TypedEntityState::new(groups, row_count));
                entity_id
            })
            .collect()
    }
}

fn fill_typed_rows(
    plan: &RollingKernelPlan,
    inputs: &[TypedGroupInput],
    event_times: &TimestampMicrosecondArray,
    states: &mut [TypedEntityState],
    builders: &mut [DerivedBuilder],
    entity_ids: &[usize],
    node_id: &str,
) -> Result<()> {
    for (row_index, &entity_id) in entity_ids.iter().enumerate() {
        let entity = &mut states[entity_id];
        entity.transition_count = entity
            .transition_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling entity transition count overflowed"))?;
        update_typed_groups(
            inputs,
            event_times.value(row_index),
            entity,
            row_index,
            plan.numerical_profile,
            plan.nan_as_value,
            node_id,
        )?;
        append_typed_outputs(&plan.outputs, builders, entity, node_id)?;
    }
    Ok(())
}

fn update_typed_groups(
    inputs: &[TypedGroupInput],
    event_time: i64,
    entity: &mut TypedEntityState,
    row_index: usize,
    numerical_profile: RollingNumericalProfile,
    nan_as_value: bool,
    node_id: &str,
) -> Result<()> {
    for (state, input) in entity.groups.iter_mut().zip(inputs) {
        state.update(
            event_time,
            input,
            row_index,
            numerical_profile,
            nan_as_value,
            entity.transition_count,
            node_id,
        )?;
    }
    Ok(())
}

fn append_typed_outputs(
    outputs: &[TypedOutputPlan],
    builders: &mut [DerivedBuilder],
    entity: &TypedEntityState,
    node_id: &str,
) -> Result<()> {
    for (builder, output) in builders.iter_mut().zip(outputs) {
        builder.append(&entity.groups, output, node_id)?;
    }
    Ok(())
}

fn compile_typed_plan(
    input_schema: &Schema,
    outputs: &[CompiledRollingOutput],
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<(Vec<TypedGroupPlan>, Vec<TypedOutputPlan>), String> {
    let groups = compile_typed_groups(input_schema, window_groups)?;
    let typed_outputs = compile_typed_outputs(outputs, &groups)?;
    if groups.is_empty() || typed_outputs.is_empty() {
        return Err("requires at least one typed rolling output".into());
    }
    Ok((groups, typed_outputs))
}

fn compile_typed_groups(
    input_schema: &Schema,
    window_groups: &[CompiledWindowGroup],
) -> std::result::Result<Vec<TypedGroupPlan>, String> {
    let mut groups = Vec::with_capacity(window_groups.len());
    for group in window_groups {
        groups.push(compile_typed_group(input_schema, group)?);
    }
    Ok(groups)
}

fn compile_typed_group(
    input_schema: &Schema,
    group: &CompiledWindowGroup,
) -> std::result::Result<TypedGroupPlan, String> {
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
        } => compile_extrema_group(input_schema, *input_index, *frame, *descending),
        CompiledWindowGroup::Pair {
            left_index,
            right_index,
            frame,
        } => compile_pair_group(input_schema, *left_index, *right_index, *frame),
        CompiledWindowGroup::Ewma {
            input_index, alpha, ..
        } => {
            require_numeric_type(input_schema, *input_index)?;
            Ok(TypedGroupPlan::Ewma {
                input_index: *input_index,
                alpha: *alpha,
            })
        }
    }
}

fn compile_numeric_group(
    input_schema: &Schema,
    input_index: usize,
    frame: CompiledFrame,
    sum_class: SumClass,
) -> std::result::Result<TypedGroupPlan, String> {
    let frame = TypedFrame::try_from(frame)?;
    match sum_class {
        SumClass::Float
            if matches!(
                input_schema.field(input_index).data_type(),
                DataType::Float32 | DataType::Float64
            ) =>
        {
            Ok(TypedGroupPlan::Numeric { input_index, frame })
        }
        SumClass::Signed => Ok(TypedGroupPlan::Signed { input_index, frame }),
        SumClass::Unsigned => Ok(TypedGroupPlan::Unsigned { input_index, frame }),
        _ => Err("requires primitive numeric window groups".into()),
    }
}

fn compile_extrema_group(
    input_schema: &Schema,
    input_index: usize,
    frame: CompiledFrame,
    descending: bool,
) -> std::result::Result<TypedGroupPlan, String> {
    let frame = TypedFrame::try_from(frame)?;
    Ok(TypedGroupPlan::Extrema {
        input_index,
        frame,
        descending,
        storage: extrema_storage(input_schema.field(input_index).data_type())?,
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
) -> std::result::Result<TypedGroupPlan, String> {
    require_numeric_type(input_schema, left_index)?;
    require_numeric_type(input_schema, right_index)?;
    Ok(TypedGroupPlan::Pair {
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

fn compile_typed_outputs(
    outputs: &[CompiledRollingOutput],
    groups: &[TypedGroupPlan],
) -> std::result::Result<Vec<TypedOutputPlan>, String> {
    let mut typed_outputs = Vec::with_capacity(outputs.len());
    for output in outputs {
        let plan = compile_typed_output(output, groups)?;
        if !typed_output_groups_are_valid(&plan, groups.len()) {
            return Err("aggregate group index is outside the typed plan".into());
        }
        typed_outputs.push(plan);
    }
    Ok(typed_outputs)
}

fn compile_typed_output(
    output: &CompiledRollingOutput,
    groups: &[TypedGroupPlan],
) -> std::result::Result<TypedOutputPlan, String> {
    match &output.evaluation {
        CompiledEvaluation::Aggregate(aggregate) => compile_aggregate_output(aggregate, groups),
        CompiledEvaluation::Pair(pair) => compile_pair_output(pair),
        CompiledEvaluation::Ewma(ewma) => compile_ewma_output(ewma),
        CompiledEvaluation::Difference(difference) => compile_difference_output(difference),
        _ => Err("requires aggregate, pair, EWMA, or fused difference outputs".into()),
    }
}

fn compile_aggregate_output(
    aggregate: &CompiledAggregate,
    groups: &[TypedGroupPlan],
) -> std::result::Result<TypedOutputPlan, String> {
    let transition = if matches!(aggregate.statistic, Statistic::Min | Statistic::Max) {
        GeneratedTransition::Extrema
    } else {
        GeneratedTransition::Numeric
    };
    require_generated_transition(aggregate.statistic.name(), transition)?;
    Ok(TypedOutputPlan {
        group: aggregate.group,
        kind: TypedOutputKind::Statistic(aggregate.statistic),
        storage: output_storage(aggregate, groups)?,
        min_periods: aggregate.min_periods,
        ddof: aggregate.ddof,
    })
}

fn compile_pair_output(
    pair: &CompiledPairAggregate,
) -> std::result::Result<TypedOutputPlan, String> {
    let primitive = if pair.correlation {
        "correlation"
    } else {
        "covariance"
    };
    require_generated_transition(primitive, GeneratedTransition::Pair)?;
    Ok(TypedOutputPlan {
        group: pair.group,
        kind: if pair.correlation {
            TypedOutputKind::Correlation
        } else {
            TypedOutputKind::Covariance
        },
        storage: OutputStorage::Float64,
        min_periods: pair.min_periods,
        ddof: pair.ddof,
    })
}

fn compile_ewma_output(ewma: &CompiledEwma) -> std::result::Result<TypedOutputPlan, String> {
    require_generated_transition("ewma", GeneratedTransition::Ewma)?;
    Ok(TypedOutputPlan {
        group: ewma.group,
        kind: TypedOutputKind::Ewma,
        storage: OutputStorage::Float64,
        min_periods: ewma.min_periods,
        ddof: 0,
    })
}

fn compile_difference_output(
    difference: &CompiledDifference,
) -> std::result::Result<TypedOutputPlan, String> {
    require_generated_transition("difference", GeneratedTransition::FusedDifference)?;
    Ok(TypedOutputPlan {
        group: 0,
        kind: TypedOutputKind::Difference {
            left: compile_typed_readout(difference.left)?,
            right: compile_typed_readout(difference.right)?,
        },
        storage: OutputStorage::Float64,
        min_periods: 0,
        ddof: 0,
    })
}

fn require_generated_transition(
    primitive: &str,
    expected: GeneratedTransition,
) -> std::result::Result<(), String> {
    let capability = generated_kernel_capability(primitive)
        .ok_or_else(|| format!("kernel census is missing primitive {primitive}"))?;
    if !capability.batch || !capability.stream {
        return Err(format!(
            "kernel census does not declare batch and stream support for {primitive}"
        ));
    }
    if capability.typed_transition != Some(expected)
        || capability.complexity != GeneratedComplexity::AmortizedConstant
    {
        return Err(format!(
            "kernel census does not declare the required typed transition for {primitive}"
        ));
    }
    Ok(())
}

pub(super) fn supports_datafusion_primitive(primitive: &str) -> bool {
    generated_kernel_capability(primitive).is_some_and(|capability| {
        capability.batch
            && capability.stream
            && capability.datafusion
            && capability.typed_transition.is_some()
            && capability.complexity == GeneratedComplexity::AmortizedConstant
    })
}

fn compile_typed_readout(
    readout: CompiledFloatReadout,
) -> std::result::Result<TypedFloatReadout, String> {
    match readout {
        CompiledFloatReadout::Ewma(readout) => Ok(TypedFloatReadout {
            group: readout.group,
            kind: TypedFloatReadoutKind::Ewma,
            min_periods: readout.min_periods,
            ddof: 0,
        }),
        CompiledFloatReadout::Aggregate(readout) => {
            let kind = match readout.statistic {
                Statistic::Mean => TypedFloatReadoutKind::Mean,
                Statistic::Variance => TypedFloatReadoutKind::Variance,
                Statistic::Stddev => TypedFloatReadoutKind::Stddev,
                _ => return Err("fused difference requires floating readouts".into()),
            };
            Ok(TypedFloatReadout {
                group: readout.group,
                kind,
                min_periods: readout.min_periods,
                ddof: readout.ddof,
            })
        }
    }
}

const fn typed_output_groups_are_valid(output: &TypedOutputPlan, group_count: usize) -> bool {
    match output.kind {
        TypedOutputKind::Difference { left, right } => {
            left.group < group_count && right.group < group_count
        }
        _ => output.group < group_count,
    }
}

fn output_storage(
    aggregate: &CompiledAggregate,
    groups: &[TypedGroupPlan],
) -> std::result::Result<OutputStorage, String> {
    if aggregate.statistic == Statistic::Count {
        return Ok(OutputStorage::Count);
    }
    if matches!(aggregate.statistic, Statistic::Min | Statistic::Max) {
        return match groups.get(aggregate.group) {
            Some(TypedGroupPlan::Extrema { storage, .. }) => Ok(*storage),
            _ => Err("typed rolling extrema output does not reference an extrema group".into()),
        };
    }
    if aggregate.statistic != Statistic::Sum {
        return Ok(OutputStorage::Float64);
    }
    match groups.get(aggregate.group) {
        Some(TypedGroupPlan::Signed { .. }) => Ok(OutputStorage::Int64),
        Some(TypedGroupPlan::Unsigned { .. }) => Ok(OutputStorage::UInt64),
        Some(TypedGroupPlan::Numeric { .. }) => Ok(OutputStorage::Float64),
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
    numerical_profile: RollingNumericalProfile,
    selection: KernelSelection,
    complexity: KernelComplexity,
    event_time_index: usize,
    order_columns: &'a [usize],
    partition_columns: &'a [usize],
    allow_order_peers: bool,
    groups: &'a [TypedGroupPlan],
    outputs: &'a [TypedOutputPlan],
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
        "version={ROLLING_KERNEL_PLAN_VERSION}|state={}|numeric={}|selection={:?}|complexity={:?}|schema={schema:?}|event_time={}|order={:?}|partition={:?}|duplicates={:?}|groups={:?}|outputs={:?}|fallback={:?}",
        input.state_layout_version,
        input.numerical_profile.name(),
        input.selection,
        input.complexity,
        input.event_time_index,
        input.order_columns,
        input.partition_columns,
        input.allow_order_peers,
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
struct TypedEntityState {
    groups: Vec<TypedWindowState>,
    transition_count: u64,
}

impl TypedEntityState {
    fn new(groups: &[TypedGroupPlan], entity_rows: usize) -> Self {
        Self {
            groups: groups
                .iter()
                .map(|group| TypedWindowState::new(*group, entity_rows))
                .collect(),
            transition_count: 0,
        }
    }

    fn force_stable_v2_rebase(&mut self, node_id: &str) -> Result<()> {
        for group in &mut self.groups {
            group.force_stable_v2_rebase(node_id)?;
        }
        Ok(())
    }

    fn estimated_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .groups
                .iter()
                .map(TypedWindowState::estimated_bytes)
                .sum::<usize>()
    }
}

#[derive(Clone, Debug)]
enum TypedWindowState {
    Numeric(Float64NumericState),
    Exact(ExactNumericState),
    Extrema(TypedExtremaState),
    Pair(Float64PairState),
    Ewma(TypedEwmaState),
}

impl TypedWindowState {
    fn new(group: TypedGroupPlan, entity_rows: usize) -> Self {
        match group {
            TypedGroupPlan::Numeric { frame, .. } => {
                Self::Numeric(Float64NumericState::new(frame, entity_rows))
            }
            TypedGroupPlan::Signed { frame, .. } => {
                Self::Exact(ExactNumericState::new(frame, SumClass::Signed, entity_rows))
            }
            TypedGroupPlan::Unsigned { frame, .. } => Self::Exact(ExactNumericState::new(
                frame,
                SumClass::Unsigned,
                entity_rows,
            )),
            TypedGroupPlan::Extrema {
                frame, descending, ..
            } => Self::Extrema(TypedExtremaState::new(frame, descending, entity_rows)),
            TypedGroupPlan::Pair { frame, .. } => {
                Self::Pair(Float64PairState::new(frame, entity_rows))
            }
            TypedGroupPlan::Ewma { alpha, .. } => Self::Ewma(TypedEwmaState::new(alpha)),
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the typed transition keeps row, numerical, and DataFusion NaN policy explicit"
    )]
    fn update(
        &mut self,
        event_time: i64,
        input: &TypedGroupInput,
        row_index: usize,
        numerical_profile: RollingNumericalProfile,
        nan_as_value: bool,
        transition_count: u64,
        node_id: &str,
    ) -> Result<()> {
        match (self, input) {
            (Self::Numeric(state), TypedGroupInput::Single(input)) => state.update(
                event_time,
                valid_float64(input, row_index, nan_as_value),
                numerical_profile,
                transition_count,
                node_id,
            ),
            (Self::Exact(state), TypedGroupInput::Signed(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExactValue::Signed(input.value(row_index))),
                node_id,
            ),
            (Self::Exact(state), TypedGroupInput::Unsigned(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExactValue::Unsigned(input.value(row_index))),
                node_id,
            ),
            (Self::Extrema(state), TypedGroupInput::Single(input)) => state.update(
                event_time,
                valid_float64(input, row_index, false).map(ExtremaValue::Float),
                node_id,
            ),
            (Self::Extrema(state), TypedGroupInput::Signed(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExtremaValue::Signed(input.value(row_index))),
                node_id,
            ),
            (Self::Extrema(state), TypedGroupInput::Unsigned(input)) => state.update(
                event_time,
                input
                    .is_valid(row_index)
                    .then(|| ExtremaValue::Unsigned(input.value(row_index))),
                node_id,
            ),
            (Self::Pair(state), TypedGroupInput::Pair(left, right)) => state.update(
                event_time,
                valid_float64_pair(left, right, row_index),
                numerical_profile,
                transition_count,
                node_id,
            ),
            (Self::Ewma(state), TypedGroupInput::Single(input)) => {
                state.update(valid_float64(input, row_index, false), node_id)
            }
            _ => Err(internal_error(
                "typed rolling group state does not match its input plan",
            )),
        }
    }

    fn force_stable_v2_rebase(&mut self, node_id: &str) -> Result<()> {
        match self {
            Self::Numeric(state) => state.rebase_stable_v2(node_id),
            Self::Pair(state) => state.rebase_stable_v2(node_id),
            Self::Exact(_) | Self::Extrema(_) | Self::Ewma(_) => Ok(()),
        }
    }

    fn estimated_bytes(&self) -> usize {
        match self {
            Self::Numeric(state) => state.estimated_bytes(),
            Self::Exact(state) => state.estimated_bytes(),
            Self::Extrema(state) => state.estimated_bytes(),
            Self::Pair(state) => state.estimated_bytes(),
            Self::Ewma(_) => TypedEwmaState::estimated_bytes(),
        }
    }
}

fn valid_float64(input: &Float64Array, row_index: usize, nan_as_value: bool) -> Option<f64> {
    if input.is_null(row_index) || (!nan_as_value && input.value(row_index).is_nan()) {
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
    valid_float64(left, row_index, false).zip(valid_float64(right, row_index, false))
}

/// One timed sample in a sliding window queue.
#[derive(Clone, Copy, Debug)]
struct TimedSample<V> {
    event_time: i64,
    value: Option<V>,
}

/// The frame-bounded sample queue every typed numeric state shares.
///
/// Each state keeps its own accumulator; the queue owns only the push,
/// expire, and front-removal mechanics so the Rows/Duration policy is
/// written once. `expire` takes the per-state removal callback so the
/// accumulator stays outside the generic.
#[derive(Clone, Debug)]
struct SlidingSampleQueue<V> {
    frame: TypedFrame,
    values: VecDeque<TimedSample<V>>,
}

impl<V: Copy> SlidingSampleQueue<V> {
    fn new(frame: TypedFrame, entity_rows: usize) -> Self {
        let capacity = frame.estimated_samples().min(entity_rows);
        Self {
            frame,
            values: VecDeque::with_capacity(capacity),
        }
    }

    fn push(&mut self, event_time: i64, value: Option<V>) {
        self.values.push_back(TimedSample { event_time, value });
    }

    fn len(&self) -> usize {
        self.values.len()
    }

    fn expire(
        &mut self,
        event_time: i64,
        mut on_remove: impl FnMut(V) -> Result<()>,
    ) -> Result<()> {
        match self.frame {
            TypedFrame::Rows(window) => {
                if self.values.len() > window {
                    self.remove_front(&mut on_remove)?;
                }
            }
            TypedFrame::Duration(micros) => {
                let bound = i128::from(event_time) - i128::from(micros);
                while self
                    .values
                    .front()
                    .is_some_and(|sample| i128::from(sample.event_time) <= bound)
                {
                    self.remove_front(&mut on_remove)?;
                }
            }
        }
        Ok(())
    }

    fn remove_front(&mut self, mut on_remove: impl FnMut(V) -> Result<()>) -> Result<()> {
        if let Some(Some(value)) = self.values.pop_front().map(|sample| sample.value) {
            on_remove(value)?;
        }
        Ok(())
    }

    /// Byte estimate with the caller's per-sample size so each state keeps
    /// its historical accounting basis (the float state sizes by the bare
    /// `Option<f64>`, not the full timed sample).
    fn estimated_bytes(&self, state_bytes: usize, sample_bytes: usize) -> usize {
        state_bytes + self.values.capacity() * sample_bytes
    }
}

#[derive(Clone, Debug)]
struct Float64NumericState {
    samples: SlidingSampleQueue<f64>,
    accumulator: WindowAccumulator,
}

#[derive(Clone, Debug)]
struct ExactNumericState {
    samples: SlidingSampleQueue<ExactValue>,
    accumulator: WindowAccumulator,
}

#[derive(Clone, Copy, Debug)]
enum ExactValue {
    Signed(i64),
    Unsigned(u64),
}

impl ExactNumericState {
    fn new(frame: TypedFrame, sum_class: SumClass, entity_rows: usize) -> Self {
        Self {
            samples: SlidingSampleQueue::new(frame, entity_rows),
            accumulator: WindowAccumulator::new(sum_class),
        }
    }

    fn update(&mut self, event_time: i64, value: Option<ExactValue>, node_id: &str) -> Result<()> {
        if let Some(value) = value {
            add_exact(&mut self.accumulator, value, node_id)?;
        }
        self.samples.push(event_time, value);
        self.expire(event_time)
    }

    fn expire(&mut self, event_time: i64) -> Result<()> {
        let accumulator = &mut self.accumulator;
        self.samples
            .expire(event_time, |value| remove_exact(accumulator, value))
    }

    fn estimated_bytes(&self) -> usize {
        self.samples
            .estimated_bytes(size_of::<Self>(), size_of::<TimedSample<ExactValue>>())
    }
}

impl Float64NumericState {
    fn new(frame: TypedFrame, entity_rows: usize) -> Self {
        Self {
            samples: SlidingSampleQueue::new(frame, entity_rows),
            accumulator: WindowAccumulator::new(SumClass::Float),
        }
    }

    fn update(
        &mut self,
        event_time: i64,
        sample: Option<f64>,
        numerical_profile: RollingNumericalProfile,
        transition_count: u64,
        node_id: &str,
    ) -> Result<()> {
        if let Some(sample) = sample {
            add_float64(&mut self.accumulator, sample, node_id)?;
        }
        self.samples.push(event_time, sample);
        self.expire(event_time)?;
        self.repair_accumulator(numerical_profile, transition_count, node_id)
    }

    fn repair_accumulator(
        &mut self,
        numerical_profile: RollingNumericalProfile,
        transition_count: u64,
        node_id: &str,
    ) -> Result<()> {
        // DataFusion 54's sliding AVG accumulator remains NaN after a NaN has
        // entered the partition, even after that row leaves a bounded frame.
        // The compatibility plan preserves that observable behavior.
        if self.accumulator.nan_count > 0 {
            return Ok(());
        }
        if numerical_profile == RollingNumericalProfile::StableV2Preview
            && stable_v2_rebase_due(
                transition_count,
                self.samples.len(),
                self.accumulator.is_non_finite(),
            )
        {
            self.rebase_stable_v2(node_id)?;
        } else if self.accumulator.is_non_finite() {
            refold_float64(&mut self.accumulator, &self.samples.values, node_id)?;
        }
        Ok(())
    }

    fn rebase_stable_v2(&mut self, node_id: &str) -> Result<()> {
        self.accumulator = stable_v2_float64_accumulator(
            self.samples.values.iter().filter_map(|sample| sample.value),
            node_id,
        )?;
        Ok(())
    }

    fn expire(&mut self, event_time: i64) -> Result<()> {
        let accumulator = &mut self.accumulator;
        self.samples
            .expire(event_time, |value| remove_float64(accumulator, value))
    }

    fn estimated_bytes(&self) -> usize {
        // Historical basis: the float state sizes by the bare Option<f64>,
        // not the full timed sample; preserve it exactly.
        self.samples
            .estimated_bytes(size_of::<Self>(), size_of::<Option<f64>>())
    }
}

#[derive(Clone, Debug)]
struct TypedExtremaState {
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

impl TypedExtremaState {
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
    samples: SlidingSampleQueue<(f64, f64)>,
    accumulator: PairAccumulator,
}

impl Float64PairState {
    fn new(frame: TypedFrame, entity_rows: usize) -> Self {
        Self {
            samples: SlidingSampleQueue::new(frame, entity_rows),
            accumulator: PairAccumulator::default(),
        }
    }

    fn update(
        &mut self,
        event_time: i64,
        value: Option<(f64, f64)>,
        numerical_profile: RollingNumericalProfile,
        transition_count: u64,
        node_id: &str,
    ) -> Result<()> {
        if let Some((left, right)) = value {
            add_pair_float64(&mut self.accumulator, left, right, node_id)?;
        }
        self.samples.push(event_time, value);
        self.expire(event_time)?;
        self.repair_accumulator(numerical_profile, transition_count, node_id)
    }

    fn repair_accumulator(
        &mut self,
        numerical_profile: RollingNumericalProfile,
        transition_count: u64,
        node_id: &str,
    ) -> Result<()> {
        if numerical_profile == RollingNumericalProfile::StableV2Preview
            && stable_v2_rebase_due(
                transition_count,
                self.samples.len(),
                self.accumulator.is_non_finite(),
            )
        {
            self.rebase_stable_v2(node_id)?;
        } else if self.accumulator.is_non_finite() {
            refold_pair_float64(&mut self.accumulator, &self.samples.values, node_id)?;
        }
        Ok(())
    }

    fn rebase_stable_v2(&mut self, node_id: &str) -> Result<()> {
        self.accumulator = stable_v2_pair_accumulator(
            self.samples.values.iter().filter_map(|sample| sample.value),
            node_id,
        )?;
        Ok(())
    }

    fn expire(&mut self, event_time: i64) -> Result<()> {
        let accumulator = &mut self.accumulator;
        self.samples.expire(event_time, |(left, right)| {
            remove_pair_float64(accumulator, left, right)
        })
    }

    fn estimated_bytes(&self) -> usize {
        self.samples
            .estimated_bytes(size_of::<Self>(), size_of::<TimedSample<(f64, f64)>>())
    }
}

#[derive(Clone, Copy, Debug)]
struct TypedEwmaState {
    alpha: f64,
    valid_count: u64,
    value: f64,
}

impl TypedEwmaState {
    const fn new(alpha: f64) -> Self {
        Self {
            alpha,
            valid_count: 0,
            value: 0.0,
        }
    }

    fn update(&mut self, sample: Option<f64>, node_id: &str) -> Result<()> {
        let Some(sample) = sample else {
            return Ok(());
        };
        self.value = if self.valid_count == 0 {
            sample
        } else {
            self.value + self.alpha * (sample - self.value)
        };
        self.valid_count = self
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling EWMA sample count overflowed"))?;
        Ok(())
    }

    const fn estimated_bytes() -> usize {
        size_of::<Self>()
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
    if sample.is_nan() {
        accumulator.nan_count = accumulator.nan_count.saturating_add(1);
        return Ok(());
    }
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
    if sample.is_nan() {
        // Deliberately sticky for DataFusion 54 AVG compatibility. The normal
        // rolling API filters NaNs before they can reach this transition.
        return Ok(());
    }
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
    values: &VecDeque<TimedSample<f64>>,
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
    values: &VecDeque<TimedSample<(f64, f64)>>,
    node_id: &str,
) -> Result<()> {
    *accumulator = PairAccumulator::default();
    for (left, right) in values.iter().filter_map(|sample| sample.value) {
        add_pair_float64(accumulator, left, right, node_id)?;
    }
    Ok(())
}

pub(super) fn stable_v2_rebase_due(
    transition_count: u64,
    retained_samples: usize,
    numerical_risk: bool,
) -> bool {
    if numerical_risk {
        return true;
    }
    if retained_samples < 2 {
        return false;
    }
    let cadence = u64::try_from(retained_samples.max(64)).unwrap_or(u64::MAX);
    transition_count % cadence == 0
}

#[derive(Default)]
struct CompensatedSum {
    value: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, sample: f64) {
        let adjusted = sample - self.correction;
        let next = self.value + adjusted;
        self.correction = (next - self.value) - adjusted;
        self.value = next;
    }
}

struct StableFloat64Fold {
    accumulator: WindowAccumulator,
    origin: Option<f64>,
    finite_count: u64,
    offsets: CompensatedSum,
    offset_squares: CompensatedSum,
}

impl StableFloat64Fold {
    fn new() -> Self {
        Self {
            accumulator: WindowAccumulator::new(SumClass::Float),
            origin: None,
            finite_count: 0,
            offsets: CompensatedSum::default(),
            offset_squares: CompensatedSum::default(),
        }
    }

    fn push(&mut self, sample: f64, node_id: &str) -> Result<()> {
        self.accumulator.valid_count = self
            .accumulator
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling valid sample count overflowed"))?;
        if sample.is_nan() {
            self.accumulator.nan_count = self.accumulator.nan_count.saturating_add(1);
            return Ok(());
        }
        if sample.is_infinite() {
            update_infinity_count(
                &mut self.accumulator.pos_inf,
                &mut self.accumulator.neg_inf,
                sample,
                true,
            );
            return Ok(());
        }
        let origin = *self.origin.get_or_insert(sample);
        let offset = sample - origin;
        self.offsets.add(offset);
        self.offset_squares.add(offset * offset);
        self.finite_count = self
            .finite_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling finite sample count overflowed"))?;
        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the stable_v2 preview retains the frozen Float64 output type"
    )]
    fn finish(mut self) -> WindowAccumulator {
        let origin = self.origin.unwrap_or(0.0);
        let finite_total = origin * self.finite_count as f64 + self.offsets.value;
        let total = match (self.accumulator.pos_inf > 0, self.accumulator.neg_inf > 0) {
            (true, true) => f64::NAN,
            (true, false) => f64::INFINITY,
            (false, true) => f64::NEG_INFINITY,
            (false, false) => finite_total,
        };
        self.accumulator.sum = Some(SumState::Float(total));
        if self.finite_count > 0 {
            let count = self.finite_count as f64;
            self.accumulator.mean = origin + self.offsets.value / count;
            self.accumulator.m2 =
                self.offset_squares.value - self.offsets.value * self.offsets.value / count;
        }
        self.accumulator
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the stable_v2 preview retains the frozen Float64 output type"
)]
pub(super) fn stable_v2_float64_accumulator(
    values: impl Iterator<Item = f64>,
    node_id: &str,
) -> Result<WindowAccumulator> {
    let mut fold = StableFloat64Fold::new();
    for sample in values {
        fold.push(sample, node_id)?;
    }
    Ok(fold.finish())
}

#[derive(Default)]
struct StablePairFold {
    accumulator: PairAccumulator,
    origin: Option<(f64, f64)>,
    finite_count: u64,
    left_offsets: CompensatedSum,
    right_offsets: CompensatedSum,
    left_squares: CompensatedSum,
    right_squares: CompensatedSum,
    cross_products: CompensatedSum,
}

impl StablePairFold {
    fn push(&mut self, left: f64, right: f64, node_id: &str) -> Result<()> {
        self.accumulator.valid_count = self
            .accumulator
            .valid_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling pair sample count overflowed"))?;
        update_pair_infinity_counts(&mut self.accumulator, left, right, true);
        if !left.is_finite() || !right.is_finite() {
            return Ok(());
        }
        let (origin_x, origin_y) = *self.origin.get_or_insert((left, right));
        let x = left - origin_x;
        let y = right - origin_y;
        self.left_offsets.add(x);
        self.right_offsets.add(y);
        self.left_squares.add(x * x);
        self.right_squares.add(y * y);
        self.cross_products.add(x * y);
        self.finite_count = self
            .finite_count
            .checked_add(1)
            .ok_or_else(|| operator_error(node_id, "rolling finite pair count overflowed"))?;
        Ok(())
    }

    #[allow(
        clippy::cast_precision_loss,
        reason = "the stable_v2 preview retains the frozen Float64 output type"
    )]
    fn finish(mut self) -> PairAccumulator {
        if let Some((origin_x, origin_y)) = self.origin {
            let count = self.finite_count as f64;
            self.accumulator.mean_x = origin_x + self.left_offsets.value / count;
            self.accumulator.mean_y = origin_y + self.right_offsets.value / count;
            self.accumulator.m2_x =
                self.left_squares.value - self.left_offsets.value * self.left_offsets.value / count;
            self.accumulator.m2_y = self.right_squares.value
                - self.right_offsets.value * self.right_offsets.value / count;
            self.accumulator.co_moment = self.cross_products.value
                - self.left_offsets.value * self.right_offsets.value / count;
        }
        self.accumulator
    }
}

pub(super) fn stable_v2_pair_accumulator(
    values: impl Iterator<Item = (f64, f64)>,
    node_id: &str,
) -> Result<PairAccumulator> {
    let mut fold = StablePairFold::default();
    for (left, right) in values {
        fold.push(left, right, node_id)?;
    }
    Ok(fold.finish())
}

enum DerivedBuilder {
    Count(UInt64Builder),
    Signed(Int64Builder, OutputStorage),
    Unsigned(UInt64Builder, OutputStorage),
    Float(Float64Builder, OutputStorage),
}

impl DerivedBuilder {
    fn new(output: TypedOutputPlan, capacity: usize) -> Self {
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
        states: &[TypedWindowState],
        output: &TypedOutputPlan,
        node_id: &str,
    ) -> Result<()> {
        if let TypedOutputKind::Difference { left, right } = output.kind {
            return self.append_difference(states, left, right);
        }
        let state = states
            .get(output.group)
            .ok_or_else(|| internal_error("typed rolling output group is out of bounds"))?;
        if typed_valid_count(state) < output.min_periods {
            self.append_null();
            return Ok(());
        }
        match state {
            TypedWindowState::Numeric(state) => self.append_numeric(state, output),
            TypedWindowState::Exact(state) => self.append_exact(state, output, node_id),
            TypedWindowState::Extrema(state) => self.append_extrema(state, output),
            TypedWindowState::Pair(state) => self.append_pair(state, output),
            TypedWindowState::Ewma(state) => self.append_ewma(state, output),
        }
    }

    fn append_numeric(
        &mut self,
        state: &Float64NumericState,
        output: &TypedOutputPlan,
    ) -> Result<()> {
        match (self, output.kind) {
            (Self::Count(builder), TypedOutputKind::Statistic(Statistic::Count)) => {
                builder.append_value(state.accumulator.valid_count);
                Ok(())
            }
            (Self::Float(builder, _), TypedOutputKind::Statistic(statistic)) => {
                append_float(builder, &state.accumulator, statistic, output.ddof)
            }
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_exact(
        &mut self,
        state: &ExactNumericState,
        output: &TypedOutputPlan,
        node_id: &str,
    ) -> Result<()> {
        match (self, output.kind) {
            (Self::Signed(builder, _), TypedOutputKind::Statistic(Statistic::Sum)) => {
                append_signed_sum(builder, &state.accumulator, node_id)
            }
            (Self::Unsigned(builder, _), TypedOutputKind::Statistic(Statistic::Sum)) => {
                append_unsigned_sum(builder, &state.accumulator, node_id)
            }
            (Self::Count(builder), TypedOutputKind::Statistic(Statistic::Count)) => {
                builder.append_value(state.accumulator.valid_count);
                Ok(())
            }
            (Self::Float(builder, _), TypedOutputKind::Statistic(statistic)) => {
                append_exact_float(builder, &state.accumulator, statistic, output.ddof)
            }
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_extrema(
        &mut self,
        state: &TypedExtremaState,
        output: &TypedOutputPlan,
    ) -> Result<()> {
        if !matches!(
            output.kind,
            TypedOutputKind::Statistic(Statistic::Min | Statistic::Max)
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

    fn append_pair(&mut self, state: &Float64PairState, output: &TypedOutputPlan) -> Result<()> {
        match (self, output.kind) {
            (
                Self::Float(builder, _),
                TypedOutputKind::Covariance | TypedOutputKind::Correlation,
            ) => append_pair(builder, &state.accumulator, output),
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_ewma(&mut self, state: &TypedEwmaState, output: &TypedOutputPlan) -> Result<()> {
        match (self, output.kind) {
            (Self::Float(builder, _), TypedOutputKind::Ewma) => {
                builder.append_value(state.value);
                Ok(())
            }
            _ => Err(typed_output_mismatch()),
        }
    }

    fn append_difference(
        &mut self,
        states: &[TypedWindowState],
        left: TypedFloatReadout,
        right: TypedFloatReadout,
    ) -> Result<()> {
        let Self::Float(builder, _) = self else {
            return Err(typed_output_mismatch());
        };
        match (
            read_typed_float(states, left)?,
            read_typed_float(states, right)?,
        ) {
            (Some(left), Some(right)) => builder.append_value(left - right),
            _ => builder.append_null(),
        }
        Ok(())
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
        internal_error(format!(
            "typed rolling output cast to {data_type} failed: {error}"
        ))
    })
}

fn append_float_extrema(builder: &mut Float64Builder, state: &TypedExtremaState) -> Result<()> {
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

fn append_signed_extrema(builder: &mut Int64Builder, state: &TypedExtremaState) -> Result<()> {
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

fn append_unsigned_extrema(builder: &mut UInt64Builder, state: &TypedExtremaState) -> Result<()> {
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

fn typed_valid_count(state: &TypedWindowState) -> u64 {
    match state {
        TypedWindowState::Numeric(state) => state.accumulator.valid_count,
        TypedWindowState::Exact(state) => state.accumulator.valid_count,
        TypedWindowState::Extrema(state) => state.valid_count,
        TypedWindowState::Pair(state) => state.accumulator.valid_count,
        TypedWindowState::Ewma(state) => state.valid_count,
    }
}

fn read_typed_float(
    states: &[TypedWindowState],
    readout: TypedFloatReadout,
) -> Result<Option<f64>> {
    let state = states
        .get(readout.group)
        .ok_or_else(|| internal_error("typed fused readout group is out of bounds"))?;
    if typed_valid_count(state) < readout.min_periods {
        return Ok(None);
    }
    if readout.kind == TypedFloatReadoutKind::Ewma {
        let TypedWindowState::Ewma(state) = state else {
            return Err(internal_error("typed fused EWMA readout state mismatch"));
        };
        return Ok(Some(state.value));
    }
    let accumulator = match state {
        TypedWindowState::Numeric(state) => &state.accumulator,
        TypedWindowState::Exact(state) => &state.accumulator,
        _ => {
            return Err(internal_error(
                "typed fused aggregate readout state mismatch",
            ));
        }
    };
    match readout.kind {
        TypedFloatReadoutKind::Mean => Ok(Some(float_mean(accumulator))),
        TypedFloatReadoutKind::Variance | TypedFloatReadoutKind::Stddev => Ok(dispersion_value(
            accumulator,
            readout.kind == TypedFloatReadoutKind::Stddev,
            readout.ddof,
        )),
        TypedFloatReadoutKind::Ewma => unreachable!("EWMA readouts return before aggregation"),
    }
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen variance output type is Float64"
)]
fn dispersion_value(
    accumulator: &WindowAccumulator,
    standard_deviation: bool,
    ddof: u8,
) -> Option<f64> {
    let divisor = accumulator.valid_count - u64::from(ddof);
    if divisor == 0 {
        return None;
    }
    if accumulator.nan_count > 0 || accumulator.pos_inf > 0 || accumulator.neg_inf > 0 {
        return Some(f64::NAN);
    }
    let variance = accumulator.m2.max(0.0) / divisor as f64;
    Some(if standard_deviation {
        variance.sqrt()
    } else {
        variance
    })
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
    output: &TypedOutputPlan,
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
        TypedOutputKind::Covariance => {
            builder.append_value(accumulator.co_moment / divisor as f64);
        }
        TypedOutputKind::Correlation => {
            append_correlation(builder, accumulator);
        }
        TypedOutputKind::Statistic(_) => {
            return Err(internal_error(
                "typed pair state received a scalar statistic",
            ));
        }
        TypedOutputKind::Ewma => {
            return Err(internal_error("typed pair state received an EWMA output"));
        }
        TypedOutputKind::Difference { .. } => {
            return Err(internal_error(
                "typed pair state received a fused difference output",
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
    if accumulator.nan_count > 0 {
        return f64::NAN;
    }
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
    if let Some(value) = dispersion_value(accumulator, statistic == Statistic::Stddev, ddof) {
        builder.append_value(value);
    } else {
        builder.append_null();
    }
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::{
        array::{Array, Float64Array, StringArray},
        datatypes::{Field, TimeUnit},
    };

    use super::*;

    fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "event_time",
                DataType::Timestamp(TimeUnit::Microsecond, None),
                true,
            ),
            Field::new("sequence", DataType::UInt64, true),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("price", DataType::Float64, true),
        ]))
    }

    fn batch(
        event_times: Vec<Option<i64>>,
        sequences: Vec<Option<u64>>,
        symbols: Vec<&str>,
        prices: Vec<Option<f64>>,
    ) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(TimestampMicrosecondArray::from(event_times)),
                Arc::new(UInt64Array::from(sequences)),
                Arc::new(StringArray::from(symbols)),
                Arc::new(Float64Array::from(prices)),
            ],
        )
        .unwrap()
    }

    fn plan(
        groups: Vec<TypedGroupPlan>,
        outputs: Vec<TypedOutputPlan>,
        numerical_profile: RollingNumericalProfile,
    ) -> RollingKernelPlan {
        RollingKernelPlan {
            version: ROLLING_KERNEL_PLAN_VERSION,
            state_layout_version: 3,
            numerical_profile,
            selection: KernelSelection::OrderedPrimitive,
            complexity: KernelComplexity::AmortizedConstant,
            event_time_index: 0,
            order_columns: vec![0, 1],
            partition_columns: vec![2],
            sequence_columns: vec![1],
            nan_as_value: false,
            groups,
            outputs,
            fallback_reason: None,
            estimated_state_bytes_per_entity: 0,
            fingerprint: "test-kernel".to_owned(),
        }
    }

    fn numeric_plan(profile: RollingNumericalProfile) -> RollingKernelPlan {
        plan(
            vec![TypedGroupPlan::Numeric {
                input_index: 3,
                frame: TypedFrame::Rows(2),
            }],
            vec![TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Statistic(Statistic::Mean),
                storage: OutputStorage::Float64,
                min_periods: 1,
                ddof: 0,
            }],
            profile,
        )
    }

    #[test]
    fn typed_state_estimated_bytes_keep_their_historical_accounting_basis() {
        use std::mem::size_of;

        let frame = TypedFrame::Rows(4);
        let mut exact = ExactNumericState::new(frame, SumClass::Signed, 8);
        let mut float = Float64NumericState::new(frame, 8);
        let mut pair = Float64PairState::new(frame, 8);
        for index in 0..4i64 {
            #[allow(clippy::cast_precision_loss, reason = "fixture values 0..4")]
            let step = index as f64;
            let _ = exact.update(index, Some(ExactValue::Signed(index)), "node");
            let _ = float.update(
                index,
                Some(step),
                RollingNumericalProfile::StableV1,
                0,
                "node",
            );
            let _ = pair.update(
                index,
                Some((step, 1.0)),
                RollingNumericalProfile::StableV1,
                0,
                "node",
            );
        }

        // The exact and pair states size by the full timed sample; the float
        // state historically sizes by the bare Option<f64>. The shared queue
        // must preserve each basis exactly.
        assert_eq!(
            exact.estimated_bytes(),
            size_of::<ExactNumericState>()
                + exact.samples.values.capacity() * size_of::<TimedSample<ExactValue>>()
        );
        assert_eq!(
            pair.estimated_bytes(),
            size_of::<Float64PairState>()
                + pair.samples.values.capacity() * size_of::<TimedSample<(f64, f64)>>()
        );
        assert_eq!(
            float.estimated_bytes(),
            size_of::<Float64NumericState>()
                + float.samples.values.capacity() * size_of::<Option<f64>>()
        );
    }

    #[test]
    fn restored_typed_state_validates_and_seeds_recurrence_metadata() {
        let kernel = plan(
            vec![
                TypedGroupPlan::Numeric {
                    input_index: 3,
                    frame: TypedFrame::Rows(64),
                },
                TypedGroupPlan::Ewma {
                    input_index: 3,
                    alpha: 0.5,
                },
            ],
            Vec::new(),
            RollingNumericalProfile::StableV2Preview,
        );
        let entities = batch(
            vec![Some(1), Some(2)],
            vec![Some(1), Some(1)],
            vec!["a", "b"],
            vec![None, None],
        );
        let seeds = vec![vec![None, Some((3, 12.5))], vec![None, None]];
        let seeded = kernel
            .seed_restored_state(
                &RollingKernelState::default(),
                &entities,
                &[64, 7],
                &seeds,
                "r",
            )
            .unwrap();

        assert_eq!(seeded.states.len(), 2);
        assert_eq!(seeded.states[0].transition_count, 64);
        let TypedWindowState::Ewma(ewma) = seeded.states[0].groups[1] else {
            panic!("expected EWMA state")
        };
        assert_eq!((ewma.valid_count, ewma.value), (3, 12.5));

        assert!(
            kernel
                .seed_restored_state(&RollingKernelState::default(), &entities, &[1], &seeds, "r",)
                .is_err()
        );
        assert!(
            kernel
                .seed_restored_state(
                    &RollingKernelState::default(),
                    &entities.slice(0, 1),
                    &[1],
                    &[vec![None]],
                    "r",
                )
                .is_err()
        );
        assert!(
            kernel
                .seed_restored_state(
                    &RollingKernelState::default(),
                    &entities.slice(0, 1),
                    &[1],
                    &[vec![Some((1, 1.0)), None]],
                    "r",
                )
                .is_err()
        );
        assert!(
            kernel
                .seed_restored_state(
                    &RollingKernelState::default(),
                    &entities.slice(0, 1),
                    &[1],
                    &[vec![None, Some((0, 1.0))]],
                    "r",
                )
                .is_err()
        );

        let foreign = RollingKernelState {
            kernel_fingerprint: Some("foreign".to_owned()),
            ..RollingKernelState::default()
        };
        assert!(
            kernel
                .seed_restored_state(&foreign, &entities, &[1, 1], &seeds, "r")
                .is_err()
        );
    }

    #[test]
    fn canonical_order_contract_covers_batch_and_stream_boundaries() {
        let kernel = numeric_plan(RollingNumericalProfile::StableV1);
        let first = batch(vec![Some(1)], vec![Some(1)], vec!["a"], vec![Some(1.0)]);
        let first_output = kernel.open_and_fill(&first, "r").unwrap().unwrap();

        let duplicate = kernel.update_and_fill(&first_output.state, &first, "r");
        assert!(duplicate.is_err());
        assert!(
            kernel
                .update_stream_and_fill(&first_output.state, &first, "r")
                .unwrap()
                .is_some()
        );

        let duplicate_within_batch = batch(
            vec![Some(1), Some(1)],
            vec![Some(1), Some(1)],
            vec!["a", "a"],
            vec![Some(1.0), Some(2.0)],
        );
        assert!(kernel.open_and_fill(&duplicate_within_batch, "r").is_err());

        let unsorted = batch(
            vec![Some(2), Some(1)],
            vec![Some(2), Some(1)],
            vec!["a", "a"],
            vec![Some(2.0), Some(1.0)],
        );
        assert!(kernel.open_and_fill(&unsorted, "r").unwrap().is_none());

        let null_time = batch(vec![None], vec![Some(1)], vec!["a"], vec![Some(1.0)]);
        assert!(kernel.open_and_fill(&null_time, "r").is_err());
        let null_sequence = batch(vec![Some(1)], vec![None], vec!["a"], vec![Some(1.0)]);
        assert!(kernel.open_and_fill(&null_sequence, "r").is_err());

        let mut general = kernel;
        general.selection = KernelSelection::General;
        assert!(general.open_and_fill(&first, "r").unwrap().is_none());
    }

    #[test]
    fn stable_preview_refolds_numeric_and_pair_windows() {
        assert!(stable_v2_rebase_due(1, 1, true));
        assert!(!stable_v2_rebase_due(1, 1, false));
        assert!(stable_v2_rebase_due(64, 64, false));

        let finite =
            stable_v2_float64_accumulator([1.0e12, 1.0e12 + 0.25, 1.0e12 + 0.5].into_iter(), "r")
                .unwrap();
        assert_eq!(finite.valid_count, 3);
        assert!(finite.m2 > 0.0);
        assert!(matches!(
            stable_v2_float64_accumulator([f64::INFINITY].into_iter(), "r")
                .unwrap()
                .sum,
            Some(SumState::Float(value)) if value == f64::INFINITY
        ));
        assert!(matches!(
            stable_v2_float64_accumulator([f64::NEG_INFINITY].into_iter(), "r")
                .unwrap()
                .sum,
            Some(SumState::Float(value)) if value == f64::NEG_INFINITY
        ));
        let mixed =
            stable_v2_float64_accumulator([f64::INFINITY, f64::NEG_INFINITY].into_iter(), "r")
                .unwrap();
        assert!(matches!(mixed.sum, Some(SumState::Float(value)) if value.is_nan()));

        let pair = stable_v2_pair_accumulator(
            [
                (1.0e12, -1.0e12),
                (1.0e12 + 1.0, -1.0e12 + 2.0),
                (f64::INFINITY, 3.0),
            ]
            .into_iter(),
            "r",
        )
        .unwrap();
        assert_eq!(pair.valid_count, 3);
        assert_eq!(pair.pos_inf_x, 1);
        assert!(pair.co_moment > 0.0);

        let mut numeric = Float64NumericState::new(TypedFrame::Rows(64), 64);
        for transition in 1_i32..=64 {
            numeric
                .update(
                    i64::from(transition),
                    Some(1.0e12 + f64::from(transition)),
                    RollingNumericalProfile::StableV2Preview,
                    u64::try_from(transition).unwrap(),
                    "r",
                )
                .unwrap();
        }
        assert_eq!(numeric.samples.len(), 64);

        let mut pairs = Float64PairState::new(TypedFrame::Rows(64), 64);
        for transition in 1_i32..=64 {
            pairs
                .update(
                    i64::from(transition),
                    Some((f64::from(transition), f64::from(transition * 2))),
                    RollingNumericalProfile::StableV2Preview,
                    u64::try_from(transition).unwrap(),
                    "r",
                )
                .unwrap();
        }
        assert_eq!(pairs.samples.len(), 64);
        assert!(pairs.accumulator.co_moment > 0.0);
    }

    #[test]
    fn primitive_state_updates_preserve_expiration_and_type_invariants() {
        let mut signed = ExactNumericState::new(TypedFrame::Rows(1), SumClass::Signed, 2);
        signed.update(1, Some(ExactValue::Signed(4)), "r").unwrap();
        signed.update(2, Some(ExactValue::Signed(-1)), "r").unwrap();
        assert!(matches!(signed.accumulator.sum, Some(SumState::Signed(-1))));

        let mut unsigned = ExactNumericState::new(TypedFrame::Duration(2), SumClass::Unsigned, 0);
        unsigned
            .update(1, Some(ExactValue::Unsigned(4)), "r")
            .unwrap();
        unsigned
            .update(3, Some(ExactValue::Unsigned(2)), "r")
            .unwrap();
        assert!(matches!(
            unsigned.accumulator.sum,
            Some(SumState::Unsigned(2))
        ));

        let mut extrema = TypedExtremaState::new(TypedFrame::Rows(2), false, 3);
        extrema
            .update(1, Some(ExtremaValue::Signed(4)), "r")
            .unwrap();
        extrema
            .update(2, Some(ExtremaValue::Signed(2)), "r")
            .unwrap();
        extrema.update(3, None, "r").unwrap();
        assert!(matches!(
            extrema.candidates.front(),
            Some((_, ExtremaValue::Signed(2)))
        ));
        assert!(
            ExtremaValue::Signed(1)
                .total_cmp(ExtremaValue::Unsigned(1))
                .is_err()
        );

        let mut ewma = TypedEwmaState::new(0.5);
        ewma.update(None, "r").unwrap();
        ewma.update(Some(10.0), "r").unwrap();
        ewma.update(Some(14.0), "r").unwrap();
        assert_eq!((ewma.valid_count, ewma.value), (2, 12.0));

        let mut pair = Float64PairState::new(TypedFrame::Duration(2), 0);
        pair.update(
            1,
            Some((1.0, 2.0)),
            RollingNumericalProfile::StableV1,
            1,
            "r",
        )
        .unwrap();
        pair.update(
            3,
            Some((3.0, 6.0)),
            RollingNumericalProfile::StableV1,
            2,
            "r",
        )
        .unwrap();
        assert_eq!(pair.accumulator.valid_count, 1);
    }

    #[test]
    fn accumulator_boundaries_fail_closed_without_wraparound() {
        let mut ewma = TypedEwmaState {
            alpha: 0.5,
            valid_count: u64::MAX,
            value: 1.0,
        };
        assert!(ewma.update(Some(2.0), "r").is_err());

        let mut wrong_float = WindowAccumulator::new(SumClass::Signed);
        assert!(add_float64(&mut wrong_float, 1.0, "r").is_err());
        let mut empty_float = WindowAccumulator::new(SumClass::Float);
        assert!(remove_float64(&mut empty_float, 1.0).is_err());
        let mut wrong_remove = WindowAccumulator::new(SumClass::Signed);
        wrong_remove.valid_count = 1;
        assert!(remove_float64(&mut wrong_remove, 1.0).is_err());

        let mut signed = WindowAccumulator::new(SumClass::Signed);
        assert!(add_exact(&mut signed, ExactValue::Unsigned(1), "r").is_err());
        let mut unsigned = WindowAccumulator::new(SumClass::Unsigned);
        assert!(remove_exact(&mut unsigned, ExactValue::Unsigned(1)).is_err());
        unsigned.valid_count = 1;
        assert!(remove_exact(&mut unsigned, ExactValue::Unsigned(1)).is_err());
        let mut mismatched = WindowAccumulator::new(SumClass::Signed);
        mismatched.valid_count = 1;
        assert!(remove_exact(&mut mismatched, ExactValue::Unsigned(1)).is_err());

        let mut pair = PairAccumulator {
            valid_count: u64::MAX,
            ..PairAccumulator::default()
        };
        assert!(add_pair_float64(&mut pair, 1.0, 2.0, "r").is_err());
        let mut empty_pair = PairAccumulator::default();
        assert!(remove_pair_float64(&mut empty_pair, 1.0, 2.0).is_err());
        add_pair_float64(&mut empty_pair, f64::INFINITY, f64::NEG_INFINITY, "r").unwrap();
        remove_pair_float64(&mut empty_pair, f64::INFINITY, f64::NEG_INFINITY).unwrap();
        assert_eq!(empty_pair.valid_count, 0);
        assert_eq!((empty_pair.pos_inf_x, empty_pair.neg_inf_y), (0, 0));
        assert_eq!(
            (
                empty_pair.mean_x,
                empty_pair.mean_y,
                empty_pair.co_moment,
                empty_pair.m2_x,
                empty_pair.m2_y,
            ),
            (0.0, 0.0, 0.0, 0.0, 0.0)
        );
    }

    #[test]
    fn typed_readouts_and_builders_enforce_state_shape_contracts() {
        let mean = TypedOutputPlan {
            group: 0,
            kind: TypedOutputKind::Statistic(Statistic::Mean),
            storage: OutputStorage::Float64,
            min_periods: 1,
            ddof: 0,
        };
        let mut numeric = Float64NumericState::new(TypedFrame::Rows(2), 2);
        numeric
            .update(1, Some(4.0), RollingNumericalProfile::StableV1, 1, "r")
            .unwrap();
        let states = vec![TypedWindowState::Numeric(numeric.clone())];

        let mut wrong_builder = DerivedBuilder::new(
            TypedOutputPlan {
                storage: OutputStorage::Count,
                ..mean
            },
            1,
        );
        assert!(wrong_builder.append(&states, &mean, "r").is_err());
        let mut out_of_bounds = DerivedBuilder::new(mean, 1);
        assert!(
            out_of_bounds
                .append(&states, &TypedOutputPlan { group: 1, ..mean }, "r")
                .is_err()
        );

        let minimum_two = TypedOutputPlan {
            min_periods: 2,
            ..mean
        };
        let mut null_builder = DerivedBuilder::new(minimum_two, 1);
        null_builder.append(&states, &minimum_two, "r").unwrap();
        assert_eq!(null_builder.finish().unwrap().null_count(), 1);

        let bad_ewma_readout = TypedFloatReadout {
            group: 0,
            kind: TypedFloatReadoutKind::Ewma,
            min_periods: 1,
            ddof: 0,
        };
        assert!(read_typed_float(&states, bad_ewma_readout).is_err());
        assert!(
            read_typed_float(
                &states,
                TypedFloatReadout {
                    group: 1,
                    ..bad_ewma_readout
                }
            )
            .is_err()
        );
        let extrema_state =
            TypedWindowState::Extrema(TypedExtremaState::new(TypedFrame::Rows(1), false, 1));
        assert!(
            read_typed_float(
                &[extrema_state],
                TypedFloatReadout {
                    group: 0,
                    kind: TypedFloatReadoutKind::Mean,
                    min_periods: 0,
                    ddof: 0,
                }
            )
            .is_err()
        );
    }

    #[test]
    fn typed_builders_reject_mismatched_accumulator_storage() {
        let mut signed = ExactNumericState::new(TypedFrame::Rows(2), SumClass::Signed, 2);
        signed.update(1, Some(ExactValue::Signed(4)), "r").unwrap();
        let signed_sum_output = TypedOutputPlan {
            group: 0,
            kind: TypedOutputKind::Statistic(Statistic::Sum),
            storage: OutputStorage::Int64,
            min_periods: 1,
            ddof: 0,
        };
        let mut signed_sum = DerivedBuilder::new(signed_sum_output, 1);
        signed_sum
            .append(&[TypedWindowState::Exact(signed)], &signed_sum_output, "r")
            .unwrap();
        assert_eq!(signed_sum.finish().unwrap().len(), 1);

        let mut wrong_extrema = TypedExtremaState::new(TypedFrame::Rows(1), false, 1);
        wrong_extrema
            .candidates
            .push_back((0, ExtremaValue::Unsigned(1)));
        assert!(append_signed_extrema(&mut Int64Builder::new(), &wrong_extrema).is_err());
        assert!(append_float_extrema(&mut Float64Builder::new(), &wrong_extrema).is_err());
        wrong_extrema.candidates.clear();
        wrong_extrema
            .candidates
            .push_back((0, ExtremaValue::Signed(1)));
        assert!(append_unsigned_extrema(&mut UInt64Builder::new(), &wrong_extrema).is_err());
    }

    #[test]
    fn typed_pair_and_float_builders_reject_invalid_output_kinds() {
        let mut numeric = Float64NumericState::new(TypedFrame::Rows(2), 2);
        numeric
            .update(1, Some(4.0), RollingNumericalProfile::StableV1, 1, "r")
            .unwrap();
        let bad_ewma_readout = TypedFloatReadout {
            group: 0,
            kind: TypedFloatReadoutKind::Ewma,
            min_periods: 1,
            ddof: 0,
        };
        let mut pair = PairAccumulator::default();
        let mut pair_builder = Float64Builder::new();
        for kind in [
            TypedOutputKind::Statistic(Statistic::Mean),
            TypedOutputKind::Ewma,
            TypedOutputKind::Difference {
                left: bad_ewma_readout,
                right: bad_ewma_readout,
            },
        ] {
            pair.valid_count = 1;
            assert!(
                append_pair(
                    &mut pair_builder,
                    &pair,
                    &TypedOutputPlan {
                        group: 0,
                        kind,
                        storage: OutputStorage::Float64,
                        min_periods: 1,
                        ddof: 0,
                    },
                )
                .is_err()
            );
        }
        pair.valid_count = 0;
        append_pair(
            &mut pair_builder,
            &pair,
            &TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Covariance,
                storage: OutputStorage::Float64,
                min_periods: 0,
                ddof: 0,
            },
        )
        .unwrap();
        pair.valid_count = 1;
        pair.pos_inf_x = 1;
        append_pair(
            &mut pair_builder,
            &pair,
            &TypedOutputPlan {
                group: 0,
                kind: TypedOutputKind::Correlation,
                storage: OutputStorage::Float64,
                min_periods: 1,
                ddof: 0,
            },
        )
        .unwrap();

        let mut invalid_float = WindowAccumulator::new(SumClass::Signed);
        invalid_float.valid_count = 1;
        assert!(append_sum(&mut Float64Builder::new(), &invalid_float).is_err());
        assert!(
            append_float(
                &mut Float64Builder::new(),
                &numeric.accumulator,
                Statistic::Count,
                0,
            )
            .is_err()
        );
        assert!(
            append_exact_float(
                &mut Float64Builder::new(),
                &invalid_float,
                Statistic::Sum,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn typed_compilation_rejects_non_numeric_and_invalid_group_shapes() {
        let all_types = Schema::new(vec![
            Field::new("i8", DataType::Int8, true),
            Field::new("i16", DataType::Int16, true),
            Field::new("i32", DataType::Int32, true),
            Field::new("i64", DataType::Int64, true),
            Field::new("u8", DataType::UInt8, true),
            Field::new("u16", DataType::UInt16, true),
            Field::new("u32", DataType::UInt32, true),
            Field::new("u64", DataType::UInt64, true),
            Field::new("f32", DataType::Float32, true),
            Field::new("f64", DataType::Float64, true),
            Field::new("text", DataType::Utf8, true),
        ]);
        for (index, storage) in [
            OutputStorage::Int8,
            OutputStorage::Int16,
            OutputStorage::Int32,
            OutputStorage::Int64,
            OutputStorage::UInt8,
            OutputStorage::UInt16,
            OutputStorage::UInt32,
            OutputStorage::UInt64,
            OutputStorage::Float32,
            OutputStorage::Float64,
        ]
        .into_iter()
        .enumerate()
        {
            assert_eq!(
                extrema_storage(all_types.field(index).data_type()).unwrap(),
                storage
            );
            require_numeric_type(&all_types, index).unwrap();
        }
        assert!(extrema_storage(&DataType::Utf8).is_err());
        assert!(require_numeric_type(&all_types, 10).is_err());
        assert!(
            compile_typed_plan(&all_types, &[], &[])
                .unwrap_err()
                .contains("at least one")
        );
        assert!(
            extrema_group_input(
                &(Arc::new(UInt64Array::from(vec![1])) as ArrayRef),
                OutputStorage::Count,
                "r",
            )
            .is_err()
        );
    }
}
