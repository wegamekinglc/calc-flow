//! Native cross-section operator: complete-group rank, percentile, z-score,
//! demean, winsorize, top/bottom selection, and mean-fill over exact-time or
//! fixed-bucket groups (SCE-00 D6, API note `symbolic-computation-engine`
//! section 3.2). One micro-batch is never
//! evidence of completeness: groups accumulate across envelopes and close
//! only by the watermark rules of D7 (or end-of-input), then emit once in
//! canonical order. The same calculation kernel serves the batch and stream
//! lifecycles; open groups are checkpointed at the aligned epoch cut.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet, HashMap},
    io::Cursor,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{ArrayRef, UInt8Array, new_null_array},
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::{
        convert::IpcSchemaEncoder,
        reader::FileReader,
        writer::{DictionaryTracker, FileWriter},
    },
    record_batch::RecordBatch,
};
use datafusion::common::ScalarValue;
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, Epoch, EventTime, JsonMap, Port, Result,
    StateHandle, TableBatch, canonical_json,
    state::{SegmentDescriptor, SegmentKind, StateInventory},
};

use super::checkpoint::{checkpoint_mismatch, compile_error, internal_error, state_format};
use super::{
    BatchOperator, BatchOperatorContext, LateMetricDelta, LatePolicySpec, OperatorMetadata,
    StreamCollector, StreamOperator, StreamOperatorContext, accumulate_late_metrics,
    expression::required_input, validate_operator_name,
};

/// Semantic configuration version of the first cross-section release.
pub const CROSS_SECTION_CONFIGURATION_VERSION: u32 = 1;
/// Durable state-layout version of the first cross-section release.
pub const CROSS_SECTION_STATE_LAYOUT_VERSION: u32 = 1;

/// Sort direction of an order-statistic output (SCE-00 D6).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SortDirection {
    /// Smallest measured value first.
    Ascending,
    /// Largest measured value first.
    Descending,
}

/// Rank tie method of an order-statistic output (SCE-00 D6).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RankTieMethod {
    /// Mean of the tied class's one-based position range.
    Average,
    /// First one-based position of the tied class.
    Min,
    /// Last one-based position of the tied class.
    Max,
}

/// Null placement of an order-statistic output (SCE-00 D6). Excluded nulls
/// produce null; included nulls form one tied class at the requested end of
/// the final sort order.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum NullPlacement {
    /// Null rows produce null and leave the ordering.
    Exclude,
    /// Null rows form the first tied class of the final sort order.
    First,
    /// Null rows form the last tied class of the final sort order.
    Last,
}

/// Frozen null/NaN policy for cross-section values (SCE-00 D3.2/D6).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CrossSectionValuePolicy {
    /// NaN is excluded from every sample and preserved at its own row;
    /// infinity stays numeric; null handling comes only from
    /// [`NullPlacement`] or row preservation.
    NanExcludePreserveV1,
}

/// Group membership rule of one cross-section operator (SCE-00 D6).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CrossSectionGroupingSpec {
    /// One group per exact event-time value and partition key.
    ExactTime,
    /// One group per fixed UTC bucket `[start, start + width)` and partition
    /// key; the origin is Unix epoch zero and negative timestamps floor
    /// toward negative infinity.
    FixedBucket {
        /// Positive bucket width in exact microseconds.
        #[schemars(range(min = 1))]
        width_micros: u64,
    },
}

/// One declared cross-section output and its output column name. Ordering
/// fields are valid only on the order-statistic primitives; the strict
/// variant shapes reject them everywhere else (SCE-00 D6).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CrossSectionOutputSpec {
    /// One-based rank over the complete group sample, returned as `float64`.
    Rank {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Sort direction of the measured value.
        direction: SortDirection,
        /// Rank tie method.
        tie_method: RankTieMethod,
        /// Null placement.
        null_placement: NullPlacement,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
    /// `(rank - 1) / (ordered_count - 1)` after the selected tie method; one
    /// ordered value is exactly `0.5`.
    Percentile {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Sort direction of the measured value.
        direction: SortDirection,
        /// Rank tie method.
        tie_method: RankTieMethod,
        /// Null placement.
        null_placement: NullPlacement,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
    /// Measured value minus the arithmetic mean of the valid sample.
    Demean {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
    /// `(value - mean) / stddev` with the selected degrees of freedom; null
    /// when the divisor is not positive or the standard deviation is zero.
    Zscore {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_samples: u64,
        /// Degrees-of-freedom adjustment; must be `0` or `1`.
        #[schemars(range(min = 0, max = 1))]
        ddof: u8,
    },
    /// Input value clamped to the Hyndman-Fan type-7 quantiles selected by
    /// `lower` and `upper`; the output preserves the input floating type.
    Winsorize {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input floating column name.
        input: String,
        /// Output column name.
        output: String,
        /// Minimum valid samples for a non-null result.
        #[schemars(range(min = 1))]
        min_samples: u64,
        /// Inclusive lower quantile probability.
        #[schemars(range(min = 0.0, max = 1.0))]
        lower: f64,
        /// Inclusive upper quantile probability.
        #[schemars(range(min = 0.0, max = 1.0))]
        upper: f64,
    },
    /// Boolean mask selecting the largest `count` valid values. Null and NaN
    /// rows produce null; boundary ties are either all included or broken by
    /// canonical row identity.
    Top {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input numeric column name.
        input: String,
        /// Output boolean column name.
        output: String,
        /// Positive number of ordered values to select.
        #[schemars(range(min = 1))]
        count: u64,
        /// Whether every value tied with the boundary is selected.
        include_ties: bool,
        /// Minimum valid samples for a non-null selection result.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
    /// Boolean mask selecting the smallest `count` valid values. Null and NaN
    /// rows produce null; boundary ties are either all included or broken by
    /// canonical row identity.
    Bottom {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input numeric column name.
        input: String,
        /// Output boolean column name.
        output: String,
        /// Positive number of ordered values to select.
        #[schemars(range(min = 1))]
        count: u64,
        /// Whether every value tied with the boundary is selected.
        include_ties: bool,
        /// Minimum valid samples for a non-null selection result.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
    /// Floating input with null rows replaced by the complete group's mean;
    /// valid and NaN rows are preserved.
    MeanFill {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input floating column name.
        input: String,
        /// Output column name.
        output: String,
        /// Minimum valid samples required before nulls are filled.
        #[schemars(range(min = 1))]
        min_samples: u64,
    },
}

impl CrossSectionOutputSpec {
    fn primitive_version(&self) -> u32 {
        match self {
            Self::Rank {
                primitive_version, ..
            }
            | Self::Percentile {
                primitive_version, ..
            }
            | Self::Demean {
                primitive_version, ..
            }
            | Self::Zscore {
                primitive_version, ..
            }
            | Self::Winsorize {
                primitive_version, ..
            }
            | Self::Top {
                primitive_version, ..
            }
            | Self::Bottom {
                primitive_version, ..
            }
            | Self::MeanFill {
                primitive_version, ..
            } => *primitive_version,
        }
    }

    fn input(&self) -> &str {
        match self {
            Self::Rank { input, .. }
            | Self::Percentile { input, .. }
            | Self::Demean { input, .. }
            | Self::Zscore { input, .. }
            | Self::Winsorize { input, .. }
            | Self::Top { input, .. }
            | Self::Bottom { input, .. }
            | Self::MeanFill { input, .. } => input,
        }
    }

    fn output(&self) -> &str {
        match self {
            Self::Rank { output, .. }
            | Self::Percentile { output, .. }
            | Self::Demean { output, .. }
            | Self::Zscore { output, .. }
            | Self::Winsorize { output, .. }
            | Self::Top { output, .. }
            | Self::Bottom { output, .. }
            | Self::MeanFill { output, .. } => output,
        }
    }

    fn min_samples(&self) -> u64 {
        match self {
            Self::Rank { min_samples, .. }
            | Self::Percentile { min_samples, .. }
            | Self::Demean { min_samples, .. }
            | Self::Zscore { min_samples, .. }
            | Self::Winsorize { min_samples, .. }
            | Self::Top { min_samples, .. }
            | Self::Bottom { min_samples, .. }
            | Self::MeanFill { min_samples, .. } => *min_samples,
        }
    }

    fn ddof(&self) -> Option<u8> {
        match self {
            Self::Zscore { ddof, .. } => Some(*ddof),
            _ => None,
        }
    }

    fn count(&self) -> Option<u64> {
        match self {
            Self::Top { count, .. } | Self::Bottom { count, .. } => Some(*count),
            _ => None,
        }
    }
}

/// Data-only declaration of one native cross-section operation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CrossSectionSpec {
    /// Semantic configuration version; must equal
    /// [`CROSS_SECTION_CONFIGURATION_VERSION`].
    pub configuration_version: u32,
    /// Durable state-layout version; must equal
    /// [`CROSS_SECTION_STATE_LAYOUT_VERSION`].
    pub state_layout_version: u32,
    /// Non-null UTC `timestamp[us]` event-time column.
    pub event_time: String,
    /// Ordered non-empty entity key used for row identity.
    pub entity_by: Vec<String>,
    /// Ordered group partition key; empty means one global group per
    /// grouping coordinate.
    #[serde(default)]
    pub partition_by: Vec<String>,
    /// Ordered non-empty sequence key; floating columns are forbidden.
    pub sequence_by: Vec<String>,
    /// Exact-time or fixed-bucket grouping.
    pub grouping: CrossSectionGroupingSpec,
    /// Cross-section outputs in semantic declaration order.
    pub outputs: Vec<CrossSectionOutputSpec>,
    /// Allowed lateness in exact microseconds (SCE-00 D7).
    pub allowed_lateness_micros: u64,
    /// Late-row policy.
    pub late_policy: LatePolicySpec,
    /// Frozen null/NaN value policy.
    pub value_policy: CrossSectionValuePolicy,
}

impl CrossSectionSpec {
    /// Validates the declaration against an exact Arrow input schema and
    /// returns the derived output schema: input fields followed by the
    /// declared outputs in order (SCE-00 D6).
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns.
    pub fn validate(&self, input_schema: &Schema) -> Result<SchemaRef> {
        validate_arguments(self)?;
        let compiled = compile_spec(self, input_schema)?;
        Ok(Arc::new(output_schema(input_schema, &compiled.outputs)))
    }
}

/// Native complete-group cross-section operator.
pub struct CrossSectionOperator {
    name: String,
    spec: CrossSectionSpec,
    input_ports: [Port; 1],
    output_ports: [Port; 1],
    compiled: CompiledCrossSectionSpec,
    state: CrossSectionStreamState,
}

impl CrossSectionOperator {
    /// Compiles one cross-section declaration against an exact Arrow input
    /// schema.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns.
    pub fn new(name: &str, input_schema: SchemaRef, spec: CrossSectionSpec) -> Result<Self> {
        validate_operator_name(name)?;
        validate_arguments(&spec)?;
        let configuration = configuration(&spec)?;
        let compiled = compile_spec_full(&spec, &input_schema, &configuration)?;
        let output_schema = Arc::new(output_schema(&input_schema, &compiled.outputs));
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [Port::with_schema_ref(
                "input",
                BatchKind::Table,
                true,
                Some(input_schema),
            )?],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            state: CrossSectionStreamState::default(),
        })
    }

    /// Returns the validated cross-section declaration.
    pub const fn spec(&self) -> &CrossSectionSpec {
        &self.spec
    }
}

impl std::fmt::Debug for CrossSectionOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("CrossSectionOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for CrossSectionOperator {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_ports(&self) -> &[Port] {
        &self.input_ports
    }

    fn output_ports(&self) -> &[Port] {
        &self.output_ports
    }

    fn configuration(&self) -> JsonMap {
        configuration(&self.spec).expect("validated cross-section configuration stays serializable")
    }
}

impl CrossSectionOperator {
    fn batch_groups(
        &self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<Groups> {
        let input = required_input(inputs, "input", &self.name, None)?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        context.run.check_cancelled()?;
        let rows = read_rows(input.table_payload()?, &self.compiled, &self.name)?;
        self.assemble(rows, &self.name)
    }

    fn grouped_batch(&self, groups: &Groups) -> Result<Batch> {
        let record = build_grouped_record(
            groups,
            &self.compiled,
            self.output_ports[0]
                .schema()
                .expect("cross-section output always has an exact schema"),
            &self.name,
        )?;
        let metadata = BatchMetadata::new(&self.name, 0, BTreeMap::new())?;
        Batch::table(vec![record], metadata)
    }
}

#[async_trait]
impl BatchOperator for CrossSectionOperator {
    /// Evaluates every group as complete at end-of-input without late-row
    /// classification (SCE-00 D7): one canonical-order output over all
    /// groups ordered by finality coordinate then group key.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let groups = self.batch_groups(inputs, context)?;
        let batch = self.grouped_batch(&groups)?;
        Ok(BTreeMap::from([("output".into(), batch)]))
    }
}

/// Live stream state owned by one cross-section operator task. Mutation is
/// confined to this value; input batches stay read-only.
#[derive(Default)]
struct CrossSectionStreamState {
    groups: Groups,
    /// Duplicate evidence for every open identity: one row identity maps to
    /// its owning open group (SCE-00 D11).
    identity_groups: BTreeMap<RowIdentity, GroupKey>,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
    last_checkpoint_epoch: Option<Epoch>,
}

/// Bounded inline manifest contribution of one cross-section checkpoint
/// (SCE-00 D11); retained rows never appear inline, only in segments.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CrossSectionSnapshotMetadata {
    state_layout_version: u32,
    configuration_hash: String,
    state_schema_fingerprint: String,
    epoch: Epoch,
    #[serde(deserialize_with = "deserialize_required_option")]
    pipeline_fingerprint: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    operator_id: Option<String>,
    #[serde(deserialize_with = "deserialize_required_option")]
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    segment_inventory: Vec<SegmentDescriptor>,
}

#[derive(Default)]
struct PreparedLateMetrics {
    late_rows: u64,
    max_lateness_micros: Option<u64>,
}

impl PreparedLateMetrics {
    fn into_delta(self) -> LateMetricDelta {
        LateMetricDelta {
            late_rows: self.late_rows,
            affected_batches: u64::from(self.late_rows > 0),
            max_lateness_micros: self.max_lateness_micros,
            ..LateMetricDelta::default()
        }
    }
}

impl CrossSectionOperator {
    fn validate_stream_data(
        &self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        if ingress != "input" {
            return Err(operator_error(
                context.operator_id(),
                &format!("unknown ingress {ingress:?}; expected \"input\""),
            ));
        }
        self.input_ports[0].validate(batch, &format!("{}.input", self.name))?;
        self.observe_context(context)?;
        if self.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        Ok(())
    }

    fn prepare_stream_data(
        &self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<(AcceptedRows, LateMetricDelta, LateMetricDelta)> {
        self.validate_stream_data(ingress, batch, context)?;
        let watermark = context.input_watermark();
        let rows = read_rows(batch.table_payload()?, &self.compiled, &self.name)?;
        let (accepted, metrics) = self.classify_envelope(rows, watermark, context.operator_id())?;
        let next_metrics = accumulate_late_metrics(self.state.metrics, metrics)?;
        Ok((accepted, metrics, next_metrics))
    }

    fn install_stream_data(
        &mut self,
        accepted: AcceptedRows,
        metrics: LateMetricDelta,
        next_metrics: LateMetricDelta,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        for (group, identity, row) in accepted {
            self.state
                .groups
                .entry(group.clone())
                .or_default()
                .insert(identity.clone(), row);
            self.state.identity_groups.insert(identity, group);
        }
        context.record_window_metrics(metrics.late_rows, metrics.max_lateness_micros, 0)?;
        self.state.metrics = next_metrics;
        self.install_context_identity(context);
        Ok(())
    }
}

#[async_trait]
impl StreamOperator for CrossSectionOperator {
    /// Classifies and buffers one input envelope atomically (SCE-00 D7): the
    /// aggregate input watermark is sampled once, every row of the envelope
    /// is classified against its own group's closing coordinate, and no row
    /// changes state, metrics, or output before the complete envelope is
    /// validated.
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let (accepted, metrics, next_metrics) =
            self.prepare_stream_data(ingress, &batch, context)?;
        self.install_stream_data(accepted, metrics, next_metrics, context)
    }

    /// Emits every newly closed group once in canonical order before the
    /// runtime forwards the watermark (SCE-00 D7 final-only output).
    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        // Cancellation is checked before any state mutation so a cancelled
        // emission leaves the buffered groups available for a retry.
        context.check_cancelled()?;
        self.observe_context(context)?;
        if self
            .state
            .last_input_watermark
            .is_some_and(|previous| watermark <= previous)
        {
            return Err(operator_error(
                context.operator_id(),
                "input watermark did not advance strictly",
            ));
        }
        let closing = self.take_closing_groups(watermark.as_micros(), context.operator_id())?;
        self.emit_groups(closing, context, output).await?;
        self.install_context_identity(context);
        self.state.last_input_watermark = Some(watermark);
        Ok(())
    }

    /// Flushes every open group once in canonical order; no sentinel
    /// watermark is synthesized and the flush releases all group state
    /// (SCE-00 D7).
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        self.observe_context(context)?;
        if self.state.ended {
            return Ok(());
        }
        let groups = std::mem::take(&mut self.state.groups);
        self.state.identity_groups.clear();
        self.emit_groups(groups, context, output).await?;
        self.install_context_identity(context);
        self.state.ended = true;
        Ok(())
    }

    /// Captures the finality frontier and every open group's complete
    /// accepted rows (SCE-00 D11) as one immutable base segment plus bounded
    /// inline metadata.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<crate::OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "cross-section checkpoint epoch did not advance strictly",
            ));
        }
        let encoded = self.encode_state(epoch)?;
        let (descriptor, segments) = match encoded {
            Some(prepared) => {
                // One shared allocation and one digest serve both the
                // snapshot and the manifest descriptor.
                let (segment_id, bytes) = prepared;
                let segment = crate::StateSegment::new(bytes);
                let descriptor = self.snapshot_segment_descriptor(epoch, &segment_id, &segment)?;
                let mut segments = BTreeMap::new();
                segments.insert(segment_id, segment);
                (Some(descriptor), segments)
            }
            None => (None, BTreeMap::new()),
        };
        let inventory = StateInventory::new(descriptor.into_iter().collect())
            .map_err(|error| checkpoint_mismatch(error.to_string()))?;
        let metadata = CrossSectionSnapshotMetadata {
            state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
            configuration_hash: self.compiled.configuration_hash.clone(),
            state_schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            epoch,
            pipeline_fingerprint: self.state.pipeline_fingerprint.clone(),
            operator_id: self.state.operator_id.clone(),
            last_input_watermark: self.state.last_input_watermark,
            next_output_sequence: self.state.next_output_sequence,
            ended: self.state.ended,
            metrics: self.state.metrics,
            segment_inventory: inventory.segments().to_vec(),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| format_error(&error))?
        else {
            return Err(internal_error(
                "cross-section snapshot metadata did not serialize as an object",
            ));
        };
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(crate::OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    /// Replaces the complete live state from one validated snapshot; a
    /// failed restore leaves the current state untouched (SCE-00 D11).
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return StreamOperator::reset(self);
        }
        let metadata = parse_snapshot_metadata(snapshot)?;
        validate_snapshot_metadata(&metadata, &self.compiled, snapshot)?;
        let (groups, identity_groups) = self.decode_state(&metadata, snapshot)?;
        self.state = CrossSectionStreamState {
            groups,
            identity_groups,
            last_input_watermark: metadata.last_input_watermark,
            next_output_sequence: metadata.next_output_sequence,
            ended: metadata.ended,
            metrics: metadata.metrics,
            pipeline_fingerprint: metadata.pipeline_fingerprint,
            operator_id: metadata.operator_id,
            last_checkpoint_epoch: Some(metadata.epoch),
        };
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = CrossSectionStreamState::default();
        Ok(())
    }
}

impl CrossSectionOperator {
    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        if self
            .state
            .pipeline_fingerprint
            .as_deref()
            .is_some_and(|value| value != context.job().fingerprint())
        {
            return Err(operator_error(
                context.operator_id(),
                "cross-section state was used with a different pipeline fingerprint",
            ));
        }
        if self
            .state
            .operator_id
            .as_deref()
            .is_some_and(|value| value != context.operator_id())
        {
            return Err(operator_error(
                context.operator_id(),
                "cross-section state was used with a different operator ID",
            ));
        }
        Ok(())
    }

    fn install_context_identity(&mut self, context: &StreamOperatorContext<'_>) {
        self.state
            .pipeline_fingerprint
            .get_or_insert_with(|| context.job().fingerprint().to_owned());
        self.state
            .operator_id
            .get_or_insert_with(|| context.operator_id().to_owned());
    }

    /// Groups accepted rows for the batch lifecycle and rejects duplicate
    /// identities before any output is produced (SCE-00 D4). The identity is
    /// unique per logical input, so duplicates are rejected across partition
    /// groups, not only within one group.
    fn assemble(&self, rows: Vec<BufferedRow>, node_id: &str) -> Result<Groups> {
        let mut groups = Groups::new();
        let mut identities = BTreeSet::new();
        for row in rows {
            let event_time = row.identity.event_time;
            if !identities.insert(row.identity.clone()) {
                return Err(operator_error(
                    node_id,
                    &format!("duplicate row identity at event_time_micros={event_time}"),
                ));
            }
            let group = self.compiled.group_key(&row, node_id)?;
            groups
                .entry(group)
                .or_default()
                .insert(row.identity.clone(), row);
        }
        Ok(groups)
    }

    /// Classifies one envelope into accepted group rows and the late-metric
    /// delta without touching live state (SCE-00 D7 envelope transaction).
    // Envelope classification keeps late checks, duplicate detection, and
    // grouping in one transactional pass with stable per-check errors.
    // #lizard forgives
    fn classify_envelope(
        &self,
        rows: Vec<BufferedRow>,
        watermark: Option<EventTime>,
        node_id: &str,
    ) -> Result<(AcceptedRows, LateMetricDelta)> {
        let mut accepted: AcceptedRows = Vec::with_capacity(rows.len());
        let mut metrics = PreparedLateMetrics::default();
        // Staged identities are unique per logical input (SCE-00 D4): a
        // duplicate is rejected across partition groups, not only within one.
        let mut staged: BTreeSet<RowIdentity> = BTreeSet::new();
        for (row_index, row) in rows.into_iter().enumerate() {
            let event_time = row.identity.event_time;
            let group = self.compiled.group_key(&row, node_id)?;
            if self.is_late(&group, watermark, row_index, event_time, node_id)? {
                record_late_row(&mut metrics, watermark, event_time, node_id)?;
                continue;
            }
            let duplicate = self.state.identity_groups.contains_key(&row.identity)
                || staged.contains(&row.identity);
            if duplicate {
                return Err(operator_error(
                    node_id,
                    &format!("duplicate row identity at event_time_micros={event_time}"),
                ));
            }
            staged.insert(row.identity.clone());
            accepted.push((group, row.identity.clone(), row));
        }
        Ok((accepted, metrics.into_delta()))
    }

    fn is_late(
        &self,
        group: &GroupKey,
        watermark: Option<EventTime>,
        row_index: usize,
        event_time: i64,
        node_id: &str,
    ) -> Result<bool> {
        let Some(watermark) = watermark else {
            return Ok(false);
        };
        let closing = self.compiled.group_closing(group, node_id)?;
        if closing > watermark.as_micros() {
            return Ok(false);
        }
        match self.spec.late_policy {
            LatePolicySpec::Error { .. } => Err(operator_error(
                node_id,
                &format!(
                    "{node_id}: late_row: envelope rejected at row_index={row_index}; event_time_micros={event_time}, closed_at_watermark_micros={}",
                    watermark.as_micros()
                ),
            )),
            LatePolicySpec::Drop { .. } => Ok(true),
        }
    }

    /// Removes and returns every group whose finality coordinate closes at
    /// or before `watermark` (SCE-00 D7; equality closes).
    fn take_closing_groups(&mut self, watermark: i64, node_id: &str) -> Result<Groups> {
        let mut closing_keys = Vec::new();
        for group in self.state.groups.keys() {
            if self.compiled.group_closing(group, node_id)? <= watermark {
                closing_keys.push(group.clone());
            }
        }
        let mut closing = Groups::new();
        for key in closing_keys {
            let rows = self
                .state
                .groups
                .remove(&key)
                .expect("closing group was present a moment ago");
            for identity in rows.keys() {
                self.state.identity_groups.remove(identity);
            }
            closing.insert(key, rows);
        }
        Ok(closing)
    }

    // Final emission keeps compute, record building, chunking, and sequence
    // accounting in one ordered pass so a partial failure leaves consistent
    // in-memory state.
    // #lizard forgives
    async fn emit_groups(
        &mut self,
        groups: Groups,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if groups.is_empty() {
            return Ok(());
        }
        let record = build_grouped_record(
            &groups,
            &self.compiled,
            self.output_ports[0]
                .schema()
                .expect("cross-section output always has an exact schema"),
            context.operator_id(),
        )?;
        let batches = chunk_output_record(
            &record,
            context.operator_id(),
            self.state.next_output_sequence,
            context.output_budget(),
        )?;
        let chunk_count = u64::try_from(batches.len()).map_err(|_| {
            operator_error(
                context.operator_id(),
                "output chunk count does not fit the sequence range",
            )
        })?;
        for batch in batches {
            output.emit("output", batch).await?;
        }
        self.state.next_output_sequence = self
            .state
            .next_output_sequence
            .checked_add(chunk_count)
            .ok_or_else(|| operator_error(context.operator_id(), "output sequence overflowed"))?;
        Ok(())
    }
}

impl CrossSectionOperator {
    fn encode_state(&self, epoch: Epoch) -> Result<Option<(String, Vec<u8>)>> {
        let row_count: usize = self.state.groups.values().map(BTreeMap::len).sum();
        if row_count == 0 {
            return Ok(None);
        }
        let pipeline_fingerprint = self.state.pipeline_fingerprint.clone().ok_or_else(|| {
            internal_error("cross-section state is missing its pipeline fingerprint")
        })?;
        let operator_id = self.state.operator_id.clone().ok_or_else(|| {
            internal_error("cross-section state is missing its operator identity")
        })?;
        let segment_id = format!("base-{:020}-00000000", epoch.as_u64());
        let bytes = encode_state_segment(
            &self.state.groups,
            self.input_ports[0]
                .schema()
                .expect("cross-section input always has an exact schema"),
            &self.compiled,
            &pipeline_fingerprint,
            &operator_id,
        )?;
        Ok(Some((segment_id, bytes)))
    }

    fn snapshot_segment_descriptor(
        &self,
        epoch: Epoch,
        segment_id: &str,
        segment: &crate::StateSegment,
    ) -> Result<SegmentDescriptor> {
        let operator_id = self.state.operator_id.as_deref().ok_or_else(|| {
            checkpoint_mismatch("cross-section segment is missing its operator identity")
        })?;
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let byte_len = u64::try_from(segment.bytes().len())
            .map_err(|_| internal_error("cross-section segment length does not fit u64"))?;
        Ok(SegmentDescriptor {
            kind: SegmentKind::Base,
            state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
            schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            handle: StateHandle::new(
                operator_id,
                epoch,
                segment_id,
                &relative_path,
                byte_len,
                segment.sha256(),
            )?,
        })
    }

    fn decode_state(
        &self,
        metadata: &CrossSectionSnapshotMetadata,
        snapshot: &crate::OperatorStateSnapshot,
    ) -> Result<(Groups, BTreeMap<RowIdentity, GroupKey>)> {
        let segments = snapshot_segments(snapshot, &metadata.segment_inventory)?;
        let Some(bytes) = segments.into_iter().next() else {
            return Ok((Groups::new(), BTreeMap::new()));
        };
        let input_schema = self.input_ports[0]
            .schema()
            .expect("cross-section input always has an exact schema");
        let groups = decode_state_segment(&bytes, input_schema.as_ref(), &self.compiled)?;
        let mut identity_groups = BTreeMap::new();
        for (group, rows) in &groups {
            for identity in rows.keys() {
                identity_groups.insert(identity.clone(), group.clone());
            }
        }
        Ok((groups, identity_groups))
    }
}

/// Every open group keyed for deterministic iteration; the map order is the
/// group output order (finality coordinate then group key, SCE-00 D4).
type Groups = BTreeMap<GroupKey, BTreeMap<RowIdentity, BufferedRow>>;

/// One envelope's validated group/identity/row triples staged for install.
type AcceptedRows = Vec<(GroupKey, RowIdentity, BufferedRow)>;

/// Group membership key: exact event time or bucket start plus the ordered
/// partition key. Ordering by `base` equals ordering by the finality
/// coordinate because one operator uses one grouping mode and lateness.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct GroupKey {
    base: i64,
    partition: Vec<Option<KeyValue>>,
}

/// Serialized state-row ordering key: group key then row identity.
fn state_fields(input_schema: &Schema) -> Vec<Field> {
    let mut fields = vec![Field::new("_state_kind", DataType::UInt8, false)];
    fields.extend(
        input_schema
            .fields()
            .iter()
            .map(|field| Field::new(field.name(), field.data_type().clone(), true)),
    );
    fields
}

fn state_schema_fingerprint(input_schema: &Schema) -> String {
    let schema = Schema::new(state_fields(input_schema));
    let mut dictionary_tracker = DictionaryTracker::new(true);
    let encoded = IpcSchemaEncoder::new()
        .with_dictionary_tracker(&mut dictionary_tracker)
        .schema_to_fb(&schema);
    hex::encode(Sha256::digest(encoded.finished_data()))
}

fn state_schema(
    input_schema: &Schema,
    compiled: &CompiledCrossSectionSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Schema {
    Schema::new_with_metadata(
        state_fields(input_schema),
        HashMap::from([
            (
                "calc_flow.state_layout_version".into(),
                CROSS_SECTION_STATE_LAYOUT_VERSION.to_string(),
            ),
            (
                "calc_flow.pipeline_fingerprint".into(),
                pipeline_fingerprint.into(),
            ),
            ("calc_flow.operator_id".into(), operator_id.into()),
            (
                "calc_flow.operator_configuration_hash".into(),
                compiled.configuration_hash.clone(),
            ),
            (
                "calc_flow.state_schema_fingerprint".into(),
                compiled.state_schema_fingerprint.clone(),
            ),
        ]),
    )
}

// State serialization writes deterministic group rows column by column with
// checked conversions for every value class.
fn state_column_values(groups: &Groups, width: usize) -> (Vec<u8>, Vec<Vec<ScalarValue>>) {
    let row_count: usize = groups.values().map(BTreeMap::len).sum();
    let mut kinds = Vec::with_capacity(row_count);
    let mut columns = vec![Vec::with_capacity(row_count); width];
    for rows in groups.values() {
        for row in rows.values() {
            kinds.push(0_u8);
            for (index, column) in columns.iter_mut().enumerate() {
                column.push(row.values[index].clone());
            }
        }
    }
    (kinds, columns)
}

fn state_arrays(kinds: Vec<u8>, columns: Vec<Vec<ScalarValue>>) -> Result<Vec<ArrayRef>> {
    let mut arrays: Vec<ArrayRef> = vec![Arc::new(UInt8Array::from(kinds))];
    for column in columns {
        arrays.push(
            ScalarValue::iter_to_array(column.into_iter()).map_err(|error| {
                state_format(format!("cross-section state array failed: {error}"))
            })?,
        );
    }
    Ok(arrays)
}

fn write_state_record(record: &RecordBatch, schema: &Schema) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    let mut writer = FileWriter::try_new(&mut bytes, schema)
        .map_err(|error| state_format(format!("cross-section state IPC header failed: {error}")))?;
    writer
        .write(record)
        .map_err(|error| state_format(format!("cross-section state IPC write failed: {error}")))?;
    writer
        .finish()
        .map_err(|error| state_format(format!("cross-section state IPC finish failed: {error}")))?;
    drop(writer);
    Ok(bytes)
}

fn encode_state_segment(
    groups: &Groups,
    input_schema: &Schema,
    compiled: &CompiledCrossSectionSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    let width = input_schema.fields().len();
    let (kinds, columns) = state_column_values(groups, width);
    let schema = state_schema(input_schema, compiled, pipeline_fingerprint, operator_id);
    let arrays = state_arrays(kinds, columns)?;
    let record = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|error| state_format(format!("cross-section state batch is invalid: {error}")))?;
    write_state_record(&record, &schema)
}

// State decode intentionally validates header metadata, shape, deterministic
// order, and per-row invariants before any state is installed.
// #lizard forgives
fn decode_state_segment(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledCrossSectionSpec,
) -> Result<Groups> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("cross-section state IPC open failed: {error}")))?;
    validate_segment_schema_metadata(reader.schema().metadata(), compiled)?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| state_format(format!("cross-section state IPC read failed: {error}")))?;
    let [record] = batches.try_into().map_err(|_| {
        state_format("cross-section state segment must contain exactly one record batch".to_owned())
    })?;
    let width = input_schema.fields().len();
    if record.num_columns() != width + 1 {
        return Err(state_format(
            "cross-section state segment column count does not match the state schema".to_owned(),
        ));
    }
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| {
            state_format("cross-section state kind column has the wrong type".to_owned())
        })?;
    let mut groups = Groups::new();
    let mut previous: Option<(GroupKey, RowIdentity)> = None;
    for row_index in 0..record.num_rows() {
        let values = (1..record.num_columns())
            .map(|index| {
                ScalarValue::try_from_array(record.column(index), row_index).map_err(|error| {
                    state_format(format!(
                        "cross-section state row could not be read: {error}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if kinds.value(row_index) != 0 {
            return Err(state_format(
                "cross-section state segment contains an unknown row kind".to_owned(),
            ));
        }
        let row = buffered_row_from_values(values, compiled)?;
        let group = compiled.group_key(&row, "cross_section")?;
        let ordering_key = (group.clone(), row.identity.clone());
        if previous
            .as_ref()
            .is_some_and(|prior| prior >= &ordering_key)
        {
            return Err(state_format(
                "cross-section state segment rows are not in deterministic key order".to_owned(),
            ));
        }
        if groups
            .entry(group)
            .or_default()
            .insert(row.identity.clone(), row)
            .is_some()
        {
            return Err(state_format(
                "cross-section state segment contains a duplicate buffered identity".to_owned(),
            ));
        }
        previous = Some(ordering_key);
    }
    Ok(groups)
}

fn validate_segment_schema_metadata(
    metadata: &HashMap<String, String>,
    compiled: &CompiledCrossSectionSpec,
) -> Result<()> {
    let expected = [
        (
            "calc_flow.state_layout_version",
            CROSS_SECTION_STATE_LAYOUT_VERSION.to_string(),
        ),
        (
            "calc_flow.operator_configuration_hash",
            compiled.configuration_hash.clone(),
        ),
        (
            "calc_flow.state_schema_fingerprint",
            compiled.state_schema_fingerprint.clone(),
        ),
    ];
    for (key, value) in expected {
        if metadata.get(key).map(String::as_str) != Some(value.as_str()) {
            return Err(checkpoint_mismatch(format!(
                "cross-section state segment metadata {key} does not match the snapshot"
            )));
        }
    }
    for key in ["calc_flow.pipeline_fingerprint", "calc_flow.operator_id"] {
        if !metadata.contains_key(key) {
            return Err(checkpoint_mismatch(format!(
                "cross-section state segment metadata {key} is missing"
            )));
        }
    }
    Ok(())
}

fn buffered_row_from_values(
    values: Vec<ScalarValue>,
    compiled: &CompiledCrossSectionSpec,
) -> Result<BufferedRow> {
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(state_format(
                "cross-section state row has a null or non-timestamp event time".to_owned(),
            ));
        }
    };
    let entity = compiled
        .entity_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "cross_section"))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], "cross_section"))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

fn parse_snapshot_metadata(
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<CrossSectionSnapshotMetadata> {
    serde_json::from_value::<CrossSectionSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))
}

fn validate_snapshot_metadata(
    metadata: &CrossSectionSnapshotMetadata,
    compiled: &CompiledCrossSectionSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    if metadata.state_layout_version != CROSS_SECTION_STATE_LAYOUT_VERSION {
        return Err(checkpoint_mismatch(format!(
            "cross-section state layout version {} does not match expected {}",
            metadata.state_layout_version, CROSS_SECTION_STATE_LAYOUT_VERSION
        )));
    }
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "cross-section operator configuration hash does not match the compiled operator",
        ));
    }
    if metadata.state_schema_fingerprint != compiled.state_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "cross-section state schema fingerprint does not match the compiled operator",
        ));
    }
    super::checkpoint::validate_inventory(
        &super::checkpoint::SnapshotContract {
            name: "cross-section",
            state_layout_version: metadata.state_layout_version,
            schema_fingerprint: &metadata.state_schema_fingerprint,
            epoch: metadata.epoch,
            pipeline_fingerprint: metadata.pipeline_fingerprint.as_deref(),
            operator_id: metadata.operator_id.as_deref(),
            segment_inventory: metadata.segment_inventory.clone(),
        },
        snapshot,
    )
}

fn snapshot_segments(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &[SegmentDescriptor],
) -> Result<Vec<Arc<Vec<u8>>>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let segment = snapshot.segments.get(segment_id).ok_or_else(|| {
                checkpoint_mismatch(format!(
                    "cross-section snapshot is missing segment {segment_id:?}"
                ))
            })?;
            // A fresh session revalidates every referenced segment byte
            // against the manifest handle before any state is installed.
            let bytes = segment.bytes();
            if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
                return Err(checkpoint_mismatch(
                    "cross-section snapshot segment byte length does not match its handle",
                ));
            }
            if hex::encode(Sha256::digest(bytes)) != descriptor.handle.sha256() {
                return Err(checkpoint_mismatch(
                    "cross-section snapshot segment checksum does not match its handle",
                ));
            }
            Ok(segment.bytes_arc())
        })
        .collect()
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn record_late_row(
    metrics: &mut PreparedLateMetrics,
    watermark: Option<EventTime>,
    event_time: i64,
    node_id: &str,
) -> Result<()> {
    let Some(watermark) = watermark else {
        return Ok(());
    };
    metrics.late_rows = metrics
        .late_rows
        .checked_add(1)
        .ok_or_else(|| operator_error(node_id, "late row counter overflowed"))?;
    let lateness = u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time))
        .map_err(|_| operator_error(node_id, "late row distance overflowed"))?;
    metrics.max_lateness_micros = Some(
        metrics
            .max_lateness_micros
            .map_or(lateness, |maximum| maximum.max(lateness)),
    );
    Ok(())
}

// Budget chunking intentionally checks per-row cost, cumulative budget, and
// sequence range in one pass so oversize output fails before enqueue.
// #lizard forgives
fn chunk_output_record(
    record: &RecordBatch,
    operator_id: &str,
    first_sequence: u64,
    budget: crate::EdgeBudget,
) -> Result<Vec<Batch>> {
    let mut batches = Vec::new();
    let mut start = 0_usize;
    let mut sequence = first_sequence;
    while start < record.num_rows() {
        let mut end = start;
        let mut bytes = 0_usize;
        while end < record.num_rows() && end - start < budget.max_rows {
            let row = record.slice(end, 1);
            let row_batch = Batch::table(vec![row], BatchMetadata::default())
                .map_err(|error| operator_error(operator_id, &error.to_string()))?;
            let row_bytes = row_batch
                .estimated_bytes()
                .map_err(|error| operator_error(operator_id, &error.to_string()))?;
            if row_bytes > budget.max_bytes {
                return Err(CalcFlowError::InvalidArgument {
                    field: "message.bytes".into(),
                    message: format!(
                        "one cross-section output row requires {row_bytes} bytes, exceeding the effective edge byte budget {}",
                        budget.max_bytes
                    ),
                });
            }
            let Some(candidate) = bytes.checked_add(row_bytes) else {
                break;
            };
            if candidate > budget.max_bytes {
                break;
            }
            bytes = candidate;
            end += 1;
        }
        if end == start {
            return Err(operator_error(
                operator_id,
                "validated cross-section output row did not fit the effective edge budget",
            ));
        }
        let metadata = BatchMetadata::new(operator_id, sequence, BTreeMap::new())?;
        batches.push(Batch::table(
            vec![record.slice(start, end - start)],
            metadata,
        )?);
        sequence = sequence.checked_add(1).ok_or_else(|| {
            operator_error(operator_id, "output sequence overflowed before emission")
        })?;
        start = end;
    }
    Ok(batches)
}

/// Reads every input row with its canonical identity; null event-time or
/// sequence values are malformed runtime data (SCE-00 D4/D12).
fn read_rows(
    table: &TableBatch,
    compiled: &CompiledCrossSectionSpec,
    node_id: &str,
) -> Result<Vec<BufferedRow>> {
    let mut rows = Vec::with_capacity(table.batches().iter().map(RecordBatch::num_rows).sum());
    for record in table.batches() {
        for row_index in 0..record.num_rows() {
            rows.push(read_row(record, row_index, compiled, node_id)?);
        }
    }
    Ok(rows)
}

fn read_row(
    record: &RecordBatch,
    row_index: usize,
    compiled: &CompiledCrossSectionSpec,
    node_id: &str,
) -> Result<BufferedRow> {
    let mut values = Vec::with_capacity(record.num_columns());
    for column in record.columns() {
        values.push(
            ScalarValue::try_from_array(column, row_index).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("cross-section input row could not be read: {error}"),
                )
            })?,
        );
    }
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(operator_error(
                node_id,
                "cross-section event-time value is null or not a microsecond timestamp",
            ));
        }
    };
    let entity = compiled
        .entity_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], node_id))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

/// Builds one output record over every group in canonical order: group order
/// by finality coordinate then group key, rows within one group by
/// `(event_time, entity..., sequence...)` (SCE-00 D4/D6).
// Record building keeps input-column reconstruction and derived-column
// alignment in one ordered pass.
// #lizard forgives
fn build_grouped_record(
    groups: &Groups,
    compiled: &CompiledCrossSectionSpec,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let input_width = compiled.input_width;
    let mut columns: Vec<Vec<ScalarValue>> = vec![Vec::new(); input_width];
    let mut derived: Vec<Vec<ScalarValue>> = vec![Vec::new(); compiled.outputs.len()];
    for rows in groups.values() {
        let group_outputs = compute_group(rows, compiled);
        for row in rows.values() {
            for (index, column) in columns.iter_mut().enumerate() {
                column.push(row.values[index].clone());
            }
        }
        for (ordinal, values) in group_outputs.into_iter().enumerate() {
            derived[ordinal].extend(values);
        }
    }
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(input_width + derived.len());
    for (index, column) in columns.into_iter().enumerate() {
        if column.is_empty() {
            arrays.push(new_null_array(output_schema.field(index).data_type(), 0));
        } else {
            arrays.push(ScalarValue::iter_to_array(column).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("cross-section output row encoding failed: {error}"),
                )
            })?);
        }
    }
    for (ordinal, column) in derived.into_iter().enumerate() {
        if column.is_empty() {
            arrays.push(new_null_array(
                output_schema.field(input_width + ordinal).data_type(),
                0,
            ));
        } else {
            arrays.push(ScalarValue::iter_to_array(column).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("cross-section derived column encoding failed: {error}"),
                )
            })?);
        }
    }
    RecordBatch::try_new(Arc::clone(output_schema), arrays).map_err(|error| {
        operator_error(
            node_id,
            &format!("cross-section output record is invalid: {error}"),
        )
    })
}

#[derive(Clone)]
struct CompiledCrossSectionSpec {
    input_width: usize,
    event_time_index: usize,
    entity_columns: Vec<CompiledKeyColumn>,
    partition_columns: Vec<CompiledKeyColumn>,
    sequence_columns: Vec<CompiledKeyColumn>,
    grouping: CrossSectionGroupingSpec,
    allowed_lateness_micros: u64,
    outputs: Vec<CompiledCrossSectionOutput>,
    configuration_hash: String,
    state_schema_fingerprint: String,
}

#[derive(Clone, Copy)]
struct CompiledKeyColumn {
    index: usize,
}

#[derive(Clone)]
struct CompiledCrossSectionOutput {
    input_index: usize,
    name: String,
    data_type: DataType,
    evaluation: CompiledEvaluation,
}

#[derive(Clone)]
enum CompiledEvaluation {
    Rank {
        direction: SortDirection,
        tie_method: RankTieMethod,
        null_placement: NullPlacement,
        min_samples: u64,
    },
    Percentile {
        direction: SortDirection,
        tie_method: RankTieMethod,
        null_placement: NullPlacement,
        min_samples: u64,
    },
    Moment {
        min_samples: u64,
        ddof: u8,
    },
    ZScore {
        min_samples: u64,
        ddof: u8,
    },
    Winsorize {
        min_samples: u64,
        lower: f64,
        upper: f64,
    },
    TopSelection {
        count: u64,
        include_ties: bool,
        min_samples: u64,
    },
    BottomSelection {
        count: u64,
        include_ties: bool,
        min_samples: u64,
    },
    MeanFill {
        min_samples: u64,
    },
}

impl CompiledCrossSectionSpec {
    /// Computes the group membership key of one row: the exact event time or
    /// the containing bucket start plus the partition key (SCE-00 D6).
    fn group_key(&self, row: &BufferedRow, node_id: &str) -> Result<GroupKey> {
        let event_time = row.identity.event_time;
        let base = match self.grouping {
            CrossSectionGroupingSpec::ExactTime => event_time,
            CrossSectionGroupingSpec::FixedBucket { width_micros } => {
                let width = bucket_width(width_micros, node_id)?;
                event_time
                    .div_euclid(width)
                    .checked_mul(width)
                    .ok_or_else(|| {
                        operator_error(node_id, "bucket start overflowed the event-time range")
                    })?
            }
        };
        let partition = self
            .partition_columns
            .iter()
            .map(|column| KeyValue::from_nullable_scalar(&row.values[column.index], node_id))
            .collect::<Result<Vec<_>>>()?;
        Ok(GroupKey { base, partition })
    }

    /// Finality coordinate of one group: `t + L` for exact groups and
    /// `bucket_end + L` for bucketed groups; equality closes (SCE-00 D7).
    fn group_closing(&self, group: &GroupKey, node_id: &str) -> Result<i64> {
        let end = match self.grouping {
            CrossSectionGroupingSpec::ExactTime => group.base,
            CrossSectionGroupingSpec::FixedBucket { width_micros } => group
                .base
                .checked_add(bucket_width(width_micros, node_id)?)
                .ok_or_else(|| {
                    operator_error(node_id, "bucket end overflowed the event-time range")
                })?,
        };
        let lateness = i64::try_from(self.allowed_lateness_micros).map_err(|_| {
            operator_error(
                node_id,
                "allowed lateness exceeds the representable event-time range",
            )
        })?;
        end.checked_add(lateness).ok_or_else(|| {
            operator_error(
                node_id,
                "finality coordinate overflowed the event-time range",
            )
        })
    }
}

fn bucket_width(width_micros: u64, node_id: &str) -> Result<i64> {
    let width = i64::try_from(width_micros).map_err(|_| {
        operator_error(
            node_id,
            "bucket width exceeds the representable event-time range",
        )
    })?;
    if width <= 0 {
        return Err(operator_error(
            node_id,
            "bucket width must be a positive number of microseconds",
        ));
    }
    Ok(width)
}

/// One entity, partition, or sequence key component in the Arrow total
/// order (null before non-null); floats compare with the IEEE total order
/// (SCE-00 D4).
#[derive(Clone, Debug)]
enum KeyValue {
    Boolean(bool),
    Signed(i64),
    Unsigned(u64),
    Float32(f32),
    Float64(f64),
    String(String),
    Date32(i32),
    Date64(i64),
    Timestamp(i64),
}

impl KeyValue {
    fn from_nullable_scalar(scalar: &ScalarValue, node_id: &str) -> Result<Option<Self>> {
        if scalar.is_null() {
            return Ok(None);
        }
        Self::from_required_scalar(scalar, node_id).map(Some)
    }

    fn from_required_scalar(scalar: &ScalarValue, node_id: &str) -> Result<Self> {
        let value = match scalar {
            ScalarValue::Boolean(value) => value.map(Self::Boolean),
            ScalarValue::Int8(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int16(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int32(value) => value.map(|value| Self::Signed(i64::from(value))),
            ScalarValue::Int64(value) => value.map(Self::Signed),
            ScalarValue::UInt8(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt16(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt32(value) => value.map(|value| Self::Unsigned(u64::from(value))),
            ScalarValue::UInt64(value) => value.map(Self::Unsigned),
            ScalarValue::Float32(value) => value.map(Self::Float32),
            ScalarValue::Float64(value) => value.map(Self::Float64),
            ScalarValue::Utf8(value) | ScalarValue::LargeUtf8(value) => {
                value.clone().map(Self::String)
            }
            ScalarValue::Date32(value) => value.map(Self::Date32),
            ScalarValue::Date64(value) => value.map(Self::Date64),
            ScalarValue::TimestampMicrosecond(value, _) => value.map(Self::Timestamp),
            other => {
                return Err(operator_error(
                    node_id,
                    &format!(
                        "cross-section key column has unsupported value type {}",
                        other.data_type()
                    ),
                ));
            }
        };
        value.ok_or_else(|| operator_error(node_id, "cross-section sequence key value is null"))
    }
}

impl PartialEq for KeyValue {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for KeyValue {}

impl PartialOrd for KeyValue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeyValue {
    fn cmp(&self, other: &Self) -> Ordering {
        fn rank(value: &KeyValue) -> u8 {
            match value {
                KeyValue::Boolean(_) => 0,
                KeyValue::Signed(_) => 1,
                KeyValue::Unsigned(_) => 2,
                KeyValue::Float32(_) => 3,
                KeyValue::Float64(_) => 4,
                KeyValue::String(_) => 5,
                KeyValue::Date32(_) => 6,
                KeyValue::Date64(_) => 7,
                KeyValue::Timestamp(_) => 8,
            }
        }
        match (self, other) {
            (Self::Boolean(left), Self::Boolean(right)) => left.cmp(right),
            (Self::Unsigned(left), Self::Unsigned(right)) => left.cmp(right),
            (Self::Float32(left), Self::Float32(right)) => left.total_cmp(right),
            (Self::Float64(left), Self::Float64(right)) => left.total_cmp(right),
            (Self::String(left), Self::String(right)) => left.cmp(right),
            (Self::Date32(left), Self::Date32(right)) => i64::from(*left).cmp(&i64::from(*right)),
            (Self::Signed(left), Self::Signed(right))
            | (Self::Date64(left), Self::Date64(right))
            | (Self::Timestamp(left), Self::Timestamp(right)) => left.cmp(right),
            _ => rank(self).cmp(&rank(other)),
        }
    }
}

/// Canonical row identity `(event_time, entity_key..., sequence_key...)`
/// (SCE-00 D4).
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct RowIdentity {
    event_time: i64,
    entity: Vec<Option<KeyValue>>,
    sequence: Vec<KeyValue>,
}

/// One accepted input row retained for final-order emission.
#[derive(Clone, Debug)]
struct BufferedRow {
    identity: RowIdentity,
    values: Vec<ScalarValue>,
}

impl BufferedRow {
    fn new(
        entity: Vec<Option<KeyValue>>,
        sequence: Vec<KeyValue>,
        event_time: i64,
        values: Vec<ScalarValue>,
    ) -> Self {
        Self {
            identity: RowIdentity {
                event_time,
                entity,
                sequence,
            },
            values,
        }
    }
}

/// One measured value's classification for the transform kernels (SCE-00
/// D3.2/D6): null and NaN are excluded from samples but stay observable at
/// their own rows.
#[derive(Clone, Copy, Debug)]
enum Sample {
    Valid(f64),
    Null,
    Nan,
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen cross-section sample type is float64"
)]
fn classify_sample(value: &ScalarValue) -> Sample {
    if value.is_null() {
        return Sample::Null;
    }
    let sample = match value {
        ScalarValue::Float32(Some(sample)) => f64::from(*sample),
        ScalarValue::Float64(Some(sample)) => *sample,
        ScalarValue::Int8(Some(sample)) => f64::from(*sample),
        ScalarValue::Int16(Some(sample)) => f64::from(*sample),
        ScalarValue::Int32(Some(sample)) => f64::from(*sample),
        ScalarValue::Int64(Some(sample)) => *sample as f64,
        ScalarValue::UInt8(Some(sample)) => f64::from(*sample),
        ScalarValue::UInt16(Some(sample)) => f64::from(*sample),
        ScalarValue::UInt32(Some(sample)) => f64::from(*sample),
        ScalarValue::UInt64(Some(sample)) => *sample as f64,
        _ => return Sample::Null,
    };
    if sample.is_nan() {
        Sample::Nan
    } else {
        Sample::Valid(sample)
    }
}

/// One measured value's exact order-statistic classification. Valid numeric
/// values keep their lossless scalar order key; float64 conversion belongs
/// only to the separate arithmetic statistic path.
#[derive(Clone, Debug)]
enum OrderSample {
    Valid(KeyValue),
    Null,
    Nan,
}

fn classify_order_sample(value: &ScalarValue) -> OrderSample {
    if value.is_null() {
        return OrderSample::Null;
    }
    let is_nan = match value {
        ScalarValue::Float32(Some(sample)) => sample.is_nan(),
        ScalarValue::Float64(Some(sample)) => sample.is_nan(),
        _ => false,
    };
    if is_nan {
        return OrderSample::Nan;
    }
    KeyValue::from_required_scalar(value, "cross_section order statistic")
        .map_or(OrderSample::Null, OrderSample::Valid)
}

/// Computes every declared output for one complete group and returns one
/// aligned column per output in declaration order (SCE-00 D6). Rows arrive
/// and return in the group's canonical order; the measured-value sort never
/// reorders output rows. Compatible outputs over one input column reuse the
/// same classified sample and ascending sort pass.
fn compute_group(
    rows: &BTreeMap<RowIdentity, BufferedRow>,
    compiled: &CompiledCrossSectionSpec,
) -> Vec<Vec<ScalarValue>> {
    let mut columns = Vec::with_capacity(compiled.outputs.len());
    let mut prepared = BTreeMap::new();
    for output in &compiled.outputs {
        let prepared = prepared
            .entry(output.input_index)
            .or_insert_with(|| PreparedCrossSectionColumn::new(rows, output.input_index));
        let column = match &output.evaluation {
            CompiledEvaluation::Rank {
                direction,
                tie_method,
                null_placement,
                min_samples,
            }
            | CompiledEvaluation::Percentile {
                direction,
                tie_method,
                null_placement,
                min_samples,
            } => {
                let percentile = matches!(output.evaluation, CompiledEvaluation::Percentile { .. });
                float64_scalars(order_statistic_column_prepared(
                    prepared,
                    *direction,
                    *tie_method,
                    *null_placement,
                    *min_samples,
                    percentile,
                ))
            }
            CompiledEvaluation::Moment { min_samples, ddof }
            | CompiledEvaluation::ZScore { min_samples, ddof } => {
                let zscore = matches!(output.evaluation, CompiledEvaluation::ZScore { .. });
                float64_scalars(statistic_column_with_accumulator(
                    &prepared.samples,
                    prepared.accumulator,
                    *min_samples,
                    *ddof,
                    zscore,
                ))
            }
            CompiledEvaluation::Winsorize {
                min_samples,
                lower,
                upper,
            } => winsorize_column(prepared, *min_samples, *lower, *upper),
            CompiledEvaluation::TopSelection {
                count,
                include_ties,
                min_samples,
            }
            | CompiledEvaluation::BottomSelection {
                count,
                include_ties,
                min_samples,
            } => {
                let top = matches!(output.evaluation, CompiledEvaluation::TopSelection { .. });
                selection_column(prepared, *count, *include_ties, *min_samples, top)
            }
            CompiledEvaluation::MeanFill { min_samples } => {
                mean_fill_column(prepared, *min_samples)
            }
        };
        columns.push(column);
    }
    columns
}

struct PreparedCrossSectionColumn {
    values: Vec<ScalarValue>,
    samples: Vec<Sample>,
    order_samples: Vec<OrderSample>,
    ascending_value_indices: Vec<usize>,
    accumulator: StatisticAccumulator,
}

impl PreparedCrossSectionColumn {
    fn new(rows: &BTreeMap<RowIdentity, BufferedRow>, input_index: usize) -> Self {
        let values = rows
            .values()
            .map(|row| row.values[input_index].clone())
            .collect::<Vec<_>>();
        let samples = values.iter().map(classify_sample).collect::<Vec<_>>();
        let order_samples = values.iter().map(classify_order_sample).collect::<Vec<_>>();
        let mut ascending_value_indices = order_samples
            .iter()
            .enumerate()
            .filter_map(|(index, sample)| matches!(sample, OrderSample::Valid(_)).then_some(index))
            .collect::<Vec<_>>();
        ascending_value_indices
            .sort_by(|left, right| compare_order_sample_indices(&order_samples, *left, *right));
        let accumulator = accumulate_statistics(&samples);
        Self {
            values,
            samples,
            order_samples,
            ascending_value_indices,
            accumulator,
        }
    }
}

fn ordered_indices(
    prepared: &PreparedCrossSectionColumn,
    direction: SortDirection,
    null_placement: NullPlacement,
) -> Vec<usize> {
    let mut values = prepared.ascending_value_indices.clone();
    if direction == SortDirection::Descending {
        values.reverse();
    }
    let nulls = prepared
        .order_samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| matches!(sample, OrderSample::Null).then_some(index))
        .collect::<Vec<_>>();
    match null_placement {
        NullPlacement::Exclude => values,
        NullPlacement::First => {
            let mut nulls = nulls;
            nulls.extend(values);
            nulls
        }
        NullPlacement::Last => {
            values.extend(nulls);
            values
        }
    }
}

fn same_order_class(samples: &[OrderSample], left: usize, right: usize) -> bool {
    match (&samples[left], &samples[right]) {
        (OrderSample::Null, OrderSample::Null) => true,
        (OrderSample::Valid(left), OrderSample::Valid(right)) => left == right,
        _ => false,
    }
}

fn compare_order_sample_indices(
    samples: &[OrderSample],
    left_index: usize,
    right_index: usize,
) -> Ordering {
    let value_order = match (&samples[left_index], &samples[right_index]) {
        (OrderSample::Valid(left), OrderSample::Valid(right)) => left.cmp(right),
        _ => Ordering::Equal,
    };
    value_order.then_with(|| left_index.cmp(&right_index))
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen rank/percentile output type is float64"
)]
fn rank_order_indices(
    samples: &[OrderSample],
    indices: &[usize],
    sample_count: usize,
    tie_method: RankTieMethod,
) -> Vec<Option<f64>> {
    let mut ranks = vec![None; sample_count];
    let mut position = 0_usize;
    while position < indices.len() {
        let mut end = position;
        while end < indices.len() && same_order_class(samples, indices[end], indices[position]) {
            end += 1;
        }
        let first_position = f64::from(u32::try_from(position + 1).unwrap_or(u32::MAX));
        let last_position = f64::from(u32::try_from(end).unwrap_or(u32::MAX));
        let rank = match tie_method {
            RankTieMethod::Average => first_position.midpoint(last_position),
            RankTieMethod::Min => first_position,
            RankTieMethod::Max => last_position,
        };
        for index in &indices[position..end] {
            ranks[*index] = Some(rank);
        }
        position = end;
    }
    ranks
}

fn render_order_statistics(
    samples: &[OrderSample],
    ranks: &[Option<f64>],
    null_placement: NullPlacement,
    min_samples: u64,
    ordered_count: usize,
    percentile: bool,
) -> Vec<Option<f64>> {
    let valid_count = samples
        .iter()
        .filter(|sample| matches!(sample, OrderSample::Valid(_)))
        .count();
    samples
        .iter()
        .enumerate()
        .map(|(index, sample)| match sample {
            OrderSample::Nan => Some(f64::NAN),
            OrderSample::Null if null_placement == NullPlacement::Exclude => None,
            _ => apply_statistic(
                ranks[index],
                valid_count,
                min_samples,
                ordered_count,
                percentile,
            ),
        })
        .collect()
}

/// Computes one rank or percentile column: the ordering places valid values
/// by direction and included nulls as one tied class at the requested end of
/// the final sort order; excluded nulls and unmet samples produce null and
/// NaN rows produce NaN (SCE-00 D6).
#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen rank/percentile output type is float64"
)]
#[cfg(test)]
fn order_statistic_column(
    samples: &[OrderSample],
    direction: SortDirection,
    tie_method: RankTieMethod,
    null_placement: NullPlacement,
    min_samples: u64,
    percentile: bool,
) -> Vec<Option<f64>> {
    let mut ascending_value_indices = samples
        .iter()
        .enumerate()
        .filter_map(|(index, sample)| matches!(sample, OrderSample::Valid(_)).then_some(index))
        .collect::<Vec<_>>();
    ascending_value_indices
        .sort_by(|left, right| compare_order_sample_indices(samples, *left, *right));
    let prepared = PreparedCrossSectionColumn {
        values: Vec::new(),
        samples: Vec::new(),
        order_samples: samples.to_vec(),
        ascending_value_indices,
        accumulator: StatisticAccumulator::default(),
    };
    order_statistic_column_prepared(
        &prepared,
        direction,
        tie_method,
        null_placement,
        min_samples,
        percentile,
    )
}

fn order_statistic_column_prepared(
    prepared: &PreparedCrossSectionColumn,
    direction: SortDirection,
    tie_method: RankTieMethod,
    null_placement: NullPlacement,
    min_samples: u64,
    percentile: bool,
) -> Vec<Option<f64>> {
    let indices = ordered_indices(prepared, direction, null_placement);
    let ordered_count = indices.len();
    let ranks = rank_order_indices(
        &prepared.order_samples,
        &indices,
        prepared.order_samples.len(),
        tie_method,
    );
    render_order_statistics(
        &prepared.order_samples,
        &ranks,
        null_placement,
        min_samples,
        ordered_count,
        percentile,
    )
}

fn float64_scalars(values: Vec<Option<f64>>) -> Vec<ScalarValue> {
    values.into_iter().map(ScalarValue::Float64).collect()
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "the validated probability bounds the integral quantile indices to this slice"
)]
fn type_seven_quantile(sorted: &[f64], probability: f64) -> f64 {
    let scaled = probability * (sorted.len() - 1) as f64;
    let lower_index = scaled.floor() as usize;
    let upper_index = scaled.ceil() as usize;
    let lower = sorted[lower_index];
    let upper = sorted[upper_index];
    if lower_index == upper_index || lower.to_bits() == upper.to_bits() {
        return lower;
    }
    let fraction = scaled - lower_index as f64;
    match (lower, upper) {
        (value, _) if value == f64::NEG_INFINITY && upper == f64::INFINITY => f64::NAN,
        (value, _) if value == f64::NEG_INFINITY => f64::NEG_INFINITY,
        (_, value) if value == f64::INFINITY => f64::INFINITY,
        _ => lower + fraction * (upper - lower),
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "float32 cross-section inputs intentionally preserve their physical type"
)]
fn float_scalar_like(value: &ScalarValue, replacement: Option<f64>) -> ScalarValue {
    match value {
        ScalarValue::Float32(_) => ScalarValue::Float32(replacement.map(|value| value as f32)),
        ScalarValue::Float64(_) => ScalarValue::Float64(replacement),
        _ => unreachable!("compiled winsorize inputs are floating"),
    }
}

fn winsorize_column(
    prepared: &PreparedCrossSectionColumn,
    min_samples: u64,
    lower_probability: f64,
    upper_probability: f64,
) -> Vec<ScalarValue> {
    let valid_count = prepared.ascending_value_indices.len() as u64;
    if valid_count < min_samples {
        return unmet_winsorize_column(prepared);
    }
    let sorted = prepared
        .ascending_value_indices
        .iter()
        .filter_map(|index| match prepared.samples[*index] {
            Sample::Valid(value) => Some(value),
            Sample::Null | Sample::Nan => None,
        })
        .collect::<Vec<_>>();
    let lower = type_seven_quantile(&sorted, lower_probability);
    let upper = type_seven_quantile(&sorted, upper_probability);
    prepared
        .values
        .iter()
        .zip(&prepared.samples)
        .map(|(original, sample)| winsorized_scalar(original, sample, lower, upper))
        .collect()
}

fn unmet_winsorize_column(prepared: &PreparedCrossSectionColumn) -> Vec<ScalarValue> {
    prepared
        .values
        .iter()
        .zip(&prepared.samples)
        .map(|(value, sample)| match sample {
            Sample::Nan => value.clone(),
            Sample::Null | Sample::Valid(_) => float_scalar_like(value, None),
        })
        .collect()
}

fn winsorized_scalar(
    original: &ScalarValue,
    sample: &Sample,
    lower: f64,
    upper: f64,
) -> ScalarValue {
    match sample {
        Sample::Null | Sample::Nan => original.clone(),
        Sample::Valid(value) => {
            float_scalar_like(original, Some(winsorized_value(*value, lower, upper)))
        }
    }
}

fn winsorized_value(value: f64, lower: f64, upper: f64) -> f64 {
    if lower.is_nan() || upper.is_nan() {
        f64::NAN
    } else if value < lower {
        lower
    } else if value > upper {
        upper
    } else {
        value
    }
}

fn descending_value_indices(prepared: &PreparedCrossSectionColumn) -> Vec<usize> {
    let ascending = &prepared.ascending_value_indices;
    let mut descending = Vec::with_capacity(ascending.len());
    let mut end = ascending.len();
    while end > 0 {
        let mut start = end - 1;
        while start > 0
            && same_order_class(
                &prepared.order_samples,
                ascending[start - 1],
                ascending[end - 1],
            )
        {
            start -= 1;
        }
        descending.extend_from_slice(&ascending[start..end]);
        end = start;
    }
    descending
}

fn selection_column(
    prepared: &PreparedCrossSectionColumn,
    count: u64,
    include_ties: bool,
    min_samples: u64,
    top: bool,
) -> Vec<ScalarValue> {
    let valid_count = prepared.ascending_value_indices.len() as u64;
    if valid_count < min_samples {
        return vec![ScalarValue::Boolean(None); prepared.samples.len()];
    }
    let ordered = if top {
        descending_value_indices(prepared)
    } else {
        prepared.ascending_value_indices.clone()
    };
    let selected_count = usize::try_from(count)
        .unwrap_or(usize::MAX)
        .min(ordered.len());
    let mut selected = ordered[..selected_count]
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if include_ties && selected_count > 0 && selected_count < ordered.len() {
        let boundary = ordered[selected_count - 1];
        selected.extend(
            ordered[selected_count..]
                .iter()
                .copied()
                .take_while(|index| same_order_class(&prepared.order_samples, boundary, *index)),
        );
    }
    prepared
        .samples
        .iter()
        .enumerate()
        .map(|(index, sample)| match sample {
            Sample::Valid(_) => ScalarValue::Boolean(Some(selected.contains(&index))),
            Sample::Null | Sample::Nan => ScalarValue::Boolean(None),
        })
        .collect()
}

fn mean_fill_column(prepared: &PreparedCrossSectionColumn, min_samples: u64) -> Vec<ScalarValue> {
    let fill =
        (prepared.accumulator.count >= min_samples).then(|| prepared.accumulator.classified_mean());
    prepared
        .values
        .iter()
        .zip(&prepared.samples)
        .map(|(original, sample)| match sample {
            Sample::Null => fill.map_or_else(
                || original.clone(),
                |mean| float_scalar_like(original, Some(mean)),
            ),
            Sample::Nan | Sample::Valid(_) => original.clone(),
        })
        .collect()
}

/// Applies the min-samples gate and the percentile transform to one rank
/// (SCE-00 D3.2/D6): an unmet sample count nulls the whole statistic, and a
/// single ordered value is exactly one half.
#[allow(clippy::cast_precision_loss, reason = "percentiles are float64")]
fn apply_statistic(
    rank: Option<f64>,
    valid_count: usize,
    min_samples: u64,
    ordered_count: usize,
    percentile: bool,
) -> Option<f64> {
    let rank = rank?;
    if (valid_count as u64) < min_samples {
        return None;
    }
    if !percentile {
        return Some(rank);
    }
    if ordered_count == 1 {
        return Some(0.5);
    }
    Some((rank - 1.0) / (ordered_count as f64 - 1.0))
}

/// One pass of West-style mean/M2 accumulation over the valid sample in
/// canonical row order; this is the deterministic shared algorithm of the
/// batch and stream lifecycles.
#[derive(Clone, Copy, Debug, Default)]
struct StatisticAccumulator {
    count: u64,
    mean: f64,
    m2: f64,
    pos_inf: u64,
    neg_inf: u64,
}

impl StatisticAccumulator {
    #[allow(
        clippy::cast_precision_loss,
        reason = "the frozen statistic output type is float64"
    )]
    fn add(&mut self, value: f64) {
        self.count += 1;
        if value.is_infinite() {
            if value > 0.0 {
                self.pos_inf += 1;
            } else {
                self.neg_inf += 1;
            }
        }
        let count = self.count as f64;
        let delta = value - self.mean;
        self.mean += delta / count;
        self.m2 += delta * (value - self.mean);
    }

    /// Classifies the sample mean from the infinity counts: both signs is
    /// the undefined `inf - inf` (NaN), one sign is that infinity, and no
    /// infinity keeps the West readout (SCE-00 D3.2).
    fn classified_mean(&self) -> f64 {
        match (self.pos_inf > 0, self.neg_inf > 0) {
            (true, true) => f64::NAN,
            (true, false) => f64::INFINITY,
            (false, true) => f64::NEG_INFINITY,
            (false, false) => self.mean,
        }
    }

    /// Classifies the sum of squared deviations: any infinity makes every
    /// deviation an `inf - inf` form, so the variance is NaN, never a
    /// silent zero (SCE-00 D3.2).
    fn classified_m2(&self) -> f64 {
        if self.pos_inf > 0 || self.neg_inf > 0 {
            f64::NAN
        } else {
            self.m2
        }
    }
}

fn accumulate_statistics(samples: &[Sample]) -> StatisticAccumulator {
    let mut accumulator = StatisticAccumulator::default();
    for sample in samples {
        if let Sample::Valid(value) = sample {
            accumulator.add(*value);
        }
    }
    accumulator
}

#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen z-score output type is float64"
)]
fn zscore_value(value: f64, mean: f64, m2: f64, divisor: u64) -> Option<f64> {
    if divisor == 0 {
        return None;
    }
    let variance = m2 / divisor as f64;
    if variance.is_nan() {
        return Some(f64::NAN);
    }
    let stddev = variance.sqrt();
    if stddev == 0.0 {
        return None;
    }
    Some((value - mean) / stddev)
}

fn statistic_value(
    sample: Sample,
    accumulator: StatisticAccumulator,
    min_samples: u64,
    ddof: u8,
    zscore: bool,
) -> Option<f64> {
    match sample {
        Sample::Null => None,
        Sample::Nan => Some(f64::NAN),
        Sample::Valid(value) => {
            if accumulator.count < min_samples {
                return None;
            }
            let mean = accumulator.classified_mean();
            if !zscore {
                return Some(value - mean);
            }
            let divisor = accumulator.count.saturating_sub(u64::from(ddof));
            zscore_value(value, mean, accumulator.classified_m2(), divisor)
        }
    }
}

/// Computes one demean or z-score column over the valid sample; null rows
/// preserve null, NaN rows preserve NaN, an unmet sample count nulls the
/// statistic, and a non-positive divisor or zero standard deviation nulls
/// the z-score (SCE-00 D6).
#[allow(
    clippy::cast_precision_loss,
    reason = "the frozen z-score output type is float64"
)]
#[cfg(test)]
fn statistic_column(
    samples: &[Sample],
    min_samples: u64,
    ddof: u8,
    zscore: bool,
) -> Vec<Option<f64>> {
    let accumulator = accumulate_statistics(samples);
    statistic_column_with_accumulator(samples, accumulator, min_samples, ddof, zscore)
}

fn statistic_column_with_accumulator(
    samples: &[Sample],
    accumulator: StatisticAccumulator,
    min_samples: u64,
    ddof: u8,
    zscore: bool,
) -> Vec<Option<f64>> {
    samples
        .iter()
        .map(|sample| statistic_value(*sample, accumulator, min_samples, ddof, zscore))
        .collect()
}

fn validate_versions(spec: &CrossSectionSpec) -> Result<()> {
    if spec.configuration_version != CROSS_SECTION_CONFIGURATION_VERSION {
        return Err(invalid_argument(
            "cross_section.configuration_version",
            "unsupported cross-section configuration version",
        ));
    }
    if spec.state_layout_version != CROSS_SECTION_STATE_LAYOUT_VERSION {
        return Err(invalid_argument(
            "cross_section.state_layout_version",
            "unsupported cross-section state layout version",
        ));
    }
    Ok(())
}

fn validate_grouping(grouping: CrossSectionGroupingSpec) -> Result<()> {
    if let CrossSectionGroupingSpec::FixedBucket { width_micros } = grouping
        && width_micros == 0
    {
        return Err(invalid_argument(
            "cross_section.grouping.width_micros",
            "must be greater than zero",
        ));
    }
    Ok(())
}

fn validate_key_declarations(spec: &CrossSectionSpec) -> Result<()> {
    if spec.entity_by.is_empty() {
        return Err(invalid_argument(
            "cross_section.entity_by",
            "must not be empty",
        ));
    }
    if spec.sequence_by.is_empty() {
        return Err(invalid_argument(
            "cross_section.sequence_by",
            "must not be empty",
        ));
    }
    validate_key_names("cross_section.entity_by", &spec.entity_by)?;
    validate_key_names("cross_section.partition_by", &spec.partition_by)?;
    validate_key_names("cross_section.sequence_by", &spec.sequence_by)
}

fn validate_late_policy(late_policy: LatePolicySpec) -> Result<()> {
    if let LatePolicySpec::Drop { metrics_version } = late_policy
        && metrics_version != 1
    {
        return Err(invalid_argument(
            "cross_section.late_policy.metrics_version",
            "unsupported late-metrics version",
        ));
    }
    Ok(())
}

fn validate_arguments(spec: &CrossSectionSpec) -> Result<()> {
    validate_versions(spec)?;
    validate_grouping(spec.grouping)?;
    validate_key_declarations(spec)?;
    validate_outputs(&spec.outputs)?;
    validate_late_policy(spec.late_policy)
}

fn validate_key_names(field: &str, columns: &[String]) -> Result<()> {
    for (index, column) in columns.iter().enumerate() {
        let indexed = format!("{field}[{index}]");
        if column.is_empty() {
            return Err(invalid_argument(&indexed, "must not be empty"));
        }
        if columns[..index].contains(column) {
            return Err(invalid_argument(
                &indexed,
                "duplicates an earlier key column",
            ));
        }
    }
    Ok(())
}

fn validate_output_version_and_counts(output: &CrossSectionOutputSpec, base: &str) -> Result<()> {
    if output.primitive_version() != 1 {
        return Err(invalid_argument(
            &format!("{base}.primitive_version"),
            "unsupported cross-section primitive version",
        ));
    }
    if output.min_samples() == 0 {
        return Err(invalid_argument(
            &format!("{base}.min_samples"),
            "must be greater than zero",
        ));
    }
    if let Some(ddof) = output.ddof()
        && ddof > 1
    {
        return Err(invalid_argument(&format!("{base}.ddof"), "must be 0 or 1"));
    }
    if output.count() == Some(0) {
        return Err(invalid_argument(
            &format!("{base}.count"),
            "must be greater than zero",
        ));
    }
    if let CrossSectionOutputSpec::Winsorize { lower, upper, .. } = output {
        if !lower.is_finite() {
            return Err(invalid_argument(&format!("{base}.lower"), "must be finite"));
        }
        if !upper.is_finite() {
            return Err(invalid_argument(&format!("{base}.upper"), "must be finite"));
        }
        if !(0.0..=1.0).contains(lower) || lower > upper {
            return Err(invalid_argument(
                &format!("{base}.lower"),
                "bounds must satisfy 0 <= lower <= upper <= 1",
            ));
        }
        if !(0.0..=1.0).contains(upper) {
            return Err(invalid_argument(
                &format!("{base}.upper"),
                "bounds must satisfy 0 <= lower <= upper <= 1",
            ));
        }
    }
    Ok(())
}

fn validate_output_names(
    output: &CrossSectionOutputSpec,
    earlier: &[CrossSectionOutputSpec],
    base: &str,
) -> Result<()> {
    if output.input().is_empty() {
        return Err(invalid_argument(
            &format!("{base}.input"),
            "must not be empty",
        ));
    }
    if output.output().is_empty() {
        return Err(invalid_argument(
            &format!("{base}.output"),
            "must not be empty",
        ));
    }
    if earlier
        .iter()
        .any(|candidate| candidate.output() == output.output())
    {
        return Err(invalid_argument(
            &format!("{base}.output"),
            "duplicates an earlier cross-section output",
        ));
    }
    Ok(())
}

fn validate_outputs(outputs: &[CrossSectionOutputSpec]) -> Result<()> {
    if outputs.is_empty() {
        return Err(invalid_argument(
            "cross_section.outputs",
            "must not be empty",
        ));
    }
    for (index, output) in outputs.iter().enumerate() {
        let base = format!("cross_section.outputs[{index}]");
        validate_output_version_and_counts(output, &base)?;
        validate_output_names(output, &outputs[..index], &base)?;
    }
    Ok(())
}

fn compile_spec(
    spec: &CrossSectionSpec,
    input_schema: &Schema,
) -> Result<CompiledCrossSectionSpec> {
    compile_spec_against_schema(spec, input_schema, String::new())
}

fn compile_spec_full(
    spec: &CrossSectionSpec,
    input_schema: &Schema,
    configuration: &JsonMap,
) -> Result<CompiledCrossSectionSpec> {
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    let configuration_hash = hex::encode(Sha256::digest(canonical.as_bytes()));
    compile_spec_against_schema(spec, input_schema, configuration_hash)
}

fn compile_spec_against_schema(
    spec: &CrossSectionSpec,
    input_schema: &Schema,
    configuration_hash: String,
) -> Result<CompiledCrossSectionSpec> {
    let event_time_index = exact_field_index(input_schema, &spec.event_time)?;
    validate_event_time(input_schema, event_time_index, &spec.event_time)?;
    let entity_columns = spec
        .entity_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Entity))
        .collect::<Result<Vec<_>>>()?;
    let partition_columns = spec
        .partition_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Partition))
        .collect::<Result<Vec<_>>>()?;
    let sequence_columns = spec
        .sequence_by
        .iter()
        .map(|column| compile_key_column(input_schema, column, KeyRole::Sequence))
        .collect::<Result<Vec<_>>>()?;
    let outputs = spec
        .outputs
        .iter()
        .enumerate()
        .map(|(ordinal, output)| compile_output(input_schema, output, ordinal))
        .collect::<Result<Vec<_>>>()?;
    Ok(CompiledCrossSectionSpec {
        input_width: input_schema.fields().len(),
        event_time_index,
        entity_columns,
        partition_columns,
        sequence_columns,
        grouping: spec.grouping,
        allowed_lateness_micros: spec.allowed_lateness_micros,
        outputs,
        configuration_hash,
        state_schema_fingerprint: state_schema_fingerprint(input_schema),
    })
}

#[derive(Clone, Copy)]
enum KeyRole {
    Entity,
    Partition,
    Sequence,
}

fn compile_key_column(
    input_schema: &Schema,
    column: &str,
    role: KeyRole,
) -> Result<CompiledKeyColumn> {
    let index = exact_field_index(input_schema, column)?;
    let field = input_schema.field(index);
    let data_type = field.data_type().clone();
    match role {
        KeyRole::Entity | KeyRole::Partition => {
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "cross-section key column {column:?} has unsupported type {data_type}"
                )));
            }
        }
        KeyRole::Sequence => {
            if field.is_nullable() {
                return Err(compile_error(format!(
                    "cross-section sequence column {column:?} must be non-nullable"
                )));
            }
            if matches!(data_type, DataType::Float32 | DataType::Float64) {
                return Err(compile_error(format!(
                    "cross-section sequence column {column:?} must not use a floating type"
                )));
            }
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "cross-section sequence column {column:?} has unsupported type {data_type}"
                )));
            }
        }
    }
    Ok(CompiledKeyColumn { index })
}

fn compile_output(
    input_schema: &Schema,
    output: &CrossSectionOutputSpec,
    ordinal: usize,
) -> Result<CompiledCrossSectionOutput> {
    if input_schema
        .fields()
        .iter()
        .any(|field| field.name() == output.output())
    {
        return Err(invalid_argument(
            &format!("cross_section.outputs[{ordinal}].output"),
            "collides with an input field name",
        ));
    }
    let input_index = exact_field_index(input_schema, output.input())?;
    let input_type = input_schema.field(input_index).data_type().clone();
    validate_output_input_type(output, &input_type)?;
    Ok(CompiledCrossSectionOutput {
        input_index,
        name: output.output().to_owned(),
        data_type: output_data_type(output, input_type),
        evaluation: compile_evaluation(output),
    })
}

fn validate_output_input_type(
    output: &CrossSectionOutputSpec,
    input_type: &DataType,
) -> Result<()> {
    if matches!(
        output,
        CrossSectionOutputSpec::Winsorize { .. } | CrossSectionOutputSpec::MeanFill { .. }
    ) && !matches!(input_type, DataType::Float32 | DataType::Float64)
    {
        return Err(compile_error(format!(
            "cross-section {} does not support column {:?} with type {}; expected float32 or float64",
            output_kind(output),
            output.input(),
            input_type
        )));
    }
    if !is_numeric(input_type) {
        return Err(compile_error(format!(
            "cross-section {} does not support column {:?} with type {}",
            output_kind(output),
            output.input(),
            input_type
        )));
    }
    Ok(())
}

fn compile_evaluation(output: &CrossSectionOutputSpec) -> CompiledEvaluation {
    match output {
        CrossSectionOutputSpec::Rank {
            direction,
            tie_method,
            null_placement,
            min_samples,
            ..
        } => CompiledEvaluation::Rank {
            direction: *direction,
            tie_method: *tie_method,
            null_placement: *null_placement,
            min_samples: *min_samples,
        },
        CrossSectionOutputSpec::Percentile {
            direction,
            tie_method,
            null_placement,
            min_samples,
            ..
        } => CompiledEvaluation::Percentile {
            direction: *direction,
            tie_method: *tie_method,
            null_placement: *null_placement,
            min_samples: *min_samples,
        },
        CrossSectionOutputSpec::Demean { min_samples, .. } => CompiledEvaluation::Moment {
            min_samples: *min_samples,
            ddof: 0,
        },
        CrossSectionOutputSpec::Zscore {
            min_samples, ddof, ..
        } => CompiledEvaluation::ZScore {
            min_samples: *min_samples,
            ddof: *ddof,
        },
        CrossSectionOutputSpec::Winsorize {
            min_samples,
            lower,
            upper,
            ..
        } => CompiledEvaluation::Winsorize {
            min_samples: *min_samples,
            lower: *lower,
            upper: *upper,
        },
        CrossSectionOutputSpec::Top {
            count,
            include_ties,
            min_samples,
            ..
        } => CompiledEvaluation::TopSelection {
            count: *count,
            include_ties: *include_ties,
            min_samples: *min_samples,
        },
        CrossSectionOutputSpec::Bottom {
            count,
            include_ties,
            min_samples,
            ..
        } => CompiledEvaluation::BottomSelection {
            count: *count,
            include_ties: *include_ties,
            min_samples: *min_samples,
        },
        CrossSectionOutputSpec::MeanFill { min_samples, .. } => CompiledEvaluation::MeanFill {
            min_samples: *min_samples,
        },
    }
}

fn output_data_type(output: &CrossSectionOutputSpec, input_type: DataType) -> DataType {
    match output {
        CrossSectionOutputSpec::Winsorize { .. } | CrossSectionOutputSpec::MeanFill { .. } => {
            input_type
        }
        CrossSectionOutputSpec::Top { .. } | CrossSectionOutputSpec::Bottom { .. } => {
            DataType::Boolean
        }
        CrossSectionOutputSpec::Rank { .. }
        | CrossSectionOutputSpec::Percentile { .. }
        | CrossSectionOutputSpec::Demean { .. }
        | CrossSectionOutputSpec::Zscore { .. } => DataType::Float64,
    }
}

fn output_kind(output: &CrossSectionOutputSpec) -> &'static str {
    match output {
        CrossSectionOutputSpec::Rank { .. } => "rank",
        CrossSectionOutputSpec::Percentile { .. } => "percentile",
        CrossSectionOutputSpec::Demean { .. } => "demean",
        CrossSectionOutputSpec::Zscore { .. } => "zscore",
        CrossSectionOutputSpec::Winsorize { .. } => "winsorize",
        CrossSectionOutputSpec::Top { .. } => "top",
        CrossSectionOutputSpec::Bottom { .. } => "bottom",
        CrossSectionOutputSpec::MeanFill { .. } => "mean_fill",
    }
}

fn exact_field_index(schema: &Schema, column: &str) -> Result<usize> {
    let matches = schema
        .fields()
        .iter()
        .enumerate()
        .filter(|(_, field)| field.name() == column)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(compile_error(format!(
            "cross-section column {column:?} does not exist in the input schema"
        ))),
        _ => Err(compile_error(format!(
            "cross-section column {column:?} is ambiguous in the input schema"
        ))),
    }
}

fn validate_event_time(schema: &Schema, index: usize, column: &str) -> Result<()> {
    let field = schema.field(index);
    if field.is_nullable() {
        return Err(compile_error(format!(
            "cross-section event-time column {column:?} must be non-nullable"
        )));
    }
    if !matches!(
        field.data_type(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) if timezone.as_ref() == "UTC"
    ) {
        return Err(compile_error(format!(
            "cross-section event-time column {column:?} must be a non-null UTC timestamp[us], found {}",
            field.data_type()
        )));
    }
    Ok(())
}

fn supports_total_order(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Boolean
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Float32
            | DataType::Float64
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
    ) || matches!(
        data_type,
        DataType::Timestamp(TimeUnit::Microsecond, timezone)
            if timezone.as_deref().is_none_or(|timezone| timezone == "UTC")
    )
}

fn is_numeric(data_type: &DataType) -> bool {
    matches!(
        data_type,
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
    )
}

fn output_schema(input_schema: &Schema, outputs: &[CompiledCrossSectionOutput]) -> Schema {
    let mut fields = input_schema.fields().to_vec();
    fields.extend(
        outputs
            .iter()
            .map(|output| Field::new(&output.name, output.data_type.clone(), true).into()),
    );
    Schema::new(fields)
}

fn configuration(spec: &CrossSectionSpec) -> Result<JsonMap> {
    let spec_json = serde_json::to_value(spec).map_err(|error| format_error(&error))?;
    Ok(JsonMap::from([
        ("kind".into(), json!("cross_section")),
        ("spec".into(), spec_json),
    ]))
}

fn invalid_argument(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn operator_error(node_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: node_id.into(),
        message: message.into(),
    }
}

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("industry", DataType::Utf8, true),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("value", DataType::Float64, true),
        ])
    }

    fn rank_output(input: &str, output: &str) -> CrossSectionOutputSpec {
        CrossSectionOutputSpec::Rank {
            primitive_version: 1,
            input: input.into(),
            output: output.into(),
            direction: SortDirection::Ascending,
            tie_method: RankTieMethod::Average,
            null_placement: NullPlacement::Exclude,
            min_samples: 1,
        }
    }

    fn valid_spec() -> CrossSectionSpec {
        CrossSectionSpec {
            configuration_version: CROSS_SECTION_CONFIGURATION_VERSION,
            state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
            event_time: "ts".into(),
            entity_by: vec!["symbol".into()],
            partition_by: vec!["industry".into()],
            sequence_by: vec!["sequence".into()],
            grouping: CrossSectionGroupingSpec::ExactTime,
            outputs: vec![rank_output("value", "rank")],
            allowed_lateness_micros: 0,
            late_policy: LatePolicySpec::Drop { metrics_version: 1 },
            value_policy: CrossSectionValuePolicy::NanExcludePreserveV1,
        }
    }

    fn compiled_spec() -> CompiledCrossSectionSpec {
        let spec = valid_spec();
        let schema = input_schema();
        let settings = configuration(&spec).unwrap();
        compile_spec_full(&spec, &schema, &settings).unwrap()
    }

    fn assert_invalid(spec: &CrossSectionSpec, expected_field: &str) {
        let error = validate_arguments(spec).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. } if field == expected_field
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn declaration_validation_reports_each_stable_field() {
        let mut spec = valid_spec();
        spec.configuration_version = 2;
        assert_invalid(&spec, "cross_section.configuration_version");

        let mut spec = valid_spec();
        spec.state_layout_version = 2;
        assert_invalid(&spec, "cross_section.state_layout_version");

        let mut spec = valid_spec();
        spec.grouping = CrossSectionGroupingSpec::FixedBucket { width_micros: 0 };
        assert_invalid(&spec, "cross_section.grouping.width_micros");

        let mut spec = valid_spec();
        spec.entity_by.clear();
        assert_invalid(&spec, "cross_section.entity_by");

        let mut spec = valid_spec();
        spec.sequence_by.clear();
        assert_invalid(&spec, "cross_section.sequence_by");

        let mut spec = valid_spec();
        spec.entity_by = vec![String::new()];
        assert_invalid(&spec, "cross_section.entity_by[0]");

        let mut spec = valid_spec();
        spec.partition_by = vec!["industry".into(), "industry".into()];
        assert_invalid(&spec, "cross_section.partition_by[1]");

        let mut spec = valid_spec();
        spec.outputs.clear();
        assert_invalid(&spec, "cross_section.outputs");

        let mut spec = valid_spec();
        spec.late_policy = LatePolicySpec::Drop { metrics_version: 2 };
        assert_invalid(&spec, "cross_section.late_policy.metrics_version");
    }

    #[test]
    fn output_validation_reports_each_stable_field() {
        let mut spec = valid_spec();
        let CrossSectionOutputSpec::Rank {
            primitive_version, ..
        } = &mut spec.outputs[0]
        else {
            unreachable!();
        };
        *primitive_version = 2;
        assert_invalid(&spec, "cross_section.outputs[0].primitive_version");

        let mut spec = valid_spec();
        let CrossSectionOutputSpec::Rank { min_samples, .. } = &mut spec.outputs[0] else {
            unreachable!();
        };
        *min_samples = 0;
        assert_invalid(&spec, "cross_section.outputs[0].min_samples");

        let mut spec = valid_spec();
        spec.outputs = vec![CrossSectionOutputSpec::Zscore {
            primitive_version: 1,
            input: "value".into(),
            output: "zscore".into(),
            min_samples: 1,
            ddof: 2,
        }];
        assert_invalid(&spec, "cross_section.outputs[0].ddof");

        let mut spec = valid_spec();
        spec.outputs = vec![rank_output("", "rank")];
        assert_invalid(&spec, "cross_section.outputs[0].input");

        let mut spec = valid_spec();
        spec.outputs = vec![rank_output("value", "")];
        assert_invalid(&spec, "cross_section.outputs[0].output");

        let mut spec = valid_spec();
        spec.outputs = vec![rank_output("value", "rank"), rank_output("value", "rank")];
        assert_invalid(&spec, "cross_section.outputs[1].output");
    }

    #[test]
    fn schema_compilation_rejects_unsupported_columns() {
        let schema = input_schema();
        assert!(valid_spec().validate(&schema).is_ok());
        assert!(exact_field_index(&schema, "missing").is_err());

        let ambiguous = Schema::new(vec![
            Field::new("duplicate", DataType::Int64, false),
            Field::new("duplicate", DataType::Int64, false),
        ]);
        assert!(exact_field_index(&ambiguous, "duplicate").is_err());

        let nullable_event_time = Schema::new(vec![Field::new(
            "ts",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            true,
        )]);
        assert!(validate_event_time(&nullable_event_time, 0, "ts").is_err());
        let wrong_event_time = Schema::new(vec![Field::new("ts", DataType::Int64, false)]);
        assert!(validate_event_time(&wrong_event_time, 0, "ts").is_err());

        let unsupported = Schema::new(vec![Field::new("key", DataType::Binary, false)]);
        assert!(compile_key_column(&unsupported, "key", KeyRole::Entity).is_err());
        assert!(compile_key_column(&unsupported, "key", KeyRole::Sequence).is_err());
        let nullable_sequence = Schema::new(vec![Field::new("key", DataType::UInt64, true)]);
        assert!(compile_key_column(&nullable_sequence, "key", KeyRole::Sequence).is_err());
        let float_sequence = Schema::new(vec![Field::new("key", DataType::Float64, false)]);
        assert!(compile_key_column(&float_sequence, "key", KeyRole::Sequence).is_err());

        let non_numeric = Schema::new(vec![Field::new("label", DataType::Utf8, false)]);
        assert!(compile_output(&non_numeric, &rank_output("label", "rank"), 0).is_err());
        assert!(compile_output(&schema, &rank_output("value", "value"), 0).is_err());
    }

    #[test]
    fn type_tables_cover_every_supported_key_and_numeric_class() {
        for data_type in [
            DataType::Boolean,
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Date32,
            DataType::Date64,
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
        ] {
            assert!(supports_total_order(&data_type), "unsupported {data_type}");
        }
        assert!(!supports_total_order(&DataType::Binary));
        for data_type in [
            DataType::Int8,
            DataType::Int16,
            DataType::Int32,
            DataType::Int64,
            DataType::UInt8,
            DataType::UInt16,
            DataType::UInt32,
            DataType::UInt64,
            DataType::Float32,
            DataType::Float64,
        ] {
            assert!(is_numeric(&data_type), "non-numeric {data_type}");
        }
        assert!(!is_numeric(&DataType::Utf8));
    }

    fn assert_key(scalar: &ScalarValue, expected: &KeyValue) {
        let actual = KeyValue::from_required_scalar(scalar, "cross_section").unwrap();
        assert_eq!(&actual, expected);
    }

    #[test]
    fn key_values_cover_every_scalar_and_total_order_variant() {
        assert_eq!(
            KeyValue::from_nullable_scalar(&ScalarValue::Utf8(None), "cross_section").unwrap(),
            None
        );
        assert_key(&ScalarValue::Boolean(Some(true)), &KeyValue::Boolean(true));
        assert_key(&ScalarValue::Int8(Some(1)), &KeyValue::Signed(1));
        assert_key(&ScalarValue::Int16(Some(2)), &KeyValue::Signed(2));
        assert_key(&ScalarValue::Int32(Some(3)), &KeyValue::Signed(3));
        assert_key(&ScalarValue::Int64(Some(4)), &KeyValue::Signed(4));
        assert_key(&ScalarValue::UInt8(Some(5)), &KeyValue::Unsigned(5));
        assert_key(&ScalarValue::UInt16(Some(6)), &KeyValue::Unsigned(6));
        assert_key(&ScalarValue::UInt32(Some(7)), &KeyValue::Unsigned(7));
        assert_key(&ScalarValue::UInt64(Some(8)), &KeyValue::Unsigned(8));
        assert_key(&ScalarValue::Float32(Some(1.5)), &KeyValue::Float32(1.5));
        assert_key(&ScalarValue::Float64(Some(2.5)), &KeyValue::Float64(2.5));
        assert_key(
            &ScalarValue::Utf8(Some("a".into())),
            &KeyValue::String("a".into()),
        );
        assert_key(
            &ScalarValue::LargeUtf8(Some("b".into())),
            &KeyValue::String("b".into()),
        );
        assert_key(&ScalarValue::Date32(Some(9)), &KeyValue::Date32(9));
        assert_key(&ScalarValue::Date64(Some(10)), &KeyValue::Date64(10));
        assert_key(
            &ScalarValue::TimestampMicrosecond(Some(11), Some(Arc::from("UTC"))),
            &KeyValue::Timestamp(11),
        );
        assert!(
            KeyValue::from_required_scalar(&ScalarValue::Binary(Some(vec![1])), "cross_section")
                .is_err()
        );
        assert!(
            KeyValue::from_required_scalar(&ScalarValue::UInt64(None), "cross_section").is_err()
        );

        let pairs = [
            (KeyValue::Boolean(false), KeyValue::Boolean(true)),
            (KeyValue::Signed(1), KeyValue::Signed(2)),
            (KeyValue::Unsigned(1), KeyValue::Unsigned(2)),
            (KeyValue::Float32(1.0), KeyValue::Float32(2.0)),
            (KeyValue::Float64(1.0), KeyValue::Float64(2.0)),
            (KeyValue::String("a".into()), KeyValue::String("b".into())),
            (KeyValue::Date32(1), KeyValue::Date32(2)),
            (KeyValue::Date64(1), KeyValue::Date64(2)),
            (KeyValue::Timestamp(1), KeyValue::Timestamp(2)),
        ];
        for (left, right) in pairs {
            assert!(left < right);
            assert_eq!(left.partial_cmp(&right), Some(Ordering::Less));
        }
        assert!(KeyValue::Boolean(true) < KeyValue::Signed(0));
    }

    #[test]
    fn order_and_statistic_helpers_cover_edge_semantics() {
        assert!((type_seven_quantile(&[0.0, 10.0], 0.25) - 2.5).abs() < f64::EPSILON);
        assert_eq!(
            type_seven_quantile(&[1.0], 0.5).to_bits(),
            1.0_f64.to_bits()
        );
        assert_eq!(
            type_seven_quantile(&[1.0, 1.0], 0.5).to_bits(),
            1.0_f64.to_bits()
        );
        assert!(type_seven_quantile(&[f64::NEG_INFINITY, f64::INFINITY], 0.5).is_nan());
        let negative = type_seven_quantile(&[f64::NEG_INFINITY, 0.0], 0.5);
        assert!(negative.is_infinite() && negative.is_sign_negative());
        let positive = type_seven_quantile(&[0.0, f64::INFINITY], 0.5);
        assert!(positive.is_infinite() && positive.is_sign_positive());

        let order_samples = [
            OrderSample::Valid(KeyValue::Float64(2.0)),
            OrderSample::Valid(KeyValue::Float64(2.0)),
            OrderSample::Valid(KeyValue::Float64(1.0)),
            OrderSample::Null,
            OrderSample::Nan,
        ];
        let ranks = order_statistic_column(
            &order_samples,
            SortDirection::Ascending,
            RankTieMethod::Average,
            NullPlacement::Exclude,
            1,
            false,
        );
        assert_eq!(&ranks[..4], &[Some(2.5), Some(2.5), Some(1.0), None]);
        assert!(ranks[4].unwrap().is_nan());
        assert_eq!(
            order_statistic_column(
                &[OrderSample::Valid(KeyValue::Float64(1.0))],
                SortDirection::Descending,
                RankTieMethod::Min,
                NullPlacement::First,
                1,
                true,
            ),
            vec![Some(0.5)]
        );
        assert_eq!(
            order_statistic_column(
                &[
                    OrderSample::Null,
                    OrderSample::Valid(KeyValue::Float64(1.0)),
                ],
                SortDirection::Descending,
                RankTieMethod::Max,
                NullPlacement::Last,
                2,
                false,
            ),
            vec![None, None]
        );

        let samples = [
            Sample::Null,
            Sample::Nan,
            Sample::Valid(1.0),
            Sample::Valid(3.0),
        ];
        let centered = statistic_column(&samples, 1, 0, false);
        assert_eq!(centered[0], None);
        assert!(centered[1].unwrap().is_nan());
        assert_eq!(&centered[2..], &[Some(-1.0), Some(1.0)]);
        assert_eq!(
            statistic_column(&[Sample::Valid(1.0)], 2, 0, false),
            vec![None]
        );
        assert_eq!(
            statistic_column(&[Sample::Valid(1.0)], 1, 1, true),
            vec![None]
        );
        assert_eq!(
            statistic_column(&[Sample::Valid(2.0), Sample::Valid(2.0)], 1, 0, true),
            vec![None, None]
        );
        let infinities = statistic_column(
            &[
                Sample::Valid(f64::INFINITY),
                Sample::Valid(f64::NEG_INFINITY),
            ],
            1,
            0,
            true,
        );
        assert!(infinities.iter().all(|value| value.unwrap().is_nan()));
        let negative = statistic_column(&[Sample::Valid(f64::NEG_INFINITY)], 1, 0, false);
        assert!(negative[0].unwrap().is_nan());
    }

    #[test]
    fn group_coordinates_reject_unrepresentable_ranges() {
        assert!(bucket_width(0, "cross_section").is_err());
        assert!(bucket_width(u64::MAX, "cross_section").is_err());

        let mut compiled = compiled_spec();
        compiled.grouping = CrossSectionGroupingSpec::FixedBucket {
            width_micros: u64::MAX,
        };
        let row = BufferedRow::new(
            vec![Some(KeyValue::String("a".into()))],
            vec![KeyValue::Unsigned(1)],
            0,
            vec![
                ScalarValue::TimestampMicrosecond(Some(0), Some(Arc::from("UTC"))),
                ScalarValue::Utf8(Some("a".into())),
                ScalarValue::Utf8(None),
                ScalarValue::UInt64(Some(1)),
                ScalarValue::Float64(Some(1.0)),
            ],
        );
        assert!(compiled.group_key(&row, "cross_section").is_err());

        compiled.grouping = CrossSectionGroupingSpec::ExactTime;
        compiled.allowed_lateness_micros = u64::MAX;
        let group = GroupKey {
            base: 0,
            partition: Vec::new(),
        };
        assert!(compiled.group_closing(&group, "cross_section").is_err());
        compiled.allowed_lateness_micros = 1;
        let group = GroupKey {
            base: i64::MAX,
            partition: Vec::new(),
        };
        assert!(compiled.group_closing(&group, "cross_section").is_err());
        compiled.grouping = CrossSectionGroupingSpec::FixedBucket { width_micros: 1 };
        compiled.allowed_lateness_micros = 0;
        assert!(compiled.group_closing(&group, "cross_section").is_err());
    }

    fn descriptor_fixture(
        compiled: &CompiledCrossSectionSpec,
        operator_id: &str,
        epoch: Epoch,
        bytes: Vec<u8>,
    ) -> (SegmentDescriptor, crate::StateSegment) {
        let segment = crate::StateSegment::new(bytes);
        let segment_id = format!("base-{:020}-00000000", epoch.as_u64());
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let handle = StateHandle::new(
            operator_id,
            epoch,
            &segment_id,
            &relative_path,
            u64::try_from(segment.bytes().len()).unwrap(),
            segment.sha256(),
        )
        .unwrap();
        (
            SegmentDescriptor {
                kind: SegmentKind::Base,
                state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
                schema_fingerprint: compiled.state_schema_fingerprint.clone(),
                handle,
            },
            segment,
        )
    }

    fn snapshot_metadata(
        compiled: &CompiledCrossSectionSpec,
        descriptors: Vec<SegmentDescriptor>,
    ) -> CrossSectionSnapshotMetadata {
        CrossSectionSnapshotMetadata {
            state_layout_version: CROSS_SECTION_STATE_LAYOUT_VERSION,
            configuration_hash: compiled.configuration_hash.clone(),
            state_schema_fingerprint: compiled.state_schema_fingerprint.clone(),
            epoch: Epoch::INITIAL,
            pipeline_fingerprint: Some("a".repeat(64)),
            operator_id: Some("cross_section".into()),
            last_input_watermark: None,
            next_output_sequence: 0,
            ended: false,
            metrics: LateMetricDelta::default(),
            segment_inventory: descriptors,
        }
    }

    #[test]
    fn segment_schema_metadata_rejects_mismatch_and_missing_identity() {
        let compiled = compiled_spec();
        let schema = state_schema(&input_schema(), &compiled, &"a".repeat(64), "cross_section");
        assert!(validate_segment_schema_metadata(schema.metadata(), &compiled).is_ok());

        let mut metadata = schema.metadata().clone();
        metadata.insert(
            "calc_flow.operator_configuration_hash".into(),
            "b".repeat(64),
        );
        assert!(validate_segment_schema_metadata(&metadata, &compiled).is_err());

        let mut metadata = schema.metadata().clone();
        metadata.remove("calc_flow.operator_id");
        assert!(validate_segment_schema_metadata(&metadata, &compiled).is_err());
    }

    #[test]
    fn snapshot_metadata_rejects_each_identity_and_inventory_mismatch() {
        let compiled = compiled_spec();
        let empty = crate::OperatorStateSnapshot::default();

        let mut metadata = snapshot_metadata(&compiled, Vec::new());
        metadata.state_layout_version = 2;
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());
        let mut metadata = snapshot_metadata(&compiled, Vec::new());
        metadata.configuration_hash = "b".repeat(64);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());
        let mut metadata = snapshot_metadata(&compiled, Vec::new());
        metadata.state_schema_fingerprint = "b".repeat(64);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());
        let mut metadata = snapshot_metadata(&compiled, Vec::new());
        metadata.pipeline_fingerprint = Some("not-a-digest".into());
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());
        let mut metadata = snapshot_metadata(&compiled, Vec::new());
        metadata.operator_id = Some(String::new());
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());

        let (descriptor, segment) =
            descriptor_fixture(&compiled, "cross_section", Epoch::INITIAL, vec![1, 2, 3]);
        let segment_id = descriptor.handle.segment_id().to_owned();
        let snapshot = crate::OperatorStateSnapshot {
            inline_metadata: JsonMap::new(),
            segments: BTreeMap::from([(segment_id, segment)]),
        };
        let metadata = snapshot_metadata(&compiled, vec![descriptor.clone()]);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &snapshot).is_ok());

        let mut wrong_layout = descriptor.clone();
        wrong_layout.state_layout_version = 2;
        let metadata = snapshot_metadata(&compiled, vec![wrong_layout]);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &snapshot).is_err());

        let (future, _) = descriptor_fixture(
            &compiled,
            "cross_section",
            Epoch::new(2).unwrap(),
            vec![1, 2, 3],
        );
        let metadata = snapshot_metadata(&compiled, vec![future]);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &snapshot).is_err());

        let (other, _) = descriptor_fixture(&compiled, "other", Epoch::INITIAL, vec![1, 2, 3]);
        let metadata = snapshot_metadata(&compiled, vec![other]);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &snapshot).is_err());

        let metadata = snapshot_metadata(&compiled, vec![descriptor.clone()]);
        assert!(validate_snapshot_metadata(&metadata, &compiled, &empty).is_err());
        let mut metadata = snapshot_metadata(&compiled, vec![descriptor]);
        metadata.pipeline_fingerprint = None;
        assert!(validate_snapshot_metadata(&metadata, &compiled, &snapshot).is_err());
    }

    #[test]
    fn snapshot_segments_revalidate_presence_length_and_checksum() {
        let compiled = compiled_spec();
        let (descriptor, segment) =
            descriptor_fixture(&compiled, "cross_section", Epoch::INITIAL, vec![1, 2, 3]);
        let segment_id = descriptor.handle.segment_id().to_owned();
        assert!(
            snapshot_segments(
                &crate::OperatorStateSnapshot::default(),
                &[descriptor.clone()]
            )
            .is_err()
        );

        let short = crate::OperatorStateSnapshot {
            inline_metadata: JsonMap::new(),
            segments: BTreeMap::from([(segment_id.clone(), crate::StateSegment::new(vec![1]))]),
        };
        assert!(snapshot_segments(&short, &[descriptor.clone()]).is_err());
        let corrupt = crate::OperatorStateSnapshot {
            inline_metadata: JsonMap::new(),
            segments: BTreeMap::from([(
                segment_id.clone(),
                crate::StateSegment::new(vec![3, 2, 1]),
            )]),
        };
        assert!(snapshot_segments(&corrupt, &[descriptor.clone()]).is_err());
        let valid = crate::OperatorStateSnapshot {
            inline_metadata: JsonMap::new(),
            segments: BTreeMap::from([(segment_id, segment)]),
        };
        assert_eq!(
            snapshot_segments(&valid, &[descriptor]).unwrap()[0].as_slice(),
            &[1, 2, 3]
        );
    }

    #[test]
    fn late_metrics_and_chunking_cover_failure_boundaries() {
        let mut metrics = PreparedLateMetrics::default();
        record_late_row(&mut metrics, None, 10, "cross_section").unwrap();
        record_late_row(
            &mut metrics,
            Some(EventTime::from_micros(20)),
            10,
            "cross_section",
        )
        .unwrap();
        assert_eq!(metrics.late_rows, 1);
        assert_eq!(metrics.max_lateness_micros, Some(10));
        metrics.late_rows = u64::MAX;
        assert!(
            record_late_row(
                &mut metrics,
                Some(EventTime::from_micros(20)),
                10,
                "cross_section",
            )
            .is_err()
        );
        let mut metrics = PreparedLateMetrics::default();
        assert!(
            record_late_row(
                &mut metrics,
                Some(EventTime::from_micros(10)),
                20,
                "cross_section",
            )
            .is_err()
        );

        let record = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "value",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(datafusion::arrow::array::Int64Array::from(vec![
                1_i64,
            ]))],
        )
        .unwrap();
        assert!(
            chunk_output_record(
                &record,
                "cross_section",
                0,
                crate::EdgeBudget::new(1, 1).unwrap(),
            )
            .is_err()
        );
        assert!(
            chunk_output_record(
                &record,
                "cross_section",
                u64::MAX,
                crate::EdgeBudget::new(1, usize::MAX).unwrap(),
            )
            .is_err()
        );
        let empty = record.slice(0, 0);
        assert!(
            chunk_output_record(&empty, "cross_section", 0, crate::EdgeBudget::default(),)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn metadata_accessors_and_empty_group_record_are_observable() {
        let spec = valid_spec();
        let operator =
            CrossSectionOperator::new("cross_section", Arc::new(input_schema()), spec).unwrap();
        assert_eq!(operator.name(), "cross_section");
        assert_eq!(operator.spec(), &valid_spec());
        assert!(format!("{operator:?}").contains("cross_section"));
        let record = build_grouped_record(
            &Groups::new(),
            &operator.compiled,
            operator.output_ports()[0].schema().unwrap(),
            operator.name(),
        )
        .unwrap();
        assert_eq!(record.num_rows(), 0);

        for output in [
            rank_output("value", "rank"),
            CrossSectionOutputSpec::Percentile {
                primitive_version: 1,
                input: "value".into(),
                output: "percentile".into(),
                direction: SortDirection::Ascending,
                tie_method: RankTieMethod::Average,
                null_placement: NullPlacement::Exclude,
                min_samples: 1,
            },
            CrossSectionOutputSpec::Demean {
                primitive_version: 1,
                input: "value".into(),
                output: "demean".into(),
                min_samples: 1,
            },
            CrossSectionOutputSpec::Zscore {
                primitive_version: 1,
                input: "value".into(),
                output: "zscore".into(),
                min_samples: 1,
                ddof: 0,
            },
        ] {
            assert!(!output_kind(&output).is_empty());
        }
        assert!(matches!(
            internal_error("test"),
            CalcFlowError::Internal { .. }
        ));
        assert!(matches!(state_format("test"), CalcFlowError::Format { .. }));
        assert!(matches!(
            compile_error("test"),
            CalcFlowError::Compile { .. }
        ));
    }
}
