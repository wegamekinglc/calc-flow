//! Native row-window rolling operator: lag and delta over entity-partitioned,
//! event-time ordered rows (SCE-00 D5, API note `symbolic-computation-engine`
//! section 3.2). The same calculation kernel serves batch and final-only
//! stream lifecycles; stream state is checkpointed at the aligned epoch cut.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap, VecDeque},
    io::Cursor,
    sync::Arc,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{ArrayRef, UInt8Array, UInt64Array, new_null_array},
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

use super::{
    BatchOperator, BatchOperatorContext, LateMetricDelta, OperatorMetadata, StreamCollector,
    StreamOperator, StreamOperatorContext, accumulate_late_metrics, expression::required_input,
    validate_operator_name,
};

/// Semantic configuration version of the first rolling operator release.
pub const ROLLING_CONFIGURATION_VERSION: u32 = 1;
/// Durable state-layout version of the first rolling operator release.
pub const ROLLING_STATE_LAYOUT_VERSION: u32 = 1;

/// Transaction scope of the `error` late-row policy (API note section 3.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum LateErrorScope {
    /// The complete input envelope is rejected atomically.
    Envelope,
}

/// Late-row handling for one rolling operator (SCE-00 D7).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LatePolicySpec {
    /// Reject the complete input envelope without state, metric, or output
    /// changes.
    Error {
        /// The only supported transaction scope.
        scope: LateErrorScope,
    },
    /// Drop each late row and record the three D7 metrics.
    Drop {
        /// Metric transaction version; must equal `1`.
        metrics_version: u32,
    },
}

/// Frozen null/NaN policy for rolling values (SCE-00 D3.2).
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RollingValuePolicy {
    /// Lag/delta preserve a null or NaN current or referenced operand.
    StatefulNumericV1,
}

/// One declared rolling output and its output column name.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum RollingOutputSpec {
    /// Value of the same column `periods` earlier in the entity total order.
    Lag {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive lag distance in rows.
        periods: u64,
    },
    /// Checked difference between the current value and the value `periods`
    /// earlier in the entity total order.
    Delta {
        /// Primitive version; must equal `1`.
        primitive_version: u32,
        /// Input column name.
        input: String,
        /// Output column name.
        output: String,
        /// Positive lag distance in rows.
        periods: u64,
    },
}

impl RollingOutputSpec {
    fn primitive_version(&self) -> u32 {
        match self {
            Self::Lag {
                primitive_version, ..
            }
            | Self::Delta {
                primitive_version, ..
            } => *primitive_version,
        }
    }

    fn input(&self) -> &str {
        match self {
            Self::Lag { input, .. } | Self::Delta { input, .. } => input,
        }
    }

    fn output(&self) -> &str {
        match self {
            Self::Lag { output, .. } | Self::Delta { output, .. } => output,
        }
    }

    fn periods(&self) -> u64 {
        match self {
            Self::Lag { periods, .. } | Self::Delta { periods, .. } => *periods,
        }
    }

    const fn is_delta(&self) -> bool {
        matches!(self, Self::Delta { .. })
    }
}

/// Data-only declaration of one native row-window rolling operation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RollingSpec {
    /// Semantic configuration version; must equal
    /// [`ROLLING_CONFIGURATION_VERSION`].
    pub configuration_version: u32,
    /// Durable state-layout version; must equal
    /// [`ROLLING_STATE_LAYOUT_VERSION`].
    pub state_layout_version: u32,
    /// Ordered non-empty entity partition key.
    pub partition_by: Vec<String>,
    /// Non-null UTC `timestamp[us]` event-time column.
    pub event_time: String,
    /// Ordered non-empty sequence key; floating columns are forbidden.
    pub sequence_by: Vec<String>,
    /// Rolling outputs in semantic declaration order.
    pub outputs: Vec<RollingOutputSpec>,
    /// Allowed lateness in exact microseconds (SCE-00 D7).
    pub allowed_lateness_micros: u64,
    /// Late-row policy.
    pub late_policy: LatePolicySpec,
    /// Frozen null/NaN value policy.
    pub value_policy: RollingValuePolicy,
}

impl RollingSpec {
    /// Validates the declaration against an exact Arrow input schema and
    /// returns the derived output schema: input fields followed by the
    /// declared outputs in order (SCE-00 D5).
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

/// Native row-window rolling operator over partitioned event-time rows.
pub struct RollingOperator {
    name: String,
    spec: RollingSpec,
    input_ports: [Port; 1],
    output_ports: [Port; 1],
    compiled: CompiledRollingSpec,
    state: RollingStreamState,
}

impl RollingOperator {
    /// Compiles one rolling declaration against an exact Arrow input schema.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for invalid declaration
    /// fields and [`CalcFlowError::Compile`] for missing, ambiguous, or
    /// unsupported input columns.
    pub fn new(name: &str, input_schema: SchemaRef, spec: RollingSpec) -> Result<Self> {
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
            state: RollingStreamState::default(),
        })
    }

    /// Returns the validated rolling declaration.
    pub const fn spec(&self) -> &RollingSpec {
        &self.spec
    }
}

impl std::fmt::Debug for RollingOperator {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RollingOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for RollingOperator {
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
        configuration(&self.spec).expect("validated rolling configuration remains serializable")
    }
}

#[async_trait]
impl BatchOperator for RollingOperator {
    /// Evaluates the complete input in canonical order without late-row
    /// classification (SCE-00 D7): every accepted row is final at
    /// end-of-input.
    async fn process(
        &mut self,
        inputs: &BTreeMap<String, Batch>,
        context: &BatchOperatorContext<'_>,
    ) -> Result<BTreeMap<String, Batch>> {
        let input = required_input(inputs, "input", &self.name, None)?;
        self.input_ports[0].validate(input, &format!("{}.input", self.name))?;
        context.run.check_cancelled()?;
        let rows = read_buffered_rows(input.table_payload()?, &self.compiled, &self.name)?;
        let ordered = sort_and_validate(rows, &self.name)?;
        let computed = compute_output_columns(
            &ordered,
            &RollingHistories::default(),
            &self.compiled,
            &self.name,
        )?;
        let record = build_output_record(
            &ordered,
            computed.columns,
            self.output_ports[0]
                .schema()
                .expect("rolling output always has an exact schema"),
            &self.name,
        )?;
        let metadata = BatchMetadata::new(&self.name, 0, BTreeMap::new())?;
        let batch = Batch::table(vec![record], metadata)?;
        Ok(BTreeMap::from([("output".into(), batch)]))
    }
}

/// Live stream state owned by one rolling operator task. Mutation is
/// confined to this value; input batches stay read-only.
#[derive(Default)]
struct RollingStreamState {
    buffer: BTreeMap<RowIdentity, BufferedRow>,
    histories: RollingHistories,
    last_input_watermark: Option<EventTime>,
    next_output_sequence: u64,
    ended: bool,
    metrics: LateMetricDelta,
    pipeline_fingerprint: Option<String>,
    operator_id: Option<String>,
    last_checkpoint_epoch: Option<Epoch>,
}

/// Bounded inline manifest contribution of one rolling checkpoint (SCE-00
/// D11); retained rows never appear inline, only in segments.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RollingSnapshotMetadata {
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

#[async_trait]
impl StreamOperator for RollingOperator {
    /// Classifies and buffers one input envelope atomically (SCE-00 D7): the
    /// aggregate input watermark is sampled once, and no row changes state,
    /// metrics, or output before the complete envelope is validated.
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if ingress != "input" {
            return Err(operator_error(
                context.operator_id(),
                &format!("unknown ingress {ingress:?}; expected \"input\""),
            ));
        }
        self.input_ports[0].validate(&batch, &format!("{}.input", self.name))?;
        self.observe_context(context)?;
        if self.state.ended {
            return Err(operator_error(
                context.operator_id(),
                "received data after end-of-input",
            ));
        }
        let watermark = context.input_watermark();
        let rows = read_buffered_rows(batch.table_payload()?, &self.compiled, &self.name)?;
        let (accepted, metrics) = self.classify_envelope(rows, watermark, context.operator_id())?;
        let next_metrics = accumulate_late_metrics(self.state.metrics, metrics)?;
        for (identity, row) in accepted {
            self.state.buffer.insert(identity, row);
        }
        context.record_window_metrics(
            metrics.late_rows,
            metrics.max_lateness_micros,
            metrics.null_event_time_rows,
        )?;
        self.state.metrics = next_metrics;
        self.install_context_identity(context);
        Ok(())
    }

    /// Emits every newly final row in canonical order before the runtime
    /// forwards the watermark (SCE-00 D7 final-only output).
    async fn on_watermark(
        &mut self,
        watermark: EventTime,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
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
        let closing = self.closing_keys(watermark.as_micros(), context.operator_id())?;
        let rows = self.take_buffered(&closing);
        self.emit_rows(rows, context, output).await?;
        self.install_context_identity(context);
        self.state.last_input_watermark = Some(watermark);
        Ok(())
    }

    /// Flushes every buffered accepted row once in canonical order; no
    /// sentinel watermark is synthesized (SCE-00 D7).
    async fn on_end(
        &mut self,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        self.observe_context(context)?;
        if self.state.ended {
            return Ok(());
        }
        let rows = self.take_all_buffered();
        self.emit_rows(rows, context, output).await?;
        self.install_context_identity(context);
        self.state.ended = true;
        Ok(())
    }

    /// Captures the finality frontier, per-entity retained history, buffered
    /// unfinalized rows, and late metrics (SCE-00 D11) as one immutable base
    /// segment plus bounded inline metadata.
    fn checkpoint(&mut self, epoch: Epoch) -> Result<crate::OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(checkpoint_mismatch(
                "rolling checkpoint epoch did not advance strictly".into(),
            ));
        }
        let encoded = self.encode_state(epoch)?;
        let (descriptor, segments) = match encoded {
            Some(prepared) => {
                let descriptor = self.snapshot_segment_descriptor(epoch, &prepared)?;
                let mut segments = BTreeMap::new();
                segments.insert(prepared.0, prepared.1);
                (Some(descriptor), segments)
            }
            None => (None, BTreeMap::new()),
        };
        let inventory = StateInventory::new(descriptor.into_iter().collect())
            .map_err(|error| checkpoint_mismatch(error.to_string()))?;
        let metadata = RollingSnapshotMetadata {
            state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
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
                "rolling snapshot metadata did not serialize as an object",
            ));
        };
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(crate::OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    /// Replaces the complete live state from one validated snapshot; a failed
    /// restore leaves the current state untouched (SCE-00 D11).
    fn restore(&mut self, snapshot: &crate::OperatorStateSnapshot) -> Result<()> {
        if snapshot.inline_metadata.is_empty() && snapshot.segments.is_empty() {
            return StreamOperator::reset(self);
        }
        let metadata = parse_snapshot_metadata(snapshot)?;
        validate_snapshot_metadata(&metadata, &self.compiled, snapshot)?;
        let restored = self.decode_state(&metadata, snapshot)?;
        self.state = RollingStreamState {
            buffer: restored.buffer,
            histories: restored.histories,
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
        self.state = RollingStreamState::default();
        Ok(())
    }
}

impl RollingOperator {
    fn observe_context(&self, context: &StreamOperatorContext<'_>) -> Result<()> {
        if self
            .state
            .pipeline_fingerprint
            .as_deref()
            .is_some_and(|value| value != context.job().fingerprint())
        {
            return Err(operator_error(
                context.operator_id(),
                "rolling state was used with a different pipeline fingerprint",
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
                "rolling state was used with a different operator ID",
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

    /// Classifies one envelope into accepted rows and the late-metric delta
    /// without touching live state (SCE-00 D7 envelope transaction).
    fn classify_envelope(
        &self,
        rows: Vec<BufferedRow>,
        watermark: Option<EventTime>,
        node_id: &str,
    ) -> Result<(BTreeMap<RowIdentity, BufferedRow>, LateMetricDelta)> {
        let mut accepted = BTreeMap::new();
        let mut metrics = PreparedLateMetrics::default();
        for (row_index, row) in rows.into_iter().enumerate() {
            if self.is_late(row.identity.event_time, watermark, row_index, node_id)? {
                record_late_row(&mut metrics, watermark, row.identity.event_time, node_id)?;
                continue;
            }
            if self.state.buffer.contains_key(&row.identity) || accepted.contains_key(&row.identity)
            {
                return Err(operator_error(
                    node_id,
                    &format!(
                        "duplicate row identity at event_time_micros={}",
                        row.identity.event_time
                    ),
                ));
            }
            accepted.insert(row.identity.clone(), row);
        }
        Ok((accepted, metrics.into_delta()))
    }

    fn is_late(
        &self,
        event_time: i64,
        watermark: Option<EventTime>,
        row_index: usize,
        node_id: &str,
    ) -> Result<bool> {
        let Some(watermark) = watermark else {
            return Ok(false);
        };
        let closing = closing_coordinate(event_time, self.spec.allowed_lateness_micros, node_id)?;
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

    fn closing_keys(&self, watermark: i64, node_id: &str) -> Result<Vec<RowIdentity>> {
        let mut keys = Vec::new();
        for identity in self.state.buffer.keys() {
            let closing = closing_coordinate(
                identity.event_time,
                self.spec.allowed_lateness_micros,
                node_id,
            )?;
            if closing <= watermark {
                keys.push(identity.clone());
            }
        }
        Ok(keys)
    }

    fn take_buffered(&mut self, keys: &[RowIdentity]) -> Vec<BufferedRow> {
        keys.iter()
            .filter_map(|key| self.state.buffer.remove(key))
            .collect()
    }

    fn take_all_buffered(&mut self) -> Vec<BufferedRow> {
        std::mem::take(&mut self.state.buffer)
            .into_values()
            .collect()
    }

    async fn emit_rows(
        &mut self,
        rows: Vec<BufferedRow>,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        if rows.is_empty() {
            return Ok(());
        }
        let computed = compute_output_columns(
            &rows,
            &self.state.histories,
            &self.compiled,
            context.operator_id(),
        )?;
        let record = build_output_record(
            &rows,
            computed.columns,
            self.output_ports[0]
                .schema()
                .expect("rolling output always has an exact schema"),
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
        self.state.histories.apply(computed.touched);
        Ok(())
    }
}

impl RollingOperator {
    fn encode_state(&self, epoch: Epoch) -> Result<Option<(String, Vec<u8>)>> {
        let row_count = self
            .state
            .histories
            .by_entity
            .values()
            .map(VecDeque::len)
            .sum::<usize>()
            + self.state.buffer.len();
        if row_count == 0 {
            return Ok(None);
        }
        let pipeline_fingerprint =
            self.state.pipeline_fingerprint.clone().ok_or_else(|| {
                internal_error("rolling state is missing its pipeline fingerprint")
            })?;
        let operator_id = self
            .state
            .operator_id
            .clone()
            .ok_or_else(|| internal_error("rolling state is missing its operator identity"))?;
        let segment_id = format!("base-{:020}-00000000", epoch.as_u64());
        let bytes = encode_state_segment(
            &self.state.histories,
            &self.state.buffer,
            self.input_ports[0]
                .schema()
                .expect("rolling input always has an exact schema"),
            &self.compiled,
            &pipeline_fingerprint,
            &operator_id,
        )?;
        Ok(Some((segment_id, bytes)))
    }

    fn snapshot_segment_descriptor(
        &self,
        epoch: Epoch,
        prepared: &(String, Vec<u8>),
    ) -> Result<SegmentDescriptor> {
        let operator_id = self.state.operator_id.as_deref().ok_or_else(|| {
            checkpoint_mismatch("rolling segment is missing its operator identity".into())
        })?;
        let (segment_id, bytes) = prepared;
        let relative_path = format!(
            "committed/{operator_id}/{:020}-{segment_id}.arrow",
            epoch.as_u64()
        );
        let byte_len = u64::try_from(bytes.len())
            .map_err(|_| internal_error("rolling segment length does not fit u64"))?;
        let sha256 = hex::encode(Sha256::digest(bytes));
        Ok(SegmentDescriptor {
            kind: SegmentKind::Base,
            state_layout_version: ROLLING_STATE_LAYOUT_VERSION,
            schema_fingerprint: self.compiled.state_schema_fingerprint.clone(),
            handle: StateHandle::new(
                operator_id,
                epoch,
                segment_id,
                &relative_path,
                byte_len,
                &sha256,
            )?,
        })
    }

    fn decode_state(
        &self,
        metadata: &RollingSnapshotMetadata,
        snapshot: &crate::OperatorStateSnapshot,
    ) -> Result<DecodedRollingState> {
        let segments = snapshot_segments(snapshot, &metadata.segment_inventory)?;
        let Some(bytes) = segments.into_iter().next() else {
            return Ok(DecodedRollingState::default());
        };
        decode_state_segment(
            &bytes,
            self.input_ports[0]
                .schema()
                .expect("rolling input always has an exact schema"),
            &self.compiled,
            metadata,
        )
    }
}

#[derive(Default)]
struct DecodedRollingState {
    buffer: BTreeMap<RowIdentity, BufferedRow>,
    histories: RollingHistories,
}

/// Serialized state-row ordering key: row kind, entity, identity, and the
/// per-entity history position.
type StateRowOrderKey = (u8, Vec<Option<KeyValue>>, RowIdentity, Option<u64>);

fn state_fields(input_schema: &Schema) -> Vec<Field> {
    let mut fields = vec![
        Field::new("_state_kind", DataType::UInt8, false),
        Field::new("_entity_position", DataType::UInt64, true),
    ];
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
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Schema {
    Schema::new_with_metadata(
        state_fields(input_schema),
        HashMap::from([
            (
                "calc_flow.state_layout_version".into(),
                ROLLING_STATE_LAYOUT_VERSION.to_string(),
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

fn encode_state_segment(
    histories: &RollingHistories,
    buffer: &BTreeMap<RowIdentity, BufferedRow>,
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    pipeline_fingerprint: &str,
    operator_id: &str,
) -> Result<Vec<u8>> {
    let width = input_schema.fields().len();
    let mut kinds = Vec::new();
    let mut positions: Vec<Option<u64>> = Vec::new();
    let mut columns: Vec<Vec<Option<ScalarValue>>> = vec![Vec::new(); width];
    let mut push_row = |kind: u8, position: Option<u64>, values: &[ScalarValue]| {
        kinds.push(kind);
        positions.push(position);
        for (index, column) in columns.iter_mut().enumerate() {
            column.push(values.get(index).cloned());
        }
    };
    for history in histories.by_entity.values() {
        for (position, values) in history.iter().enumerate() {
            let position = u64::try_from(position)
                .map_err(|_| internal_error("rolling history position does not fit u64"))?;
            push_row(0, Some(position), values);
        }
    }
    for row in buffer.values() {
        push_row(1, None, &row.values);
    }
    let schema = state_schema(input_schema, compiled, pipeline_fingerprint, operator_id);
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(UInt8Array::from(kinds)),
        Arc::new(UInt64Array::from(positions)),
    ];
    for column in columns {
        arrays.push(
            ScalarValue::iter_to_array(
                column
                    .into_iter()
                    .map(|value| value.expect("rolling state rows carry full typed values")),
            )
            .map_err(|error| state_format(format!("rolling state array failed: {error}")))?,
        );
    }
    let record = RecordBatch::try_new(Arc::new(schema.clone()), arrays)
        .map_err(|error| state_format(format!("rolling state batch is invalid: {error}")))?;
    let mut bytes = Vec::new();
    {
        let mut writer = FileWriter::try_new(&mut bytes, &schema)
            .map_err(|error| state_format(format!("rolling state IPC header failed: {error}")))?;
        writer
            .write(&record)
            .map_err(|error| state_format(format!("rolling state IPC write failed: {error}")))?;
        writer
            .finish()
            .map_err(|error| state_format(format!("rolling state IPC finish failed: {error}")))?;
    }
    Ok(bytes)
}

fn decode_state_segment(
    bytes: &[u8],
    input_schema: &Schema,
    compiled: &CompiledRollingSpec,
    metadata: &RollingSnapshotMetadata,
) -> Result<DecodedRollingState> {
    let reader = FileReader::try_new(Cursor::new(bytes), None)
        .map_err(|error| state_format(format!("rolling state IPC open failed: {error}")))?;
    validate_segment_schema_metadata(reader.schema().metadata(), metadata, compiled)?;
    let batches = reader
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| state_format(format!("rolling state IPC read failed: {error}")))?;
    let [record] = batches.try_into().map_err(|_| {
        state_format("rolling state segment must contain exactly one record batch".to_owned())
    })?;
    let width = input_schema.fields().len();
    if record.num_columns() != width + 2 {
        return Err(state_format(
            "rolling state segment column count does not match the state schema".to_owned(),
        ));
    }
    let kinds = record
        .column(0)
        .as_any()
        .downcast_ref::<UInt8Array>()
        .ok_or_else(|| state_format("rolling state kind column has the wrong type".to_owned()))?;
    let positions = record
        .column(1)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| {
            state_format("rolling state position column has the wrong type".to_owned())
        })?;
    let mut decoded = DecodedRollingState::default();
    let mut previous: Option<StateRowOrderKey> = None;
    for row_index in 0..record.num_rows() {
        let values = (2..record.num_columns())
            .map(|index| {
                ScalarValue::try_from_array(record.column(index), row_index).map_err(|error| {
                    state_format(format!("rolling state row could not be read: {error}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let position = positions.iter().nth(row_index).flatten();
        decode_state_row(
            kinds.value(row_index),
            position,
            values,
            &mut decoded,
            compiled,
            &mut previous,
        )?;
    }
    validate_decoded_state(&decoded, compiled)?;
    Ok(decoded)
}

fn validate_segment_schema_metadata(
    metadata: &HashMap<String, String>,
    snapshot: &RollingSnapshotMetadata,
    compiled: &CompiledRollingSpec,
) -> Result<()> {
    let expected = [
        (
            "calc_flow.state_layout_version",
            ROLLING_STATE_LAYOUT_VERSION.to_string(),
        ),
        (
            "calc_flow.pipeline_fingerprint",
            snapshot.pipeline_fingerprint.clone().unwrap_or_default(),
        ),
        (
            "calc_flow.operator_id",
            snapshot.operator_id.clone().unwrap_or_default(),
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
                "rolling state segment metadata {key} does not match the snapshot"
            )));
        }
    }
    Ok(())
}

fn decode_state_row(
    kind: u8,
    position: Option<u64>,
    values: Vec<ScalarValue>,
    decoded: &mut DecodedRollingState,
    compiled: &CompiledRollingSpec,
    previous: &mut Option<StateRowOrderKey>,
) -> Result<()> {
    let row = buffered_row_from_values(values, compiled)?;
    let ordering_key = (
        kind,
        row.identity.entity.clone(),
        row.identity.clone(),
        position,
    );
    if let Some(prior) = previous.as_ref()
        && !state_rows_in_order(prior, &ordering_key)
    {
        return Err(state_format(
            "rolling state segment rows are not in deterministic key order".to_owned(),
        ));
    }
    match kind {
        0 => {
            let history = decoded
                .histories
                .by_entity
                .entry(row.identity.entity.clone())
                .or_default();
            let expected = u64::try_from(history.len()).unwrap_or(u64::MAX);
            if position != Some(expected) {
                return Err(state_format(
                    "rolling state segment history positions are not contiguous".to_owned(),
                ));
            }
            history.push_back(row.values);
        }
        1 => {
            if decoded.buffer.insert(row.identity.clone(), row).is_some() {
                return Err(state_format(
                    "rolling state segment contains a duplicate buffered identity".to_owned(),
                ));
            }
        }
        other => {
            return Err(state_format(format!(
                "rolling state segment contains unknown row kind {other}"
            )));
        }
    }
    *previous = Some(ordering_key);
    Ok(())
}

fn state_rows_in_order(prior: &StateRowOrderKey, current: &StateRowOrderKey) -> bool {
    if prior.0 != current.0 {
        return prior.0 < current.0;
    }
    match prior.0 {
        0 => {
            if prior.1 != current.1 {
                return prior.1 < current.1;
            }
            match (prior.3, current.3) {
                (Some(left), Some(right)) => left < right,
                _ => false,
            }
        }
        _ => prior.2 < current.2,
    }
}

fn buffered_row_from_values(
    values: Vec<ScalarValue>,
    compiled: &CompiledRollingSpec,
) -> Result<BufferedRow> {
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(state_format(
                "rolling state row has a null or non-timestamp event time".to_owned(),
            ));
        }
    };
    let entity = compiled
        .partition_columns
        .iter()
        .map(|column| KeyValue::from_nullable_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    let sequence = compiled
        .sequence_columns
        .iter()
        .map(|column| KeyValue::from_required_scalar(&values[column.index], "rolling"))
        .collect::<Result<Vec<_>>>()?;
    Ok(BufferedRow::new(entity, sequence, event_time, values))
}

fn validate_decoded_state(
    decoded: &DecodedRollingState,
    compiled: &CompiledRollingSpec,
) -> Result<()> {
    let max_periods = usize::try_from(compiled.max_periods)
        .map_err(|_| internal_error("rolling max periods does not fit usize"))?;
    for history in decoded.histories.by_entity.values() {
        if history.len() > max_periods {
            return Err(state_format(
                "rolling state segment retains more history than the declared frames".to_owned(),
            ));
        }
    }
    Ok(())
}

fn parse_snapshot_metadata(
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<RollingSnapshotMetadata> {
    serde_json::from_value::<RollingSnapshotMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| format_error(&error))
}

fn validate_snapshot_metadata(
    metadata: &RollingSnapshotMetadata,
    compiled: &CompiledRollingSpec,
    snapshot: &crate::OperatorStateSnapshot,
) -> Result<StateInventory> {
    if metadata.state_layout_version != ROLLING_STATE_LAYOUT_VERSION {
        return Err(checkpoint_mismatch(format!(
            "rolling state layout version {} does not match expected {}",
            metadata.state_layout_version, ROLLING_STATE_LAYOUT_VERSION
        )));
    }
    if metadata.configuration_hash != compiled.configuration_hash {
        return Err(checkpoint_mismatch(
            "rolling operator configuration hash does not match the compiled operator".into(),
        ));
    }
    if metadata.state_schema_fingerprint != compiled.state_schema_fingerprint {
        return Err(checkpoint_mismatch(
            "rolling state schema fingerprint does not match the compiled operator".into(),
        ));
    }
    let inventory = StateInventory::new(metadata.segment_inventory.clone())
        .map_err(|error| checkpoint_mismatch(error.to_string()))?;
    for descriptor in inventory.segments() {
        if descriptor.state_layout_version != ROLLING_STATE_LAYOUT_VERSION
            || descriptor.schema_fingerprint != compiled.state_schema_fingerprint
        {
            return Err(checkpoint_mismatch(
                "rolling segment inventory layout or schema does not match the compiled operator"
                    .into(),
            ));
        }
        if descriptor.handle.epoch() > metadata.epoch {
            return Err(checkpoint_mismatch(
                "rolling segment inventory contains a future epoch".into(),
            ));
        }
        if metadata.operator_id.as_deref() != Some(descriptor.handle.operator_id()) {
            return Err(checkpoint_mismatch(
                "rolling segment inventory operator does not match snapshot metadata".into(),
            ));
        }
    }
    let expected_ids = inventory
        .segments()
        .iter()
        .map(|descriptor| descriptor.handle.segment_id().to_owned())
        .collect::<Vec<_>>();
    let actual_ids = snapshot.segments.keys().cloned().collect::<Vec<_>>();
    if expected_ids != actual_ids {
        return Err(checkpoint_mismatch(
            "rolling snapshot segment IDs are missing, extra, duplicated, or non-canonical".into(),
        ));
    }
    if !snapshot.segments.is_empty()
        && (metadata.pipeline_fingerprint.is_none() || metadata.operator_id.is_none())
    {
        return Err(checkpoint_mismatch(
            "rolling segments require pipeline and operator identity metadata".into(),
        ));
    }
    if let Some(fingerprint) = metadata.pipeline_fingerprint.as_deref()
        && (fingerprint.len() != 64
            || !fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    {
        return Err(checkpoint_mismatch(
            "rolling pipeline fingerprint is not lowercase SHA-256".into(),
        ));
    }
    if metadata
        .operator_id
        .as_deref()
        .is_some_and(|operator_id| operator_id.is_empty() || operator_id.contains('\0'))
    {
        return Err(checkpoint_mismatch(
            "rolling operator ID is empty or contains NUL".into(),
        ));
    }
    Ok(inventory)
}

fn snapshot_segments(
    snapshot: &crate::OperatorStateSnapshot,
    inventory: &[SegmentDescriptor],
) -> Result<Vec<Vec<u8>>> {
    inventory
        .iter()
        .map(|descriptor| {
            let segment_id = descriptor.handle.segment_id();
            let bytes = snapshot.segments.get(segment_id).cloned().ok_or_else(|| {
                checkpoint_mismatch(format!(
                    "rolling snapshot is missing segment {segment_id:?}"
                ))
            })?;
            if u64::try_from(bytes.len()).ok() != Some(descriptor.handle.byte_len()) {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment byte length does not match its handle".into(),
                ));
            }
            if hex::encode(Sha256::digest(&bytes)) != descriptor.handle.sha256() {
                return Err(checkpoint_mismatch(
                    "rolling snapshot segment checksum does not match its handle".into(),
                ));
            }
            Ok(bytes)
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

fn closing_coordinate(event_time: i64, allowed_lateness_micros: u64, node_id: &str) -> Result<i64> {
    let lateness = i64::try_from(allowed_lateness_micros).map_err(|_| {
        operator_error(
            node_id,
            "allowed lateness exceeds the representable event-time range",
        )
    })?;
    event_time.checked_add(lateness).ok_or_else(|| {
        operator_error(
            node_id,
            "finality coordinate overflowed the event-time range",
        )
    })
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
                        "one rolling output row requires {row_bytes} bytes, exceeding the effective edge byte budget {}",
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
                "validated rolling output row did not fit the effective edge budget",
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
fn read_buffered_rows(
    table: &TableBatch,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<Vec<BufferedRow>> {
    let mut rows = Vec::with_capacity(table.batches().iter().map(RecordBatch::num_rows).sum());
    for record in table.batches() {
        for row_index in 0..record.num_rows() {
            rows.push(read_buffered_row(record, row_index, compiled, node_id)?);
        }
    }
    Ok(rows)
}

fn read_buffered_row(
    record: &RecordBatch,
    row_index: usize,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<BufferedRow> {
    let mut values = Vec::with_capacity(record.num_columns());
    for column in record.columns() {
        values.push(
            ScalarValue::try_from_array(column, row_index).map_err(|error| {
                operator_error(
                    node_id,
                    &format!("rolling input row could not be read: {error}"),
                )
            })?,
        );
    }
    let event_time = match &values[compiled.event_time_index] {
        ScalarValue::TimestampMicrosecond(Some(value), _) => *value,
        _ => {
            return Err(operator_error(
                node_id,
                "rolling event-time value is null or not a microsecond timestamp",
            ));
        }
    };
    let entity = compiled
        .partition_columns
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

/// Sorts accepted rows into the canonical observable order and rejects
/// duplicate identities before any output is produced (SCE-00 D4).
fn sort_and_validate(mut rows: Vec<BufferedRow>, node_id: &str) -> Result<Vec<BufferedRow>> {
    rows.sort_by(|left, right| left.identity.cmp(&right.identity));
    if let Some(duplicate) = rows
        .windows(2)
        .find(|pair| pair[0].identity == pair[1].identity)
    {
        return Err(operator_error(
            node_id,
            &format!(
                "duplicate row identity at event_time_micros={}",
                duplicate[0].identity.event_time
            ),
        ));
    }
    Ok(rows)
}

/// Builds one output record: canonical-order input columns followed by the
/// derived rolling outputs (SCE-00 D5).
fn build_output_record(
    rows: &[BufferedRow],
    derived: Vec<ArrayRef>,
    output_schema: &SchemaRef,
    node_id: &str,
) -> Result<RecordBatch> {
    let input_width = rows.first().map_or_else(
        || output_schema.fields().len() - derived.len(),
        |row| row.values.len(),
    );
    let mut columns = Vec::with_capacity(input_width + derived.len());
    for index in 0..input_width {
        if rows.is_empty() {
            columns.push(new_null_array(output_schema.field(index).data_type(), 0));
            continue;
        }
        columns.push(
            ScalarValue::iter_to_array(rows.iter().map(|row| row.values[index].clone())).map_err(
                |error| {
                    operator_error(
                        node_id,
                        &format!("rolling output row encoding failed: {error}"),
                    )
                },
            )?,
        );
    }
    columns.extend(derived);
    RecordBatch::try_new(Arc::clone(output_schema), columns).map_err(|error| {
        operator_error(
            node_id,
            &format!("rolling output record is invalid: {error}"),
        )
    })
}

#[derive(Clone)]
struct CompiledRollingSpec {
    event_time_index: usize,
    partition_columns: Vec<CompiledKeyColumn>,
    sequence_columns: Vec<CompiledKeyColumn>,
    outputs: Vec<CompiledRollingOutput>,
    max_periods: u64,
    configuration_hash: String,
    state_schema_fingerprint: String,
}

#[derive(Clone)]
struct CompiledKeyColumn {
    index: usize,
}

#[derive(Clone)]
struct CompiledRollingOutput {
    input_index: usize,
    name: String,
    input_type: DataType,
    output_type: DataType,
    is_delta: bool,
    periods: u64,
}

/// One entity or sequence key component in the Arrow total order (null
/// before non-null); floats compare with the IEEE total order (SCE-00 D4).
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
                        "rolling key column has unsupported value type {}",
                        other.data_type()
                    ),
                ));
            }
        };
        value.ok_or_else(|| operator_error(node_id, "rolling sequence key value is null"))
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

/// Per-entity retained tails of the last `max_periods` referenced values.
#[derive(Clone, Debug, Default)]
struct RollingHistories {
    by_entity: BTreeMap<Vec<Option<KeyValue>>, VecDeque<Vec<ScalarValue>>>,
}

/// Kernel-produced per-entity history replacements (entity key, new tail).
type HistoryUpdates = Vec<(Vec<Option<KeyValue>>, VecDeque<Vec<ScalarValue>>)>;

impl RollingHistories {
    fn apply(&mut self, touched: HistoryUpdates) {
        for (entity, history) in touched {
            self.by_entity.insert(entity, history);
        }
    }
}

/// Kernel result: derived output columns plus the per-entity history updates
/// the caller installs only after complete success (transactional state).
#[derive(Debug)]
struct ComputedOutputs {
    columns: Vec<ArrayRef>,
    touched: HistoryUpdates,
}

/// Computes every declared rolling output over `rows` in canonical order,
/// reading entity histories without mutating them (SCE-00 D5: batch and
/// stream lifecycles share this kernel and row order).
fn compute_output_columns(
    rows: &[BufferedRow],
    histories: &RollingHistories,
    compiled: &CompiledRollingSpec,
    node_id: &str,
) -> Result<ComputedOutputs> {
    if rows.is_empty() {
        return Ok(ComputedOutputs {
            columns: compiled
                .outputs
                .iter()
                .map(|output| new_null_array(&output.output_type, 0))
                .collect(),
            touched: Vec::new(),
        });
    }
    let mut entities: BTreeMap<&Vec<Option<KeyValue>>, Vec<usize>> = BTreeMap::new();
    for (index, row) in rows.iter().enumerate() {
        entities
            .entry(&row.identity.entity)
            .or_default()
            .push(index);
    }
    let mut derived: Vec<Vec<Option<ScalarValue>>> = compiled
        .outputs
        .iter()
        .map(|_| vec![None; rows.len()])
        .collect();
    let mut touched = Vec::with_capacity(entities.len());
    for (entity, indices) in entities {
        let history = histories.by_entity.get(entity).cloned().unwrap_or_default();
        let mut next_history = history.clone();
        for (position, &row_index) in indices.iter().enumerate() {
            for (ordinal, output) in compiled.outputs.iter().enumerate() {
                derived[ordinal][row_index] = Some(compute_output_value(
                    rows, &indices, &history, position, row_index, output, node_id,
                )?);
            }
        }
        for &row_index in &indices {
            next_history.push_back(rows[row_index].values.clone());
        }
        while next_history.len() > usize::try_from(compiled.max_periods).unwrap_or(usize::MAX) {
            next_history.pop_front();
        }
        touched.push((entity.clone(), next_history));
    }
    let columns = derived
        .into_iter()
        .zip(&compiled.outputs)
        .map(|(column, output)| {
            ScalarValue::iter_to_array(
                column
                    .into_iter()
                    .map(|value| value.unwrap_or_else(|| typed_null(&output.input_type))),
            )
            .map_err(|error| {
                operator_error(
                    node_id,
                    &format!("rolling output column encoding failed: {error}"),
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ComputedOutputs { columns, touched })
}

fn compute_output_value(
    rows: &[BufferedRow],
    indices: &[usize],
    history: &VecDeque<Vec<ScalarValue>>,
    position: usize,
    row_index: usize,
    output: &CompiledRollingOutput,
    node_id: &str,
) -> Result<ScalarValue> {
    let periods = usize::try_from(output.periods)
        .map_err(|_| operator_error(node_id, "rolling periods does not fit usize"))?;
    let referenced = if position + history.len() < periods {
        None
    } else if position >= periods {
        Some(rows[indices[position - periods]].values[output.input_index].clone())
    } else {
        Some(history[history.len() + position - periods][output.input_index].clone())
    };
    if !output.is_delta {
        return Ok(referenced.unwrap_or_else(|| typed_null(&output.input_type)));
    }
    let current = &rows[row_index].values[output.input_index];
    if current.is_null() {
        return Ok(typed_null(&output.input_type));
    }
    let Some(reference) = referenced.filter(|value| !value.is_null()) else {
        return Ok(typed_null(&output.input_type));
    };
    current.sub_checked(&reference).map_err(|error| {
        operator_error(
            node_id,
            &format!("rolling delta failed with checked arithmetic: {error}"),
        )
    })
}

fn typed_null(data_type: &DataType) -> ScalarValue {
    ScalarValue::try_from(data_type).unwrap_or(ScalarValue::Null)
}

fn validate_arguments(spec: &RollingSpec) -> Result<()> {
    if spec.configuration_version != ROLLING_CONFIGURATION_VERSION {
        return Err(invalid_argument(
            "rolling.configuration_version",
            "unsupported rolling configuration version",
        ));
    }
    if spec.state_layout_version != ROLLING_STATE_LAYOUT_VERSION {
        return Err(invalid_argument(
            "rolling.state_layout_version",
            "unsupported rolling state layout version",
        ));
    }
    validate_key_names("rolling.partition_by", &spec.partition_by)?;
    validate_key_names("rolling.sequence_by", &spec.sequence_by)?;
    validate_outputs(&spec.outputs)?;
    if let LatePolicySpec::Drop { metrics_version } = spec.late_policy
        && metrics_version != 1
    {
        return Err(invalid_argument(
            "rolling.late_policy.metrics_version",
            "unsupported late-metrics version",
        ));
    }
    Ok(())
}

fn validate_key_names(field: &str, columns: &[String]) -> Result<()> {
    if columns.is_empty() {
        return Err(invalid_argument(field, "must not be empty"));
    }
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

fn validate_outputs(outputs: &[RollingOutputSpec]) -> Result<()> {
    if outputs.is_empty() {
        return Err(invalid_argument("rolling.outputs", "must not be empty"));
    }
    for (index, output) in outputs.iter().enumerate() {
        let base = format!("rolling.outputs[{index}]");
        if output.primitive_version() != 1 {
            return Err(invalid_argument(
                &format!("{base}.primitive_version"),
                "unsupported rolling primitive version",
            ));
        }
        if output.periods() == 0 {
            return Err(invalid_argument(
                &format!("{base}.periods"),
                "must be greater than zero",
            ));
        }
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
        if outputs[..index]
            .iter()
            .any(|earlier| earlier.output() == output.output())
        {
            return Err(invalid_argument(
                &format!("{base}.output"),
                "duplicates an earlier rolling output",
            ));
        }
    }
    Ok(())
}

fn compile_spec(spec: &RollingSpec, input_schema: &Schema) -> Result<CompiledRollingSpec> {
    compile_spec_against_schema(spec, input_schema, String::new())
}

fn compile_spec_full(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration: &JsonMap,
) -> Result<CompiledRollingSpec> {
    let canonical = canonical_json(&Value::Object(configuration.clone().into_iter().collect()))?;
    let configuration_hash = hex::encode(Sha256::digest(canonical.as_bytes()));
    compile_spec_against_schema(spec, input_schema, configuration_hash)
}

fn compile_spec_against_schema(
    spec: &RollingSpec,
    input_schema: &Schema,
    configuration_hash: String,
) -> Result<CompiledRollingSpec> {
    let event_time_index = exact_field_index(input_schema, &spec.event_time)?;
    validate_event_time(input_schema, event_time_index, &spec.event_time)?;
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
    let max_periods = spec
        .outputs
        .iter()
        .map(RollingOutputSpec::periods)
        .max()
        .unwrap_or(1);
    Ok(CompiledRollingSpec {
        event_time_index,
        partition_columns,
        sequence_columns,
        outputs,
        max_periods,
        configuration_hash,
        state_schema_fingerprint: state_schema_fingerprint(input_schema),
    })
}

#[derive(Clone, Copy)]
enum KeyRole {
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
        KeyRole::Partition => {
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling partition column {column:?} has unsupported type {data_type}"
                )));
            }
        }
        KeyRole::Sequence => {
            if field.is_nullable() {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must be non-nullable"
                )));
            }
            if matches!(data_type, DataType::Float32 | DataType::Float64) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} must not use a floating type"
                )));
            }
            if !supports_total_order(&data_type) {
                return Err(compile_error(format!(
                    "rolling sequence column {column:?} has unsupported type {data_type}"
                )));
            }
        }
    }
    Ok(CompiledKeyColumn { index })
}

fn compile_output(
    input_schema: &Schema,
    output: &RollingOutputSpec,
    ordinal: usize,
) -> Result<CompiledRollingOutput> {
    if input_schema
        .fields()
        .iter()
        .any(|field| field.name() == output.output())
    {
        return Err(invalid_argument(
            &format!("rolling.outputs[{ordinal}].output"),
            "collides with an input field name",
        ));
    }
    let input_index = exact_field_index(input_schema, output.input())?;
    let input_type = input_schema.field(input_index).data_type().clone();
    if output.is_delta() && !is_numeric(&input_type) {
        return Err(compile_error(format!(
            "rolling delta does not support column {:?} with type {input_type}",
            output.input()
        )));
    }
    Ok(CompiledRollingOutput {
        input_index,
        name: output.output().to_owned(),
        output_type: input_type.clone(),
        input_type,
        is_delta: output.is_delta(),
        periods: output.periods(),
    })
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
            "rolling column {column:?} does not exist in the input schema"
        ))),
        _ => Err(compile_error(format!(
            "rolling column {column:?} is ambiguous in the input schema"
        ))),
    }
}

fn validate_event_time(schema: &Schema, index: usize, column: &str) -> Result<()> {
    let field = schema.field(index);
    if field.is_nullable() {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be non-nullable"
        )));
    }
    if !matches!(
        field.data_type(),
        DataType::Timestamp(TimeUnit::Microsecond, Some(timezone)) if timezone.as_ref() == "UTC"
    ) {
        return Err(compile_error(format!(
            "rolling event-time column {column:?} must be a non-null UTC timestamp[us], found {}",
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

fn output_schema(input_schema: &Schema, outputs: &[CompiledRollingOutput]) -> Schema {
    let mut fields = input_schema.fields().to_vec();
    fields.extend(
        outputs
            .iter()
            .map(|output| Field::new(&output.name, output.output_type.clone(), true).into()),
    );
    Schema::new(fields)
}

fn configuration(spec: &RollingSpec) -> Result<JsonMap> {
    let spec_json = serde_json::to_value(spec).map_err(|error| format_error(&error))?;
    Ok(JsonMap::from([
        ("kind".into(), json!("rolling")),
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

fn checkpoint_mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}

fn internal_error(message: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: message.into(),
    }
}

fn state_format(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

fn compile_error(message: String) -> CalcFlowError {
    CalcFlowError::Compile { message }
}

fn format_error(error: &serde_json::Error) -> CalcFlowError {
    CalcFlowError::Format {
        message: error.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::arrow::datatypes::{DataType, Field, Schema, TimeUnit};
    use serde_json::{Value, json};

    use super::*;
    use crate::{CalcFlowError, OperatorMetadata};

    fn input_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ])
    }

    fn valid_spec_json() -> Value {
        json!({
            "configuration_version": 1,
            "state_layout_version": 1,
            "partition_by": ["symbol"],
            "event_time": "ts",
            "sequence_by": ["sequence"],
            "outputs": [
                {
                    "kind": "lag",
                    "primitive_version": 1,
                    "input": "price",
                    "output": "price_lag_1",
                    "periods": 1
                },
                {
                    "kind": "delta",
                    "primitive_version": 1,
                    "input": "volume",
                    "output": "volume_delta_1",
                    "periods": 1
                }
            ],
            "allowed_lateness_micros": 0,
            "late_policy": {"kind": "error", "scope": "envelope"},
            "value_policy": "stateful_numeric_v1"
        })
    }

    fn valid_spec() -> RollingSpec {
        serde_json::from_value(valid_spec_json()).unwrap()
    }

    fn with_field(schema: &Schema, index: usize, replacement: &Field) -> Schema {
        let fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(position, field)| {
                if position == index {
                    replacement.clone().into()
                } else {
                    field.clone()
                }
            })
            .collect::<Vec<_>>();
        Schema::new(fields)
    }

    // ------------------------------------------------------------------
    // Strict serialized model
    // ------------------------------------------------------------------

    #[test]
    fn canonical_lag_delta_spec_round_trips_the_frozen_json() {
        let spec: RollingSpec = serde_json::from_value(valid_spec_json()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), valid_spec_json());
    }

    #[test]
    fn drop_late_policy_uses_the_exact_frozen_shape() {
        let mut document = valid_spec_json();
        document["late_policy"] = json!({"kind": "drop", "metrics_version": 1});
        let spec: RollingSpec = serde_json::from_value(document.clone()).unwrap();
        assert_eq!(serde_json::to_value(&spec).unwrap(), document);
    }

    #[test]
    fn unknown_spec_field_is_rejected() {
        let mut document = valid_spec_json();
        document["unexpected"] = json!(true);
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn missing_semantic_field_is_rejected() {
        let mut document = valid_spec_json();
        document.as_object_mut().unwrap().remove("value_policy");
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn unknown_output_kind_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][0] = json!({
            "kind": "mean",
            "primitive_version": 1,
            "input": "price",
            "output": "price_mean_20",
            "frame": {"kind": "rows", "size": 20},
            "min_periods": 1
        });
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn lag_output_rejects_aggregate_only_fields() {
        for field in ["frame", "min_periods", "ddof"] {
            let mut document = valid_spec_json();
            document["outputs"][0][field] = json!(1);
            assert!(
                serde_json::from_value::<RollingSpec>(document).is_err(),
                "lag accepted aggregate-only field {field}"
            );
        }
    }

    #[test]
    fn unknown_value_policy_is_rejected() {
        let mut document = valid_spec_json();
        document["value_policy"] = json!("lenient");
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    #[test]
    fn error_late_policy_rejects_metrics_version_and_drop_rejects_scope() {
        let mut document = valid_spec_json();
        document["late_policy"] =
            json!({"kind": "error", "scope": "envelope", "metrics_version": 1});
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
        let mut document = valid_spec_json();
        document["late_policy"] =
            json!({"kind": "drop", "metrics_version": 1, "scope": "envelope"});
        assert!(serde_json::from_value::<RollingSpec>(document).is_err());
    }

    // ------------------------------------------------------------------
    // Declaration and schema validation
    // ------------------------------------------------------------------

    #[test]
    fn valid_spec_derives_the_output_schema() {
        let output_schema = valid_spec().validate(&input_schema()).unwrap();
        let expected = Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
            Field::new("price_lag_1", DataType::Float64, true),
            Field::new("volume_delta_1", DataType::Int64, true),
        ]);
        assert_eq!(output_schema.as_ref(), &expected);
    }

    #[test]
    fn unsupported_configuration_version_is_rejected() {
        let mut spec = valid_spec();
        spec.configuration_version = 2;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.configuration_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unsupported_state_layout_version_is_rejected() {
        let mut spec = valid_spec();
        spec.state_layout_version = 0;
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.state_layout_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn empty_partition_by_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.partition_by"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn duplicate_partition_column_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = vec!["symbol".into(), "symbol".into()];
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.partition_by[1]"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_partition_column_is_rejected() {
        let mut spec = valid_spec();
        spec.partition_by = vec!["industry".into()];
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn unsupported_partition_column_type_is_rejected() {
        let schema = with_field(
            &input_schema(),
            1,
            &Field::new("symbol", DataType::LargeBinary, false),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_event_time_column_is_rejected() {
        let mut spec = valid_spec();
        spec.event_time = "event_ts".into();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn nullable_event_time_is_rejected() {
        let schema = with_field(
            &input_schema(),
            0,
            &Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                true,
            ),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn non_utc_or_coarse_event_time_is_rejected() {
        for data_type in [
            DataType::Timestamp(TimeUnit::Microsecond, None),
            DataType::Timestamp(TimeUnit::Millisecond, Some(Arc::from("UTC"))),
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("Asia/Shanghai"))),
            DataType::Int64,
        ] {
            let schema = with_field(
                &input_schema(),
                0,
                &Field::new("ts", data_type.clone(), false),
            );
            let error = valid_spec().validate(&schema).unwrap_err();
            assert!(
                matches!(error, CalcFlowError::Compile { .. }),
                "event-time type {data_type} was accepted"
            );
        }
    }

    #[test]
    fn empty_sequence_by_is_rejected() {
        let mut spec = valid_spec();
        spec.sequence_by = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.sequence_by"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn nullable_sequence_column_is_rejected() {
        let schema = with_field(
            &input_schema(),
            2,
            &Field::new("sequence", DataType::UInt64, true),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn floating_sequence_column_is_rejected() {
        let schema = with_field(
            &input_schema(),
            2,
            &Field::new("sequence", DataType::Float64, false),
        );
        let error = valid_spec().validate(&schema).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn empty_outputs_are_rejected() {
        let mut spec = valid_spec();
        spec.outputs = Vec::new();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn zero_periods_is_rejected() {
        for index in [0, 1] {
            let mut document = valid_spec_json();
            document["outputs"][index]["periods"] = json!(0);
            let spec: RollingSpec = serde_json::from_value(document).unwrap();
            let error = spec.validate(&input_schema()).unwrap_err();
            assert!(
                matches!(
                    error,
                    CalcFlowError::InvalidArgument { ref field, .. }
                        if field == &format!("rolling.outputs[{index}].periods")
                ),
                "unexpected error: {error}"
            );
        }
    }

    #[test]
    fn unsupported_primitive_version_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][0]["primitive_version"] = json!(2);
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[0].primitive_version"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn missing_output_input_column_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][0]["input"] = json!("close");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn delta_on_non_numeric_column_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["input"] = json!("label");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Compile { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn lag_preserves_any_input_type() {
        let mut document = valid_spec_json();
        document["outputs"][0]["input"] = json!("label");
        document["outputs"][0]["output"] = json!("label_lag_1");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let output_schema = spec.validate(&input_schema()).unwrap();
        assert_eq!(
            output_schema.field_with_name("label_lag_1").unwrap(),
            &Field::new("label_lag_1", DataType::Utf8, true)
        );
    }

    #[test]
    fn duplicate_output_name_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["output"] = json!("price_lag_1");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[1].output"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn output_name_colliding_with_an_input_field_is_rejected() {
        let mut document = valid_spec_json();
        document["outputs"][1]["output"] = json!("volume");
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.outputs[1].output"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn drop_metrics_version_must_be_one() {
        let mut document = valid_spec_json();
        document["late_policy"] = json!({"kind": "drop", "metrics_version": 2});
        let spec: RollingSpec = serde_json::from_value(document).unwrap();
        let error = spec.validate(&input_schema()).unwrap_err();
        assert!(
            matches!(
                error,
                CalcFlowError::InvalidArgument { ref field, .. }
                    if field == "rolling.late_policy.metrics_version"
            ),
            "unexpected error: {error}"
        );
    }

    // ------------------------------------------------------------------
    // Shared lag/delta kernel
    // ------------------------------------------------------------------

    fn ts_scalar(value: i64) -> ScalarValue {
        ScalarValue::TimestampMicrosecond(Some(value), Some(Arc::from("UTC")))
    }

    fn full_row(
        event_time: i64,
        symbol: &str,
        sequence: u64,
        rest: Vec<ScalarValue>,
    ) -> BufferedRow {
        let mut values = vec![
            ts_scalar(event_time),
            ScalarValue::Utf8(Some(symbol.into())),
            ScalarValue::UInt64(Some(sequence)),
        ];
        values.extend(rest);
        while values.len() < 6 {
            values.push(match values.len() {
                3 => ScalarValue::Float64(None),
                4 => ScalarValue::Int64(None),
                _ => ScalarValue::Utf8(None),
            });
        }
        BufferedRow::new(
            vec![Some(KeyValue::String(symbol.into()))],
            vec![KeyValue::Unsigned(sequence)],
            event_time,
            values,
        )
    }

    fn kernel_schema() -> Schema {
        Schema::new(vec![
            Field::new(
                "ts",
                DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
                false,
            ),
            Field::new("symbol", DataType::Utf8, false),
            Field::new("sequence", DataType::UInt64, false),
            Field::new("price", DataType::Float64, true),
            Field::new("volume", DataType::Int64, true),
            Field::new("label", DataType::Utf8, true),
        ])
    }

    fn kernel_spec(outputs: Value) -> RollingSpec {
        let mut document = valid_spec_json();
        document["partition_by"] = json!(["symbol"]);
        document["event_time"] = json!("ts");
        document["sequence_by"] = json!(["sequence"]);
        document["outputs"] = outputs;
        serde_json::from_value(document).unwrap()
    }

    fn lag_price(periods: u64) -> Value {
        json!({
            "kind": "lag",
            "primitive_version": 1,
            "input": "price",
            "output": "price_lag",
            "periods": periods
        })
    }

    fn delta_price(periods: u64) -> Value {
        json!({
            "kind": "delta",
            "primitive_version": 1,
            "input": "price",
            "output": "price_delta",
            "periods": periods
        })
    }

    fn delta_volume(periods: u64) -> Value {
        json!({
            "kind": "delta",
            "primitive_version": 1,
            "input": "volume",
            "output": "volume_delta",
            "periods": periods
        })
    }

    fn compute(
        spec: &RollingSpec,
        histories: &RollingHistories,
        rows: &[BufferedRow],
    ) -> Result<ComputedOutputs> {
        let compiled = compile_spec(spec, &kernel_schema())?;
        compute_output_columns(rows, histories, &compiled, "rolling")
    }

    fn float_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<f64>> {
        outputs.columns[index]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    fn signed_column(outputs: &ComputedOutputs, index: usize) -> Vec<Option<i64>> {
        outputs.columns[index]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Int64Array>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn lag_references_the_previous_row_within_each_entity() {
        let spec = kernel_spec(json!([lag_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(1, "b", 1, vec![ScalarValue::Float64(Some(10.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
            full_row(2, "b", 2, vec![ScalarValue::Float64(Some(20.0))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![None, None, Some(1.0), Some(10.0), Some(2.0)]
        );
    }

    #[test]
    fn lag_periods_span_the_shared_history_across_segmentation() {
        let spec = kernel_spec(json!([lag_price(2)]));
        let mut histories = RollingHistories::default();
        let first = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(Some(1.0))]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(2.0))]),
        ];
        let outputs = compute(&spec, &histories, &first).unwrap();
        assert_eq!(float_column(&outputs, 0), vec![None, None]);
        histories.apply(outputs.touched);

        let second = vec![
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
            full_row(5, "a", 5, vec![ScalarValue::Float64(Some(5.0))]),
        ];
        let outputs = compute(&spec, &histories, &second).unwrap();
        assert_eq!(
            float_column(&outputs, 0),
            vec![Some(1.0), Some(2.0), Some(3.0)]
        );
    }

    #[test]
    fn lag_preserves_null_and_nan_at_the_referenced_position() {
        let spec = kernel_spec(json!([lag_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(f64::NAN))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(3.0))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(4.0))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .unwrap();
        assert!(values.is_null(0));
        assert!(values.is_null(1));
        assert!(values.value(2).is_nan());
        assert_eq!(values.value(3).to_bits(), 3.0_f64.to_bits());
    }

    #[test]
    fn lag_works_for_non_numeric_columns() {
        let spec = kernel_spec(json!([{
            "kind": "lag",
            "primitive_version": 1,
            "input": "label",
            "output": "label_lag",
            "periods": 1
        }]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(None),
                    ScalarValue::Utf8(Some("x".into())),
                ],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(None),
                    ScalarValue::Utf8(Some("y".into())),
                ],
            ),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StringArray>()
            .unwrap();
        assert!(values.is_null(0));
        assert_eq!(values.value(1), "x");
    }

    #[test]
    fn delta_subtracts_the_referenced_value_with_checked_integer_math() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(7))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(10))],
            ),
            full_row(
                3,
                "a",
                3,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(4))],
            ),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        assert_eq!(signed_column(&outputs, 0), vec![None, Some(3), Some(-6)]);
    }

    #[test]
    fn delta_integer_overflow_is_a_data_error() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX)),
                ],
            ),
        ];
        let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { ref node_id, .. } if node_id == "rolling"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn delta_preserves_null_and_propagates_nan() {
        let spec = kernel_spec(json!([delta_price(1)]));
        let rows = vec![
            full_row(1, "a", 1, vec![ScalarValue::Float64(None)]),
            full_row(2, "a", 2, vec![ScalarValue::Float64(Some(1.5))]),
            full_row(3, "a", 3, vec![ScalarValue::Float64(Some(f64::NAN))]),
            full_row(4, "a", 4, vec![ScalarValue::Float64(Some(2.5))]),
            full_row(5, "a", 5, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
            full_row(6, "a", 6, vec![ScalarValue::Float64(Some(f64::INFINITY))]),
        ];
        let outputs = compute(&spec, &RollingHistories::default(), &rows).unwrap();
        let values = outputs.columns[0]
            .as_any()
            .downcast_ref::<datafusion::arrow::array::Float64Array>()
            .unwrap();
        assert!(values.is_null(0));
        assert!(values.is_null(1));
        assert!(values.value(2).is_nan());
        assert!(values.value(3).is_nan());
        assert_eq!(values.value(4).to_bits(), f64::INFINITY.to_bits());
        assert!(values.value(5).is_nan());
    }

    #[test]
    fn delta_unsigned_underflow_is_a_data_error() {
        let spec = kernel_spec(json!([{
            "kind": "delta",
            "primitive_version": 1,
            "input": "sequence",
            "output": "sequence_delta",
            "periods": 1
        }]));
        let rows = vec![full_row(1, "a", 10, vec![]), full_row(2, "a", 3, vec![])];
        let error = compute(&spec, &RollingHistories::default(), &rows).unwrap_err();
        assert!(
            matches!(error, CalcFlowError::Operator { .. }),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn history_is_truncated_to_the_maximum_declared_periods() {
        let spec = kernel_spec(json!([lag_price(2), lag_price(1)]));
        let mut histories = RollingHistories::default();
        for batch in 0..3_u32 {
            let rows = (0..4_u32)
                .map(|index| {
                    let sequence = batch * 4 + index + 1;
                    full_row(
                        i64::from(sequence),
                        "a",
                        u64::from(sequence),
                        vec![ScalarValue::Float64(Some(f64::from(sequence)))],
                    )
                })
                .collect::<Vec<_>>();
            let outputs = compute(&spec, &histories, &rows).unwrap();
            histories.apply(outputs.touched);
        }
        for history in histories.by_entity.values() {
            assert!(history.len() <= 2);
        }
    }

    #[test]
    fn failed_delta_leaves_histories_untouched() {
        let spec = kernel_spec(json!([delta_volume(1)]));
        let histories = RollingHistories::default();
        let rows = vec![
            full_row(
                1,
                "a",
                1,
                vec![ScalarValue::Float64(None), ScalarValue::Int64(Some(-1))],
            ),
            full_row(
                2,
                "a",
                2,
                vec![
                    ScalarValue::Float64(None),
                    ScalarValue::Int64(Some(i64::MAX)),
                ],
            ),
        ];
        assert!(compute(&spec, &histories, &rows).is_err());
        assert!(histories.by_entity.is_empty());
    }

    // ------------------------------------------------------------------
    // Operator construction and metadata
    // ------------------------------------------------------------------

    #[test]
    fn operator_exposes_exact_ports_and_frozen_configuration() {
        let operator =
            RollingOperator::new("rolling_features", Arc::new(input_schema()), valid_spec())
                .unwrap();
        assert_eq!(operator.name(), "rolling_features");
        let [input] = operator.input_ports() else {
            panic!("rolling exposes one input port");
        };
        assert_eq!(input.name(), "input");
        assert!(input.required());
        assert_eq!(input.schema().unwrap().as_ref(), &input_schema());
        let [output] = operator.output_ports() else {
            panic!("rolling exposes one output port");
        };
        assert_eq!(output.name(), "output");
        assert!(output.required());
        assert_eq!(
            output.schema().unwrap().as_ref(),
            valid_spec().validate(&input_schema()).unwrap().as_ref()
        );
        assert_eq!(
            serde_json::to_value(operator.configuration()).unwrap(),
            json!({
                "kind": "rolling",
                "spec": valid_spec_json(),
            })
        );
    }
}
