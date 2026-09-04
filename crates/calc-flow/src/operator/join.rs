use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    io::Cursor,
    sync::Arc,
    time::Duration,
};

use async_trait::async_trait;
use datafusion::arrow::{
    array::{
        Array, ArrayAccessor, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray,
        DictionaryArray, FixedSizeListArray, LargeBinaryArray, LargeListArray, LargeListViewArray,
        LargeStringArray, ListArray, ListViewArray, MapArray, PrimitiveArray, RunArray,
        StringArray, StringViewArray, StructArray, TimestampMicrosecondArray,
        TimestampMillisecondArray, TimestampNanosecondArray, TimestampSecondArray, UInt64Array,
        UnionArray,
    },
    compute::concat,
    datatypes::{
        ArrowPrimitiveType, DataType, Field, Int8Type, Int16Type, Int32Type, Int64Type,
        IntervalUnit, Schema, SchemaRef, TimeUnit, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
    },
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, DataFusionRuntime, Epoch,
    EventTime, IngressProgress, JsonMap, OperatorStateSnapshot, Port, Result, StateSegment,
    StreamCollector, StreamOperator, StreamOperatorContext, UdfRegistrySnapshot,
};

use super::{OperatorMetadata, StreamRuntimeState, is_portable_identifier, validate_operator_name};

/// The fixed logical bookkeeping charge for one retained Join row.
pub const STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1: u64 = 64;

/// Largest integer that round-trips exactly through ordinary JSON numbers.
pub const STREAM_JOIN_MAX_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Supported Join semantics.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum StreamJoinType {
    /// Emit every pair with equal non-null keys and an event time inside the
    /// configured inclusive interval.
    Inner,
}

/// Inclusive event-time distance around one left row.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct JoinTimeBounds {
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    before_micros: u64,
    #[schemars(range(min = 0, max = 9_007_199_254_740_991_u64))]
    after_micros: u64,
}

impl JoinTimeBounds {
    /// Creates exact, non-negative microsecond bounds.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when either duration has
    /// sub-microsecond precision or exceeds the exact JSON integer domain.
    pub fn new(before: Duration, after: Duration) -> Result<Self> {
        Ok(Self {
            before_micros: exact_safe_duration_micros(before, "stream_join.bounds.before_micros")?,
            after_micros: exact_safe_duration_micros(after, "stream_join.bounds.after_micros")?,
        })
    }

    pub(crate) fn from_micros(before_micros: u64, after_micros: u64) -> Result<Self> {
        validate_safe_integer(before_micros, false, "stream_join.bounds.before_micros")?;
        validate_safe_integer(after_micros, false, "stream_join.bounds.after_micros")?;
        Ok(Self {
            before_micros,
            after_micros,
        })
    }

    /// Returns the exact preceding distance in microseconds.
    pub const fn before_micros(self) -> u64 {
        self.before_micros
    }

    /// Returns the exact following distance in microseconds.
    pub const fn after_micros(self) -> u64 {
        self.after_micros
    }

    /// Returns the preceding distance.
    pub const fn before(self) -> Duration {
        Duration::from_micros(self.before_micros)
    }

    /// Returns the following distance.
    pub const fn after(self) -> Duration {
        Duration::from_micros(self.after_micros)
    }

    pub(crate) fn contains_pair(self, left_micros: i64, right_micros: i64) -> bool {
        let left = i128::from(left_micros);
        let right = i128::from(right_micros);
        right >= left - i128::from(self.before_micros)
            && right <= left + i128::from(self.after_micros)
    }
}

impl<'de> Deserialize<'de> for JoinTimeBounds {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            before_micros: u64,
            after_micros: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::from_micros(fields.before_micros, fields.after_micros).map_err(D::Error::custom)
    }
}

/// Hard logical state and per-input fan-out limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[allow(
    clippy::struct_field_names,
    reason = "the frozen public JSON field names all use the max_ limit prefix"
)]
#[serde(deny_unknown_fields)]
pub struct JoinStateLimits {
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_rows_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_state_bytes_per_side: u64,
    #[schemars(range(min = 1, max = 9_007_199_254_740_991_u64))]
    max_matches_per_input_batch: u64,
}

impl JoinStateLimits {
    /// Creates required positive Join limits.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when a value is zero or is
    /// larger than [`STREAM_JOIN_MAX_SAFE_JSON_INTEGER`].
    pub fn new(
        max_state_rows_per_side: u64,
        max_state_bytes_per_side: u64,
        max_matches_per_input_batch: u64,
    ) -> Result<Self> {
        validate_safe_integer(
            max_state_rows_per_side,
            true,
            "stream_join.limits.max_state_rows_per_side",
        )?;
        validate_safe_integer(
            max_state_bytes_per_side,
            true,
            "stream_join.limits.max_state_bytes_per_side",
        )?;
        validate_safe_integer(
            max_matches_per_input_batch,
            true,
            "stream_join.limits.max_matches_per_input_batch",
        )?;
        Ok(Self {
            max_state_rows_per_side,
            max_state_bytes_per_side,
            max_matches_per_input_batch,
        })
    }

    /// Maximum retained rows on either side.
    pub const fn max_state_rows_per_side(self) -> u64 {
        self.max_state_rows_per_side
    }

    /// Maximum logical retained bytes on either side.
    pub const fn max_state_bytes_per_side(self) -> u64 {
        self.max_state_bytes_per_side
    }

    /// Maximum pairs one accepted input batch may emit.
    pub const fn max_matches_per_input_batch(self) -> u64 {
        self.max_matches_per_input_batch
    }
}

impl<'de> Deserialize<'de> for JoinStateLimits {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[allow(
            clippy::struct_field_names,
            reason = "the wire DTO must preserve the frozen max_ field names"
        )]
        #[serde(deny_unknown_fields)]
        struct Fields {
            max_state_rows_per_side: u64,
            max_state_bytes_per_side: u64,
            max_matches_per_input_batch: u64,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(
            fields.max_state_rows_per_side,
            fields.max_state_bytes_per_side,
            fields.max_matches_per_input_batch,
        )
        .map_err(D::Error::custom)
    }
}

/// Immutable declaration for a two-input bounded inner stream Join.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct StreamJoinSpec {
    join_type: StreamJoinType,
    left_keys: Vec<String>,
    right_keys: Vec<String>,
    left_event_time: String,
    right_event_time: String,
    bounds: JoinTimeBounds,
    limits: JoinStateLimits,
    left_prefix: String,
    right_prefix: String,
}

impl StreamJoinSpec {
    /// Canonical `left_prefix` default materialized after a clean raw pass.
    pub const DEFAULT_LEFT_PREFIX: &'static str = "left";

    /// Canonical `right_prefix` default materialized after a clean raw pass.
    pub const DEFAULT_RIGHT_PREFIX: &'static str = "right";

    /// Creates an inner Join with canonical `left` and `right` prefixes.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for empty, duplicate, or
    /// unequally sized key declarations and invalid event-time names.
    pub fn inner<L, R, LI, RI>(
        left_keys: L,
        right_keys: R,
        left_event_time: &str,
        right_event_time: &str,
        bounds: JoinTimeBounds,
        limits: JoinStateLimits,
    ) -> Result<Self>
    where
        L: IntoIterator<Item = LI>,
        R: IntoIterator<Item = RI>,
        LI: Into<String>,
        RI: Into<String>,
    {
        let left_keys = left_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        let right_keys = right_keys.into_iter().map(Into::into).collect::<Vec<_>>();
        validate_key_names(&left_keys, &right_keys)?;
        validate_column_name(left_event_time, "stream_join.left_event_time")?;
        validate_column_name(right_event_time, "stream_join.right_event_time")?;
        Ok(Self {
            join_type: StreamJoinType::Inner,
            left_keys,
            right_keys,
            left_event_time: left_event_time.into(),
            right_event_time: right_event_time.into(),
            bounds,
            limits,
            left_prefix: "left".into(),
            right_prefix: "right".into(),
        })
    }

    /// Replaces both output prefixes without mutating the original value.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] unless both values are
    /// distinct portable identifiers.
    pub fn with_prefixes(mut self, left_prefix: &str, right_prefix: &str) -> Result<Self> {
        validate_prefixes(left_prefix, right_prefix)?;
        self.left_prefix = left_prefix.into();
        self.right_prefix = right_prefix.into();
        Ok(self)
    }

    pub const fn join_type(&self) -> StreamJoinType {
        self.join_type
    }

    pub fn left_keys(&self) -> &[String] {
        &self.left_keys
    }

    pub fn right_keys(&self) -> &[String] {
        &self.right_keys
    }

    pub fn left_event_time(&self) -> &str {
        &self.left_event_time
    }

    pub fn right_event_time(&self) -> &str {
        &self.right_event_time
    }

    pub const fn bounds(&self) -> JoinTimeBounds {
        self.bounds
    }

    pub const fn limits(&self) -> JoinStateLimits {
        self.limits
    }

    pub fn left_prefix(&self) -> &str {
        &self.left_prefix
    }

    pub fn right_prefix(&self) -> &str {
        &self.right_prefix
    }
}

impl<'de> Deserialize<'de> for StreamJoinSpec {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            join_type: StreamJoinType,
            left_keys: Vec<String>,
            right_keys: Vec<String>,
            left_event_time: String,
            right_event_time: String,
            bounds: JoinTimeBounds,
            limits: JoinStateLimits,
            #[serde(default = "default_left_prefix")]
            left_prefix: String,
            #[serde(default = "default_right_prefix")]
            right_prefix: String,
        }

        let fields = Fields::deserialize(deserializer)?;
        if fields.join_type != StreamJoinType::Inner {
            return Err(D::Error::custom("only inner stream joins are supported"));
        }
        Self::inner(
            fields.left_keys,
            fields.right_keys,
            &fields.left_event_time,
            &fields.right_event_time,
            fields.bounds,
            fields.limits,
        )
        .and_then(|spec| spec.with_prefixes(&fields.left_prefix, &fields.right_prefix))
        .map_err(D::Error::custom)
    }
}

/// Payload-free status snapshot for one side of a retained Join state
/// (api note "Payload-free Join status").
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamJoinSideStatus {
    /// Logically retained rows on this side.
    pub retained_rows: u64,
    /// Versioned logical byte charge of the retained rows.
    pub retained_bytes: u64,
    /// Rows removed by watermark eviction or a side End.
    pub evicted_rows: u64,
    /// Rows dropped as late under this side's watermark.
    pub late_rows: u64,
    /// Input batches that contained at least one late row.
    pub late_affected_batches: u64,
    /// Largest observed lateness, if any late row was seen.
    pub max_lateness: Option<Duration>,
    /// Rows dropped because their event time was null.
    pub null_event_time_rows: u64,
    /// Rows dropped because a key component was null.
    pub null_key_rows: u64,
}

/// Payload-free Join status for one node (api note "Payload-free Join status").
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StreamJoinStatus {
    /// Retained left-side state.
    pub left: StreamJoinSideStatus,
    /// Retained right-side state.
    pub right: StreamJoinSideStatus,
    /// Match rows emitted so far.
    pub emitted_match_rows: u64,
    /// State-limit admission failures.
    pub state_limit_failures: u64,
    /// Match-limit admission failures.
    pub match_limit_failures: u64,
}

fn side_status(metrics: &SideMetrics) -> StreamJoinSideStatus {
    StreamJoinSideStatus {
        retained_rows: metrics.retained_rows,
        retained_bytes: metrics.retained_bytes,
        evicted_rows: metrics.evicted_rows,
        late_rows: metrics.late_rows,
        late_affected_batches: metrics.late_affected_batches,
        max_lateness: metrics.max_lateness_micros.map(Duration::from_micros),
        null_event_time_rows: metrics.null_event_time_rows,
        null_key_rows: metrics.null_key_rows,
    }
}

/// Stateful two-input bounded event-time Join.
pub struct StreamJoinOperator {
    name: String,
    spec: StreamJoinSpec,
    input_ports: [Port; 2],
    output_ports: [Port; 1],
    compiled: CompiledJoin,
    runtime: StreamRuntimeState,
    state: StreamJoinState,
}

#[derive(Clone)]
struct CompiledJoin {
    left_key_indices: Vec<usize>,
    right_key_indices: Vec<usize>,
    left_event_time_index: usize,
    right_event_time_index: usize,
    equality_query: String,
}

/// Scratch-table alias holding the admitted rows of the current input batch.
const PROBE_TABLE: &str = "probe_input";
/// Scratch-table alias holding the opposite side's retained state rows.
const STATE_TABLE: &str = "state_input";
/// Renamed key column prefix shared by both scratch tables.
const KEY_COLUMN_PREFIX: &str = "__cf_join_key_";
/// Position of one admitted row inside [`PROBE_TABLE`].
const PROBE_POS_COLUMN: &str = "__cf_join_pos";
/// Retained-state row id inside [`STATE_TABLE`].
const STATE_RID_COLUMN: &str = "__cf_join_row_id";

/// One incoming row that passed null-key, null-event-time, and lateness admission.
struct AdmittedRow {
    record: RecordBatch,
    event_time: EventTime,
    row_id: u64,
    retain: bool,
}

/// Scratch accumulator for one input batch's admission pass.
struct AdmissionBundle {
    next_row_id: u64,
    metrics: SideMetrics,
    admitted: Vec<AdmittedRow>,
    had_late: bool,
}

/// Why an incoming physical row was dropped during admission.
#[derive(Clone, Copy)]
enum DropKind {
    NullEventTime,
    NullKey,
    Late(u64),
}

/// Classified disposition of one incoming physical row.
enum RowAdmission {
    Dropped(DropKind),
    Admitted(EventTime),
}

/// One time-qualified key-equal pair, ordered for emission.
struct MatchedPair {
    pos: usize,
    opposite_index: usize,
}

#[derive(Clone)]
struct StoredRow {
    record: RecordBatch,
    event_time: EventTime,
    row_id: u64,
    charge: u64,
    encoded_key: Arc<Vec<u8>>,
}

/// One side of the Join state, in durable-identity order.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum JoinSide {
    Left,
    Right,
}

impl JoinSide {
    const fn as_str(self) -> &'static str {
        match self {
            JoinSide::Left => "left",
            JoinSide::Right => "right",
        }
    }
}

/// One dirty state change since the last captured checkpoint (spec FR45).
#[derive(Clone)]
enum PendingOp {
    Upsert {
        side: JoinSide,
        row_id: u64,
        event_time: EventTime,
        encoded_key: Arc<Vec<u8>>,
        record: RecordBatch,
        charge: u64,
    },
    Tombstone {
        side: JoinSide,
        row_id: u64,
        event_time: EventTime,
        encoded_key: Arc<Vec<u8>>,
    },
}

impl PendingOp {
    fn identity(&self) -> (JoinSide, u64) {
        match self {
            PendingOp::Upsert { side, row_id, .. } | PendingOp::Tombstone { side, row_id, .. } => {
                (*side, *row_id)
            }
        }
    }
}

/// Prepared checkpoint segments and the dirty log (spec FR45/FR47).
///
/// Bulk encoding and compaction are prepared during data/progress handlers;
/// `checkpoint` only shares the prepared segment allocations and encodes the
/// dirty ops.
#[derive(Default)]
struct DeltaTracking {
    base: BTreeMap<&'static str, StateSegment>,
    segments: BTreeMap<(u64, &'static str), StateSegment>,
    pending: Vec<PendingOp>,
    segments_since_base: u32,
    needs_compaction: bool,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct SideMetrics {
    retained_rows: u64,
    retained_bytes: u64,
    evicted_rows: u64,
    late_rows: u64,
    late_affected_batches: u64,
    max_lateness_micros: Option<u64>,
    null_event_time_rows: u64,
    null_key_rows: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
struct JoinMetrics {
    left: SideMetrics,
    right: SideMetrics,
    emitted_match_rows: u64,
    state_limit_failures: u64,
    match_limit_failures: u64,
}

#[derive(Default)]
struct StreamJoinState {
    left: Vec<StoredRow>,
    right: Vec<StoredRow>,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    last_checkpoint_epoch: Option<Epoch>,
    deltas: DeltaTracking,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct JoinCheckpointMetadata {
    layout_version: u32,
    spec: StreamJoinSpec,
    next_left_row_id: u64,
    next_right_row_id: u64,
    next_output_sequence: u64,
    metrics: JoinMetrics,
    ended: bool,
    epoch: u64,
}

struct PreparedJoinBatch {
    outputs: Vec<RecordBatch>,
    retained: Vec<StoredRow>,
    next_row_id: u64,
    metrics: SideMetrics,
}

impl StreamJoinOperator {
    /// Compiles one Join declaration against two exact Arrow schemas.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for declaration errors and
    /// [`CalcFlowError::Compile`] for incompatible schemas.
    pub fn new(
        name: &str,
        left_schema: SchemaRef,
        right_schema: SchemaRef,
        spec: StreamJoinSpec,
    ) -> Result<Self> {
        validate_operator_name(name)?;
        validate_payload_charge_support(&left_schema, "left")?;
        validate_payload_charge_support(&right_schema, "right")?;
        let (output_schema, compiled) = compile_schemas(&left_schema, &right_schema, &spec)?;
        Ok(Self {
            name: name.into(),
            spec,
            input_ports: [
                Port::with_schema_ref("left", BatchKind::Table, true, Some(left_schema))?,
                Port::with_schema_ref("right", BatchKind::Table, true, Some(right_schema))?,
            ],
            output_ports: [Port::with_schema_ref(
                "output",
                BatchKind::Table,
                true,
                Some(output_schema),
            )?],
            compiled,
            runtime: StreamRuntimeState::new(),
            state: StreamJoinState::default(),
        })
    }

    /// Returns the immutable Join declaration.
    pub const fn spec(&self) -> &StreamJoinSpec {
        &self.spec
    }

    /// Returns a payload-free status snapshot of the retained Join state.
    pub fn status(&self) -> StreamJoinStatus {
        StreamJoinStatus {
            left: side_status(&self.state.metrics.left),
            right: side_status(&self.state.metrics.right),
            emitted_match_rows: self.state.metrics.emitted_match_rows,
            state_limit_failures: self.state.metrics.state_limit_failures,
            match_limit_failures: self.state.metrics.match_limit_failures,
        }
    }

    pub(crate) fn set_stream_resources(
        &mut self,
        config: DataFusionConfig,
        udfs: UdfRegistrySnapshot,
    ) {
        self.runtime.set_resources(config, udfs, Vec::new());
    }

    pub(crate) const fn stream_runtime_initialized(&self) -> bool {
        self.runtime.is_initialized()
    }

    pub(crate) fn output_frontier_candidate(
        &self,
        progress: &crate::IngressProgressSnapshot,
    ) -> Result<Option<EventTime>> {
        let left = progress.get("left").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing left ingress progress for output frontier",
            )
        })?;
        let right = progress.get("right").ok_or_else(|| {
            operator_error(
                &self.name,
                "missing right ingress progress for output frontier",
            )
        })?;
        let left_live = left.state() != crate::IngressState::Ended;
        let right_live = right.state() != crate::IngressState::Ended;
        let candidate = match (left_live, right_live) {
            (true, true) => match (left.watermark(), right.watermark()) {
                (Some(left), Some(right)) => Some(
                    (i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros))
                        .min(
                            i128::from(right.as_micros())
                                - i128::from(self.spec.bounds.after_micros),
                        ),
                ),
                _ => None,
            },
            (true, false) => left.watermark().map(|left| {
                i128::from(left.as_micros()) - i128::from(self.spec.bounds.before_micros)
            }),
            (false, true) => right.watermark().map(|right| {
                i128::from(right.as_micros()) - i128::from(self.spec.bounds.after_micros)
            }),
            (false, false) => None,
        };
        candidate
            .filter(|candidate| *candidate >= i128::from(i64::MIN))
            .map(|candidate| {
                i64::try_from(candidate)
                    .map(EventTime::from_micros)
                    .map_err(|_| operator_error(&self.name, "output frontier exceeds EventTime"))
            })
            .transpose()
    }

    async fn prepare_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedJoinBatch> {
        let plan = self.begin_batch(ingress, batch)?;
        let mut bundle = self.admission_bundle(&plan);
        for record in batch.table_payload()?.batches() {
            self.admit_record(record, &plan, ingress, context, &mut bundle)?;
        }
        bundle.finish(&self.name)?;
        let outputs = self.evaluate_matches(&plan, &bundle.admitted).await?;
        self.finish_prepared(&plan, bundle, outputs)
    }

    fn finish_prepared(
        &mut self,
        plan: &SidePlan,
        bundle: AdmissionBundle,
        outputs: Vec<RecordBatch>,
    ) -> Result<PreparedJoinBatch> {
        let retained = retained_rows(&bundle.admitted, &plan.key_indices, &self.name)?;
        self.validate_state_admission(plan.incoming_is_left, &retained)?;
        Ok(PreparedJoinBatch {
            outputs,
            retained,
            next_row_id: bundle.next_row_id,
            metrics: bundle.metrics,
        })
    }

    /// Validates the ingress and port contract before any admission work.
    fn begin_batch(&self, ingress: &str, batch: &Batch) -> Result<SidePlan> {
        if self.state.ended {
            return Err(operator_error(
                &self.name,
                "received data after end-of-input",
            ));
        }
        let plan = self.side_plan(ingress)?;
        self.input_ports[plan.port_index].validate(batch, &format!("{}.{}", self.name, ingress))?;
        Ok(plan)
    }

    fn side_plan(&self, ingress: &str) -> Result<SidePlan> {
        match ingress {
            "left" => Ok(SidePlan {
                incoming_is_left: true,
                port_index: 0,
                event_time_index: self.compiled.left_event_time_index,
                key_indices: self.compiled.left_key_indices.clone(),
            }),
            "right" => Ok(SidePlan {
                incoming_is_left: false,
                port_index: 1,
                event_time_index: self.compiled.right_event_time_index,
                key_indices: self.compiled.right_key_indices.clone(),
            }),
            _ => Err(operator_error(
                &self.name,
                &format!("unknown ingress {ingress:?}; expected left or right"),
            )),
        }
    }

    fn admission_bundle(&self, plan: &SidePlan) -> AdmissionBundle {
        let (next_row_id, metrics) = if plan.incoming_is_left {
            (self.state.next_left_row_id, self.state.metrics.left.clone())
        } else {
            (
                self.state.next_right_row_id,
                self.state.metrics.right.clone(),
            )
        };
        AdmissionBundle {
            next_row_id,
            metrics,
            admitted: Vec::new(),
            had_late: false,
        }
    }

    fn admit_record(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
        bundle: &mut AdmissionBundle,
    ) -> Result<()> {
        let side_progress = context.ingress_progress().get(ingress);
        let opposite_progress = context.ingress_progress().get(if plan.incoming_is_left {
            "right"
        } else {
            "left"
        });
        for row_index in 0..record.num_rows() {
            let row_id = bundle.reserve_row_id(&self.name)?;
            match self.classify_row(record, plan, row_index, side_progress, ingress)? {
                RowAdmission::Dropped(kind) => bundle.note_dropped(kind, &self.name)?,
                RowAdmission::Admitted(event_time) => bundle.push_admitted(AdmittedRow {
                    record: record.slice(row_index, 1),
                    event_time,
                    row_id,
                    retain: should_retain(
                        plan.incoming_is_left,
                        event_time,
                        opposite_progress,
                        self.spec.bounds,
                    ),
                }),
            }
        }
        Ok(())
    }

    fn classify_row(
        &self,
        record: &RecordBatch,
        plan: &SidePlan,
        row_index: usize,
        side_progress: Option<IngressProgress>,
        ingress: &str,
    ) -> Result<RowAdmission> {
        let Some(event_time) = event_time_at(
            record,
            plan.event_time_index,
            row_index,
            &self.name,
            ingress,
        )?
        else {
            return Ok(RowAdmission::Dropped(DropKind::NullEventTime));
        };
        if plan
            .key_indices
            .iter()
            .any(|&index| record.column(index).is_null(row_index))
        {
            return Ok(RowAdmission::Dropped(DropKind::NullKey));
        }
        match late_lateness(event_time, side_progress, &self.name)? {
            Some(lateness) => Ok(RowAdmission::Dropped(DropKind::Late(lateness))),
            None => Ok(RowAdmission::Admitted(event_time)),
        }
    }

    /// Runs the batched key-equality probe and emits one output row per pair.
    ///
    /// Key equality executes as one `DataFusion` join over scratch tables per
    /// input batch; the time bound stays in checked `i128` Rust arithmetic.
    async fn evaluate_matches(
        &mut self,
        plan: &SidePlan,
        admitted: &[AdmittedRow],
    ) -> Result<Vec<RecordBatch>> {
        let runtime = self.runtime.runtime()?;
        let opposite = if plan.incoming_is_left {
            self.state.right.as_slice()
        } else {
            self.state.left.as_slice()
        };
        if admitted.is_empty() || opposite.is_empty() {
            return Ok(Vec::new());
        }
        let matched = matched_pairs(
            runtime,
            &self.compiled,
            &self.spec,
            plan,
            admitted,
            opposite,
            &self.name,
        )
        .await?;
        enforce_match_limit(
            matched.len(),
            &mut self.state.metrics.match_limit_failures,
            self.spec.limits.max_matches_per_input_batch,
            &self.name,
        )?;
        let output_schema = self.output_ports[0]
            .schema()
            .expect("stream Join output always has an exact schema");
        materialize_outputs(
            output_schema,
            admitted,
            opposite,
            &matched,
            plan.incoming_is_left,
            &self.name,
        )
    }

    fn validate_state_admission(
        &mut self,
        incoming_is_left: bool,
        retained: &[StoredRow],
    ) -> Result<()> {
        let current = if incoming_is_left {
            &self.state.left
        } else {
            &self.state.right
        };
        let (rows, bytes) = prospective_state_charge(current, retained, &self.name)?;
        if rows > self.spec.limits.max_state_rows_per_side
            || bytes > self.spec.limits.max_state_bytes_per_side
        {
            self.state.metrics.state_limit_failures = checked_metric(
                self.state.metrics.state_limit_failures,
                1,
                &self.name,
                "state_limit_failures",
            )?;
            return Err(operator_reason(
                &self.name,
                crate::StreamingFailureReason::JoinStateLimitExceeded,
                "retained state limit exceeded",
            ));
        }
        Ok(())
    }

    fn validate_restored_limits(&self, left: &[StoredRow], right: &[StoredRow]) -> Result<()> {
        for (side, rows) in [("left", left), ("right", right)] {
            let row_count =
                u64::try_from(rows.len()).map_err(|_| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} row count is too large", self.name),
                })?;
            let byte_count = rows
                .iter()
                .try_fold(0_u64, |total, row| total.checked_add(row.charge))
                .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {:?} {side} byte charge overflowed", self.name),
                })?;
            if row_count > self.spec.limits.max_state_rows_per_side
                || byte_count > self.spec.limits.max_state_bytes_per_side
            {
                return Err(CalcFlowError::CheckpointMismatch {
                    message: format!(
                        "stream Join {:?} restored {side} state exceeds configured limits",
                        self.name
                    ),
                });
            }
        }
        Ok(())
    }

    async fn emit_prepared(
        &mut self,
        prepared: &PreparedJoinBatch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let sequence_count = u64::try_from(prepared.outputs.len())
            .map_err(|_| counter_overflow(&self.name, "output sequence"))?;
        self.state
            .next_output_sequence
            .checked_add(sequence_count)
            .ok_or_else(|| counter_overflow(&self.name, "output sequence"))?;
        for record in &prepared.outputs {
            let sequence = self.state.next_output_sequence;
            emit_output_row(record, sequence, context, output).await?;
            self.state.next_output_sequence += 1;
        }
        Ok(())
    }

    fn commit_prepared(&mut self, ingress: &str, prepared: PreparedJoinBatch) -> Result<()> {
        let metrics = prepared.metrics;
        let side = if ingress == "left" {
            JoinSide::Left
        } else {
            JoinSide::Right
        };
        for row in &prepared.retained {
            self.state.deltas.pending.push(PendingOp::Upsert {
                side,
                row_id: row.row_id,
                event_time: row.event_time,
                encoded_key: Arc::clone(&row.encoded_key),
                record: row.record.clone(),
                charge: row.charge,
            });
        }
        if side == JoinSide::Left {
            self.state.next_left_row_id = prepared.next_row_id;
            self.state.left.extend(prepared.retained);
            self.state.metrics.left = metrics;
            refresh_retained_metrics(&mut self.state.metrics.left, &self.state.left, &self.name)?;
        } else {
            self.state.next_right_row_id = prepared.next_row_id;
            self.state.right.extend(prepared.retained);
            self.state.metrics.right = metrics;
            refresh_retained_metrics(&mut self.state.metrics.right, &self.state.right, &self.name)?;
        }
        Ok(())
    }

    fn decode_restored_sides(
        &self,
        snapshot: &OperatorStateSnapshot,
    ) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
        restore_sides_from_segments(
            snapshot,
            self.input_schema(0),
            self.input_schema(1),
            &self.compiled.left_key_indices,
            &self.compiled.right_key_indices,
            &self.name,
        )
    }

    fn input_schema(&self, port_index: usize) -> &SchemaRef {
        self.input_ports[port_index]
            .schema()
            .expect("stream Join inputs always have an exact schema")
    }

    fn validate_restored_join_rows(
        &self,
        metadata: &JoinCheckpointMetadata,
        left: &[StoredRow],
        right: &[StoredRow],
    ) -> Result<()> {
        validate_restored_rows(
            left,
            metadata.next_left_row_id,
            self.compiled.left_event_time_index,
            &self.compiled.left_key_indices,
            &self.name,
            "left",
        )?;
        validate_restored_rows(
            right,
            metadata.next_right_row_id,
            self.compiled.right_event_time_index,
            &self.compiled.right_key_indices,
            &self.name,
            "right",
        )
    }
}

impl fmt::Debug for StreamJoinOperator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StreamJoinOperator")
            .field("name", &self.name)
            .field("spec", &self.spec)
            .field("input_ports", &self.input_ports)
            .field("output_ports", &self.output_ports)
            .finish_non_exhaustive()
    }
}

impl OperatorMetadata for StreamJoinOperator {
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
        let value = serde_json::to_value(&self.spec)
            .expect("validated stream Join configuration remains serializable");
        let Value::Object(values) = value else {
            unreachable!("stream Join configuration serializes as an object")
        };
        values.into_iter().collect()
    }
}

#[async_trait]
impl StreamOperator for StreamJoinOperator {
    async fn process_data(
        &mut self,
        ingress: &str,
        batch: Batch,
        context: &StreamOperatorContext<'_>,
        output: &mut dyn StreamCollector,
    ) -> Result<()> {
        context.check_cancelled()?;
        if self.state.deltas.needs_compaction {
            compact_base(&mut self.state, &self.name)?;
        }
        let prepared = self.prepare_batch(ingress, &batch, context).await?;
        self.emit_prepared(&prepared, context, output).await?;
        let emitted = u64::try_from(prepared.outputs.len())
            .map_err(|_| counter_overflow(&self.name, "emitted rows"))?;
        self.state.metrics.emitted_match_rows = checked_metric(
            self.state.metrics.emitted_match_rows,
            emitted,
            &self.name,
            "emitted_match_rows",
        )?;
        self.commit_prepared(ingress, prepared)?;
        Ok(())
    }

    async fn on_ingress_progress(
        &mut self,
        ingress: &str,
        context: &StreamOperatorContext<'_>,
    ) -> Result<()> {
        let progress = context.ingress_progress().get(ingress).ok_or_else(|| {
            operator_error(
                &self.name,
                &format!("missing progress for ingress {ingress:?}"),
            )
        })?;
        if self.state.deltas.needs_compaction {
            compact_base(&mut self.state, &self.name)?;
        }
        match ingress {
            "left" => {
                evict_opposite(
                    &mut self.state.right,
                    progress,
                    self.spec.bounds.before_micros,
                    &mut self.state.metrics.right,
                    JoinSide::Right,
                    &mut self.state.deltas.pending,
                    &self.name,
                )?;
            }
            "right" => {
                evict_opposite(
                    &mut self.state.left,
                    progress,
                    self.spec.bounds.after_micros,
                    &mut self.state.metrics.left,
                    JoinSide::Left,
                    &mut self.state.deltas.pending,
                    &self.name,
                )?;
            }
            _ => {
                return Err(operator_error(
                    &self.name,
                    &format!("unknown ingress progress {ingress:?}"),
                ));
            }
        }
        Ok(())
    }

    async fn on_watermark(
        &mut self,
        _watermark: EventTime,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        Ok(())
    }

    async fn on_end(
        &mut self,
        _context: &StreamOperatorContext<'_>,
        _output: &mut dyn StreamCollector,
    ) -> Result<()> {
        let left_identities = self
            .state
            .left
            .iter()
            .map(|row| (row.row_id, row.event_time, Arc::clone(&row.encoded_key)))
            .collect::<Vec<_>>();
        record_tombstones(
            &mut self.state.deltas.pending,
            JoinSide::Left,
            left_identities,
        );
        let right_identities = self
            .state
            .right
            .iter()
            .map(|row| (row.row_id, row.event_time, Arc::clone(&row.encoded_key)))
            .collect::<Vec<_>>();
        record_tombstones(
            &mut self.state.deltas.pending,
            JoinSide::Right,
            right_identities,
        );
        self.state.left.clear();
        self.state.right.clear();
        self.state.metrics.left.retained_rows = 0;
        self.state.metrics.left.retained_bytes = 0;
        self.state.metrics.right.retained_rows = 0;
        self.state.metrics.right.retained_bytes = 0;
        self.state.ended = true;
        Ok(())
    }

    fn reset(&mut self) -> Result<()> {
        self.state = StreamJoinState::default();
        Ok(())
    }

    fn checkpoint(&mut self, epoch: Epoch) -> Result<OperatorStateSnapshot> {
        if self
            .state
            .last_checkpoint_epoch
            .is_some_and(|previous| epoch <= previous)
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint epoch did not advance strictly",
                    self.name
                ),
            });
        }
        let metadata = JoinCheckpointMetadata {
            layout_version: 1,
            spec: self.spec.clone(),
            next_left_row_id: self.state.next_left_row_id,
            next_right_row_id: self.state.next_right_row_id,
            next_output_sequence: self.state.next_output_sequence,
            metrics: self.state.metrics.clone(),
            ended: self.state.ended,
            epoch: epoch.as_u64(),
        };
        let Value::Object(inline_metadata) =
            serde_json::to_value(metadata).map_err(|error| CalcFlowError::Internal {
                message: format!("stream Join checkpoint metadata encoding failed: {error}"),
            })?
        else {
            unreachable!("stream Join checkpoint metadata is an object")
        };
        // O(dirty/segment metadata) capture (spec FR47): prepared base and
        // carried delta segments share their allocations without copying, and
        // only the dirty ops since the last epoch encode here.
        let mut segments = BTreeMap::new();
        for (side, segment) in &self.state.deltas.base {
            segments.insert(format!("{side}-base"), segment.clone());
        }
        for ((segment_epoch, side), segment) in &self.state.deltas.segments {
            segments.insert(format!("{side}-delta-{segment_epoch}"), segment.clone());
        }
        if !self.state.deltas.pending.is_empty() {
            for (side, bytes) in encode_pending_delta(&self.state, epoch, &self.name)? {
                let segment = StateSegment::new(bytes);
                self.state
                    .deltas
                    .segments
                    .insert((epoch.as_u64(), side.as_str()), segment.clone());
                segments.insert(
                    format!("{}-delta-{}", side.as_str(), epoch.as_u64()),
                    segment,
                );
            }
            self.state.deltas.pending.clear();
            self.state.deltas.segments_since_base += 1;
            if self.state.deltas.segments_since_base >= JOIN_DELTA_COMPACTION_SEGMENTS {
                self.state.deltas.needs_compaction = true;
            }
        }
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = decode_join_metadata(snapshot, &self.name)?;
        if !checkpoint_metadata_compatible(&metadata, &self.spec) {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint layout or specification is incompatible",
                    self.name
                ),
            });
        }
        let (left, right) = self.decode_restored_sides(snapshot)?;
        self.validate_restored_join_rows(&metadata, &left, &right)?;
        restored_retained_metrics_match(&metadata.metrics, &left, &right, &self.name)?;
        self.validate_restored_limits(&left, &right)?;
        let carried = carried_delta_segments(snapshot, &self.name)?;
        let base = carried_base_segments(snapshot);
        let segments_since_base =
            u32::try_from(carried.len()).map_err(|_| counter_overflow(&self.name, "segments"))?;
        self.state = StreamJoinState {
            left,
            right,
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            last_checkpoint_epoch: Epoch::new(metadata.epoch),
            deltas: DeltaTracking {
                base,
                segments: carried,
                segments_since_base,
                ..DeltaTracking::default()
            },
        };
        Ok(())
    }
}

/// Compile-time per-ingress lookup for one input batch.
struct SidePlan {
    incoming_is_left: bool,
    port_index: usize,
    event_time_index: usize,
    key_indices: Vec<usize>,
}

impl AdmissionBundle {
    fn reserve_row_id(&mut self, operator_id: &str) -> Result<u64> {
        let row_id = self.next_row_id;
        self.next_row_id = self
            .next_row_id
            .checked_add(1)
            .ok_or_else(|| counter_overflow(operator_id, "row_id"))?;
        Ok(row_id)
    }

    fn note_dropped(&mut self, kind: DropKind, operator_id: &str) -> Result<()> {
        match kind {
            DropKind::NullEventTime => {
                self.metrics.null_event_time_rows = checked_metric(
                    self.metrics.null_event_time_rows,
                    1,
                    operator_id,
                    "null_event_time_rows",
                )?;
            }
            DropKind::NullKey => {
                self.metrics.null_key_rows =
                    checked_metric(self.metrics.null_key_rows, 1, operator_id, "null_key_rows")?;
            }
            DropKind::Late(lateness) => {
                self.metrics.late_rows =
                    checked_metric(self.metrics.late_rows, 1, operator_id, "late_rows")?;
                self.metrics.max_lateness_micros = Some(
                    self.metrics
                        .max_lateness_micros
                        .map_or(lateness, |current| current.max(lateness)),
                );
                self.had_late = true;
            }
        }
        Ok(())
    }

    fn push_admitted(&mut self, row: AdmittedRow) {
        self.admitted.push(row);
    }

    fn finish(&mut self, operator_id: &str) -> Result<()> {
        if self.had_late {
            self.metrics.late_affected_batches = checked_metric(
                self.metrics.late_affected_batches,
                1,
                operator_id,
                "late_affected_batches",
            )?;
        }
        Ok(())
    }
}

fn checkpoint_metadata_compatible(
    metadata: &JoinCheckpointMetadata,
    spec: &StreamJoinSpec,
) -> bool {
    metadata.layout_version == 1 && metadata.spec == *spec
}

fn decode_join_metadata(
    snapshot: &OperatorStateSnapshot,
    operator_id: &str,
) -> Result<JoinCheckpointMetadata> {
    serde_json::from_value::<JoinCheckpointMetadata>(Value::Object(
        snapshot.inline_metadata.clone().into_iter().collect(),
    ))
    .map_err(|error| CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} metadata is invalid: {error}"),
    })
}

/// Recomputes retained charges and rejects checkpoints that disagree with them.
fn restored_retained_metrics_match(
    metrics: &JoinMetrics,
    left: &[StoredRow],
    right: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    let mut left_metrics = metrics.left.clone();
    let mut right_metrics = metrics.right.clone();
    refresh_retained_metrics(&mut left_metrics, left, operator_id)?;
    refresh_retained_metrics(&mut right_metrics, right, operator_id)?;
    if side_retained_matches(&metrics.left, &left_metrics)
        && side_retained_matches(&metrics.right, &right_metrics)
    {
        return Ok(());
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} restored state charge is inconsistent"),
    })
}

fn side_retained_matches(recorded: &SideMetrics, recomputed: &SideMetrics) -> bool {
    recorded.retained_rows == recomputed.retained_rows
        && recorded.retained_bytes == recomputed.retained_bytes
}

/// Prospective (rows, bytes) charge if `retained` were installed next to `current`.
fn prospective_state_charge(
    current: &[StoredRow],
    retained: &[StoredRow],
    operator_id: &str,
) -> Result<(u64, u64)> {
    let rows = state_row_count(current, operator_id)?
        .checked_add(state_row_count(retained, operator_id)?)
        .ok_or_else(|| counter_overflow(operator_id, "state rows"))?;
    let bytes = current
        .iter()
        .chain(retained)
        .try_fold(0_u64, |total, row| total.checked_add(row.charge))
        .ok_or_else(|| counter_overflow(operator_id, "state bytes"))?;
    Ok((rows, bytes))
}

fn state_row_count(rows: &[StoredRow], operator_id: &str) -> Result<u64> {
    u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "state rows"))
}

/// Emits one prepared output row under the effective edge byte budget.
async fn emit_output_row(
    record: &RecordBatch,
    sequence: u64,
    context: &StreamOperatorContext<'_>,
    output: &mut dyn StreamCollector,
) -> Result<()> {
    let metadata = BatchMetadata::new(context.operator_id(), sequence, BTreeMap::new())?;
    let batch = Batch::table(vec![record.clone()], metadata)?;
    if batch.estimated_bytes()? > context.output_budget().max_bytes {
        return Err(CalcFlowError::InvalidArgument {
            field: "message.bytes".into(),
            message: "one stream Join output row exceeds the effective edge byte budget".into(),
        });
    }
    output.emit("output", batch).await
}

fn late_lateness(
    event_time: EventTime,
    progress: Option<IngressProgress>,
    operator_id: &str,
) -> Result<Option<u64>> {
    let Some(watermark) = progress.and_then(IngressProgress::watermark) else {
        return Ok(None);
    };
    if event_time >= watermark {
        return Ok(None);
    }
    let lateness =
        u64::try_from(i128::from(watermark.as_micros()) - i128::from(event_time.as_micros()))
            .map_err(|_| counter_overflow(operator_id, "lateness"))?;
    Ok(Some(lateness))
}

/// Runs the batched key-equality probe and returns the time-qualified pairs
/// in emission order.
async fn matched_pairs(
    runtime: &DataFusionRuntime,
    compiled: &CompiledJoin,
    spec: &StreamJoinSpec,
    plan: &SidePlan,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    operator_id: &str,
) -> Result<Vec<MatchedPair>> {
    let probe = probe_key_batch(admitted, &plan.key_indices)?;
    let state_keys = state_key_batch(opposite, compiled, plan)?;
    let tables = equality_tables(probe, state_keys)?;
    let result = runtime
        .sql(&compiled.equality_query, &tables, Some(operator_id))
        .await?;
    let equal_pairs = decode_key_pairs(&result)?;
    Ok(filter_and_order_pairs(
        &spec.bounds,
        plan,
        admitted,
        opposite,
        equal_pairs,
    ))
}

fn retained_rows(
    admitted: &[AdmittedRow],
    key_indices: &[usize],
    operator_id: &str,
) -> Result<Vec<StoredRow>> {
    admitted
        .iter()
        .filter(|row| row.retain)
        .map(|row| {
            let charge = state_row_charge(&row.record, 0, key_indices, operator_id)?;
            Ok(StoredRow {
                encoded_key: Arc::new(encode_join_key_v1(&row.record, 0, key_indices)?),
                record: row.record.clone(),
                event_time: row.event_time,
                row_id: row.row_id,
                charge,
            })
        })
        .collect()
}

fn enforce_match_limit(
    count: usize,
    failures: &mut u64,
    limit: u64,
    operator_id: &str,
) -> Result<()> {
    let count = u64::try_from(count).map_err(|_| counter_overflow(operator_id, "match_count"))?;
    if count > limit {
        *failures = checked_metric(*failures, 1, operator_id, "match_limit_failures")?;
        return Err(operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinMatchLimitExceeded,
            "input batch match limit exceeded",
        ));
    }
    Ok(())
}

/// Builds the admitted-row scratch table with renamed key columns.
fn probe_key_batch(admitted: &[AdmittedRow], key_indices: &[usize]) -> Result<RecordBatch> {
    let records = admitted.iter().map(|row| &row.record).collect::<Vec<_>>();
    let positions = UInt64Array::from_iter_values(
        (0..admitted.len()).map(|index| u64::try_from(index).expect("row count fits u64")),
    );
    key_probe_batch(&records, key_indices, PROBE_POS_COLUMN, &positions)
}

/// Builds the retained-state scratch table with renamed key columns and row ids.
fn state_key_batch(
    opposite: &[StoredRow],
    compiled: &CompiledJoin,
    plan: &SidePlan,
) -> Result<RecordBatch> {
    let key_indices = if plan.incoming_is_left {
        &compiled.right_key_indices
    } else {
        &compiled.left_key_indices
    };
    let records = opposite.iter().map(|row| &row.record).collect::<Vec<_>>();
    let row_ids = UInt64Array::from_iter_values(opposite.iter().map(|row| row.row_id));
    key_probe_batch(&records, key_indices, STATE_RID_COLUMN, &row_ids)
}

fn key_probe_batch(
    records: &[&RecordBatch],
    key_indices: &[usize],
    extra_name: &str,
    extra: &UInt64Array,
) -> Result<RecordBatch> {
    let first = records
        .first()
        .expect("join probe batches always have at least one row");
    let source_schema = first.schema();
    let mut fields = Vec::with_capacity(key_indices.len() + 1);
    let mut columns = Vec::with_capacity(key_indices.len() + 1);
    for (position, &key_index) in key_indices.iter().enumerate() {
        let source = source_schema.field(key_index);
        fields.push(Field::new(
            format!("{KEY_COLUMN_PREFIX}{position}"),
            source.data_type().clone(),
            source.is_nullable(),
        ));
        let slices = records
            .iter()
            .map(|record| record.column(key_index).as_ref())
            .collect::<Vec<_>>();
        columns.push(concat_key_column(&slices)?);
    }
    fields.push(Field::new(extra_name, DataType::UInt64, false));
    columns.push(Arc::new(extra.clone()));
    RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).map_err(|error| {
        CalcFlowError::Internal {
            message: format!("stream Join equality probe assembly failed: {error}"),
        }
    })
}

fn concat_key_column(slices: &[&dyn Array]) -> Result<ArrayRef> {
    concat(slices).map_err(|error| CalcFlowError::Internal {
        message: format!("stream Join key column concatenation failed: {error}"),
    })
}

fn equality_tables(probe: RecordBatch, state_keys: RecordBatch) -> Result<BTreeMap<String, Batch>> {
    Ok(BTreeMap::from([
        (
            PROBE_TABLE.into(),
            Batch::table(vec![probe], BatchMetadata::default())?,
        ),
        (
            STATE_TABLE.into(),
            Batch::table(vec![state_keys], BatchMetadata::default())?,
        ),
    ]))
}

fn decode_key_pairs(result: &Batch) -> Result<Vec<(u64, u64)>> {
    let mut pairs = Vec::new();
    for record in result.table_payload()?.batches() {
        let positions = u64_column(record, 0, "probe position")?;
        let row_ids = u64_column(record, 1, "state row id")?;
        for row_index in 0..record.num_rows() {
            pairs.push((positions.value(row_index), row_ids.value(row_index)));
        }
    }
    Ok(pairs)
}

fn u64_column<'a>(
    record: &'a RecordBatch,
    column_index: usize,
    field: &str,
) -> Result<&'a UInt64Array> {
    record
        .column(column_index)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("stream Join equality result is missing the {field} column"),
        })
}

fn filter_and_order_pairs(
    bounds: &JoinTimeBounds,
    plan: &SidePlan,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    equal_pairs: Vec<(u64, u64)>,
) -> Vec<MatchedPair> {
    let row_id_index = index_by_row_id(opposite);
    let mut matched = equal_pairs
        .into_iter()
        .filter_map(|(pos, rid)| {
            let pos = usize::try_from(pos).expect("probe positions index admitted rows");
            let opposite_index = *row_id_index
                .get(&rid)
                .expect("state row ids index retained rows");
            let incoming = &admitted[pos];
            let candidate = &opposite[opposite_index];
            let in_bounds = if plan.incoming_is_left {
                bounds.contains_pair(
                    incoming.event_time.as_micros(),
                    candidate.event_time.as_micros(),
                )
            } else {
                bounds.contains_pair(
                    candidate.event_time.as_micros(),
                    incoming.event_time.as_micros(),
                )
            };
            in_bounds.then_some(MatchedPair {
                pos,
                opposite_index,
            })
        })
        .collect::<Vec<_>>();
    matched.sort_by_key(|pair| {
        let row = &opposite[pair.opposite_index];
        (pair.pos, row.event_time, row.row_id)
    });
    matched
}

fn index_by_row_id(opposite: &[StoredRow]) -> BTreeMap<u64, usize> {
    opposite
        .iter()
        .enumerate()
        .map(|(index, row)| (row.row_id, index))
        .collect()
}

fn materialize_outputs(
    output_schema: &SchemaRef,
    admitted: &[AdmittedRow],
    opposite: &[StoredRow],
    matched: &[MatchedPair],
    incoming_is_left: bool,
    operator_id: &str,
) -> Result<Vec<RecordBatch>> {
    matched
        .iter()
        .map(|pair| {
            let incoming = &admitted[pair.pos];
            let candidate = &opposite[pair.opposite_index];
            let (left, right) = if incoming_is_left {
                (&incoming.record, &candidate.record)
            } else {
                (&candidate.record, &incoming.record)
            };
            output_record(output_schema, left, right, operator_id)
        })
        .collect()
}

fn output_record(
    output_schema: &SchemaRef,
    left: &RecordBatch,
    right: &RecordBatch,
    operator_id: &str,
) -> Result<RecordBatch> {
    let columns = left
        .columns()
        .iter()
        .chain(right.columns())
        .cloned()
        .collect::<Vec<_>>();
    RecordBatch::try_new(Arc::clone(output_schema), columns)
        .map_err(|error| operator_error(operator_id, &format!("output projection failed: {error}")))
}

fn exact_safe_duration_micros(duration: Duration, field: &str) -> Result<u64> {
    if duration.subsec_nanos() % 1_000 != 0 {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be an exact multiple of one microsecond".into(),
        });
    }
    let micros =
        u64::try_from(duration.as_micros()).map_err(|_| CalcFlowError::InvalidArgument {
            field: field.into(),
            message: format!("must be at most {STREAM_JOIN_MAX_SAFE_JSON_INTEGER}"),
        })?;
    validate_safe_integer(micros, false, field)?;
    Ok(micros)
}

fn validate_safe_integer(value: u64, positive: bool, field: &str) -> Result<()> {
    if (positive && value == 0) || value > STREAM_JOIN_MAX_SAFE_JSON_INTEGER {
        return Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: if positive {
                format!("must be in 1..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            } else {
                format!("must be in 0..={STREAM_JOIN_MAX_SAFE_JSON_INTEGER}")
            },
        });
    }
    Ok(())
}

fn validate_key_names(left: &[String], right: &[String]) -> Result<()> {
    if left.is_empty() || left.len() != right.len() {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.keys".into(),
            message: "left_keys and right_keys must be non-empty and equally sized".into(),
        });
    }
    for (side, keys) in [("left", left), ("right", right)] {
        if keys.iter().any(String::is_empty)
            || keys.iter().collect::<BTreeSet<_>>().len() != keys.len()
        {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("stream_join.{side}_keys"),
                message: "must contain unique non-empty column names".into(),
            });
        }
    }
    Ok(())
}

fn validate_column_name(value: &str, field: &str) -> Result<()> {
    if value.is_empty() {
        Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must name one column".into(),
        })
    } else {
        Ok(())
    }
}

fn validate_prefixes(left: &str, right: &str) -> Result<()> {
    if !is_portable_identifier(left) || !is_portable_identifier(right) || left == right {
        return Err(CalcFlowError::InvalidArgument {
            field: "stream_join.prefixes".into(),
            message: "must be distinct non-empty portable identifiers".into(),
        });
    }
    Ok(())
}

fn compile_schemas(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(SchemaRef, CompiledJoin)> {
    validate_unique_fields(left, "left")?;
    validate_unique_fields(right, "right")?;
    let (left_key_indices, right_key_indices) = compile_key_pair_indices(left, right, spec)?;
    let left_event_time_index = event_time_index(left, &spec.left_event_time, "left_event_time")?;
    let right_event_time_index =
        event_time_index(right, &spec.right_event_time, "right_event_time")?;
    let fields = prefixed_output_fields(left, right, spec);
    Ok((
        Arc::new(Schema::new(fields)),
        CompiledJoin {
            left_key_indices,
            right_key_indices,
            left_event_time_index,
            right_event_time_index,
            equality_query: equality_query(spec.left_keys.len()),
        },
    ))
}

fn compile_key_pair_indices(
    left: &Schema,
    right: &Schema,
    spec: &StreamJoinSpec,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut left_key_indices = Vec::with_capacity(spec.left_keys.len());
    let mut right_key_indices = Vec::with_capacity(spec.right_keys.len());
    for (index, (left_key, right_key)) in spec.left_keys.iter().zip(&spec.right_keys).enumerate() {
        let left_field = field_by_name(left, left_key, "left_keys", index)?;
        let right_field = field_by_name(right, right_key, "right_keys", index)?;
        validate_key_pair_types(index, left_field, right_field)?;
        left_key_indices.push(
            left.index_of(left_key)
                .expect("field lookup succeeded above"),
        );
        right_key_indices.push(
            right
                .index_of(right_key)
                .expect("field lookup succeeded above"),
        );
    }
    Ok((left_key_indices, right_key_indices))
}

fn validate_key_pair_types(index: usize, left_field: &Field, right_field: &Field) -> Result<()> {
    if left_field.data_type() != right_field.data_type()
        || !supported_key_type(left_field.data_type())
    {
        return Err(CalcFlowError::Compile {
            message: format!(
                "stream Join key pair {index} requires identical supported Arrow types; left is {} and right is {}",
                left_field.data_type(),
                right_field.data_type()
            ),
        });
    }
    Ok(())
}

fn event_time_index(schema: &Schema, name: &str, field: &str) -> Result<usize> {
    validate_event_time(schema, name, field)?;
    Ok(schema
        .index_of(name)
        .expect("event-time lookup succeeded above"))
}

fn prefixed_output_fields(left: &Schema, right: &Schema, spec: &StreamJoinSpec) -> Vec<Arc<Field>> {
    left.fields()
        .iter()
        .map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.left_prefix,
                field.name()
            )))
        })
        .chain(right.fields().iter().map(|field| {
            Arc::new(field.as_ref().clone().with_name(format!(
                "{}__{}",
                spec.right_prefix,
                field.name()
            )))
        }))
        .collect()
}

/// One batched key-equality probe over the per-batch scratch tables.
///
/// The query returns the admitted-row position and the retained-state row id
/// for every key-equal pair; the closed time bound is applied afterwards in
/// checked `i128` Rust arithmetic.
fn equality_query(key_count: usize) -> String {
    let equality = (0..key_count)
        .map(|position| {
            let column = quote_identifier(&format!("{KEY_COLUMN_PREFIX}{position}"));
            format!("{PROBE_TABLE}.{column} = {STATE_TABLE}.{column}")
        })
        .collect::<Vec<_>>()
        .join(" AND ");
    format!(
        "SELECT {PROBE_TABLE}.{pos}, {STATE_TABLE}.{rid} FROM {PROBE_TABLE} INNER JOIN {STATE_TABLE} ON {equality}",
        pos = quote_identifier(PROBE_POS_COLUMN),
        rid = quote_identifier(STATE_RID_COLUMN),
    )
}

fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

/// Reports whether the frozen v1 state-charge table covers `data_type`
/// recursively; a genuinely new Arrow payload type fails construction instead
/// of being charged as zero (spec FR16/D16).
fn payload_charge_supported(data_type: &DataType) -> bool {
    match data_type {
        DataType::Null
        | DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(_)
        | DataType::Decimal32(_, _)
        | DataType::Decimal64(_, _)
        | DataType::Decimal128(_, _)
        | DataType::Decimal256(_, _)
        | DataType::FixedSizeBinary(_)
        | DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => payload_charge_supported(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| payload_charge_supported(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .all(|(_type_id, field)| payload_charge_supported(field.data_type())),
        DataType::Dictionary(_, value) => payload_charge_supported(value),
        DataType::RunEndEncoded(_, values) => payload_charge_supported(values.data_type()),
    }
}

/// Rejects schemas whose payload types have no versioned state charge.
fn validate_payload_charge_support(schema: &Schema, side: &str) -> Result<()> {
    for field in schema.fields() {
        if !payload_charge_supported(field.data_type()) {
            return Err(CalcFlowError::InvalidArgument {
                field: format!("stream_join.{side}_schema.{}", field.name()),
                message: format!(
                    "unsupported_payload_type: {} has no versioned state charge",
                    field.data_type()
                ),
            });
        }
    }
    Ok(())
}

fn validate_unique_fields(schema: &Schema, side: &str) -> Result<()> {
    let mut names = BTreeSet::new();
    if schema.fields().is_empty()
        || schema
            .fields()
            .iter()
            .any(|field| !names.insert(field.name()))
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream Join {side} schema must be non-empty with unique field names"),
        });
    }
    Ok(())
}

fn field_by_name<'a>(
    schema: &'a Schema,
    name: &str,
    field: &str,
    index: usize,
) -> Result<&'a Field> {
    schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field}[{index}] names missing column {name:?}"),
        })
}

fn validate_event_time(schema: &Schema, name: &str, field: &str) -> Result<()> {
    let column = schema
        .field_with_name(name)
        .map_err(|_| CalcFlowError::Compile {
            message: format!("stream_join.{field} names missing column {name:?}"),
        })?;
    let DataType::Timestamp(_, timezone) = column.data_type() else {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} must be an Arrow timestamp"),
        });
    };
    if timezone
        .as_deref()
        .is_some_and(|timezone| timezone != "UTC")
    {
        return Err(CalcFlowError::Compile {
            message: format!("stream_join.{field} timestamp timezone must be UTC or absent"),
        });
    }
    Ok(())
}

pub(crate) fn supported_key_type(data_type: &DataType) -> bool {
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
            | DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Date32
            | DataType::Date64
            | DataType::Timestamp(_, _)
    )
}

const JOIN_STATE_MAGIC: &[u8; 8] = b"CFJOIN1\0";

/// Rebuilds one canonical base from live state, replacing every carried
/// delta (spec FR45). Compaction runs only inside data/progress handlers.
fn compact_base(state: &mut StreamJoinState, operator_id: &str) -> Result<()> {
    let left = encode_side(&state.left, operator_id, "left")?;
    let right = encode_side(&state.right, operator_id, "right")?;
    state.deltas.base = BTreeMap::from([
        ("left", StateSegment::new(left)),
        ("right", StateSegment::new(right)),
    ]);
    state.deltas.segments.clear();
    state.deltas.pending.clear();
    state.deltas.segments_since_base = 0;
    state.deltas.needs_compaction = false;
    Ok(())
}

/// Number of carried delta segments that triggers compaction on the next
/// data/progress handler (spec FR45).
const JOIN_DELTA_COMPACTION_SEGMENTS: u32 = 4;

const JOIN_DELTA_MAGIC: &[u8; 8] = b"CFJDLT1\0";
const JOIN_DELTA_UPSERT_TAG: u8 = 1;
const JOIN_DELTA_TOMBSTONE_TAG: u8 = 2;

/// Encodes the dirty ops of one epoch into per-side delta segments.
///
/// Upserts encode the records they carry from admission, so the encode cost is
/// proportional to the dirty set, never to the full state (spec FR47).
fn encode_pending_delta(
    state: &StreamJoinState,
    epoch: Epoch,
    operator_id: &str,
) -> Result<Vec<(JoinSide, Vec<u8>)>> {
    let mut encoded = Vec::new();
    for side in [JoinSide::Left, JoinSide::Right] {
        let ops = state
            .deltas
            .pending
            .iter()
            .filter(|op| op.side() == side)
            .collect::<Vec<_>>();
        if ops.is_empty() {
            continue;
        }
        let mut segment = Vec::new();
        segment.extend_from_slice(JOIN_DELTA_MAGIC);
        segment.extend_from_slice(&ops.len().to_le_bytes());
        for op in ops {
            segment.push(match op {
                PendingOp::Upsert { .. } => JOIN_DELTA_UPSERT_TAG,
                PendingOp::Tombstone { .. } => JOIN_DELTA_TOMBSTONE_TAG,
            });
            let (row_id, event_time, encoded_key) = match op {
                PendingOp::Upsert {
                    row_id,
                    event_time,
                    encoded_key,
                    ..
                }
                | PendingOp::Tombstone {
                    row_id,
                    event_time,
                    encoded_key,
                    ..
                } => (*row_id, *event_time, encoded_key.as_slice()),
            };
            segment.extend_from_slice(&row_id.to_le_bytes());
            segment.extend_from_slice(&event_time.as_micros().to_le_bytes());
            segment.extend_from_slice(
                &u64::try_from(encoded_key.len())
                    .map_err(|_| counter_overflow(operator_id, "encoded key length"))?
                    .to_le_bytes(),
            );
            segment.extend_from_slice(encoded_key);
            if let PendingOp::Upsert { record, charge, .. } = op {
                segment.extend_from_slice(&charge.to_le_bytes());
                let ipc = encode_row_ipc(record, operator_id, side.as_str())?;
                segment.extend_from_slice(
                    &u64::try_from(ipc.len())
                        .map_err(|_| counter_overflow(operator_id, "IPC length"))?
                        .to_le_bytes(),
                );
                segment.extend_from_slice(&ipc);
            }
        }
        let _ = epoch;
        encoded.push((side, segment));
    }
    Ok(encoded)
}

impl PendingOp {
    const fn side(&self) -> JoinSide {
        match self {
            PendingOp::Upsert { side, .. } | PendingOp::Tombstone { side, .. } => *side,
        }
    }
}

/// Encodes one stored row's Arrow IPC payload.
fn encode_row_ipc(record: &RecordBatch, operator_id: &str, side: &str) -> Result<Vec<u8>> {
    let mut ipc = Vec::new();
    {
        let mut writer =
            StreamWriter::try_new(&mut ipc, record.schema().as_ref()).map_err(|error| {
                CalcFlowError::Internal {
                    message: format!(
                        "stream Join {operator_id:?} {side} IPC writer failed: {error}"
                    ),
                }
            })?;
        writer
            .write(record)
            .and_then(|()| writer.finish())
            .map_err(|error| CalcFlowError::Internal {
                message: format!("stream Join {operator_id:?} {side} IPC encoding failed: {error}"),
            })?;
    }
    Ok(ipc)
}

/// Restores both sides by folding the base segment and the delta segments in
/// ascending `(epoch, segment_id)` order; later operations win (spec FR45).
fn restore_sides_from_segments(
    snapshot: &OperatorStateSnapshot,
    left_schema: &SchemaRef,
    right_schema: &SchemaRef,
    left_key_indices: &[usize],
    right_key_indices: &[usize],
    operator_id: &str,
) -> Result<(Vec<StoredRow>, Vec<StoredRow>)> {
    let mut inventory: Vec<(&str, SegmentKind)> = Vec::new();
    for segment_id in snapshot.segments.keys() {
        inventory.push((
            segment_id.as_str(),
            parse_segment_kind(segment_id, operator_id)?,
        ));
    }
    if inventory.is_empty() {
        return Err(CalcFlowError::CheckpointMismatch {
            message: format!("stream Join {operator_id:?} segment inventory is empty"),
        });
    }
    let mut left = fold_side(
        &snapshot.segments,
        &inventory,
        JoinSide::Left,
        left_schema,
        left_key_indices,
        operator_id,
    )?;
    let mut right = fold_side(
        &snapshot.segments,
        &inventory,
        JoinSide::Right,
        right_schema,
        right_key_indices,
        operator_id,
    )?;
    left.sort_by(identity_order);
    right.sort_by(identity_order);
    Ok((left, right))
}

fn identity_order(left: &StoredRow, right: &StoredRow) -> std::cmp::Ordering {
    (
        left.encoded_key.as_slice(),
        left.event_time.as_micros(),
        left.row_id,
    )
        .cmp(&(
            right.encoded_key.as_slice(),
            right.event_time.as_micros(),
            right.row_id,
        ))
}

/// Rebuilds the carried delta-segment map from a restored snapshot so the
/// next checkpoint carries them forward without re-encoding.
type CarriedDeltaSegments = BTreeMap<(u64, &'static str), StateSegment>;

fn carried_base_segments(snapshot: &OperatorStateSnapshot) -> BTreeMap<&'static str, StateSegment> {
    let mut base = BTreeMap::new();
    for (segment_id, segment) in &snapshot.segments {
        if let Some(side) = ["left", "right"]
            .into_iter()
            .find(|side| segment_id == &format!("{side}-base"))
        {
            base.insert(side, segment.clone());
        }
    }
    base
}

fn carried_delta_segments(
    snapshot: &OperatorStateSnapshot,
    operator_id: &str,
) -> Result<CarriedDeltaSegments> {
    let mut carried = BTreeMap::new();
    for (segment_id, segment) in &snapshot.segments {
        let side = ["left", "right"]
            .into_iter()
            .find(|side| segment_id.starts_with(side) && segment_id.contains("-delta-"));
        let Some(side) = side else {
            continue;
        };
        let epoch = segment_id
            .rsplit('-')
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .ok_or_else(|| CalcFlowError::CheckpointMismatch {
                message: format!("stream Join {operator_id:?} segment id is invalid"),
            })?;
        carried.insert((epoch, side), segment.clone());
    }
    Ok(carried)
}

enum SegmentKind {
    Base,
    Delta(u64),
}

fn parse_segment_kind(segment_id: &str, operator_id: &str) -> Result<SegmentKind> {
    for side in ["left", "right"] {
        if segment_id == format!("{side}-base") {
            return Ok(SegmentKind::Base);
        }
        if let Some(rest) = segment_id.strip_prefix(&format!("{side}-delta-")) {
            let epoch = rest
                .parse::<u64>()
                .map_err(|_| CalcFlowError::CheckpointMismatch {
                    message: format!("stream Join {operator_id:?} segment id is invalid"),
                })?;
            return Ok(SegmentKind::Delta(epoch));
        }
    }
    Err(CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} segment id is invalid"),
    })
}

/// Folds one side's base and deltas into durable-identity order.
fn fold_side(
    segments: &BTreeMap<String, StateSegment>,
    inventory: &[(&str, SegmentKind)],
    side: JoinSide,
    schema: &SchemaRef,
    key_indices: &[usize],
    operator_id: &str,
) -> Result<Vec<StoredRow>> {
    let mut folded: BTreeMap<(Vec<u8>, i64, u64), StoredRow> = BTreeMap::new();
    let side_str = side.as_str();
    let mut ordered: Vec<(&str, &SegmentKind, &StateSegment)> = inventory
        .iter()
        .filter(|(segment_id, _)| segment_id.starts_with(side_str))
        .map(|(segment_id, kind)| (*segment_id, kind, &segments[*segment_id]))
        .collect();
    // BTreeMap iteration gives ascending segment ids; the base folds first.
    ordered.sort_by_key(|(segment_id, kind, _)| match kind {
        SegmentKind::Base => (u64::MIN, (*segment_id).to_owned()),
        SegmentKind::Delta(epoch) => (*epoch, (*segment_id).to_owned()),
    });
    for (_segment_id, kind, segment) in ordered {
        let bytes = segment.bytes();
        match kind {
            SegmentKind::Base => {
                for row in decode_side(bytes, schema, key_indices, operator_id, side_str)? {
                    folded.insert(
                        (
                            row.encoded_key.to_vec(),
                            row.event_time.as_micros(),
                            row.row_id,
                        ),
                        row,
                    );
                }
            }
            SegmentKind::Delta(_) => {
                decode_delta_segment(
                    bytes,
                    schema,
                    key_indices,
                    operator_id,
                    side_str,
                    &mut folded,
                )?;
            }
        }
    }
    Ok(folded.into_values().collect())
}

/// Applies one delta segment's upserts and tombstones to the fold.
fn decode_delta_segment(
    bytes: &[u8],
    schema: &SchemaRef,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
    folded: &mut BTreeMap<(Vec<u8>, i64, u64), StoredRow>,
) -> Result<()> {
    let mut offset = 0_usize;
    if take_segment_bytes(bytes, &mut offset, JOIN_DELTA_MAGIC.len())? != JOIN_DELTA_MAGIC {
        return Err(checkpoint_error(
            operator_id,
            side,
            "delta magic is invalid",
        ));
    }
    let op_count = usize::try_from(read_segment_u64(bytes, &mut offset)?)
        .map_err(|_| checkpoint_error(operator_id, side, "delta op count is invalid"))?;
    let mut seen_identities = BTreeSet::new();
    for _ in 0..op_count {
        let tag = *take_segment_bytes(bytes, &mut offset, 1)?
            .first()
            .expect("one tag byte was taken");
        let row_id = read_segment_u64(bytes, &mut offset)?;
        let event_time = EventTime::from_micros(read_segment_i64(bytes, &mut offset)?);
        let key_length = usize::try_from(read_segment_u64(bytes, &mut offset)?)
            .map_err(|_| checkpoint_error(operator_id, side, "delta key length is invalid"))?;
        let encoded_key = take_segment_bytes(bytes, &mut offset, key_length)?.to_vec();
        let identity = (encoded_key.clone(), event_time.as_micros(), row_id);
        if !seen_identities.insert(identity.clone()) {
            return Err(checkpoint_error(
                operator_id,
                side,
                "delta segment repeats one row identity",
            ));
        }
        match tag {
            JOIN_DELTA_UPSERT_TAG => {
                let charge = read_segment_u64(bytes, &mut offset)?;
                let ipc_length = usize::try_from(read_segment_u64(bytes, &mut offset)?)
                    .map_err(|_| checkpoint_error(operator_id, side, "IPC length is invalid"))?;
                let ipc = take_segment_bytes(bytes, &mut offset, ipc_length)?;
                let record = decode_ipc_row(ipc, schema, operator_id, side)?;
                if encode_join_key_v1(&record, 0, key_indices)? != encoded_key {
                    return Err(checkpoint_error(
                        operator_id,
                        side,
                        "delta upsert key does not match its record",
                    ));
                }
                folded.insert(
                    identity,
                    StoredRow {
                        record,
                        event_time,
                        row_id,
                        charge,
                        encoded_key: Arc::new(encoded_key),
                    },
                );
            }
            JOIN_DELTA_TOMBSTONE_TAG => {
                folded.remove(&identity);
            }
            _ => {
                return Err(checkpoint_error(
                    operator_id,
                    side,
                    "delta op tag is invalid",
                ));
            }
        }
    }
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "delta segment has trailing bytes",
        ));
    }
    Ok(())
}

fn encode_side(rows: &[StoredRow], operator_id: &str, side: &str) -> Result<Vec<u8>> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by(|a, b| identity_order(a, b));
    let mut output = Vec::new();
    output.extend_from_slice(JOIN_STATE_MAGIC);
    output.extend_from_slice(
        &u64::try_from(ordered.len())
            .map_err(|_| counter_overflow(operator_id, "checkpoint rows"))?
            .to_le_bytes(),
    );
    for row in ordered {
        let mut ipc = Vec::new();
        {
            let mut writer = StreamWriter::try_new(&mut ipc, row.record.schema().as_ref())
                .map_err(|error| CalcFlowError::Internal {
                    message: format!(
                        "stream Join {operator_id:?} {side} IPC writer failed: {error}"
                    ),
                })?;
            writer
                .write(&row.record)
                .and_then(|()| writer.finish())
                .map_err(|error| CalcFlowError::Internal {
                    message: format!(
                        "stream Join {operator_id:?} {side} IPC encoding failed: {error}"
                    ),
                })?;
        }
        output.extend_from_slice(&row.row_id.to_le_bytes());
        output.extend_from_slice(&row.event_time.as_micros().to_le_bytes());
        output.extend_from_slice(&row.charge.to_le_bytes());
        output.extend_from_slice(
            &u64::try_from(ipc.len())
                .map_err(|_| counter_overflow(operator_id, "IPC length"))?
                .to_le_bytes(),
        );
        output.extend_from_slice(&ipc);
    }
    Ok(output)
}

fn decode_side(
    bytes: &[u8],
    expected_schema: &SchemaRef,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let mut offset = 0_usize;
    let row_count = decode_side_header(bytes, &mut offset, operator_id, side)?;
    let rows = decode_side_rows(
        bytes,
        &mut offset,
        row_count,
        expected_schema,
        key_indices,
        operator_id,
        side,
    )?;
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state segment has trailing bytes",
        ));
    }
    Ok(rows)
}

/// Validates the state magic and returns the declared row count.
fn decode_side_header(
    bytes: &[u8],
    offset: &mut usize,
    operator_id: &str,
    side: &str,
) -> Result<u64> {
    if take_segment_bytes(bytes, offset, JOIN_STATE_MAGIC.len())? != JOIN_STATE_MAGIC {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state magic is invalid",
        ));
    }
    read_segment_u64(bytes, offset)
}

fn decode_side_rows(
    bytes: &[u8],
    offset: &mut usize,
    row_count: u64,
    expected_schema: &SchemaRef,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let row_capacity = decode_row_capacity(row_count, bytes.len(), operator_id, side)?;
    let mut rows = Vec::with_capacity(row_capacity);
    for _ in 0..row_count {
        rows.push(decode_stored_row(
            bytes,
            offset,
            expected_schema,
            key_indices,
            operator_id,
            side,
        )?);
    }
    Ok(rows)
}

fn decode_row_capacity(
    row_count: u64,
    segment_len: usize,
    operator_id: &str,
    side: &str,
) -> Result<usize> {
    usize::try_from(row_count)
        .ok()
        .filter(|count| *count <= segment_len)
        .ok_or_else(|| checkpoint_error(operator_id, side, "row count is invalid"))
}

fn decode_stored_row(
    bytes: &[u8],
    offset: &mut usize,
    expected_schema: &SchemaRef,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<StoredRow> {
    let row_id = read_segment_u64(bytes, offset)?;
    let event_time = EventTime::from_micros(read_segment_i64(bytes, offset)?);
    let charge = read_segment_u64(bytes, offset)?;
    let ipc_length = usize::try_from(read_segment_u64(bytes, offset)?)
        .map_err(|_| checkpoint_error(operator_id, side, "IPC length is invalid"))?;
    let ipc = take_segment_bytes(bytes, offset, ipc_length)?;
    let record = decode_ipc_row(ipc, expected_schema, operator_id, side)?;
    let encoded_key = Arc::new(encode_join_key_v1(&record, 0, key_indices)?);
    Ok(StoredRow {
        record,
        event_time,
        row_id,
        charge,
        encoded_key,
    })
}

fn decode_ipc_row(
    ipc: &[u8],
    expected_schema: &SchemaRef,
    operator_id: &str,
    side: &str,
) -> Result<RecordBatch> {
    let mut reader = StreamReader::try_new(Cursor::new(ipc), None).map_err(|error| {
        checkpoint_error(
            operator_id,
            side,
            &format!("IPC header is invalid: {error}"),
        )
    })?;
    if reader.schema().as_ref() != expected_schema.as_ref() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC schema is incompatible",
        ));
    }
    let record = reader
        .next()
        .transpose()
        .map_err(|error| {
            checkpoint_error(operator_id, side, &format!("IPC row is invalid: {error}"))
        })?
        .filter(|record| record.num_rows() == 1)
        .ok_or_else(|| checkpoint_error(operator_id, side, "IPC must contain one row"))?;
    if reader.next().is_some() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "IPC contains extra record batches",
        ));
    }
    Ok(record)
}

fn take_segment_bytes<'a>(bytes: &'a [u8], offset: &mut usize, length: usize) -> Result<&'a [u8]> {
    let end = offset
        .checked_add(length)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment offset overflowed".into(),
        })?;
    let value = bytes
        .get(*offset..end)
        .ok_or_else(|| CalcFlowError::CheckpointMismatch {
            message: "stream Join state segment is truncated".into(),
        })?;
    *offset = end;
    Ok(value)
}

fn read_segment_u64(bytes: &[u8], offset: &mut usize) -> Result<u64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(u64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn read_segment_i64(bytes: &[u8], offset: &mut usize) -> Result<i64> {
    let value = take_segment_bytes(bytes, offset, 8)?;
    Ok(i64::from_le_bytes(
        value.try_into().expect("exact eight-byte segment slice"),
    ))
}

fn validate_restored_rows(
    rows: &[StoredRow],
    next_row_id: u64,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let mut identities = BTreeSet::new();
    for row in rows {
        validate_restored_row_identity(row, &mut identities, next_row_id, operator_id, side)?;
        validate_restored_row_payload(row, event_index, key_indices, operator_id, side)?;
    }
    Ok(())
}

fn validate_restored_row_identity(
    row: &StoredRow,
    identities: &mut BTreeSet<(EventTime, u64)>,
    next_row_id: u64,
    operator_id: &str,
    side: &str,
) -> Result<()> {
    if row.row_id >= next_row_id || !identities.insert((row.event_time, row.row_id)) {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row identity is invalid",
        ));
    }
    Ok(())
}

fn validate_restored_row_payload(
    row: &StoredRow,
    event_index: usize,
    key_indices: &[usize],
    operator_id: &str,
    side: &str,
) -> Result<()> {
    let restored_event_time = event_time_at(&row.record, event_index, 0, operator_id, side)?
        .ok_or_else(|| checkpoint_error(operator_id, side, "stored event time is null"))?;
    let restored_charge = state_row_charge(&row.record, 0, key_indices, operator_id)?;
    if restored_event_time != row.event_time || restored_charge != row.charge {
        return Err(checkpoint_error(
            operator_id,
            side,
            "row event time or charge is inconsistent",
        ));
    }
    Ok(())
}

fn checkpoint_error(operator_id: &str, side: &str, message: &str) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch {
        message: format!("stream Join {operator_id:?} {side} {message}"),
    }
}

fn should_retain(
    incoming_is_left: bool,
    event_time: EventTime,
    opposite: Option<IngressProgress>,
    bounds: JoinTimeBounds,
) -> bool {
    let Some(opposite) = opposite else {
        return true;
    };
    if opposite.state() == crate::IngressState::Ended {
        return false;
    }
    let Some(watermark) = opposite.watermark() else {
        return true;
    };
    let extension = if incoming_is_left {
        bounds.after_micros
    } else {
        bounds.before_micros
    };
    i128::from(event_time.as_micros()) + i128::from(extension) >= i128::from(watermark.as_micros())
}

fn evict_opposite(
    rows: &mut Vec<StoredRow>,
    progress: IngressProgress,
    extension_micros: u64,
    metrics: &mut SideMetrics,
    side: JoinSide,
    pending: &mut Vec<PendingOp>,
    operator_id: &str,
) -> Result<()> {
    let before = rows.len();
    let mut evicted_identities = Vec::new();
    if progress.state() == crate::IngressState::Ended {
        evicted_identities.extend(
            rows.iter()
                .map(|row| (row.row_id, row.event_time, Arc::clone(&row.encoded_key))),
        );
        rows.clear();
    } else if let Some(watermark) = progress.watermark() {
        let mut index = 0;
        while index < rows.len() {
            let expired = i128::from(rows[index].event_time.as_micros())
                + i128::from(extension_micros)
                < i128::from(watermark.as_micros());
            if expired {
                let row = rows.remove(index);
                evicted_identities.push((row.row_id, row.event_time, Arc::clone(&row.encoded_key)));
            } else {
                index += 1;
            }
        }
    }
    record_tombstones(pending, side, evicted_identities);
    let evicted = u64::try_from(before - rows.len())
        .map_err(|_| counter_overflow(operator_id, "evicted rows"))?;
    metrics.evicted_rows =
        checked_metric(metrics.evicted_rows, evicted, operator_id, "evicted_rows")?;
    refresh_retained_metrics(metrics, rows, operator_id)
}

/// Records evictions in the dirty log: an upsert still waiting for its first
/// checkpoint coalesces away; a captured row leaves a durable tombstone.
fn record_tombstones(
    pending: &mut Vec<PendingOp>,
    side: JoinSide,
    evicted: Vec<(u64, EventTime, Arc<Vec<u8>>)>,
) {
    // Pending identities are unique per side, so one indexed pass keeps the
    // coalescing cost proportional to the dirty log, never quadratic.
    let pending_identities: BTreeSet<(JoinSide, u64)> =
        pending.iter().map(PendingOp::identity).collect();
    let coalesced: BTreeSet<(JoinSide, u64)> = evicted
        .iter()
        .map(|(row_id, _, _)| (side, *row_id))
        .filter(|identity| pending_identities.contains(identity))
        .collect();
    if !coalesced.is_empty() {
        pending.retain(|op| !coalesced.contains(&op.identity()));
    }
    for (row_id, event_time, encoded_key) in evicted {
        if coalesced.contains(&(side, row_id)) {
            continue;
        }
        pending.push(PendingOp::Tombstone {
            side,
            row_id,
            event_time,
            encoded_key,
        });
    }
}

fn refresh_retained_metrics(
    metrics: &mut SideMetrics,
    rows: &[StoredRow],
    operator_id: &str,
) -> Result<()> {
    metrics.retained_rows =
        u64::try_from(rows.len()).map_err(|_| counter_overflow(operator_id, "retained rows"))?;
    metrics.retained_bytes = rows
        .iter()
        .try_fold(0_u64, |total, row| total.checked_add(row.charge))
        .ok_or_else(|| counter_overflow(operator_id, "retained bytes"))?;
    Ok(())
}

fn event_time_at(
    record: &RecordBatch,
    column_index: usize,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<Option<EventTime>> {
    let array = record.column(column_index).as_ref();
    if array.is_null(row_index) {
        return Ok(None);
    }
    let data_type = record.schema().field(column_index).data_type().clone();
    let value = match &data_type {
        DataType::Timestamp(TimeUnit::Second, _) => {
            downcast_timestamp::<TimestampSecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            downcast_timestamp::<TimestampMillisecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            downcast_timestamp::<TimestampMicrosecondArray>(array, row_index, operator_id, side)?
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            downcast_timestamp::<TimestampNanosecondArray>(array, row_index, operator_id, side)?
        }
        _ => {
            return Err(operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time is not a timestamp"),
            ));
        }
    };
    EventTime::import_timestamp(value, &data_type, &format!("stream_join.{side}_event_time"))
        .map(Some)
        .map_err(|_| {
            operator_reason(
                operator_id,
                crate::StreamingFailureReason::JoinTimeConversionFailed,
                &format!("{side} event time cannot be represented"),
            )
        })
}

fn downcast_timestamp<T>(
    array: &dyn Array,
    row_index: usize,
    operator_id: &str,
    side: &str,
) -> Result<i64>
where
    T: Array + 'static,
    for<'a> &'a T: TimestampValue,
{
    let typed = array.as_any().downcast_ref::<T>().ok_or_else(|| {
        operator_reason(
            operator_id,
            crate::StreamingFailureReason::JoinTimeConversionFailed,
            &format!("{side} timestamp array type mismatch"),
        )
    })?;
    Ok(typed.timestamp_value(row_index))
}

trait TimestampValue {
    fn timestamp_value(self, row_index: usize) -> i64;
}

macro_rules! impl_timestamp_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl TimestampValue for &$ty {
                fn timestamp_value(self, row_index: usize) -> i64 {
                    self.value(row_index)
                }
            }
        )+
    };
}

impl_timestamp_value!(
    TimestampSecondArray,
    TimestampMillisecondArray,
    TimestampMicrosecondArray,
    TimestampNanosecondArray,
);

fn state_row_charge(
    record: &RecordBatch,
    row_index: usize,
    key_indices: &[usize],
    operator_id: &str,
) -> Result<u64> {
    let encoded_key = encode_join_key_v1(record, row_index, key_indices)?;
    let key_bytes = u64::try_from(encoded_key.len())
        .map_err(|_| counter_overflow(operator_id, "encoded key"))?;
    let payload = record.columns().iter().try_fold(0_u64, |total, column| {
        let charge = logical_cell_charge(column.as_ref(), row_index)?;
        total
            .checked_add(charge)
            .ok_or_else(|| counter_overflow(operator_id, "logical payload bytes"))
    })?;
    STREAM_JOIN_STATE_ROW_OVERHEAD_BYTES_V1
        .checked_add(key_bytes)
        .and_then(|value| value.checked_add(16))
        .and_then(|value| value.checked_add(payload))
        .ok_or_else(|| counter_overflow(operator_id, "state row charge"))
}

/// Logical charge of one non-null cell under the frozen V1 accounting table,
/// including its validity byte.
fn logical_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    if array.is_null(row_index) {
        return Ok(1);
    }
    let data_type = array.data_type();
    if let Some(value) = fixed_cell_charge(data_type) {
        return validity_wrapped(value);
    }
    let Some(value) = variable_cell_charge(array, row_index)? else {
        return sized_cell_charge(array, row_index).and_then(validity_wrapped);
    };
    validity_wrapped(value)
}

fn fixed_cell_charge(data_type: &DataType) -> Option<u64> {
    if matches!(data_type, DataType::Null) {
        return Some(0);
    }
    if matches!(
        data_type,
        DataType::Boolean | DataType::Int8 | DataType::UInt8
    ) {
        return Some(1);
    }
    if matches!(
        data_type,
        DataType::Int16 | DataType::UInt16 | DataType::Float16
    ) {
        return Some(2);
    }
    if is_four_byte_cell(data_type) {
        return Some(4);
    }
    if is_eight_byte_cell(data_type) {
        return Some(8);
    }
    fixed_wide_cell_charge(data_type)
}

fn fixed_wide_cell_charge(data_type: &DataType) -> Option<u64> {
    if is_sixteen_byte_cell(data_type) {
        return Some(16);
    }
    if matches!(data_type, DataType::Decimal256(_, _)) {
        return Some(32);
    }
    None
}

fn is_four_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int32
            | DataType::UInt32
            | DataType::Float32
            | DataType::Date32
            | DataType::Time32(_)
            | DataType::Interval(IntervalUnit::YearMonth)
            | DataType::Decimal32(_, _)
    )
}

fn is_eight_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Int64
            | DataType::UInt64
            | DataType::Float64
            | DataType::Date64
            | DataType::Time64(_)
            | DataType::Timestamp(_, _)
            | DataType::Duration(_)
            | DataType::Interval(IntervalUnit::DayTime)
            | DataType::Decimal64(_, _)
    )
}

fn is_sixteen_byte_cell(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Interval(IntervalUnit::MonthDayNano) | DataType::Decimal128(_, _)
    )
}

fn variable_cell_charge(array: &dyn Array, row_index: usize) -> Result<Option<u64>> {
    if let DataType::FixedSizeBinary(size) = array.data_type() {
        return fixed_size_binary_charge(*size).map(Some);
    }
    string_cell_charge(array, row_index)
        .or(binary_cell_charge(array, row_index))
        .transpose()
}

fn fixed_size_binary_charge(size: i32) -> Result<u64> {
    u64::try_from(size).map_err(|_| CalcFlowError::Internal {
        message: "negative FixedSizeBinary width".into(),
    })
}

fn string_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Utf8 => Some(downcast_cell_charge::<StringArray>(
            array, row_index, 4, "Utf8",
        )),
        DataType::LargeUtf8 => Some(downcast_cell_charge::<LargeStringArray>(
            array,
            row_index,
            8,
            "LargeUtf8",
        )),
        DataType::Utf8View => Some(downcast_cell_charge::<StringViewArray>(
            array, row_index, 16, "Utf8View",
        )),
        _ => None,
    }
}

fn binary_cell_charge(array: &dyn Array, row_index: usize) -> Option<Result<u64>> {
    match array.data_type() {
        DataType::Binary => Some(downcast_cell_charge::<BinaryArray>(
            array, row_index, 4, "Binary",
        )),
        DataType::LargeBinary => Some(downcast_cell_charge::<LargeBinaryArray>(
            array,
            row_index,
            8,
            "LargeBinary",
        )),
        DataType::BinaryView => Some(downcast_cell_charge::<BinaryViewArray>(
            array,
            row_index,
            16,
            "BinaryView",
        )),
        _ => None,
    }
}

/// Downcasts one variable-length array kind and charges prefix plus value.
fn downcast_cell_charge<T>(
    array: &dyn Array,
    row_index: usize,
    prefix: u64,
    label: &str,
) -> Result<u64>
where
    T: Array + 'static,
    for<'a> &'a T: CellBytes,
{
    let typed = array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} array type mismatch"),
        })?;
    prefix_cell_charge(prefix, typed.cell_len(row_index), label)
}

fn prefix_cell_charge(prefix: u64, len: usize, label: &str) -> Result<u64> {
    prefix
        .checked_add(u64::try_from(len).unwrap_or(u64::MAX))
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("{label} cell charge overflow"),
        })
}

trait CellBytes {
    fn cell_len(self, row_index: usize) -> usize;
}

macro_rules! impl_cell_bytes {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl CellBytes for &$ty {
                fn cell_len(self, row_index: usize) -> usize {
                    self.value(row_index).len()
                }
            }
        )+
    };
}

impl_cell_bytes!(
    StringArray,
    LargeStringArray,
    StringViewArray,
    BinaryArray,
    LargeBinaryArray,
    BinaryViewArray,
);

/// Charges nested and dictionary-encoded cells from the frozen logical table
/// (state-byte accounting v1); every traversal is by logical value, never by
/// buffer capacity.
fn sized_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    match array.data_type() {
        DataType::List(_) => {
            let typed = list_array::<ListArray>(array)?;
            list_cell_charge(typed.value(row_index).as_ref(), 4)
        }
        DataType::LargeList(_) => {
            let typed = list_array::<LargeListArray>(array)?;
            list_cell_charge(typed.value(row_index).as_ref(), 8)
        }
        DataType::Map(_, _) => {
            let typed = list_array::<MapArray>(array)?;
            let entries = typed.value(row_index);
            let entries: &dyn Array = &entries;
            list_cell_charge(entries, 4)
        }
        DataType::ListView(_) => {
            let typed = list_array::<ListViewArray>(array)?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 8)
        }
        DataType::LargeListView(_) => {
            let typed = list_array::<LargeListViewArray>(array)?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 16)
        }
        DataType::FixedSizeList(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| charge_type_mismatch("FixedSizeList"))?;
            let child = typed.value(row_index);
            let child: &dyn Array = &child;
            list_cell_charge(child, 0)
        }
        DataType::Struct(_) => {
            let typed = array
                .as_any()
                .downcast_ref::<StructArray>()
                .ok_or_else(|| charge_type_mismatch("Struct"))?;
            typed.columns().iter().try_fold(0_u64, |total, column| {
                total
                    .checked_add(logical_cell_charge(column.as_ref(), row_index)?)
                    .ok_or_else(|| charge_overflow("struct cell"))
            })
        }
        DataType::Union(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<UnionArray>()
                .ok_or_else(|| charge_type_mismatch("Union"))?;
            let child = typed.child(typed.type_id(row_index));
            let charge = logical_cell_charge(child.as_ref(), typed.value_offset(row_index))?;
            charge
                .checked_add(1)
                .ok_or_else(|| charge_overflow("union cell"))
        }
        DataType::Dictionary(_, _) => {
            let typed = array
                .as_any()
                .downcast_ref::<DictionaryArray<Int32Type>>()
                .ok_or_else(|| charge_type_mismatch("Dictionary"))?;
            let index =
                typed
                    .keys()
                    .value(row_index)
                    .try_into()
                    .map_err(|_| CalcFlowError::Internal {
                        message: "dictionary key does not fit usize".into(),
                    })?;
            logical_cell_charge(typed.values().as_ref(), index)
        }
        DataType::RunEndEncoded(..) => {
            let typed = array
                .as_any()
                .downcast_ref::<RunArray<Int32Type>>()
                .ok_or_else(|| charge_type_mismatch("RunEndEncoded"))?;
            logical_cell_charge(typed.values().as_ref(), typed.get_physical_index(row_index))
        }
        _ => Err(unsupported_payload_type(array.data_type())),
    }
}

/// Charges one list-like child slice: prefix plus each child cell.
fn list_cell_charge(child: &(dyn Array + '_), prefix: u64) -> Result<u64> {
    let sum = (0..child.len()).try_fold(0_u64, |total, index| {
        total
            .checked_add(logical_cell_charge(child, index)?)
            .ok_or_else(|| charge_overflow("list child cell"))
    })?;
    prefix
        .checked_add(sum)
        .ok_or_else(|| charge_overflow("list cell"))
}

fn list_array<'a, T>(array: &'a dyn Array) -> Result<&'a T>
where
    &'a T: ArrayAccessor,
    T: 'static + Array,
{
    array
        .as_any()
        .downcast_ref::<T>()
        .ok_or_else(|| charge_type_mismatch("list"))
}

fn charge_type_mismatch(label: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("{label} array type mismatch in logical charge"),
    }
}

fn charge_overflow(label: &str) -> CalcFlowError {
    CalcFlowError::Internal {
        message: format!("logical {label} charge overflow"),
    }
}

/// Reports the construction-time rejection for payload types outside the
/// frozen state-byte accounting table (spec FR16/D16).
fn unsupported_payload_type(data_type: &DataType) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "stream_join.schema".into(),
        message: format!("unsupported_payload_type: {data_type} has no versioned state charge"),
    }
}

/// Version-1 type-tagged, length-delimited join key encoding (spec FR44).
///
/// Each key column contributes one block: tag byte, timezone length, timezone
/// bytes, value length, and raw value bytes. The encoding is stable and
/// unambiguous across units, timezone metadata, and column order.
fn encode_join_key_v1(
    record: &RecordBatch,
    row_index: usize,
    key_indices: &[usize],
) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    for &index in key_indices {
        append_key_block(&mut encoded, record.column(index).as_ref(), row_index)?;
    }
    Ok(encoded)
}

fn append_key_block(encoded: &mut Vec<u8>, array: &dyn Array, row_index: usize) -> Result<()> {
    let data_type = array.data_type();
    let tag = key_type_tag(data_type)?;
    let timezone = match data_type {
        DataType::Timestamp(_, Some(tz)) => tz.as_bytes(),
        _ => &[],
    };
    let value = key_value_bytes(array, row_index)?;
    encoded.push(tag);
    encoded.extend_from_slice(
        &u32::try_from(timezone.len())
            .map_err(|_| charge_overflow("key timezone length"))?
            .to_le_bytes(),
    );
    encoded.extend_from_slice(timezone);
    encoded.extend_from_slice(
        &u32::try_from(value.len())
            .map_err(|_| charge_overflow("key value length"))?
            .to_le_bytes(),
    );
    encoded.extend_from_slice(&value);
    Ok(())
}

fn key_type_tag(data_type: &DataType) -> Result<u8> {
    Ok(match data_type {
        DataType::Boolean => 1,
        DataType::Int8 => 2,
        DataType::Int16 => 3,
        DataType::Int32 => 4,
        DataType::Int64 => 5,
        DataType::UInt8 => 6,
        DataType::UInt16 => 7,
        DataType::UInt32 => 8,
        DataType::UInt64 => 9,
        DataType::Utf8 => 10,
        DataType::LargeUtf8 => 11,
        DataType::Date32 => 12,
        DataType::Date64 => 13,
        DataType::Timestamp(TimeUnit::Second, _) => 14,
        DataType::Timestamp(TimeUnit::Millisecond, _) => 15,
        DataType::Timestamp(TimeUnit::Microsecond, _) => 16,
        DataType::Timestamp(TimeUnit::Nanosecond, _) => 17,
        _ => {
            return Err(CalcFlowError::Internal {
                message: format!("join key type {data_type} has no version-1 tag"),
            });
        }
    })
}

fn key_value_bytes(array: &dyn Array, row_index: usize) -> Result<Vec<u8>> {
    let data_type = array.data_type().clone();
    if let DataType::Timestamp(unit, _) = &data_type {
        return timestamp_key_bytes(array, row_index, *unit);
    }
    Ok(match &data_type {
        DataType::Boolean => {
            let typed = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| charge_type_mismatch("Boolean"))?;
            vec![u8::from(typed.value(row_index))]
        }
        DataType::Int8 => {
            let typed = primitive::<Int8Type>(array)?;
            vec![
                u8::try_from(typed.value(row_index)).map_err(|_| CalcFlowError::Internal {
                    message: "int8 key does not fit u8".into(),
                })?,
            ]
        }
        DataType::Int16 => primitive::<Int16Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::Int32 | DataType::Date32 => primitive::<Int32Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::Int64 | DataType::Date64 => primitive::<Int64Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::UInt8 => vec![primitive::<UInt8Type>(array)?.value(row_index)],
        DataType::UInt16 => primitive::<UInt16Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::UInt32 => primitive::<UInt32Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::UInt64 => primitive::<UInt64Type>(array)?
            .value(row_index)
            .to_le_bytes()
            .to_vec(),
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .ok_or_else(|| charge_type_mismatch("Utf8 key"))?
            .value(row_index)
            .as_bytes()
            .to_vec(),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .ok_or_else(|| charge_type_mismatch("LargeUtf8 key"))?
            .value(row_index)
            .as_bytes()
            .to_vec(),
        _ => {
            return Err(CalcFlowError::Internal {
                message: format!("join key type {data_type} has no version-1 encoding"),
            });
        }
    })
}

fn primitive<T>(array: &dyn Array) -> Result<&PrimitiveArray<T>>
where
    T: ArrowPrimitiveType,
{
    array
        .as_any()
        .downcast_ref::<PrimitiveArray<T>>()
        .ok_or_else(|| charge_type_mismatch("primitive key"))
}

fn validity_wrapped(value_bytes: u64) -> Result<u64> {
    value_bytes
        .checked_add(1)
        .ok_or_else(|| CalcFlowError::Internal {
            message: "logical cell charge overflow".into(),
        })
}

fn checked_metric(current: u64, delta: u64, operator_id: &str, field: &str) -> Result<u64> {
    current
        .checked_add(delta)
        .ok_or_else(|| counter_overflow(operator_id, field))
}

fn counter_overflow(operator_id: &str, field: &str) -> CalcFlowError {
    operator_reason(
        operator_id,
        crate::StreamingFailureReason::JoinCounterOverflow,
        &format!("{field} counter overflowed"),
    )
}

fn operator_reason(
    operator_id: &str,
    reason_code: crate::StreamingFailureReason,
    message: &str,
) -> CalcFlowError {
    CalcFlowError::OperatorReason {
        node_id: operator_id.into(),
        reason_code,
        message: message.into(),
    }
}

fn operator_error(operator_id: &str, message: &str) -> CalcFlowError {
    CalcFlowError::Operator {
        node_id: operator_id.into(),
        message: message.into(),
    }
}

fn default_left_prefix() -> String {
    "left".into()
}

fn default_right_prefix() -> String {
    "right".into()
}

/// Encodes one timestamp key value as its native unit's little-endian bytes.
/// The key block already carries the unit-specific type tag and timezone, so
/// cross-unit unambiguity does not depend on converting to microseconds here.
fn timestamp_key_bytes(array: &dyn Array, row_index: usize, unit: TimeUnit) -> Result<Vec<u8>> {
    let raw = match unit {
        TimeUnit::Second => array
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Millisecond => array
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Microsecond => array
            .as_any()
            .downcast_ref::<TimestampMicrosecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
        TimeUnit::Nanosecond => array
            .as_any()
            .downcast_ref::<TimestampNanosecondArray>()
            .ok_or_else(|| charge_type_mismatch("timestamp key"))?
            .value(row_index),
    };
    Ok(raw.to_le_bytes().to_vec())
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{
            BinaryArray, FixedSizeBinaryArray, Int64Array, StringArray, TimestampMicrosecondArray,
        },
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{
        BatchMetadata, CancellationToken, EdgeBudget, EdgeCollector, IngressProgress,
        IngressProgressSnapshot, IngressState, JsonMap, OperatorMetadata, StreamJobContext,
        StreamMessageKind, StreamOperator,
    };

    fn left_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]))
    }

    fn right_schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]))
    }

    fn spec() -> StreamJoinSpec {
        StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1_000).unwrap(),
        )
        .unwrap()
        .with_prefixes("authorization", "payment")
        .unwrap()
    }

    fn left_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    left_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(Int64Array::from(vec![42; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    fn right_batch(times: Vec<i64>) -> Batch {
        let rows = times.len();
        Batch::table(
            vec![
                RecordBatch::try_new(
                    right_schema(),
                    vec![
                        Arc::new(Int64Array::from(vec![7; rows])),
                        Arc::new(TimestampMicrosecondArray::from(times).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"; rows])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap()
    }

    #[test]
    fn derives_exact_prefixed_ports() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();

        assert_eq!(
            operator
                .input_ports()
                .iter()
                .map(Port::name)
                .collect::<Vec<_>>(),
            ["left", "right"]
        );
        let output = operator.output_ports()[0].schema().unwrap();
        assert_eq!(
            output
                .fields()
                .iter()
                .map(|field| field.name().as_str())
                .collect::<Vec<_>>(),
            [
                "authorization__account_id",
                "authorization__authorized_at",
                "authorization__amount",
                "payment__account_id",
                "payment__paid_at",
                "payment__status",
            ]
        );
        assert!(output.field(2).is_nullable());
        assert!(output.field(5).is_nullable());
    }

    #[test]
    fn rejects_values_beyond_the_json_safe_integer_domain() {
        let too_large = STREAM_JOIN_MAX_SAFE_JSON_INTEGER + 1;
        assert!(JoinStateLimits::new(too_large, 1, 1).is_err());
        assert!(JoinTimeBounds::new(Duration::from_micros(too_large), Duration::ZERO).is_err());
        assert!(JoinStateLimits::new(1, 1, 1).is_ok());
    }

    #[test]
    fn inclusive_bound_helper_accepts_both_edges() {
        let bounds =
            JoinTimeBounds::new(Duration::from_micros(10), Duration::from_micros(20)).unwrap();
        assert!(bounds.contains_pair(100, 90));
        assert!(bounds.contains_pair(100, 120));
        assert!(!bounds.contains_pair(100, 89));
        assert!(!bounds.contains_pair(100, 121));
    }

    #[test]
    fn output_frontier_uses_live_idle_and_ended_formulas() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let snapshot = |left_state: IngressState,
                        left: Option<i64>,
                        right_state: IngressState,
                        right: Option<i64>| {
            IngressProgressSnapshot::new(BTreeMap::from([
                (
                    "left".into(),
                    IngressProgress::new(left_state, left.map(EventTime::from_micros)),
                ),
                (
                    "right".into(),
                    IngressProgress::new(right_state, right.map(EventTime::from_micros)),
                ),
            ]))
        };
        let micros = |value: Option<EventTime>| value.map(EventTime::as_micros);

        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Idle,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Active,
                        Some(400_000_000),
                        IngressState::Ended,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(100_000_000)
        );
        assert_eq!(
            micros(
                operator
                    .output_frontier_candidate(&snapshot(
                        IngressState::Ended,
                        Some(400_000_000),
                        IngressState::Active,
                        Some(100_000_000),
                    ))
                    .unwrap()
            ),
            Some(40_000_000)
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Ended,
                    Some(400_000_000),
                    IngressState::Ended,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
        assert_eq!(
            operator
                .output_frontier_candidate(&snapshot(
                    IngressState::Active,
                    None,
                    IngressState::Active,
                    Some(100_000_000),
                ))
                .unwrap(),
            None
        );
    }

    #[tokio::test]
    async fn emits_duplicate_pairs_at_both_inclusive_boundaries() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job = StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        );
        let context = StreamOperatorContext::new(&job, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![100, 100]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data(
                "right",
                right_batch(vec![-299_999_900, 60_000_100, 60_000_101]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 4);
        assert!(
            outputs
                .iter()
                .all(|message| message.kind() == StreamMessageKind::Data)
        );
        assert_eq!(
            outputs
                .iter()
                .map(|message| message.as_data().unwrap().metadata().sequence())
                .collect::<Vec<_>>(),
            [0, 1, 2, 3]
        );
    }

    fn job() -> StreamJobContext {
        StreamJobContext::new(
            1,
            "fingerprint",
            JsonMap::new(),
            None,
            CancellationToken::new(),
        )
    }

    fn progress_context(
        job_context: &StreamJobContext,
        left: (IngressState, Option<i64>),
        right: (IngressState, Option<i64>),
    ) -> StreamOperatorContext<'_> {
        let snapshot = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(left.0, left.1.map(EventTime::from_micros)),
            ),
            (
                "right".into(),
                IngressProgress::new(right.0, right.1.map(EventTime::from_micros)),
            ),
        ]));
        StreamOperatorContext::for_task(
            job_context,
            "match",
            None,
            snapshot,
            EdgeBudget::default(),
            Arc::new(NoopLateMetrics),
        )
    }

    struct NoopLateMetrics;

    impl crate::operator::LateMetricSink for NoopLateMetrics {
        fn record(&self, _delta: crate::operator::LateMetricDelta) -> Result<()> {
            Ok(())
        }
    }

    fn reason_of(error: &CalcFlowError) -> Option<crate::StreamingFailureReason> {
        match error {
            CalcFlowError::OperatorReason { reason_code, .. } => Some(*reason_code),
            _ => None,
        }
    }

    fn checkpoint_metadata(operator: &mut StreamJoinOperator, epoch: u64) -> JsonMap {
        operator
            .checkpoint(Epoch::new(epoch).unwrap())
            .unwrap()
            .inline_metadata
    }

    #[tokio::test]
    async fn late_rows_are_dropped_with_metrics_and_never_retained() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = progress_context(
            &job_context,
            (IngressState::Active, Some(500_000_000)),
            (IngressState::Active, None),
        );
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data(
                "left",
                left_batch(vec![100_000_000, 499_999_999]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        let state = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(state.contains("\"late_rows\":2"), "{state}");
        assert!(
            state.contains("\"late_affected_batches\":1")
                && state.contains("\"max_lateness_micros\":400000000"),
            "{state}"
        );
        assert!(
            state.contains("\"retained_rows\":0"),
            "late rows must never be retained: {state}"
        );
    }

    #[tokio::test]
    async fn null_event_time_and_null_key_rows_are_counted_not_stored() {
        let nullable_time_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                true,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let mut operator = StreamJoinOperator::new(
            "match",
            Arc::clone(&nullable_time_schema),
            Arc::clone(&nullable_key_schema),
            spec(),
        )
        .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let null_time = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_time_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(
                            TimestampMicrosecondArray::from(vec![None::<i64>]).with_timezone("UTC"),
                        ),
                        Arc::new(Int64Array::from(vec![42])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();
        let nullable_key_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, true),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let null_key = Batch::table(
            vec![
                RecordBatch::try_new(
                    Arc::clone(&nullable_key_schema),
                    vec![
                        Arc::new(Int64Array::from(vec![None::<i64>])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["paid"])),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        operator
            .process_data("left", null_time, &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", null_key, &context, &mut collector)
            .await
            .unwrap();

        assert!(collector.drain("output").is_empty());
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["null_event_time_rows"], 1);
        assert_eq!(metadata["metrics"]["right"]["null_key_rows"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn watermark_progress_evicts_expired_opposite_rows_and_end_clears_them() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data(
                "left",
                left_batch(vec![0, 100_000_000]),
                &context,
                &mut collector,
            )
            .await
            .unwrap();
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 2);

        let eviction = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Active, Some(150_000_000)),
        );
        operator
            .on_ingress_progress("right", &eviction)
            .await
            .unwrap();
        let metadata = checkpoint_metadata(&mut operator, 2);
        let left_metrics = serde_json::to_string(&metadata["metrics"]["left"]).unwrap();
        assert!(
            left_metrics.contains("\"retained_rows\":1")
                && left_metrics.contains("\"evicted_rows\":1"),
            "{left_metrics}"
        );

        let ended = progress_context(
            &job_context,
            (IngressState::Active, Some(1_000_000)),
            (IngressState::Ended, Some(50_000_000)),
        );
        operator.on_ingress_progress("right", &ended).await.unwrap();
        let metadata = checkpoint_metadata(&mut operator, 3);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }

    #[tokio::test]
    async fn unknown_ingress_and_data_after_end_fail_loudly() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let unknown = operator
            .process_data("middle", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(unknown.to_string().contains("unknown ingress"), "{unknown}");

        operator.on_end(&context, &mut collector).await.unwrap();
        let after_end = operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert!(
            after_end.to_string().contains("data after end-of-input"),
            "{after_end}"
        );

        let progress = IngressProgressSnapshot::new(BTreeMap::from([
            (
                "left".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "right".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
            (
                "middle".into(),
                IngressProgress::new(IngressState::Active, Some(EventTime::from_micros(1))),
            ),
        ]));
        let unknown_progress = operator
            .on_ingress_progress(
                "middle",
                &StreamOperatorContext::for_task(
                    &job_context,
                    "match",
                    None,
                    progress,
                    EdgeBudget::default(),
                    Arc::new(NoopLateMetrics),
                ),
            )
            .await
            .unwrap_err();
        assert!(
            unknown_progress.to_string().contains("unknown ingress"),
            "{unknown_progress}"
        );
    }

    #[tokio::test]
    async fn state_row_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(1, 1_000_000, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["state_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 1);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn match_limit_failure_is_atomic_with_typed_reason() {
        let limited = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 1_000_000, 1).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), limited).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        operator
            .process_data("left", left_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap();
        let failure = operator
            .process_data("right", right_batch(vec![0, 0]), &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinMatchLimitExceeded)
        );

        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["match_limit_failures"], 1);
        assert_eq!(metadata["metrics"]["right"]["retained_rows"], 0);
        assert!(collector.drain("output").is_empty());
    }

    #[tokio::test]
    async fn checkpoint_and_restore_round_trip_preserves_state_and_counters() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        operator
            .process_data("right", right_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        collector.drain("output");
        let snapshot = operator.checkpoint(Epoch::new(7).unwrap()).unwrap();

        let same_epoch = operator.checkpoint(Epoch::new(7).unwrap()).unwrap_err();
        assert!(
            same_epoch.to_string().contains("did not advance"),
            "{same_epoch}"
        );

        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        let round_trip = restored.checkpoint(Epoch::new(8).unwrap()).unwrap();
        let mut expected_metadata = snapshot.inline_metadata.clone();
        expected_metadata.insert("epoch".into(), 8.into());
        assert_eq!(round_trip.inline_metadata, expected_metadata);
        assert_eq!(round_trip.segments, snapshot.segments);

        let mut collector = EdgeCollector::new(restored.output_ports().to_vec());
        restored
            .process_data("right", right_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let outputs = collector.drain("output");
        assert_eq!(outputs.len(), 1, "restored left state must still match");
    }

    #[tokio::test]
    async fn checkpoint_encodes_dirty_upserts_from_carried_records_not_live_state() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0, 1, 2]), &context, &mut collector)
            .await
            .unwrap();
        // The dirty log must encode from the records it carries at admission:
        // the checkpoint path may not scan live state per dirty row, or capture
        // cost grows with the total retained state instead of the dirty set.
        operator.state.left.clear();
        let snapshot = operator.checkpoint(Epoch::INITIAL).unwrap();

        let mut restored =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        restored.restore(&snapshot).unwrap();
        assert_eq!(restored.status().left.retained_rows, 3);
    }

    #[tokio::test]
    async fn checkpoint_shares_carried_segment_allocations_across_epochs() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let first = operator.checkpoint(Epoch::INITIAL).unwrap();
        operator
            .process_data("left", left_batch(vec![1]), &context, &mut collector)
            .await
            .unwrap();
        let second = operator.checkpoint(Epoch::INITIAL.next().unwrap()).unwrap();

        // Capture cost must stay proportional to the dirty set (spec FR47): a
        // segment the operator already encoded is carried into the next
        // snapshot by sharing its allocation, never by copying its bytes.
        let mut shared = 0_usize;
        for (segment_id, carried) in &second.segments {
            if let Some(original) = first.segments.get(segment_id) {
                assert!(
                    Arc::ptr_eq(&original.bytes_arc(), &carried.bytes_arc()),
                    "carried segment {segment_id:?} must share its allocation"
                );
                shared += 1;
            }
        }
        assert_eq!(shared, 1, "epoch 1 delta carries into epoch 2");
        assert!(
            second.segments.contains_key("left-delta-2"),
            "epoch 2 dirty ops encode a fresh segment"
        );
    }

    #[tokio::test]
    async fn restore_rejects_tampered_checkpoints() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());
        operator
            .process_data("left", left_batch(vec![0]), &context, &mut collector)
            .await
            .unwrap();
        let snapshot = operator.checkpoint(Epoch::new(1).unwrap()).unwrap();

        let fresh = |snapshot: &OperatorStateSnapshot| {
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec())
                .unwrap()
                .restore(snapshot)
        };

        let mut bad_magic = snapshot.clone();
        bad_magic
            .segments
            .insert("left-delta-1".into(), StateSegment::new(vec![0_u8; 8]));
        assert!(fresh(&bad_magic).is_err(), "invalid magic must be rejected");

        let mut short_inventory = snapshot.clone();
        short_inventory.segments.remove("left-delta-1");
        assert!(
            fresh(&short_inventory).is_err(),
            "missing segment must be rejected"
        );

        let mut truncated = snapshot.clone();
        let segment = truncated.segments.get_mut("left-delta-1").unwrap();
        let mut bytes = segment.bytes().to_vec();
        bytes.truncate(bytes.len() - 1);
        *segment = StateSegment::new(bytes);
        assert!(
            fresh(&truncated).is_err(),
            "truncated segment must be rejected"
        );

        let mut wrong_layout = snapshot.clone();
        wrong_layout
            .inline_metadata
            .insert("layout_version".into(), 2.into());
        assert!(
            fresh(&wrong_layout).is_err(),
            "layout bump must be rejected"
        );

        let mut wrong_metrics = snapshot.clone();
        wrong_metrics
            .inline_metadata
            .entry("metrics".into())
            .or_default()["left"]["retained_rows"] = 99.into();
        assert!(
            fresh(&wrong_metrics).is_err(),
            "inconsistent retained metrics must be rejected"
        );

        let mut wrong_limits = snapshot.clone();
        wrong_limits
            .inline_metadata
            .entry("spec".into())
            .or_default()["limits"]["max_state_rows_per_side"] = 5.into();
        assert!(
            fresh(&wrong_limits).is_err(),
            "spec change must be rejected"
        );

        let mut bad_metadata = snapshot.clone();
        bad_metadata
            .inline_metadata
            .insert("layout_version".into(), "not-a-number".into());
        assert!(
            fresh(&bad_metadata).is_err(),
            "invalid metadata must be rejected"
        );

        assert!(fresh(&snapshot).is_ok(), "the untampered snapshot restores");
    }

    #[test]
    fn reset_clears_state_for_reuse() {
        let mut operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        let record = left_batch(vec![0]).table_payload().unwrap().batches()[0].slice(0, 1);
        operator.state.left.push(StoredRow {
            encoded_key: Arc::new(encode_join_key_v1(&record, 0, &[0]).unwrap()),
            record,
            event_time: EventTime::from_micros(0),
            row_id: 0,
            charge: 64,
        });
        operator.state.metrics.left.retained_rows = 1;

        operator.reset().unwrap();

        assert!(operator.state.left.is_empty());
        assert_eq!(operator.state.metrics.left.retained_rows, 0);
    }

    #[test]
    fn metadata_exposes_data_only_configuration_and_debug() {
        let operator =
            StreamJoinOperator::new("match", left_schema(), right_schema(), spec()).unwrap();
        assert_eq!(operator.name(), "match");
        assert_eq!(operator.input_ports().len(), 2);
        assert_eq!(operator.output_ports().len(), 1);
        let configuration = operator.configuration();
        assert_eq!(configuration["join_type"], "inner");
        assert_eq!(configuration["left_event_time"], "authorized_at");
        assert!(!configuration.contains_key("callable"));
        let debug = format!("{operator:?}");
        assert!(debug.contains("match"), "{debug}");
    }

    #[test]
    fn spec_validation_rejects_invalid_declarations() {
        let bounds =
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap();
        let limits = JoinStateLimits::new(100, 1_000_000, 1_000).unwrap();

        let empty_keys = StreamJoinSpec::inner(
            Vec::<String>::new(),
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_keys.is_err());

        let unequal = StreamJoinSpec::inner(
            ["a", "b"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(unequal.is_err());

        let duplicate = StreamJoinSpec::inner(
            ["account_id", "account_id"],
            ["account_id", "account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        );
        assert!(duplicate.is_err());

        let empty_event_time = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "",
            "paid_at",
            bounds,
            limits,
        );
        assert!(empty_event_time.is_err());

        let valid = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            bounds,
            limits,
        )
        .unwrap();
        assert!(valid.clone().with_prefixes("same", "same").is_err());
        assert!(valid.clone().with_prefixes("not valid", "right").is_err());

        let prefixed = valid.with_prefixes("authorization", "payment").unwrap();
        assert_eq!(prefixed.left_keys(), ["account_id"]);
        assert_eq!(prefixed.right_keys(), ["account_id"]);
        assert_eq!(prefixed.left_event_time(), "authorized_at");
        assert_eq!(prefixed.right_event_time(), "paid_at");
        assert_eq!(prefixed.left_prefix(), "authorization");
        assert_eq!(prefixed.right_prefix(), "payment");
        assert_eq!(prefixed.join_type(), StreamJoinType::Inner);
        assert_eq!(prefixed.bounds().before(), Duration::from_secs(300));
        assert_eq!(prefixed.bounds().after(), Duration::from_secs(60));
        assert_eq!(prefixed.limits().max_state_rows_per_side(), 100);
        assert_eq!(prefixed.limits().max_state_bytes_per_side(), 1_000_000);
        assert_eq!(prefixed.limits().max_matches_per_input_batch(), 1_000);
        assert!(format!("{prefixed:?}").contains("StreamJoinSpec"));
    }

    #[test]
    fn serde_round_trips_and_rejects_unknown_or_wrong_kind_fields() {
        let source = r#"{
            "join_type": "inner",
            "left_keys": ["account_id"],
            "right_keys": ["account_id"],
            "left_event_time": "authorized_at",
            "right_event_time": "paid_at",
            "bounds": {"before_micros": 300000000, "after_micros": 60000000},
            "limits": {
                "max_state_rows_per_side": 100,
                "max_state_bytes_per_side": 1000000,
                "max_matches_per_input_batch": 1000
            }
        }"#;
        let parsed: StreamJoinSpec = serde_json::from_str(source).unwrap();
        assert_eq!(parsed.left_prefix(), "left");
        assert_eq!(parsed.right_prefix(), "right");
        let encoded = serde_json::to_value(&parsed).unwrap();
        assert_eq!(encoded["bounds"]["before_micros"], 300_000_000);

        let unknown = source.replace(
            "\"join_type\": \"inner\",",
            "\"join_type\": \"inner\", \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown).is_err());

        let outer_join = source.replace("\"inner\"", "\"outer\"");
        assert!(serde_json::from_str::<StreamJoinSpec>(&outer_join).is_err());

        let unknown_bound = source.replace(
            "\"before_micros\": 300000000,",
            "\"before_micros\": 300000000, \"extra\": 1,",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_bound).is_err());

        let unknown_limit = source.replace(
            "\"max_matches_per_input_batch\": 1000",
            "\"max_matches_per_input_batch\": 1000, \"extra\": 1",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&unknown_limit).is_err());

        let zero_limit = source.replace(
            "\"max_state_rows_per_side\": 100",
            "\"max_state_rows_per_side\": 0",
        );
        assert!(serde_json::from_str::<StreamJoinSpec>(&zero_limit).is_err());
    }

    #[tokio::test]
    async fn event_time_columns_accept_every_timestamp_unit_and_reject_others() {
        for (unit, _value) in [
            (TimeUnit::Second, 1_i64),
            (TimeUnit::Millisecond, 1_000),
            (TimeUnit::Microsecond, 1_000_000),
            (TimeUnit::Nanosecond, 1_000_000_000),
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("account_id", DataType::Int64, false),
                Field::new(
                    "authorized_at",
                    DataType::Timestamp(unit, Some("UTC".into())),
                    false,
                ),
                Field::new("amount", DataType::Int64, true),
            ]));
            let operator = StreamJoinOperator::new("match", schema, right_schema(), spec());
            assert!(operator.is_ok(), "unit {unit:?} must be supported");
        }

        let not_a_timestamp = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new("authorized_at", DataType::Int64, false),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", not_a_timestamp, right_schema(), spec());
        assert!(
            rejected.is_err(),
            "non-timestamp event time must be rejected"
        );

        let missing_key = Arc::new(Schema::new(vec![
            Field::new("ledger_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
        ]));
        let rejected = StreamJoinOperator::new("match", missing_key, right_schema(), spec());
        assert!(rejected.is_err(), "missing key column must be rejected");

        let wrong_key_type = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Float64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", wrong_key_type, right_schema(), spec());
        assert!(rejected.is_err(), "unsupported key type must be rejected");

        let zoned = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("America/New_York".into())),
                false,
            ),
            Field::new("amount", DataType::Int64, true),
        ]));
        let rejected = StreamJoinOperator::new("match", zoned, right_schema(), spec());
        assert!(rejected.is_err(), "non-UTC timezone must be rejected");
    }

    #[tokio::test]
    async fn variable_width_payloads_are_charged_and_limited_deterministically() {
        let payload_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "authorized_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("notes", DataType::Utf8, true),
            Field::new("blob", DataType::Binary, true),
            Field::new("tag", DataType::FixedSizeBinary(4), true),
        ]));
        let keys_schema = Arc::new(Schema::new(vec![
            Field::new("account_id", DataType::Int64, false),
            Field::new(
                "paid_at",
                DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())),
                false,
            ),
            Field::new("status", DataType::Utf8, true),
        ]));
        let tiny = StreamJoinSpec::inner(
            ["account_id"],
            ["account_id"],
            "authorized_at",
            "paid_at",
            JoinTimeBounds::new(Duration::from_secs(300), Duration::from_secs(60)).unwrap(),
            JoinStateLimits::new(100, 96, 1_000).unwrap(),
        )
        .unwrap();
        let mut operator =
            StreamJoinOperator::new("match", Arc::clone(&payload_schema), keys_schema, tiny)
                .unwrap();
        let job_context = job();
        let context = StreamOperatorContext::new(&job_context, "match", None);
        let mut collector = EdgeCollector::new(operator.output_ports().to_vec());

        let batch = Batch::table(
            vec![
                RecordBatch::try_new(
                    payload_schema,
                    vec![
                        Arc::new(Int64Array::from(vec![7])),
                        Arc::new(TimestampMicrosecondArray::from(vec![0]).with_timezone("UTC")),
                        Arc::new(StringArray::from(vec!["0123456789"])),
                        Arc::new(BinaryArray::from_opt_vec(vec![Some(&[0_u8; 8][..])])),
                        Arc::new(
                            FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                                vec![Some([1_u8; 4])].into_iter(),
                                4,
                            )
                            .unwrap(),
                        ),
                    ],
                )
                .unwrap(),
            ],
            BatchMetadata::default(),
        )
        .unwrap();

        let failure = operator
            .process_data("left", batch, &context, &mut collector)
            .await
            .unwrap_err();
        assert_eq!(
            reason_of(&failure),
            Some(crate::StreamingFailureReason::JoinStateLimitExceeded)
        );
        let metadata = checkpoint_metadata(&mut operator, 1);
        assert_eq!(metadata["metrics"]["left"]["retained_rows"], 0);
    }
}
