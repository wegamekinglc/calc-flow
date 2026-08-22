#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Duration};

    use datafusion::arrow::{
        array::{Int64Array, StringArray, TimestampMicrosecondArray},
        datatypes::{DataType, Field, Schema, TimeUnit},
        record_batch::RecordBatch,
    };

    use super::*;
    use crate::{
        BatchMetadata, CancellationToken, EdgeCollector, IngressProgress, IngressProgressSnapshot,
        IngressState, JsonMap, OperatorMetadata, StreamJobContext, StreamMessageKind,
        StreamOperator,
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
}
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
        Array, BinaryArray, BinaryViewArray, LargeBinaryArray, LargeStringArray, StringArray,
        StringViewArray, TimestampMicrosecondArray, TimestampMillisecondArray,
        TimestampNanosecondArray, TimestampSecondArray,
    },
    datatypes::{DataType, Field, Schema, SchemaRef, TimeUnit},
    ipc::{reader::StreamReader, writer::StreamWriter},
    record_batch::RecordBatch,
};
use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;

use crate::{
    Batch, BatchKind, BatchMetadata, CalcFlowError, DataFusionConfig, Epoch, EventTime,
    IngressProgress, JsonMap, OperatorStateSnapshot, Port, Result, StreamCollector, StreamOperator,
    StreamOperatorContext, UdfRegistrySnapshot,
};

use super::{OperatorMetadata, StreamRuntimeState, is_identifier, validate_operator_name};

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

#[derive(Clone)]
struct StoredRow {
    record: RecordBatch,
    event_time: EventTime,
    row_id: u64,
    charge: u64,
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

    #[allow(
        clippy::too_many_lines,
        reason = "transactional Join admission keeps row IDs, matches, and limits in one scratch-state boundary"
    )]
    async fn prepare_batch(
        &mut self,
        ingress: &str,
        batch: &Batch,
        context: &StreamOperatorContext<'_>,
    ) -> Result<PreparedJoinBatch> {
        if self.state.ended {
            return Err(operator_error(
                &self.name,
                "received data after end-of-input",
            ));
        }
        let (input_port_index, event_index, key_indices, opposite, incoming_is_left) = match ingress
        {
            "left" => (
                0,
                self.compiled.left_event_time_index,
                self.compiled.left_key_indices.clone(),
                self.state.right.clone(),
                true,
            ),
            "right" => (
                1,
                self.compiled.right_event_time_index,
                self.compiled.right_key_indices.clone(),
                self.state.left.clone(),
                false,
            ),
            _ => {
                return Err(operator_error(
                    &self.name,
                    &format!("unknown ingress {ingress:?}; expected left or right"),
                ));
            }
        };
        self.input_ports[input_port_index]
            .validate(batch, &format!("{}.{}", self.name, ingress))?;
        let side_progress = context.ingress_progress().get(ingress);
        let opposite_progress =
            context
                .ingress_progress()
                .get(if incoming_is_left { "right" } else { "left" });
        let mut next_row_id = if incoming_is_left {
            self.state.next_left_row_id
        } else {
            self.state.next_right_row_id
        };
        let mut metrics = if incoming_is_left {
            self.state.metrics.left.clone()
        } else {
            self.state.metrics.right.clone()
        };
        let mut outputs = Vec::new();
        let mut retained = Vec::new();
        let mut batch_had_late = false;
        for record in batch.table_payload()?.batches() {
            for row_index in 0..record.num_rows() {
                let row_id = next_row_id;
                next_row_id = next_row_id
                    .checked_add(1)
                    .ok_or_else(|| counter_overflow(&self.name, "row_id"))?;
                let Some(event_time) =
                    event_time_at(record, event_index, row_index, &self.name, ingress)?
                else {
                    metrics.null_event_time_rows = checked_metric(
                        metrics.null_event_time_rows,
                        1,
                        &self.name,
                        "null_event_time_rows",
                    )?;
                    continue;
                };
                if key_indices
                    .iter()
                    .any(|&index| record.column(index).is_null(row_index))
                {
                    metrics.null_key_rows =
                        checked_metric(metrics.null_key_rows, 1, &self.name, "null_key_rows")?;
                    continue;
                }
                if let Some(watermark) = side_progress.and_then(IngressProgress::watermark)
                    && event_time < watermark
                {
                    let lateness = u64::try_from(
                        i128::from(watermark.as_micros()) - i128::from(event_time.as_micros()),
                    )
                    .map_err(|_| counter_overflow(&self.name, "lateness"))?;
                    metrics.late_rows =
                        checked_metric(metrics.late_rows, 1, &self.name, "late_rows")?;
                    metrics.max_lateness_micros = Some(
                        metrics
                            .max_lateness_micros
                            .map_or(lateness, |current| current.max(lateness)),
                    );
                    batch_had_late = true;
                    continue;
                }
                let incoming = record.slice(row_index, 1);
                let mut candidates = opposite
                    .iter()
                    .filter(|candidate| {
                        if incoming_is_left {
                            self.spec.bounds.contains_pair(
                                event_time.as_micros(),
                                candidate.event_time.as_micros(),
                            )
                        } else {
                            self.spec.bounds.contains_pair(
                                candidate.event_time.as_micros(),
                                event_time.as_micros(),
                            )
                        }
                    })
                    .collect::<Vec<_>>();
                candidates.sort_by_key(|candidate| (candidate.event_time, candidate.row_id));
                for candidate in candidates {
                    let (left, right) = if incoming_is_left {
                        (&incoming, &candidate.record)
                    } else {
                        (&candidate.record, &incoming)
                    };
                    if self.pair_matches(left, right).await? {
                        outputs.push(self.output_record(left, right)?);
                        let match_count = u64::try_from(outputs.len())
                            .map_err(|_| counter_overflow(&self.name, "match_count"))?;
                        if match_count > self.spec.limits.max_matches_per_input_batch {
                            self.state.metrics.match_limit_failures = checked_metric(
                                self.state.metrics.match_limit_failures,
                                1,
                                &self.name,
                                "match_limit_failures",
                            )?;
                            return Err(operator_reason(
                                &self.name,
                                crate::StreamingFailureReason::JoinMatchLimitExceeded,
                                "input batch match limit exceeded",
                            ));
                        }
                    }
                }
                if should_retain(
                    incoming_is_left,
                    event_time,
                    opposite_progress,
                    self.spec.bounds,
                ) {
                    let charge = state_row_charge(record, row_index, &key_indices, &self.name)?;
                    retained.push(StoredRow {
                        record: incoming,
                        event_time,
                        row_id,
                        charge,
                    });
                }
            }
        }
        if batch_had_late {
            metrics.late_affected_batches = checked_metric(
                metrics.late_affected_batches,
                1,
                &self.name,
                "late_affected_batches",
            )?;
        }
        self.validate_state_admission(incoming_is_left, &retained)?;
        Ok(PreparedJoinBatch {
            outputs,
            retained,
            next_row_id,
            metrics,
        })
    }

    async fn pair_matches(&mut self, left: &RecordBatch, right: &RecordBatch) -> Result<bool> {
        let tables = BTreeMap::from([
            (
                "left_input".into(),
                Batch::table(vec![left.clone()], BatchMetadata::default())?,
            ),
            (
                "right_input".into(),
                Batch::table(vec![right.clone()], BatchMetadata::default())?,
            ),
        ]);
        let result = self
            .runtime
            .runtime()?
            .sql(&self.compiled.equality_query, &tables, Some(&self.name))
            .await?;
        Ok(result.num_rows() == 1)
    }

    fn output_record(&self, left: &RecordBatch, right: &RecordBatch) -> Result<RecordBatch> {
        let columns = left
            .columns()
            .iter()
            .chain(right.columns())
            .cloned()
            .collect::<Vec<_>>();
        RecordBatch::try_new(
            Arc::clone(
                self.output_ports[0]
                    .schema()
                    .expect("stream Join output always has an exact schema"),
            ),
            columns,
        )
        .map_err(|error| operator_error(&self.name, &format!("output projection failed: {error}")))
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
        let rows = u64::try_from(current.len())
            .ok()
            .and_then(|count| count.checked_add(u64::try_from(retained.len()).ok()?))
            .ok_or_else(|| counter_overflow(&self.name, "state rows"))?;
        let bytes = current
            .iter()
            .chain(retained)
            .try_fold(0_u64, |total, row| total.checked_add(row.charge))
            .ok_or_else(|| counter_overflow(&self.name, "state bytes"))?;
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
            let metadata = BatchMetadata::new(
                context.operator_id(),
                self.state.next_output_sequence,
                BTreeMap::new(),
            )?;
            let batch = Batch::table(vec![record.clone()], metadata)?;
            if batch.estimated_bytes()? > context.output_budget().max_bytes {
                return Err(CalcFlowError::InvalidArgument {
                    field: "message.bytes".into(),
                    message: "one stream Join output row exceeds the effective edge byte budget"
                        .into(),
                });
            }
            output.emit("output", batch).await?;
            self.state.next_output_sequence += 1;
        }
        Ok(())
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
        if ingress == "left" {
            self.state.next_left_row_id = prepared.next_row_id;
            self.state.left.extend(prepared.retained);
            self.state.metrics.left = prepared.metrics;
            refresh_retained_metrics(&mut self.state.metrics.left, &self.state.left, &self.name)?;
        } else {
            self.state.next_right_row_id = prepared.next_row_id;
            self.state.right.extend(prepared.retained);
            self.state.metrics.right = prepared.metrics;
            refresh_retained_metrics(&mut self.state.metrics.right, &self.state.right, &self.name)?;
        }
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
        match ingress {
            "left" => {
                evict_opposite(
                    &mut self.state.right,
                    progress,
                    self.spec.bounds.before_micros,
                    &mut self.state.metrics.right,
                    &self.name,
                )?;
            }
            "right" => {
                evict_opposite(
                    &mut self.state.left,
                    progress,
                    self.spec.bounds.after_micros,
                    &mut self.state.metrics.left,
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
        let segments = BTreeMap::from([
            (
                "left-v1".into(),
                encode_side(&self.state.left, &self.name, "left")?,
            ),
            (
                "right-v1".into(),
                encode_side(&self.state.right, &self.name, "right")?,
            ),
        ]);
        self.state.last_checkpoint_epoch = Some(epoch);
        Ok(OperatorStateSnapshot {
            inline_metadata: inline_metadata.into_iter().collect(),
            segments,
        })
    }

    fn restore(&mut self, snapshot: &OperatorStateSnapshot) -> Result<()> {
        let metadata = serde_json::from_value::<JoinCheckpointMetadata>(Value::Object(
            snapshot.inline_metadata.clone().into_iter().collect(),
        ))
        .map_err(|error| CalcFlowError::CheckpointMismatch {
            message: format!("stream Join {:?} metadata is invalid: {error}", self.name),
        })?;
        if metadata.layout_version != 1 || metadata.spec != self.spec {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} checkpoint layout or specification is incompatible",
                    self.name
                ),
            });
        }
        if snapshot
            .segments
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != ["left-v1", "right-v1"]
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!("stream Join {:?} segment inventory is invalid", self.name),
            });
        }
        let left_schema = self.input_ports[0]
            .schema()
            .expect("stream Join left input always has an exact schema");
        let right_schema = self.input_ports[1]
            .schema()
            .expect("stream Join right input always has an exact schema");
        let left = decode_side(
            &snapshot.segments["left-v1"],
            left_schema,
            &self.name,
            "left",
        )?;
        let right = decode_side(
            &snapshot.segments["right-v1"],
            right_schema,
            &self.name,
            "right",
        )?;
        validate_restored_rows(
            &left,
            metadata.next_left_row_id,
            self.compiled.left_event_time_index,
            &self.compiled.left_key_indices,
            &self.name,
            "left",
        )?;
        validate_restored_rows(
            &right,
            metadata.next_right_row_id,
            self.compiled.right_event_time_index,
            &self.compiled.right_key_indices,
            &self.name,
            "right",
        )?;
        let mut left_metrics = metadata.metrics.left.clone();
        let mut right_metrics = metadata.metrics.right.clone();
        refresh_retained_metrics(&mut left_metrics, &left, &self.name)?;
        refresh_retained_metrics(&mut right_metrics, &right, &self.name)?;
        if left_metrics.retained_rows != metadata.metrics.left.retained_rows
            || left_metrics.retained_bytes != metadata.metrics.left.retained_bytes
            || right_metrics.retained_rows != metadata.metrics.right.retained_rows
            || right_metrics.retained_bytes != metadata.metrics.right.retained_bytes
        {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "stream Join {:?} restored state charge is inconsistent",
                    self.name
                ),
            });
        }
        self.validate_restored_limits(&left, &right)?;
        self.state = StreamJoinState {
            left,
            right,
            next_left_row_id: metadata.next_left_row_id,
            next_right_row_id: metadata.next_right_row_id,
            next_output_sequence: metadata.next_output_sequence,
            metrics: metadata.metrics,
            ended: metadata.ended,
            last_checkpoint_epoch: Epoch::new(metadata.epoch),
        };
        Ok(())
    }
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
    if !is_identifier(left) || !is_identifier(right) || left == right {
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
    let mut left_key_indices = Vec::with_capacity(spec.left_keys.len());
    let mut right_key_indices = Vec::with_capacity(spec.right_keys.len());
    for (index, (left_key, right_key)) in spec.left_keys.iter().zip(&spec.right_keys).enumerate() {
        let left_field = field_by_name(left, left_key, "left_keys", index)?;
        let right_field = field_by_name(right, right_key, "right_keys", index)?;
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
    validate_event_time(left, &spec.left_event_time, "left_event_time")?;
    validate_event_time(right, &spec.right_event_time, "right_event_time")?;
    let left_event_time_index = left
        .index_of(&spec.left_event_time)
        .expect("event-time lookup succeeded above");
    let right_event_time_index = right
        .index_of(&spec.right_event_time)
        .expect("event-time lookup succeeded above");
    let fields = left
        .fields()
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
        .collect::<Vec<_>>();
    let equality = spec
        .left_keys
        .iter()
        .zip(&spec.right_keys)
        .map(|(left, right)| {
            format!(
                "left_input.{} = right_input.{}",
                quote_identifier(left),
                quote_identifier(right)
            )
        })
        .collect::<Vec<_>>()
        .join(" AND ");
    Ok((
        Arc::new(Schema::new(fields)),
        CompiledJoin {
            left_key_indices,
            right_key_indices,
            left_event_time_index,
            right_event_time_index,
            equality_query: format!(
                "SELECT 1 AS matched FROM left_input INNER JOIN right_input ON {equality}"
            ),
        },
    ))
}

fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
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

fn supported_key_type(data_type: &DataType) -> bool {
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

fn encode_side(rows: &[StoredRow], operator_id: &str, side: &str) -> Result<Vec<u8>> {
    let mut ordered = rows.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|row| (row.event_time, row.row_id));
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
    operator_id: &str,
    side: &str,
) -> Result<Vec<StoredRow>> {
    let mut offset = 0_usize;
    if take_segment_bytes(bytes, &mut offset, JOIN_STATE_MAGIC.len())? != JOIN_STATE_MAGIC {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state magic is invalid",
        ));
    }
    let row_count = read_segment_u64(bytes, &mut offset)?;
    let row_capacity = usize::try_from(row_count)
        .ok()
        .filter(|count| *count <= bytes.len())
        .ok_or_else(|| checkpoint_error(operator_id, side, "row count is invalid"))?;
    let mut rows = Vec::with_capacity(row_capacity);
    for _ in 0..row_count {
        let row_id = read_segment_u64(bytes, &mut offset)?;
        let event_time = EventTime::from_micros(read_segment_i64(bytes, &mut offset)?);
        let charge = read_segment_u64(bytes, &mut offset)?;
        let ipc_length = usize::try_from(read_segment_u64(bytes, &mut offset)?)
            .map_err(|_| checkpoint_error(operator_id, side, "IPC length is invalid"))?;
        let ipc = take_segment_bytes(bytes, &mut offset, ipc_length)?;
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
        rows.push(StoredRow {
            record,
            event_time,
            row_id,
            charge,
        });
    }
    if offset != bytes.len() {
        return Err(checkpoint_error(
            operator_id,
            side,
            "state segment has trailing bytes",
        ));
    }
    Ok(rows)
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
        if row.row_id >= next_row_id || !identities.insert((row.event_time, row.row_id)) {
            return Err(checkpoint_error(
                operator_id,
                side,
                "row identity is invalid",
            ));
        }
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
    operator_id: &str,
) -> Result<()> {
    let before = rows.len();
    if progress.state() == crate::IngressState::Ended {
        rows.clear();
    } else if let Some(watermark) = progress.watermark() {
        rows.retain(|row| {
            i128::from(row.event_time.as_micros()) + i128::from(extension_micros)
                >= i128::from(watermark.as_micros())
        });
    }
    let evicted = u64::try_from(before - rows.len())
        .map_err(|_| counter_overflow(operator_id, "evicted rows"))?;
    metrics.evicted_rows =
        checked_metric(metrics.evicted_rows, evicted, operator_id, "evicted_rows")?;
    refresh_retained_metrics(metrics, rows, operator_id)
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
    let key_bytes = key_indices.iter().try_fold(0_u64, |total, &index| {
        let charge = logical_cell_charge(record.column(index).as_ref(), row_index)?;
        total
            .checked_add(charge)
            .ok_or_else(|| counter_overflow(operator_id, "encoded key length"))
    })?;
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

#[allow(
    clippy::too_many_lines,
    reason = "the exhaustive Arrow charging table mirrors the frozen V1 accounting contract"
)]
fn logical_cell_charge(array: &dyn Array, row_index: usize) -> Result<u64> {
    if array.is_null(row_index) {
        return Ok(1);
    }
    let value_bytes = match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean | DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(datafusion::arrow::datatypes::IntervalUnit::YearMonth)
        | DataType::Decimal32(_, _) => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_)
        | DataType::Interval(datafusion::arrow::datatypes::IntervalUnit::DayTime)
        | DataType::Decimal64(_, _) => 8,
        DataType::Interval(datafusion::arrow::datatypes::IntervalUnit::MonthDayNano)
        | DataType::Decimal128(_, _) => 16,
        DataType::Decimal256(_, _) => 32,
        DataType::FixedSizeBinary(size) => {
            u64::try_from(*size).map_err(|_| CalcFlowError::Internal {
                message: "negative FixedSizeBinary width".into(),
            })?
        }
        DataType::Utf8 => {
            let array = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Utf8 array type mismatch".into(),
                })?;
            4_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Utf8 cell charge overflow".into(),
                })?
        }
        DataType::LargeUtf8 => {
            let array = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "LargeUtf8 array type mismatch".into(),
                })?;
            8_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "LargeUtf8 cell charge overflow".into(),
                })?
        }
        DataType::Binary => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Binary array type mismatch".into(),
                })?;
            4_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Binary cell charge overflow".into(),
                })?
        }
        DataType::LargeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<LargeBinaryArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "LargeBinary array type mismatch".into(),
                })?;
            8_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "LargeBinary cell charge overflow".into(),
                })?
        }
        DataType::Utf8View => {
            let array = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Utf8View array type mismatch".into(),
                })?;
            16_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "Utf8View cell charge overflow".into(),
                })?
        }
        DataType::BinaryView => {
            let array = array
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "BinaryView array type mismatch".into(),
                })?;
            16_u64
                .checked_add(u64::try_from(array.value(row_index).len()).unwrap_or(u64::MAX))
                .ok_or_else(|| CalcFlowError::Internal {
                    message: "BinaryView cell charge overflow".into(),
                })?
        }
        _ => u64::try_from(
            array
                .slice(row_index, 1)
                .to_data()
                .get_slice_memory_size()
                .map_err(|error| CalcFlowError::Internal {
                    message: format!("logical payload charge failed: {error}"),
                })?,
        )
        .map_err(|_| CalcFlowError::Internal {
            message: "logical payload charge exceeds UInt64".into(),
        })?,
    };
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
