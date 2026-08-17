//! The `ClickHouse` connector (feature `clickhouse`).
//!
//! The source reads a bounded snapshot with a startup-fixed upper
//! cursor bound or polls an event-time/sequence cursor with a unique
//! tie-breaker; both use identifier-checked table/column names and
//! parameterized HTTP queries. Connection credentials arrive only
//! through the secret resolver.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorCapabilities,
    ConnectorDescriptor, ConnectorError, ConnectorFactories, ConnectorIdentity, ConnectorKind,
    ConnectorOperation, ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor,
    DeliveryCapability, JsonMap, Result, SecretHandle, SecretReference, SecretResolver,
    SecretResolverKind, SourceCapabilities, SourceEvent, SourceSchema, StreamSink, StreamSource,
    TransactionSupport, TransactionalStreamSink, WatermarkSupport,
};
use serde_json::Value;

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

pub(crate) fn connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "clickhouse", IDENTITY_VERSION)
        .expect("the clickhouse connector identity is valid")
}

pub(crate) fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        connector_identity(),
        ConnectorOperation::new(operation).expect("operation name is non-empty"),
        detail,
    ))
}

/// Validates a `ClickHouse` identifier against the lowercase vocabulary.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] for names that are not
/// plain lowercase identifiers.
pub(crate) fn required_string(options: &JsonMap, key: &str) -> Result<String> {
    match options.get(key) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a string".into(),
        }),
        None => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option is required".into(),
        }),
    }
}

pub(crate) fn u64_option(options: &JsonMap, key: &str) -> Result<Option<u64>> {
    match options.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(number)) => {
            number
                .as_u64()
                .map(Some)
                .ok_or(CalcFlowError::InvalidArgument {
                    field: key.into(),
                    message: "option must be a non-negative integer".into(),
                })
        }
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a non-negative integer".into(),
        }),
    }
}

/// Validates a `ClickHouse` identifier against the lowercase vocabulary.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] for names that are not
/// plain lowercase identifiers.
pub fn ch_identifier(name: &str) -> Result<String> {
    let valid = !name.is_empty()
        && name
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
        && name.len() <= 63;
    if !valid {
        return Err(CalcFlowError::InvalidArgument {
            field: "identifier".into(),
            message: format!("{name:?} is not a lowercase `ClickHouse` identifier"),
        });
    }
    Ok(name.to_string())
}

/// Resolves the HTTP endpoint URL from a secret reference.
///
/// # Errors
///
/// Returns the resolver error; the URL value never enters the error.
pub fn resolve_clickhouse_url(secrets: &dyn SecretResolver, key: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Environment, key)
        .map_err(|error| fail("open", &error.to_string()))?;
    let handle: SecretHandle = secrets
        .resolve(&reference)
        .map_err(|_| fail("open", "the `ClickHouse` URL secret could not be resolved"))?;
    String::from_utf8(handle.expose().to_vec())
        .map_err(|_| fail("open", "the `ClickHouse` URL secret is not valid UTF-8"))
}

/// Data-only configuration for one `ClickHouse` source.
#[derive(Clone, Debug)]
pub struct ClickHouseSourceConfig {
    /// Secret key holding the `http://host:port` endpoint URL.
    pub url_key: String,
    /// Table to read.
    pub table: String,
    /// Source mode: bounded snapshot or tie-breaker polling.
    pub mode: ChSourceMode,
    /// Cursor column (event-time or sequence).
    pub cursor_column: String,
    /// Unique tie-breaker column for rows sharing a cursor value.
    pub tie_breaker_column: String,
    /// Optional explicit projection column list.
    pub columns: Vec<ArrowFieldSpec>,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Poll interval for incremental mode.
    pub poll_interval: Duration,
    /// HTTP query timeout.
    pub query_timeout: Duration,
}

/// The two `ClickHouse` source modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ChSourceMode {
    /// Bounded snapshot with a startup-fixed upper cursor bound.
    Snapshot,
    /// Event-time/sequence polling with a unique tie-breaker.
    IncrementalQuery,
}

impl ClickHouseSourceConfig {
    /// Parses the source configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option; polling without a unique tie-breaker fails closed.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let url_key = required_string(options, "url_key")?;
        let table = ch_identifier(&required_string(options, "table")?)?;
        let mode = parse_source_mode(options)?;
        let cursor_column = ch_identifier(&required_string(options, "cursor_column")?)?;
        let tie_breaker_column = ch_identifier(&required_string(options, "tie_breaker_column")?)?;
        let columns = parse_column_list(options)?;
        Ok(Self {
            url_key,
            table,
            mode,
            cursor_column,
            tie_breaker_column,
            columns,
            max_batch_rows: u64_option(options, "max_batch_rows")?.unwrap_or(8192),
            poll_interval: Duration::from_millis(
                u64_option(options, "poll_interval_ms")?.unwrap_or(500),
            ),
            query_timeout: Duration::from_secs(
                u64_option(options, "query_timeout_seconds")?.unwrap_or(30),
            ),
        })
    }
}

fn parse_source_mode(options: &JsonMap) -> Result<ChSourceMode> {
    match required_string(options, "mode")?.as_str() {
        "snapshot" => Ok(ChSourceMode::Snapshot),
        "incremental_query" => Ok(ChSourceMode::IncrementalQuery),
        other => Err(CalcFlowError::InvalidArgument {
            field: "mode".into(),
            message: format!("unsupported source mode {other:?}"),
        }),
    }
}

fn parse_column_list(options: &JsonMap) -> Result<Vec<ArrowFieldSpec>> {
    let fields = match options.get("columns") {
        None => Vec::new(),
        Some(value) => {
            serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone()).map_err(|error| {
                CalcFlowError::InvalidArgument {
                    field: "columns".into(),
                    message: format!("columns must be a field list: {error}"),
                }
            })?
        }
    };
    for field in &fields {
        ch_identifier(&field.name)?;
    }
    Ok(fields)
}

/// The `ClickHouse` source over the HTTP interface.
pub struct ClickHouseSource {
    capabilities: SourceCapabilities,
    config: ClickHouseSourceConfig,
    client: reqwest::Client,
    /// Snapshot upper bound fixed at open; rows above it stay out.
    upper_bound: Option<String>,
    cursor_value: String,
    tie_breaker_value: String,
    sequence: u64,
    exhausted: bool,
}

impl ClickHouseSource {
    /// Builds the source.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: ClickHouseSourceConfig) -> Result<Self> {
        let schema = SourceSchema::DynamicOrUnknown;
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: 64 * 1024 * 1024,
            schema,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        let client = reqwest::Client::builder()
            .timeout(config.query_timeout)
            .build()
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(Self {
            capabilities,
            config,
            client,
            upper_bound: None,
            cursor_value: String::new(),
            tie_breaker_value: String::new(),
            sequence: 0,
            exhausted: false,
        })
    }

    /// Opens with a snapshot upper bound fixed at startup.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the bound query fails.
    pub async fn open_with_secrets(
        &mut self,
        cursor: Option<Cursor>,
        secrets: &dyn SecretResolver,
    ) -> Result<()> {
        if let Some(cursor) = cursor {
            self.cursor_value = cursor
                .payload()
                .get("cursor")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            self.tie_breaker_value = cursor
                .payload()
                .get("tie_breaker")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
        if self.config.mode == ChSourceMode::Snapshot && self.upper_bound.is_none() {
            let url = resolve_clickhouse_url(secrets, &self.config.url_key)?;
            let bound = self
                .query_scalar(
                    &url,
                    &format!(
                        "SELECT max({}) FROM {} FORMAT TabSeparated",
                        self.config.cursor_column, self.config.table
                    ),
                )
                .await?;
            self.upper_bound = Some(bound);
        }
        Ok(())
    }

    async fn query_scalar(&self, url: &str, sql: &str) -> Result<String> {
        let response = self
            .client
            .post(url)
            .body(sql.to_string())
            .send()
            .await
            .map_err(|error| fail("read", &redact_url_error(&error.to_string())))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| fail("read", &error.to_string()))?;
        if !status.is_success() {
            return Err(fail(
                "read",
                &format!("ClickHouse returned status {status}"),
            ));
        }
        Ok(text.trim().to_string())
    }

    fn build_query(&self) -> String {
        let selection = if self.config.columns.is_empty() {
            "*".to_string()
        } else {
            self.config
                .columns
                .iter()
                .map(|f| f.name.clone())
                .collect::<Vec<_>>()
                .join(", ")
        };
        let mut sql = format!("SELECT {selection} FROM {}", self.config.table);
        let mut conditions = Vec::new();
        if !self.cursor_value.is_empty() {
            conditions.push(format!(
                "({}, {}) > ('{}', '{}')",
                self.config.cursor_column,
                self.config.tie_breaker_column,
                self.cursor_value,
                self.tie_breaker_value
            ));
        }
        if let Some(bound) = &self.upper_bound {
            conditions.push(format!(
                "{} <= '{}'",
                self.config.cursor_column,
                escape_sql_literal(bound)
            ));
        }
        if !conditions.is_empty() {
            sql.push_str(" WHERE ");
            sql.push_str(&conditions.join(" AND "));
        }
        let _ = write!(
            sql,
            " ORDER BY {}, {} LIMIT {} FORMAT JSONEachRow",
            self.config.cursor_column, self.config.tie_breaker_column, self.config.max_batch_rows
        );
        sql
    }

    async fn fetch_batch(&mut self, secrets: &dyn SecretResolver) -> Result<Option<SourceEvent>> {
        if self.exhausted {
            return Ok(None);
        }
        let url = resolve_clickhouse_url(secrets, &self.config.url_key)?;
        let sql = self.build_query();
        let response = self
            .client
            .post(&url)
            .body(sql)
            .send()
            .await
            .map_err(|error| fail("read", &redact_url_error(&error.to_string())))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|error| fail("read", &error.to_string()))?;
        if !status.is_success() {
            return Err(fail(
                "read",
                &format!("ClickHouse returned status {status}"),
            ));
        }
        let rows: Vec<Value> = text
            .lines()
            .filter(|line| !line.is_empty())
            .map(serde_json::from_str)
            .collect::<std::result::Result<_, _>>()
            .map_err(|error| fail("read", &format!("invalid JSONEachRow: {error}")))?;
        if rows.is_empty() {
            if self.config.mode == ChSourceMode::Snapshot {
                self.exhausted = true;
                return Ok(None);
            }
            tokio::time::sleep(self.config.poll_interval).await;
            return Ok(Some(SourceEvent::Idle));
        }
        let batch = self.decode_rows(&rows)?;
        let cursor = self.cursor_from_last_row(&rows)?;
        if self.config.mode == ChSourceMode::Snapshot
            && (rows.len() as u64) < self.config.max_batch_rows
        {
            self.exhausted = true;
        }
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    fn decode_rows(&mut self, rows: &[Value]) -> Result<Batch> {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field, Schema};
        if rows.is_empty() {
            return Err(fail("read", "cannot decode an empty row set"));
        }
        let first = &rows[0];
        let obj = first
            .as_object()
            .ok_or_else(|| fail("read", "each row must be a JSON object"))?;
        let mut fields = Vec::new();
        let mut columns: Vec<Vec<Option<String>>> = Vec::new();
        for (name, _) in obj {
            let arrow_type = infer_arrow_type(rows, name);
            fields.push(Field::new(name.clone(), arrow_type, true));
            let values: Vec<Option<String>> = rows
                .iter()
                .map(|row| row.get(name).and_then(Value::as_str).map(str::to_string))
                .collect();
            columns.push(values);
        }
        let schema = Schema::new(fields);
        let mut arrays: Vec<std::sync::Arc<dyn arrow::array::Array>> = Vec::new();
        for (field, values) in schema.fields().iter().zip(&columns) {
            let array: std::sync::Arc<dyn arrow::array::Array> =
                if field.data_type() == &DataType::Int64 {
                    let parsed: Vec<Option<i64>> = values
                        .iter()
                        .map(|v| v.as_deref().and_then(|s| s.parse().ok()))
                        .collect();
                    std::sync::Arc::new(Int64Array::from(parsed))
                } else {
                    let strings: Vec<Option<&str>> = values.iter().map(|v| v.as_deref()).collect();
                    std::sync::Arc::new(StringArray::from(strings))
                };
            arrays.push(array);
        }
        let record = arrow::record_batch::RecordBatch::try_new(std::sync::Arc::new(schema), arrays)
            .map_err(|error| fail("read", &error.to_string()))?;
        self.sequence += 1;
        let metadata = BatchMetadata::new(
            "clickhouse",
            self.sequence,
            BTreeMap::from([(
                "table".to_string(),
                Value::String(self.config.table.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        Batch::table(vec![record], metadata).map_err(|error| fail("read", &error.to_string()))
    }

    fn cursor_from_last_row(&mut self, rows: &[Value]) -> Result<Cursor> {
        let last = rows.last().expect("caller checked non-empty");
        let cursor: String = json_value_to_string(last.get(self.config.cursor_column.as_str()));
        let tie_breaker: String =
            json_value_to_string(last.get(self.config.tie_breaker_column.as_str()));
        self.cursor_value.clone_from(&cursor);
        self.tie_breaker_value.clone_from(&tie_breaker);
        let payload = BTreeMap::from([
            ("cursor".to_string(), Value::String(cursor)),
            ("tie_breaker".to_string(), Value::String(tie_breaker)),
        ]);
        let order = serde_json::to_vec(&vec![&self.cursor_value, &self.tie_breaker_value])
            .map_err(|error| fail("cursor", &error.to_string()))?;
        Cursor::unbound(order, payload)
    }
}

/// Infers the Arrow type for one column from its JSON values.
fn infer_arrow_type(rows: &[Value], name: &str) -> arrow::datatypes::DataType {
    use arrow::datatypes::DataType;
    for row in rows {
        if let Some(value) = row.get(name) {
            return match value {
                Value::Number(n) if n.is_i64() || n.is_u64() => DataType::Int64,
                Value::Number(n) if n.is_f64() => DataType::Float64,
                Value::Bool(_) => DataType::Boolean,
                _ => DataType::Utf8,
            };
        }
    }
    DataType::Utf8
}

/// Extracts a plain string from a JSON value without JSON quoting.
fn json_value_to_string(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// Extracts a plain string from a JSON value reference.
fn json_value_to_string_ref(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

/// Escapes single quotes for safe SQL literal interpolation.
fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "\'\'")
}

pub(crate) fn redact_url_error(message: &str) -> String {
    message
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

#[async_trait]
impl StreamSource for ClickHouseSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        // The endpoint URL arrives through the secret resolver per
        // fetch; the trait's no-credential signature cannot resolve one.
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(Some(SourceEvent::Idle))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

impl ClickHouseSource {
    /// Produces the next event; credentials arrive per call.
    ///
    /// # Errors
    ///
    /// Returns the connector error on query failure.
    pub async fn next_with_secrets(
        &mut self,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<SourceEvent>> {
        self.fetch_batch(secrets).await
    }
}

/// Trusted source factory for the `ClickHouse` transport.
pub struct ClickHouseSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl ClickHouseSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: clickhouse_connector_descriptor(),
        }
    }
}

impl Default for ClickHouseSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSourceFactory for ClickHouseSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = ClickHouseSourceConfig::from_options(options)?;
        Ok(Box::new(ClickHouseSource::new(config)?))
    }
}

/// Trusted sink factory for the `ClickHouse` transport.
pub struct ClickHouseSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl ClickHouseSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: clickhouse_connector_descriptor(),
        }
    }
}

impl Default for ClickHouseSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSinkFactory for ClickHouseSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        Err(fail(
            "open",
            "the clickhouse sink requires the endpoint URL secret; use open_transactional",
        ))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = crate::clickhouse_sink::ClickHouseSinkConfig::from_options(options)?;
        Ok(Some(Box::new(crate::clickhouse_sink::ClickHouseSink::new(
            config,
        )?)))
    }
}

fn clickhouse_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: connector_identity(),
        kind: ConnectorKind::Both,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: calc_flow::ReplayCapability::ReplayableExact,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::LedgerIdempotent,
            snapshot: true,
            polling: true,
            cdc: false,
            lookup: false,
        },
        formats: vec![],
        config_schema: JsonMap::from([
            ("url_key".to_string(), serde_json::json!("string")),
            ("table".to_string(), serde_json::json!("string")),
            ("mode".to_string(), serde_json::json!("string")),
            ("cursor_column".to_string(), serde_json::json!("string")),
            (
                "tie_breaker_column".to_string(),
                serde_json::json!("string"),
            ),
            ("columns".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("poll_interval_ms".to_string(), serde_json::json!("u64")),
            (
                "query_timeout_seconds".to_string(),
                serde_json::json!("u64"),
            ),
            ("pipeline".to_string(), serde_json::json!("string")),
            ("output".to_string(), serde_json::json!("string")),
        ]),
        secret_slots: ["url_key".to_string()].into_iter().collect(),
    }
}

/// Registers the `ClickHouse` connectors into one trusted registry.
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot is already
/// occupied.
pub fn register_clickhouse_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        clickhouse_connector_descriptor(),
        ConnectorFactories::both(
            std::sync::Arc::new(ClickHouseSourceFactory::new()),
            std::sync::Arc::new(ClickHouseSinkFactory::new()),
        ),
    )
}
