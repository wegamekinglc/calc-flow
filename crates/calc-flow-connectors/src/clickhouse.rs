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
    DecodeBounds, DeliveryCapability, FormatDecoder, JsonMap, Result, SecretHandle,
    SecretReference, SecretResolver, SecretResolverKind, SourceCapabilities, SourceEvent,
    SourceSchema, StreamSink, StreamSource, TransactionSupport, TransactionalStreamSink,
    WatermarkSupport,
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
pub fn resolve_clickhouse_url(secrets: &dyn SecretResolver, slot: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Registered, slot)
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
    /// Table to read.
    pub table: String,
    /// Source mode: bounded snapshot or tie-breaker polling.
    pub mode: ChSourceMode,
    /// Cursor column (event-time or sequence).
    pub cursor_column: String,
    /// Unique tie-breaker column for rows sharing a cursor value.
    pub tie_breaker_column: String,
    /// Explicit user assertion that the composite cursor is unique.
    pub tie_breaker_unique: bool,
    /// Optional explicit projection column list.
    pub columns: Vec<ArrowFieldSpec>,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Byte bound of one decoded response and decoded batch.
    pub max_batch_bytes: u64,
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
        if options.contains_key("url_key") {
            return Err(CalcFlowError::InvalidArgument {
                field: "options".into(),
                message: "the endpoint URL must use a secret reference".into(),
            });
        }
        let table = ch_identifier(&required_string(options, "table")?)?;
        let mode = parse_source_mode(options)?;
        let cursor_column = ch_identifier(&required_string(options, "cursor_column")?)?;
        let tie_breaker_column = ch_identifier(&required_string(options, "tie_breaker_column")?)?;
        if cursor_column == tie_breaker_column {
            return Err(CalcFlowError::InvalidArgument {
                field: "tie_breaker_column".into(),
                message: "tie-breaker column must differ from cursor column".into(),
            });
        }
        let tie_breaker_unique = match options.get("tie_breaker_unique") {
            Some(Value::Bool(true)) => true,
            Some(Value::Bool(false)) | None => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "tie_breaker_unique".into(),
                    message: "an explicit true uniqueness assertion is required".into(),
                });
            }
            Some(_) => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "tie_breaker_unique".into(),
                    message: "option must be a boolean".into(),
                });
            }
        };
        let columns = parse_column_list(options)?;
        if columns.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "columns".into(),
                message: "ClickHouse sources require a frozen Arrow schema".into(),
            });
        }
        if !columns.iter().any(|field| field.name == cursor_column)
            || !columns.iter().any(|field| field.name == tie_breaker_column)
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "columns".into(),
                message: "frozen schema must include cursor and tie-breaker columns".into(),
            });
        }
        crate::arrow_schema::schema_from_spec(&columns)?;
        let max_batch_rows = positive_option(options, "max_batch_rows", 8192)?;
        let max_batch_bytes = positive_option(options, "max_batch_bytes", 64 * 1024 * 1024)?;
        let query_timeout_seconds = positive_option(options, "query_timeout_seconds", 30)?;
        Ok(Self {
            table,
            mode,
            cursor_column,
            tie_breaker_column,
            tie_breaker_unique,
            columns,
            max_batch_rows,
            max_batch_bytes,
            poll_interval: Duration::from_millis(
                u64_option(options, "poll_interval_ms")?.unwrap_or(500),
            ),
            query_timeout: Duration::from_secs(query_timeout_seconds),
        })
    }
}

fn positive_option(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
    let value = u64_option(options, key)?.unwrap_or(default);
    if value == 0 {
        Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be greater than zero".into(),
        })
    } else {
        Ok(value)
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
    upper_bound: Option<(String, String)>,
    cursor_type: Option<String>,
    tie_breaker_type: Option<String>,
    cursor_value: String,
    tie_breaker_value: String,
    page: u64,
    sequence: u64,
    exhausted: bool,
    endpoint_url: Option<String>,
}

impl ClickHouseSource {
    /// Builds the source.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: ClickHouseSourceConfig) -> Result<Self> {
        let schema = SourceSchema::Exact(crate::arrow_schema::schema_from_spec(&config.columns)?);
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_batch_bytes).unwrap_or(usize::MAX),
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
            cursor_type: None,
            tie_breaker_type: None,
            cursor_value: String::new(),
            tie_breaker_value: String::new(),
            page: 0,
            sequence: 0,
            exhausted: false,
            endpoint_url: None,
        })
    }

    fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.endpoint_url = Some(endpoint_url);
        self
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
        let url = resolve_clickhouse_url(secrets, "url")?;
        self.endpoint_url = Some(url.clone());
        self.initialize(cursor.as_ref(), &url).await
    }

    async fn initialize(&mut self, cursor: Option<&Cursor>, url: &str) -> Result<()> {
        if let Some(cursor) = cursor {
            self.restore_cursor(cursor)?;
        }
        self.load_cursor_types(url).await?;
        if self.config.mode == ChSourceMode::Snapshot
            && self.upper_bound.is_none()
            && !self.exhausted
        {
            self.upper_bound = self.query_upper_bound(url).await?;
            self.exhausted = self.upper_bound.is_none();
        }
        Ok(())
    }

    fn restore_cursor(&mut self, cursor: &Cursor) -> Result<()> {
        let required = |key: &str| {
            cursor
                .payload()
                .get(key)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| fail("cursor", &format!("cursor field {key:?} is missing")))
        };
        self.cursor_value = required("cursor")?;
        self.tie_breaker_value = required("tie_breaker")?;
        self.page = cursor
            .payload()
            .get("page")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("cursor", "cursor page is missing"))?;
        if self.config.mode == ChSourceMode::Snapshot {
            self.upper_bound = Some((required("upper_cursor")?, required("upper_tie_breaker")?));
        }
        if cursor.order() != self.page.to_be_bytes() {
            return Err(fail("cursor", "cursor order does not match its page"));
        }
        Ok(())
    }

    async fn load_cursor_types(&mut self, url: &str) -> Result<()> {
        let sql = "SELECT name, type FROM system.columns \
                   WHERE database = currentDatabase() AND table = {table:String} \
                     AND (name = {cursor_name:String} OR name = {tie_name:String}) \
                   FORMAT JSONEachRow";
        let rows = self
            .query_rows(
                url,
                sql,
                &[
                    ("param_table".into(), self.config.table.clone()),
                    (
                        "param_cursor_name".into(),
                        self.config.cursor_column.clone(),
                    ),
                    (
                        "param_tie_name".into(),
                        self.config.tie_breaker_column.clone(),
                    ),
                ],
            )
            .await?;
        let mut types = BTreeMap::new();
        for row in rows {
            let name = row.get("name").and_then(Value::as_str);
            let data_type = row.get("type").and_then(Value::as_str);
            if let (Some(name), Some(data_type)) = (name, data_type) {
                validate_cursor_type(data_type)?;
                types.insert(name.to_string(), data_type.to_string());
            }
        }
        self.cursor_type = types.get(&self.config.cursor_column).cloned();
        self.tie_breaker_type = types.get(&self.config.tie_breaker_column).cloned();
        if self.cursor_type.is_none() || self.tie_breaker_type.is_none() {
            return Err(fail(
                "open",
                "cursor or tie-breaker column is absent from the ClickHouse table",
            ));
        }
        Ok(())
    }

    async fn query_upper_bound(&self, url: &str) -> Result<Option<(String, String)>> {
        let sql = format!(
            "SELECT {} AS cursor, {} AS tie_breaker FROM {} \
             ORDER BY {} DESC, {} DESC LIMIT 1 FORMAT JSONEachRow",
            self.config.cursor_column,
            self.config.tie_breaker_column,
            self.config.table,
            self.config.cursor_column,
            self.config.tie_breaker_column,
        );
        self.query_rows(url, &sql, &[]).await?.first().map_or_else(
            || Ok(None),
            |row| {
                Ok(Some((
                    required_json_cursor(row.get("cursor"), "upper cursor")?,
                    required_json_cursor(row.get("tie_breaker"), "upper tie-breaker")?,
                )))
            },
        )
    }

    async fn query_rows(
        &self,
        url: &str,
        sql: &str,
        params: &[(String, String)],
    ) -> Result<Vec<Value>> {
        let mut response = self
            .client
            .post(url)
            .query(params)
            .body(sql.to_string())
            .send()
            .await
            .map_err(|error| fail("read", &redact_url_error(&error.to_string())))?;
        let status = response.status();
        if response
            .content_length()
            .is_some_and(|bytes| bytes > self.config.max_batch_bytes)
        {
            return Err(fail("read", "ClickHouse response exceeds max_batch_bytes"));
        }
        if !status.is_success() {
            return Err(fail(
                "read",
                &format!("ClickHouse returned status {status}"),
            ));
        }
        let mut bytes = Vec::new();
        while let Some(chunk) = response
            .chunk()
            .await
            .map_err(|error| fail("read", &redact_url_error(&error.to_string())))?
        {
            let next_len = bytes
                .len()
                .checked_add(chunk.len())
                .ok_or_else(|| fail("read", "ClickHouse response length exhausted usize"))?;
            if u64::try_from(next_len).unwrap_or(u64::MAX) > self.config.max_batch_bytes {
                return Err(fail("read", "ClickHouse response exceeds max_batch_bytes"));
            }
            bytes.extend_from_slice(&chunk);
        }
        std::str::from_utf8(&bytes)
            .map_err(|_| fail("read", "ClickHouse response is not UTF-8"))?
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .map_err(|error| fail("read", &format!("invalid JSONEachRow: {error}")))
            })
            .collect()
    }

    fn build_query(&self) -> Result<(String, Vec<(String, String)>)> {
        let selection = self
            .config
            .columns
            .iter()
            .map(|field| field.name.clone())
            .collect::<Vec<_>>()
            .join(", ");
        let mut sql = format!("SELECT {selection} FROM {}", self.config.table);
        let mut conditions = Vec::new();
        let mut params = Vec::new();
        let cursor_type = self
            .cursor_type
            .as_deref()
            .ok_or_else(|| fail("read", "cursor type was not frozen during open"))?;
        let tie_type = self
            .tie_breaker_type
            .as_deref()
            .ok_or_else(|| fail("read", "tie-breaker type was not frozen during open"))?;
        if !self.cursor_value.is_empty() {
            conditions.push(format!(
                "({}, {}) > ({{cursor:{cursor_type}}}, {{tie_breaker:{tie_type}}})",
                self.config.cursor_column, self.config.tie_breaker_column,
            ));
            params.extend([
                ("param_cursor".into(), self.cursor_value.clone()),
                ("param_tie_breaker".into(), self.tie_breaker_value.clone()),
            ]);
        }
        if let Some((upper_cursor, upper_tie_breaker)) = &self.upper_bound {
            conditions.push(format!(
                "({}, {}) <= ({{upper_cursor:{cursor_type}}}, {{upper_tie_breaker:{tie_type}}})",
                self.config.cursor_column, self.config.tie_breaker_column,
            ));
            params.extend([
                ("param_upper_cursor".into(), upper_cursor.clone()),
                ("param_upper_tie_breaker".into(), upper_tie_breaker.clone()),
            ]);
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
        Ok((sql, params))
    }

    async fn fetch_batch(&mut self, url: &str) -> Result<Option<SourceEvent>> {
        if self.exhausted {
            return Ok(None);
        }
        let (sql, params) = self.build_query()?;
        let rows = self.query_rows(url, &sql, &params).await?;
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
        if rows.is_empty() {
            return Err(fail("read", "cannot decode an empty row set"));
        }
        let json = rows
            .iter()
            .map(serde_json::to_string)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|error| fail("read", &format!("could not encode JSON rows: {error}")))?
            .join("\n");
        let decoded = crate::json_lines::JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?
            .decode(
                json.as_bytes(),
                &DecodeBounds::new(self.config.max_batch_rows, self.config.max_batch_bytes)?,
                &self.config.columns,
            )?;
        let records = decoded.table_payload()?.batches().to_vec();
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("read", "source sequence exhausted"))?;
        let metadata = BatchMetadata::new(
            "clickhouse",
            self.sequence,
            BTreeMap::from([(
                "table".to_string(),
                Value::String(self.config.table.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        Batch::table(records, metadata).map_err(|error| fail("read", &error.to_string()))
    }

    fn cursor_from_last_row(&mut self, rows: &[Value]) -> Result<Cursor> {
        let last = rows.last().expect("caller checked non-empty");
        let cursor = required_json_cursor(
            last.get(self.config.cursor_column.as_str()),
            "cursor column",
        )?;
        let tie_breaker = required_json_cursor(
            last.get(self.config.tie_breaker_column.as_str()),
            "tie-breaker column",
        )?;
        self.cursor_value.clone_from(&cursor);
        self.tie_breaker_value.clone_from(&tie_breaker);
        self.page = self
            .page
            .checked_add(1)
            .ok_or_else(|| fail("cursor", "cursor page exhausted"))?;
        let mut payload = BTreeMap::from([
            ("cursor".to_string(), Value::String(cursor)),
            ("tie_breaker".to_string(), Value::String(tie_breaker)),
            ("page".to_string(), Value::from(self.page)),
        ]);
        if let Some((upper_cursor, upper_tie_breaker)) = &self.upper_bound {
            payload.insert("upper_cursor".into(), Value::String(upper_cursor.clone()));
            payload.insert(
                "upper_tie_breaker".into(),
                Value::String(upper_tie_breaker.clone()),
            );
        }
        Cursor::unbound(self.page.to_be_bytes().to_vec(), payload)
    }
}

/// Extracts a plain string from a JSON value without JSON quoting.
fn required_json_cursor(value: Option<&Value>, field: &str) -> Result<String> {
    match value {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(Value::Number(value)) => Ok(value.to_string()),
        Some(Value::Bool(value)) => Ok(value.to_string()),
        _ => Err(fail("cursor", &format!("{field} is null or non-scalar"))),
    }
}

fn validate_cursor_type(data_type: &str) -> Result<()> {
    let scalar = [
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "UInt128",
        "UInt256",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "Int128",
        "Int256",
        "Date",
        "Date32",
        "DateTime",
        "DateTime64",
        "Decimal",
        "Decimal32",
        "Decimal64",
        "Decimal128",
        "Decimal256",
        "String",
        "FixedString",
        "UUID",
    ]
    .iter()
    .any(|prefix| {
        data_type == *prefix
            || data_type
                .strip_prefix(prefix)
                .is_some_and(|suffix| suffix.starts_with('(') && suffix.ends_with(')'))
    });
    let safe_vocabulary = !data_type.is_empty()
        && data_type.len() <= 128
        && data_type.chars().all(|character| {
            character.is_ascii_alphanumeric()
                || matches!(character, '_' | '(' | ')' | ',' | ' ' | '\'')
        });
    if scalar && safe_vocabulary {
        Ok(())
    } else {
        Err(fail(
            "open",
            &format!("unsupported cursor type {data_type:?}"),
        ))
    }
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

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        let url = self.endpoint_url.clone().ok_or_else(|| {
            fail(
                "open",
                "the clickhouse source was not opened through its trusted factory",
            )
        })?;
        self.initialize(cursor.as_ref(), &url).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let url = self.endpoint_url.clone().ok_or_else(|| {
            fail(
                "read",
                "the clickhouse source was not opened through its trusted factory",
            )
        })?;
        self.fetch_batch(&url).await
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
        let url = resolve_clickhouse_url(secrets, "url")?;
        self.fetch_batch(&url).await
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

    fn validate(&self, options: &JsonMap) -> Result<()> {
        ClickHouseSourceConfig::from_options(options).map(drop)
    }

    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = ClickHouseSourceConfig::from_options(options)?;
        let endpoint_url = resolve_clickhouse_url(secrets, "url")?;
        Ok(Box::new(
            ClickHouseSource::new(config)?.with_endpoint(endpoint_url),
        ))
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

    fn validate(&self, options: &JsonMap) -> Result<()> {
        crate::clickhouse_sink::ClickHouseSinkConfig::from_options(options).map(drop)
    }

    fn capabilities(&self, options: &JsonMap) -> Result<ConnectorCapabilities> {
        let config = crate::clickhouse_sink::ClickHouseSinkConfig::from_options(options)?;
        let mut capabilities = self.descriptor.capabilities;
        if config.retry_deduplicated {
            capabilities.transaction = TransactionSupport::RetryDeduplicated;
        }
        Ok(capabilities)
    }

    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        let config = crate::clickhouse_sink::ClickHouseSinkConfig::from_options(options)?;
        let endpoint_url = resolve_clickhouse_url(secrets, "url")?;
        Ok(Box::new(
            crate::clickhouse_sink::OrdinaryClickHouseSink::new(config, endpoint_url)?,
        ))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = crate::clickhouse_sink::ClickHouseSinkConfig::from_options(options)?;
        let endpoint_url = resolve_clickhouse_url(secrets, "url")?;
        Ok(Some(Box::new(
            crate::clickhouse_sink::ClickHouseSink::new(config)?.with_endpoint(endpoint_url),
        )))
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
            transaction: TransactionSupport::None,
            snapshot: true,
            polling: true,
            cdc: false,
            lookup: false,
        },
        formats: vec![],
        config_schema: JsonMap::from([
            ("table".to_string(), serde_json::json!("string")),
            ("mode".to_string(), serde_json::json!("string")),
            ("cursor_column".to_string(), serde_json::json!("string")),
            (
                "tie_breaker_column".to_string(),
                serde_json::json!("string"),
            ),
            ("tie_breaker_unique".to_string(), serde_json::json!("bool")),
            ("columns".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("poll_interval_ms".to_string(), serde_json::json!("u64")),
            (
                "query_timeout_seconds".to_string(),
                serde_json::json!("u64"),
            ),
            ("pipeline".to_string(), serde_json::json!("string")),
            ("output".to_string(), serde_json::json!("string")),
            ("retry_deduplicated".to_string(), serde_json::json!("bool")),
            ("max_block_rows".to_string(), serde_json::json!("u64")),
            ("max_block_bytes".to_string(), serde_json::json!("u64")),
        ]),
        secret_slots: ["url".to_string()].into_iter().collect(),
        required_secret_slots: ["url".to_string()].into_iter().collect(),
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

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ClickHouseSourceConfig {
        ClickHouseSourceConfig::from_options(&BTreeMap::from([
            ("table".into(), serde_json::json!("events")),
            ("mode".into(), serde_json::json!("snapshot")),
            ("cursor_column".into(), serde_json::json!("updated_at")),
            ("tie_breaker_column".into(), serde_json::json!("id")),
            ("tie_breaker_unique".into(), serde_json::json!(true)),
            (
                "columns".into(),
                serde_json::json!([
                    {"name": "updated_at", "data_type": "string", "nullable": false},
                    {"name": "id", "data_type": "int64", "nullable": false},
                    {"name": "label", "data_type": "string", "nullable": false}
                ]),
            ),
        ]))
        .unwrap()
    }

    #[test]
    fn cursor_values_are_query_parameters_and_snapshot_bound_is_composite() {
        let mut source = ClickHouseSource::new(config()).unwrap();
        source.cursor_type = Some("DateTime64(3, 'UTC')".into());
        source.tie_breaker_type = Some("Int64".into());
        source.cursor_value = "2026-01-01 00:00:00.000' OR 1".into();
        source.tie_breaker_value = "7".into();
        source.upper_bound = Some(("2026-01-02 00:00:00.000".into(), "99".into()));

        let (sql, params) = source.build_query().unwrap();
        assert!(!sql.contains("OR 1"));
        assert!(sql.contains("{cursor:DateTime64(3, 'UTC')}"));
        assert!(sql.contains("{upper_tie_breaker:Int64}"));
        assert_eq!(params[0].1, "2026-01-01 00:00:00.000' OR 1");
        assert_eq!(params.len(), 4);
    }

    #[test]
    fn snapshot_cursor_persists_bound_and_uses_monotonic_page_order() {
        let mut source = ClickHouseSource::new(config()).unwrap();
        source.upper_bound = Some(("2026-01-02".into(), "9".into()));
        let cursor = source
            .cursor_from_last_row(&[serde_json::json!({
                "updated_at": "2026-01-01",
                "id": 7,
                "label": "event"
            })])
            .unwrap();
        assert_eq!(cursor.order(), 1_u64.to_be_bytes());
        assert_eq!(
            cursor.payload()["upper_tie_breaker"],
            serde_json::json!("9")
        );

        let mut restored = ClickHouseSource::new(config()).unwrap();
        restored.restore_cursor(&cursor).unwrap();
        assert_eq!(
            restored.upper_bound,
            Some(("2026-01-02".into(), "9".into()))
        );
        assert_eq!(restored.page, 1);
    }

    #[test]
    fn cursor_type_vocabulary_fails_closed() {
        for supported in ["UInt64", "DateTime64(3, 'UTC')", "Decimal(18, 4)", "UUID"] {
            validate_cursor_type(supported).unwrap();
        }
        for rejected in ["Nullable(UInt64)", "Float64", "UInt64); DROP TABLE events"] {
            assert!(validate_cursor_type(rejected).is_err(), "{rejected}");
        }
    }

    #[test]
    fn explicit_schema_decodes_numeric_json_without_nulling_values() {
        let mut source = ClickHouseSource::new(config()).unwrap();
        let batch = source
            .decode_rows(&[serde_json::json!({
                "updated_at": "2026-01-01",
                "id": 7,
                "label": "event"
            })])
            .unwrap();
        let record = &batch.table_payload().unwrap().batches()[0];
        let ids = record
            .column(1)
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(ids.value(0), 7);
    }
}
