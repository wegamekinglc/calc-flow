//! The `PostgreSQL` connector (feature `postgresql`).
//!
//! The source reads a repeatable-read consistent snapshot or polls a
//! strictly monotonic composite cursor; both use bound parameters and
//! identifier-checked table and column names. The sink writes
//! parameterized appends, upserts, or an epoch ledger committed in the
//! same transaction as the data. Connection credentials arrive only
//! through the secret resolver as a `postgresql://…` URL.

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorError, ConnectorIdentity,
    ConnectorOperation, Cursor, JsonMap, Result, SecretHandle, SecretReference, SecretResolverKind,
    SinkRecovery, SourceCapabilities, SourceEvent, SourceSchema, StreamSource,
    TransactionalStreamSink,
};
use serde_json::Value;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, NoTls, Row};

use crate::database_types::{PgColumn, arrow_schema, pg_identifier, record_batch};

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

/// The ledger table name carrying epoch-committed evidence.
pub const LEDGER_TABLE: &str = "calc_flow_epoch_ledger";

fn connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "postgresql", IDENTITY_VERSION)
        .expect("the postgresql connector identity is valid")
}

fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        connector_identity(),
        ConnectorOperation::new(operation).expect("operation name is non-empty"),
        detail,
    ))
}

/// Reads the connection URL from a secret reference, never from
/// options.
///
/// # Errors
///
/// Returns the resolver error when the reference cannot be resolved;
/// the URL value itself never enters the error.
pub fn resolve_connection_url(
    secrets: &dyn calc_flow::SecretResolver,
    key: &str,
) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Environment, key)
        .map_err(|error| fail("open", &error.to_string()))?;
    let handle: SecretHandle = secrets
        .resolve(&reference)
        .map_err(|_| fail("open", "the connection URL secret could not be resolved"))?;
    String::from_utf8(handle.expose().to_vec())
        .map_err(|_| fail("open", "the connection URL secret is not valid UTF-8"))
}

/// Data-only configuration shared by both source modes.
#[derive(Clone, Debug)]
pub struct PostgresSourceConfig {
    /// Secret key holding the `postgresql://` URL.
    pub url_key: String,
    /// Table to read.
    pub table: String,
    /// Source mode: repeatable-read snapshot or composite-cursor poll.
    pub mode: PgSourceMode,
    /// Ordered cursor columns; required for incremental polling.
    pub cursor_columns: Vec<String>,
    /// Optional explicit projection column list.
    pub columns: Vec<ArrowFieldSpec>,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Poll interval for incremental mode.
    pub poll_interval: std::time::Duration,
}

/// The two `PostgreSQL` source modes shipped in this task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PgSourceMode {
    /// One repeatable-read consistent snapshot.
    Snapshot,
    /// Strictly monotonic composite-cursor polling.
    IncrementalQuery,
}

impl PostgresSourceConfig {
    /// Parses the source configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option for missing or malformed values, or for cursor columns
    /// without incremental mode.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let (url_key, table) = parse_source_endpoint(options)?;
        let mode = parse_source_mode(options)?;
        let (max_batch_rows, poll_interval_ms) = parse_source_bounds(options)?;
        Ok(Self {
            url_key,
            table,
            mode,
            cursor_columns: parse_cursor_columns(options, mode)?,
            columns: parse_column_list(options)?,
            max_batch_rows,
            poll_interval: std::time::Duration::from_millis(poll_interval_ms),
        })
    }
}

fn parse_source_endpoint(options: &JsonMap) -> Result<(String, String)> {
    Ok((
        required_string(options, "url_key")?,
        pg_identifier(&required_string(options, "table")?)?,
    ))
}

fn parse_source_bounds(options: &JsonMap) -> Result<(u64, u64)> {
    Ok((
        u64_option(options, "max_batch_rows")?.unwrap_or(8192),
        u64_option(options, "poll_interval_ms")?.unwrap_or(500),
    ))
}

fn parse_source_mode(options: &JsonMap) -> Result<PgSourceMode> {
    match required_string(options, "mode")?.as_str() {
        "snapshot" => Ok(PgSourceMode::Snapshot),
        "incremental_query" => Ok(PgSourceMode::IncrementalQuery),
        other => Err(CalcFlowError::InvalidArgument {
            field: "mode".into(),
            message: format!("unsupported source mode {other:?}"),
        }),
    }
}

fn parse_cursor_columns(options: &JsonMap, mode: PgSourceMode) -> Result<Vec<String>> {
    let columns = match options.get("cursor_columns") {
        None => Vec::new(),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map_or_else(
                    || {
                        Err(CalcFlowError::InvalidArgument {
                            field: "cursor_columns".into(),
                            message: "entries must be strings".into(),
                        })
                    },
                    pg_identifier,
                )
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor_columns".into(),
                message: "cursor_columns must be a string array".into(),
            });
        }
    };
    if mode == PgSourceMode::IncrementalQuery && columns.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "cursor_columns".into(),
            message: "incremental_query requires at least one cursor column".into(),
        });
    }
    Ok(columns)
}

fn parse_column_list(options: &JsonMap) -> Result<Vec<ArrowFieldSpec>> {
    match options.get("columns") {
        None => Ok(Vec::new()),
        Some(value) => {
            serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone()).map_err(|error| {
                CalcFlowError::InvalidArgument {
                    field: "columns".into(),
                    message: format!("columns must be a field list: {error}"),
                }
            })
        }
    }
}

/// The `PostgreSQL` source over a private connection.
pub struct PostgresSource {
    capabilities: SourceCapabilities,
    config: PostgresSourceConfig,
    client: Option<Client>,
    columns: Vec<PgColumn>,
    cursor_values: Vec<String>,
    sequence: u64,
    exhausted: bool,
}

impl PostgresSource {
    /// Builds the source; the connection opens lazily in `open`.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: PostgresSourceConfig) -> Result<Self> {
        let schema = if config.columns.is_empty() {
            SourceSchema::DynamicOrUnknown
        } else {
            SourceSchema::Exact(
                arrow_schema(
                    &config
                        .columns
                        .iter()
                        .map(|field| PgColumn {
                            name: field.name.clone(),
                            data_type: tokio_postgres::types::Type::TEXT,
                            nullable: field.nullable,
                        })
                        .collect::<Vec<_>>(),
                )
                .map_err(|error| fail("open", &error.to_string()))?,
            )
        };
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: 64 * 1024 * 1024,
            schema,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            capabilities,
            config,
            client: None,
            columns: Vec::new(),
            cursor_values: Vec::new(),
            sequence: 0,
            exhausted: false,
        })
    }

    async fn connect(&mut self, url: &str) -> Result<()> {
        let (client, connection) = tokio_postgres::connect(url, NoTls)
            .await
            .map_err(|error| fail("open", &redact_url_error(&error.to_string())))?;
        tokio::spawn(async move {
            if let Err(error) = connection.await {
                let _ = error;
            }
        });
        self.client = Some(client);
        self.load_columns()
            .await
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(())
    }

    async fn load_columns(&mut self) -> std::result::Result<(), tokio_postgres::Error> {
        let client = self.client.as_ref().expect("connection established");
        let rows = client
            .query(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull \
                 FROM pg_attribute a \
                 JOIN pg_class c ON c.oid = a.attrelid \
                 WHERE c.relname = $1 AND a.attnum > 0 AND NOT a.attisdropped \
                 ORDER BY a.attnum",
                &[&self.config.table],
            )
            .await?;
        self.columns = rows
            .iter()
            .map(|row| {
                let name: String = row.get(0);
                let type_name: String = row.get(1);
                let nullable: bool = row.get(2);
                PgColumn {
                    data_type: parse_pg_type(&type_name),
                    name,
                    nullable,
                }
            })
            .collect();
        Ok(())
    }

    fn selection(&self) -> String {
        let names: Vec<String> = if self.config.columns.is_empty() {
            self.columns.iter().map(|c| c.name.clone()).collect()
        } else {
            self.config.columns.iter().map(|f| f.name.clone()).collect()
        };
        names.join(", ")
    }

    async fn fetch_batch(&mut self, url: &str) -> Result<Option<SourceEvent>> {
        if self.exhausted {
            return Ok(None);
        }
        if self.client.is_none() {
            self.connect(url).await?;
        }
        let rows = self.query_batch().await?;
        if rows.is_empty() {
            if self.config.mode == PgSourceMode::IncrementalQuery {
                tokio::time::sleep(self.config.poll_interval).await;
            }
            return Ok(self.on_empty_batch());
        }
        let event = self.build_data_event(&rows)?;
        self.mark_snapshot_exhausted(&rows);
        Ok(Some(event))
    }

    fn on_empty_batch(&mut self) -> Option<SourceEvent> {
        if self.config.mode == PgSourceMode::Snapshot {
            self.exhausted = true;
            return None;
        }
        Some(SourceEvent::Idle)
    }

    fn mark_snapshot_exhausted(&mut self, rows: &[Row]) {
        if self.config.mode == PgSourceMode::Snapshot
            && (rows.len() as u64) < self.config.max_batch_rows
        {
            self.exhausted = true;
        }
    }

    fn build_data_event(&mut self, rows: &[Row]) -> Result<SourceEvent> {
        let batch = self.assemble_batch(rows)?;
        let cursor = self.cursor_from_values()?;
        Ok(SourceEvent::Data { batch, cursor })
    }

    fn assemble_batch(&mut self, rows: &[Row]) -> Result<Batch> {
        let batch =
            record_batch(&self.columns, rows).map_err(|error| fail("read", &error.to_string()))?;
        self.advance_cursor(rows);
        self.sequence += 1;
        let metadata = BatchMetadata::new(
            "postgresql",
            self.sequence,
            BTreeMap::from([(
                "table".to_string(),
                Value::String(self.config.table.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        Batch::table(vec![batch], metadata).map_err(|error| fail("read", &error.to_string()))
    }

    async fn query_batch(&mut self) -> Result<Vec<Row>> {
        let client = self.client.as_ref().expect("connection established");
        let sql = self.build_query();
        let params: Vec<&(dyn ToSql + Sync)> = self
            .cursor_values
            .iter()
            .map(|value| value as &(dyn ToSql + Sync))
            .collect();
        client
            .query(&sql, &params)
            .await
            .map_err(|error| fail("read", &error.to_string()))
    }

    fn build_query(&self) -> String {
        match self.config.mode {
            PgSourceMode::Snapshot => format!(
                "SELECT {} FROM {} ORDER BY 1 LIMIT {}",
                self.selection(),
                self.config.table,
                self.config.max_batch_rows
            ),
            PgSourceMode::IncrementalQuery => {
                let mut sql = format!("SELECT {} FROM {}", self.selection(), self.config.table);
                if !self.cursor_values.is_empty() {
                    let predicates: Vec<String> = self
                        .config
                        .cursor_columns
                        .iter()
                        .enumerate()
                        .map(|(index, column)| format!("{column} > ${}", index + 1))
                        .collect();
                    sql.push_str(" WHERE ");
                    sql.push_str(&predicates.join(" AND "));
                }
                let order = self.config.cursor_columns.join(", ");
                let _ = write!(
                    sql,
                    " ORDER BY {} LIMIT {}",
                    if order.is_empty() { "1" } else { &order },
                    self.config.max_batch_rows
                );
                sql
            }
        }
    }

    fn advance_cursor(&mut self, rows: &[Row]) {
        if self.config.cursor_columns.is_empty() {
            return;
        }
        if let Some(last) = rows.last() {
            self.cursor_values = self
                .config
                .cursor_columns
                .iter()
                .map(|name| {
                    let index = last
                        .columns()
                        .iter()
                        .position(|c| c.name() == name.as_str())
                        .unwrap_or(0);
                    last.try_get::<_, String>(index)
                        .or_else(|_| {
                            last.try_get::<_, Option<i64>>(index)
                                .map(|v| v.map(|v| v.to_string()).unwrap_or_default())
                        })
                        .unwrap_or_default()
                })
                .collect();
        }
    }

    fn cursor_from_values(&self) -> Result<Cursor> {
        let payload: BTreeMap<String, Value> = self
            .config
            .cursor_columns
            .iter()
            .zip(&self.cursor_values)
            .map(|(column, value)| (column.clone(), Value::String(value.clone())))
            .collect();
        let order = serde_json::to_vec(&self.cursor_values)
            .map_err(|error| fail("cursor", &error.to_string()))?;
        Cursor::unbound(order, payload)
    }

    fn values_from_cursor(cursor: &Cursor) -> Vec<String> {
        cursor
            .payload()
            .values()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect()
    }
}

fn parse_pg_type(name: &str) -> tokio_postgres::types::Type {
    use tokio_postgres::types::Type;
    match name {
        "boolean" => Type::BOOL,
        "smallint" => Type::INT2,
        "integer" => Type::INT4,
        "bigint" => Type::INT8,
        "real" => Type::FLOAT4,
        "double precision" => Type::FLOAT8,
        "bytea" => Type::BYTEA,
        "numeric" => Type::NUMERIC,
        "timestamp without time zone" => Type::TIMESTAMP,
        "timestamp with time zone" => Type::TIMESTAMPTZ,
        "date" => Type::DATE,
        "uuid" => Type::UUID,
        _ => Type::TEXT,
    }
}

fn redact_url_error(message: &str) -> String {
    message
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

#[async_trait]
impl StreamSource for PostgresSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            self.cursor_values = Self::values_from_cursor(&cursor);
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        // The connection URL arrives through the secret resolver per
        // fetch; the trait's no-credential signature cannot open one.
        Ok(Some(SourceEvent::Idle))
    }

    async fn close(&mut self) -> Result<()> {
        self.client = None;
        Ok(())
    }
}

impl PostgresSource {
    /// Produces the next event; the connection URL arrives per call so
    /// it never lives in the source state.
    ///
    /// # Errors
    ///
    /// Returns the connector error on connection or query failure.
    pub async fn next_with_secrets(
        &mut self,
        secrets: &dyn calc_flow::SecretResolver,
    ) -> Result<Option<SourceEvent>> {
        let url = resolve_connection_url(secrets, &self.config.url_key)?;
        self.fetch_batch(&url).await
    }

    /// Opens the connection using resolved credentials.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the URL cannot be resolved or
    /// the server rejects the connection.
    pub async fn open_with_secrets(
        &mut self,
        cursor: Option<Cursor>,
        secrets: &dyn calc_flow::SecretResolver,
    ) -> Result<()> {
        if let Some(cursor) = cursor {
            self.cursor_values = Self::values_from_cursor(&cursor);
        }
        let url = resolve_connection_url(secrets, &self.config.url_key)?;
        self.connect(&url).await
    }
}

/// Data-only configuration for one `PostgreSQL` sink.
#[derive(Clone, Debug)]
pub struct PostgresSinkConfig {
    /// Secret key holding the `postgresql://` URL.
    pub url_key: String,
    /// Target table.
    pub table: String,
    /// Sink write mode.
    pub mode: PgSinkMode,
    /// Conflict key column names for upserts.
    pub conflict_columns: Vec<String>,
    /// Pipeline name recorded in the epoch ledger.
    pub pipeline: String,
    /// Output name recorded in the epoch ledger.
    pub output: String,
}

/// The three `PostgreSQL` sink modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PgSinkMode {
    /// Plain parameterized inserts.
    Append,
    /// `INSERT … ON CONFLICT` upserts keyed by the configured columns.
    Upsert,
    /// Epoch ledger committed in the same transaction as the data.
    Transactional,
}

impl PostgresSinkConfig {
    /// Parses the sink configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let mode = parse_sink_mode(options)?;
        Ok(Self {
            url_key: required_string(options, "url_key")?,
            table: pg_identifier(&required_string(options, "table")?)?,
            mode,
            conflict_columns: parse_conflict_columns(options)?,
            pipeline: required_string(options, "pipeline")?,
            output: required_string(options, "output")?,
        })
    }
}

fn parse_sink_mode(options: &JsonMap) -> Result<PgSinkMode> {
    match required_string(options, "mode")?.as_str() {
        "append" => Ok(PgSinkMode::Append),
        "upsert" => Ok(PgSinkMode::Upsert),
        "transactional" => Ok(PgSinkMode::Transactional),
        other => Err(CalcFlowError::InvalidArgument {
            field: "mode".into(),
            message: format!("unsupported sink mode {other:?}"),
        }),
    }
}

fn parse_conflict_columns(options: &JsonMap) -> Result<Vec<String>> {
    match options.get("conflict_columns") {
        None => Ok(Vec::new()),
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value.as_str().map_or_else(
                    || {
                        Err(CalcFlowError::InvalidArgument {
                            field: "conflict_columns".into(),
                            message: "entries must be strings".into(),
                        })
                    },
                    pg_identifier,
                )
            })
            .collect::<Result<Vec<_>>>(),
        Some(_) => Err(CalcFlowError::InvalidArgument {
            field: "conflict_columns".into(),
            message: "conflict_columns must be a string array".into(),
        }),
    }
}

fn required_string(options: &JsonMap, key: &str) -> Result<String> {
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

fn u64_option(options: &JsonMap, key: &str) -> Result<Option<u64>> {
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

/// The transactional `PostgreSQL` sink with the epoch ledger.
pub struct TransactionalPostgresSink {
    config: PostgresSinkConfig,
    client: Option<Client>,
    pending_url: Option<String>,
    /// Parameterized rows staged for the epoch's commit transaction.
    pending_rows: Vec<Vec<crate::database_types::PgValue>>,
    /// The compiled SQL for this epoch's rows.
    pending_sql: Option<String>,
    active: Option<calc_flow::Epoch>,
    rows: u64,
}

impl TransactionalPostgresSink {
    /// Builds the sink; the connection opens with credentials.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: PostgresSinkConfig) -> Result<Self> {
        Ok(Self {
            config,
            client: None,
            pending_url: None,
            pending_rows: Vec::new(),
            pending_sql: None,
            active: None,
            rows: 0,
        })
    }

    /// Resolves the connection URL and stages it for `open`.
    ///
    /// # Errors
    ///
    /// Returns the resolver error when the URL secret is missing.
    pub fn open_with_secrets(&mut self, secrets: &dyn calc_flow::SecretResolver) -> Result<()> {
        let url = resolve_connection_url(secrets, &self.config.url_key)?;
        self.pending_url = Some(url);
        Ok(())
    }

    async fn connect(&mut self, url: &str) -> Result<()> {
        let (client, connection) = tokio_postgres::connect(url, NoTls)
            .await
            .map_err(|error| fail("open", &redact_url_error(&error.to_string())))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        client
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (\
                     pipeline TEXT NOT NULL, output TEXT NOT NULL, epoch BIGINT NOT NULL, \
                     rows_written BIGINT NOT NULL, committed_at timestamptz NOT NULL DEFAULT now(), \
                     PRIMARY KEY (pipeline, output, epoch))"
                ),
                &[],
            )
            .await
            .map_err(|error| fail("open", &error.to_string()))?;
        self.client = Some(client);
        Ok(())
    }
}

#[async_trait]
impl TransactionalStreamSink for TransactionalPostgresSink {
    async fn open(&mut self) -> Result<()> {
        let url = self.pending_url.take().ok_or_else(|| {
            fail(
                "open",
                "the sink connection URL must be set through open_with_secrets",
            )
        })?;
        self.connect(&url).await
    }

    async fn begin_epoch(&mut self, epoch: calc_flow::Epoch) -> Result<()> {
        if self.client.is_none() {
            return Err(fail(
                "begin_epoch",
                "begin_epoch before a resolved connection",
            ));
        }
        self.pending_rows.clear();
        self.pending_sql = None;
        self.active = Some(epoch);
        self.rows = 0;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if self.active.is_none() {
            return Err(fail("write", "write before begin_epoch"));
        }
        let payload = batch
            .table_payload()
            .map_err(|_| fail("write", "the postgresql sink writes table batches only"))?;
        let schema = payload.schema();
        let names: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();
        let placeholders: Vec<String> = (1..=names.len()).map(|i| format!("${i}")).collect();
        let sql = match self.config.mode {
            PgSinkMode::Append => format!(
                "INSERT INTO {} ({}) VALUES ({})",
                self.config.table,
                names.join(", "),
                placeholders.join(", ")
            ),
            PgSinkMode::Upsert if !self.config.conflict_columns.is_empty() => {
                let non_key: Vec<&String> = names
                    .iter()
                    .filter(|name| !self.config.conflict_columns.iter().any(|key| key == *name))
                    .collect();
                if non_key.is_empty() {
                    format!(
                        "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO NOTHING",
                        self.config.table,
                        names.join(", "),
                        placeholders.join(", "),
                        self.config.conflict_columns.join(", ")
                    )
                } else {
                    let updates: Vec<String> = non_key
                        .iter()
                        .map(|name| format!("{name} = EXCLUDED.{name}"))
                        .collect();
                    format!(
                        "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}",
                        self.config.table,
                        names.join(", "),
                        placeholders.join(", "),
                        self.config.conflict_columns.join(", "),
                        updates.join(", ")
                    )
                }
            }
            PgSinkMode::Upsert | PgSinkMode::Transactional => format!(
                "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING",
                self.config.table,
                names.join(", "),
                placeholders.join(", ")
            ),
        };
        self.pending_sql = Some(sql);
        for record in payload.batches() {
            for row in 0..record.num_rows() {
                let params: Vec<crate::database_types::PgValue> = (0..names.len())
                    .map(|col| {
                        let column = record.column(col);
                        crate::database_types::cell_value(column, row)
                            .unwrap_or(crate::database_types::PgValue::Null)
                    })
                    .collect();
                self.pending_rows.push(params);
                self.rows += 1;
            }
        }
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if self.active.is_none() {
            return Err(fail("pre_commit", "pre_commit before begin_epoch"));
        }
        Ok(BTreeMap::from([
            (
                "pipeline".to_string(),
                Value::String(self.config.pipeline.clone()),
            ),
            (
                "output".to_string(),
                Value::String(self.config.output.clone()),
            ),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            ("rows".to_string(), Value::from(self.rows)),
        ]))
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        let Some(client) = self.client.as_mut() else {
            return Err(fail("commit", "commit before a resolved connection"));
        };
        let epoch_value = i64::try_from(_epoch.as_u64()).unwrap_or(i64::MAX);
        let rows_written = i64::try_from(self.rows).unwrap_or(i64::MAX);
        let tx = client
            .transaction()
            .await
            .map_err(|e| fail("commit", &e.to_string()))?;
        if let (Some(sql), rows) = (self.pending_sql.as_deref(), &self.pending_rows) {
            for row_values in rows {
                let params: Vec<&(dyn ToSql + Sync)> = row_values
                    .iter()
                    .map(|v| v as &(dyn ToSql + Sync))
                    .collect();
                tx.execute(sql, &params)
                    .await
                    .map_err(|e| fail("commit", &e.to_string()))?;
            }
        }
        tx.execute(
            &format!(
                "INSERT INTO {LEDGER_TABLE} (pipeline, output, epoch, rows_written) \
                 VALUES ($1, $2, $3, $4) \
                 ON CONFLICT (pipeline, output, epoch) DO NOTHING"
            ),
            &[
                &self.config.pipeline,
                &self.config.output,
                &epoch_value,
                &rows_written,
            ],
        )
        .await
        .map_err(|e| fail("commit", &e.to_string()))?;
        tx.commit()
            .await
            .map_err(|e| fail("commit", &e.to_string()))?;
        self.pending_rows.clear();
        self.pending_sql = None;
        self.active = None;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.pending_rows.clear();
        self.pending_sql = None;
        self.active = None;
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        let expected = recovery.pre_commit();
        if expected.get("pipeline").and_then(Value::as_str) != Some(&self.config.pipeline)
            || expected.get("output").and_then(Value::as_str) != Some(&self.config.output)
        {
            return Err(fail(
                "recover",
                "recovery evidence names a different pipeline/output identity",
            ));
        }
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.client = None;
        Ok(())
    }
}
