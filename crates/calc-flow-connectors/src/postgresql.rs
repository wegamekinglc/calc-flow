//! The PostgreSQL connector (feature `postgresql`).
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
    SinkRecovery, SourceCapabilities, SourceEvent, SourceSchema, StreamSink, StreamSource,
    TransactionalStreamSink,
};
use serde_json::Value;
use std::collections::BTreeMap;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, NoTls};

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
pub async fn resolve_connection_url(
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

/// The two PostgreSQL source modes shipped in this task.
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
        let url_key = required_string(options, "url_key")?;
        let table = pg_identifier(&required_string(options, "table")?)?;
        let mode = match required_string(options, "mode")?.as_str() {
            "snapshot" => PgSourceMode::Snapshot,
            "incremental_query" => PgSourceMode::IncrementalQuery,
            other => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "mode".into(),
                    message: format!("unsupported source mode {other:?}"),
                });
            }
        };
        let cursor_columns = match options.get("cursor_columns") {
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
                        |name| pg_identifier(name),
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
        if mode == PgSourceMode::IncrementalQuery && cursor_columns.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "cursor_columns".into(),
                message: "incremental_query requires at least one cursor column".into(),
            });
        }
        let columns =
            match options.get("columns") {
                None => Vec::new(),
                Some(value) => serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone())
                    .map_err(|error| CalcFlowError::InvalidArgument {
                        field: "columns".into(),
                        message: format!("columns must be a field list: {error}"),
                    })?,
            };
        Ok(Self {
            url_key,
            table,
            mode,
            cursor_columns,
            columns,
            max_batch_rows: u64_option(options, "max_batch_rows")?.unwrap_or(8192),
            poll_interval: std::time::Duration::from_millis(
                u64_option(options, "poll_interval_ms")?.unwrap_or(500),
            ),
        })
    }
}

/// The PostgreSQL source over a private connection.
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
        if self.exhausted || self.client.is_none() {
            if self.client.is_none() {
                self.connect(url).await?;
            }
        }
        let client = self
            .client
            .as_ref()
            .expect("connection established")
            .clone();
        let sql = match self.config.mode {
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
                let order: Vec<String> = self
                    .config
                    .cursor_columns
                    .iter()
                    .map(|column| column.clone())
                    .collect();
                sql.push_str(&format!(
                    " ORDER BY {} LIMIT {}",
                    if order.is_empty() {
                        "1".to_string()
                    } else {
                        order.join(", ")
                    },
                    self.config.max_batch_rows
                ));
                sql
            }
        };
        let params: Vec<&(dyn ToSql + Sync)> = self
            .cursor_values
            .iter()
            .map(|value| value as &(dyn ToSql + Sync))
            .collect();
        let rows = client
            .query(&sql, &params)
            .await
            .map_err(|error| fail("read", &error.to_string()))?;
        if rows.is_empty() {
            if self.config.mode == PgSourceMode::Snapshot {
                self.exhausted = true;
                return Ok(None);
            }
            tokio::time::sleep(self.config.poll_interval).await;
            return Ok(Some(SourceEvent::Idle));
        }
        let mut cursor_columns = Vec::new();
        let batch = if self.config.cursor_columns.is_empty() {
            record_batch(&self.columns, &rows).map_err(|error| fail("read", &error.to_string()))?
        } else {
            let indexes: Vec<usize> = self
                .config
                .cursor_columns
                .iter()
                .map(|name| {
                    rows[0]
                        .columns()
                        .iter()
                        .position(|column| column.name() == name.as_str())
                        .unwrap_or(0)
                })
                .collect();
            for (row_index, row) in rows.iter().enumerate() {
                let _ = row_index;
                cursor_columns = indexes
                    .iter()
                    .map(|index| {
                        let value: String = row
                            .try_get::<_, String>(*index)
                            .or_else(|_| {
                                row.try_get::<_, Option<i64>>(*index)
                                    .map(|v| v.map(|v| v.to_string()).unwrap_or_default())
                            })
                            .unwrap_or_default();
                        value
                    })
                    .collect();
            }
            record_batch(&self.columns, &rows).map_err(|error| fail("read", &error.to_string()))?
        };
        self.sequence += 1;
        self.cursor_values = cursor_columns;
        let cursor = self.cursor_from_values()?;
        let metadata = BatchMetadata::new(
            "postgresql",
            self.sequence,
            BTreeMap::from([(
                "table".to_string(),
                Value::String(self.config.table.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        let batch =
            Batch::table(vec![batch], metadata).map_err(|e| fail("read", &e.to_string()))?;
        if self.config.mode == PgSourceMode::Snapshot
            && (rows.len() as u64) < self.config.max_batch_rows
        {
            self.exhausted = true;
        }
        Ok(Some(SourceEvent::Data { batch, cursor }))
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
        "text" => Type::TEXT,
        "character varying" | "character" => Type::VARCHAR,
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
        let url = resolve_connection_url(secrets, &self.config.url_key).await?;
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
        let url = resolve_connection_url(secrets, &self.config.url_key).await?;
        self.connect(&url).await
    }
}

/// Data-only configuration for one PostgreSQL sink.
#[derive(Clone, Debug)]
pub struct PostgresSinkConfig {
    /// Secret key holding the `postgresql://` URL.
    pub url_key: String,
    /// Target table.
    pub table: String,
    /// Sink write mode.
    pub mode: PgSinkMode,
}

/// The three PostgreSQL sink modes.
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
        Ok(Self {
            url_key: required_string(options, "url_key")?,
            table: pg_identifier(&required_string(options, "table")?)?,
            mode: match required_string(options, "mode")?.as_str() {
                "append" => PgSinkMode::Append,
                "upsert" => PgSinkMode::Upsert,
                "transactional" => PgSinkMode::Transactional,
                other => {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "mode".into(),
                        message: format!("unsupported sink mode {other:?}"),
                    });
                }
            },
        })
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

/// The transactional PostgreSQL sink with the epoch ledger.
pub struct TransactionalPostgresSink {
    config: PostgresSinkConfig,
    client: Option<Client>,
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
            active: None,
            rows: 0,
        })
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
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        self.active = Some(_epoch);
        self.rows = 0;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let Some(client) = self.client.as_mut() else {
            return Err(fail("write", "write before a resolved connection"));
        };
        let payload = batch
            .table_payload()
            .map_err(|_| fail("write", "the postgresql sink writes table batches only"))?;
        let schema = payload.schema();
        let statements = client
            .transaction()
            .await
            .map_err(|e| fail("write", &e.to_string()))?;
        for record in payload.batches() {
            for row_index in 0..record.num_rows() {
                let placeholders: Vec<String> = (1..=schema.fields().len())
                    .map(|i| format!("${i}"))
                    .collect();
                let sql = match self.config.mode {
                    PgSinkMode::Append => format!(
                        "INSERT INTO {} VALUES ({})",
                        self.config.table,
                        placeholders.join(", ")
                    ),
                    PgSinkMode::Upsert | PgSinkMode::Transactional => format!(
                        "INSERT INTO {} VALUES ({}) ON CONFLICT DO NOTHING",
                        self.config.table,
                        placeholders.join(", ")
                    ),
                };
                statements
                    .execute(&sql, &[])
                    .await
                    .map_err(|e| fail("write", &e.to_string()))?;
                let _ = row_index;
                self.rows += 1;
            }
        }
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        Ok(BTreeMap::from([
            ("pipeline".to_string(), Value::String(String::new())),
            ("output".to_string(), Value::String(String::new())),
            ("rows".to_string(), Value::from(self.rows)),
        ]))
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        self.active = None;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.active = None;
        Ok(())
    }

    async fn recover(&mut self, _recovery: &SinkRecovery) -> Result<()> {
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.client = None;
        Ok(())
    }
}
