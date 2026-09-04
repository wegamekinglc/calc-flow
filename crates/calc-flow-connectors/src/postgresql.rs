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
    ArrowFieldSpec, Batch, BatchMetadata, CalcFlowError, ConnectorCapabilities,
    ConnectorDescriptor, ConnectorError, ConnectorFactories, ConnectorIdentity, ConnectorKind,
    ConnectorOperation, ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor,
    DeliveryCapability, JsonMap, Result, SecretHandle, SecretReference, SecretResolver,
    SecretResolverKind, SinkRecovery, SourceCapabilities, SourceEvent, SourceSchema, StreamSink,
    StreamSource, TransactionSupport, TransactionalStreamSink, WatermarkSupport,
};
use serde_json::Value;
use sha2::{Digest as _, Sha256};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::str::FromStr as _;
use tokio_postgres::types::ToSql;
use tokio_postgres::{Client, Row};

use crate::database_types::{PgColumn, arrow_schema, pg_identifier, record_batch};
use crate::options::{bool_option, required_string, u64_option};

const PREPARED_SEGMENT_ID: &str = "prepared-rows";

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

pub(crate) type ConnectionDriver =
    tokio::task::JoinHandle<std::result::Result<(), tokio_postgres::Error>>;

pub(crate) async fn connect_postgres(url: &str) -> Result<(Client, ConnectionDriver)> {
    let config = tokio_postgres::Config::from_str(url)
        .map_err(|_| fail("open", "the connection URL could not be parsed"))?;
    let _ = rustls::crypto::ring::default_provider().install_default();
    let tls = tokio_postgres_rustls::MakeRustlsConnect::with_webpki_roots();
    let (client, connection) = config
        .connect(tls)
        .await
        .map_err(|error| fail("open", &redact_url_error(&error.to_string())))?;
    Ok((client, tokio::spawn(connection)))
}

pub(crate) async fn settle_connection(
    client: &mut Option<Client>,
    driver: &mut Option<ConnectionDriver>,
) -> Result<()> {
    *client = None;
    let Some(driver) = driver.take() else {
        return Ok(());
    };
    if !driver.is_finished() {
        driver.abort();
    }
    match driver.await {
        Ok(Ok(())) => Ok(()),
        Ok(Err(_)) => Err(fail("close", "the PostgreSQL connection driver failed")),
        Err(error) if error.is_cancelled() => Ok(()),
        Err(_) => Err(fail("close", "the PostgreSQL connection driver panicked")),
    }
}

/// Reads the connection URL from a secret reference, never from
/// options.
///
/// # Errors
///
/// Returns the resolver error when the reference cannot be resolved;
/// the URL value itself never enters the error.
pub fn resolve_connection_url(secrets: &dyn SecretResolver, slot: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Registered, slot)
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
    /// In-memory byte bound of one decoded batch.
    pub max_batch_bytes: u64,
    /// Maximum row changes buffered for one CDC transaction.
    pub max_transaction_rows: u64,
    /// Maximum tuple bytes buffered for one CDC transaction.
    pub max_transaction_bytes: u64,
    /// Poll interval for incremental mode.
    pub poll_interval: std::time::Duration,
    /// Existing durable logical replication slot for `logical_cdc`.
    pub slot: Option<String>,
    /// Publication streamed by `logical_cdc`.
    pub publication: Option<String>,
    /// Explicit durable slot lifecycle policy.
    pub slot_policy: Option<PgSlotPolicy>,
    /// Require complete old rows for update/delete events.
    pub require_before: bool,
}

/// `PostgreSQL` source modes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PgSourceMode {
    /// One repeatable-read consistent snapshot.
    Snapshot,
    /// Strictly monotonic composite-cursor polling.
    IncrementalQuery,
    /// Commit-ordered changes from a durable `pgoutput` slot.
    LogicalCdc,
}

/// Durable replication-slot ownership policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PgSlotPolicy {
    /// Reuse a pre-provisioned slot and never drop it automatically.
    RequireExisting,
    /// Create a new durable slot and copy its exported snapshot before CDC.
    CreateWithSnapshot,
    /// Explicitly replace an inactive slot, then copy a new exported snapshot.
    RecreateWithSnapshot,
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
        if options.contains_key("url_key") {
            return Err(CalcFlowError::InvalidArgument {
                field: "options".into(),
                message: "the connection URL must use a secret reference".into(),
            });
        }
        let table = parse_source_endpoint(options)?;
        let mode = parse_source_mode(options)?;
        let (
            max_batch_rows,
            max_batch_bytes,
            max_transaction_rows,
            max_transaction_bytes,
            poll_interval_ms,
        ) = parse_source_bounds(options)?;
        let cursor_columns = parse_cursor_columns(options, mode)?;
        let columns = parse_column_list(options)?;
        crate::database_types::pg_identifiers(&columns)?;
        if mode == PgSourceMode::LogicalCdc && columns.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "columns".into(),
                message: "logical_cdc requires the complete frozen table schema".into(),
            });
        }
        let (slot, publication, slot_policy) = if mode == PgSourceMode::LogicalCdc {
            (
                Some(pg_identifier(&required_string(options, "slot")?)?),
                Some(pg_identifier(&required_string(options, "publication")?)?),
                Some(parse_slot_policy(options)?),
            )
        } else {
            (None, None, None)
        };
        let require_before = bool_option(options, "require_before")?.unwrap_or(false);
        if !columns.is_empty()
            && cursor_columns
                .iter()
                .any(|cursor| !columns.iter().any(|field| field.name == *cursor))
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "columns".into(),
                message: "explicit projections must include every cursor column".into(),
            });
        }
        Ok(Self {
            table,
            mode,
            cursor_columns,
            columns,
            max_batch_rows,
            max_batch_bytes,
            max_transaction_rows,
            max_transaction_bytes,
            poll_interval: std::time::Duration::from_millis(poll_interval_ms),
            slot,
            publication,
            slot_policy,
            require_before,
        })
    }
}

fn parse_slot_policy(options: &JsonMap) -> Result<PgSlotPolicy> {
    match required_string(options, "slot_policy")?.as_str() {
        "require_existing" => Ok(PgSlotPolicy::RequireExisting),
        "create_with_snapshot" => Ok(PgSlotPolicy::CreateWithSnapshot),
        "recreate_with_snapshot" => Ok(PgSlotPolicy::RecreateWithSnapshot),
        other => Err(CalcFlowError::InvalidArgument {
            field: "slot_policy".into(),
            message: format!(
                "unsupported slot policy {other:?}; slot creation and replacement must be explicit"
            ),
        }),
    }
}

fn parse_source_endpoint(options: &JsonMap) -> Result<String> {
    pg_identifier(&required_string(options, "table")?)
}

fn parse_source_bounds(options: &JsonMap) -> Result<(u64, u64, u64, u64, u64)> {
    let max_batch_rows = positive_u64_option(options, "max_batch_rows", 8192)?;
    let max_batch_bytes = positive_u64_option(options, "max_batch_bytes", 64 * 1024 * 1024)?;
    let max_transaction_rows = positive_u64_option(
        options,
        "max_transaction_rows",
        max_batch_rows.max(1_000_000),
    )?;
    let max_transaction_bytes = positive_u64_option(
        options,
        "max_transaction_bytes",
        max_batch_bytes.max(256 * 1024 * 1024),
    )?;
    if max_transaction_rows < max_batch_rows {
        return Err(CalcFlowError::InvalidArgument {
            field: "max_transaction_rows".into(),
            message: "must be greater than or equal to max_batch_rows".into(),
        });
    }
    if max_transaction_bytes < max_batch_bytes {
        return Err(CalcFlowError::InvalidArgument {
            field: "max_transaction_bytes".into(),
            message: "must be greater than or equal to max_batch_bytes".into(),
        });
    }
    Ok((
        max_batch_rows,
        max_batch_bytes,
        max_transaction_rows,
        max_transaction_bytes,
        positive_u64_option(options, "poll_interval_ms", 500)?,
    ))
}

fn positive_u64_option(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
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

fn parse_source_mode(options: &JsonMap) -> Result<PgSourceMode> {
    match required_string(options, "mode")?.as_str() {
        "snapshot" => Ok(PgSourceMode::Snapshot),
        "incremental_query" => Ok(PgSourceMode::IncrementalQuery),
        "logical_cdc" => Ok(PgSourceMode::LogicalCdc),
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
    connection_driver: Option<ConnectionDriver>,
    columns: Vec<PgColumn>,
    cursor_values: Vec<String>,
    sequence: u64,
    exhausted: bool,
    endpoint_url: Option<String>,
    snapshot_offset: u64,
    snapshot_transaction_open: bool,
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
            replay_positioning: match config.mode {
                PgSourceMode::Snapshot => calc_flow::ReplayPositioning::Unsupported,
                PgSourceMode::IncrementalQuery => {
                    calc_flow::ReplayPositioning::ExactPauseReportAndSeek
                }
                PgSourceMode::LogicalCdc => {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "mode".into(),
                        message: "logical_cdc uses the dedicated pgoutput source".into(),
                    });
                }
            },
            delivery: match config.mode {
                PgSourceMode::Snapshot => calc_flow::SourceDeliveryCapability::Lossy,
                PgSourceMode::IncrementalQuery => calc_flow::SourceDeliveryCapability::Lossless,
                PgSourceMode::LogicalCdc => unreachable!("validated above"),
            },
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_batch_bytes).unwrap_or(usize::MAX),
            schema,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            capabilities,
            config,
            client: None,
            connection_driver: None,
            columns: Vec::new(),
            cursor_values: Vec::new(),
            sequence: 0,
            exhausted: false,
            endpoint_url: None,
            snapshot_offset: 0,
            snapshot_transaction_open: false,
        })
    }

    fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.endpoint_url = Some(endpoint_url);
        self
    }

    async fn connect(&mut self, url: &str) -> Result<()> {
        let (client, connection_driver) = connect_postgres(url).await?;
        self.connection_driver = Some(connection_driver);
        self.client = Some(client);
        if let Err(error) = self.load_columns().await {
            let primary = fail("open", &error.to_string());
            let _ = settle_connection(&mut self.client, &mut self.connection_driver).await;
            return Err(primary);
        }
        if let Err(error) = self.validate_cursor_columns() {
            let _ = settle_connection(&mut self.client, &mut self.connection_driver).await;
            return Err(error);
        }
        if self.config.mode == PgSourceMode::IncrementalQuery
            && let Err(error) = self.validate_cursor_uniqueness().await
        {
            let _ = settle_connection(&mut self.client, &mut self.connection_driver).await;
            return Err(error);
        }
        if self.config.mode == PgSourceMode::Snapshot {
            if let Err(error) = self
                .client
                .as_ref()
                .expect("connection established")
                .batch_execute("BEGIN TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
                .await
            {
                let primary = fail("open", &error.to_string());
                let _ = settle_connection(&mut self.client, &mut self.connection_driver).await;
                return Err(primary);
            }
            self.snapshot_transaction_open = true;
        }
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

    fn validate_cursor_columns(&self) -> Result<()> {
        for name in &self.config.cursor_columns {
            let Some(column) = self.columns.iter().find(|column| column.name == *name) else {
                return Err(fail(
                    "open",
                    &format!("cursor column {name:?} does not exist in the source table"),
                ));
            };
            if !cursor_type_supported(&column.data_type) {
                return Err(fail(
                    "open",
                    &format!(
                        "cursor column {name:?} uses unsupported type {}",
                        column.data_type.name()
                    ),
                ));
            }
        }
        Ok(())
    }

    async fn validate_cursor_uniqueness(&self) -> Result<()> {
        let client = self.client.as_ref().expect("connection established");
        let indexes = client
            .query(
                "SELECT array_agg(a.attname ORDER BY key.ordinality) \
                 FROM pg_index i \
                 JOIN pg_class c ON c.oid = i.indrelid \
                 JOIN pg_namespace n ON n.oid = c.relnamespace \
                 JOIN LATERAL unnest(i.indkey) WITH ORDINALITY AS key(attnum, ordinality) \
                   ON key.ordinality <= i.indnkeyatts \
                 JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = key.attnum \
                 WHERE c.relname = $1 AND n.nspname = ANY(current_schemas(false)) \
                   AND i.indisunique AND i.indisvalid \
                 GROUP BY i.indexrelid",
                &[&self.config.table],
            )
            .await
            .map_err(|error| fail("open", &error.to_string()))?;
        let cursor_is_unique = indexes.iter().any(|row| {
            let columns: Vec<String> = row.get(0);
            columns
                .iter()
                .all(|column| self.config.cursor_columns.contains(column))
        });
        if !cursor_is_unique {
            return Err(fail(
                "open",
                "incremental cursor must include every column of a valid unique index",
            ));
        }
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
        Ok(Some(event))
    }

    fn on_empty_batch(&mut self) -> Option<SourceEvent> {
        if self.config.mode == PgSourceMode::Snapshot {
            self.exhausted = true;
            return None;
        }
        Some(SourceEvent::Idle)
    }

    fn build_data_event(&mut self, rows: &[Row]) -> Result<SourceEvent> {
        let batch = self.assemble_batch(rows)?;
        if self.config.mode == PgSourceMode::Snapshot {
            self.snapshot_offset = self
                .snapshot_offset
                .checked_add(u64::try_from(rows.len()).unwrap_or(u64::MAX))
                .ok_or_else(|| fail("cursor", "snapshot offset exhausted"))?;
            if u64::try_from(rows.len()).unwrap_or(u64::MAX) < self.config.max_batch_rows {
                self.exhausted = true;
            }
        }
        let cursor = self.cursor_from_values()?;
        Ok(SourceEvent::Data { batch, cursor })
    }

    fn assemble_batch(&mut self, rows: &[Row]) -> Result<Batch> {
        let batch =
            record_batch(&self.columns, rows).map_err(|error| fail("read", &error.to_string()))?;
        self.advance_cursor(rows);
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("read", "source sequence exhausted u64"))?;
        let metadata = BatchMetadata::new(
            "postgresql",
            self.sequence,
            BTreeMap::from([(
                "table".to_string(),
                Value::String(self.config.table.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        let batch = Batch::table(vec![batch], metadata)
            .map_err(|error| fail("read", &error.to_string()))?;
        if u64::try_from(batch.estimated_bytes()?).unwrap_or(u64::MAX) > self.config.max_batch_bytes
        {
            return Err(fail(
                "read",
                "decoded PostgreSQL batch exceeds max_batch_bytes",
            ));
        }
        Ok(batch)
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
                "SELECT {} FROM {} ORDER BY ctid LIMIT {} OFFSET {}",
                self.selection(),
                self.config.table,
                self.config.max_batch_rows,
                self.snapshot_offset
            ),
            PgSourceMode::IncrementalQuery => {
                let mut sql = format!("SELECT {} FROM {}", self.selection(), self.config.table);
                if !self.cursor_values.is_empty() {
                    sql.push_str(" WHERE ");
                    sql.push_str(&self.composite_cursor_predicate());
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
            PgSourceMode::LogicalCdc => unreachable!("logical CDC uses its dedicated source"),
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
                    cursor_cell_to_string(last, index, last.columns()[index].type_())
                })
                .collect();
        }
    }

    fn cursor_from_values(&self) -> Result<Cursor> {
        if self.config.mode == PgSourceMode::Snapshot {
            return Cursor::unbound(
                self.snapshot_offset.to_be_bytes().to_vec(),
                BTreeMap::from([(
                    "snapshot_offset".to_string(),
                    Value::from(self.snapshot_offset),
                )]),
            );
        }
        let mut payload: BTreeMap<String, Value> = self
            .config
            .cursor_columns
            .iter()
            .zip(&self.cursor_values)
            .map(|(column, value)| (column.clone(), Value::String(value.clone())))
            .collect();
        payload.insert("sequence".into(), Value::from(self.sequence));
        Cursor::unbound(self.sequence.to_be_bytes().to_vec(), payload)
    }

    fn restore_cursor(&mut self, cursor: &Cursor) -> Result<()> {
        if self.config.mode == PgSourceMode::Snapshot {
            return Err(fail(
                "open",
                "snapshot mode cannot restore the same repeatable-read transaction",
            ));
        }
        let sequence = cursor
            .payload()
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("open", "PostgreSQL cursor sequence is missing"))?;
        if cursor.order() != sequence.to_be_bytes() {
            return Err(fail(
                "open",
                "PostgreSQL cursor order does not match its sequence",
            ));
        }
        self.cursor_values = self
            .config
            .cursor_columns
            .iter()
            .map(|column| {
                Ok(cursor
                    .payload()
                    .get(column)
                    .and_then(Value::as_str)
                    .ok_or_else(|| fail("open", "PostgreSQL cursor field is missing"))?
                    .to_string())
            })
            .collect::<Result<Vec<_>>>()?;
        self.sequence = sequence;
        Ok(())
    }

    fn composite_cursor_predicate(&self) -> String {
        let values = self
            .config
            .cursor_columns
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let data_type = self
                    .columns
                    .iter()
                    .find(|column| column.name == *name)
                    .map_or(&tokio_postgres::types::Type::TEXT, |column| {
                        &column.data_type
                    });
                format!(
                    "CAST(${} AS text)::{}",
                    index + 1,
                    cursor_sql_type(data_type)
                )
            })
            .collect::<Vec<_>>();
        format!(
            "({}) > ({})",
            self.config.cursor_columns.join(", "),
            values.join(", ")
        )
    }
}

pub(crate) fn parse_pg_type(name: &str) -> tokio_postgres::types::Type {
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
        "json" => Type::JSON,
        "jsonb" => Type::JSONB,
        _ => Type::TEXT,
    }
}

fn cursor_sql_type(data_type: &tokio_postgres::types::Type) -> &'static str {
    use tokio_postgres::types::Type;
    match data_type.clone() {
        Type::BOOL => "boolean",
        Type::INT2 => "smallint",
        Type::INT4 => "integer",
        Type::INT8 => "bigint",
        Type::FLOAT4 => "real",
        Type::FLOAT8 => "double precision",
        Type::TIMESTAMP => "timestamp",
        Type::TIMESTAMPTZ => "timestamptz",
        Type::DATE => "date",
        Type::UUID => "uuid",
        _ => "text",
    }
}

fn cursor_type_supported(data_type: &tokio_postgres::types::Type) -> bool {
    use tokio_postgres::types::Type;
    matches!(
        data_type.clone(),
        Type::BOOL
            | Type::INT2
            | Type::INT4
            | Type::INT8
            | Type::FLOAT4
            | Type::FLOAT8
            | Type::TEXT
            | Type::VARCHAR
            | Type::BPCHAR
            | Type::NAME
            | Type::TIMESTAMP
            | Type::TIMESTAMPTZ
            | Type::DATE
            | Type::UUID
    )
}

fn cursor_cell_to_string(
    row: &Row,
    index: usize,
    data_type: &tokio_postgres::types::Type,
) -> String {
    use tokio_postgres::types::Type;
    macro_rules! value {
        ($ty:ty) => {
            row.try_get::<_, Option<$ty>>(index)
                .ok()
                .flatten()
                .map(|value| value.to_string())
                .unwrap_or_default()
        };
    }
    match data_type.clone() {
        Type::BOOL => value!(bool),
        Type::INT2 => value!(i16),
        Type::INT4 => value!(i32),
        Type::INT8 => value!(i64),
        Type::FLOAT4 => value!(f32),
        Type::FLOAT8 => value!(f64),
        Type::TEXT | Type::VARCHAR | Type::BPCHAR | Type::NAME => value!(String),
        Type::TIMESTAMP => value!(chrono::NaiveDateTime),
        Type::TIMESTAMPTZ => value!(chrono::DateTime<chrono::Utc>),
        Type::DATE => value!(chrono::NaiveDate),
        Type::UUID => value!(uuid::Uuid),
        _ => String::new(),
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
            self.restore_cursor(&cursor)?;
        }
        let url = self.endpoint_url.clone().ok_or_else(|| {
            fail(
                "open",
                "the postgresql source was not opened through its trusted factory",
            )
        })?;
        self.connect(&url).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let url = self.endpoint_url.clone().ok_or_else(|| {
            fail(
                "read",
                "the postgresql source was not opened through its trusted factory",
            )
        })?;
        self.fetch_batch(&url).await
    }

    async fn close(&mut self) -> Result<()> {
        if self.snapshot_transaction_open {
            if let Some(client) = self.client.as_ref() {
                client
                    .batch_execute("ROLLBACK")
                    .await
                    .map_err(|error| fail("close", &error.to_string()))?;
            }
            self.snapshot_transaction_open = false;
        }
        settle_connection(&mut self.client, &mut self.connection_driver).await
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
        secrets: &dyn SecretResolver,
    ) -> Result<Option<SourceEvent>> {
        let url = resolve_connection_url(secrets, "url")?;
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
        secrets: &dyn SecretResolver,
    ) -> Result<()> {
        if let Some(cursor) = cursor {
            self.restore_cursor(&cursor)?;
        }
        let url = resolve_connection_url(secrets, "url")?;
        self.connect(&url).await
    }
}

/// Data-only configuration for one `PostgreSQL` sink.
#[derive(Clone, Debug)]
pub struct PostgresSinkConfig {
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
    /// Maximum rows staged for one transactional epoch.
    pub max_epoch_rows: u64,
    /// Maximum encoded bytes staged for one transactional epoch.
    pub max_epoch_bytes: u64,
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
        if options.contains_key("url_key") {
            return Err(CalcFlowError::InvalidArgument {
                field: "options".into(),
                message: "the connection URL must use a secret reference".into(),
            });
        }
        let mode = parse_sink_mode(options)?;
        let conflict_columns = parse_conflict_columns(options)?;
        if mode == PgSinkMode::Upsert && conflict_columns.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "conflict_columns".into(),
                message: "upsert mode requires at least one explicit conflict column".into(),
            });
        }
        Ok(Self {
            table: pg_identifier(&required_string(options, "table")?)?,
            mode,
            conflict_columns,
            pipeline: required_string(options, "pipeline")?,
            output: required_string(options, "output")?,
            max_epoch_rows: positive_sink_bound(options, "max_epoch_rows", 8192)?,
            max_epoch_bytes: positive_sink_bound(options, "max_epoch_bytes", 64 * 1024 * 1024)?,
        })
    }
}

fn positive_sink_bound(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
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

fn compile_insert_sql(config: &PostgresSinkConfig, names: &[String]) -> Result<String> {
    if names.is_empty() {
        return Ok(String::new());
    }
    let names = names
        .iter()
        .map(|name| pg_identifier(name))
        .collect::<Result<Vec<_>>>()?;
    let placeholders = (1..=names.len())
        .map(|index| format!("${index}"))
        .collect::<Vec<_>>();
    match config.mode {
        PgSinkMode::Append | PgSinkMode::Transactional => Ok(format!(
            "INSERT INTO {} ({}) VALUES ({})",
            config.table,
            names.join(", "),
            placeholders.join(", ")
        )),
        PgSinkMode::Upsert => {
            let non_key = names
                .iter()
                .filter(|name| !config.conflict_columns.contains(name))
                .collect::<Vec<_>>();
            if non_key.is_empty() {
                Ok(format!(
                    "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO NOTHING",
                    config.table,
                    names.join(", "),
                    placeholders.join(", "),
                    config.conflict_columns.join(", ")
                ))
            } else {
                let updates = non_key
                    .iter()
                    .map(|name| format!("{name} = EXCLUDED.{name}"))
                    .collect::<Vec<_>>();
                Ok(format!(
                    "INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) DO UPDATE SET {}",
                    config.table,
                    names.join(", "),
                    placeholders.join(", "),
                    config.conflict_columns.join(", "),
                    updates.join(", ")
                ))
            }
        }
    }
}

fn sink_schema_hash(schema: &arrow::datatypes::Schema) -> String {
    let mut hasher = Sha256::new();
    for field in schema.fields() {
        hasher.update(field.name().as_bytes());
        hasher.update([0]);
        hasher.update(format!("{:?}", field.data_type()).as_bytes());
        hasher.update([u8::from(field.is_nullable())]);
    }
    hex::encode(hasher.finalize())
}

fn evidence_rows(rows: &[Vec<crate::database_types::PgValue>]) -> Value {
    Value::Array(
        rows.iter()
            .map(|row| {
                Value::Array(
                    row.iter()
                        .map(crate::database_types::PgValue::to_evidence)
                        .collect(),
                )
            })
            .collect(),
    )
}

fn decode_evidence_rows(
    value: &Value,
    width: usize,
) -> Result<Vec<Vec<crate::database_types::PgValue>>> {
    value
        .as_array()
        .ok_or_else(|| fail("commit", "pre-commit rows must be an array"))?
        .iter()
        .map(|row| {
            let cells = row
                .as_array()
                .ok_or_else(|| fail("commit", "pre-commit row must be an array"))?;
            if cells.len() != width {
                return Err(fail(
                    "commit",
                    "pre-commit row width does not match its column list",
                ));
            }
            cells
                .iter()
                .map(crate::database_types::PgValue::from_evidence)
                .collect()
        })
        .collect()
}

/// The ordinary at-least-once `PostgreSQL` append/upsert sink.
pub struct PostgresSink {
    config: PostgresSinkConfig,
    client: Option<Client>,
    connection_driver: Option<ConnectionDriver>,
    pending_url: Option<String>,
}

impl PostgresSink {
    /// Builds an ordinary append/upsert sink.
    ///
    /// # Errors
    ///
    /// Returns an error when the transactional mode is routed through the
    /// ordinary lifecycle.
    pub fn new(config: PostgresSinkConfig) -> Result<Self> {
        if config.mode == PgSinkMode::Transactional {
            return Err(fail(
                "open",
                "transactional mode requires the epoch-transactional lifecycle",
            ));
        }
        Ok(Self {
            config,
            client: None,
            connection_driver: None,
            pending_url: None,
        })
    }

    fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.pending_url = Some(endpoint_url);
        self
    }
}

#[async_trait]
impl StreamSink for PostgresSink {
    async fn open(&mut self) -> Result<()> {
        let url = self.pending_url.take().ok_or_else(|| {
            fail(
                "open",
                "the sink connection URL must be set through its trusted factory",
            )
        })?;
        let (client, connection_driver) = connect_postgres(&url).await?;
        self.connection_driver = Some(connection_driver);
        self.client = Some(client);
        Ok(())
    }

    // This method validates the complete Arrow-to-PostgreSQL row contract
    // before mutating the pending transaction buffer.
    // #lizard forgives
    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let payload = batch
            .table_payload()
            .map_err(|_| fail("write", "the postgresql sink writes table batches only"))?;
        let names = payload
            .schema()
            .fields()
            .iter()
            .map(|field| pg_identifier(field.name()))
            .collect::<Result<Vec<_>>>()?;
        let sql = compile_insert_sql(&self.config, &names)?;
        let client = self
            .client
            .as_mut()
            .ok_or_else(|| fail("write", "write before open"))?;
        let transaction = client
            .transaction()
            .await
            .map_err(|error| fail("write", &error.to_string()))?;
        for record in payload.batches() {
            for row in 0..record.num_rows() {
                let values = (0..names.len())
                    .map(|column| crate::database_types::cell_value(record.column(column), row))
                    .collect::<Result<Vec<_>>>()?;
                let params = values
                    .iter()
                    .map(|value| value as &(dyn ToSql + Sync))
                    .collect::<Vec<_>>();
                transaction
                    .execute(&sql, &params)
                    .await
                    .map_err(|error| fail("write", &error.to_string()))?;
            }
        }
        transaction
            .commit()
            .await
            .map_err(|error| fail("write", &error.to_string()))
    }

    async fn close(&mut self) -> Result<()> {
        settle_connection(&mut self.client, &mut self.connection_driver).await
    }
}

/// The transactional `PostgreSQL` sink with the epoch ledger.
pub struct TransactionalPostgresSink {
    config: PostgresSinkConfig,
    client: Option<Client>,
    connection_driver: Option<ConnectionDriver>,
    pending_url: Option<String>,
    /// Parameterized rows staged for the epoch's commit transaction.
    pending_rows: Vec<Vec<crate::database_types::PgValue>>,
    /// The compiled SQL for this epoch's rows.
    pending_sql: Option<String>,
    pending_columns: Vec<String>,
    pending_schema_hash: Option<String>,
    active: Option<calc_flow::Epoch>,
    rows: u64,
    pending_bytes: u64,
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
            connection_driver: None,
            pending_url: None,
            pending_rows: Vec::new(),
            pending_sql: None,
            pending_columns: Vec::new(),
            pending_schema_hash: None,
            active: None,
            rows: 0,
            pending_bytes: 2,
        })
    }

    /// Resolves the connection URL and stages it for `open`.
    ///
    /// # Errors
    ///
    /// Returns the resolver error when the URL secret is missing.
    pub fn open_with_secrets(&mut self, secrets: &dyn SecretResolver) -> Result<()> {
        let url = resolve_connection_url(secrets, "url")?;
        self.pending_url = Some(url);
        Ok(())
    }

    fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.pending_url = Some(endpoint_url);
        self
    }

    async fn connect(&mut self, url: &str) -> Result<()> {
        let (client, connection_driver) = connect_postgres(url).await?;
        self.connection_driver = Some(connection_driver);
        self.client = Some(client);
        let create_result = self
            .client
            .as_ref()
            .expect("connection established")
            .execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS {LEDGER_TABLE} (\
                     pipeline TEXT NOT NULL, output TEXT NOT NULL, epoch BIGINT NOT NULL, \
                     rows_written BIGINT NOT NULL, committed_at timestamptz NOT NULL DEFAULT now(), \
                     PRIMARY KEY (pipeline, output, epoch))"
                ),
                &[],
            )
            .await;
        if let Err(error) = create_result {
            let primary = fail("open", &error.to_string());
            let _ = settle_connection(&mut self.client, &mut self.connection_driver).await;
            return Err(primary);
        }
        Ok(())
    }

    fn prepared_evidence(&self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        let prepared_rows = serde_json::to_vec(&evidence_rows(&self.pending_rows))
            .map_err(|error| fail("pre_commit", &error.to_string()))?;
        Ok(BTreeMap::from([
            (
                "pipeline".to_string(),
                Value::String(self.config.pipeline.clone()),
            ),
            (
                "output".to_string(),
                Value::String(self.config.output.clone()),
            ),
            (
                "target".to_string(),
                Value::String(self.config.table.clone()),
            ),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            ("rows".to_string(), Value::from(self.rows)),
            (
                "segment_id".to_string(),
                Value::String(PREPARED_SEGMENT_ID.into()),
            ),
            (
                "segment_bytes".to_string(),
                Value::from(u64::try_from(prepared_rows.len()).unwrap_or(u64::MAX)),
            ),
            (
                "segment_sha256".to_string(),
                Value::String(hex::encode(Sha256::digest(&prepared_rows))),
            ),
            (
                "columns".to_string(),
                Value::Array(
                    self.pending_columns
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .collect(),
                ),
            ),
            (
                "schema_hash".to_string(),
                Value::String(
                    self.pending_schema_hash
                        .clone()
                        .unwrap_or_else(|| hex::encode(Sha256::digest([]))),
                ),
            ),
        ]))
    }

    // Durable evidence validation is an atomic trust boundary. All identity,
    // epoch, schema, SQL, row, and checksum checks stay together and fail closed.
    // #lizard forgives
    fn validate_evidence(
        &self,
        epoch: calc_flow::Epoch,
        evidence: &JsonMap,
        rows: Vec<Vec<crate::database_types::PgValue>>,
    ) -> Result<PreparedPostgresCommit> {
        let string = |field: &str| {
            evidence
                .get(field)
                .and_then(Value::as_str)
                .map(str::to_string)
                .ok_or_else(|| fail("commit", &format!("pre-commit field {field:?} is missing")))
        };
        if string("pipeline")? != self.config.pipeline
            || string("output")? != self.config.output
            || string("target")? != self.config.table
        {
            return Err(fail(
                "commit",
                "pre-commit evidence names a different sink identity",
            ));
        }
        if evidence.get("epoch").and_then(Value::as_u64) != Some(epoch.as_u64()) {
            return Err(fail(
                "commit",
                "pre-commit evidence names a different epoch",
            ));
        }
        if string("segment_id")? != PREPARED_SEGMENT_ID {
            return Err(fail("commit", "pre-commit segment identity is invalid"));
        }
        let columns = evidence
            .get("columns")
            .and_then(Value::as_array)
            .ok_or_else(|| fail("commit", "pre-commit columns are missing"))?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .ok_or_else(|| fail("commit", "pre-commit column must be a string"))
                    .and_then(pg_identifier)
            })
            .collect::<Result<Vec<_>>>()?;
        let sql = compile_insert_sql(&self.config, &columns)?;
        let schema_hash = string("schema_hash")?;
        if schema_hash.len() != 64 || !schema_hash.bytes().all(|byte| byte.is_ascii_hexdigit()) {
            return Err(fail(
                "commit",
                "pre-commit evidence has an invalid schema hash",
            ));
        }
        if rows.iter().any(|row| row.len() != columns.len()) {
            return Err(fail(
                "commit",
                "pre-commit row width does not match its column list",
            ));
        }
        let prepared_rows = serde_json::to_vec(&evidence_rows(&rows))
            .map_err(|error| fail("commit", &error.to_string()))?;
        if evidence.get("segment_bytes").and_then(Value::as_u64)
            != Some(u64::try_from(prepared_rows.len()).unwrap_or(u64::MAX))
        {
            return Err(fail(
                "commit",
                "pre-commit segment byte count does not match its prepared rows",
            ));
        }
        if string("segment_sha256")? != hex::encode(Sha256::digest(&prepared_rows)) {
            return Err(fail(
                "commit",
                "pre-commit segment checksum does not match its prepared rows",
            ));
        }
        let expected_rows = evidence
            .get("rows")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("commit", "pre-commit row count is missing"))?;
        let actual_rows = u64::try_from(rows.len()).unwrap_or(u64::MAX);
        if expected_rows != actual_rows {
            return Err(fail(
                "commit",
                "pre-commit row count does not match its prepared rows",
            ));
        }
        Ok(PreparedPostgresCommit { sql, rows })
    }

    // Prepared-transaction recovery deliberately handles every idempotent
    // PostgreSQL outcome in one state transition.
    // #lizard forgives
    async fn commit_prepared(
        &mut self,
        epoch: calc_flow::Epoch,
        prepared: &PreparedPostgresCommit,
    ) -> Result<()> {
        let Some(client) = self.client.as_mut() else {
            return Err(fail("commit", "commit before a resolved connection"));
        };
        let epoch_value = i64::try_from(epoch.as_u64())
            .map_err(|_| fail("commit", "epoch exceeds the PostgreSQL ledger BIGINT range"))?;
        let rows_written = i64::try_from(prepared.rows.len())
            .map_err(|_| fail("commit", "row count exceeds the PostgreSQL BIGINT range"))?;
        let tx = client
            .transaction()
            .await
            .map_err(|error| fail("commit", &error.to_string()))?;
        let inserted = tx
            .query_opt(
                &format!(
                    "INSERT INTO {LEDGER_TABLE} (pipeline, output, epoch, rows_written) \
                     VALUES ($1, $2, $3, $4) \
                     ON CONFLICT (pipeline, output, epoch) DO NOTHING \
                     RETURNING rows_written"
                ),
                &[
                    &self.config.pipeline,
                    &self.config.output,
                    &epoch_value,
                    &rows_written,
                ],
            )
            .await
            .map_err(|error| fail("commit", &error.to_string()))?;
        if inserted.is_some() {
            for row_values in &prepared.rows {
                let params = row_values
                    .iter()
                    .map(|value| value as &(dyn ToSql + Sync))
                    .collect::<Vec<_>>();
                tx.execute(&prepared.sql, &params)
                    .await
                    .map_err(|error| fail("commit", &error.to_string()))?;
            }
        } else {
            let recorded: i64 = tx
                .query_one(
                    &format!(
                        "SELECT rows_written FROM {LEDGER_TABLE} \
                         WHERE pipeline = $1 AND output = $2 AND epoch = $3"
                    ),
                    &[&self.config.pipeline, &self.config.output, &epoch_value],
                )
                .await
                .map_err(|error| fail("commit", &error.to_string()))?
                .get(0);
            if recorded != rows_written {
                return Err(fail(
                    "commit",
                    "existing epoch ledger row count conflicts with recovery evidence",
                ));
            }
        }
        tx.commit()
            .await
            .map_err(|error| fail("commit", &error.to_string()))
    }
}

struct PreparedPostgresCommit {
    sql: String,
    rows: Vec<Vec<crate::database_types::PgValue>>,
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
        self.pending_columns.clear();
        self.pending_schema_hash = None;
        self.active = Some(epoch);
        self.rows = 0;
        self.pending_bytes = 2;
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
        let names = schema
            .fields()
            .iter()
            .map(|field| pg_identifier(field.name()))
            .collect::<Result<Vec<_>>>()?;
        let sql = compile_insert_sql(&self.config, &names)?;
        let schema_hash = sink_schema_hash(schema.as_ref());
        if self
            .pending_sql
            .as_ref()
            .is_some_and(|expected| expected != &sql)
            || self
                .pending_schema_hash
                .as_ref()
                .is_some_and(|expected| expected != &schema_hash)
        {
            return Err(fail(
                "write",
                "all batches in one epoch must use the same Arrow schema",
            ));
        }
        self.pending_sql = Some(sql);
        self.pending_columns.clone_from(&names);
        self.pending_schema_hash = Some(schema_hash);
        for record in payload.batches() {
            for row in 0..record.num_rows() {
                let params = (0..names.len())
                    .map(|column| crate::database_types::cell_value(record.column(column), row))
                    .collect::<Result<Vec<_>>>()?;
                let row_bytes = serde_json::to_vec(&Value::Array(
                    params
                        .iter()
                        .map(crate::database_types::PgValue::to_evidence)
                        .collect(),
                ))
                .map_err(|error| fail("write", &error.to_string()))?;
                let next_rows = self
                    .rows
                    .checked_add(1)
                    .ok_or_else(|| fail("write", "epoch row count exhausted"))?;
                let separator = u64::from(self.rows != 0);
                let next_bytes = self
                    .pending_bytes
                    .checked_add(u64::try_from(row_bytes.len()).unwrap_or(u64::MAX))
                    .and_then(|bytes| bytes.checked_add(separator))
                    .ok_or_else(|| fail("write", "epoch byte count exhausted"))?;
                if next_rows > self.config.max_epoch_rows
                    || next_bytes > self.config.max_epoch_bytes
                {
                    return Err(fail("write", "epoch rows exceed configured bounds"));
                }
                self.pending_rows.push(params);
                self.rows = next_rows;
                self.pending_bytes = next_bytes;
            }
        }
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if self.active != Some(epoch) {
            return Err(fail("pre_commit", "pre_commit names an inactive epoch"));
        }
        self.prepared_evidence(epoch)
    }

    async fn pre_commit_segments(
        &mut self,
        epoch: calc_flow::Epoch,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        if self.active != Some(epoch) {
            return Err(fail(
                "pre_commit",
                "pre_commit segments name an inactive epoch",
            ));
        }
        let bytes = serde_json::to_vec(&evidence_rows(&self.pending_rows))
            .map_err(|error| fail("pre_commit", &error.to_string()))?;
        Ok(BTreeMap::from([(PREPARED_SEGMENT_ID.into(), bytes)]))
    }

    async fn commit(&mut self, epoch: calc_flow::Epoch, pre_commit: &JsonMap) -> Result<()> {
        let prepared = self.validate_evidence(epoch, pre_commit, self.pending_rows.clone())?;
        self.commit_prepared(epoch, &prepared).await?;
        self.pending_rows.clear();
        self.pending_sql = None;
        self.pending_columns.clear();
        self.pending_schema_hash = None;
        self.active = None;
        self.rows = 0;
        self.pending_bytes = 2;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        self.pending_rows.clear();
        self.pending_sql = None;
        self.pending_columns.clear();
        self.pending_schema_hash = None;
        self.active = None;
        self.rows = 0;
        self.pending_bytes = 2;
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        let bytes = recovery
            .segments()
            .get(PREPARED_SEGMENT_ID)
            .ok_or_else(|| fail("recover", "prepared rows segment is missing"))?;
        let value: Value = serde_json::from_slice(bytes)
            .map_err(|_| fail("recover", "prepared rows segment is invalid JSON"))?;
        let width = recovery
            .pre_commit()
            .get("columns")
            .and_then(Value::as_array)
            .map(Vec::len)
            .ok_or_else(|| fail("recover", "pre-commit columns are missing"))?;
        let rows = decode_evidence_rows(&value, width)?;
        let prepared = self.validate_evidence(recovery.epoch(), recovery.pre_commit(), rows)?;
        self.commit_prepared(recovery.epoch(), &prepared).await
    }

    async fn close(&mut self) -> Result<()> {
        settle_connection(&mut self.client, &mut self.connection_driver).await
    }
}

/// Trusted source factory for the `PostgreSQL` transport.
pub struct PostgresSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl PostgresSourceFactory {
    /// Creates the source factory.
    pub fn new() -> Self {
        Self {
            descriptor: postgresql_connector_descriptor(),
        }
    }
}

impl Default for PostgresSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSourceFactory for PostgresSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    fn validate(&self, options: &JsonMap) -> Result<()> {
        PostgresSourceConfig::from_options(options).map(drop)
    }

    fn capabilities(&self, options: &JsonMap) -> Result<ConnectorCapabilities> {
        let config = PostgresSourceConfig::from_options(options)?;
        let mut capabilities = self.descriptor.capabilities;
        match config.mode {
            PgSourceMode::Snapshot => {
                capabilities.delivery = DeliveryCapability::BestEffort;
                capabilities.replay = calc_flow::ReplayCapability::Unreplayable;
            }
            PgSourceMode::IncrementalQuery | PgSourceMode::LogicalCdc => {
                capabilities.delivery = DeliveryCapability::AtLeastOnce;
                capabilities.replay = calc_flow::ReplayCapability::ReplayableExact;
            }
        }
        Ok(capabilities)
    }

    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = PostgresSourceConfig::from_options(options)?;
        let endpoint_url = resolve_connection_url(secrets, "url")?;
        if config.mode == PgSourceMode::LogicalCdc {
            Ok(Box::new(crate::postgresql_cdc::PostgresCdcSource::new(
                config,
                endpoint_url,
            )?))
        } else {
            Ok(Box::new(
                PostgresSource::new(config)?.with_endpoint(endpoint_url),
            ))
        }
    }
}

/// Trusted sink factory for the `PostgreSQL` transport.
pub struct PostgresSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl PostgresSinkFactory {
    /// Creates the sink factory.
    pub fn new() -> Self {
        Self {
            descriptor: postgresql_connector_descriptor(),
        }
    }
}

impl Default for PostgresSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSinkFactory for PostgresSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    fn validate(&self, options: &JsonMap) -> Result<()> {
        PostgresSinkConfig::from_options(options).map(drop)
    }

    fn capabilities(&self, options: &JsonMap) -> Result<ConnectorCapabilities> {
        let config = PostgresSinkConfig::from_options(options)?;
        let mut capabilities = self.descriptor.capabilities;
        if config.mode != PgSinkMode::Transactional {
            capabilities.transaction = TransactionSupport::None;
        }
        Ok(capabilities)
    }

    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        let config = PostgresSinkConfig::from_options(options)?;
        let endpoint_url = resolve_connection_url(secrets, "url")?;
        Ok(Box::new(
            PostgresSink::new(config)?.with_endpoint(endpoint_url),
        ))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = PostgresSinkConfig::from_options(options)?;
        if config.mode != PgSinkMode::Transactional {
            return Ok(None);
        }
        let endpoint_url = resolve_connection_url(secrets, "url")?;
        Ok(Some(Box::new(
            TransactionalPostgresSink::new(config)?.with_endpoint(endpoint_url),
        )))
    }
}

fn postgresql_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: connector_identity(),
        kind: ConnectorKind::Both,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: calc_flow::ReplayCapability::Unreplayable,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::LedgerIdempotent,
            snapshot: true,
            polling: true,
            cdc: true,
            lookup: false,
        },
        formats: Vec::new(),
        config_schema: JsonMap::from([
            ("table".to_string(), serde_json::json!("string")),
            ("mode".to_string(), serde_json::json!("string")),
            ("cursor_columns".to_string(), serde_json::json!("array")),
            ("columns".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("max_transaction_rows".to_string(), serde_json::json!("u64")),
            (
                "max_transaction_bytes".to_string(),
                serde_json::json!("u64"),
            ),
            ("poll_interval_ms".to_string(), serde_json::json!("u64")),
            ("conflict_columns".to_string(), serde_json::json!("array")),
            ("pipeline".to_string(), serde_json::json!("string")),
            ("output".to_string(), serde_json::json!("string")),
            ("max_epoch_rows".to_string(), serde_json::json!("u64")),
            ("max_epoch_bytes".to_string(), serde_json::json!("u64")),
            ("slot".to_string(), serde_json::json!("string")),
            ("publication".to_string(), serde_json::json!("string")),
            ("slot_policy".to_string(), serde_json::json!("string")),
            ("require_before".to_string(), serde_json::json!("bool")),
        ]),
        secret_slots: ["url".to_string()].into_iter().collect(),
        required_secret_slots: ["url".to_string()].into_iter().collect(),
    }
}

/// Registers both `PostgreSQL` connector directions.
///
/// # Errors
///
/// Returns the registry conflict error when the connector slot is occupied.
pub fn register_postgresql_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        postgresql_connector_descriptor(),
        ConnectorFactories::both(
            std::sync::Arc::new(PostgresSourceFactory::new()),
            std::sync::Arc::new(PostgresSinkFactory::new()),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_options(mode: &str) -> JsonMap {
        BTreeMap::from([
            ("table".into(), Value::String("orders".into())),
            ("mode".into(), Value::String(mode.into())),
        ])
    }

    fn sink_options(mode: &str) -> JsonMap {
        BTreeMap::from([
            ("table".into(), Value::String("orders".into())),
            ("mode".into(), Value::String(mode.into())),
            ("pipeline".into(), Value::String("pipeline".into())),
            ("output".into(), Value::String("output".into())),
        ])
    }

    #[test]
    fn source_configuration_rejects_ambiguous_modes_and_unbounded_values() {
        let mut candidate = source_options("incremental_query");
        assert!(PostgresSourceConfig::from_options(&candidate).is_err());

        candidate.insert("cursor_columns".into(), serde_json::json!(["id"]));
        candidate.insert(
            "columns".into(),
            serde_json::json!([{"name": "label", "data_type": "string", "nullable": false}]),
        );
        assert!(PostgresSourceConfig::from_options(&candidate).is_err());

        for field in [
            "max_batch_rows",
            "max_batch_bytes",
            "max_transaction_rows",
            "max_transaction_bytes",
            "poll_interval_ms",
        ] {
            let mut candidate = source_options("snapshot");
            candidate.insert(field.into(), Value::from(0));
            assert!(
                PostgresSourceConfig::from_options(&candidate).is_err(),
                "{field}"
            );
        }

        candidate = source_options("snapshot");
        candidate.insert("url_key".into(), Value::String("legacy".into()));
        assert!(PostgresSourceConfig::from_options(&candidate).is_err());
    }

    #[test]
    fn logical_cdc_configuration_accepts_explicit_frozen_contract() {
        let mut candidate = source_options("logical_cdc");
        candidate.extend([
            ("slot".into(), Value::String("orders_slot".into())),
            (
                "publication".into(),
                Value::String("orders_publication".into()),
            ),
            (
                "slot_policy".into(),
                Value::String("require_existing".into()),
            ),
            (
                "columns".into(),
                serde_json::json!([{"name": "id", "data_type": "int64", "nullable": false}]),
            ),
            ("require_before".into(), Value::Bool(true)),
        ]);
        let config = PostgresSourceConfig::from_options(&candidate).unwrap();
        assert_eq!(config.mode, PgSourceMode::LogicalCdc);
        assert_eq!(config.slot.as_deref(), Some("orders_slot"));
        assert_eq!(config.slot_policy, Some(PgSlotPolicy::RequireExisting));
        assert!(config.require_before);
    }

    #[test]
    fn parameterized_insert_sql_covers_append_and_upsert_shapes() {
        let append = PostgresSinkConfig::from_options(&sink_options("append")).unwrap();
        assert_eq!(
            compile_insert_sql(&append, &["id".into(), "label".into()]).unwrap(),
            "INSERT INTO orders (id, label) VALUES ($1, $2)"
        );

        let mut candidate = sink_options("upsert");
        candidate.insert("conflict_columns".into(), serde_json::json!(["id"]));
        let upsert = PostgresSinkConfig::from_options(&candidate).unwrap();
        assert_eq!(
            compile_insert_sql(&upsert, &["id".into(), "label".into()]).unwrap(),
            "INSERT INTO orders (id, label) VALUES ($1, $2) ON CONFLICT (id) DO UPDATE SET label = EXCLUDED.label"
        );
        assert_eq!(
            compile_insert_sql(&upsert, &["id".into()]).unwrap(),
            "INSERT INTO orders (id) VALUES ($1) ON CONFLICT (id) DO NOTHING"
        );
    }

    #[test]
    fn transactional_evidence_round_trips_and_detects_tampering_offline() {
        let config = PostgresSinkConfig::from_options(&sink_options("transactional")).unwrap();
        let mut sink = TransactionalPostgresSink::new(config).unwrap();
        sink.pending_rows = vec![vec![crate::database_types::PgValue::Int64(7)]];
        sink.pending_columns = vec!["id".into()];
        sink.pending_schema_hash = Some("a".repeat(64));
        sink.rows = 1;
        let epoch = calc_flow::Epoch::INITIAL;
        let evidence = sink.prepared_evidence(epoch).unwrap();
        let prepared = sink
            .validate_evidence(epoch, &evidence, sink.pending_rows.clone())
            .unwrap();
        assert_eq!(prepared.rows.len(), 1);
        assert_eq!(prepared.sql, "INSERT INTO orders (id) VALUES ($1)");

        let encoded = evidence_rows(&sink.pending_rows);
        assert_eq!(
            decode_evidence_rows(&encoded, 1).unwrap(),
            sink.pending_rows
        );
        assert!(decode_evidence_rows(&encoded, 2).is_err());

        let mut tampered = evidence;
        tampered.insert("rows".into(), Value::from(2));
        assert!(
            sink.validate_evidence(epoch, &tampered, sink.pending_rows.clone())
                .is_err()
        );
    }

    #[test]
    fn reviewed_cursor_type_matrix_is_explicit() {
        use tokio_postgres::types::Type;

        let supported = [
            Type::BOOL,
            Type::INT2,
            Type::INT4,
            Type::INT8,
            Type::FLOAT4,
            Type::FLOAT8,
            Type::TEXT,
            Type::TIMESTAMP,
            Type::TIMESTAMPTZ,
            Type::DATE,
            Type::UUID,
        ];
        for data_type in supported {
            assert!(cursor_type_supported(&data_type), "{data_type}");
            assert_ne!(cursor_sql_type(&data_type), "");
        }
        assert!(!cursor_type_supported(&Type::BYTEA));
        assert_eq!(parse_pg_type("future"), Type::TEXT);
    }

    #[test]
    fn composite_cursor_uses_lexicographic_row_comparison() {
        let config = PostgresSourceConfig::from_options(&BTreeMap::from([
            ("table".into(), serde_json::json!("orders")),
            ("mode".into(), serde_json::json!("incremental_query")),
            (
                "cursor_columns".into(),
                serde_json::json!(["updated_at", "id"]),
            ),
        ]))
        .unwrap();
        let mut source = PostgresSource::new(config).unwrap();
        source.columns = vec![
            PgColumn {
                name: "updated_at".into(),
                data_type: tokio_postgres::types::Type::TIMESTAMPTZ,
                nullable: false,
            },
            PgColumn {
                name: "id".into(),
                data_type: tokio_postgres::types::Type::INT8,
                nullable: false,
            },
        ];
        source.cursor_values = vec!["2026-08-19T00:00:00Z".into(), "42".into()];

        let sql = source.build_query();
        assert!(
            sql.contains(
                "(updated_at, id) > (CAST($1 AS text)::timestamptz, CAST($2 AS text)::bigint)"
            ),
            "{sql}"
        );
        assert!(!sql.contains("updated_at > $1 AND id > $2"), "{sql}");
    }

    #[test]
    fn snapshot_query_advances_by_the_checkpointed_offset() {
        let config = PostgresSourceConfig::from_options(&BTreeMap::from([
            ("table".into(), serde_json::json!("orders")),
            ("mode".into(), serde_json::json!("snapshot")),
        ]))
        .unwrap();
        let mut source = PostgresSource::new(config).unwrap();
        source.snapshot_offset = 16;
        assert!(source.build_query().ends_with("LIMIT 8192 OFFSET 16"));
        assert_eq!(
            source.cursor_from_values().unwrap().payload()["snapshot_offset"],
            serde_json::json!(16)
        );
    }

    #[tokio::test]
    async fn connection_driver_is_joined_and_removed() {
        let mut client = None;
        let mut driver = Some(tokio::spawn(std::future::pending()));

        tokio::time::timeout(
            std::time::Duration::from_millis(100),
            settle_connection(&mut client, &mut driver),
        )
        .await
        .expect("close must cancel a connection driver that cannot finish itself")
        .expect("a clean driver settles");

        assert!(driver.is_none(), "the owner must not retain a joined task");
    }
}
