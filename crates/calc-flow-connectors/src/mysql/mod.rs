//! `MySQL` 8.4 `InnoDB` snapshot, cursor-polling, and transactional connectors.
//!
//! Enable `mysql` and register the factories in a trusted registry. Connection
//! URLs are resolved from the `url` secret slot. TLS verifies certificates by
//! default. See `docs/connectors.md` for cursor assumptions and type mappings.
//!
//! ```
//! use calc_flow::ConnectorRegistry;
//! use calc_flow_connectors::register_mysql_connectors;
//! let mut registry = ConnectorRegistry::new();
//! register_mysql_connectors(&mut registry)?;
//! # Ok::<(), calc_flow::CalcFlowError>(())
//! ```

mod config;
mod sink;
mod source;
mod types;

use async_trait::async_trait;
use calc_flow::{
    CalcFlowError, ConnectorCapabilities, ConnectorDescriptor, ConnectorError, ConnectorFactories,
    ConnectorIdentity, ConnectorKind, ConnectorOperation, ConnectorRegistry, ConnectorSinkFactory,
    ConnectorSourceFactory, DeliveryCapability, JsonMap, ReplayCapability, Result, SecretReference,
    SecretResolver, SecretResolverKind, StreamSink, StreamSource, TransactionSupport,
    TransactionalStreamSink, WatermarkSupport,
};
use config::{ConnectionConfig, SinkConfig, SinkMode, SourceConfig};
use mysql_async::{Conn, Opts, OptsBuilder, SslOpts, prelude::Queryable};
use std::{future::Future, sync::Arc, time::Duration};

/// Version of the `MySQL` connector's data-only identity.
pub const IDENTITY_VERSION: &str = "1.0.0";

fn identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "mysql", IDENTITY_VERSION)
        .expect("static identity")
}

fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        identity(),
        ConnectorOperation::new(operation).expect("static operation"),
        detail,
    ))
}

async fn database<T>(
    operation: &str,
    timeout: Duration,
    future: impl Future<Output = mysql_async::Result<T>>,
) -> Result<T> {
    tokio::time::timeout(timeout, future)
        .await
        .map_err(|_| fail(operation, "MySQL operation timed out"))?
        .map_err(|error| match error {
            mysql_async::Error::Server(server) => fail(
                operation,
                &format!(
                    "MySQL server error {} (SQLSTATE {})",
                    server.code, server.state
                ),
            ),
            _ => fail(
                operation,
                "MySQL transport, authentication, or value conversion failed",
            ),
        })
}

fn endpoint(secrets: &dyn SecretResolver) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Registered, "url")?;
    let secret = secrets
        .resolve(&reference)
        .map_err(|_| fail("open", "the url secret could not be resolved"))?;
    String::from_utf8(secret.expose().to_vec())
        .map_err(|_| fail("open", "the url secret must be UTF-8"))
}

async fn connect(url: &str, config: &ConnectionConfig, bytes: u64) -> Result<Conn> {
    // Feature unification may enable both rustls providers. Respect an
    // embedding application's provider, and select ring when none is set.
    let _ = rustls::crypto::ring::default_provider().install_default();
    let opts = Opts::from_url(url).map_err(|_| fail("open", "invalid MySQL connection URL"))?;
    if opts.db_name().is_none_or(str::is_empty) {
        return Err(fail("open", "the URL must select a database"));
    }
    let opts = OptsBuilder::from_opts(opts)
        .prefer_socket(false).socket(None::<String>).secure_auth(true)
        .ssl_opts(config.tls.then(SslOpts::default))
        .max_allowed_packet(Some(usize::try_from(bytes).unwrap_or(usize::MAX)))
        .init(vec!["SET time_zone = '+00:00'", "SET SESSION sql_mode = 'STRICT_ALL_TABLES,NO_ZERO_DATE,NO_ZERO_IN_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION'"]);
    database("open", config.timeout, Conn::new(opts)).await
}

async fn require_innodb(conn: &mut Conn, table: &str, timeout: Duration) -> Result<()> {
    let engine: Option<String> = database("open", timeout, conn.exec_first(
        "SELECT ENGINE FROM information_schema.TABLES WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = ?", (table,)
    )).await?;
    if engine.as_deref() != Some("InnoDB") {
        return Err(fail(
            "open",
            "the target must be an existing InnoDB base table",
        ));
    }
    Ok(())
}

fn descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: identity(),
        kind: ConnectorKind::Both,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: ReplayCapability::Unreplayable,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::LedgerIdempotent,
            snapshot: true,
            polling: true,
            cdc: false,
            lookup: false,
        },
        formats: Vec::new(),
        config_schema: [
            ("table", "string"),
            ("mode", "string"),
            ("columns", "array"),
            ("cursor_columns", "array"),
            ("assume_monotonic_cursor", "boolean"),
            ("max_batch_rows", "u64"),
            ("max_batch_bytes", "u64"),
            ("poll_interval_ms", "u64"),
            ("tls", "boolean"),
            ("timeout_seconds", "u64"),
            ("pipeline", "string"),
            ("output", "string"),
            ("max_epoch_rows", "u64"),
            ("max_epoch_bytes", "u64"),
        ]
        .into_iter()
        .map(|(key, value)| (key.into(), serde_json::json!(value)))
        .collect(),
        secret_slots: ["url".into()].into_iter().collect(),
        required_secret_slots: ["url".into()].into_iter().collect(),
    }
}

/// Trusted `MySQL` snapshot and incremental-query source factory.
pub struct MySqlSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl MySqlSourceFactory {
    /// Creates a factory without opening a connection.
    pub fn new() -> Self {
        Self {
            descriptor: descriptor(),
        }
    }
}
impl Default for MySqlSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSourceFactory for MySqlSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }
    fn validate(&self, options: &JsonMap) -> Result<()> {
        SourceConfig::parse(options).map(drop)
    }
    fn capabilities(&self, options: &JsonMap) -> Result<ConnectorCapabilities> {
        let config = SourceConfig::parse(options)?;
        let mut caps = self.descriptor.capabilities;
        caps.transaction = TransactionSupport::None;
        if config.incremental {
            caps.replay = ReplayCapability::ReplayableExact;
        } else {
            caps.delivery = DeliveryCapability::BestEffort;
        }
        Ok(caps)
    }
    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        Ok(Box::new(source::MySqlSource::new(
            SourceConfig::parse(options)?,
            endpoint(secrets)?,
        )?))
    }
}

/// Trusted `MySQL` append, upsert, and epoch-ledger sink factory.
pub struct MySqlSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl MySqlSinkFactory {
    /// Creates a factory without opening a connection.
    pub fn new() -> Self {
        Self {
            descriptor: descriptor(),
        }
    }
}
impl Default for MySqlSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSinkFactory for MySqlSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }
    fn validate(&self, options: &JsonMap) -> Result<()> {
        SinkConfig::parse(options).map(drop)
    }
    fn capabilities(&self, options: &JsonMap) -> Result<ConnectorCapabilities> {
        let config = SinkConfig::parse(options)?;
        let mut caps = self.descriptor.capabilities;
        if config.mode != SinkMode::Transactional {
            caps.transaction = TransactionSupport::None;
        }
        Ok(caps)
    }
    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        Ok(Box::new(sink::MySqlSink::new(
            SinkConfig::parse(options)?,
            endpoint(secrets)?,
        )))
    }
    async fn open_transactional(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = SinkConfig::parse(options)?;
        if config.mode != SinkMode::Transactional {
            return Ok(None);
        }
        Ok(Some(Box::new(sink::MySqlSink::new(
            config,
            endpoint(secrets)?,
        ))))
    }
}

/// Registers both `MySQL` directions in the trusted process-local registry.
///
/// # Errors
/// Returns a registry conflict when this identity is already registered.
pub fn register_mysql_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        descriptor(),
        ConnectorFactories::both(
            Arc::new(MySqlSourceFactory::new()),
            Arc::new(MySqlSinkFactory::new()),
        ),
    )
}
