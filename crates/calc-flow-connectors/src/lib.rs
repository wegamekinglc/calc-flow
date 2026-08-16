//! Connector implementations for the Calc-Flow continuous runtime.
//!
//! This crate owns the transport and codec glue; contracts, validation,
//! and the trusted registry live in the `calc-flow` core crate. Every
//! connector registers through [`calc_flow::ConnectorRegistry`] and
//! implements the A6-public source and sink lifecycles, so plans built
//! from data-only documents keep resolving factories through trusted
//! process-local code.
//!
//! Feature gates follow the frozen M6 decision: lightweight pure-Rust
//! format codecs (CSV, newline JSON) always compile, while the Parquet
//! codec and the file transport compile behind the default `file`
//! feature.
//!
//! # Example
//!
//! ```
//! use calc_flow::ConnectorRegistry;
//! use calc_flow_connectors::register_file_connectors;
//!
//! let mut registry = ConnectorRegistry::new();
//! register_file_connectors(&mut registry).expect("file connectors register");
//! assert_eq!(
//!     registry
//!         .snapshot()
//!         .identities()
//!         .iter()
//!         .map(|identity| identity.name.to_string())
//!         .collect::<Vec<String>>(),
//!     vec!["file"]
//! );
//! ```

pub mod arrow_schema;
pub mod csv;
pub mod json_lines;
#[cfg(feature = "file")]
pub mod parquet;

#[cfg(feature = "file")]
mod file;
#[cfg(feature = "file")]
mod file_sink;
#[cfg(feature = "kafka")]
pub mod kafka;

#[cfg(feature = "file")]
pub use file::{FileFormat, FileSource, FileSourceConfig};
#[cfg(feature = "file")]
pub use file_sink::{FileSinkConfig, TransactionalParquetSink};

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use calc_flow::{
    Batch, ConnectorCapabilities, ConnectorDescriptor, ConnectorFactories, ConnectorIdentity,
    ConnectorKind, ConnectorRegistry, ConnectorRegistrySnapshot, ConnectorSinkFactory,
    ConnectorSourceFactory, DeliveryCapability, Epoch, FormatDescriptor, FormatIdentity, JsonMap,
    Result, SecretResolver, StreamSink, StreamSource, TransactionSupport, TransactionalStreamSink,
    WatermarkSupport,
};

/// The connector identity of the built-in file transport (feature
/// `file`).
pub const FILE_CONNECTOR_VERSION: &str = "2.0.0";

fn file_connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "file", FILE_CONNECTOR_VERSION)
        .expect("the file connector identity is valid")
}

fn file_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: file_connector_identity(),
        kind: ConnectorKind::Both,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: calc_flow::ReplayCapability::ReplayableExact,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::PreCommitCommit,
            snapshot: true,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: vec![
            FormatIdentity::new(csv::IDENTITY, csv::IDENTITY_VERSION).expect("csv identity"),
            FormatIdentity::new(json_lines::IDENTITY, json_lines::IDENTITY_VERSION)
                .expect("json identity"),
            #[cfg(feature = "file")]
            FormatIdentity::new(parquet::IDENTITY, parquet::IDENTITY_VERSION)
                .expect("parquet identity"),
        ],
        config_schema: JsonMap::from([
            ("path".to_string(), serde_json::json!("string")),
            ("format".to_string(), serde_json::json!("string")),
            ("header".to_string(), serde_json::json!("boolean")),
            ("schema".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("max_file_bytes".to_string(), serde_json::json!("u64")),
            // Sink-side option naming the output directory under `path`.
            ("output".to_string(), serde_json::json!("string")),
        ]),
        secret_slots: BTreeSet::new(),
    }
}

/// Trusted source factory for the file transport (feature `file`).
#[cfg(feature = "file")]
pub struct FileSourceFactory {
    descriptor: ConnectorDescriptor,
}

#[cfg(feature = "file")]
impl FileSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: file_connector_descriptor(),
        }
    }
}

#[cfg(feature = "file")]
impl Default for FileSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "file")]
#[async_trait]
impl ConnectorSourceFactory for FileSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = FileSourceConfig::from_options(options)?;
        Ok(Box::new(FileSource::new(config)?))
    }
}

/// Trusted sink factory for the file transport (feature `file`).
#[cfg(feature = "file")]
pub struct FileSinkFactory {
    descriptor: ConnectorDescriptor,
}

#[cfg(feature = "file")]
impl FileSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: file_connector_descriptor(),
        }
    }
}

#[cfg(feature = "file")]
impl Default for FileSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

/// Ordinary (at-least-once) file sink writing Parquet parts directly.
///
/// The exactly-once path uses [`TransactionalParquetSink`]; the ordinary
/// sink exists for plans whose delivery requirement stops at
/// at-least-once.
#[cfg(feature = "file")]
pub struct OrdinaryParquetSink {
    inner: TransactionalParquetSink,
    epoch: Epoch,
}

#[cfg(feature = "file")]
impl OrdinaryParquetSink {
    /// Builds the sink over one auto-committing epoch directory.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: FileSinkConfig, epoch: Epoch) -> Result<Self> {
        Ok(Self {
            inner: TransactionalParquetSink::new(config)?,
            epoch,
        })
    }
}

#[cfg(feature = "file")]
#[async_trait]
impl StreamSink for OrdinaryParquetSink {
    async fn open(&mut self) -> Result<()> {
        self.inner.open().await?;
        self.inner.begin_epoch(self.epoch).await
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        self.inner.write(batch).await
    }

    async fn close(&mut self) -> Result<()> {
        let evidence = self.inner.pre_commit(self.epoch).await?;
        self.inner.commit(self.epoch, &evidence).await?;
        self.inner.close().await
    }
}

#[cfg(feature = "file")]
#[async_trait]
impl ConnectorSinkFactory for FileSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        let config = FileSinkConfig::from_options(options)?;
        Ok(Box::new(OrdinaryParquetSink::new(config, Epoch::INITIAL)?))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = FileSinkConfig::from_options(options)?;
        Ok(Some(Box::new(TransactionalParquetSink::new(config)?)))
    }
}

/// The connector identity of the Kafka transport (feature `kafka`).
#[cfg(feature = "kafka")]
pub const KAFKA_CONNECTOR_VERSION: &str = kafka::IDENTITY_VERSION;

/// Trusted source factory for the Kafka transport (feature `kafka`).
#[cfg(feature = "kafka")]
pub struct KafkaSourceFactory {
    descriptor: ConnectorDescriptor,
}

#[cfg(feature = "kafka")]
impl KafkaSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: kafka_connector_descriptor(),
        }
    }
}

#[cfg(feature = "kafka")]
impl Default for KafkaSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "kafka")]
#[async_trait]
impl ConnectorSourceFactory for KafkaSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = kafka::KafkaSourceConfig::from_options(options)?;
        Ok(Box::new(kafka::KafkaSource::new(config)?))
    }
}

/// Trusted sink factory for the Kafka transport (feature `kafka`).
#[cfg(feature = "kafka")]
pub struct KafkaSinkFactory {
    descriptor: ConnectorDescriptor,
}

#[cfg(feature = "kafka")]
impl KafkaSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: kafka_connector_descriptor(),
        }
    }
}

#[cfg(feature = "kafka")]
impl Default for KafkaSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "kafka")]
#[async_trait]
impl ConnectorSinkFactory for KafkaSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        let config = kafka::KafkaSinkConfig::from_options(options)?;
        Ok(Box::new(kafka::OrdinaryKafkaSink::new(config)?))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = kafka::KafkaSinkConfig::from_options(options)?;
        Ok(Some(Box::new(kafka::TransactionalKafkaSink::new(config)?)))
    }
}

#[cfg(feature = "kafka")]
fn kafka_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: ConnectorIdentity::new("calc-flow-connectors", "kafka", kafka::IDENTITY_VERSION)
            .expect("the kafka connector identity is valid"),
        kind: ConnectorKind::Both,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: calc_flow::ReplayCapability::ReplayableExact,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::PreCommitCommit,
            snapshot: false,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: vec![
            FormatIdentity::new(json_lines::IDENTITY, json_lines::IDENTITY_VERSION)
                .expect("json identity"),
            FormatIdentity::new(csv::IDENTITY, csv::IDENTITY_VERSION).expect("csv identity"),
        ],
        config_schema: JsonMap::from([
            ("bootstrap_servers".to_string(), serde_json::json!("string")),
            ("topic".to_string(), serde_json::json!("string")),
            ("partitions".to_string(), serde_json::json!("array")),
            ("auto_offset_reset".to_string(), serde_json::json!("string")),
            ("format".to_string(), serde_json::json!("string")),
            ("schema".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("transactional_id".to_string(), serde_json::json!("string")),
        ]),
        secret_slots: BTreeSet::new(),
    }
}

/// Registers the Kafka connectors into one trusted registry (feature
/// `kafka`).
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot or format
/// identity is already occupied.
#[cfg(feature = "kafka")]
pub fn register_kafka_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        kafka_connector_descriptor(),
        ConnectorFactories::both(
            Arc::new(KafkaSourceFactory::new()),
            Arc::new(KafkaSinkFactory::new()),
        ),
    )
}

/// Registers the built-in format codecs (always available).
///
/// # Errors
///
/// Returns the registry conflict error when a format identity is already
/// registered.
pub fn register_format_codecs(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_format(FormatDescriptor {
        identity: FormatIdentity::new(csv::IDENTITY, csv::IDENTITY_VERSION)?,
    })?;
    registry.register_format(FormatDescriptor {
        identity: FormatIdentity::new(json_lines::IDENTITY, json_lines::IDENTITY_VERSION)?,
    })?;
    Ok(())
}

/// Registers the built-in file connectors and their format codecs into
/// one trusted registry (feature `file`).
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot or format
/// identity is already occupied.
#[cfg(feature = "file")]
pub fn register_file_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    register_format_codecs(registry)?;
    registry.register_format(FormatDescriptor {
        identity: FormatIdentity::new(parquet::IDENTITY, parquet::IDENTITY_VERSION)?,
    })?;
    registry.register_connector(
        file_connector_descriptor(),
        ConnectorFactories::both(
            Arc::new(FileSourceFactory::new()),
            Arc::new(FileSinkFactory::new()),
        ),
    )
}

/// Resolves the file source factory through one captured snapshot.
///
/// # Errors
///
/// Returns the resolution error when the snapshot has no file connector.
#[cfg(feature = "file")]
pub fn resolve_file_source(
    snapshot: &ConnectorRegistrySnapshot,
) -> Result<Arc<dyn ConnectorSourceFactory>> {
    snapshot.resolve_source(&file_connector_identity())
}
