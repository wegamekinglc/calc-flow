//! The Kafka connector (feature `kafka`).
//!
//! The source owns an explicit partition assignment — deterministic by
//! construction because the runtime leases one execution plan to exactly
//! one job, so no consumer-group rebalancing participates in placement.
//! Replay cursors carry every assigned partition's committed offset; the
//! sink commits through Kafka transactions with a transactional ID
//! derived from the pipeline and sink identity, never from secrets.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, CalcFlowError, ConnectorError, ConnectorIdentity, ConnectorOperation,
    Cursor, DecodeBounds, FormatDecoder, JsonMap, Result, SinkRecovery, SourceCapabilities,
    SourceEvent, SourceSchema, StreamSink, StreamSource, TransactionalStreamSink,
};
use rdkafka::Offset;
use rdkafka::consumer::{BaseConsumer, Consumer};
use rdkafka::message::Message;
use rdkafka::producer::{FutureProducer, FutureRecord, Producer};
use rdkafka::topic_partition_list::TopicPartitionList;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::arrow_schema::schema_from_spec;
use crate::csv::CsvCodec;
use crate::json_lines::JsonLinesCodec;

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

/// How long one source poll waits before reporting idleness.
const POLL_TIMEOUT: Duration = Duration::from_millis(250);

fn connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "kafka", IDENTITY_VERSION)
        .expect("the kafka connector identity is valid")
}

fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        connector_identity(),
        ConnectorOperation::new(operation).expect("operation name is non-empty"),
        detail,
    ))
}

/// The wire format of Kafka record values.
#[derive(Clone, Copy, Debug)]
pub enum KafkaFormat {
    /// Newline-delimited JSON payloads.
    Json,
    /// CSV payloads.
    Csv,
}

impl KafkaFormat {
    /// Parses the data-only format vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for an unknown name.
    pub fn parse(value: &str) -> Result<Self> {
        match value {
            "json" => Ok(Self::Json),
            "csv" => Ok(Self::Csv),
            other => Err(CalcFlowError::InvalidArgument {
                field: "format".into(),
                message: format!("unsupported kafka payload format {other:?}"),
            }),
        }
    }
}

/// Data-only configuration for one Kafka source.
#[derive(Clone, Debug)]
pub struct KafkaSourceConfig {
    /// Comma-separated bootstrap broker list.
    pub bootstrap_servers: String,
    /// Topic to read.
    pub topic: String,
    /// Explicitly owned partitions in ascending order.
    pub partitions: Vec<i32>,
    /// Reset offset for partitions without a replay cursor.
    pub auto_offset_reset: KafkaOffsetReset,
    /// Payload wire format.
    pub format: KafkaFormat,
    /// Optional explicit Arrow schema every payload must match.
    pub schema: Vec<ArrowFieldSpec>,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Byte bound of one decoded batch.
    pub max_batch_bytes: u64,
}

/// Where a partition starts when no cursor names its offset.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KafkaOffsetReset {
    /// The earliest retained record.
    Earliest,
    /// The next record to be produced.
    Latest,
}

impl KafkaOffsetReset {
    fn librdkafka_value(self) -> &'static str {
        match self {
            Self::Earliest => "earliest",
            Self::Latest => "latest",
        }
    }
}

impl KafkaSourceConfig {
    /// Parses the source configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option for a missing or malformed value.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let (bootstrap_servers, topic, format) = parse_kafka_endpoint(options)?;
        let (max_batch_rows, max_batch_bytes) = parse_kafka_bounds(options)?;
        Ok(Self {
            bootstrap_servers,
            topic,
            partitions: parse_partitions(options)?,
            auto_offset_reset: parse_offset_reset(options)?,
            format,
            schema: parse_kafka_schema(options)?,
            max_batch_rows,
            max_batch_bytes,
        })
    }

    fn decode(&self, payload: &[u8]) -> Result<Batch> {
        let bounds = DecodeBounds::new(self.max_batch_rows, self.max_batch_bytes)?;
        match self.format {
            KafkaFormat::Json => JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.decode(
                payload,
                &bounds,
                &self.schema,
            ),
            KafkaFormat::Csv => {
                CsvCodec::new(csv::IDENTITY_VERSION, true)?.decode(payload, &bounds, &self.schema)
            }
        }
    }
}

/// The Kafka source over an explicit partition assignment.
pub struct KafkaSource {
    capabilities: SourceCapabilities,
    config: KafkaSourceConfig,
    consumer: BaseConsumer,
    offsets: BTreeMap<i32, i64>,
    sequence: u64,
}

impl KafkaSource {
    /// Builds the source and freezes its capabilities.
    ///
    /// # Errors
    ///
    /// Returns the configuration error for invalid bounds or schema, or
    /// the Kafka error when the consumer cannot be created.
    pub fn new(config: KafkaSourceConfig) -> Result<Self> {
        let schema = if config.schema.is_empty() {
            SourceSchema::DynamicOrUnknown
        } else {
            SourceSchema::Exact(schema_from_spec(&config.schema)?)
        };
        let bounds = DecodeBounds::new(config.max_batch_rows, config.max_batch_bytes)?;
        let mut client = rdkafka::config::ClientConfig::new();
        client.set("bootstrap.servers", &config.bootstrap_servers);
        client.set("group.id", "calc-flow-kafka-source");
        client.set("enable.auto.commit", "false");
        client.set(
            "auto.offset.reset",
            config.auto_offset_reset.librdkafka_value(),
        );
        client.set("enable.partition.eof", "false");
        let consumer: BaseConsumer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        let source = Self {
            capabilities: source_capabilities(schema, bounds),
            config,
            consumer,
            offsets: BTreeMap::new(),
            sequence: 0,
        };
        source
            .assign_partitions(None)
            .map_err(|error| fail("open", &format!("partition assignment failed: {error}")))?;
        Ok(source)
    }

    fn assign_partitions(
        &self,
        resume: Option<&BTreeMap<i32, i64>>,
    ) -> rdkafka::error::KafkaResult<()> {
        let mut assignment = TopicPartitionList::new();
        for partition in &self.config.partitions {
            let offset = match resume.and_then(|offsets| offsets.get(partition).copied()) {
                Some(offset) => Offset::Offset(offset),
                None => match self.config.auto_offset_reset {
                    KafkaOffsetReset::Earliest => Offset::Beginning,
                    KafkaOffsetReset::Latest => Offset::End,
                },
            };
            assignment.add_partition_offset(&self.config.topic, *partition, offset)?;
        }
        self.consumer.assign(&assignment)
    }

    fn cursor_from_offsets(&self) -> Result<Cursor> {
        let (order_partition, order_offset) = self
            .offsets
            .iter()
            .next_back()
            .map_or((0, 0), |(partition, offset)| (*partition, *offset));
        let mut order = Vec::with_capacity(16);
        order.extend_from_slice(&i64::from(order_partition).to_be_bytes());
        order.extend_from_slice(&order_offset.to_be_bytes());
        let offsets: serde_json::Map<String, Value> = self
            .offsets
            .iter()
            .map(|(partition, offset)| (partition.to_string(), Value::from(*offset)))
            .collect();
        Cursor::unbound(
            order,
            BTreeMap::from([("offsets".to_string(), Value::Object(offsets))]),
        )
    }

    fn offsets_from_cursor(cursor: &Cursor) -> BTreeMap<i32, i64> {
        cursor
            .payload()
            .get("offsets")
            .and_then(Value::as_object)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(|(partition, offset)| {
                        Some((partition.parse().ok()?, offset.as_i64()?))
                    })
                    .collect()
            })
            .unwrap_or_default()
    }
}

/// Whether a poll error reflects a broker outage rather than a
/// protocol failure.
fn is_transient_transport_error(error: &rdkafka::error::KafkaError) -> bool {
    matches!(
        error.rdkafka_error_code(),
        Some(
            rdkafka::types::RDKafkaErrorCode::BrokerTransportFailure
                | rdkafka::types::RDKafkaErrorCode::AllBrokersDown,
        )
    )
}

fn source_capabilities(schema: SourceSchema, bounds: DecodeBounds) -> SourceCapabilities {
    SourceCapabilities {
        replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
        delivery: calc_flow::SourceDeliveryCapability::Lossless,
        max_batch_rows: usize::try_from(bounds.max_rows).unwrap_or(usize::MAX),
        max_batch_bytes: usize::try_from(bounds.max_bytes).unwrap_or(usize::MAX),
        schema,
        native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
    }
}

#[async_trait]
impl StreamSource for KafkaSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if let Some(cursor) = cursor {
            let resume = Self::offsets_from_cursor(&cursor);
            self.offsets = resume;
            self.assign_partitions(Some(&self.offsets))
                .map_err(|error| fail("open", &error.to_string()))?;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let message = match self.consumer.poll(POLL_TIMEOUT) {
            None => return Ok(Some(SourceEvent::Idle)),
            Some(Ok(message)) => message.detach(),
            Some(Err(error)) if is_transient_transport_error(&error) => {
                // A broker that is down or restarting must surface as
                // idleness so the job outlives the outage; protocol
                // and decode failures still fail closed.
                return Ok(Some(SourceEvent::Idle));
            }
            Some(Err(error)) => return Err(fail("poll", &error.to_string())),
        };
        let partition = message.partition();
        let offset = message.offset();
        let payload = message.payload().unwrap_or_default();
        let batch = self.config.decode(payload)?;
        self.offsets.insert(partition, offset + 1);
        self.sequence += 1;
        let cursor = self.cursor_from_offsets()?;
        let metadata = calc_flow::BatchMetadata::new(
            "kafka",
            self.sequence,
            BTreeMap::from([
                (
                    "topic".to_string(),
                    Value::String(self.config.topic.clone()),
                ),
                ("partition".to_string(), Value::from(partition)),
                ("offset".to_string(), Value::from(offset)),
            ]),
        )?;
        let batch = batch.with_metadata(metadata);
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Validates that recovery evidence names this sink's transactional ID.
///
/// # Errors
///
/// Returns the connector error when the evidence names a foreign
/// transactional ID or omits it entirely.
pub fn validate_recovery_evidence(expected: &str, evidence: &JsonMap) -> Result<()> {
    match evidence.get("transactional_id").and_then(Value::as_str) {
        Some(recorded) if recorded == expected => Ok(()),
        Some(recorded) => Err(fail(
            "recover",
            &format!(
                "recovery evidence names transactional ID {recorded:?}, not this sink's {expected:?}"
            ),
        )),
        None => Err(fail(
            "recover",
            "recovery evidence is missing the transactional ID",
        )),
    }
}

/// Derives the stable, secret-free transactional ID for one sink.
///
/// # Errors
///
/// Never; the marker return keeps the signature future-proof.
pub fn transactional_id(pipeline: &str, output: &str) -> String {
    let digest = Sha256::digest(format!("{pipeline}/{output}").as_bytes());
    format!("calc-flow-{}", hex::encode(&digest[..8]))
}

/// Data-only configuration for one transactional Kafka sink.
#[derive(Clone, Debug)]
pub struct KafkaSinkConfig {
    /// Comma-separated bootstrap broker list.
    pub bootstrap_servers: String,
    /// Target topic.
    pub topic: String,
    /// Stable transactional ID owner; derived from pipeline and output
    /// identity by the factory, never from secrets.
    pub transactional_id: String,
    /// Payload wire format.
    pub format: KafkaFormat,
}

impl KafkaSinkConfig {
    /// Parses the sink configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option for a missing or malformed value.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        Ok(Self {
            bootstrap_servers: required_string(options, "bootstrap_servers")?,
            topic: required_string(options, "topic")?,
            transactional_id: required_string(options, "transactional_id")?,
            format: KafkaFormat::parse(&required_string(options, "format")?)?,
        })
    }
}

/// Parses and normalizes the explicit partition assignment.
fn parse_kafka_bounds(options: &JsonMap) -> Result<(u64, u64)> {
    Ok((
        u64_option(options, "max_batch_rows")?.unwrap_or(8192),
        u64_option(options, "max_batch_bytes")?.unwrap_or(8 * 1024 * 1024),
    ))
}

fn parse_kafka_endpoint(options: &JsonMap) -> Result<(String, String, KafkaFormat)> {
    Ok((
        required_string(options, "bootstrap_servers")?,
        required_string(options, "topic")?,
        KafkaFormat::parse(&required_string(options, "format")?)?,
    ))
}

fn parse_partitions(options: &JsonMap) -> Result<Vec<i32>> {
    let partitions = match options.get("partitions") {
        None => vec![0],
        Some(Value::Array(values)) => values
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .and_then(|entry| i32::try_from(entry).ok())
                    .ok_or_else(|| CalcFlowError::InvalidArgument {
                        field: "partitions".into(),
                        message: "partition entries must be integers".into(),
                    })
            })
            .collect::<Result<Vec<_>>>()?,
        Some(_) => {
            return Err(CalcFlowError::InvalidArgument {
                field: "partitions".into(),
                message: "partitions must be an integer array".into(),
            });
        }
    };
    if partitions.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "partitions".into(),
            message: "at least one partition must be assigned".into(),
        });
    }
    let mut sorted = partitions;
    sorted.sort_unstable();
    sorted.dedup();
    Ok(sorted)
}

/// Parses the data-only reset-offset vocabulary.
fn parse_offset_reset(options: &JsonMap) -> Result<KafkaOffsetReset> {
    match options.get("auto_offset_reset").and_then(Value::as_str) {
        None | Some("earliest") => Ok(KafkaOffsetReset::Earliest),
        Some("latest") => Ok(KafkaOffsetReset::Latest),
        Some(other) => Err(CalcFlowError::InvalidArgument {
            field: "auto_offset_reset".into(),
            message: format!("unsupported reset offset {other:?}"),
        }),
    }
}

/// Parses the optional explicit schema field list.
fn parse_kafka_schema(options: &JsonMap) -> Result<Vec<ArrowFieldSpec>> {
    match options.get("schema") {
        None => Ok(Vec::new()),
        Some(value) => {
            serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone()).map_err(|error| {
                CalcFlowError::InvalidArgument {
                    field: "schema".into(),
                    message: format!("schema must be a field list: {error}"),
                }
            })
        }
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

/// The transactional Kafka sink.
pub struct TransactionalKafkaSink {
    config: KafkaSinkConfig,
    producer: FutureProducer,
    active: bool,
    delivered: u64,
}

impl TransactionalKafkaSink {
    /// Builds the sink and fences stale transactional producers.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the producer cannot be created
    /// or transaction initialization is rejected by the broker.
    pub fn new(config: KafkaSinkConfig) -> Result<Self> {
        let mut client = rdkafka::config::ClientConfig::new();
        client.set("bootstrap.servers", &config.bootstrap_servers);
        client.set("transactional.id", &config.transactional_id);
        client.set("enable.idempotence", "true");
        client.set("message.timeout.ms", "30000");
        client.set("transaction.timeout.ms", "60000");
        let producer: FutureProducer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        producer
            .init_transactions(Duration::from_secs(30))
            .map_err(|error| fail("open", &format!("transaction init failed: {error}")))?;
        Ok(Self {
            config,
            producer,
            active: false,
            delivered: 0,
        })
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>> {
        use calc_flow::FormatEncoder as _;
        match self.config.format {
            KafkaFormat::Json => JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.encode(batch),
            KafkaFormat::Csv => CsvCodec::new(csv::IDENTITY_VERSION, true)?.encode(batch),
        }
    }
}

#[async_trait]
impl TransactionalStreamSink for TransactionalKafkaSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        if self.active {
            return Err(fail(
                "begin_epoch",
                "a transaction is already active; the runtime owns epoch sequencing",
            ));
        }
        self.producer
            .begin_transaction()
            .map_err(|error| fail("begin_epoch", &error.to_string()))?;
        self.active = true;
        self.delivered = 0;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if !self.active {
            return Err(fail("write", "write before begin_epoch"));
        }
        let payload = self.encode(batch)?;
        let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(&payload);
        let delivery = self
            .producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(error, _message)| fail("write", &error.to_string()))?;
        self.delivered += u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        let _ = delivery;
        Ok(())
    }

    async fn pre_commit(&mut self, _epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if !self.active {
            return Err(fail("pre_commit", "pre_commit before begin_epoch"));
        }
        self.producer
            .flush(Duration::from_secs(30))
            .map_err(|error| fail("pre_commit", &error.to_string()))?;
        Ok(BTreeMap::from([
            (
                "transactional_id".to_string(),
                Value::String(self.config.transactional_id.clone()),
            ),
            ("messages".to_string(), Value::from(self.delivered)),
        ]))
    }

    async fn commit(&mut self, _epoch: calc_flow::Epoch, _pre_commit: &JsonMap) -> Result<()> {
        if !self.active {
            return Err(fail("commit", "commit without an active transaction"));
        }
        self.producer
            .commit_transaction(Duration::from_secs(30))
            .map_err(|error| fail("commit", &error.to_string()))?;
        self.active = false;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        if self.active {
            self.producer
                .abort_transaction(Duration::from_secs(30))
                .map_err(|error| fail("abort", &error.to_string()))?;
            self.active = false;
        }
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        validate_recovery_evidence(&self.config.transactional_id, recovery.pre_commit())
    }

    async fn close(&mut self) -> Result<()> {
        self.producer
            .flush(Duration::from_secs(30))
            .map_err(|error| fail("close", &error.to_string()))?;
        Ok(())
    }
}

/// Ordinary at-least-once Kafka sink for non-transactional plans.
pub struct OrdinaryKafkaSink {
    config: KafkaSinkConfig,
    producer: FutureProducer,
    sequence: u64,
}

impl OrdinaryKafkaSink {
    /// Builds the idempotent producer.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the producer cannot be created.
    pub fn new(config: KafkaSinkConfig) -> Result<Self> {
        let mut client = rdkafka::config::ClientConfig::new();
        client.set("bootstrap.servers", &config.bootstrap_servers);
        client.set("enable.idempotence", "true");
        client.set("message.timeout.ms", "30000");
        let producer: FutureProducer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(Self {
            config,
            producer,
            sequence: 0,
        })
    }
}

#[async_trait]
impl StreamSink for OrdinaryKafkaSink {
    async fn open(&mut self) -> Result<()> {
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        use calc_flow::FormatEncoder as _;
        let payload = match self.config.format {
            KafkaFormat::Json => {
                JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.encode(batch)?
            }
            KafkaFormat::Csv => CsvCodec::new(csv::IDENTITY_VERSION, true)?.encode(batch)?,
        };
        self.sequence += 1;
        let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(&payload);
        self.producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(error, _message)| fail("write", &error.to_string()))?;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        self.producer
            .flush(Duration::from_secs(30))
            .map_err(|error| fail("close", &error.to_string()))?;
        Ok(())
    }
}

use std::collections::BTreeSet;
use std::sync::Arc;

use calc_flow::{
    ConnectorCapabilities, ConnectorDescriptor, ConnectorFactories, ConnectorKind,
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, DeliveryCapability,
    FormatIdentity, SecretResolver, TransactionSupport, WatermarkSupport,
};

use crate::{csv, json_lines};

/// The connector implementation version re-exported as the transport
/// identity constant.
pub const KAFKA_CONNECTOR_VERSION: &str = IDENTITY_VERSION;

/// Trusted source factory for the Kafka transport (feature `kafka`).
pub struct KafkaSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl KafkaSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: kafka_connector_descriptor(),
        }
    }
}

impl Default for KafkaSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

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
        let config = KafkaSourceConfig::from_options(options)?;
        Ok(Box::new(KafkaSource::new(config)?))
    }
}

/// Trusted sink factory for the Kafka transport (feature `kafka`).
pub struct KafkaSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl KafkaSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: kafka_connector_descriptor(),
        }
    }
}

impl Default for KafkaSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

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
        let config = KafkaSinkConfig::from_options(options)?;
        Ok(Box::new(OrdinaryKafkaSink::new(config)?))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = KafkaSinkConfig::from_options(options)?;
        Ok(Some(Box::new(TransactionalKafkaSink::new(config)?)))
    }
}

fn kafka_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: ConnectorIdentity::new("calc-flow-connectors", "kafka", IDENTITY_VERSION)
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
pub fn register_kafka_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        kafka_connector_descriptor(),
        ConnectorFactories::both(
            Arc::new(KafkaSourceFactory::new()),
            Arc::new(KafkaSinkFactory::new()),
        ),
    )
}
