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
        let bootstrap_servers = required_string(options, "bootstrap_servers")?;
        let topic = required_string(options, "topic")?;
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
        let auto_offset_reset = match options.get("auto_offset_reset").and_then(Value::as_str) {
            None | Some("earliest") => KafkaOffsetReset::Earliest,
            Some("latest") => KafkaOffsetReset::Latest,
            Some(other) => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "auto_offset_reset".into(),
                    message: format!("unsupported reset offset {other:?}"),
                });
            }
        };
        let format = KafkaFormat::parse(&required_string(options, "format")?)?;
        let schema =
            match options.get("schema") {
                None => Vec::new(),
                Some(value) => serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone())
                    .map_err(|error| CalcFlowError::InvalidArgument {
                        field: "schema".into(),
                        message: format!("schema must be a field list: {error}"),
                    })?,
            };
        Ok(Self {
            bootstrap_servers,
            topic,
            partitions: sorted,
            auto_offset_reset,
            format,
            schema,
            max_batch_rows: u64_option(options, "max_batch_rows")?.unwrap_or(8192),
            max_batch_bytes: u64_option(options, "max_batch_bytes")?.unwrap_or(8 * 1024 * 1024),
        })
    }

    fn decode(&self, payload: &[u8]) -> Result<Batch> {
        let bounds = DecodeBounds::new(self.max_batch_rows, self.max_batch_bytes)?;
        match self.format {
            KafkaFormat::Json => JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?.decode(
                payload,
                &bounds,
                &self.schema,
            ),
            KafkaFormat::Csv => CsvCodec::new(crate::csv::IDENTITY_VERSION, true)?.decode(
                payload,
                &bounds,
                &self.schema,
            ),
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
            KafkaFormat::Json => {
                JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?.encode(batch)
            }
            KafkaFormat::Csv => CsvCodec::new(crate::csv::IDENTITY_VERSION, true)?.encode(batch),
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
        let _ = recovery;
        Ok(())
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
                JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?.encode(batch)?
            }
            KafkaFormat::Csv => CsvCodec::new(crate::csv::IDENTITY_VERSION, true)?.encode(batch)?,
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
