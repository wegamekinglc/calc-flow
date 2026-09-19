//! The Kafka connector (feature `kafka`).
//!
//! The source owns an explicit partition assignment — deterministic by
//! construction because the runtime leases one execution plan to exactly
//! one job, so no consumer-group rebalancing participates in placement.
//! Replay cursors carry every assigned partition's committed offset; the
//! sink commits through Kafka transactions with a transactional ID
//! derived from the pipeline and sink identity, never from secrets.

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, CalcFlowError, ConnectorError, ConnectorIdentity, ConnectorOperation,
    Cursor, DecodeBounds, FormatDecoder, JsonMap, Result, SinkRecovery, SourceCapabilities,
    SourceEvent, SourceSchema, StreamSink, StreamSource, TransactionalStreamSink,
};
use rdkafka::Offset;
use rdkafka::admin::{AdminClient, AdminOptions, ResourceSpecifier};
use rdkafka::client::DefaultClientContext;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::Message;
use rdkafka::producer::{FutureProducer, FutureRecord, Producer};
use rdkafka::topic_partition_list::TopicPartitionList;
use serde_json::Value;
use sha2::{Digest as _, Sha256};

use crate::arrow_schema::schema_from_spec;
use crate::csv::CsvCodec;
use crate::json_lines::JsonLinesCodec;
use crate::options::{required_string, u64_option};
use crate::protobuf::{self, ProtobufCodec};

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
    /// One protobuf message per record value, decoded through a
    /// runtime-loaded descriptor set.
    Protobuf,
    /// A trusted decoder registered out-of-band, selected by the
    /// `decoder` option identity.
    Custom,
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
            "protobuf" => Ok(Self::Protobuf),
            "custom" => Ok(Self::Custom),
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
    /// Protobuf descriptor-set path (required for the protobuf format).
    pub descriptor_set: Option<String>,
    /// Fully-qualified protobuf message name (required for the protobuf
    /// format).
    pub message: Option<String>,
    /// Registered decoder identity (required for the custom format).
    pub decoder: Option<FormatIdentity>,
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
        let schema = parse_kafka_schema(options)?;
        let (descriptor_set, message, decoder) = parse_format_companions(options, format, &schema)?;
        Ok(Self {
            bootstrap_servers,
            topic,
            partitions: parse_partitions(options)?,
            auto_offset_reset: parse_offset_reset(options)?,
            format,
            descriptor_set,
            message,
            decoder,
            schema,
            max_batch_rows,
            max_batch_bytes,
        })
    }

    fn decoder(&self, decoders: &KafkaDecoderRegistry) -> Result<KafkaDecoder> {
        match self.format {
            KafkaFormat::Json => Ok(KafkaDecoder::Json(JsonLinesCodec::new(
                json_lines::IDENTITY_VERSION,
            )?)),
            KafkaFormat::Csv => Ok(KafkaDecoder::Csv(CsvCodec::new(
                csv::IDENTITY_VERSION,
                true,
            )?)),
            KafkaFormat::Protobuf => self.protobuf_decoder(),
            KafkaFormat::Custom => {
                let identity =
                    self.decoder
                        .as_ref()
                        .ok_or_else(|| CalcFlowError::InvalidArgument {
                            field: "decoder".into(),
                            message: "custom payloads require a decoder identity".into(),
                        })?;
                Ok(KafkaDecoder::Custom(decoders.resolve(identity)?))
            }
        }
    }

    fn protobuf_decoder(&self) -> Result<KafkaDecoder> {
        let descriptor_set =
            self.descriptor_set
                .as_deref()
                .ok_or_else(|| CalcFlowError::InvalidArgument {
                    field: "descriptor_set".into(),
                    message: "protobuf payloads require a descriptor set path".into(),
                })?;
        let message = self
            .message
            .as_deref()
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "message".into(),
                message: "protobuf payloads require a message name".into(),
            })?;
        Ok(KafkaDecoder::Protobuf(ProtobufCodec::new(
            protobuf::IDENTITY_VERSION,
            Path::new(descriptor_set),
            message,
        )?))
    }

    #[cfg(test)]
    fn decode(&self, payload: &[u8], decoders: &KafkaDecoderRegistry) -> Result<Batch> {
        self.decoder(decoders)?.decode(payload, self)
    }
}

/// The prepared payload decoders, built once when a source opens so the
/// protobuf descriptor pool loads exactly once per job.
enum KafkaDecoder {
    Json(JsonLinesCodec),
    Csv(CsvCodec),
    Protobuf(ProtobufCodec),
    Custom(Arc<dyn FormatDecoder>),
}

impl KafkaDecoder {
    fn decode(&self, payload: &[u8], config: &KafkaSourceConfig) -> Result<Batch> {
        let bounds = DecodeBounds::new(config.max_batch_rows, config.max_batch_bytes)?;
        match self {
            Self::Json(codec) => codec.decode(payload, &bounds, &config.schema),
            Self::Csv(codec) => codec.decode(payload, &bounds, &config.schema),
            Self::Protobuf(codec) => codec.decode(payload, &bounds, &config.schema),
            Self::Custom(decoder) => decoder.decode(payload, &bounds, &config.schema),
        }
    }
}

/// Shared registry of trusted payload decoders for the custom Kafka format.
///
/// Decoders register under their [`FormatDecoder::identity`]; data-only
/// project options then name the identity to select one. The registry is
/// cheap to clone and registrations stay visible through every clone, so
/// a factory snapshotted into a plan still observes decoders registered
/// before the job opens. The built-in `json`, `csv`, and `protobuf`
/// format names are reserved and cannot be shadowed.
#[derive(Clone, Default)]
pub struct KafkaDecoderRegistry {
    decoders: Arc<std::sync::RwLock<BTreeMap<FormatIdentity, Arc<dyn FormatDecoder>>>>,
}

impl std::fmt::Debug for KafkaDecoderRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let decoders = self
            .decoders
            .read()
            .expect("the decoder registry lock is never poisoned by debugging");
        formatter
            .debug_struct("KafkaDecoderRegistry")
            .field("identities", &decoders.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl KafkaDecoderRegistry {
    /// Registers one decoder under its identity.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the identity shadows
    /// a built-in format name, or [`CalcFlowError::Conflict`] when the
    /// identity is already registered.
    ///
    /// # Panics
    ///
    /// Never in practice; the `expect` documents that the registry lock can
    /// only be poisoned by a panic while held, which registration itself
    /// never triggers.
    pub fn register(&self, decoder: Arc<dyn FormatDecoder>) -> Result<()> {
        let identity = decoder.identity();
        if matches!(
            identity.name.as_ref(),
            "json" | "csv" | "protobuf" | "custom"
        ) {
            return Err(CalcFlowError::InvalidArgument {
                field: "decoder".into(),
                message: format!(
                    "decoder name {:?} shadows a built-in kafka payload format",
                    identity.name
                ),
            });
        }
        let mut decoders = self
            .decoders
            .write()
            .expect("the decoder registry lock is never poisoned by registration");
        if decoders.insert(identity.clone(), decoder).is_some() {
            return Err(CalcFlowError::Conflict {
                resource: "kafka decoder".into(),
                key: format!("{}/{}", identity.name, identity.version),
            });
        }
        Ok(())
    }

    /// Resolves one registered decoder by identity.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the identity when
    /// no decoder is registered for it.
    ///
    /// # Panics
    ///
    /// Never in practice; the `expect` documents that the registry lock can
    /// only be poisoned by a panic while held, which resolution itself
    /// never triggers.
    pub fn resolve(&self, identity: &FormatIdentity) -> Result<Arc<dyn FormatDecoder>> {
        self.decoders
            .read()
            .expect("the decoder registry lock is never poisoned by resolution")
            .get(identity)
            .cloned()
            .ok_or_else(|| CalcFlowError::InvalidArgument {
                field: "decoder".into(),
                message: format!(
                    "no kafka decoder registered for {}/{}",
                    identity.name, identity.version
                ),
            })
    }
}

/// The Kafka source over an explicit partition assignment.
pub struct KafkaSource {
    capabilities: SourceCapabilities,
    config: KafkaSourceConfig,
    decoder: KafkaDecoder,
    consumer: StreamConsumer,
    offsets: BTreeMap<i32, i64>,
    sequence: u64,
}

impl KafkaSource {
    /// Builds the source with no custom decoders and freezes its
    /// capabilities.
    ///
    /// # Errors
    ///
    /// Returns the configuration error for invalid bounds, schema, or
    /// protobuf descriptors, or the Kafka error when the consumer cannot
    /// be created. The custom payload format requires
    /// [`KafkaSource::with_decoders`].
    pub fn new(config: KafkaSourceConfig) -> Result<Self> {
        Self::with_decoders(config, &KafkaDecoderRegistry::default())
    }

    /// Builds the source over one shared custom decoder registry and
    /// freezes its capabilities.
    ///
    /// # Errors
    ///
    /// Returns the configuration error for invalid bounds, schema, or
    /// protobuf descriptors, the resolution error when the custom format
    /// names an unregistered decoder, or the Kafka error when the consumer
    /// cannot be created.
    pub fn with_decoders(
        config: KafkaSourceConfig,
        decoders: &KafkaDecoderRegistry,
    ) -> Result<Self> {
        let schema = if config.schema.is_empty() {
            SourceSchema::DynamicOrUnknown
        } else {
            SourceSchema::Exact(schema_from_spec(&config.schema)?)
        };
        let bounds = DecodeBounds::new(config.max_batch_rows, config.max_batch_bytes)?;
        let decoder = config.decoder(decoders)?;
        let mut client = rdkafka::config::ClientConfig::new();
        client.set("bootstrap.servers", &config.bootstrap_servers);
        client.set("group.id", "calc-flow-kafka-source");
        client.set("enable.auto.commit", "false");
        client.set(
            "auto.offset.reset",
            config.auto_offset_reset.librdkafka_value(),
        );
        client.set("enable.partition.eof", "false");
        let consumer: StreamConsumer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        let source = Self {
            capabilities: source_capabilities(schema, bounds),
            config,
            decoder,
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
        let offsets: serde_json::Map<String, Value> = self
            .offsets
            .iter()
            .map(|(partition, offset)| (partition.to_string(), Value::from(*offset)))
            .collect();
        Cursor::unbound(
            self.sequence.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("offsets".to_string(), Value::Object(offsets)),
                ("sequence".to_string(), Value::from(self.sequence)),
            ]),
        )
    }

    fn refresh_assigned_positions(&mut self) -> Result<()> {
        let positions = self
            .consumer
            .position()
            .map_err(|error| fail("poll", &error.to_string()))?;
        let mut offsets = BTreeMap::new();
        for element in positions.elements_for_topic(&self.config.topic) {
            let offset = match element.offset() {
                Offset::Offset(offset) if offset >= 0 => offset,
                _ => {
                    return Err(fail(
                        "poll",
                        "Kafka did not resolve every assigned partition position",
                    ));
                }
            };
            offsets.insert(element.partition(), offset);
        }
        let expected = self.config.partitions.clone();
        let actual = offsets.keys().copied().collect::<Vec<_>>();
        if actual != expected {
            return Err(fail(
                "poll",
                "Kafka position set does not match the frozen assignment",
            ));
        }
        self.offsets = offsets;
        Ok(())
    }

    // Cursor validation intentionally fails each malformed durable field at the
    // boundary before any consumer state changes.
    // #lizard forgives
    fn state_from_cursor(&self, cursor: &Cursor) -> Result<(BTreeMap<i32, i64>, u64)> {
        let sequence = cursor
            .payload()
            .get("sequence")
            .and_then(Value::as_u64)
            .ok_or_else(|| fail("open", "Kafka cursor sequence is missing"))?;
        if cursor.order() != sequence.to_be_bytes() {
            return Err(fail(
                "open",
                "Kafka cursor order does not match its sequence",
            ));
        }
        let entries = cursor
            .payload()
            .get("offsets")
            .and_then(Value::as_object)
            .ok_or_else(|| fail("open", "Kafka cursor partition offsets are missing"))?;
        let offsets = entries
            .iter()
            .map(|(partition, offset)| {
                let partition = partition
                    .parse::<i32>()
                    .map_err(|_| fail("open", "Kafka cursor partition is not an i32"))?;
                let offset = offset
                    .as_i64()
                    .filter(|offset| *offset >= 0)
                    .ok_or_else(|| fail("open", "Kafka cursor offset is not non-negative"))?;
                Ok((partition, offset))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let expected = self.config.partitions.clone();
        let actual = offsets.keys().copied().collect::<Vec<_>>();
        if actual != expected {
            return Err(fail(
                "open",
                "Kafka cursor partition set does not match the frozen assignment",
            ));
        }
        Ok((offsets, sequence))
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
            let (resume, sequence) = self.state_from_cursor(&cursor)?;
            self.offsets = resume;
            self.sequence = sequence;
            self.assign_partitions(Some(&self.offsets))
                .map_err(|error| fail("open", &error.to_string()))?;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let message = match tokio::time::timeout(POLL_TIMEOUT, self.consumer.recv()).await {
            Err(_) => return Ok(Some(SourceEvent::Idle)),
            Ok(Ok(message)) => message.detach(),
            Ok(Err(error)) if is_transient_transport_error(&error) => {
                // A broker that is down or restarting must surface as
                // idleness so the job outlives the outage; protocol
                // and decode failures still fail closed.
                return Ok(Some(SourceEvent::Idle));
            }
            Ok(Err(error)) => return Err(fail("poll", &error.to_string())),
        };
        let partition = message.partition();
        let offset = message.offset();
        let payload = message.payload().unwrap_or_default();
        let batch = self.decoder.decode(payload, &self.config)?;
        let next_offset = offset
            .checked_add(1)
            .ok_or_else(|| fail("poll", "Kafka offset exhausted i64"))?;
        self.refresh_assigned_positions()?;
        self.offsets.insert(partition, next_offset);
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("poll", "Kafka cursor sequence exhausted u64"))?;
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
    /// Dedicated, one-partition compacted epoch-ledger topic.
    pub ledger_topic: String,
    /// Stable transactional ID owner; derived from pipeline and output
    /// identity by the factory, never from secrets.
    pub transactional_id: String,
    /// Payload wire format.
    pub format: KafkaFormat,
    /// Maximum source rows staged in one epoch.
    pub max_epoch_rows: u64,
    /// Maximum encoded record bytes staged in one epoch.
    pub max_epoch_bytes: u64,
}

impl KafkaSinkConfig {
    /// Parses the sink configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option for a missing or malformed value.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        if options.contains_key("transactional_id") {
            return Err(CalcFlowError::InvalidArgument {
                field: "transactional_id".into(),
                message: "transactional IDs are derived from pipeline and output identity".into(),
            });
        }
        let pipeline = required_string(options, "pipeline")?;
        let output = required_string(options, "output")?;
        Ok(Self {
            bootstrap_servers: required_string(options, "bootstrap_servers")?,
            topic: required_string(options, "topic")?,
            ledger_topic: required_string(options, "ledger_topic")?,
            transactional_id: transactional_id(&pipeline, &output),
            format: parse_sink_format(options)?,
            max_epoch_rows: positive_kafka_option(options, "max_epoch_rows", 1_000_000)?,
            max_epoch_bytes: positive_kafka_option(options, "max_epoch_bytes", 256 * 1024 * 1024)?,
        })
    }
}

/// Parses the sink payload format, rejecting the source-only codecs.
fn parse_sink_format(options: &JsonMap) -> Result<KafkaFormat> {
    let format = KafkaFormat::parse(&required_string(options, "format")?)?;
    if matches!(format, KafkaFormat::Protobuf | KafkaFormat::Custom) {
        return Err(CalcFlowError::InvalidArgument {
            field: "format".into(),
            message:
                "protobuf and custom payloads decode from Kafka only; sinks encode json and csv"
                    .into(),
        });
    }
    Ok(format)
}

fn positive_kafka_option(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
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

/// Parses and normalizes the explicit partition assignment.
fn parse_kafka_bounds(options: &JsonMap) -> Result<(u64, u64)> {
    Ok((
        positive_kafka_option(options, "max_batch_rows", 8192)?,
        positive_kafka_option(options, "max_batch_bytes", 8 * 1024 * 1024)?,
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

/// Validates the format companion options against the parsed format.
fn parse_format_companions(
    options: &JsonMap,
    format: KafkaFormat,
    schema: &[ArrowFieldSpec],
) -> Result<(Option<String>, Option<String>, Option<FormatIdentity>)> {
    let (descriptor_set, message) = parse_protobuf_companions(options, format, schema)?;
    let decoder = parse_custom_companion(options, format)?;
    Ok((descriptor_set, message, decoder))
}

fn parse_protobuf_companions(
    options: &JsonMap,
    format: KafkaFormat,
    schema: &[ArrowFieldSpec],
) -> Result<(Option<String>, Option<String>)> {
    if !matches!(format, KafkaFormat::Protobuf) {
        for key in ["descriptor_set", "message"] {
            if options.contains_key(key) {
                return Err(CalcFlowError::InvalidArgument {
                    field: key.into(),
                    message: "option applies to the protobuf payload format only".into(),
                });
            }
        }
        return Ok((None, None));
    }
    if schema.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "schema".into(),
            message: "protobuf payloads require an explicit schema".into(),
        });
    }
    Ok((
        Some(required_string(options, "descriptor_set")?),
        Some(required_string(options, "message")?),
    ))
}

fn parse_custom_companion(
    options: &JsonMap,
    format: KafkaFormat,
) -> Result<Option<FormatIdentity>> {
    if !matches!(format, KafkaFormat::Custom) {
        if options.contains_key("decoder") {
            return Err(CalcFlowError::InvalidArgument {
                field: "decoder".into(),
                message: "option applies to the custom payload format only".into(),
            });
        }
        return Ok(None);
    }
    let decoder = options
        .get("decoder")
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "decoder".into(),
            message: "custom payloads require a decoder identity".into(),
        })?;
    let name = decoder.get("name").and_then(Value::as_str).ok_or_else(|| {
        CalcFlowError::InvalidArgument {
            field: "decoder".into(),
            message: "decoder identity requires a name string".into(),
        }
    })?;
    let version = decoder
        .get("version")
        .and_then(Value::as_str)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "decoder".into(),
            message: "decoder identity requires a version string".into(),
        })?;
    Ok(Some(FormatIdentity::new(name, version)?))
}

/// The transactional Kafka sink.
pub struct TransactionalKafkaSink {
    config: KafkaSinkConfig,
    producer: FutureProducer,
    active: bool,
    delivered: u64,
    pending_records: Vec<Vec<u8>>,
    pending_bytes: u64,
}

const PREPARED_RECORDS_SEGMENT: &str = "records";

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
            pending_records: Vec::new(),
            pending_bytes: 0,
        })
    }

    fn encode(&self, batch: &Batch) -> Result<Vec<u8>> {
        use calc_flow::FormatEncoder as _;
        match self.config.format {
            KafkaFormat::Json => JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.encode(batch),
            KafkaFormat::Csv => CsvCodec::new(csv::IDENTITY_VERSION, true)?.encode(batch),
            KafkaFormat::Protobuf | KafkaFormat::Custom => Err(fail(
                "encode",
                "protobuf and custom payloads decode from Kafka only; sinks encode json and csv",
            )),
        }
    }

    async fn write_ledger_marker(&self, epoch: calc_flow::Epoch, evidence: &JsonMap) -> Result<()> {
        let segment_sha256 = evidence
            .get("segment_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("commit", "prepared segment hash is missing"))?;
        let payload = serde_json::to_vec(&BTreeMap::from([
            ("epoch", Value::from(epoch.as_u64())),
            (
                "transactional_id",
                Value::String(self.config.transactional_id.clone()),
            ),
            ("segment_sha256", Value::String(segment_sha256.to_string())),
        ]))
        .map_err(|error| fail("commit", &error.to_string()))?;
        let key = self.config.transactional_id.as_bytes().to_vec();
        let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.ledger_topic)
            .partition(0)
            .key(&key)
            .payload(&payload);
        self.producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(error, _)| fail("commit", &error.to_string()))?;
        Ok(())
    }

    // The ledger scan is one bounded protocol state machine; splitting its
    // termination branches would obscure which Kafka event closes recovery.
    // #lizard forgives
    async fn latest_ledger_marker(&self) -> Result<Option<KafkaLedgerMarker>> {
        let mut client = rdkafka::config::ClientConfig::new();
        client.set("bootstrap.servers", &self.config.bootstrap_servers);
        client.set(
            "group.id",
            format!("{}-recovery", self.config.transactional_id),
        );
        client.set("enable.auto.commit", "false");
        client.set("enable.partition.eof", "true");
        client.set("isolation.level", "read_committed");
        let consumer: StreamConsumer = client
            .create()
            .map_err(|error| fail("recover", &error.to_string()))?;
        let mut assignment = TopicPartitionList::new();
        assignment
            .add_partition_offset(&self.config.ledger_topic, 0, Offset::Beginning)
            .map_err(|error| fail("recover", &error.to_string()))?;
        consumer
            .assign(&assignment)
            .map_err(|error| fail("recover", &error.to_string()))?;
        let scan = async {
            let mut latest = None;
            loop {
                match consumer.recv().await {
                    Ok(message)
                        if message.key() == Some(self.config.transactional_id.as_bytes()) =>
                    {
                        let payload = message
                            .payload()
                            .ok_or_else(|| fail("recover", "Kafka ledger marker is empty"))?;
                        let marker: KafkaLedgerMarker = serde_json::from_slice(payload)
                            .map_err(|_| fail("recover", "Kafka ledger marker is malformed"))?;
                        if marker.transactional_id != self.config.transactional_id {
                            return Err(fail(
                                "recover",
                                "Kafka ledger marker names another transactional ID",
                            ));
                        }
                        latest = Some(marker);
                    }
                    Ok(_) => {}
                    Err(rdkafka::error::KafkaError::PartitionEOF(_)) => return Ok(latest),
                    Err(error) => return Err(fail("recover", &error.to_string())),
                }
            }
        };
        tokio::time::timeout(Duration::from_secs(30), scan)
            .await
            .map_err(|_| fail("recover", "Kafka ledger scan timed out"))?
    }

    // Preflight reports each broker contract failure distinctly before the
    // transactional producer can publish user data.
    // #lizard forgives
    async fn preflight_ledger(&self) -> Result<()> {
        let metadata = self
            .producer
            .client()
            .fetch_metadata(Some(&self.config.ledger_topic), Duration::from_secs(10))
            .map_err(|error| fail("open", &error.to_string()))?;
        let topic = metadata
            .topics()
            .iter()
            .find(|topic| topic.name() == self.config.ledger_topic)
            .ok_or_else(|| fail("open", "Kafka ledger topic does not exist"))?;
        if topic.partitions().len() != 1 {
            return Err(fail(
                "open",
                "Kafka ledger topic must have exactly one partition",
            ));
        }
        let admin: AdminClient<DefaultClientContext> = rdkafka::config::ClientConfig::new()
            .set("bootstrap.servers", &self.config.bootstrap_servers)
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        let results = admin
            .describe_configs(
                &[ResourceSpecifier::Topic(&self.config.ledger_topic)],
                &AdminOptions::new().operation_timeout(Some(Duration::from_secs(10))),
            )
            .await
            .map_err(|error| fail("open", &error.to_string()))?;
        let resource = results
            .into_iter()
            .next()
            .ok_or_else(|| fail("open", "Kafka ledger topic config is missing"))?
            .map_err(|error| fail("open", &error.to_string()))?;
        let cleanup = resource
            .get("cleanup.policy")
            .and_then(|entry| entry.value.as_deref())
            .ok_or_else(|| fail("open", "Kafka ledger cleanup.policy is missing"))?;
        if cleanup != "compact" {
            return Err(fail(
                "open",
                "Kafka ledger topic must use cleanup.policy=compact without delete retention",
            ));
        }
        Ok(())
    }
}

#[derive(serde::Deserialize)]
struct KafkaLedgerMarker {
    epoch: u64,
    transactional_id: String,
    segment_sha256: String,
}

fn encode_records(records: &[Vec<u8>]) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(
        &u64::try_from(records.len())
            .map_err(|_| fail("pre_commit", "record count exceeds u64"))?
            .to_be_bytes(),
    );
    for record in records {
        encoded.extend_from_slice(
            &u64::try_from(record.len())
                .map_err(|_| fail("pre_commit", "record length exceeds u64"))?
                .to_be_bytes(),
        );
        encoded.extend_from_slice(record);
    }
    Ok(encoded)
}

// The durable record framing decoder keeps every bounds check adjacent to the
// cursor it protects so truncated evidence always fails closed.
// #lizard forgives
fn decode_records(encoded: &[u8]) -> Result<Vec<Vec<u8>>> {
    let mut offset = 0_usize;
    let take_u64 = |offset: &mut usize| -> Result<u64> {
        let end = offset
            .checked_add(8)
            .ok_or_else(|| fail("recover", "prepared record segment offset exhausted"))?;
        let bytes: [u8; 8] = encoded
            .get(*offset..end)
            .ok_or_else(|| fail("recover", "prepared record segment is truncated"))?
            .try_into()
            .expect("slice length checked");
        *offset = end;
        Ok(u64::from_be_bytes(bytes))
    };
    let count = usize::try_from(take_u64(&mut offset)?).map_err(|_| {
        fail(
            "recover",
            "prepared record count does not fit this platform",
        )
    })?;
    let mut records = Vec::with_capacity(count);
    for _ in 0..count {
        let len = usize::try_from(take_u64(&mut offset)?).map_err(|_| {
            fail(
                "recover",
                "prepared record length does not fit this platform",
            )
        })?;
        let end = offset
            .checked_add(len)
            .ok_or_else(|| fail("recover", "prepared record offset exhausted"))?;
        records.push(
            encoded
                .get(offset..end)
                .ok_or_else(|| fail("recover", "prepared record segment is truncated"))?
                .to_vec(),
        );
        offset = end;
    }
    if offset != encoded.len() {
        return Err(fail(
            "recover",
            "prepared record segment has trailing bytes",
        ));
    }
    Ok(records)
}

fn validate_prepared_evidence(
    config: &KafkaSinkConfig,
    epoch: calc_flow::Epoch,
    evidence: &JsonMap,
    records: &[Vec<u8>],
) -> Result<()> {
    validate_recovery_evidence(&config.transactional_id, evidence)?;
    let protocol = |message: String| fail("recover", &message);
    crate::evidence::check_epoch(evidence, epoch).map_err(protocol)?;
    if crate::evidence::string_field(evidence, "ledger_topic").map_err(protocol)?
        != config.ledger_topic
    {
        return Err(fail(
            "recover",
            "prepared Kafka evidence names another sink",
        ));
    }
    crate::evidence::check_segment_id(evidence, PREPARED_RECORDS_SEGMENT).map_err(protocol)?;
    let segment = encode_records(records)?;
    crate::evidence::check_segment(evidence, &segment).map_err(protocol)?;
    Ok(())
}

#[async_trait]
impl TransactionalStreamSink for TransactionalKafkaSink {
    async fn open(&mut self) -> Result<()> {
        self.preflight_ledger().await
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
        self.pending_records.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if !self.active {
            return Err(fail("write", "write before begin_epoch"));
        }
        let payload = self.encode(batch)?;
        let rows = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        let next_rows = self
            .delivered
            .checked_add(rows)
            .ok_or_else(|| fail("write", "Kafka epoch row count exhausted u64"))?;
        let next_bytes = self
            .pending_bytes
            .checked_add(u64::try_from(payload.len()).unwrap_or(u64::MAX))
            .ok_or_else(|| fail("write", "Kafka epoch byte count exhausted u64"))?;
        if next_rows > self.config.max_epoch_rows || next_bytes > self.config.max_epoch_bytes {
            return Err(fail(
                "write",
                "Kafka epoch exceeds configured staging bounds",
            ));
        }
        let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(&payload);
        let delivery = self
            .producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(error, _message)| fail("write", &error.to_string()))?;
        self.pending_records.push(payload);
        self.pending_bytes = next_bytes;
        self.delivered = next_rows;
        let _ = delivery;
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if !self.active {
            return Err(fail("pre_commit", "pre_commit before begin_epoch"));
        }
        self.producer
            .flush(Duration::from_secs(30))
            .map_err(|error| fail("pre_commit", &error.to_string()))?;
        let segment = encode_records(&self.pending_records)?;
        Ok(BTreeMap::from([
            (
                "transactional_id".to_string(),
                Value::String(self.config.transactional_id.clone()),
            ),
            ("messages".to_string(), Value::from(self.delivered)),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            (
                "ledger_topic".to_string(),
                Value::String(self.config.ledger_topic.clone()),
            ),
            (
                "segment_id".to_string(),
                Value::String(PREPARED_RECORDS_SEGMENT.into()),
            ),
            (
                "segment_bytes".to_string(),
                Value::from(u64::try_from(segment.len()).unwrap_or(u64::MAX)),
            ),
            (
                "segment_sha256".to_string(),
                Value::String(hex::encode(Sha256::digest(&segment))),
            ),
        ]))
    }

    async fn pre_commit_segments(
        &mut self,
        _epoch: calc_flow::Epoch,
    ) -> Result<BTreeMap<String, Vec<u8>>> {
        Ok(BTreeMap::from([(
            PREPARED_RECORDS_SEGMENT.into(),
            encode_records(&self.pending_records)?,
        )]))
    }

    async fn commit(&mut self, epoch: calc_flow::Epoch, pre_commit: &JsonMap) -> Result<()> {
        if !self.active {
            return Err(fail("commit", "commit without an active transaction"));
        }
        validate_prepared_evidence(&self.config, epoch, pre_commit, &self.pending_records)?;
        self.write_ledger_marker(epoch, pre_commit).await?;
        self.producer
            .commit_transaction(Duration::from_secs(30))
            .map_err(|error| fail("commit", &error.to_string()))?;
        self.active = false;
        self.pending_records.clear();
        self.pending_bytes = 0;
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
        self.pending_records.clear();
        self.pending_bytes = 0;
        Ok(())
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        validate_recovery_evidence(&self.config.transactional_id, recovery.pre_commit())?;
        let segment = recovery
            .segments()
            .get(PREPARED_RECORDS_SEGMENT)
            .ok_or_else(|| fail("recover", "prepared Kafka record segment is missing"))?;
        let records = decode_records(segment)?;
        validate_prepared_evidence(
            &self.config,
            recovery.epoch(),
            recovery.pre_commit(),
            &records,
        )?;
        if let Some(marker) = self.latest_ledger_marker().await? {
            if marker.epoch > recovery.epoch().as_u64() {
                return Ok(());
            }
            if marker.epoch == recovery.epoch().as_u64() {
                let expected_hash = recovery.pre_commit()["segment_sha256"]
                    .as_str()
                    .ok_or_else(|| fail("recover", "prepared segment hash is missing"))?;
                if marker.segment_sha256 != expected_hash {
                    return Err(fail(
                        "recover",
                        "ledger marker hash disagrees with durable prepared records",
                    ));
                }
                return Ok(());
            }
        }
        self.producer
            .begin_transaction()
            .map_err(|error| fail("recover", &error.to_string()))?;
        for payload in &records {
            let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(payload);
            self.producer
                .send(record, Duration::from_secs(10))
                .await
                .map_err(|(error, _)| fail("recover", &error.to_string()))?;
        }
        self.write_ledger_marker(recovery.epoch(), recovery.pre_commit())
            .await?;
        self.producer
            .commit_transaction(Duration::from_secs(30))
            .map_err(|error| fail("recover", &error.to_string()))
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
            KafkaFormat::Protobuf | KafkaFormat::Custom => {
                return Err(fail(
                    "encode",
                    "protobuf and custom payloads decode from Kafka only; sinks encode json and csv",
                ));
            }
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
    FormatDescriptor, FormatIdentity, SecretResolver, TransactionSupport, WatermarkSupport,
};

use crate::{csv, json_lines};

/// The connector implementation version re-exported as the transport
/// identity constant.
pub const KAFKA_CONNECTOR_VERSION: &str = IDENTITY_VERSION;

/// Trusted source factory for the Kafka transport (feature `kafka`).
pub struct KafkaSourceFactory {
    descriptor: ConnectorDescriptor,
    decoders: KafkaDecoderRegistry,
}

impl KafkaSourceFactory {
    /// Creates the factory with no custom decoders.
    pub fn new() -> Self {
        Self {
            descriptor: kafka_connector_descriptor(),
            decoders: KafkaDecoderRegistry::default(),
        }
    }

    /// Registers one trusted custom payload decoder.
    ///
    /// # Errors
    ///
    /// Returns the registry error when the decoder identity shadows a
    /// built-in format name or is already registered.
    pub fn with_decoder(self, decoder: Arc<dyn FormatDecoder>) -> Result<Self> {
        self.decoders.register(decoder)?;
        Ok(self)
    }

    /// Shares one custom decoder registry with the factory; registrations
    /// stay visible through the shared handle.
    #[must_use]
    pub fn with_decoders(self, decoders: KafkaDecoderRegistry) -> Self {
        Self { decoders, ..self }
    }

    /// The shared custom decoder registry of this factory.
    pub fn decoders(&self) -> &KafkaDecoderRegistry {
        &self.decoders
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

    fn validate(&self, options: &JsonMap) -> Result<()> {
        KafkaSourceConfig::from_options(options).map(drop)
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = KafkaSourceConfig::from_options(options)?;
        Ok(Box::new(KafkaSource::with_decoders(
            config,
            &self.decoders,
        )?))
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

    fn validate(&self, options: &JsonMap) -> Result<()> {
        KafkaSinkConfig::from_options(options).map(drop)
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
            transaction: TransactionSupport::LedgerIdempotent,
            snapshot: false,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: vec![
            FormatIdentity::new(json_lines::IDENTITY, json_lines::IDENTITY_VERSION)
                .expect("json identity"),
            FormatIdentity::new(csv::IDENTITY, csv::IDENTITY_VERSION).expect("csv identity"),
            FormatIdentity::new(protobuf::IDENTITY, protobuf::IDENTITY_VERSION)
                .expect("protobuf identity"),
            FormatIdentity::new(crate::CUSTOM_FORMAT_IDENTITY, crate::CUSTOM_FORMAT_VERSION)
                .expect("custom identity"),
        ],
        config_schema: JsonMap::from([
            ("bootstrap_servers".to_string(), serde_json::json!("string")),
            ("topic".to_string(), serde_json::json!("string")),
            ("partitions".to_string(), serde_json::json!("array")),
            ("auto_offset_reset".to_string(), serde_json::json!("string")),
            ("format".to_string(), serde_json::json!("string")),
            ("descriptor_set".to_string(), serde_json::json!("string")),
            ("message".to_string(), serde_json::json!("string")),
            ("decoder".to_string(), serde_json::json!("object")),
            ("schema".to_string(), serde_json::json!("array")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("ledger_topic".to_string(), serde_json::json!("string")),
            ("pipeline".to_string(), serde_json::json!("string")),
            ("output".to_string(), serde_json::json!("string")),
            ("max_epoch_rows".to_string(), serde_json::json!("u64")),
            ("max_epoch_bytes".to_string(), serde_json::json!("u64")),
        ]),
        secret_slots: BTreeSet::new(),
        required_secret_slots: BTreeSet::new(),
    }
}

/// Registers the Kafka connectors and their protobuf format codec into
/// one trusted registry (feature `kafka`).
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot or format
/// identity is already occupied.
pub fn register_kafka_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    register_kafka_connectors_with_decoders(registry, KafkaDecoderRegistry::default())
}

/// Registers the Kafka connectors over one shared custom decoder registry
/// (feature `kafka`).
///
/// Registrations into the shared registry stay visible through the
/// snapshot taken from `registry`, so decoders may register until the job
/// opens.
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot or format
/// identity is already occupied.
pub fn register_kafka_connectors_with_decoders(
    registry: &mut ConnectorRegistry,
    decoders: KafkaDecoderRegistry,
) -> Result<()> {
    registry.register_format(FormatDescriptor {
        identity: FormatIdentity::new(protobuf::IDENTITY, protobuf::IDENTITY_VERSION)?,
    })?;
    registry.register_connector(
        kafka_connector_descriptor(),
        ConnectorFactories::both(
            Arc::new(KafkaSourceFactory::new().with_decoders(decoders)),
            Arc::new(KafkaSinkFactory::new()),
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn source_options(format: &str) -> JsonMap {
        BTreeMap::from([
            (
                "bootstrap_servers".into(),
                Value::String("127.0.0.1:1".into()),
            ),
            ("topic".into(), Value::String("events".into())),
            ("format".into(), Value::String(format.into())),
            (
                "schema".into(),
                serde_json::json!([
                    {"name": "id", "data_type": "int64", "nullable": false},
                    {"name": "label", "data_type": "string", "nullable": false}
                ]),
            ),
        ])
    }

    #[test]
    fn json_and_csv_payloads_decode_against_the_frozen_schema() {
        let decoders = KafkaDecoderRegistry::default();
        let json = KafkaSourceConfig::from_options(&source_options("json")).unwrap();
        let batch = json
            .decode(b"{\"id\":1,\"label\":\"one\"}\n", &decoders)
            .unwrap();
        assert_eq!(batch.num_rows(), 1);

        let csv = KafkaSourceConfig::from_options(&source_options("csv")).unwrap();
        let batch = csv.decode(b"id,label\n2,two\n", &decoders).unwrap();
        assert_eq!(batch.num_rows(), 1);
        assert!(
            csv.decode(b"id,label\nnot-an-int,two\n", &decoders)
                .is_err()
        );
    }

    #[test]
    fn configuration_rejects_ambiguous_partitions_formats_and_bounds() {
        let mut candidate = source_options("future");
        assert!(KafkaSourceConfig::from_options(&candidate).is_err());

        candidate = source_options("json");
        candidate.insert("partitions".into(), Value::Array(Vec::new()));
        assert!(KafkaSourceConfig::from_options(&candidate).is_err());

        candidate = source_options("json");
        candidate.insert("partitions".into(), serde_json::json!([1, 0, 1]));
        assert_eq!(
            KafkaSourceConfig::from_options(&candidate)
                .unwrap()
                .partitions,
            vec![0, 1]
        );

        for field in ["max_batch_rows", "max_batch_bytes"] {
            candidate = source_options("json");
            candidate.insert(field.into(), Value::from(0));
            assert!(
                KafkaSourceConfig::from_options(&candidate).is_err(),
                "{field}"
            );
        }
    }

    #[tokio::test]
    async fn durable_cursor_shape_is_strict_and_partition_bound() {
        let mut candidate = source_options("json");
        candidate.insert("partitions".into(), serde_json::json!([0, 2]));
        let mut source =
            KafkaSource::new(KafkaSourceConfig::from_options(&candidate).unwrap()).unwrap();
        source.offsets = BTreeMap::from([(0, 7), (2, 9)]);
        source.sequence = 3;
        let cursor = source.cursor_from_offsets().unwrap();
        let (offsets, sequence) = source.state_from_cursor(&cursor).unwrap();
        assert_eq!(offsets, source.offsets);
        assert_eq!(sequence, 3);
        source.open(Some(cursor)).await.unwrap();

        let wrong_order = Cursor::unbound(
            2_u64.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("offsets".into(), serde_json::json!({"0": 7, "2": 9})),
                ("sequence".into(), Value::from(3)),
            ]),
        )
        .unwrap();
        assert!(source.state_from_cursor(&wrong_order).is_err());

        let wrong_partitions = Cursor::unbound(
            3_u64.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("offsets".into(), serde_json::json!({"0": 7})),
                ("sequence".into(), Value::from(3)),
            ]),
        )
        .unwrap();
        assert!(source.state_from_cursor(&wrong_partitions).is_err());
    }

    #[test]
    fn protobuf_source_options_validate_their_required_companions() {
        assert!(matches!(
            KafkaFormat::parse("protobuf").expect("protobuf is a known payload format"),
            KafkaFormat::Protobuf
        ));

        let mut missing_descriptor = source_options("protobuf");
        missing_descriptor.insert("message".into(), Value::String("events.Order".into()));
        let error = KafkaSourceConfig::from_options(&missing_descriptor)
            .expect_err("protobuf without a descriptor set fails closed");
        assert!(error.to_string().contains("descriptor_set"), "{error}");

        let mut missing_message = source_options("protobuf");
        missing_message.insert("descriptor_set".into(), Value::String("orders.pb".into()));
        let error = KafkaSourceConfig::from_options(&missing_message)
            .expect_err("protobuf without a message name fails closed");
        assert!(error.to_string().contains("message"), "{error}");

        let mut missing_schema = source_options("protobuf");
        missing_schema.insert("descriptor_set".into(), Value::String("orders.pb".into()));
        missing_schema.insert("message".into(), Value::String("events.Order".into()));
        missing_schema.remove("schema");
        let error = KafkaSourceConfig::from_options(&missing_schema)
            .expect_err("protobuf without an explicit schema fails closed");
        assert!(error.to_string().contains("schema"), "{error}");
    }

    #[test]
    fn protobuf_payloads_decode_through_the_source_config() {
        let directory = tempfile::tempdir().expect("tempdir");
        let path = protobuf::fixtures::write_descriptor_set(directory.path());
        let mut options = source_options("protobuf");
        options.insert(
            "descriptor_set".into(),
            Value::String(path.to_string_lossy().into_owned()),
        );
        options.insert(
            "message".into(),
            Value::String(protobuf::fixtures::ORDER_MESSAGE.into()),
        );
        let config = KafkaSourceConfig::from_options(&options).expect("protobuf config parses");
        let payload = protobuf::fixtures::order_payload(&[
            ("id", prost_reflect::Value::I64(3)),
            ("label", prost_reflect::Value::String("three".to_string())),
        ]);
        let decoders = KafkaDecoderRegistry::default();
        let batch = config
            .decode(&payload, &decoders)
            .expect("protobuf payload decodes");
        assert_eq!(batch.num_rows(), 1);

        let mut stale = config.clone();
        stale.message = Some("events.Missing".into());
        assert!(
            stale.decode(&payload, &decoders).is_err(),
            "an unknown message name fails at decoder construction"
        );
    }

    struct RenamedJsonDecoder;

    impl FormatDecoder for RenamedJsonDecoder {
        fn identity(&self) -> FormatIdentity {
            FormatIdentity::new("orders-json", "1").expect("decoder identity")
        }

        fn decode(
            &self,
            bytes: &[u8],
            bounds: &DecodeBounds,
            schema: &[ArrowFieldSpec],
        ) -> Result<Batch> {
            JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.decode(bytes, bounds, schema)
        }
    }

    struct JsonShadowDecoder;

    impl FormatDecoder for JsonShadowDecoder {
        fn identity(&self) -> FormatIdentity {
            FormatIdentity::new("json", "9").expect("decoder identity")
        }

        fn decode(
            &self,
            bytes: &[u8],
            bounds: &DecodeBounds,
            schema: &[ArrowFieldSpec],
        ) -> Result<Batch> {
            JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.decode(bytes, bounds, schema)
        }
    }

    fn custom_source_options() -> JsonMap {
        let mut options = source_options("custom");
        options.insert(
            "decoder".into(),
            serde_json::json!({"name": "orders-json", "version": "1"}),
        );
        options
    }

    #[test]
    fn custom_format_validates_its_decoder_companion() {
        let mut missing = source_options("custom");
        let error = KafkaSourceConfig::from_options(&missing)
            .expect_err("custom without a decoder identity fails closed");
        assert!(error.to_string().contains("decoder"), "{error}");

        missing.insert("decoder".into(), serde_json::json!({"name": "orders-json"}));
        let error = KafkaSourceConfig::from_options(&missing)
            .expect_err("a decoder identity without a version fails closed");
        assert!(error.to_string().contains("version"), "{error}");

        let mut wrong_format = source_options("json");
        wrong_format.insert(
            "decoder".into(),
            serde_json::json!({"name": "orders-json", "version": "1"}),
        );
        let error = KafkaSourceConfig::from_options(&wrong_format)
            .expect_err("the decoder option applies to the custom format only");
        assert!(error.to_string().contains("custom"), "{error}");

        for key in ["descriptor_set", "message"] {
            let mut misplaced = source_options("custom");
            misplaced.insert(
                "decoder".into(),
                serde_json::json!({"name": "orders-json", "version": "1"}),
            );
            misplaced.insert(key.into(), Value::String("value".into()));
            let error = KafkaSourceConfig::from_options(&misplaced)
                .expect_err("protobuf companions apply to the protobuf format only");
            assert!(error.to_string().contains("protobuf"), "{key}: {error}");
        }
    }

    #[test]
    fn custom_decoders_register_resolve_and_decode() {
        let registry = KafkaDecoderRegistry::default();
        registry
            .register(Arc::new(RenamedJsonDecoder))
            .expect("decoder registers");
        let error = registry
            .register(Arc::new(RenamedJsonDecoder))
            .expect_err("duplicate identities conflict");
        assert!(matches!(error, CalcFlowError::Conflict { .. }), "{error}");
        let error = registry
            .register(Arc::new(JsonShadowDecoder))
            .expect_err("built-in format names are reserved");
        assert!(error.to_string().contains("shadows"), "{error}");
        let missing = FormatIdentity::new("missing", "1").expect("identity");
        assert!(
            registry.resolve(&missing).is_err(),
            "unknown identities fail"
        );

        let config =
            KafkaSourceConfig::from_options(&custom_source_options()).expect("custom parses");
        let batch = config
            .decode(b"{\"id\":1,\"label\":\"one\"}\n", &registry)
            .expect("the registered decoder decodes");
        assert_eq!(batch.num_rows(), 1);

        let factory = KafkaSourceFactory::new()
            .with_decoder(Arc::new(RenamedJsonDecoder))
            .expect("factory accepts the decoder");
        factory
            .validate(&custom_source_options())
            .expect("option shape validates without opening");

        let mut unknown = custom_source_options();
        unknown.insert(
            "decoder".into(),
            serde_json::json!({"name": "missing", "version": "1"}),
        );
        factory
            .validate(&unknown)
            .expect("validation stays data-only; resolution happens at open");
        let error = KafkaSource::with_decoders(
            KafkaSourceConfig::from_options(&unknown).expect("options parse"),
            &KafkaDecoderRegistry::default(),
        )
        .err()
        .expect("an unregistered decoder identity fails at open");
        assert!(error.to_string().contains("missing/1"), "{error}");
    }

    #[test]
    fn sinks_reject_the_source_only_protobuf_format() {
        for format in ["protobuf", "custom"] {
            let options = BTreeMap::from([
                (
                    "bootstrap_servers".into(),
                    Value::String("127.0.0.1:1".into()),
                ),
                ("topic".into(), Value::String("events".into())),
                ("ledger_topic".into(), Value::String("events-ledger".into())),
                ("pipeline".into(), Value::String("orders".into())),
                ("output".into(), Value::String("events".into())),
                ("format".into(), Value::String(format.into())),
            ]);
            let error = KafkaSinkConfig::from_options(&options)
                .expect_err("sinks cannot encode source-only payloads");
            assert!(error.to_string().contains("format"), "{format}: {error}");
        }
    }

    #[test]
    fn source_capabilities_preserve_exact_schema_and_bounds() {
        let config = KafkaSourceConfig::from_options(&source_options("json")).unwrap();
        let schema = SourceSchema::Exact(schema_from_spec(&config.schema).unwrap());
        let bounds = DecodeBounds::new(config.max_batch_rows, config.max_batch_bytes).unwrap();
        let capabilities = source_capabilities(schema.clone(), bounds);
        let SourceSchema::Exact(actual) = capabilities.schema else {
            panic!("the frozen Kafka schema must remain exact")
        };
        let SourceSchema::Exact(expected) = schema else {
            unreachable!("the fixture constructs an exact schema")
        };
        assert_eq!(actual, expected);
        assert_eq!(capabilities.max_batch_rows, 8192);
        assert_eq!(capabilities.max_batch_bytes, 8 * 1024 * 1024);
        assert_eq!(
            capabilities.native_watermarks,
            calc_flow::NativeWatermarkCapability::NeverEmits
        );
    }
}
