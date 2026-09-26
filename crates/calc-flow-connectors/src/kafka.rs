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
use std::sync::OnceLock;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, CalcFlowError, ConnectorError, ConnectorIdentity, ConnectorOperation,
    Cursor, DecodeBounds, FormatDecoder, JsonMap, Result, SecretHandle, SecretReference,
    SecretResolver, SecretResolverKind, SinkRecovery, SourceCapabilities, SourceEvent,
    SourceSchema, StreamSink, StreamSource, TransactionalStreamSink,
};
use rdkafka::admin::{AdminClient, AdminOptions, ResourceSpecifier};
use rdkafka::client::DefaultClientContext;
use rdkafka::consumer::{Consumer, StreamConsumer};
use rdkafka::message::Message;
use rdkafka::producer::{FutureProducer, FutureRecord, Producer};
use rdkafka::topic_partition_list::TopicPartitionList;
use rdkafka::{Offset, config::ClientConfig};
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

type ProducerCall =
    Box<dyn FnOnce(&FutureProducer) -> rdkafka::error::KafkaResult<()> + Send + 'static>;

struct ProducerCommand {
    producer: FutureProducer,
    call: ProducerCall,
    response: tokio::sync::oneshot::Sender<std::result::Result<(), String>>,
}

struct ProducerLifecycle(tokio::sync::mpsc::UnboundedSender<ProducerCommand>);

impl ProducerLifecycle {
    fn new() -> Self {
        Self(spawn_producer_actor())
    }
}

fn spawn_producer_actor() -> tokio::sync::mpsc::UnboundedSender<ProducerCommand> {
    let (sender, mut receiver) = tokio::sync::mpsc::unbounded_channel::<ProducerCommand>();
    tokio::spawn(async move {
        while let Some(command) = receiver.recv().await {
            let ProducerCommand {
                producer,
                call,
                response,
            } = command;
            let result = tokio::task::spawn_blocking(move || call(&producer))
                .await
                .map_err(|error| format!("Kafka worker failed: {error}"))
                .and_then(|result| result.map_err(|error| error.to_string()));
            let _ = response.send(result);
        }
    });
    sender
}

async fn send_producer_call(
    sender: &tokio::sync::mpsc::UnboundedSender<ProducerCommand>,
    producer: &FutureProducer,
    operation: &'static str,
    call: impl FnOnce(&FutureProducer) -> rdkafka::error::KafkaResult<()> + Send + 'static,
) -> Result<()> {
    let (response, completed) = tokio::sync::oneshot::channel();
    sender
        .send(ProducerCommand {
            producer: producer.clone(),
            call: Box::new(call),
            response,
        })
        .map_err(|_| fail(operation, "Kafka lifecycle worker stopped"))?;
    completed
        .await
        .map_err(|error| fail(operation, &format!("Kafka worker stopped: {error}")))?
        .map_err(|error| fail(operation, &error))
}

fn start_transaction_init(
    producer: &FutureProducer,
) -> tokio::sync::oneshot::Receiver<std::result::Result<(), String>> {
    let producer = producer.clone();
    let (response, completed) = tokio::sync::oneshot::channel();
    std::thread::spawn(move || {
        let result = producer
            .init_transactions(Duration::from_secs(30))
            .map_err(|error| error.to_string());
        let _ = response.send(result);
    });
    completed
}

async fn finish_transaction_init(
    completion: &mut tokio::sync::oneshot::Receiver<std::result::Result<(), String>>,
) -> Result<()> {
    completion
        .await
        .map_err(|error| fail("open", &format!("Kafka init worker stopped: {error}")))?
        .map_err(|error| fail("open", &error))
}

async fn blocking_producer_call(
    producer: &FutureProducer,
    lifecycle: &OnceLock<ProducerLifecycle>,
    operation: &'static str,
    call: impl FnOnce(&FutureProducer) -> rdkafka::error::KafkaResult<()> + Send + 'static,
) -> Result<()> {
    let lifecycle = lifecycle.get_or_init(ProducerLifecycle::new);
    send_producer_call(&lifecycle.0, producer, operation, call).await
}

/// Attaches one record's coordinates to its decode failure.
///
/// The codec detail is dropped rather than forwarded: Arrow and Python
/// client messages may echo payload bytes, and connector error details
/// must stay payload-free. The codec identity and operation survive, and
/// the topic, partition, and offset pinpoint the poison record for
/// triage and dead-lettering.
fn decode_failure(topic: &str, partition: i32, offset: i64, error: CalcFlowError) -> CalcFlowError {
    let detail = format!(
        "topic {topic:?} partition {partition} offset {offset}: record payload could not be decoded"
    );
    match error {
        CalcFlowError::Connector(inner) => CalcFlowError::Connector(ConnectorError::new(
            inner.identity,
            inner.operation,
            &detail,
        )),
        _ => fail("poll", &detail),
    }
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

/// Transport security selected for every Kafka client in one binding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KafkaSecurityProtocol {
    /// No TLS or SASL authentication.
    Plaintext,
    /// TLS transport without SASL authentication.
    Ssl,
    /// SASL authentication over an unencrypted transport.
    SaslPlaintext,
    /// SASL authentication over TLS.
    SaslSsl,
}

impl KafkaSecurityProtocol {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "plaintext" => Ok(Self::Plaintext),
            "ssl" => Ok(Self::Ssl),
            "sasl_plaintext" => Ok(Self::SaslPlaintext),
            "sasl_ssl" => Ok(Self::SaslSsl),
            _ => Err(security_option_error(
                "security_protocol",
                "expected plaintext, ssl, sasl_plaintext, or sasl_ssl",
            )),
        }
    }

    fn librdkafka_value(self) -> &'static str {
        match self {
            Self::Plaintext => "plaintext",
            Self::Ssl => "ssl",
            Self::SaslPlaintext => "sasl_plaintext",
            Self::SaslSsl => "sasl_ssl",
        }
    }

    fn uses_sasl(self) -> bool {
        matches!(self, Self::SaslPlaintext | Self::SaslSsl)
    }

    fn uses_tls(self) -> bool {
        matches!(self, Self::Ssl | Self::SaslSsl)
    }
}

/// Data-only Kafka security options; passwords arrive through secret slots.
#[derive(Clone, Debug)]
pub struct KafkaSecurityConfig {
    /// Transport security mode.
    pub protocol: KafkaSecurityProtocol,
    /// Optional trusted CA bundle path for TLS connections.
    pub ssl_ca_location: Option<String>,
    /// SASL mechanism: `PLAIN`, `SCRAM-SHA-256`, or `SCRAM-SHA-512`.
    pub sasl_mechanism: Option<String>,
    /// SASL username; the password uses the `sasl_password` secret slot.
    pub sasl_username: Option<String>,
}

impl KafkaSecurityConfig {
    fn from_options(options: &JsonMap) -> Result<Self> {
        if options.contains_key("sasl_password") {
            return Err(security_option_error(
                "sasl_password",
                "passwords must use the sasl_password secret slot",
            ));
        }
        let protocol = KafkaSecurityProtocol::parse(
            optional_security_string(options, "security_protocol")?
                .as_deref()
                .unwrap_or("plaintext"),
        )?;
        let ssl_ca_location = optional_security_string(options, "ssl_ca_location")?;
        let sasl_mechanism = optional_security_string(options, "sasl_mechanism")?;
        let sasl_username = optional_security_string(options, "sasl_username")?;
        let config = Self {
            protocol,
            ssl_ca_location,
            sasl_mechanism,
            sasl_username,
        };
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<()> {
        if self
            .ssl_ca_location
            .as_ref()
            .is_some_and(|path| path.trim().is_empty())
        {
            return Err(security_option_error(
                "ssl_ca_location",
                "must be a non-empty string",
            ));
        }
        if self.ssl_ca_location.is_some() && !self.protocol.uses_tls() {
            return Err(security_option_error(
                "ssl_ca_location",
                "a CA path requires ssl or sasl_ssl",
            ));
        }
        if self.protocol.uses_sasl() {
            if !matches!(
                self.sasl_mechanism.as_deref(),
                Some("PLAIN" | "SCRAM-SHA-256" | "SCRAM-SHA-512")
            ) {
                return Err(security_option_error(
                    "sasl_mechanism",
                    "SASL requires PLAIN, SCRAM-SHA-256, or SCRAM-SHA-512",
                ));
            }
            if self
                .sasl_username
                .as_ref()
                .is_none_or(|value| value.trim().is_empty())
            {
                return Err(security_option_error(
                    "sasl_username",
                    "SASL requires a username",
                ));
            }
        } else if self.sasl_mechanism.is_some() || self.sasl_username.is_some() {
            return Err(security_option_error(
                "sasl_mechanism",
                "SASL options require sasl_plaintext or sasl_ssl",
            ));
        }
        Ok(())
    }

    fn resolve_password(&self, secrets: &dyn SecretResolver) -> Result<Option<SecretHandle>> {
        let reference = SecretReference::new(SecretResolverKind::Registered, "sasl_password")
            .map_err(|_| fail("open", "the SASL password reference is invalid"))?;
        if !self.protocol.uses_sasl() {
            return match secrets.has_reference(&reference) {
                Some(true) => Err(fail(
                    "open",
                    "sasl_password was supplied for a non-SASL Kafka binding",
                )),
                Some(false) => Ok(None),
                None => Err(fail(
                    "open",
                    "the resolver cannot confirm whether sasl_password was supplied",
                )),
            };
        }
        secrets
            .resolve(&reference)
            .map(Some)
            .map_err(|_| fail("open", "the SASL password secret could not be resolved"))
    }
}

fn security_option_error(field: &str, message: &str) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: field.into(),
        message: message.into(),
    }
}

fn optional_security_string(options: &JsonMap, key: &str) -> Result<Option<String>> {
    match options.get(key) {
        None => Ok(None),
        Some(Value::String(value)) if !value.trim().is_empty() => Ok(Some(value.clone())),
        _ => Err(security_option_error(key, "must be a non-empty string")),
    }
}

fn kafka_client_config(
    bootstrap_servers: &str,
    security: &KafkaSecurityConfig,
    password: Option<&SecretHandle>,
) -> Result<ClientConfig> {
    security.validate()?;
    let mut client = ClientConfig::new();
    client.set("bootstrap.servers", bootstrap_servers);
    client.set("security.protocol", security.protocol.librdkafka_value());
    if let Some(path) = &security.ssl_ca_location {
        client.set("ssl.ca.location", path);
    }
    if security.protocol.uses_sasl() {
        let password = password.ok_or_else(|| {
            fail(
                "open",
                "the SASL password secret is required for this Kafka binding",
            )
        })?;
        let password = std::str::from_utf8(password.expose())
            .map_err(|_| fail("open", "the SASL password secret is not valid UTF-8"))?;
        if password.is_empty() {
            return Err(fail("open", "the SASL password secret is empty"));
        }
        client.set(
            "sasl.mechanism",
            security.sasl_mechanism.as_deref().ok_or_else(|| {
                security_option_error("sasl_mechanism", "SASL mechanism is missing")
            })?,
        );
        client.set(
            "sasl.username",
            security.sasl_username.as_deref().ok_or_else(|| {
                security_option_error("sasl_username", "SASL username is missing")
            })?,
        );
        client.set("sasl.password", password);
    } else if password.is_some() {
        return Err(fail(
            "open",
            "SASL password supplied for a non-SASL Kafka binding",
        ));
    }
    Ok(client)
}

/// Data-only configuration for one Kafka source.
#[derive(Clone, Debug)]
pub struct KafkaSourceConfig {
    /// Comma-separated bootstrap broker list.
    pub bootstrap_servers: String,
    /// Transport security shared by the source consumer.
    pub security: KafkaSecurityConfig,
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
            security: KafkaSecurityConfig::from_options(options)?,
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
        Self::with_decoders_and_password(config, decoders, None)
    }

    fn with_decoders_and_password(
        config: KafkaSourceConfig,
        decoders: &KafkaDecoderRegistry,
        password: Option<&SecretHandle>,
    ) -> Result<Self> {
        let schema = if config.schema.is_empty() {
            SourceSchema::DynamicOrUnknown
        } else {
            SourceSchema::Exact(schema_from_spec(&config.schema)?)
        };
        let bounds = DecodeBounds::new(config.max_batch_rows, config.max_batch_bytes)?;
        let decoder = config.decoder(decoders)?;
        let mut client =
            kafka_client_config(&config.bootstrap_servers, &config.security, password)?;
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
        let batch = self
            .decoder
            .decode(payload, &self.config)
            .map_err(|error| decode_failure(&self.config.topic, partition, offset, error))?;
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
    /// Transport security shared by producers and recovery clients.
    pub security: KafkaSecurityConfig,
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
            security: KafkaSecurityConfig::from_options(options)?,
            topic: required_string(options, "topic")?,
            ledger_topic: required_string(options, "ledger_topic")?,
            transactional_id: transactional_id(&pipeline, &output),
            format: parse_sink_format(options)?,
            max_epoch_rows: positive_kafka_option(options, "max_epoch_rows", 1_000_000)?,
            max_epoch_bytes: positive_kafka_option(options, "max_epoch_bytes", 256 * 1024 * 1024)?,
        })
    }
}

/// The shared rejection of the source-only payload formats, kept in one
/// place so sink validation and sink encoding cannot drift apart.
const SINK_FORMAT_MESSAGE: &str =
    "protobuf and custom payloads decode from Kafka only; sinks encode json and csv";

/// Guards one parsed format against the source-only codecs.
fn ensure_sink_format(format: KafkaFormat) -> Result<()> {
    if matches!(format, KafkaFormat::Protobuf | KafkaFormat::Custom) {
        return Err(CalcFlowError::InvalidArgument {
            field: "format".into(),
            message: SINK_FORMAT_MESSAGE.into(),
        });
    }
    Ok(())
}

/// Parses the sink payload format, rejecting the source-only codecs.
fn parse_sink_format(options: &JsonMap) -> Result<KafkaFormat> {
    let format = KafkaFormat::parse(&required_string(options, "format")?)?;
    ensure_sink_format(format)?;
    Ok(format)
}

/// Encodes one batch into the sink payload for its format.
///
/// # Errors
///
/// Returns the codec's safe encode error, or the shared sink-format
/// rejection when the format is source-only — unreachable for options
/// parsed through [`KafkaSinkConfig::from_options`].
fn encode_kafka_payload(format: KafkaFormat, batch: &Batch) -> Result<Vec<u8>> {
    use calc_flow::FormatEncoder as _;
    match format {
        KafkaFormat::Json => JsonLinesCodec::new(json_lines::IDENTITY_VERSION)?.encode(batch),
        KafkaFormat::Csv => CsvCodec::new(csv::IDENTITY_VERSION, true)?.encode(batch),
        KafkaFormat::Protobuf | KafkaFormat::Custom => Err(fail("encode", SINK_FORMAT_MESSAGE)),
    }
}

fn kafka_format_name(format: KafkaFormat) -> &'static str {
    match format {
        KafkaFormat::Json => "json",
        KafkaFormat::Csv => "csv",
        KafkaFormat::Protobuf => "protobuf",
        KafkaFormat::Custom => "custom",
    }
}

fn kafka_schema_hash(format: KafkaFormat, schema: Option<&arrow::datatypes::Schema>) -> String {
    let fields = schema
        .map(|schema| {
            schema
                .fields()
                .iter()
                .map(|field| {
                    (
                        field.name(),
                        format!("{:?}", field.data_type()),
                        field.is_nullable(),
                        field.metadata().iter().collect::<BTreeMap<_, _>>(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let metadata = schema.map(|schema| schema.metadata().iter().collect::<BTreeMap<_, _>>());
    let bytes = serde_json::to_vec(&(kafka_format_name(format), fields, metadata))
        .expect("data-only Kafka schema identity serializes");
    crate::evidence::sha256_hex(&bytes)
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
        reject_misplaced_option(options, "descriptor_set", "protobuf")?;
        reject_misplaced_option(options, "message", "protobuf")?;
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
        reject_misplaced_option(options, "decoder", "custom")?;
        return Ok(None);
    }
    let decoder = options
        .get("decoder")
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "decoder".into(),
            message: "custom payloads require a decoder identity".into(),
        })?;
    let name = required_decoder_field(decoder, "name")?;
    let version = required_decoder_field(decoder, "version")?;
    Ok(Some(FormatIdentity::new(name, version)?))
}

fn reject_misplaced_option(options: &JsonMap, key: &str, format: &str) -> Result<()> {
    if options.contains_key(key) {
        return Err(CalcFlowError::InvalidArgument {
            field: key.into(),
            message: format!("option applies to the {format} payload format only"),
        });
    }
    Ok(())
}

fn required_decoder_field<'a>(decoder: &'a Value, key: &str) -> Result<&'a str> {
    decoder
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| CalcFlowError::InvalidArgument {
            field: "decoder".into(),
            message: format!("decoder identity requires a {key} string"),
        })
}

/// The transactional Kafka sink.
pub struct TransactionalKafkaSink {
    config: KafkaSinkConfig,
    password: Option<SecretHandle>,
    producer: FutureProducer,
    lifecycle: OnceLock<ProducerLifecycle>,
    init_inflight: Option<tokio::sync::oneshot::Receiver<std::result::Result<(), String>>>,
    active: bool,
    delivered: u64,
    pending_records: Vec<Vec<u8>>,
    pending_bytes: u64,
    pending_schema_hash: Option<String>,
}

const PREPARED_RECORDS_SEGMENT: &str = "records";

impl TransactionalKafkaSink {
    /// Builds the sink. [`TransactionalStreamSink::open`] fences stale producers.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the producer cannot be created.
    pub fn new(config: KafkaSinkConfig) -> Result<Self> {
        Self::new_with_password(config, None)
    }

    fn new_with_password(config: KafkaSinkConfig, password: Option<SecretHandle>) -> Result<Self> {
        let mut client = kafka_client_config(
            &config.bootstrap_servers,
            &config.security,
            password.as_ref(),
        )?;
        client.set("transactional.id", &config.transactional_id);
        client.set("enable.idempotence", "true");
        client.set("message.timeout.ms", "30000");
        client.set("transaction.timeout.ms", "60000");
        let producer: FutureProducer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(Self {
            config,
            password,
            producer,
            lifecycle: OnceLock::new(),
            init_inflight: None,
            active: false,
            delivered: 0,
            pending_records: Vec::new(),
            pending_bytes: 0,
            pending_schema_hash: None,
        })
    }

    async fn write_ledger_marker(&self, epoch: calc_flow::Epoch, evidence: &JsonMap) -> Result<()> {
        let segment_sha256 = evidence
            .get("segment_sha256")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("commit", "prepared segment hash is missing"))?;
        let schema_hash = evidence
            .get("schema_hash")
            .and_then(Value::as_str)
            .ok_or_else(|| fail("commit", "prepared schema hash is missing"))?;
        let payload = serde_json::to_vec(&BTreeMap::from([
            ("epoch", Value::from(epoch.as_u64())),
            (
                "transactional_id",
                Value::String(self.config.transactional_id.clone()),
            ),
            ("topic", Value::String(self.config.topic.clone())),
            (
                "format",
                Value::String(kafka_format_name(self.config.format).into()),
            ),
            ("schema_hash", Value::String(schema_hash.into())),
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
        let mut client = kafka_client_config(
            &self.config.bootstrap_servers,
            &self.config.security,
            self.password.as_ref(),
        )?;
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
        let admin: AdminClient<DefaultClientContext> = kafka_client_config(
            &self.config.bootstrap_servers,
            &self.config.security,
            self.password.as_ref(),
        )?
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
    topic: Option<String>,
    format: Option<String>,
    schema_hash: Option<String>,
    segment_sha256: String,
}

fn validate_ledger_marker_target(marker: &KafkaLedgerMarker, target: &str) -> Result<()> {
    match marker.topic.as_deref() {
        Some(recorded) if recorded == target => Ok(()),
        _ => Err(fail(
            "recover",
            "Kafka ledger marker target topic differs from this sink",
        )),
    }
}

fn validate_ledger_marker_schema(marker: &KafkaLedgerMarker, evidence: &JsonMap) -> Result<()> {
    if marker.schema_hash.as_deref() == evidence.get("schema_hash").and_then(Value::as_str)
        && marker.schema_hash.is_some()
    {
        Ok(())
    } else {
        Err(fail(
            "recover",
            "Kafka ledger marker schema differs from durable evidence",
        ))
    }
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
    live_schema_hash: Option<&str>,
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
    if crate::evidence::string_field(evidence, "topic").map_err(protocol)? != config.topic {
        return Err(fail(
            "recover",
            "prepared Kafka evidence names another target topic",
        ));
    }
    if crate::evidence::string_field(evidence, "format").map_err(protocol)?
        != kafka_format_name(config.format)
    {
        return Err(fail(
            "recover",
            "prepared Kafka evidence names another wire format",
        ));
    }
    crate::evidence::check_schema_hash(evidence).map_err(protocol)?;
    if live_schema_hash
        .is_some_and(|hash| evidence.get("schema_hash").and_then(Value::as_str) != Some(hash))
    {
        return Err(fail(
            "commit",
            "prepared Kafka schema differs from the active epoch",
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
        let completion = self
            .init_inflight
            .get_or_insert_with(|| start_transaction_init(&self.producer));
        let initialized = finish_transaction_init(completion).await;
        self.init_inflight = None;
        initialized?;
        self.preflight_ledger().await
    }

    async fn settle_open(&mut self) -> Result<()> {
        if let Some(mut completion) = self.init_inflight.take() {
            finish_transaction_init(&mut completion).await?;
        }
        Ok(())
    }

    async fn begin_epoch(&mut self, _epoch: calc_flow::Epoch) -> Result<()> {
        if self.active {
            return Err(fail(
                "begin_epoch",
                "a transaction is already active; the runtime owns epoch sequencing",
            ));
        }
        self.active = true;
        blocking_producer_call(&self.producer, &self.lifecycle, "begin_epoch", |producer| {
            producer.begin_transaction()
        })
        .await?;
        self.delivered = 0;
        self.pending_records.clear();
        self.pending_bytes = 0;
        self.pending_schema_hash = None;
        Ok(())
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        if !self.active {
            return Err(fail("write", "write before begin_epoch"));
        }
        let payload = encode_kafka_payload(self.config.format, batch)?;
        let table = batch
            .table_payload()
            .map_err(|_| fail("write", "the Kafka sink writes table batches only"))?;
        let schema_hash = kafka_schema_hash(self.config.format, Some(table.schema().as_ref()));
        if self
            .pending_schema_hash
            .as_ref()
            .is_some_and(|pending| pending != &schema_hash)
        {
            return Err(fail(
                "write",
                "all batches in one epoch must use the same Arrow schema",
            ));
        }
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
        self.pending_schema_hash = Some(schema_hash);
        let _ = delivery;
        Ok(())
    }

    async fn pre_commit(&mut self, epoch: calc_flow::Epoch) -> Result<JsonMap> {
        if !self.active {
            return Err(fail("pre_commit", "pre_commit before begin_epoch"));
        }
        blocking_producer_call(&self.producer, &self.lifecycle, "pre_commit", |producer| {
            producer.flush(Duration::from_secs(30))
        })
        .await?;
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
                "topic".to_string(),
                Value::String(self.config.topic.clone()),
            ),
            (
                "format".to_string(),
                Value::String(kafka_format_name(self.config.format).into()),
            ),
            (
                "schema_hash".to_string(),
                Value::String(
                    self.pending_schema_hash
                        .clone()
                        .unwrap_or_else(|| kafka_schema_hash(self.config.format, None)),
                ),
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
        let live_schema_hash = self
            .pending_schema_hash
            .clone()
            .unwrap_or_else(|| kafka_schema_hash(self.config.format, None));
        validate_prepared_evidence(
            &self.config,
            epoch,
            pre_commit,
            &self.pending_records,
            Some(&live_schema_hash),
        )?;
        self.write_ledger_marker(epoch, pre_commit).await?;
        blocking_producer_call(&self.producer, &self.lifecycle, "commit", |producer| {
            producer.commit_transaction(Duration::from_secs(30))
        })
        .await?;
        self.active = false;
        self.pending_records.clear();
        self.pending_bytes = 0;
        self.pending_schema_hash = None;
        Ok(())
    }

    async fn abort(
        &mut self,
        _epoch: calc_flow::Epoch,
        _pre_commit: Option<&JsonMap>,
    ) -> Result<()> {
        if self.active {
            blocking_producer_call(&self.producer, &self.lifecycle, "abort", |producer| {
                producer.abort_transaction(Duration::from_secs(30))
            })
            .await?;
            self.active = false;
        }
        self.pending_records.clear();
        self.pending_bytes = 0;
        self.pending_schema_hash = None;
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
            None,
        )?;
        if let Some(marker) = self.latest_ledger_marker().await? {
            validate_ledger_marker_target(&marker, &self.config.topic)?;
            if marker.format.as_deref() != Some(kafka_format_name(self.config.format)) {
                return Err(fail(
                    "recover",
                    "Kafka ledger marker wire format differs from this sink",
                ));
            }
            if !marker.schema_hash.as_ref().is_some_and(|hash| {
                hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit())
            }) {
                return Err(fail(
                    "recover",
                    "Kafka ledger marker schema hash is missing or invalid",
                ));
            }
            if marker.epoch > recovery.epoch().as_u64() {
                return Ok(());
            }
            if marker.epoch == recovery.epoch().as_u64() {
                validate_ledger_marker_schema(&marker, recovery.pre_commit())?;
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
        blocking_producer_call(&self.producer, &self.lifecycle, "recover", |producer| {
            producer.begin_transaction()
        })
        .await?;
        for payload in &records {
            let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(payload);
            self.producer
                .send(record, Duration::from_secs(10))
                .await
                .map_err(|(error, _)| fail("recover", &error.to_string()))?;
        }
        self.write_ledger_marker(recovery.epoch(), recovery.pre_commit())
            .await?;
        blocking_producer_call(&self.producer, &self.lifecycle, "recover", |producer| {
            producer.commit_transaction(Duration::from_secs(30))
        })
        .await
    }

    async fn close(&mut self) -> Result<()> {
        blocking_producer_call(&self.producer, &self.lifecycle, "close", |producer| {
            producer.flush(Duration::from_secs(30))
        })
        .await?;
        Ok(())
    }
}

/// Ordinary at-least-once Kafka sink for non-transactional plans.
pub struct OrdinaryKafkaSink {
    config: KafkaSinkConfig,
    producer: FutureProducer,
    lifecycle: OnceLock<ProducerLifecycle>,
    sequence: u64,
}

impl OrdinaryKafkaSink {
    /// Builds the idempotent producer.
    ///
    /// # Errors
    ///
    /// Returns the connector error when the producer cannot be created.
    pub fn new(config: KafkaSinkConfig) -> Result<Self> {
        Self::new_with_password(config, None)
    }

    fn new_with_password(config: KafkaSinkConfig, password: Option<&SecretHandle>) -> Result<Self> {
        let mut client =
            kafka_client_config(&config.bootstrap_servers, &config.security, password)?;
        client.set("enable.idempotence", "true");
        client.set("message.timeout.ms", "30000");
        let producer: FutureProducer = client
            .create()
            .map_err(|error| fail("open", &error.to_string()))?;
        Ok(Self {
            config,
            producer,
            lifecycle: OnceLock::new(),
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
        let payload = encode_kafka_payload(self.config.format, batch)?;
        self.sequence += 1;
        let record = FutureRecord::<Vec<u8>, Vec<u8>>::to(&self.config.topic).payload(&payload);
        self.producer
            .send(record, Duration::from_secs(10))
            .await
            .map_err(|(error, _message)| fail("write", &error.to_string()))?;
        Ok(())
    }

    async fn close(&mut self) -> Result<()> {
        blocking_producer_call(&self.producer, &self.lifecycle, "close", |producer| {
            producer.flush(Duration::from_secs(30))
        })
        .await?;
        Ok(())
    }
}

use std::collections::BTreeSet;
use std::sync::Arc;

use calc_flow::{
    ConnectorCapabilities, ConnectorDescriptor, ConnectorFactories, ConnectorKind,
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, DeliveryCapability,
    FormatDescriptor, FormatIdentity, TransactionSupport, WatermarkSupport,
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
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = KafkaSourceConfig::from_options(options)?;
        let password = config.security.resolve_password(secrets)?;
        Ok(Box::new(KafkaSource::with_decoders_and_password(
            config,
            &self.decoders,
            password.as_ref(),
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
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>> {
        let config = KafkaSinkConfig::from_options(options)?;
        let password = config.security.resolve_password(secrets)?;
        Ok(Box::new(OrdinaryKafkaSink::new_with_password(
            config,
            password.as_ref(),
        )?))
    }

    async fn open_transactional(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        let config = KafkaSinkConfig::from_options(options)?;
        let password = config.security.resolve_password(secrets)?;
        Ok(Some(Box::new(TransactionalKafkaSink::new_with_password(
            config, password,
        )?)))
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
            ("security_protocol".to_string(), serde_json::json!("string")),
            ("ssl_ca_location".to_string(), serde_json::json!("string")),
            ("sasl_mechanism".to_string(), serde_json::json!("string")),
            ("sasl_username".to_string(), serde_json::json!("string")),
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
        secret_slots: ["sasl_password".to_string()].into_iter().collect(),
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

    #[tokio::test]
    async fn producer_lifecycle_call_runs_off_the_async_worker() {
        let producer: FutureProducer = rdkafka::config::ClientConfig::new()
            .set("bootstrap.servers", "127.0.0.1:1")
            .create()
            .unwrap();
        let async_worker = std::thread::current().id();
        let lifecycle = OnceLock::new();
        blocking_producer_call(&producer, &lifecycle, "test", move |_| {
            assert_ne!(std::thread::current().id(), async_worker);
            Ok(())
        })
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn dropped_lifecycle_wait_keeps_late_begin_before_abort() {
        let producer: FutureProducer = rdkafka::config::ClientConfig::new()
            .set("bootstrap.servers", "127.0.0.1:1")
            .create()
            .unwrap();
        let lifecycle = OnceLock::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (started, started_rx) = std::sync::mpsc::channel();
        let begin_order = Arc::clone(&order);
        let mut begin = Box::pin(blocking_producer_call(
            &producer,
            &lifecycle,
            "begin",
            move |_| {
                started.send(()).unwrap();
                std::thread::sleep(Duration::from_millis(50));
                begin_order.lock().unwrap().push("begin");
                Ok(())
            },
        ));
        assert!(
            tokio::time::timeout(Duration::from_millis(1), begin.as_mut())
                .await
                .is_err()
        );
        started_rx.recv().unwrap();
        drop(begin);
        let abort_order = Arc::clone(&order);
        blocking_producer_call(&producer, &lifecycle, "abort", move |_| {
            abort_order.lock().unwrap().push("abort");
            Ok(())
        })
        .await
        .unwrap();
        assert_eq!(&*order.lock().unwrap(), &["begin", "abort"]);
    }

    #[tokio::test]
    async fn cancelled_open_retains_native_init_until_settled() {
        let config = KafkaSinkConfig::from_options(&sink_options("json")).unwrap();
        let mut sink = TransactionalKafkaSink::new(config).unwrap();
        let (response, completion) = tokio::sync::oneshot::channel();
        sink.init_inflight = Some(completion);
        {
            let settlement = sink.settle_open();
            tokio::pin!(settlement);
            tokio::select! {
                result = &mut settlement => panic!("init settled before its native worker: {result:?}"),
                () = tokio::task::yield_now() => {}
            }
            response.send(Ok(())).unwrap();
            settlement.await.unwrap();
        }
        assert!(sink.init_inflight.is_none());
    }

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
    fn decode_failures_carry_record_coordinates_without_payload() {
        let decoders = KafkaDecoderRegistry::default();
        let config = KafkaSourceConfig::from_options(&source_options("json")).unwrap();
        let decoder = config.decoder(&decoders).unwrap();
        let poison = b"{\"id\":\"sentinel-leak-marker\",\"label\":\"one\"}\n";
        let error = decoder
            .decode(poison, &config)
            .expect_err("a poison record fails to decode");
        let wrapped = decode_failure(&config.topic, 3, 42, error);
        let message = wrapped.to_string();
        assert!(message.contains("topic \"events\""), "{message}");
        assert!(message.contains("partition 3"), "{message}");
        assert!(message.contains("offset 42"), "{message}");
        assert!(!message.contains("sentinel-leak-marker"), "{message}");
        assert!(matches!(wrapped, CalcFlowError::Connector(_)), "{wrapped}");
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

    fn sink_options(format: &str) -> JsonMap {
        BTreeMap::from([
            (
                "bootstrap_servers".into(),
                Value::String("127.0.0.1:1".into()),
            ),
            ("topic".into(), Value::String("events".into())),
            ("ledger_topic".into(), Value::String("events-ledger".into())),
            ("pipeline".into(), Value::String("orders".into())),
            ("output".into(), Value::String("events".into())),
            ("format".into(), Value::String(format.into())),
        ])
    }

    #[test]
    fn prepared_recovery_evidence_binds_target_topic() {
        let config = KafkaSinkConfig::from_options(&sink_options("json")).expect("config");
        let records = vec![b"one".to_vec()];
        let segment = encode_records(&records).expect("segment");
        let evidence = BTreeMap::from([
            (
                "transactional_id".into(),
                Value::String(config.transactional_id.clone()),
            ),
            (
                "epoch".into(),
                Value::from(calc_flow::Epoch::INITIAL.as_u64()),
            ),
            (
                "ledger_topic".into(),
                Value::String(config.ledger_topic.clone()),
            ),
            ("topic".into(), Value::String(config.topic.clone())),
            ("format".into(), Value::String("json".into())),
            ("schema_hash".into(), Value::String("a".repeat(64))),
            (
                "segment_id".into(),
                Value::String(PREPARED_RECORDS_SEGMENT.into()),
            ),
            ("segment_bytes".into(), Value::from(segment.len() as u64)),
            (
                "segment_sha256".into(),
                Value::String(hex::encode(Sha256::digest(&segment))),
            ),
        ]);
        validate_prepared_evidence(
            &config,
            calc_flow::Epoch::INITIAL,
            &evidence,
            &records,
            Some(&"a".repeat(64)),
        )
        .expect("matching topic is recoverable");
        let mut wrong_schema = evidence.clone();
        wrong_schema.insert("schema_hash".into(), Value::String("b".repeat(64)));
        assert!(
            validate_prepared_evidence(
                &config,
                calc_flow::Epoch::INITIAL,
                &wrong_schema,
                &records,
                Some(&"a".repeat(64)),
            )
            .is_err(),
            "live evidence cannot claim another Arrow schema"
        );
        let mut foreign = evidence.clone();
        foreign.insert("topic".into(), Value::String("other-topic".into()));
        assert!(
            validate_prepared_evidence(
                &config,
                calc_flow::Epoch::INITIAL,
                &foreign,
                &records,
                None
            )
            .is_err(),
            "another target topic must fail closed"
        );
        let mut missing = evidence;
        missing.remove("topic");
        assert!(
            validate_prepared_evidence(
                &config,
                calc_flow::Epoch::INITIAL,
                &missing,
                &records,
                None
            )
            .is_err(),
            "missing target topic must fail closed"
        );
        missing.insert("topic".into(), Value::String(config.topic.clone()));
        missing.insert("format".into(), Value::String("csv".into()));
        assert!(
            validate_prepared_evidence(
                &config,
                calc_flow::Epoch::INITIAL,
                &missing,
                &records,
                None
            )
            .is_err(),
            "a different wire format must fail closed"
        );
    }

    #[test]
    fn ledger_marker_binds_target_topic() {
        let marker = |topic: Option<&str>| {
            let mut payload = serde_json::json!({
                "epoch": 1,
                "transactional_id": "calc-flow-test",
                "schema_hash": "a".repeat(64),
                "segment_sha256": "a".repeat(64),
            });
            if let Some(topic) = topic {
                payload["topic"] = Value::String(topic.into());
            }
            serde_json::from_value::<KafkaLedgerMarker>(payload).expect("marker parses")
        };
        validate_ledger_marker_target(&marker(Some("events")), "events")
            .expect("matching target recovers");
        assert!(validate_ledger_marker_target(&marker(Some("other")), "events").is_err());
        assert!(validate_ledger_marker_target(&marker(None), "events").is_err());
        let expected = BTreeMap::from([("schema_hash".into(), Value::String("a".repeat(64)))]);
        validate_ledger_marker_schema(&marker(Some("events")), &expected)
            .expect("matching schema recovers");
        let changed = BTreeMap::from([("schema_hash".into(), Value::String("b".repeat(64)))]);
        assert!(validate_ledger_marker_schema(&marker(Some("events")), &changed).is_err());
    }

    #[test]
    fn schema_hash_distinguishes_equal_wire_payloads() {
        use arrow::datatypes::{DataType, Field, Schema};

        let narrow = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let wide = Schema::new(vec![Field::new("a", DataType::Int64, false)]);
        assert_ne!(
            kafka_schema_hash(KafkaFormat::Json, Some(&narrow)),
            kafka_schema_hash(KafkaFormat::Json, Some(&wide)),
        );
    }

    #[test]
    fn sasl_tls_options_use_a_secret_and_configure_librdkafka() {
        let mut options = sink_options("json");
        options.insert("security_protocol".into(), Value::String("sasl_ssl".into()));
        options.insert(
            "sasl_mechanism".into(),
            Value::String("SCRAM-SHA-512".into()),
        );
        options.insert("sasl_username".into(), Value::String("worker".into()));
        options.insert(
            "ssl_ca_location".into(),
            Value::String("/tmp/ca.pem".into()),
        );
        calc_flow::validate_connector_options(&kafka_connector_descriptor(), &options)
            .expect("security options are declared");
        let config = KafkaSinkConfig::from_options(&options).expect("SASL over TLS parses");
        let password = SecretHandle::from_bytes(b"private-password");
        let client =
            kafka_client_config(&config.bootstrap_servers, &config.security, Some(&password))
                .expect("resolved password configures client");
        assert_eq!(client.get("security.protocol"), Some("sasl_ssl"));
        assert_eq!(client.get("sasl.mechanism"), Some("SCRAM-SHA-512"));
        assert_eq!(client.get("sasl.username"), Some("worker"));
        assert_eq!(client.get("sasl.password"), Some("private-password"));
        assert_eq!(client.get("ssl.ca.location"), Some("/tmp/ca.pem"));
        assert!(kafka_client_config(&config.bootstrap_servers, &config.security, None).is_err());

        options.insert("sasl_password".into(), Value::String("literal".into()));
        assert!(
            calc_flow::validate_connector_options(&kafka_connector_descriptor(), &options).is_err(),
            "passwords are secret references, never project options"
        );
        assert!(KafkaSinkConfig::from_options(&options).is_err());
    }

    #[test]
    fn security_options_reject_incomplete_or_incompatible_combinations() {
        let mut options = sink_options("json");
        options.insert("security_protocol".into(), Value::String("sasl_ssl".into()));
        assert!(KafkaSinkConfig::from_options(&options).is_err());
        options.insert("sasl_mechanism".into(), Value::String("PLAIN".into()));
        options.insert("sasl_username".into(), Value::String("worker".into()));
        assert!(KafkaSinkConfig::from_options(&options).is_ok());
        options.insert(
            "security_protocol".into(),
            Value::String("plaintext".into()),
        );
        assert!(KafkaSinkConfig::from_options(&options).is_err());
        options.remove("sasl_mechanism");
        options.remove("sasl_username");
        options.insert(
            "ssl_ca_location".into(),
            Value::String("/tmp/ca.pem".into()),
        );
        assert!(KafkaSinkConfig::from_options(&options).is_err());

        let mut public_config = KafkaSinkConfig::from_options(&sink_options("json")).unwrap();
        public_config.security.protocol = KafkaSecurityProtocol::SaslSsl;
        assert!(OrdinaryKafkaSink::new(public_config).is_err());
    }

    struct PasswordResolver;

    impl SecretResolver for PasswordResolver {
        fn has_reference(&self, reference: &SecretReference) -> Option<bool> {
            Some(reference.key == "sasl_password")
        }

        fn resolve(&self, reference: &SecretReference) -> Result<SecretHandle> {
            assert_eq!(reference.key, "sasl_password");
            Ok(SecretHandle::from_bytes(b"private-password"))
        }
    }

    struct UnresolvedPasswordReference;

    impl SecretResolver for UnresolvedPasswordReference {
        fn has_reference(&self, reference: &SecretReference) -> Option<bool> {
            Some(reference.key == "sasl_password")
        }

        fn resolve(&self, _reference: &SecretReference) -> Result<SecretHandle> {
            Err(CalcFlowError::NotFound {
                resource: "secret".into(),
                key: "sasl_password".into(),
            })
        }
    }

    struct UnknownPasswordReference;

    impl SecretResolver for UnknownPasswordReference {
        fn resolve(&self, _reference: &SecretReference) -> Result<SecretHandle> {
            Err(CalcFlowError::NotFound {
                resource: "secret".into(),
                key: "sasl_password".into(),
            })
        }
    }

    #[tokio::test]
    async fn factories_resolve_sasl_password_for_source_and_sink() {
        let security = [
            ("security_protocol".into(), Value::String("sasl_ssl".into())),
            ("sasl_mechanism".into(), Value::String("PLAIN".into())),
            ("sasl_username".into(), Value::String("worker".into())),
        ];
        let mut source = source_options("json");
        source.extend(security.clone());
        KafkaSourceFactory::new()
            .open(&source, &PasswordResolver)
            .await
            .expect("source factory uses password secret");

        let mut sink = sink_options("json");
        sink.extend(security);
        KafkaSinkFactory::new()
            .open(&sink, &PasswordResolver)
            .await
            .expect("sink factory uses password secret");
    }

    #[tokio::test]
    async fn plaintext_binding_rejects_supplied_password_secret() {
        let error = KafkaSinkFactory::new()
            .open(&sink_options("json"), &PasswordResolver)
            .await
            .err()
            .expect("plaintext must reject a supplied SASL password");
        assert!(error.to_string().contains("sasl_password"), "{error}");
        let error = KafkaSinkFactory::new()
            .open(&sink_options("json"), &UnresolvedPasswordReference)
            .await
            .err()
            .expect("an unresolved declared password still blocks plaintext");
        assert!(error.to_string().contains("sasl_password"), "{error}");
        let error = KafkaSinkFactory::new()
            .open(&sink_options("json"), &UnknownPasswordReference)
            .await
            .err()
            .expect("unknown secret presence cannot allow plaintext");
        assert!(error.to_string().contains("sasl_password"), "{error}");
    }

    fn sample_batch() -> Batch {
        use arrow::array::Int64Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let record = RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 2]))])
            .expect("record batch");
        Batch::table(
            vec![record],
            calc_flow::BatchMetadata::new("test", 1, BTreeMap::new()).unwrap(),
        )
        .unwrap()
    }

    #[test]
    fn sinks_reject_the_source_only_protobuf_format() {
        for format in ["protobuf", "custom"] {
            let error = KafkaSinkConfig::from_options(&sink_options(format))
                .expect_err("sinks cannot encode source-only payloads");
            assert!(error.to_string().contains("format"), "{format}: {error}");
        }
    }

    #[test]
    fn sink_parsing_and_encoding_share_one_format_guard() {
        for format in ["protobuf", "custom"] {
            let parse_error = KafkaSinkConfig::from_options(&sink_options(format))
                .expect_err("source-only formats are rejected at parse");
            let parsed = KafkaFormat::parse(format).expect("known format");
            let encode_error = encode_kafka_payload(parsed, &sample_batch())
                .expect_err("source-only formats are rejected at encode");
            let parse_message = parse_error.to_string();
            let encode_message = encode_error.to_string();
            assert!(
                parse_message.contains(SINK_FORMAT_MESSAGE),
                "{format}: {parse_message}"
            );
            assert!(
                encode_message.contains(SINK_FORMAT_MESSAGE),
                "{format}: {encode_message}"
            );
        }

        let json = encode_kafka_payload(KafkaFormat::Json, &sample_batch())
            .expect("json encodes through the shared helper");
        assert_eq!(json, b"{\"a\":1}\n{\"a\":2}\n".to_vec());
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
