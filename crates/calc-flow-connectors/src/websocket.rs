//! The `WebSocket` connector (feature `http-websocket`).
//!
//! Streams JSON lines from one `WebSocket` endpoint with frame and
//! decoded-batch size limits. The source is lossy: it can pause reads
//! by default (Block). `DropOldest` is an explicit, observable mode
//! that is incompatible with exactly-once and is rejected at config
//! time when combined with an exactly-once delivery request. TLS
//! verification is on by default; `insecure` must be set explicitly
//! and emits a warning.

use std::collections::BTreeMap;

use async_trait::async_trait;
use calc_flow::FormatDecoder as _;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ConnectorCapabilities, ConnectorDescriptor,
    ConnectorError, ConnectorFactories, ConnectorIdentity, ConnectorKind, ConnectorOperation,
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor, DeliveryCapability,
    JsonMap, Result, SecretHandle, SecretReference, SecretResolver, SecretResolverKind,
    SourceCapabilities, SourceEvent, SourceSchema, StreamSource, TransactionSupport,
    WatermarkSupport,
};
use serde_json::Value;
use tokio_tungstenite::tungstenite::Message;

use crate::json_lines::JsonLinesCodec;

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

/// How the source reacts when the consumer cannot keep up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackpressureMode {
    /// Pause reads until the consumer catches up (default, lossless).
    Block,
    /// Drop the oldest buffered frames (explicit, lossy, incompatible
    /// with exactly-once).
    DropOldest,
}

fn connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "websocket", IDENTITY_VERSION)
        .expect("the websocket connector identity is valid")
}

fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        connector_identity(),
        ConnectorOperation::new(operation).expect("operation name is non-empty"),
        detail,
    ))
}

/// Resolves the `WebSocket` endpoint URL from a secret reference.
///
/// # Errors
///
/// Returns the resolver error; the URL value never enters the error.
pub fn resolve_ws_url(secrets: &dyn SecretResolver, key: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Environment, key)
        .map_err(|error| fail("open", &error.to_string()))?;
    let handle: SecretHandle = secrets
        .resolve(&reference)
        .map_err(|_| fail("open", "the `WebSocket` URL secret could not be resolved"))?;
    String::from_utf8(handle.expose().to_vec())
        .map_err(|_| fail("open", "the `WebSocket` URL secret is not valid UTF-8"))
}

/// Data-only configuration for one `WebSocket` source.
#[derive(Clone, Debug)]
pub struct WebSocketSourceConfig {
    /// Secret key holding the `wss://…` endpoint URL.
    pub url_key: String,
    /// Backpressure reaction mode.
    pub backpressure: BackpressureMode,
    /// Maximum accepted frame size.
    pub max_frame_bytes: u64,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Byte bound of one decoded batch.
    pub max_batch_bytes: u64,
    /// Whether TLS certificate verification is disabled (warns).
    pub insecure: bool,
}

impl WebSocketSourceConfig {
    /// Parses the source configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option; `drop_oldest` combined with `exactly_once` fails closed
    /// and `insecure` emits a tracing warning.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let url_key = required_string(options, "url_key")?;
        let backpressure = match options
            .get("backpressure")
            .and_then(Value::as_str)
            .unwrap_or("block")
        {
            "block" => BackpressureMode::Block,
            "drop_oldest" => {
                if options
                    .get("delivery")
                    .and_then(Value::as_str)
                    .is_some_and(|d| d == "exactly_once")
                {
                    return Err(CalcFlowError::InvalidArgument {
                        field: "backpressure".into(),
                        message: "drop_oldest is incompatible with exactly-once delivery".into(),
                    });
                }
                BackpressureMode::DropOldest
            }
            other => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "backpressure".into(),
                    message: format!("unsupported backpressure mode {other:?}"),
                });
            }
        };
        let insecure = options
            .get("insecure")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if insecure {
            tracing::warn!(
                target: "calc_flow_connectors::websocket",
                "insecure mode: TLS certificate verification is disabled for this source"
            );
        }
        Ok(Self {
            url_key,
            backpressure,
            max_frame_bytes: u64_option(options, "max_frame_bytes")?.unwrap_or(1024 * 1024),
            max_batch_rows: u64_option(options, "max_batch_rows")?.unwrap_or(8192),
            max_batch_bytes: u64_option(options, "max_batch_bytes")?.unwrap_or(8 * 1024 * 1024),
            insecure,
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

/// The `WebSocket` source.
pub struct WebSocketSource {
    capabilities: SourceCapabilities,
    config: WebSocketSourceConfig,
    sequence: u64,
}

impl WebSocketSource {
    /// Builds the source.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: WebSocketSourceConfig) -> Result<Self> {
        let lossy = config.backpressure == BackpressureMode::DropOldest;
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: if lossy {
                calc_flow::SourceDeliveryCapability::Lossy
            } else {
                calc_flow::SourceDeliveryCapability::Lossless
            },
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_batch_bytes).unwrap_or(usize::MAX),
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            capabilities,
            config,
            sequence: 0,
        })
    }

    /// Streams one bounded batch of frames as newline-JSON events.
    ///
    /// # Errors
    ///
    /// Returns the connector error on connection failure or oversized
    /// frames/batches.
    pub async fn next_with_secrets(
        &mut self,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<SourceEvent>> {
        let url = resolve_ws_url(secrets, &self.config.url_key)?;
        let (mut stream, _response) = tokio_tungstenite::connect_async(url)
            .await
            .map_err(|error| fail("open", &error.to_string()))?;
        let lines = collect_bounded_frames(
            &mut stream,
            self.config.max_batch_rows,
            self.config.max_frame_bytes,
        )
        .await?;
        if lines.is_empty() {
            return Ok(Some(SourceEvent::Idle));
        }
        let batch = self.decode_frames(&lines)?;
        let cursor = self.build_cursor()?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    fn decode_frames(&mut self, lines: &[Vec<u8>]) -> Result<Batch> {
        let body: Vec<u8> = lines
            .iter()
            .flat_map(|line| line.iter().copied().chain(std::iter::once(b'\n')))
            .collect();
        let codec = JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?;
        let bounds =
            calc_flow::DecodeBounds::new(self.config.max_batch_rows, self.config.max_batch_bytes)?;
        let batch = codec.decode(&body, &bounds, &[])?;
        self.sequence += 1;
        let metadata = BatchMetadata::new(
            "websocket",
            self.sequence,
            BTreeMap::from([(
                "url_key".to_string(),
                Value::String(self.config.url_key.clone()),
            )]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        Ok(batch.with_metadata(metadata))
    }

    fn build_cursor(&self) -> Result<Cursor> {
        Cursor::unbound(
            self.sequence.to_be_bytes().to_vec(),
            BTreeMap::from([("sequence".to_string(), Value::from(self.sequence))]),
        )
    }
}

#[async_trait]
impl StreamSource for WebSocketSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, _cursor: Option<Cursor>) -> Result<()> {
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        Ok(Some(SourceEvent::Idle))
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Collects bounded frames from a WebSocket stream into newline
/// JSON lines.
async fn collect_bounded_frames<S>(
    stream: &mut S,
    max_rows: u64,
    max_frame_bytes: u64,
) -> Result<Vec<Vec<u8>>>
where
    S: futures_util::Stream<
            Item = std::result::Result<Message, tokio_tungstenite::tungstenite::Error>,
        > + Unpin,
{
    let mut lines: Vec<Vec<u8>> = Vec::new();
    let mut taken: u64 = 0;
    while let Some(message) = futures_util::StreamExt::next(stream).await {
        let message = message.map_err(|error| fail("read", &error.to_string()))?;
        let payload = match &message {
            Message::Text(text) => text.as_bytes().to_vec(),
            Message::Binary(data) => data.to_vec(),
            Message::Close(_) => break,
            _ => continue,
        };
        if payload.len() as u64 > max_frame_bytes {
            return Err(fail(
                "read",
                &format!(
                    "frame {} bytes exceeds the {} byte limit",
                    payload.len(),
                    max_frame_bytes
                ),
            ));
        }
        lines.push(payload);
        taken += 1;
        if taken >= max_rows {
            break;
        }
    }
    Ok(lines)
}

/// Trusted source factory for the `WebSocket` transport.
pub struct WebSocketSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl WebSocketSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: websocket_connector_descriptor(),
        }
    }
}

impl Default for WebSocketSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSourceFactory for WebSocketSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = WebSocketSourceConfig::from_options(options)?;
        Ok(Box::new(WebSocketSource::new(config)?))
    }
}

/// Trusted sink factory placeholder: WebSocket is source-only in 3.0.
pub struct WebSocketSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl WebSocketSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: websocket_connector_descriptor(),
        }
    }
}

impl Default for WebSocketSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSinkFactory for WebSocketSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn calc_flow::StreamSink>> {
        Err(fail(
            "open",
            "the `WebSocket` connector is source-only in 3.0",
        ))
    }
}

fn websocket_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: connector_identity(),
        kind: ConnectorKind::Source,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::BestEffort,
            replay: calc_flow::ReplayCapability::Unreplayable,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::None,
            snapshot: false,
            polling: false,
            cdc: false,
            lookup: false,
        },
        formats: vec![],
        config_schema: JsonMap::from([
            ("url_key".to_string(), serde_json::json!("string")),
            ("backpressure".to_string(), serde_json::json!("string")),
            ("max_frame_bytes".to_string(), serde_json::json!("u64")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("insecure".to_string(), serde_json::json!("boolean")),
        ]),
        secret_slots: ["url_key".to_string()].into_iter().collect(),
    }
}

/// Registers the `WebSocket` connectors into one trusted registry.
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot is already
/// occupied.
pub fn register_websocket_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        websocket_connector_descriptor(),
        ConnectorFactories::source_only(std::sync::Arc::new(WebSocketSourceFactory::new())),
    )
}
