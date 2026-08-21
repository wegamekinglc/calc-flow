//! The `WebSocket` connector (feature `websocket`).
//!
//! Streams JSON lines from one `WebSocket` endpoint with frame and
//! decoded-batch size limits. The source is unreplayable and therefore
//! best-effort across process failure; it pauses live reads by default
//! (`Block`). `DropOldest` is an explicit, observable mode
//! that is incompatible with exactly-once and is rejected at config
//! time when combined with an exactly-once delivery request. TLS
//! verification is on by default; `insecure` must be set explicitly
//! and emits a warning.

use std::{
    collections::{BTreeMap, VecDeque},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Duration,
};

use async_trait::async_trait;
use calc_flow::FormatDecoder as _;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ConnectorCapabilities, ConnectorDescriptor,
    ConnectorError, ConnectorFactories, ConnectorIdentity, ConnectorKind, ConnectorOperation,
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor, DeliveryCapability,
    FormatIdentity, JsonMap, Result, SecretHandle, SecretReference, SecretResolver,
    SecretResolverKind, SourceCapabilities, SourceEvent, SourceSchema, StreamSource,
    TransactionSupport, WatermarkSupport,
};
use serde_json::Value;
use tokio::{
    sync::{Mutex, Notify},
    task::JoinHandle,
};
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream, tungstenite::Message};

use crate::json_lines::JsonLinesCodec;

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

/// How the source reacts when the consumer cannot keep up.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BackpressureMode {
    /// Pause live reads until the consumer catches up (default).
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
pub fn resolve_ws_url(secrets: &dyn SecretResolver, slot: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Registered, slot)
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
        if options.contains_key("url_key") {
            return Err(CalcFlowError::InvalidArgument {
                field: "options".into(),
                message: "the endpoint URL must use a secret reference".into(),
            });
        }
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
        if backpressure == BackpressureMode::Block
            && options
                .get("delivery")
                .and_then(Value::as_str)
                .is_some_and(|delivery| delivery == "exactly_once")
        {
            return Err(CalcFlowError::InvalidArgument {
                field: "delivery".into(),
                message: "WebSocket is unreplayable and cannot provide exactly-once delivery"
                    .into(),
            });
        }
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
        Self {
            backpressure,
            max_frame_bytes: positive_option(options, "max_frame_bytes", 1024 * 1024)?,
            max_batch_rows: positive_option(options, "max_batch_rows", 8192)?,
            max_batch_bytes: positive_option(options, "max_batch_bytes", 8 * 1024 * 1024)?,
            insecure,
        }
        .validate_bounds()
    }

    fn validate_bounds(self) -> Result<Self> {
        if self.max_frame_bytes >= self.max_batch_bytes {
            return Err(CalcFlowError::InvalidArgument {
                field: "max_frame_bytes".into(),
                message: "must be smaller than max_batch_bytes so the JSON line delimiter fits"
                    .into(),
            });
        }
        Ok(self)
    }
}

fn positive_option(options: &JsonMap, key: &str, default: u64) -> Result<u64> {
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
    endpoint_url: Option<String>,
    queue: Option<Arc<FrameQueue>>,
    reader: Option<JoinHandle<()>>,
}

impl WebSocketSource {
    /// Builds the source.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn new(config: WebSocketSourceConfig) -> Result<Self> {
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::Unsupported,
            delivery: calc_flow::SourceDeliveryCapability::Lossy,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_batch_bytes).unwrap_or(usize::MAX),
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            capabilities,
            config,
            sequence: 0,
            endpoint_url: None,
            queue: None,
            reader: None,
        })
    }

    fn with_endpoint(mut self, endpoint_url: String) -> Self {
        self.endpoint_url = Some(endpoint_url);
        self
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
        if self.endpoint_url.is_none() {
            self.endpoint_url = Some(resolve_ws_url(secrets, "url")?);
        }
        if self.reader.is_none() {
            self.open(None).await?;
        }
        self.next().await
    }

    fn decode_frames(&mut self, lines: &[Vec<u8>], dropped_frames: u64) -> Result<Batch> {
        let body: Vec<u8> = lines
            .iter()
            .flat_map(|line| line.iter().copied().chain(std::iter::once(b'\n')))
            .collect();
        let codec = JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?;
        let bounds =
            calc_flow::DecodeBounds::new(self.config.max_batch_rows, self.config.max_batch_bytes)?;
        let batch = codec.decode(&body, &bounds, &[])?;
        self.sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| fail("read", "WebSocket cursor sequence exhausted u64"))?;
        let metadata = BatchMetadata::new(
            "websocket",
            self.sequence,
            BTreeMap::from([("dropped_frames".to_string(), Value::from(dropped_frames))]),
        )
        .map_err(|error| fail("read", &error.to_string()))?;
        Ok(batch.with_metadata(metadata))
    }

    fn build_cursor(&self, dropped_frames: u64) -> Result<Cursor> {
        Cursor::unbound(
            self.sequence.to_be_bytes().to_vec(),
            BTreeMap::from([
                ("sequence".to_string(), Value::from(self.sequence)),
                ("dropped_frames".to_string(), Value::from(dropped_frames)),
            ]),
        )
    }
}

#[async_trait]
impl StreamSource for WebSocketSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if cursor.is_some() {
            return Err(fail(
                "open",
                "WebSocket sources cannot restore from a cursor",
            ));
        }
        if self.reader.is_some() {
            return Err(fail("open", "WebSocket source is already open"));
        }
        let url = self.endpoint_url.as_deref().ok_or_else(|| {
            fail(
                "open",
                "the `WebSocket` source was not opened through its trusted factory",
            )
        })?;
        let stream = connect_websocket(url, self.config.insecure).await?;
        let queue = Arc::new(FrameQueue::new(&self.config)?);
        self.reader = Some(tokio::spawn(read_frames(stream, Arc::clone(&queue))));
        self.queue = Some(queue);
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        let queue = self.queue.as_ref().ok_or_else(|| {
            fail(
                "read",
                "the `WebSocket` source must be opened before it is read",
            )
        })?;
        let Some((frames, dropped_frames)) = queue.take_batch().await? else {
            return Ok(None);
        };
        let batch = self.decode_frames(&frames, dropped_frames)?;
        let cursor = self.build_cursor(dropped_frames)?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    async fn close(&mut self) -> Result<()> {
        if let Some(queue) = self.queue.take() {
            queue.cancel();
        }
        if let Some(reader) = self.reader.take() {
            reader
                .await
                .map_err(|error| fail("close", &format!("reader task failed: {error}")))?;
        }
        Ok(())
    }
}

type Socket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

async fn connect_websocket(url: &str, insecure: bool) -> Result<Socket> {
    let connected = if insecure {
        let mut tls = rustls::ClientConfig::builder()
            .with_root_certificates(rustls::RootCertStore::empty())
            .with_no_client_auth();
        // This bypass is reachable only through the explicit `insecure: true`
        // connector option, which is off by default and emits a warning.
        tls.dangerous() // nosemgrep: rust.lang.security.rustls-dangerous.rustls-dangerous
            .set_certificate_verifier(Arc::new(InsecureCertificateVerifier));
        tokio_tungstenite::connect_async_tls_with_config(
            url,
            None,
            false,
            Some(tokio_tungstenite::Connector::Rustls(Arc::new(tls))),
        )
        .await
    } else {
        tokio_tungstenite::connect_async(url).await
    };
    connected
        .map(|(stream, _)| stream)
        .map_err(|_| fail("open", "WebSocket connection failed"))
}

#[derive(Debug)]
struct InsecureCertificateVerifier;

impl rustls::client::danger::ServerCertVerifier for InsecureCertificateVerifier {
    fn verify_server_cert(
        &self,
        _end_entity: &rustls::pki_types::CertificateDer<'_>,
        _intermediates: &[rustls::pki_types::CertificateDer<'_>],
        _server_name: &rustls::pki_types::ServerName<'_>,
        _ocsp: &[u8],
        _now: rustls::pki_types::UnixTime,
    ) -> std::result::Result<rustls::client::danger::ServerCertVerified, rustls::Error> {
        Ok(rustls::client::danger::ServerCertVerified::assertion())
    }

    fn verify_tls12_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn verify_tls13_signature(
        &self,
        _message: &[u8],
        _cert: &rustls::pki_types::CertificateDer<'_>,
        _dss: &rustls::DigitallySignedStruct,
    ) -> std::result::Result<rustls::client::danger::HandshakeSignatureValid, rustls::Error> {
        Ok(rustls::client::danger::HandshakeSignatureValid::assertion())
    }

    fn supported_verify_schemes(&self) -> Vec<rustls::SignatureScheme> {
        vec![
            rustls::SignatureScheme::ECDSA_NISTP256_SHA256,
            rustls::SignatureScheme::ECDSA_NISTP384_SHA384,
            rustls::SignatureScheme::ED25519,
            rustls::SignatureScheme::RSA_PKCS1_SHA256,
            rustls::SignatureScheme::RSA_PKCS1_SHA384,
            rustls::SignatureScheme::RSA_PKCS1_SHA512,
            rustls::SignatureScheme::RSA_PSS_SHA256,
            rustls::SignatureScheme::RSA_PSS_SHA384,
            rustls::SignatureScheme::RSA_PSS_SHA512,
        ]
    }
}

struct FrameQueue {
    mode: BackpressureMode,
    max_rows: usize,
    max_bytes: usize,
    max_frame_bytes: usize,
    state: Mutex<FrameQueueState>,
    data_ready: Notify,
    space_ready: Notify,
    cancel_ready: Notify,
    cancelled: AtomicBool,
}

#[derive(Default)]
struct FrameQueueState {
    frames: VecDeque<Vec<u8>>,
    bytes: usize,
    dropped_frames: u64,
    terminal: Option<QueueTerminal>,
}

enum QueueTerminal {
    Closed,
    Failed(String),
}

impl FrameQueue {
    fn new(config: &WebSocketSourceConfig) -> Result<Self> {
        Ok(Self {
            mode: config.backpressure,
            max_rows: usize::try_from(config.max_batch_rows)
                .map_err(|_| fail("open", "max_batch_rows does not fit this platform"))?,
            max_bytes: usize::try_from(config.max_batch_bytes)
                .map_err(|_| fail("open", "max_batch_bytes does not fit this platform"))?,
            max_frame_bytes: usize::try_from(config.max_frame_bytes)
                .map_err(|_| fail("open", "max_frame_bytes does not fit this platform"))?,
            state: Mutex::new(FrameQueueState::default()),
            data_ready: Notify::new(),
            space_ready: Notify::new(),
            cancel_ready: Notify::new(),
            cancelled: AtomicBool::new(false),
        })
    }

    // Queue admission is one synchronized state transition; the branches encode
    // cancellation, overflow, block, and drop-oldest behavior under one lock.
    // #lizard forgives
    async fn push(&self, frame: Vec<u8>) -> Result<bool> {
        if frame.len() > self.max_frame_bytes {
            return Err(fail(
                "read",
                &format!(
                    "frame {} bytes exceeds the {} byte limit",
                    frame.len(),
                    self.max_frame_bytes
                ),
            ));
        }
        let frame_bytes = frame
            .len()
            .checked_add(1)
            .ok_or_else(|| fail("read", "frame byte size exhausted usize"))?;
        loop {
            if self.cancelled.load(Ordering::Acquire) {
                return Ok(false);
            }
            let notified = self.space_ready.notified();
            let mut state = self.state.lock().await;
            if self.mode == BackpressureMode::DropOldest {
                while !state.frames.is_empty()
                    && (state.frames.len() >= self.max_rows
                        || state.bytes.saturating_add(frame_bytes) > self.max_bytes)
                {
                    let removed = state.frames.pop_front().expect("queue is non-empty");
                    state.bytes = state.bytes.saturating_sub(removed.len().saturating_add(1));
                    state.dropped_frames = state.dropped_frames.saturating_add(1);
                }
            }
            if state.frames.len() < self.max_rows
                && state.bytes.saturating_add(frame_bytes) <= self.max_bytes
            {
                state.bytes += frame_bytes;
                state.frames.push_back(frame);
                drop(state);
                self.data_ready.notify_one();
                return Ok(true);
            }
            drop(state);
            notified.await;
        }
    }

    async fn take_batch(&self) -> Result<Option<(Vec<Vec<u8>>, u64)>> {
        loop {
            let notified = self.data_ready.notified();
            let mut state = self.state.lock().await;
            if !state.frames.is_empty() {
                let frames: Vec<Vec<u8>> = state.frames.drain(..).collect();
                state.bytes = 0;
                let dropped_frames = state.dropped_frames;
                drop(state);
                self.space_ready.notify_one();
                return Ok(Some((frames, dropped_frames)));
            }
            match state.terminal.as_ref() {
                Some(QueueTerminal::Closed) => return Ok(None),
                Some(QueueTerminal::Failed(detail)) => return Err(fail("read", detail)),
                None => {}
            }
            drop(state);
            notified.await;
        }
    }

    async fn finish(&self, terminal: QueueTerminal) {
        self.state.lock().await.terminal = Some(terminal);
        self.data_ready.notify_one();
        self.space_ready.notify_one();
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.data_ready.notify_one();
        self.space_ready.notify_one();
        self.cancel_ready.notify_one();
    }
}

async fn read_frames(mut stream: Socket, queue: Arc<FrameQueue>) {
    use futures_util::StreamExt as _;

    loop {
        if queue.cancelled.load(Ordering::Acquire) {
            let _ = tokio::time::timeout(Duration::from_secs(1), stream.close(None)).await;
            queue.finish(QueueTerminal::Closed).await;
            return;
        }
        tokio::select! {
            () = queue.cancel_ready.notified() => {}
            message = stream.next() => {
                let payload = match message {
                    Some(Ok(Message::Text(text))) => Some(text.as_bytes().to_vec()),
                    Some(Ok(Message::Binary(data))) => Some(data.to_vec()),
                    Some(Ok(Message::Close(_))) | None => {
                        queue.finish(QueueTerminal::Closed).await;
                        return;
                    }
                    Some(Ok(_)) => None,
                    Some(Err(_)) => {
                        queue
                            .finish(QueueTerminal::Failed("WebSocket transport failed".into()))
                            .await;
                        return;
                    }
                };
                if let Some(payload) = payload {
                    match queue.push(payload).await {
                        Ok(true) => {}
                        Ok(false) => {
                            queue.finish(QueueTerminal::Closed).await;
                            return;
                        }
                        Err(error) => {
                            queue.finish(QueueTerminal::Failed(error.to_string())).await;
                            return;
                        }
                    }
                }
            }
        }
    }
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

    fn validate(&self, options: &JsonMap) -> Result<()> {
        WebSocketSourceConfig::from_options(options).map(drop)
    }

    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = WebSocketSourceConfig::from_options(options)?;
        let endpoint_url = resolve_ws_url(secrets, "url")?;
        Ok(Box::new(
            WebSocketSource::new(config)?.with_endpoint(endpoint_url),
        ))
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
        formats: vec![
            FormatIdentity::new(
                crate::json_lines::IDENTITY,
                crate::json_lines::IDENTITY_VERSION,
            )
            .expect("json-lines identity"),
        ],
        config_schema: JsonMap::from([
            ("backpressure".to_string(), serde_json::json!("string")),
            ("max_frame_bytes".to_string(), serde_json::json!("u64")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
            ("max_batch_bytes".to_string(), serde_json::json!("u64")),
            ("insecure".to_string(), serde_json::json!("boolean")),
        ]),
        secret_slots: ["url".to_string()].into_iter().collect(),
        required_secret_slots: ["url".to_string()].into_iter().collect(),
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
        ConnectorFactories::source_only(Arc::new(WebSocketSourceFactory::new())),
    )
}
