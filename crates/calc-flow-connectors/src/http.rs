//! The `HTTP` polling connector (feature `http-websocket`).
//!
//! Polls one `HTTP` endpoint with response size, timeout, and retry
//! limits. Conditional requests carry `ETag`/Last-Modified as an optional
//! replay cursor; a 304 response is a poll boundary, not data. TLS
//! verification is on by default — `insecure` must be set explicitly
//! and emits a warning. Authorization headers and credentialed URLs
//! never enter errors, logs, or cursor payloads.

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use calc_flow::{
    Batch, BatchMetadata, CalcFlowError, ConnectorCapabilities, ConnectorDescriptor,
    ConnectorError, ConnectorFactories, ConnectorIdentity, ConnectorKind, ConnectorOperation,
    ConnectorRegistry, ConnectorSinkFactory, ConnectorSourceFactory, Cursor, DeliveryCapability,
    FormatDecoder, JsonMap, Result, SecretHandle, SecretReference, SecretResolver,
    SecretResolverKind, SourceCapabilities, SourceEvent, SourceSchema, StreamSource,
    TransactionSupport, WatermarkSupport,
};
use serde_json::Value;

use crate::json_lines::JsonLinesCodec;

/// The connector implementation version.
pub const IDENTITY_VERSION: &str = "2.0.0";

fn connector_identity() -> ConnectorIdentity {
    ConnectorIdentity::new("calc-flow-connectors", "http", IDENTITY_VERSION)
        .expect("the http connector identity is valid")
}

fn fail(operation: &str, detail: &str) -> CalcFlowError {
    CalcFlowError::Connector(ConnectorError::new(
        connector_identity(),
        ConnectorOperation::new(operation).expect("operation name is non-empty"),
        detail,
    ))
}

/// Resolves the `HTTP` endpoint URL from a secret reference.
///
/// # Errors
///
/// Returns the resolver error; the URL value never enters the error.
pub fn resolve_http_url(secrets: &dyn SecretResolver, key: &str) -> Result<String> {
    let reference = SecretReference::new(SecretResolverKind::Environment, key)
        .map_err(|error| fail("open", &error.to_string()))?;
    let handle: SecretHandle = secrets
        .resolve(&reference)
        .map_err(|_| fail("open", "the `HTTP` URL secret could not be resolved"))?;
    String::from_utf8(handle.expose().to_vec())
        .map_err(|_| fail("open", "the `HTTP` URL secret is not valid UTF-8"))
}

/// Resolves an optional authorization header value from a secret.
///
/// # Errors
///
/// Returns the resolver error when the reference cannot resolve.
pub fn resolve_auth_header(secrets: &dyn SecretResolver, key: &str) -> Result<Option<String>> {
    let reference = SecretReference::new(SecretResolverKind::Environment, key)
        .map_err(|error| fail("open", &error.to_string()))?;
    match secrets.resolve(&reference) {
        Ok(handle) => String::from_utf8(handle.expose().to_vec())
            .map(Some)
            .map_err(|_| fail("open", "the auth header secret is not valid UTF-8")),
        Err(_) => Ok(None),
    }
}

/// Data-only configuration for one `HTTP` polling source.
#[derive(Clone, Debug)]
pub struct HttpSourceConfig {
    /// Secret key holding the `https://…` endpoint URL.
    pub url_key: String,
    /// Optional secret key holding an Authorization header value.
    pub auth_key: Option<String>,
    /// Poll interval between requests.
    pub poll_interval: Duration,
    /// Per-request timeout.
    pub timeout: Duration,
    /// Maximum accepted response body size.
    pub max_response_bytes: u64,
    /// Whether to send If-None-Match / If-Modified-Since (`ETag` replay).
    pub conditional: bool,
    /// Whether TLS certificate verification is disabled (warns).
    pub insecure: bool,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
}

impl HttpSourceConfig {
    /// Parses the source configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] naming the offending
    /// option; `insecure` also emits a tracing warning.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let url_key = required_string(options, "url_key")?;
        let auth_key = match options.get("auth_key") {
            None | Some(Value::Null) => None,
            Some(Value::String(key)) => Some(key.clone()),
            Some(_) => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "auth_key".into(),
                    message: "auth_key must be a string".into(),
                });
            }
        };
        let insecure = options
            .get("insecure")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        if insecure {
            tracing::warn!(
                target: "calc_flow_connectors::http",
                "insecure mode: TLS certificate verification is disabled for this source"
            );
        }
        Ok(Self {
            url_key,
            auth_key,
            poll_interval: Duration::from_millis(
                u64_option(options, "poll_interval_ms")?.unwrap_or(1000),
            ),
            timeout: Duration::from_secs(u64_option(options, "timeout_seconds")?.unwrap_or(30)),
            max_response_bytes: u64_option(options, "max_response_bytes")?
                .unwrap_or(8 * 1024 * 1024),
            conditional: options
                .get("conditional")
                .and_then(Value::as_bool)
                .unwrap_or(true),
            insecure,
            max_batch_rows: u64_option(options, "max_batch_rows")?.unwrap_or(8192),
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

/// The `HTTP` polling source.
pub struct HttpSource {
    capabilities: SourceCapabilities,
    config: HttpSourceConfig,
    client: reqwest::Client,
    etag: Option<String>,
    last_modified: Option<String>,
    sequence: u64,
}

impl HttpSource {
    /// Builds the source; TLS verification defaults to on.
    ///
    /// # Errors
    ///
    /// Returns the client construction error.
    pub fn new(config: HttpSourceConfig) -> Result<Self> {
        let mut builder = reqwest::Client::builder().timeout(config.timeout);
        if config.insecure {
            builder = builder.danger_accept_invalid_certs(true);
        }
        let client = builder
            .build()
            .map_err(|error| fail("open", &error.to_string()))?;
        let replayable = config.conditional;
        let capabilities = SourceCapabilities {
            replay_positioning: if replayable {
                calc_flow::ReplayPositioning::ExactPauseReportAndSeek
            } else {
                calc_flow::ReplayPositioning::Unsupported
            },
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(config.max_batch_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(config.max_response_bytes).unwrap_or(usize::MAX),
            schema: SourceSchema::DynamicOrUnknown,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            capabilities,
            config,
            client,
            etag: None,
            last_modified: None,
            sequence: 0,
        })
    }

    /// Opens at a cursor carrying the `ETag`/Last-Modified replay state.
    ///
    /// # Errors
    ///
    /// Returns the configuration error.
    pub fn open_with_secrets(
        &mut self,
        cursor: Option<Cursor>,
        _secrets: &dyn SecretResolver,
    ) -> Result<()> {
        if let Some(cursor) = cursor {
            self.etag = cursor
                .payload()
                .get("etag")
                .and_then(Value::as_str)
                .map(str::to_string);
            self.last_modified = cursor
                .payload()
                .get("last_modified")
                .and_then(Value::as_str)
                .map(str::to_string);
        }
        Ok(())
    }

    fn build_request(&self, url: &str, auth: Option<&str>) -> reqwest::RequestBuilder {
        let mut request = self.client.get(url);
        if let Some(auth_value) = auth {
            request = request.header("Authorization", auth_value);
        }
        if self.config.conditional {
            if let Some(etag) = &self.etag {
                request = request.header("If-None-Match", etag);
            }
            if let Some(last_modified) = &self.last_modified {
                request = request.header("If-Modified-Since", last_modified);
            }
        }
        request
    }

    /// Produces the next event; a 304 surfaces as Idle.
    ///
    /// # Errors
    ///
    /// Returns the connector error on request failure or oversized
    /// responses.
    pub async fn next_with_secrets(
        &mut self,
        secrets: &dyn SecretResolver,
    ) -> Result<Option<SourceEvent>> {
        let url = resolve_http_url(secrets, &self.config.url_key)?;
        let auth = self.resolve_auth(secrets)?;
        let response = self.fetch(&url, auth.as_deref()).await?;
        if response.status() == reqwest::StatusCode::NOT_MODIFIED {
            tokio::time::sleep(self.config.poll_interval).await;
            return Ok(Some(SourceEvent::Idle));
        }
        let body = self.read_body(response).await?;
        let batch = self.decode_body(&body)?;
        let cursor = self.cursor_from_state()?;
        Ok(Some(SourceEvent::Data { batch, cursor }))
    }

    fn resolve_auth(&self, secrets: &dyn SecretResolver) -> Result<Option<String>> {
        match &self.config.auth_key {
            Some(key) => resolve_auth_header(secrets, key),
            None => Ok(None),
        }
    }

    async fn fetch(&self, url: &str, auth: Option<&str>) -> Result<reqwest::Response> {
        let request = self.build_request(url, auth);
        let response = request
            .send()
            .await
            .map_err(|error| fail("poll", &redact_url(&error.to_string())))?;
        if !response.status().is_success() {
            return Err(fail(
                "poll",
                &format!("endpoint returned status {}", response.status()),
            ));
        }
        Ok(response)
    }

    async fn read_body(&mut self, response: reqwest::Response) -> Result<bytes::Bytes> {
        self.etag = response
            .headers()
            .get("etag")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        self.last_modified = response
            .headers()
            .get("last-modified")
            .and_then(|v| v.to_str().ok())
            .map(str::to_string);
        let content_length = response.content_length().unwrap_or(0);
        if content_length > self.config.max_response_bytes {
            return Err(fail(
                "poll",
                &format!(
                    "response body {} bytes exceeds the {} byte limit",
                    content_length, self.config.max_response_bytes
                ),
            ));
        }
        let body = response
            .bytes()
            .await
            .map_err(|error| fail("poll", &error.to_string()))?;
        if body.len() as u64 > self.config.max_response_bytes {
            return Err(fail(
                "poll",
                &format!(
                    "response body {} bytes exceeds the {} byte limit",
                    body.len(),
                    self.config.max_response_bytes
                ),
            ));
        }
        Ok(body)
    }

    fn decode_body(&mut self, body: &[u8]) -> Result<Batch> {
        let codec = JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?;
        let bounds = calc_flow::DecodeBounds::new(
            self.config.max_batch_rows,
            self.config.max_response_bytes,
        )?;
        let batch = codec.decode(body, &bounds, &[])?;
        self.sequence += 1;
        let metadata = BatchMetadata::new(
            "http",
            self.sequence,
            BTreeMap::from([(
                "url_key".to_string(),
                Value::String(self.config.url_key.clone()),
            )]),
        )
        .map_err(|error| fail("poll", &error.to_string()))?;
        Ok(batch.with_metadata(metadata))
    }

    fn cursor_from_state(&self) -> Result<Cursor> {
        let mut payload = BTreeMap::new();
        if let Some(etag) = &self.etag {
            payload.insert("etag".to_string(), Value::String(etag.clone()));
        }
        if let Some(last_modified) = &self.last_modified {
            payload.insert(
                "last_modified".to_string(),
                Value::String(last_modified.clone()),
            );
        }
        let order = serde_json::to_vec(&vec![&self.etag, &self.last_modified])
            .map_err(|error| fail("cursor", &error.to_string()))?;
        Cursor::unbound(order, payload)
    }
}

fn redact_url(message: &str) -> String {
    message
        .split_whitespace()
        .take(4)
        .collect::<Vec<_>>()
        .join(" ")
}

#[async_trait]
impl StreamSource for HttpSource {
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

/// Trusted source factory for the `HTTP` transport.
pub struct HttpSourceFactory {
    descriptor: ConnectorDescriptor,
}

impl HttpSourceFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: http_connector_descriptor(),
        }
    }
}

impl Default for HttpSourceFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSourceFactory for HttpSourceFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>> {
        let config = HttpSourceConfig::from_options(options)?;
        Ok(Box::new(HttpSource::new(config)?))
    }
}

/// Trusted sink factory placeholder: HTTP is source-only in 3.0.
pub struct HttpSinkFactory {
    descriptor: ConnectorDescriptor,
}

impl HttpSinkFactory {
    /// Creates the factory.
    pub fn new() -> Self {
        Self {
            descriptor: http_connector_descriptor(),
        }
    }
}

impl Default for HttpSinkFactory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ConnectorSinkFactory for HttpSinkFactory {
    fn descriptor(&self) -> &ConnectorDescriptor {
        &self.descriptor
    }

    async fn open(
        &self,
        _options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn calc_flow::StreamSink>> {
        Err(fail("open", "the `HTTP` connector is source-only in 3.0"))
    }
}

fn http_connector_descriptor() -> ConnectorDescriptor {
    ConnectorDescriptor {
        identity: connector_identity(),
        kind: ConnectorKind::Source,
        capabilities: ConnectorCapabilities {
            delivery: DeliveryCapability::AtLeastOnce,
            replay: calc_flow::ReplayCapability::ReplayableExact,
            watermark: WatermarkSupport::GeneratedOnly,
            transaction: TransactionSupport::None,
            snapshot: false,
            polling: true,
            cdc: false,
            lookup: false,
        },
        formats: vec![],
        config_schema: JsonMap::from([
            ("url_key".to_string(), serde_json::json!("string")),
            ("auth_key".to_string(), serde_json::json!("string")),
            ("poll_interval_ms".to_string(), serde_json::json!("u64")),
            ("timeout_seconds".to_string(), serde_json::json!("u64")),
            ("max_response_bytes".to_string(), serde_json::json!("u64")),
            ("conditional".to_string(), serde_json::json!("boolean")),
            ("insecure".to_string(), serde_json::json!("boolean")),
            ("max_batch_rows".to_string(), serde_json::json!("u64")),
        ]),
        secret_slots: ["url_key".to_string(), "auth_key".to_string()]
            .into_iter()
            .collect(),
    }
}

/// Registers the `HTTP` connectors into one trusted registry.
///
/// # Errors
///
/// Returns the registry conflict error when a connector slot is already
/// occupied.
pub fn register_http_connectors(registry: &mut ConnectorRegistry) -> Result<()> {
    registry.register_connector(
        http_connector_descriptor(),
        ConnectorFactories::source_only(std::sync::Arc::new(HttpSourceFactory::new())),
    )
}
