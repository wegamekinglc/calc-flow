//! The trusted connector registry and its plan-scoped immutable snapshot.
//!
//! Registration happens in trusted process-local code; compilation captures
//! a [`ConnectorRegistrySnapshot`] that later registrations cannot affect.
//! Projects select connectors by data-only identity, and factories resolve
//! exclusively through the snapshot.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;

use crate::connector::capability::{
    ConnectorDescriptor, ConnectorIdentity, ConnectorKind, FormatDescriptor, FormatIdentity,
};
use crate::connector::secret::SecretResolver;
use crate::continuous::{StreamSink, StreamSource, TransactionalStreamSink};
use crate::json::JsonMap;
use crate::{CalcFlowError, Result};

/// Opens source connectors for one registered identity.
#[async_trait]
pub trait ConnectorSourceFactory: Send + Sync {
    /// The data-only descriptor this factory registers.
    fn descriptor(&self) -> &ConnectorDescriptor;

    /// Opens one source connector.
    ///
    /// # Errors
    ///
    /// Returns a safe error that carries the connector identity and
    /// operation, preferably as [`crate::connector::ConnectorError`]; it must
    /// not include secret values, credentialed URLs, raw frames, or query
    /// bodies.
    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSource>>;
}

/// Opens sink connectors for one registered identity.
#[async_trait]
pub trait ConnectorSinkFactory: Send + Sync {
    /// The data-only descriptor this factory registers.
    fn descriptor(&self) -> &ConnectorDescriptor;

    /// Opens one ordinary sink connector.
    ///
    /// # Errors
    ///
    /// Returns a safe error that carries the connector identity and
    /// operation, preferably as [`crate::connector::ConnectorError`]; it must
    /// not include secret values, credentialed URLs, raw frames, or query
    /// bodies.
    async fn open(
        &self,
        options: &JsonMap,
        secrets: &dyn SecretResolver,
    ) -> Result<Box<dyn StreamSink>>;

    /// Opens one transactional sink connector when the implementation
    /// provides it.
    ///
    /// The default returns `None` for sinks whose descriptor already rules
    /// out exactly-once during delivery derivation.
    ///
    /// # Errors
    ///
    /// Returns a safe error when the transactional implementation fails to
    /// open.
    async fn open_transactional(
        &self,
        _options: &JsonMap,
        _secrets: &dyn SecretResolver,
    ) -> Result<Option<Box<dyn TransactionalStreamSink>>> {
        Ok(None)
    }
}

/// Trusted factories bound to one registered connector.
#[derive(Clone, Default)]
pub struct ConnectorFactories {
    /// Factory for the source direction, when registered.
    pub source: Option<Arc<dyn ConnectorSourceFactory>>,
    /// Factory for the sink direction, when registered.
    pub sink: Option<Arc<dyn ConnectorSinkFactory>>,
}

impl ConnectorFactories {
    /// Binds only a source factory.
    pub fn source_only(source: Arc<dyn ConnectorSourceFactory>) -> Self {
        Self {
            source: Some(source),
            sink: None,
        }
    }

    /// Binds only a sink factory.
    pub fn sink_only(sink: Arc<dyn ConnectorSinkFactory>) -> Self {
        Self {
            source: None,
            sink: Some(sink),
        }
    }

    /// Binds both directions.
    pub fn both(
        source: Arc<dyn ConnectorSourceFactory>,
        sink: Arc<dyn ConnectorSinkFactory>,
    ) -> Self {
        Self {
            source: Some(source),
            sink: Some(sink),
        }
    }
}

struct RegisteredConnector {
    source: Option<Arc<dyn ConnectorSourceFactory>>,
    sink: Option<Arc<dyn ConnectorSinkFactory>>,
}

/// The mutable trusted registry connectors register into.
#[derive(Default)]
pub struct ConnectorRegistry {
    connectors: BTreeMap<ConnectorIdentity, RegisteredConnector>,
    formats: BTreeMap<FormatIdentity, FormatDescriptor>,
}

impl std::fmt::Debug for ConnectorRegistry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ConnectorRegistry")
            .field(
                "connectors",
                &self
                    .connectors
                    .keys()
                    .map(|identity| {
                        format!(
                            "{}/{}/{}",
                            identity.provider, identity.name, identity.version
                        )
                    })
                    .collect::<Vec<String>>(),
            )
            .field(
                "formats",
                &self
                    .formats
                    .keys()
                    .map(|identity| format!("{}/{}", identity.name, identity.version))
                    .collect::<Vec<String>>(),
            )
            .finish()
    }
}

impl ConnectorRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers one connector with its trusted factories.
    ///
    /// Registration is atomic: a duplicate identity, a kind/factory
    /// mismatch, or a factory descriptor disagreeing with the registered
    /// identity leaves the registry unchanged.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Conflict`] for a duplicate identity and
    /// [`CalcFlowError::InvalidArgument`] when the declared kind lacks its
    /// matching factory or a factory carries a different identity.
    pub fn register_connector(
        &mut self,
        descriptor: ConnectorDescriptor,
        factories: ConnectorFactories,
    ) -> Result<()> {
        if self.connectors.contains_key(&descriptor.identity) {
            let identity = &descriptor.identity;
            return Err(CalcFlowError::Conflict {
                resource: "connector".into(),
                key: format!(
                    "{}/{}/{}",
                    identity.provider, identity.name, identity.version
                ),
            });
        }
        match descriptor.kind {
            ConnectorKind::Source if factories.source.is_none() => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "factories".into(),
                    message: "source-kind connector requires a source factory".into(),
                });
            }
            ConnectorKind::Sink if factories.sink.is_none() => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "factories".into(),
                    message: "sink-kind connector requires a sink factory".into(),
                });
            }
            ConnectorKind::Both if factories.source.is_none() || factories.sink.is_none() => {
                return Err(CalcFlowError::InvalidArgument {
                    field: "factories".into(),
                    message: "both-kind connector requires source and sink factories".into(),
                });
            }
            ConnectorKind::Source | ConnectorKind::Sink | ConnectorKind::Both => {}
        }
        if let Some(source) = &factories.source {
            if source.descriptor().identity != descriptor.identity {
                return Err(CalcFlowError::InvalidArgument {
                    field: "factories".into(),
                    message: "source factory descriptor identity does not match the registration"
                        .into(),
                });
            }
        }
        if let Some(sink) = &factories.sink {
            if sink.descriptor().identity != descriptor.identity {
                return Err(CalcFlowError::InvalidArgument {
                    field: "factories".into(),
                    message: "sink factory descriptor identity does not match the registration"
                        .into(),
                });
            }
        }
        self.connectors.insert(
            descriptor.identity,
            RegisteredConnector {
                source: factories.source,
                sink: factories.sink,
            },
        );
        Ok(())
    }

    /// Registers one format codec descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Conflict`] for a duplicate identity.
    pub fn register_format(&mut self, descriptor: FormatDescriptor) -> Result<()> {
        if self.formats.contains_key(&descriptor.identity) {
            let identity = &descriptor.identity;
            return Err(CalcFlowError::Conflict {
                resource: "format".into(),
                key: format!("{}/{}", identity.name, identity.version),
            });
        }
        self.formats.insert(descriptor.identity.clone(), descriptor);
        Ok(())
    }

    /// Captures the immutable plan-scoped snapshot.
    pub fn snapshot(&self) -> ConnectorRegistrySnapshot {
        ConnectorRegistrySnapshot {
            connectors: Arc::new(
                self.connectors
                    .iter()
                    .map(|(identity, registered)| {
                        (
                            identity.clone(),
                            SnapshotConnector {
                                source: registered.source.clone(),
                                sink: registered.sink.clone(),
                            },
                        )
                    })
                    .collect(),
            ),
            formats: Arc::new(self.formats.clone()),
        }
    }
}

struct SnapshotConnector {
    source: Option<Arc<dyn ConnectorSourceFactory>>,
    sink: Option<Arc<dyn ConnectorSinkFactory>>,
}

type SnapshotConnectorMap = Arc<BTreeMap<ConnectorIdentity, SnapshotConnector>>;

/// The immutable registry view a compiled plan captures.
///
/// Registrations made after capture are invisible to the snapshot.
#[derive(Clone)]
pub struct ConnectorRegistrySnapshot {
    connectors: SnapshotConnectorMap,
    formats: Arc<BTreeMap<FormatIdentity, FormatDescriptor>>,
}

impl ConnectorRegistrySnapshot {
    /// Resolves the source factory for one identity.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::NotFound`] naming the identity when no
    /// registered source matches, before any construction side effect.
    pub fn resolve_source(
        &self,
        identity: &ConnectorIdentity,
    ) -> Result<Arc<dyn ConnectorSourceFactory>> {
        self.connectors
            .get(identity)
            .and_then(|connector| connector.source.clone())
            .ok_or_else(|| CalcFlowError::NotFound {
                resource: "connector".into(),
                key: format!(
                    "{}/{}/{}",
                    identity.provider, identity.name, identity.version
                ),
            })
    }

    /// Resolves the sink factory for one identity.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::NotFound`] naming the identity when no
    /// registered sink matches, before any construction side effect.
    pub fn resolve_sink(
        &self,
        identity: &ConnectorIdentity,
    ) -> Result<Arc<dyn ConnectorSinkFactory>> {
        self.connectors
            .get(identity)
            .and_then(|connector| connector.sink.clone())
            .ok_or_else(|| CalcFlowError::NotFound {
                resource: "connector".into(),
                key: format!(
                    "{}/{}/{}",
                    identity.provider, identity.name, identity.version
                ),
            })
    }

    /// Resolves one format descriptor.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::NotFound`] naming the identity when the
    /// format was not registered.
    pub fn resolve_format(&self, identity: &FormatIdentity) -> Result<FormatDescriptor> {
        self.formats
            .get(identity)
            .cloned()
            .ok_or_else(|| CalcFlowError::NotFound {
                resource: "format".into(),
                key: format!("{}/{}", identity.name, identity.version),
            })
    }

    /// Lists registered connector identities in deterministic order.
    pub fn identities(&self) -> Vec<ConnectorIdentity> {
        self.connectors.keys().cloned().collect()
    }

    /// Lists registered format identities in deterministic order.
    pub fn format_identities(&self) -> Vec<FormatIdentity> {
        self.formats.keys().cloned().collect()
    }
}
