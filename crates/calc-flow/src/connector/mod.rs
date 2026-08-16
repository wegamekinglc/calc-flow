//! Core connector contracts for the 3.0 connector surface.
//!
//! The core crate owns the trusted registry, the independent capability
//! vocabulary, secret references, the bounded format layer, and delivery
//! derivation. Connector implementations and their heavy clients live in
//! the separate `calc-flow-connectors` crate and register through
//! [`ConnectorRegistry`]; compiled plans capture an immutable
//! [`ConnectorRegistrySnapshot`] that later registrations cannot affect.
//!
//! # Example
//!
//! ```
//! use calc_flow::{ConnectorRegistry, FormatDescriptor, FormatIdentity};
//!
//! let mut registry = ConnectorRegistry::new();
//! registry
//!     .register_format(FormatDescriptor {
//!         identity: FormatIdentity::new("csv", "1").expect("valid identity"),
//!     })
//!     .expect("first registration succeeds");
//! let duplicate = registry
//!     .register_format(FormatDescriptor {
//!         identity: FormatIdentity::new("csv", "1").expect("valid identity"),
//!     })
//!     .expect_err("duplicate identities are rejected");
//! assert!(duplicate.to_string().contains("format"));
//!
//! let snapshot = registry.snapshot();
//! assert!(snapshot
//!     .resolve_format(&FormatIdentity::new("csv", "1").expect("valid identity"))
//!     .is_ok());
//! assert!(snapshot
//!     .resolve_format(&FormatIdentity::new("parquet", "1").expect("valid identity"))
//!     .is_err());
//! ```

mod capability;
mod format;
mod registry;
mod secret;

use std::fmt;
use std::sync::Arc;

use thiserror::Error;

use crate::{CalcFlowError, Result};

/// Stable name of one connector lifecycle operation.
///
/// The name identifies the failing step (`open_source`, `commit`, ...)
/// without carrying payload bytes; each connector implementation owns its
/// stable vocabulary.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct ConnectorOperation(Arc<str>);

impl ConnectorOperation {
    /// Builds an operation name from a non-empty string.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the name is empty.
    pub fn new(name: &str) -> Result<Self> {
        if name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "connector operation".into(),
                message: "connector operation name must not be empty".into(),
            });
        }
        Ok(Self(Arc::from(name)))
    }

    /// Returns the stable operation name.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ConnectorOperation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

/// Safe failure projection from one connector operation.
///
/// The fields carry the connector identity, the stable operation name, and a
/// payload-free detail string. Secret values, credentialed URLs, raw frames,
/// and query bodies never enter these fields; connector implementations
/// project client failures through this type instead of forwarding client
/// error strings.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
#[error("connector {identity} failed during {operation}: {detail}")]
pub struct ConnectorError {
    /// Identity of the connector that failed.
    pub identity: ConnectorIdentity,
    /// Stable name of the failing operation.
    pub operation: ConnectorOperation,
    /// Payload-free human-readable detail.
    pub detail: String,
}

impl ConnectorError {
    /// Assembles a connector failure projection.
    pub fn new(identity: ConnectorIdentity, operation: ConnectorOperation, detail: &str) -> Self {
        Self {
            identity,
            operation,
            detail: detail.to_string(),
        }
    }
}

pub use capability::{
    ConnectorCapabilities, ConnectorDescriptor, ConnectorIdentity, ConnectorKind,
    DeliveryCapability, DeliveryParticipant, DeliveryProof, FormatDescriptor, FormatIdentity,
    ParticipantRole, ReplayCapability, TransactionSupport, WatermarkSupport,
    validate_connector_options, validate_delivery_guarantee,
};
pub use format::{DecodeBounds, FormatDecoder, FormatEncoder};
pub use registry::{
    ConnectorFactories, ConnectorRegistry, ConnectorRegistrySnapshot, ConnectorSinkFactory,
    ConnectorSourceFactory,
};
pub use secret::{
    EnvironmentSecretResolver, SecretHandle, SecretReference, SecretResolver, SecretResolverKind,
};
