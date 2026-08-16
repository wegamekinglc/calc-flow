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
