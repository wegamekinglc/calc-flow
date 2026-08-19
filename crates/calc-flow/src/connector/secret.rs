//! Secret references and the resolver contract.
//!
//! Connector configuration never carries secret values: it names secret
//! slots as [`SecretReference`] values, and a trusted
//! [`SecretResolver`] resolves them only when a connector opens. Resolved
//! values live in [`SecretHandle`], which cannot serialize and renders a
//! fixed redaction marker.

use std::fmt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{CalcFlowError, Result};

/// Where a secret reference is resolved from.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SecretResolverKind {
    /// Resolved from a process environment variable.
    Environment,
    /// Resolved from a file path.
    File,
    /// Resolved from a trusted in-process registry.
    Registered,
}

/// A named pointer to a secret; the only secret-shaped value a data-only
/// document may carry.
#[derive(Clone, Debug, Eq, Hash, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct SecretReference {
    /// Resolution source.
    pub resolver: SecretResolverKind,
    /// Resolver-specific key, for example the variable name.
    pub key: String,
}

impl SecretReference {
    /// Builds a reference with a non-empty key.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the key is empty.
    pub fn new(resolver: SecretResolverKind, key: &str) -> Result<Self> {
        if key.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "secret key".into(),
                message: "secret reference key must not be empty".into(),
            });
        }
        Ok(Self {
            resolver,
            key: key.to_string(),
        })
    }
}

/// A resolved secret value.
///
/// The handle is not `Clone` and not `Serialize`; its `Debug` and `Display`
/// render the fixed marker `<redacted secret>`. Use [`SecretHandle::expose`]
/// only inside connector code that constructs client credentials.
pub struct SecretHandle(Vec<u8>);

impl fmt::Debug for SecretHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted secret>")
    }
}

impl SecretHandle {
    /// Wraps resolved secret bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self(bytes.to_vec())
    }

    /// Returns the secret bytes for connector-side credential assembly.
    ///
    /// Callers must never format, log, or persist the returned slice.
    pub fn expose(&self) -> &[u8] {
        &self.0
    }
}

impl fmt::Display for SecretHandle {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("<redacted secret>")
    }
}

/// Resolves secret references for connector opens.
pub trait SecretResolver: Send + Sync {
    /// Resolves one reference to its secret handle.
    ///
    /// # Errors
    ///
    /// Returns a safe error when the reference cannot be resolved; the
    /// error must not include the secret value.
    fn resolve(&self, reference: &SecretReference) -> Result<SecretHandle>;
}

/// Resolves [`SecretResolverKind::Environment`] references from the process
/// environment.
#[derive(Clone, Copy, Debug, Default)]
pub struct EnvironmentSecretResolver;

impl SecretResolver for EnvironmentSecretResolver {
    fn resolve(&self, reference: &SecretReference) -> Result<SecretHandle> {
        std::env::var(&reference.key)
            .map(|value| SecretHandle::from_bytes(value.as_bytes()))
            .map_err(|_| CalcFlowError::NotFound {
                resource: "secret".into(),
                key: reference.key.clone(),
            })
    }
}
