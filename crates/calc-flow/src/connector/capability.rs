//! Connector identity and the independent capability vocabulary.
//!
//! The M6 specification forbids a single coarse capability flag: delivery,
//! replay, watermark, transaction, snapshot, polling, CDC, and lookup are
//! separate axes on [`ConnectorCapabilities`], and the projection methods
//! convert them into the runtime-native capability types that whole-job
//! preflight already validates.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::Arc;

use crate::continuous::{
    NativeWatermarkCapability, ReplayPositioning, SinkDelivery, SourceDeliveryCapability,
};
use crate::json::JsonMap;
use crate::pipeline::DeliveryGuarantee;
use crate::{CalcFlowError, Result};

/// Trusted identity of one connector implementation.
///
/// A connector is `(provider, name, version)`: the provider names the
/// trusted source of the implementation (`calc-flow-connectors` for
/// built-ins), the name is the transport, and the version is the connector
/// implementation version, not the transport protocol version.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct ConnectorIdentity {
    /// Trusted provider namespace of the implementation.
    pub provider: Arc<str>,
    /// Transport name, for example `file` or `kafka`.
    pub name: Arc<str>,
    /// Connector implementation version.
    pub version: Arc<str>,
}

impl ConnectorIdentity {
    /// Builds an identity from non-empty components.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when any component is
    /// empty.
    pub fn new(provider: &str, name: &str, version: &str) -> Result<Self> {
        if provider.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "provider".into(),
                message: "connector provider must not be empty".into(),
            });
        }
        if name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "name".into(),
                message: "connector name must not be empty".into(),
            });
        }
        if version.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "version".into(),
                message: "connector version must not be empty".into(),
            });
        }
        Ok(Self {
            provider: Arc::from(provider),
            name: Arc::from(name),
            version: Arc::from(version),
        })
    }
}

impl fmt::Display for ConnectorIdentity {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}/{}/{}",
            self.provider, self.name, self.version
        )
    }
}

/// Which lifecycle directions one connector implements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConnectorKind {
    /// Reads external data into the graph.
    Source,
    /// Writes graph output to an external system.
    Sink,
    /// Reads and writes through one registered identity.
    Both,
}

/// Delivery strength one participant can uphold on its own.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeliveryCapability {
    /// Accepted events may be lost before runtime observation.
    BestEffort,
    /// Accepted events survive until an exact cursor can replay them.
    AtLeastOnce,
    /// Replay plus a transactional or idempotent commit protocol.
    ExactlyOnce,
}

/// Whether a source can replay from an exact accepted cut.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayCapability {
    /// The source pauses, reports, and later seeks to the exact cut.
    ReplayableExact,
    /// Exact recovery is unavailable.
    Unreplayable,
}

/// Native watermark behavior of a connector.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WatermarkSupport {
    /// The connector emits event-time watermarks itself.
    Native,
    /// Watermarks come only from the runtime watermark policy.
    GeneratedOnly,
}

/// Transactional commit protocol strength of a sink.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TransactionSupport {
    /// No transactional or epoch-idempotent protocol.
    None,
    /// Two-phase pre-commit and commit after manifest durability.
    PreCommitCommit,
    /// Idempotent ledger keyed by pipeline, sink, and epoch.
    LedgerIdempotent,
}

/// Independent capability axes declared by one connector descriptor.
///
/// The mode axes are availability flags frozen by the M6 specification; a
/// two-state enum would add no information over `bool`.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConnectorCapabilities {
    /// Delivery strength the participant can uphold alone.
    pub delivery: DeliveryCapability,
    /// Exact replay availability.
    pub replay: ReplayCapability,
    /// Native watermark emission.
    pub watermark: WatermarkSupport,
    /// Sink transaction protocol.
    pub transaction: TransactionSupport,
    /// Consistent snapshot reading mode.
    pub snapshot: bool,
    /// Strictly monotonic composite-cursor polling mode.
    pub polling: bool,
    /// Append-only change-event capture mode.
    pub cdc: bool,
    /// Point-lookup mode; no 3.0 connector ships it.
    pub lookup: bool,
}

impl ConnectorCapabilities {
    /// Projects replay capability onto the runtime-native enum.
    pub fn replay_positioning(&self) -> ReplayPositioning {
        match self.replay {
            ReplayCapability::ReplayableExact => ReplayPositioning::ExactPauseReportAndSeek,
            ReplayCapability::Unreplayable => ReplayPositioning::Unsupported,
        }
    }

    /// Projects delivery capability onto the runtime-native enum.
    pub fn source_delivery(&self) -> SourceDeliveryCapability {
        match self.delivery {
            DeliveryCapability::BestEffort => SourceDeliveryCapability::Lossy,
            DeliveryCapability::AtLeastOnce | DeliveryCapability::ExactlyOnce => {
                SourceDeliveryCapability::Lossless
            }
        }
    }

    /// Projects watermark capability onto the runtime-native enum.
    pub fn native_watermarks(&self) -> NativeWatermarkCapability {
        match self.watermark {
            WatermarkSupport::Native => NativeWatermarkCapability::EmitsNative,
            WatermarkSupport::GeneratedOnly => NativeWatermarkCapability::NeverEmits,
        }
    }

    /// Projects transaction support onto the runtime-native sink delivery.
    pub fn sink_delivery(&self) -> SinkDelivery {
        match self.transaction {
            TransactionSupport::None => SinkDelivery::Ordinary,
            TransactionSupport::PreCommitCommit => SinkDelivery::Transactional,
            TransactionSupport::LedgerIdempotent => SinkDelivery::EpochIdempotent {
                mechanism: "ledger".into(),
                retention: crate::state::RetentionClass::Bounded,
            },
        }
    }

    fn exactly_once_capable(self, role: ParticipantRole) -> bool {
        match role {
            ParticipantRole::Source => {
                self.replay == ReplayCapability::ReplayableExact
                    && self.delivery != DeliveryCapability::BestEffort
            }
            ParticipantRole::Sink => self.transaction != TransactionSupport::None,
        }
    }
}

/// Data-only description of one registered connector.
#[derive(Clone, Debug)]
pub struct ConnectorDescriptor {
    /// Trusted identity of the implementation.
    pub identity: ConnectorIdentity,
    /// Lifecycle directions the connector implements.
    pub kind: ConnectorKind,
    /// Declared capability axes.
    pub capabilities: ConnectorCapabilities,
    /// Formats the connector accepts.
    pub formats: Vec<FormatIdentity>,
    /// Allowed option keys mapped to schema data.
    pub config_schema: JsonMap,
    /// Named secret slots that must arrive as secret references.
    pub secret_slots: BTreeSet<String>,
}

/// Data-only description of one registered format codec.
#[derive(Clone, Debug)]
pub struct FormatDescriptor {
    /// Identity of the codec.
    pub identity: FormatIdentity,
}

/// Trusted identity of one format codec.
#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct FormatIdentity {
    /// Codec name, for example `csv` or `parquet`.
    pub name: Arc<str>,
    /// Codec version.
    pub version: Arc<str>,
}

impl FormatIdentity {
    /// Builds a format identity from non-empty components.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when any component is
    /// empty.
    pub fn new(name: &str, version: &str) -> Result<Self> {
        if name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "format name".into(),
                message: "format name must not be empty".into(),
            });
        }
        if version.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "format version".into(),
                message: "format version must not be empty".into(),
            });
        }
        Ok(Self {
            name: Arc::from(name),
            version: Arc::from(version),
        })
    }
}

/// Validates connector options against a descriptor before any factory
/// invocation.
///
/// Option keys must be declared by `config_schema`, and declared secret
/// slots must never carry literal values in options: secrets arrive only as
/// [`crate::connector::SecretReference`] values.
///
/// # Errors
///
/// Returns [`CalcFlowError::InvalidArgument`] naming the offending key for
/// an undeclared option or a secret slot carrying a literal value.
pub fn validate_connector_options(
    descriptor: &ConnectorDescriptor,
    options: &JsonMap,
) -> Result<()> {
    for key in options.keys() {
        if descriptor.secret_slots.contains(key) {
            return Err(CalcFlowError::InvalidArgument {
                field: key.clone(),
                message: "secret values must use a secret reference, not connector options".into(),
            });
        }
        if !descriptor.config_schema.contains_key(key) {
            return Err(CalcFlowError::InvalidArgument {
                field: key.clone(),
                message: "unknown connector option".into(),
            });
        }
    }
    Ok(())
}

/// Lifecycle role of one delivery participant in the reachable path.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParticipantRole {
    /// A source binding feeding the graph.
    Source,
    /// A sink binding receiving one output.
    Sink,
}

/// One reachable participant considered by delivery derivation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeliveryParticipant {
    /// Stable path naming the participant in diagnostics.
    pub path: String,
    /// Source or sink role.
    pub role: ParticipantRole,
    /// Declared capability axes.
    pub capabilities: ConnectorCapabilities,
}

/// Requested and effective delivery guarantee for one plan output.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeliveryProof {
    /// Guarantee the plan requested.
    pub requested: DeliveryGuarantee,
    /// Guarantee the reachable participants can prove.
    pub effective: DeliveryGuarantee,
}

/// Derives the effective delivery guarantee for a requested level.
///
/// A requested [`DeliveryGuarantee::ExactlyOnce`] fails when any reachable
/// participant cannot uphold it; the error names every incapable
/// participant path so compilation stops before any connector lifecycle
/// side effect. Lower requests record their proof without upgrading.
///
/// # Errors
///
/// Returns [`CalcFlowError::Compile`] listing every incapable participant
/// path when exactly-once was requested and cannot be proven.
pub fn validate_delivery_guarantee(
    requested: DeliveryGuarantee,
    participants: &[DeliveryParticipant],
) -> Result<DeliveryProof> {
    let effective = match requested {
        DeliveryGuarantee::AtLeastOnce => DeliveryGuarantee::AtLeastOnce,
        DeliveryGuarantee::ExactlyOnce => {
            let incapable: Vec<&str> = participants
                .iter()
                .filter(|participant| {
                    !participant
                        .capabilities
                        .exactly_once_capable(participant.role)
                })
                .map(|participant| participant.path.as_str())
                .collect();
            if incapable.is_empty() {
                DeliveryGuarantee::ExactlyOnce
            } else {
                return Err(CalcFlowError::Compile {
                    message: format!(
                        "exactly-once delivery is not provable; incapable participants: {}",
                        incapable.join(", ")
                    ),
                });
            }
        }
    };
    Ok(DeliveryProof {
        requested,
        effective,
    })
}
