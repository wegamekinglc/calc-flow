use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::{StateHandle, backend::validate_sha256};
use crate::{
    CalcFlowError, Epoch, EventTime, JsonMap, Result, canonical_json,
    json::{parse_json_value, validate_json_depth_at, validate_portable_identifier},
};

/// The final v3 checkpoint-manifest format version.
pub const MANIFEST_FORMAT_VERSION: u32 = 3;
/// Maximum accepted size of one v3 checkpoint-manifest document.
pub const MAX_MANIFEST_DOCUMENT_BYTES: usize = 10 * 1024 * 1024;

/// Recovery state represented by an M4 manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecoveryStatus {
    /// The manifest contains a complete final state for its epoch.
    Final,
}

/// Connector cursor metadata stored in one source entry.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CursorManifestEntry {
    /// Connector-defined stable cursor ordering identifier.
    pub order: String,
    /// Strict bounded connector cursor payload.
    pub payload: JsonMap,
}

/// Persisted input-progress state for one operator ingress.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ManifestIngressState {
    /// The ingress participates in aggregate progress.
    Active,
    /// The ingress is temporarily excluded from aggregate progress.
    Idle,
    /// The ingress has permanently ended.
    Ended,
}

/// Persisted progress for one operator ingress.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorIngressManifestEntry {
    /// Current active, idle, or ended state.
    pub state: ManifestIngressState,
    /// Last accepted watermark, or `None` while undefined.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub watermark: Option<EventTime>,
}

/// Persisted source progress and watermark-policy state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceManifestEntry {
    /// Connector cursor; the nullable field must be present in JSON.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub cursor: Option<CursorManifestEntry>,
    /// Hash of the prepared source identity.
    pub identity_hash: String,
    /// Next source sequence coordinate.
    pub sequence: u64,
    /// Whether the source reached its terminal position.
    pub ended: bool,
    /// Persisted watermark-generator state.
    pub watermark_policy: SourceWatermarkManifestState,
}

/// Persisted operator progress and immutable state handles.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorManifestEntry {
    /// Stable ingress ID to persisted progress.
    pub progress: BTreeMap<String, OperatorIngressManifestEntry>,
    /// Small bounded operator-owned metadata.
    pub inline_metadata: JsonMap,
    /// Canonically ordered committed state handles.
    pub segments: Vec<StateHandle>,
}

/// Retention guarantee for an epoch-idempotent sink mechanism.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RetentionClass {
    /// Idempotency records do not expire.
    Unbounded,
    /// Idempotency records have a bounded retention horizon.
    Bounded,
}

/// Delivery state persisted for one sink.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SinkDeliveryManifest {
    /// Ordinary at-least-once delivery.
    Ordinary,
    /// Delivery deduplicated by an epoch-aware sink mechanism.
    EpochIdempotent {
        /// Stable sink mechanism identifier.
        mechanism: String,
        /// Retention class of the mechanism's deduplication records.
        retention: RetentionClass,
    },
    /// Transactional sink delivery.
    Transactional,
}

/// Persisted delivery and optional pre-commit state for one sink.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SinkManifestEntry {
    /// Delivery mechanism state.
    pub delivery: SinkDeliveryManifest,
    /// Connector-owned bounded pre-commit metadata.
    #[serde(deserialize_with = "deserialize_required_option")]
    pub pre_commit: Option<JsonMap>,
}

/// Persisted source watermark-generator state.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum SourceWatermarkManifestState {
    /// Watermarks supplied directly by the connector.
    SourceProvided {
        /// Last emitted watermark, or `None` while undefined.
        #[serde(deserialize_with = "deserialize_required_option")]
        last_emitted_micros: Option<EventTime>,
        /// Whether the source was idle.
        idle: bool,
    },
    /// Generated bounded-out-of-orderness watermarks.
    BoundedOutOfOrderness {
        /// Maximum observed event time, or `None` before data.
        #[serde(deserialize_with = "deserialize_required_option")]
        observed_max_micros: Option<EventTime>,
        /// Last emitted watermark, or `None` while undefined.
        #[serde(deserialize_with = "deserialize_required_option")]
        last_emitted_micros: Option<EventTime>,
        /// Whether the source was idle.
        idle: bool,
    },
    /// Watermark generation disabled for this source.
    Disabled {
        /// Whether the source was idle.
        idle: bool,
    },
}

/// Caller-supplied fields used to construct one validated manifest.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CheckpointManifestFields {
    /// Logical pipeline name.
    pub pipeline_name: String,
    /// Semantic pipeline fingerprint.
    pub pipeline_fingerprint: String,
    /// Hash of the effective runtime configuration.
    pub runtime_config_hash: String,
    /// Checkpoint epoch represented by the manifest.
    pub epoch: Epoch,
    /// Manifest creation timestamp.
    pub created_at: DateTime<Utc>,
    /// Final recovery status.
    pub recovery_status: RecoveryStatus,
    /// Stable source ID to persisted source state.
    pub sources: BTreeMap<String, SourceManifestEntry>,
    /// Stable operator ID to persisted operator state.
    pub operators: BTreeMap<String, OperatorManifestEntry>,
    /// Stable sink ID to persisted sink state.
    pub sinks: BTreeMap<String, SinkManifestEntry>,
}

/// Exact prepared-job identity expected while loading a manifest.
pub struct ManifestExpectation<'a> {
    /// Expected pipeline name.
    pub pipeline_name: &'a str,
    /// Expected semantic pipeline fingerprint.
    pub pipeline_fingerprint: &'a str,
    /// Expected effective runtime-configuration hash used only for diagnostics.
    pub runtime_config_hash: &'a str,
    /// Expected checkpoint epoch.
    pub epoch: Epoch,
    /// Exact expected source IDs.
    pub source_ids: &'a BTreeSet<String>,
    /// Exact expected operator IDs.
    pub operator_ids: &'a BTreeSet<String>,
    /// Exact expected sink IDs.
    pub sink_ids: &'a BTreeSet<String>,
}

/// Canonical v3 checkpoint manifest.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct CheckpointManifest {
    format_version: u32,
    pipeline_name: String,
    pipeline_fingerprint: String,
    runtime_config_hash: String,
    epoch: Epoch,
    created_at: DateTime<Utc>,
    recovery_status: RecoveryStatus,
    sources: BTreeMap<String, SourceManifestEntry>,
    operators: BTreeMap<String, OperatorManifestEntry>,
    sinks: BTreeMap<String, SinkManifestEntry>,
    state_checksum: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SerializedManifest {
    format_version: u32,
    pipeline_name: String,
    pipeline_fingerprint: String,
    runtime_config_hash: String,
    epoch: Epoch,
    created_at: DateTime<Utc>,
    recovery_status: RecoveryStatus,
    sources: BTreeMap<String, SourceManifestEntry>,
    operators: BTreeMap<String, OperatorManifestEntry>,
    sinks: BTreeMap<String, SinkManifestEntry>,
    state_checksum: String,
}

impl From<SerializedManifest> for CheckpointManifest {
    fn from(fields: SerializedManifest) -> Self {
        Self {
            format_version: fields.format_version,
            pipeline_name: fields.pipeline_name,
            pipeline_fingerprint: fields.pipeline_fingerprint,
            runtime_config_hash: fields.runtime_config_hash,
            epoch: fields.epoch,
            created_at: fields.created_at,
            recovery_status: fields.recovery_status,
            sources: fields.sources,
            operators: fields.operators,
            sinks: fields.sinks,
            state_checksum: fields.state_checksum,
        }
    }
}

impl CheckpointManifest {
    /// Constructs a validated manifest and computes its state checksum.
    ///
    /// # Errors
    ///
    /// Returns a validation error for an invalid identity, JSON payload,
    /// handle ownership coordinate, or document size.
    pub fn new(fields: CheckpointManifestFields) -> Result<Self> {
        let mut manifest = Self {
            format_version: MANIFEST_FORMAT_VERSION,
            pipeline_name: fields.pipeline_name,
            pipeline_fingerprint: fields.pipeline_fingerprint,
            runtime_config_hash: fields.runtime_config_hash,
            epoch: fields.epoch,
            created_at: fields.created_at,
            recovery_status: fields.recovery_status,
            sources: fields.sources,
            operators: fields.operators,
            sinks: fields.sinks,
            state_checksum: String::new(),
        };
        manifest.validate_contents()?;
        manifest.state_checksum = manifest.recompute_state_checksum()?;
        manifest.validate_internal()?;
        manifest.ensure_size_bound()?;
        Ok(manifest)
    }

    /// Parses one bounded, duplicate-key-free, strict manifest document.
    ///
    /// # Errors
    ///
    /// Fails before exposing a value when the byte bound, JSON depth, version,
    /// typed fields, handle ownership, or state checksum is invalid.
    pub fn from_bytes(document: &[u8]) -> Result<Self> {
        validate_document_size(document)?;
        let value = parse_json_value(document, "checkpoint manifest")?;
        validate_document_version(&value)?;
        let fields: SerializedManifest =
            serde_json::from_value(value).map_err(|error| format_error(error.to_string()))?;
        let manifest = Self::from(fields);
        manifest.validate_internal()?;
        Ok(manifest)
    }

    /// Validates this manifest against one exact prepared-job expectation.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::CheckpointMismatch`] for a semantic identity,
    /// epoch, ID-set, or checksum mismatch. A runtime-configuration hash
    /// difference is diagnostics-only and does not fail validation.
    pub fn validate(&self, expected: &ManifestExpectation<'_>) -> Result<()> {
        self.validate_internal()?;
        self.validate_identity(expected)?;
        self.validate_epoch(expected.epoch)?;
        validate_id_set("source", self.sources.keys(), expected.source_ids)?;
        validate_id_set("operator", self.operators.keys(), expected.operator_ids)?;
        validate_id_set("sink", self.sinks.keys(), expected.sink_ids)?;
        Ok(())
    }

    fn validate_identity(&self, expected: &ManifestExpectation<'_>) -> Result<()> {
        validate_expected("pipeline name", &self.pipeline_name, expected.pipeline_name)?;
        validate_expected(
            "pipeline fingerprint",
            &self.pipeline_fingerprint,
            expected.pipeline_fingerprint,
        )
    }

    pub(crate) fn runtime_config_changed(&self, expected: &ManifestExpectation<'_>) -> bool {
        self.runtime_config_hash != expected.runtime_config_hash
    }

    fn validate_epoch(&self, expected: Epoch) -> Result<()> {
        if self.epoch != expected {
            return Err(mismatch(format!(
                "manifest epoch {} does not match expected epoch {}",
                self.epoch.as_u64(),
                expected.as_u64()
            )));
        }
        Ok(())
    }

    /// Returns canonical compact JSON bytes for this manifest.
    ///
    /// # Errors
    ///
    /// Returns a validation or format error when the manifest is invalid or
    /// exceeds the document bound.
    pub fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate_internal()?;
        let value = serde_json::to_value(self).map_err(|error| format_error(error.to_string()))?;
        let bytes = canonical_json(&value)?.into_bytes();
        if bytes.len() > MAX_MANIFEST_DOCUMENT_BYTES {
            return Err(format_error(format!(
                "manifest exceeds the {MAX_MANIFEST_DOCUMENT_BYTES}-byte limit"
            )));
        }
        Ok(bytes)
    }

    /// Recomputes the deterministic checksum over source, operator, and sink state.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Format`] if state cannot be canonically serialized.
    pub fn recompute_state_checksum(&self) -> Result<String> {
        let value = json!({
            "operators": &self.operators,
            "sinks": &self.sinks,
            "sources": &self.sources,
        });
        Ok(hex::encode(Sha256::digest(
            canonical_json(&value)?.as_bytes(),
        )))
    }

    /// Returns the manifest format version.
    pub const fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Returns the logical pipeline name.
    pub fn pipeline_name(&self) -> &str {
        &self.pipeline_name
    }

    /// Returns the semantic pipeline fingerprint.
    pub fn pipeline_fingerprint(&self) -> &str {
        &self.pipeline_fingerprint
    }

    /// Returns the runtime configuration hash.
    pub fn runtime_config_hash(&self) -> &str {
        &self.runtime_config_hash
    }

    /// Returns the checkpoint epoch.
    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Returns the creation timestamp.
    pub const fn created_at(&self) -> DateTime<Utc> {
        self.created_at
    }

    /// Returns the recovery status.
    pub const fn recovery_status(&self) -> RecoveryStatus {
        self.recovery_status
    }

    /// Returns source entries in stable ID order.
    pub const fn sources(&self) -> &BTreeMap<String, SourceManifestEntry> {
        &self.sources
    }

    /// Returns operator entries in stable ID order.
    pub const fn operators(&self) -> &BTreeMap<String, OperatorManifestEntry> {
        &self.operators
    }

    /// Returns sink entries in stable ID order.
    pub const fn sinks(&self) -> &BTreeMap<String, SinkManifestEntry> {
        &self.sinks
    }

    /// Returns the deterministic state checksum.
    pub fn state_checksum(&self) -> &str {
        &self.state_checksum
    }

    fn validate_internal(&self) -> Result<()> {
        if self.format_version != MANIFEST_FORMAT_VERSION {
            return Err(CalcFlowError::UnsupportedVersion {
                expected: MANIFEST_FORMAT_VERSION,
                found: self.format_version,
            });
        }
        self.validate_contents()?;
        validate_sha256("state_checksum", &self.state_checksum)?;
        let expected = self.recompute_state_checksum()?;
        if self.state_checksum != expected {
            return Err(mismatch(
                "manifest state checksum does not match its contents".into(),
            ));
        }
        Ok(())
    }

    fn validate_contents(&self) -> Result<()> {
        validate_portable_identifier("pipeline_name", &self.pipeline_name)?;
        validate_sha256("pipeline_fingerprint", &self.pipeline_fingerprint)?;
        validate_sha256("runtime_config_hash", &self.runtime_config_hash)?;
        validate_sources(&self.sources)?;
        let mut identities = BTreeSet::new();
        let mut paths = BTreeSet::new();
        validate_operators(&self.operators, self.epoch, &mut identities, &mut paths)?;
        validate_sinks(&self.sinks)
    }

    fn ensure_size_bound(&self) -> Result<()> {
        self.canonical_bytes().map(|_| ())
    }
}

fn validate_document_size(document: &[u8]) -> Result<()> {
    if document.len() > MAX_MANIFEST_DOCUMENT_BYTES {
        Err(format_error(format!(
            "manifest exceeds the {MAX_MANIFEST_DOCUMENT_BYTES}-byte limit"
        )))
    } else {
        Ok(())
    }
}

fn validate_document_version(value: &Value) -> Result<()> {
    let object = value
        .as_object()
        .ok_or_else(|| format_error("checkpoint manifest must contain a JSON object".into()))?;
    let found = object
        .get("format_version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| format_error("manifest format_version must be a u32".into()))?;
    if found == MANIFEST_FORMAT_VERSION {
        Ok(())
    } else {
        Err(CalcFlowError::UnsupportedVersion {
            expected: MANIFEST_FORMAT_VERSION,
            found,
        })
    }
}

fn validate_sources(sources: &BTreeMap<String, SourceManifestEntry>) -> Result<()> {
    for (source_id, source) in sources {
        validate_portable_identifier("sources.id", source_id)?;
        validate_sha256("sources.identity_hash", &source.identity_hash)?;
        if let Some(cursor) = &source.cursor {
            validate_portable_identifier("sources.cursor.order", &cursor.order)?;
            validate_json_map(&cursor.payload, "source cursor payload")?;
        }
    }
    Ok(())
}

fn validate_operators(
    operators: &BTreeMap<String, OperatorManifestEntry>,
    manifest_epoch: Epoch,
    identities: &mut BTreeSet<(String, Epoch, String)>,
    paths: &mut BTreeSet<String>,
) -> Result<()> {
    for (operator_id, operator) in operators {
        validate_portable_identifier("operators.id", operator_id)?;
        validate_json_map(&operator.inline_metadata, "operator inline metadata")?;
        validate_ingress_ids(operator)?;
        validate_operator_handles(operator_id, operator, manifest_epoch, identities, paths)?;
    }
    Ok(())
}

fn validate_ingress_ids(operator: &OperatorManifestEntry) -> Result<()> {
    for ingress_id in operator.progress.keys() {
        validate_portable_identifier("operators.progress.id", ingress_id)?;
    }
    Ok(())
}

fn validate_operator_handles(
    operator_id: &str,
    operator: &OperatorManifestEntry,
    manifest_epoch: Epoch,
    identities: &mut BTreeSet<(String, Epoch, String)>,
    paths: &mut BTreeSet<String>,
) -> Result<()> {
    let mut previous = None;
    for handle in &operator.segments {
        validate_operator_handle(operator_id, handle, manifest_epoch, previous)?;
        record_unique_handle(handle, identities, paths)?;
        previous = Some(handle);
    }
    Ok(())
}

fn validate_operator_handle(
    operator_id: &str,
    handle: &StateHandle,
    manifest_epoch: Epoch,
    previous: Option<&StateHandle>,
) -> Result<()> {
    handle.validate_for(operator_id, handle.epoch())?;
    if handle.epoch() > manifest_epoch {
        return Err(mismatch(format!(
            "state handle epoch {} is newer than manifest epoch {}",
            handle.epoch().as_u64(),
            manifest_epoch.as_u64()
        )));
    }
    if previous.is_some_and(|value| value >= handle) {
        return Err(format_error(format!(
            "operator {operator_id:?} state handles are not in canonical order"
        )));
    }
    Ok(())
}

fn record_unique_handle(
    handle: &StateHandle,
    identities: &mut BTreeSet<(String, Epoch, String)>,
    paths: &mut BTreeSet<String>,
) -> Result<()> {
    let identity = (
        handle.operator_id().into(),
        handle.epoch(),
        handle.segment_id().into(),
    );
    if !identities.insert(identity) {
        return Err(format_error("duplicate state handle identity".into()));
    }
    if !paths.insert(handle.relative_path().into()) {
        return Err(format_error("duplicate committed state path".into()));
    }
    Ok(())
}

fn validate_sinks(sinks: &BTreeMap<String, SinkManifestEntry>) -> Result<()> {
    for (sink_id, sink) in sinks {
        validate_portable_identifier("sinks.id", sink_id)?;
        if let SinkDeliveryManifest::EpochIdempotent { mechanism, .. } = &sink.delivery {
            validate_portable_identifier("sinks.delivery.mechanism", mechanism)?;
        }
        if let Some(pre_commit) = &sink.pre_commit {
            validate_json_map(pre_commit, "sink pre-commit metadata")?;
        }
    }
    Ok(())
}

impl<'de> Deserialize<'de> for CheckpointManifest {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let manifest = Self::from(SerializedManifest::deserialize(deserializer)?);
        manifest.validate_internal().map_err(D::Error::custom)?;
        Ok(manifest)
    }
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer)
}

fn validate_json_map(values: &JsonMap, label: &str) -> Result<()> {
    validate_json_depth_at(
        &Value::Object(values.clone().into_iter().collect()),
        label,
        0,
    )
}

fn validate_expected(label: &str, found: &str, expected: &str) -> Result<()> {
    if found == expected {
        Ok(())
    } else {
        Err(mismatch(format!(
            "manifest {label} {found:?} does not match expected {expected:?}"
        )))
    }
}

fn validate_id_set<'a>(
    label: &str,
    found: impl Iterator<Item = &'a String>,
    expected: &BTreeSet<String>,
) -> Result<()> {
    let found = found.cloned().collect::<BTreeSet<_>>();
    if found == *expected {
        Ok(())
    } else {
        Err(mismatch(format!(
            "manifest {label} IDs do not match the prepared job"
        )))
    }
}

fn mismatch(message: String) -> CalcFlowError {
    CalcFlowError::CheckpointMismatch { message }
}

fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}
