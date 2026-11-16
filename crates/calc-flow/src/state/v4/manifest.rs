use std::collections::{BTreeMap, BTreeSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

use super::super::manifest::{
    CursorManifestEntry, OperatorIngressManifestEntry, SinkDeliveryManifest,
};
use super::{MAX_MANIFEST_DOCUMENT_BYTES, Result, V4Handle, format_error, mismatch};
use crate::{
    CalcFlowError, EventTime, JsonMap, StaticInputDigest, canonical_json, json::parse_json_value,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::state) enum JobKind {
    Runner,
    Session,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct SourceHistoryV4 {
    pub(in crate::state) format_version: u32,
    pub(in crate::state) contract: String,
    pub(in crate::state) inline_metadata: JsonMap,
    pub(in crate::state) segments: Vec<V4Handle>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct SourceV4 {
    #[serde(deserialize_with = "deserialize_required_option")]
    pub(in crate::state) cursor: Option<CursorManifestEntry>,
    pub(in crate::state) identity_hash: String,
    pub(in crate::state) next_sequence: u64,
    pub(in crate::state) ended: bool,
    pub(in crate::state) watermark_policy: super::super::manifest::SourceWatermarkManifestState,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub(in crate::state) history: Option<SourceHistoryV4>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(in crate::state) enum OperatorV4 {
    Table {
        progress: BTreeMap<String, OperatorIngressManifestEntry>,
        #[serde(deserialize_with = "deserialize_required_option")]
        output_frontier: Option<EventTime>,
        state_fingerprint: String,
        partitions: Vec<PartitionV4>,
    },
    Opaque {
        progress: BTreeMap<String, OperatorIngressManifestEntry>,
        #[serde(deserialize_with = "deserialize_required_option")]
        output_frontier: Option<EventTime>,
        state_fingerprint: String,
        inline_metadata: JsonMap,
        blocks: Vec<V4Handle>,
    },
}

impl OperatorV4 {
    pub(in crate::state) fn partitions(&self) -> Option<&[PartitionV4]> {
        match self {
            Self::Table { partitions, .. } => Some(partitions),
            Self::Opaque { .. } => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct PartitionV4 {
    pub(in crate::state) key_group: u32,
    pub(in crate::state) root: V4Handle,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct SinkV4 {
    pub(in crate::state) delivery: SinkDeliveryManifest,
    #[serde(deserialize_with = "deserialize_required_option")]
    pub(in crate::state) pre_commit: Option<JsonMap>,
    pub(in crate::state) blocks: Vec<V4Handle>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(in crate::state) struct V4Manifest {
    pub(in crate::state) format_version: u32,
    pub(in crate::state) pipeline_name: String,
    pub(in crate::state) pipeline_fingerprint: String,
    pub(in crate::state) runtime_config_hash: String,
    pub(in crate::state) epoch: u64,
    pub(in crate::state) created_at: DateTime<Utc>,
    pub(in crate::state) recovery_status: String,
    pub(in crate::state) job_kind: JobKind,
    pub(in crate::state) recovery_fingerprint: String,
    pub(in crate::state) sources: BTreeMap<String, SourceV4>,
    pub(in crate::state) operators: BTreeMap<String, OperatorV4>,
    pub(in crate::state) sinks: BTreeMap<String, SinkV4>,
    pub(in crate::state) static_inputs: BTreeMap<String, StaticInputDigest>,
    pub(in crate::state) manifest_fingerprint: String,
}

#[derive(Clone, Debug)]
pub(in crate::state) struct ManifestExpectationV4 {
    pub(in crate::state) pipeline_name: String,
    pub(in crate::state) pipeline_fingerprint: String,
    pub(in crate::state) recovery_fingerprint: String,
    pub(in crate::state) job_kind: JobKind,
    pub(in crate::state) epoch: u64,
    pub(in crate::state) source_ids: BTreeSet<String>,
    pub(in crate::state) operator_ids: BTreeSet<String>,
    pub(in crate::state) sink_ids: BTreeSet<String>,
    pub(in crate::state) static_inputs: BTreeMap<String, StaticInputDigest>,
    pub(in crate::state) table_partitions: BTreeMap<String, Vec<u32>>,
}

impl V4Manifest {
    pub(in crate::state) fn from_bytes(document: &[u8]) -> Result<Self> {
        if document.len() > MAX_MANIFEST_DOCUMENT_BYTES {
            return Err(format_error(
                "checkpoint manifest exceeds the 10 MiB limit".into(),
            ));
        }
        let value = parse_json_value(document, "checkpoint manifest")?;
        validate_document_version(&value)?;
        let manifest: Self = serde_json::from_value(value).map_err(|error| {
            format_error(format!("checkpoint manifest schema mismatch: {error}"))
        })?;
        manifest.validate_fingerprint()?;
        manifest.validate_structure()
    }

    pub(in crate::state) fn canonical_bytes(&self) -> Result<Vec<u8>> {
        self.validate_fingerprint()?;
        let bytes = canonical_json(&serde_json::to_value(self).map_err(|error| {
            format_error(format!("checkpoint manifest serialization failed: {error}"))
        })?)?
        .into_bytes();
        if bytes.len() > MAX_MANIFEST_DOCUMENT_BYTES {
            return Err(format_error(
                "checkpoint manifest exceeds the 10 MiB limit".into(),
            ));
        }
        Ok(bytes)
    }

    pub(in crate::state) fn validate(&self, expected: &ManifestExpectationV4) -> Result<()> {
        self.validate_fingerprint()?;
        self.validate_structure()?;
        if self.pipeline_name != expected.pipeline_name
            || self.pipeline_fingerprint != expected.pipeline_fingerprint
        {
            return Err(mismatch("pipeline identity mismatch".into()));
        }
        if self.recovery_fingerprint != expected.recovery_fingerprint {
            return Err(mismatch("recovery fingerprint mismatch".into()));
        }
        if self.job_kind != expected.job_kind || self.epoch != expected.epoch {
            return Err(mismatch("checkpoint manifest identity mismatch".into()));
        }
        validate_ids("source", &self.sources, &expected.source_ids)?;
        validate_ids("operator", &self.operators, &expected.operator_ids)?;
        validate_ids("sink", &self.sinks, &expected.sink_ids)?;
        if self.static_inputs != expected.static_inputs {
            return Err(mismatch("static-input participants differ".into()));
        }
        for (operator, groups) in &expected.table_partitions {
            let Some(actual_operator) = self.operators.get(operator) else {
                return Err(mismatch("operator participants differ".into()));
            };
            let actual = actual_operator
                .partitions()
                .map(|partitions| {
                    partitions
                        .iter()
                        .map(|partition| partition.key_group)
                        .collect::<Vec<_>>()
                })
                .ok_or_else(|| mismatch("operator is not a table operator".into()))?;
            if &actual != groups {
                return Err(mismatch(
                    "partition groups differ from prepared key groups".into(),
                ));
            }
        }
        Ok(())
    }

    pub(in crate::state) fn v3_rejection_message() -> &'static str {
        "checkpoint manifest v3 is not restorable; start a new lineage"
    }

    fn validate_fingerprint(&self) -> Result<()> {
        let expected = manifest_fingerprint(self)?;
        if self.manifest_fingerprint != expected {
            return Err(mismatch(
                "manifest fingerprint does not match its contents".into(),
            ));
        }
        Ok(())
    }

    pub(in crate::state) fn validate_structure(&self) -> Result<()> {
        if self.format_version != 4 {
            return Err(CalcFlowError::UnsupportedVersion {
                expected: 4,
                found: self.format_version,
            });
        }
        if self.epoch == 0 || self.recovery_status != "final" {
            return Err(format_error(
                "checkpoint manifest epoch or recovery status is invalid".into(),
            ));
        }
        validate_participant_rules(self)?;
        for source in self.sources.values() {
            if let Some(history) = &source.history {
                if history.format_version != 1 {
                    return Err(format_error(
                        "checkpoint-v4 source-history version is unsupported".into(),
                    ));
                }
                for handle in &history.segments {
                    handle.validate(self.epoch)?;
                }
            }
        }
        for operator in self.operators.values() {
            validate_operator(operator, self.epoch)?;
        }
        for sink in self.sinks.values() {
            for block in &sink.blocks {
                block.validate(self.epoch)?;
            }
        }
        Ok(())
    }
}

pub(in crate::state) fn manifest_fingerprint(manifest: &V4Manifest) -> Result<String> {
    let mut value = serde_json::to_value(manifest).map_err(|error| {
        format_error(format!("checkpoint manifest serialization failed: {error}"))
    })?;
    value
        .as_object_mut()
        .ok_or_else(|| format_error("checkpoint manifest is not an object".into()))?
        .remove("manifest_fingerprint");
    Ok(hex::encode(Sha256::digest(
        canonical_json(&value)?.as_bytes(),
    )))
}

pub(in crate::state) fn recovery_fingerprint(identity: &Value) -> Result<String> {
    Ok(hex::encode(Sha256::digest(
        canonical_json(identity)?.as_bytes(),
    )))
}

fn deserialize_required_option<'de, D, T>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    T::deserialize(deserializer).map(Some)
}

fn validate_document_version(value: &Value) -> Result<()> {
    let found = value
        .get("format_version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or_else(|| format_error("manifest format_version must be a u32".into()))?;
    if found == 3 {
        return Err(CalcFlowError::UnsupportedVersion {
            expected: 4,
            found: 3,
        });
    }
    if found != 4 {
        return Err(CalcFlowError::UnsupportedVersion { expected: 4, found });
    }
    Ok(())
}

fn validate_participant_rules(manifest: &V4Manifest) -> Result<()> {
    match manifest.job_kind {
        JobKind::Runner
            if manifest.sources.is_empty()
                || manifest.operators.is_empty()
                || manifest.sinks.is_empty() =>
        {
            return Err(format_error(
                "runner manifest participants must be non-empty".into(),
            ));
        }
        JobKind::Session
            if manifest.sources.len() != 1
                || manifest.operators.is_empty()
                || !manifest.sinks.is_empty() =>
        {
            return Err(format_error(
                "session manifest must have one source and no sinks".into(),
            ));
        }
        _ => {}
    }
    Ok(())
}

fn validate_operator(operator: &OperatorV4, epoch: u64) -> Result<()> {
    match operator {
        OperatorV4::Table { partitions, .. } => {
            let groups = partitions
                .iter()
                .map(|partition| partition.key_group)
                .collect::<Vec<_>>();
            let unique = groups.iter().copied().collect::<BTreeSet<_>>();
            if unique.len() != groups.len() || groups.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(mismatch(
                    "partition groups differ from prepared key groups".into(),
                ));
            }
            for partition in partitions {
                partition.root.validate(epoch)?;
            }
        }
        OperatorV4::Opaque { blocks, .. } => {
            for block in blocks {
                block.validate(epoch)?;
            }
        }
    }
    Ok(())
}

fn validate_ids<T>(
    kind: &str,
    actual: &BTreeMap<String, T>,
    expected: &BTreeSet<String>,
) -> Result<()> {
    if actual.keys().collect::<Vec<_>>() != expected.iter().collect::<Vec<_>>() {
        return Err(mismatch(format!("{kind} participants differ")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{JobKind, ManifestExpectationV4, V4Manifest, manifest_fingerprint};

    const GOLDEN: &str = include_str!("../../../tests/fixtures/checkpoint-v4/manifest-v4.json");

    fn expectation(manifest: &V4Manifest) -> ManifestExpectationV4 {
        ManifestExpectationV4 {
            pipeline_name: manifest.pipeline_name.clone(),
            pipeline_fingerprint: manifest.pipeline_fingerprint.clone(),
            recovery_fingerprint: manifest.recovery_fingerprint.clone(),
            job_kind: JobKind::Runner,
            epoch: manifest.epoch,
            source_ids: manifest.sources.keys().cloned().collect(),
            operator_ids: manifest.operators.keys().cloned().collect(),
            sink_ids: manifest.sinks.keys().cloned().collect(),
            static_inputs: manifest.static_inputs.clone(),
            table_partitions: [("rolling".to_owned(), vec![0, 1])].into_iter().collect(),
        }
    }

    #[test]
    fn golden_manifest_roundtrips_and_rejects_v3() {
        let manifest = V4Manifest::from_bytes(GOLDEN.as_bytes()).unwrap();
        assert_eq!(
            manifest.canonical_bytes().unwrap(),
            GOLDEN.trim_end().as_bytes()
        );
        manifest.validate(&expectation(&manifest)).unwrap();
        let v3 = GOLDEN.replace("\"format_version\":4", "\"format_version\":3");
        let error = V4Manifest::from_bytes(v3.as_bytes()).unwrap_err();
        assert!(matches!(
            error,
            crate::CalcFlowError::UnsupportedVersion {
                expected: 4,
                found: 3
            }
        ));
        assert_eq!(
            V4Manifest::v3_rejection_message(),
            "checkpoint manifest v3 is not restorable; start a new lineage"
        );
    }

    #[test]
    fn fingerprint_and_participant_mutations_fail_closed() {
        let manifest = V4Manifest::from_bytes(GOLDEN.as_bytes()).unwrap();
        let mut identity = manifest.clone();
        identity.pipeline_fingerprint = "a".repeat(64);
        identity.manifest_fingerprint = manifest_fingerprint(&identity).unwrap();
        assert!(identity.validate(&expectation(&manifest)).is_err());
        let mut participants = manifest.clone();
        participants.sources.remove("input");
        participants.manifest_fingerprint = manifest_fingerprint(&participants).unwrap();
        assert!(participants.validate(&expectation(&manifest)).is_err());
        let mut stale = manifest;
        stale.epoch += 1;
        assert!(V4Manifest::from_bytes(&serde_json::to_vec(&stale).unwrap()).is_err());
    }
}
