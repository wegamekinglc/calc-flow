//! Native selection of the highest complete runner checkpoint-v4 manifest.

use std::{collections::BTreeMap, path::Path};

use crate::state::PreparedManifestIdentity;
use crate::state::v4::{JobKind, ManifestExpectationV4, V4Manifest, mismatch};
use crate::{Epoch, Result};

#[derive(Debug)]
pub(in crate::runtime::streaming) struct SelectedRunnerV4 {
    pub manifest: V4Manifest,
    pub next_epoch: Epoch,
    pub runtime_config_changed: bool,
}

pub(super) async fn select_latest(
    manifest_root: &Path,
    identity: &PreparedManifestIdentity,
) -> Result<Option<SelectedRunnerV4>> {
    reject_legacy_manifests(manifest_root).await?;
    let directory = manifest_root.join("lineage/manifests");
    let mut entries = match tokio::fs::read_dir(&directory).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => {
            return Err(crate::CalcFlowError::Io {
                path: directory.to_string_lossy().into_owned(),
                source: error,
            });
        }
    };
    let mut selected: Option<(u64, V4Manifest)> = None;
    while let Some(entry) =
        entries
            .next_entry()
            .await
            .map_err(|error| crate::CalcFlowError::Io {
                path: directory.to_string_lossy().into_owned(),
                source: error,
            })?
    {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let Some(epoch) = name
            .strip_prefix("epoch-")
            .and_then(|value| value.strip_suffix(".json"))
            .and_then(|value| value.parse::<u64>().ok())
        else {
            continue;
        };
        let document = tokio::fs::read(&path)
            .await
            .map_err(|error| crate::CalcFlowError::Io {
                path: path.to_string_lossy().into_owned(),
                source: error,
            })?;
        let manifest = V4Manifest::from_bytes(&document)?;
        if manifest.job_kind != JobKind::Runner || manifest.recovery_status != "final" {
            return Err(mismatch("selected runner manifest is not complete".into()));
        }
        manifest.validate(&ManifestExpectationV4 {
            pipeline_name: identity.pipeline_name.clone(),
            pipeline_fingerprint: identity.pipeline_fingerprint.clone(),
            recovery_fingerprint: identity.pipeline_fingerprint.clone(),
            job_kind: JobKind::Runner,
            epoch,
            source_ids: identity.source_ids.clone(),
            operator_ids: identity.operator_ids.clone(),
            sink_ids: identity.sink_ids.clone(),
            static_inputs: identity.static_inputs.clone(),
            table_partitions: BTreeMap::new(),
        })?;
        if selected
            .as_ref()
            .is_none_or(|(current, _)| epoch > *current)
        {
            selected = Some((epoch, manifest));
        }
    }
    selected
        .map(|(epoch, manifest)| {
            let next_epoch = Epoch::new(epoch.checked_add(1).ok_or_else(|| {
                mismatch("selected checkpoint epoch overflows the next epoch".into())
            })?)
            .expect("epoch plus one is nonzero");
            Ok(SelectedRunnerV4 {
                runtime_config_changed: manifest.runtime_config_hash
                    != identity.runtime_config_hash,
                manifest,
                next_epoch,
            })
        })
        .transpose()
}

async fn reject_legacy_manifests(manifest_root: &Path) -> Result<()> {
    let mut entries = match tokio::fs::read_dir(manifest_root).await {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(crate::CalcFlowError::Io {
                path: manifest_root.to_string_lossy().into_owned(),
                source: error,
            });
        }
    };
    while let Some(entry) =
        entries
            .next_entry()
            .await
            .map_err(|error| crate::CalcFlowError::Io {
                path: manifest_root.to_string_lossy().into_owned(),
                source: error,
            })?
    {
        let path = entry.path();
        if path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| {
                name.starts_with("manifest-")
                    && Path::new(name)
                        .extension()
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
            })
        {
            return Err(mismatch(V4Manifest::v3_rejection_message().into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use chrono::Utc;

    use super::{select_latest, *};
    use crate::JsonMap;
    use crate::state::v4::{OperatorV4, SinkV4, SourceV4, V4Store, manifest_fingerprint};
    use crate::state::{SinkDeliveryManifest, SourceWatermarkManifestState};

    fn identity() -> PreparedManifestIdentity {
        PreparedManifestIdentity {
            pipeline_name: "pipeline".into(),
            pipeline_fingerprint: "fingerprint".into(),
            runtime_config_hash: "runtime".into(),
            source_ids: BTreeSet::from(["source".into()]),
            operator_ids: BTreeSet::from(["operator".into()]),
            sink_ids: BTreeSet::from(["sink".into()]),
            static_inputs: BTreeMap::new(),
        }
    }

    fn manifest(identity: &PreparedManifestIdentity, epoch: u64) -> V4Manifest {
        let mut manifest = V4Manifest {
            format_version: 4,
            pipeline_name: identity.pipeline_name.clone(),
            pipeline_fingerprint: identity.pipeline_fingerprint.clone(),
            runtime_config_hash: identity.runtime_config_hash.clone(),
            epoch,
            created_at: Utc::now(),
            recovery_status: "final".into(),
            job_kind: JobKind::Runner,
            recovery_fingerprint: identity.pipeline_fingerprint.clone(),
            sources: BTreeMap::from([(
                "source".into(),
                SourceV4 {
                    cursor: None,
                    identity_hash: "identity".into(),
                    next_sequence: 0,
                    ended: false,
                    watermark_policy: SourceWatermarkManifestState::Disabled { idle: false },
                    history: None,
                },
            )]),
            operators: BTreeMap::from([(
                "operator".into(),
                OperatorV4::Opaque {
                    progress: BTreeMap::new(),
                    output_frontier: None,
                    state_fingerprint: "state".into(),
                    inline_metadata: JsonMap::new(),
                    blocks: Vec::new(),
                },
            )]),
            sinks: BTreeMap::from([(
                "sink".into(),
                SinkV4 {
                    delivery: SinkDeliveryManifest::Ordinary,
                    pre_commit: None,
                    blocks: Vec::new(),
                },
            )]),
            static_inputs: BTreeMap::new(),
            manifest_fingerprint: String::new(),
        };
        manifest.manifest_fingerprint = manifest_fingerprint(&manifest).unwrap();
        manifest
    }

    #[tokio::test]
    async fn selects_highest_complete_v4_epoch() {
        let root = tempfile::tempdir().unwrap();
        let identity = identity();
        let mut store = V4Store::new(root.path().join("lineage"));
        store.publish_manifest(&manifest(&identity, 1)).unwrap();
        store.publish_manifest(&manifest(&identity, 2)).unwrap();
        let selected = select_latest(root.path(), &identity)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(selected.manifest.epoch, 2);
        assert_eq!(selected.next_epoch.as_u64(), 3);
    }

    #[tokio::test]
    async fn rejects_legacy_manifest_instead_of_restoring_it() {
        let root = tempfile::tempdir().unwrap();
        tokio::fs::write(
            root.path().join("manifest-00000000000000000001.json"),
            b"{}",
        )
        .await
        .unwrap();
        let error = select_latest(root.path(), &identity()).await.unwrap_err();
        assert!(matches!(
            error,
            crate::CalcFlowError::CheckpointMismatch { .. }
        ));
    }
}
