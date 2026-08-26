use async_trait::async_trait;
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};

use crate::{CalcFlowError, Epoch, Result, json::validate_portable_identifier};

/// An immutable reference to one committed state segment.
///
/// The relative path is a portable path below the backend's committed
/// subtree. Logical identities remain metadata and are never interpreted as
/// caller-selected filesystem paths.
#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct StateHandle {
    /// Stable logical operator or sink identity.
    operator_id: String,
    /// Checkpoint epoch that created the segment.
    epoch: Epoch,
    /// Stable segment identity within the operator epoch.
    segment_id: String,
    /// Portable path below the managed state root.
    relative_path: String,
    /// Exact committed byte length.
    byte_len: u64,
    /// Lowercase hexadecimal SHA-256 of the committed bytes.
    sha256: String,
}

impl StateHandle {
    /// Constructs and validates an immutable committed-segment handle.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when an identity, relative
    /// path, or checksum is not portable and canonical.
    pub fn new(
        operator_id: &str,
        epoch: Epoch,
        segment_id: &str,
        relative_path: &str,
        byte_len: u64,
        sha256: &str,
    ) -> Result<Self> {
        validate_portable_identifier("operator_id", operator_id)?;
        validate_portable_identifier("segment_id", segment_id)?;
        validate_committed_relative_path(relative_path)?;
        validate_sha256("sha256", sha256)?;
        Ok(Self {
            operator_id: operator_id.into(),
            epoch,
            segment_id: segment_id.into(),
            relative_path: relative_path.into(),
            byte_len,
            sha256: sha256.into(),
        })
    }

    /// Returns the stable logical state-owner identity.
    pub fn operator_id(&self) -> &str {
        &self.operator_id
    }

    /// Returns the checkpoint epoch that created the segment.
    pub const fn epoch(&self) -> Epoch {
        self.epoch
    }

    /// Returns the stable segment identity within the operator epoch.
    pub fn segment_id(&self) -> &str {
        &self.segment_id
    }

    /// Returns the portable path below the managed state root.
    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// Returns the exact committed byte length.
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Returns the lowercase hexadecimal SHA-256 of the committed bytes.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Validates that this handle belongs to one expected state owner and epoch.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the handle itself is
    /// non-canonical, or [`CalcFlowError::CheckpointMismatch`] before any
    /// segment load when either ownership coordinate differs.
    pub fn validate_for(&self, expected_owner: &str, expected_epoch: Epoch) -> Result<()> {
        self.validate_owner(expected_owner)?;
        if self.epoch != expected_epoch {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "state handle epoch {} does not match expected epoch {}",
                    self.epoch.as_u64(),
                    expected_epoch.as_u64()
                ),
            });
        }
        Ok(())
    }

    /// Validates that this handle belongs to one expected state owner.
    ///
    /// Recovery loads validate ownership with this method rather than by
    /// epoch: a carried segment keeps the epoch that created it, so one
    /// manifest legitimately references handles committed at different epochs.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the handle itself is
    /// non-canonical, or [`CalcFlowError::CheckpointMismatch`] before any
    /// segment load when the owner differs.
    pub(crate) fn validate_owner(&self, expected_owner: &str) -> Result<()> {
        validate_portable_identifier("operator_id", &self.operator_id)?;
        validate_portable_identifier("segment_id", &self.segment_id)?;
        validate_committed_relative_path(&self.relative_path)?;
        validate_sha256("sha256", &self.sha256)?;
        if self.operator_id != expected_owner {
            return Err(CalcFlowError::CheckpointMismatch {
                message: format!(
                    "state handle owner {:?} does not match expected owner {expected_owner:?}",
                    self.operator_id
                ),
            });
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for StateHandle {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            operator_id: String,
            epoch: Epoch,
            segment_id: String,
            relative_path: String,
            byte_len: u64,
            sha256: String,
        }

        let fields = Fields::deserialize(deserializer)?;
        Self::new(
            &fields.operator_id,
            fields.epoch,
            &fields.segment_id,
            &fields.relative_path,
            fields.byte_len,
            &fields.sha256,
        )
        .map_err(D::Error::custom)
    }
}

/// Validated identity of one continuous-job state lineage.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StateLineageKey {
    pipeline_name: String,
    pipeline_fingerprint: String,
}

impl StateLineageKey {
    /// Constructs a validated lineage key.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] when the pipeline identity
    /// is not portable or the semantic fingerprint is not lowercase SHA-256.
    pub fn new(pipeline_name: &str, pipeline_fingerprint: &str) -> Result<Self> {
        validate_portable_identifier("pipeline_name", pipeline_name)?;
        validate_sha256("pipeline_fingerprint", pipeline_fingerprint)?;
        Ok(Self {
            pipeline_name: pipeline_name.into(),
            pipeline_fingerprint: pipeline_fingerprint.into(),
        })
    }

    /// Returns the logical pipeline name.
    pub fn pipeline_name(&self) -> &str {
        &self.pipeline_name
    }

    /// Returns the semantic pipeline fingerprint.
    pub fn pipeline_fingerprint(&self) -> &str {
        &self.pipeline_fingerprint
    }
}

/// Backend-neutral factory for exclusive lineage sessions.
#[async_trait]
pub trait StateBackend: Send + Sync {
    /// Opens the only segment-operation surface for one lineage.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Conflict`] when the backend identity and
    /// lineage are already exclusively leased.
    async fn open_lineage(&self, key: &StateLineageKey) -> Result<Box<dyn StateLineageBackend>>;
}

/// Segment operations available only while one lineage lease is owned.
#[async_trait]
pub trait StateLineageBackend: Send + Sync {
    /// Returns the deterministic backend identity hash.
    fn identity_hash(&self) -> &str;

    /// Writes bytes to the managed staging subtree.
    async fn stage_segment(&self, handle: &StateHandle, bytes: &[u8]) -> Result<()>;

    /// Re-reads and validates a staged segment's length and checksum.
    async fn validate_segment(&self, handle: &StateHandle) -> Result<()>;

    /// Atomically publishes one previously validated segment.
    async fn publish_segment(&self, handle: &StateHandle) -> Result<()>;

    /// Loads committed bytes after validating length and checksum.
    async fn load_segment(&self, handle: &StateHandle) -> Result<Vec<u8>>;

    /// Collects committed segments unreachable from the retained handles.
    async fn collect_orphans(&self, retained: &[StateHandle]) -> Result<usize>;
}

pub(crate) fn validate_sha256(field: &str, value: &str) -> Result<()> {
    if value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        Ok(())
    } else {
        Err(CalcFlowError::InvalidArgument {
            field: field.into(),
            message: "must be exactly 64 lowercase hexadecimal SHA-256 characters".into(),
        })
    }
}

fn validate_committed_relative_path(value: &str) -> Result<()> {
    let components = value.split('/').collect::<Vec<_>>();
    let valid_component = |component: &&str| {
        !component.is_empty()
            && *component != "."
            && *component != ".."
            && component
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.'))
    };
    if value.contains('\0')
        || value.contains('\\')
        || components.len() < 2
        || components.first() != Some(&"committed")
        || !components.iter().all(valid_component)
    {
        Err(CalcFlowError::InvalidArgument {
            field: "relative_path".into(),
            message: "must be a portable path inside the committed subtree".into(),
        })
    } else {
        Ok(())
    }
}
