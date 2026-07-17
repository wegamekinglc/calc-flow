use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize, de::Error as _};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::json::{parse_json_value, validate_json_depth, validate_json_depth_at};
use crate::project_store::{WriteMode, atomic_write, bounded_read, delete_file};
use crate::{CalcFlowError, Result};

/// The only checkpoint format accepted by the Rust v2 runtime.
pub const CHECKPOINT_FORMAT_VERSION: u32 = 2;
/// Maximum accepted size of one stored checkpoint document.
pub const MAX_CHECKPOINT_DOCUMENT_BYTES: usize = 10 * 1024 * 1024;

/// Durable state committed after source data has reached every sink.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Checkpoint {
    pub format_version: u32,
    pub pipeline_name: String,
    pub pipeline_fingerprint: String,
    pub source_cursor: Option<Value>,
    pub sequence: u64,
    pub state: BTreeMap<String, Value>,
    pub created_at: DateTime<Utc>,
}

impl Checkpoint {
    /// Constructs a validated v2 checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::InvalidArgument`] for empty identity fields or state keys.
    pub fn new(
        pipeline_name: &str,
        pipeline_fingerprint: &str,
        source_cursor: Option<Value>,
        sequence: u64,
        state: BTreeMap<String, Value>,
        created_at: DateTime<Utc>,
    ) -> Result<Self> {
        let checkpoint = Self {
            format_version: CHECKPOINT_FORMAT_VERSION,
            pipeline_name: pipeline_name.into(),
            pipeline_fingerprint: pipeline_fingerprint.into(),
            source_cursor: source_cursor.filter(|value| !value.is_null()),
            sequence,
            state,
            created_at,
        };
        checkpoint.validate()?;
        Ok(checkpoint)
    }

    fn validate(&self) -> Result<()> {
        if self.format_version != CHECKPOINT_FORMAT_VERSION {
            return Err(CalcFlowError::UnsupportedVersion {
                expected: CHECKPOINT_FORMAT_VERSION,
                found: self.format_version,
            });
        }
        if self.pipeline_name.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "pipeline_name".into(),
                message: "must not be empty".into(),
            });
        }
        if self.pipeline_fingerprint.is_empty() {
            return Err(CalcFlowError::InvalidArgument {
                field: "pipeline_fingerprint".into(),
                message: "must not be empty".into(),
            });
        }
        if self.source_cursor.as_ref().is_some_and(Value::is_null) {
            return Err(CalcFlowError::InvalidArgument {
                field: "source_cursor".into(),
                message: "use None instead of a JSON null cursor".into(),
            });
        }
        if self.state.keys().any(String::is_empty) {
            return Err(CalcFlowError::InvalidArgument {
                field: "state".into(),
                message: "node IDs must not be empty".into(),
            });
        }
        if let Some(source_cursor) = &self.source_cursor {
            validate_json_depth(source_cursor, "checkpoint source cursor")?;
        }
        for (node_id, state) in &self.state {
            validate_json_depth_at(state, &format!("checkpoint state for node {node_id:?}"), 1)?;
        }
        Ok(())
    }
}

impl<'de> Deserialize<'de> for Checkpoint {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Fields {
            #[serde(deserialize_with = "deserialize_checkpoint_version")]
            format_version: u32,
            pipeline_name: String,
            pipeline_fingerprint: String,
            #[serde(default, deserialize_with = "deserialize_present_nullable_value")]
            source_cursor: NullableValueField,
            sequence: u64,
            state: BTreeMap<String, Value>,
            created_at: DateTime<Utc>,
        }

        let fields = Fields::deserialize(deserializer)?;
        let source_cursor = match fields.source_cursor {
            NullableValueField::Missing => {
                return Err(D::Error::missing_field("source_cursor"));
            }
            NullableValueField::Present(value) => value,
        };
        let checkpoint = Self {
            format_version: fields.format_version,
            pipeline_name: fields.pipeline_name,
            pipeline_fingerprint: fields.pipeline_fingerprint,
            source_cursor,
            sequence: fields.sequence,
            state: fields.state,
            created_at: fields.created_at,
        };
        checkpoint.validate().map_err(D::Error::custom)?;
        Ok(checkpoint)
    }
}

fn deserialize_present_nullable_value<'de, D>(
    deserializer: D,
) -> std::result::Result<NullableValueField, D::Error>
where
    D: Deserializer<'de>,
{
    Option::<Value>::deserialize(deserializer).map(NullableValueField::Present)
}

#[derive(Default)]
enum NullableValueField {
    #[default]
    Missing,
    Present(Option<Value>),
}

/// Asynchronous checkpoint persistence contract.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    /// Loads a checkpoint by pipeline name, returning `None` when absent.
    async fn load(&self, pipeline_name: &str) -> Result<Option<Checkpoint>>;
    /// Atomically creates or replaces a checkpoint.
    async fn save(&self, checkpoint: &Checkpoint) -> Result<()>;
    /// Deletes a checkpoint. Missing checkpoints are an idempotent success.
    async fn delete(&self, pipeline_name: &str) -> Result<()>;
}

/// Atomic file-backed checkpoint storage using SHA-256 filenames.
#[derive(Clone, Debug)]
pub struct FileCheckpointStore {
    directory: PathBuf,
}

impl FileCheckpointStore {
    /// Creates the checkpoint directory and resolves it to a canonical root.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Io`] if the directory cannot be created or resolved.
    pub async fn new(directory: impl AsRef<Path>) -> Result<Self> {
        let requested = directory.as_ref().to_owned();
        tokio::fs::create_dir_all(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        let directory = tokio::fs::canonicalize(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        Ok(Self { directory })
    }

    fn path_for(&self, pipeline_name: &str) -> PathBuf {
        let digest = Sha256::digest(pipeline_name.as_bytes());
        self.directory.join(format!("{}.json", hex::encode(digest)))
    }
}

#[async_trait]
impl CheckpointStore for FileCheckpointStore {
    async fn load(&self, pipeline_name: &str) -> Result<Option<Checkpoint>> {
        let path = self.path_for(pipeline_name);
        let Some(bytes) = bounded_read(path, MAX_CHECKPOINT_DOCUMENT_BYTES).await? else {
            return Ok(None);
        };
        let value = parse_json_value(&bytes, "checkpoint document")?;
        if !value.is_object() {
            return Err(format_error(
                "checkpoint document must contain an object".into(),
            ));
        }
        reject_checkpoint_version(&value)?;
        let checkpoint: Checkpoint =
            serde_json::from_value(value).map_err(|error| format_error(error.to_string()))?;
        checkpoint.validate()?;
        if checkpoint.pipeline_name != pipeline_name {
            return Err(format_error(format!(
                "stored checkpoint pipeline name {:?} does not match key {pipeline_name:?}",
                checkpoint.pipeline_name
            )));
        }
        Ok(Some(checkpoint))
    }

    async fn save(&self, checkpoint: &Checkpoint) -> Result<()> {
        checkpoint.validate()?;
        let mut bytes = canonical_pretty_json(checkpoint)?.into_bytes();
        bytes.push(b'\n');
        if bytes.len() > MAX_CHECKPOINT_DOCUMENT_BYTES {
            return Err(format_error(format!(
                "checkpoint exceeds the {MAX_CHECKPOINT_DOCUMENT_BYTES}-byte limit"
            )));
        }
        atomic_write(
            self.directory.clone(),
            self.path_for(&checkpoint.pipeline_name),
            bytes,
            WriteMode::Replace,
            "checkpoint",
            &checkpoint.pipeline_name,
        )
        .await
    }

    async fn delete(&self, pipeline_name: &str) -> Result<()> {
        delete_file(
            self.directory.clone(),
            self.path_for(pipeline_name),
            true,
            "checkpoint",
            pipeline_name,
        )
        .await
    }
}

fn canonical_pretty_json(value: &impl Serialize) -> Result<String> {
    let value = serde_json::to_value(value).map_err(|error| format_error(error.to_string()))?;
    let compact = crate::canonical_json(&value)?;
    let sorted: Value =
        serde_json::from_str(&compact).map_err(|error| format_error(error.to_string()))?;
    serde_json::to_string_pretty(&sorted).map_err(|error| format_error(error.to_string()))
}

fn reject_checkpoint_version(value: &Value) -> Result<()> {
    let Some(version) = value.get("format_version").and_then(Value::as_u64) else {
        return Ok(());
    };
    let version = u32::try_from(version)
        .map_err(|_| format_error("checkpoint format version is outside the u32 range".into()))?;
    if version != CHECKPOINT_FORMAT_VERSION {
        return Err(CalcFlowError::UnsupportedVersion {
            expected: CHECKPOINT_FORMAT_VERSION,
            found: version,
        });
    }
    Ok(())
}

fn deserialize_checkpoint_version<'de, D>(deserializer: D) -> std::result::Result<u32, D::Error>
where
    D: Deserializer<'de>,
{
    let version = u32::deserialize(deserializer)?;
    if version == CHECKPOINT_FORMAT_VERSION {
        Ok(version)
    } else {
        Err(D::Error::custom(format!(
            "checkpoint format version {version} is unsupported; expected {CHECKPOINT_FORMAT_VERSION}"
        )))
    }
}

fn format_error(message: String) -> CalcFlowError {
    CalcFlowError::Format { message }
}

fn io_error(path: &Path, source: std::io::Error) -> CalcFlowError {
    CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    }
}
