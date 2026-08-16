//! The transactional Parquet sink (feature `file`).
//!
//! Each epoch stages Parquet part files under a hidden staging directory
//! on the same filesystem as the target, publishes a manifest before
//! durable publication, and commits by atomically renaming the staged
//! epoch directory to its final `epoch=<n>` location. Committing the same
//! epoch again is idempotent, staging artifacts never count as committed
//! output, and unrelated user files outside the managed epoch directories
//! are never touched.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use async_trait::async_trait;
use calc_flow::{
    Batch, ConnectorError, ConnectorIdentity, ConnectorOperation, Epoch, FormatEncoder, JsonMap,
    Result, SinkRecovery, TransactionalStreamSink,
};
use serde_json::Value;

use crate::parquet::ParquetCodec;

/// The sink connector identity.
pub const IDENTITY_VERSION: &str = "2.0.0";

/// Data-only configuration for one transactional Parquet sink.
#[derive(Clone, Debug)]
pub struct FileSinkConfig {
    /// Root directory that owns `<output>/epoch=<n>` outputs.
    pub root: PathBuf,
    /// Stable output directory name under the root.
    pub output: String,
}

impl FileSinkConfig {
    /// Parses the sink configuration from connector options.
    ///
    /// # Errors
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] for a missing
    /// or non-string option.
    pub fn from_options(options: &JsonMap) -> Result<Self> {
        let root = required_string(options, "path")?;
        let output = required_string(options, "output")?;
        if output.contains('/') || output.contains('\\') || output == "." || output == ".." {
            return Err(calc_flow::CalcFlowError::InvalidArgument {
                field: "output".into(),
                message: "output must be a single directory name".into(),
            });
        }
        Ok(Self {
            root: PathBuf::from(root),
            output,
        })
    }

    fn output_dir(&self) -> PathBuf {
        self.root.join(&self.output)
    }

    fn staging_root(&self) -> PathBuf {
        self.output_dir().join(".staging")
    }

    fn staging_dir(&self, epoch: Epoch) -> PathBuf {
        self.staging_root()
            .join(format!("epoch={}", epoch.as_u64()))
    }

    fn final_dir(&self, epoch: Epoch) -> PathBuf {
        self.output_dir().join(format!("epoch={}", epoch.as_u64()))
    }
}

fn required_string(options: &JsonMap, key: &str) -> Result<String> {
    match options.get(key) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(calc_flow::CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option must be a string".into(),
        }),
        None => Err(calc_flow::CalcFlowError::InvalidArgument {
            field: key.into(),
            message: "option is required".into(),
        }),
    }
}

/// The transactional Parquet file sink.
pub struct TransactionalParquetSink {
    config: FileSinkConfig,
    codec: ParquetCodec,
    epoch: Option<Epoch>,
    part: u32,
    parts: Vec<String>,
    rows: u64,
}

impl TransactionalParquetSink {
    /// Builds the sink.
    ///
    /// # Errors
    ///
    /// Returns the codec construction error.
    pub fn new(config: FileSinkConfig) -> Result<Self> {
        Ok(Self {
            config,
            codec: ParquetCodec::new(crate::parquet::IDENTITY_VERSION)?,
            epoch: None,
            part: 0,
            parts: Vec::new(),
            rows: 0,
        })
    }

    fn identity() -> ConnectorIdentity {
        ConnectorIdentity::new("calc-flow-connectors", "file", IDENTITY_VERSION)
            .expect("the file sink identity is valid")
    }

    fn map_io(
        operation: &'static str,
        path: PathBuf,
    ) -> impl Fn(std::io::Error) -> calc_flow::CalcFlowError {
        move |error| Self::fail(operation, &path, &error.to_string())
    }

    fn fail(operation: &str, path: &Path, detail: &str) -> calc_flow::CalcFlowError {
        calc_flow::CalcFlowError::Connector(ConnectorError::new(
            Self::identity(),
            ConnectorOperation::new(operation).expect("operation name is non-empty"),
            &format!("{}: {detail}", path.display()),
        ))
    }

    async fn blocking<T, F>(&self, path: PathBuf, operation: &'static str, work: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce() -> std::result::Result<T, calc_flow::CalcFlowError> + Send + 'static,
    {
        tokio::task::spawn_blocking(work)
            .await
            .map_err(|error| Self::fail(operation, &path, &error.to_string()))?
    }

    fn manifest_evidence(&self, epoch: Epoch, parts: &[String], rows: u64) -> JsonMap {
        BTreeMap::from([
            (
                "output".to_string(),
                Value::String(self.config.output.clone()),
            ),
            ("epoch".to_string(), Value::from(epoch.as_u64())),
            (
                "parts".to_string(),
                Value::Array(
                    parts
                        .iter()
                        .map(|part| Value::String(part.clone()))
                        .collect(),
                ),
            ),
            ("rows".to_string(), Value::from(rows)),
        ])
    }
}

fn write_manifest(
    staging: &Path,
    evidence: &JsonMap,
) -> std::result::Result<(), calc_flow::CalcFlowError> {
    let encoded = serde_json::to_vec(evidence).map_err(|error| {
        TransactionalParquetSink::fail("pre_commit", staging, &error.to_string())
    })?;
    let manifest = staging.join("manifest.json");
    let temp = staging.join("manifest.json.tmp");
    std::fs::write(&temp, &encoded).map_err(|error| {
        TransactionalParquetSink::fail("pre_commit", &manifest, &error.to_string())
    })?;
    sync_file(&temp)?;
    std::fs::rename(&temp, &manifest).map_err(|error| {
        TransactionalParquetSink::fail("pre_commit", &manifest, &error.to_string())
    })?;
    sync_directory(staging)?;
    Ok(())
}

fn sync_file(path: &Path) -> std::result::Result<(), calc_flow::CalcFlowError> {
    let file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(path)
        .map_err(|error| TransactionalParquetSink::fail("pre_commit", path, &error.to_string()))?;
    file.sync_all()
        .map_err(|error| TransactionalParquetSink::fail("pre_commit", path, &error.to_string()))?;
    Ok(())
}

fn sync_directory(directory: &Path) -> std::result::Result<(), calc_flow::CalcFlowError> {
    // Windows cannot fsync a directory through a file handle; durability
    // there rests on the per-file syncs.
    #[cfg(unix)]
    {
        let file = std::fs::File::open(directory).map_err(|error| {
            TransactionalParquetSink::fail("pre_commit", directory, &error.to_string())
        })?;
        file.sync_all().map_err(|error| {
            TransactionalParquetSink::fail("pre_commit", directory, &error.to_string())
        })?;
    }
    #[cfg(not(unix))]
    let _ = directory;
    Ok(())
}

/// Commits one staged epoch: replays idempotently when the final
/// directory already carries matching manifest evidence, otherwise
/// renames the staged directory into place atomically.
fn commit_staged(staging: &Path, final_dir: &Path, evidence: &JsonMap) -> Result<()> {
    if final_dir.exists() {
        replay_committed_epoch(staging, final_dir, evidence)
    } else {
        rename_staged_epoch(staging, final_dir)
    }
}

fn replay_committed_epoch(staging: &Path, final_dir: &Path, evidence: &JsonMap) -> Result<()> {
    let committed = read_manifest(final_dir).ok_or_else(|| {
        TransactionalParquetSink::fail(
            "commit",
            final_dir,
            "committed epoch is missing its manifest",
        )
    })?;
    if committed != *evidence {
        return Err(TransactionalParquetSink::fail(
            "commit",
            final_dir,
            "committed epoch manifest disagrees with the replayed pre-commit evidence",
        ));
    }
    if staging.exists() {
        std::fs::remove_dir_all(staging).map_err(TransactionalParquetSink::map_io(
            "commit",
            staging.to_path_buf(),
        ))?;
    }
    Ok(())
}

fn rename_staged_epoch(staging: &Path, final_dir: &Path) -> Result<()> {
    if !staging.exists() {
        return Err(TransactionalParquetSink::fail(
            "commit",
            staging,
            "staged epoch is missing; nothing to commit",
        ));
    }
    std::fs::rename(staging, final_dir).map_err(TransactionalParquetSink::map_io(
        "commit",
        final_dir.to_path_buf(),
    ))?;
    sync_directory(final_dir.parent().unwrap_or(final_dir))
}

fn read_manifest(dir: &Path) -> Option<JsonMap> {
    let bytes = std::fs::read(dir.join("manifest.json")).ok()?;
    serde_json::from_slice(&bytes).ok()
}

#[async_trait]
impl TransactionalStreamSink for TransactionalParquetSink {
    async fn open(&mut self) -> Result<()> {
        let staging_root = self.config.staging_root();
        self.blocking(staging_root.clone(), "open", move || {
            std::fs::create_dir_all(&staging_root)
                .map_err(Self::map_io("open", staging_root.clone()))?;
            Ok(())
        })
        .await
    }

    async fn begin_epoch(&mut self, epoch: Epoch) -> Result<()> {
        let staging = self.config.staging_dir(epoch);
        let staging_root = self.config.staging_root();
        self.epoch = Some(epoch);
        self.part = 0;
        self.parts.clear();
        self.rows = 0;
        self.blocking(staging.clone(), "begin_epoch", move || {
            if staging.exists() {
                std::fs::remove_dir_all(&staging)
                    .map_err(Self::map_io("begin_epoch", staging.clone()))?;
            }
            std::fs::create_dir_all(&staging).map_err(Self::map_io("begin_epoch", staging_root))?;
            Ok(())
        })
        .await
    }

    async fn write(&mut self, batch: &Batch) -> Result<()> {
        let Some(epoch) = self.epoch else {
            return Err(Self::fail(
                "write",
                &self.config.output_dir(),
                "write before begin_epoch",
            ));
        };
        let encoded = self.codec.encode(batch)?;
        let path = self
            .config
            .staging_dir(epoch)
            .join(format!("part-{:04}.parquet", self.part));
        self.rows += u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
        self.part += 1;
        let part_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        self.parts.push(part_name);
        self.blocking(path.clone(), "write", move || {
            std::fs::write(&path, encoded).map_err(Self::map_io("write", path))
        })
        .await
    }

    async fn pre_commit(&mut self, epoch: Epoch) -> Result<JsonMap> {
        let Some(active) = self.epoch else {
            return Err(Self::fail(
                "pre_commit",
                &self.config.output_dir(),
                "pre_commit before begin_epoch",
            ));
        };
        if active != epoch {
            return Err(Self::fail(
                "pre_commit",
                &self.config.output_dir(),
                "pre_commit names an inactive epoch",
            ));
        }
        let staging = self.config.staging_dir(epoch);
        for part in &self.parts {
            let path = staging.join(part);
            self.blocking(path.clone(), "pre_commit", move || sync_file(&path))
                .await?;
        }
        let evidence = self.manifest_evidence(epoch, &self.parts, self.rows);
        let evidence_for_manifest = evidence.clone();
        self.blocking(staging.clone(), "pre_commit", move || {
            write_manifest(&staging, &evidence_for_manifest)
        })
        .await?;
        Ok(evidence)
    }

    async fn commit(&mut self, epoch: Epoch, pre_commit: &JsonMap) -> Result<()> {
        let staging = self.config.staging_dir(epoch);
        let final_dir = self.config.final_dir(epoch);
        let evidence = pre_commit.clone();
        self.blocking(final_dir.clone(), "commit", move || {
            commit_staged(&staging, &final_dir, &evidence)
        })
        .await
    }

    async fn abort(&mut self, epoch: Epoch, _pre_commit: Option<&JsonMap>) -> Result<()> {
        let staging = self.config.staging_dir(epoch);
        self.epoch = None;
        self.parts.clear();
        self.rows = 0;
        self.blocking(staging.clone(), "abort", move || {
            if staging.exists() {
                std::fs::remove_dir_all(&staging).map_err(|error| {
                    TransactionalParquetSink::fail("abort", &staging, &error.to_string())
                })?;
            }
            Ok(())
        })
        .await
    }

    async fn recover(&mut self, recovery: &SinkRecovery) -> Result<()> {
        let final_dir = self.config.final_dir(recovery.epoch());
        let evidence = recovery.pre_commit().clone();
        self.blocking(final_dir.clone(), "recover", move || {
            if final_dir.exists() {
                let committed = read_manifest(&final_dir).ok_or_else(|| {
                    TransactionalParquetSink::fail(
                        "recover",
                        &final_dir,
                        "committed epoch is missing its manifest",
                    )
                })?;
                if committed != evidence {
                    return Err(TransactionalParquetSink::fail(
                        "recover",
                        &final_dir,
                        "committed epoch manifest disagrees with the recovery evidence",
                    ));
                }
            }
            Ok(())
        })
        .await
    }

    async fn close(&mut self) -> Result<()> {
        Ok(())
    }
}
