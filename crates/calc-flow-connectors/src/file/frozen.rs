use super::{FileFormat, FileSource, FileSourceConfig};
use async_trait::async_trait;
use calc_flow::{
    Cursor, Result, SourceCapabilities, SourceEvent, SourceHistoryContext, SourceHistoryLimits,
    SourceHistoryManifestEntry, SourceHistorySpec, StreamSource,
};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    path::{Component, PathBuf},
};

const CONTRACT: &str = "file_frozen_v1";

/// A file snapshot archived in managed checkpoint storage before opening.
///
/// Recovery reads the archive even when the original files were changed or removed.
/// Use explicit source bindings with a managed `StreamingRunner`.
pub struct FrozenFileSource {
    inner: FileSource,
    limits: SourceHistoryLimits,
}

impl FrozenFileSource {
    /// Constructs a source without filesystem access.
    ///
    /// # Errors
    /// Returns a configuration error for invalid schema or decode limits.
    pub fn new(config: FileSourceConfig) -> Result<Self> {
        Self::with_limits(config, SourceHistoryLimits::default())
    }

    /// Constructs a source with explicit immutable-history limits.
    ///
    /// # Errors
    /// Returns a configuration error for invalid history or decode limits.
    pub fn with_limits(config: FileSourceConfig, limits: SourceHistoryLimits) -> Result<Self> {
        SourceHistorySpec::new(CONTRACT, limits)?;
        Ok(Self {
            inner: FileSource::new(config)?,
            limits,
        })
    }

    fn configuration(&self) -> Value {
        let (format, header) = match self.inner.config.format {
            FileFormat::Csv { header } => ("csv", header),
            FileFormat::JsonLines => ("json", false),
            FileFormat::Parquet => ("parquet", false),
        };
        json!({
            "format": format,
            "header": header,
            "schema": self.inner.config.schema,
            "max_batch_rows": self.inner.config.max_batch_rows,
            "max_batch_bytes": self.inner.config.max_batch_bytes,
            "max_file_bytes": self.inner.config.max_file_bytes,
        })
    }

    fn filenames(&self, history: &SourceHistoryManifestEntry) -> Result<Vec<PathBuf>> {
        let invalid = || calc_flow::CalcFlowError::CheckpointMismatch {
            message: "invalid frozen file history".into(),
        };
        if history.contract != CONTRACT
            || history.format_version != calc_flow::SOURCE_HISTORY_FORMAT_VERSION
            || history.inline_metadata.len() != 2
            || history.inline_metadata.get("configuration") != Some(&self.configuration())
        {
            return Err(invalid());
        }
        let files = history
            .inline_metadata
            .get("files")
            .and_then(Value::as_array)
            .ok_or_else(invalid)?;
        if files.len() != history.segments.len() || files.len() > self.limits.max_segments {
            return Err(invalid());
        }
        let mut paths = Vec::with_capacity(files.len());
        for (index, file) in files.iter().enumerate() {
            let name = file.as_str().ok_or_else(invalid)?;
            let path = PathBuf::from(name);
            if path.components().count() != 1
                || !matches!(path.components().next(), Some(Component::Normal(_)))
                || paths.last().is_some_and(|last| last >= &path)
                || history.segments[index].segment_id() != segment_id(index)
                || history.segments[index].byte_len() > self.inner.config.max_file_bytes
            {
                return Err(invalid());
            }
            paths.push(path);
        }
        Ok(paths)
    }

    async fn capture(&mut self, history: &SourceHistoryContext) -> Result<()> {
        let files = history
            .discover_files(
                &self.inner.config.path,
                self.inner.config.format.expected_extension(),
            )
            .await?;
        let mut names = Vec::with_capacity(files.len());
        for (index, path) in files.iter().enumerate() {
            let name = path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| FileSource::fail("open", path, "file identity must be UTF-8"))?;
            history
                .archive_file(&segment_id(index), path, self.inner.config.max_file_bytes)
                .await?;
            names.push(json!(name));
        }
        history.seal(BTreeMap::from([
            ("files".into(), Value::Array(names)),
            ("configuration".into(), self.configuration()),
        ]))
    }
}

pub(super) fn segment_id(index: usize) -> String {
    format!("file-{index:020}")
}

#[async_trait]
impl StreamSource for FrozenFileSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.inner.capabilities()
    }

    fn history_spec(&self) -> Option<SourceHistorySpec> {
        Some(SourceHistorySpec {
            contract: CONTRACT.into(),
            limits: self.limits,
        })
    }

    fn validate_history(&self, history: &SourceHistoryManifestEntry) -> Result<()> {
        self.filenames(history).map(|_| ())
    }

    async fn prepare_history(&mut self, history: SourceHistoryContext) -> Result<()> {
        if history.manifest().is_none() {
            self.capture(&history).await?;
        }
        self.inner.files = self.filenames(&history.manifest().ok_or_else(|| {
            calc_flow::CalcFlowError::Internal {
                message: "missing frozen descriptor".into(),
            }
        })?)?;
        self.inner.history = Some(history);
        Ok(())
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if self.inner.history.is_none() {
            return Err(FileSource::fail(
                "open",
                &self.inner.config.path,
                "frozen history requires managed runtime preparation",
            ));
        }
        self.inner.open(cursor).await
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        self.inner.next().await
    }

    async fn close(&mut self) -> Result<()> {
        self.inner.close().await
    }
}
