//! The file/directory snapshot source (feature `file`).
//!
//! The source scans one file or one flat directory of regular files,
//! orders them by stable file identity (lexicographic file name), and
//! emits bounded immutable batches through the registered format codecs.
//! Replay cursors carry the file identity plus the consumed row offset;
//! newline JSON advances per bounded chunk, while CSV and Parquet emit
//! one bounded batch per file so checkpoint cuts always land on file
//! boundaries. Discovery fails closed on symlinks, subdirectories,
//! unexpected extensions, path traversal, and files above the configured
//! byte ceiling.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use calc_flow::{
    ArrowFieldSpec, Batch, ConnectorError, ConnectorIdentity, ConnectorOperation, Cursor,
    DecodeBounds, FormatDecoder, Result, SourceCapabilities, SourceEvent, SourceSchema,
    StreamSource,
};
use serde_json::Value;

mod frozen;
pub use frozen::FrozenFileSource;

use crate::arrow_schema::{codec_connector_identity, schema_from_spec};
use crate::csv::CsvCodec;
use crate::json_lines::JsonLinesCodec;
use crate::options::{bool_option, required_str, u64_option};

/// Default ceiling for one decoded file payload.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 256 * 1024 * 1024;
/// Default row bound for one decoded batch; fits the default edge
/// budget of 10,000 rows.
pub const DEFAULT_MAX_BATCH_ROWS: u64 = 8_192;
/// Default byte bound for one decoded batch.
pub const DEFAULT_MAX_BATCH_BYTES: u64 = 8 * 1024 * 1024;

/// The selected wire format and its codec options.
#[derive(Clone, Debug)]
pub enum FileFormat {
    /// RFC 4180-style CSV with an optional header record.
    Csv { header: bool },
    /// Newline-delimited JSON objects.
    JsonLines,
    /// Parquet columnar files.
    Parquet,
}

impl FileFormat {
    /// Parses the data-only format vocabulary.
    ///
    /// # Errors
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] for an
    /// unknown format name.
    pub fn parse(value: &str, header: bool) -> Result<Self> {
        match value {
            "csv" => Ok(Self::Csv { header }),
            "json" => Ok(Self::JsonLines),
            "parquet" => Ok(Self::Parquet),
            other => Err(calc_flow::CalcFlowError::InvalidArgument {
                field: "format".into(),
                message: format!("unsupported file format {other:?}"),
            }),
        }
    }

    fn expected_extension(&self) -> &'static str {
        match self {
            Self::Csv { .. } => "csv",
            Self::JsonLines => "json",
            Self::Parquet => "parquet",
        }
    }
}

/// Data-only configuration for one file snapshot source.
#[derive(Clone, Debug)]
pub struct FileSourceConfig {
    /// File or flat directory to scan.
    pub path: PathBuf,
    /// Wire format of the scanned files.
    pub format: FileFormat,
    /// Optional explicit Arrow schema every file must match.
    pub schema: Vec<ArrowFieldSpec>,
    /// Row bound of one decoded batch.
    pub max_batch_rows: u64,
    /// Byte bound of one decoded batch.
    pub max_batch_bytes: u64,
    /// Ceiling for one decoded file payload.
    pub max_file_bytes: u64,
}

impl FileSourceConfig {
    /// Parses the configuration from connector options.
    ///
    /// # Errors
    ///
    /// Option-key admission against the registered descriptor happens in
    /// the core registry; this parser validates the values it reads.
    ///
    /// Returns [`calc_flow::CalcFlowError::InvalidArgument`] naming the
    /// offending option for a missing path or format, a path containing
    /// `..` traversal, or a non-positive bound.
    pub fn from_options(options: &calc_flow::JsonMap) -> Result<Self> {
        let path = parse_path(options)?;
        let format = FileFormat::parse(
            required_str(options, "format")?,
            bool_option(options, "header")?.unwrap_or(true),
        )?;
        let (max_batch_rows, max_batch_bytes, max_file_bytes) = parse_bounds(options)?;
        Ok(Self {
            path,
            format,
            schema: parse_schema(options)?,
            max_batch_rows,
            max_batch_bytes,
            max_file_bytes,
        })
    }

    fn bounds(&self) -> Result<DecodeBounds> {
        DecodeBounds::new(self.max_batch_rows, self.max_batch_bytes)
    }
}

fn parse_bounds(options: &calc_flow::JsonMap) -> Result<(u64, u64, u64)> {
    Ok((
        u64_option(options, "max_batch_rows")?.unwrap_or(DEFAULT_MAX_BATCH_ROWS),
        u64_option(options, "max_batch_bytes")?.unwrap_or(DEFAULT_MAX_BATCH_BYTES),
        u64_option(options, "max_file_bytes")?.unwrap_or(DEFAULT_MAX_FILE_BYTES),
    ))
}

fn parse_path(options: &calc_flow::JsonMap) -> Result<PathBuf> {
    let path = PathBuf::from(required_str(options, "path")?);
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir | Component::CurDir))
    {
        return Err(calc_flow::CalcFlowError::InvalidArgument {
            field: "path".into(),
            message: "file source paths must name a file or directory without traversal".into(),
        });
    }
    Ok(path)
}

fn parse_schema(options: &calc_flow::JsonMap) -> Result<Vec<ArrowFieldSpec>> {
    match options.get("schema") {
        None => Ok(Vec::new()),
        Some(value) => {
            serde_json::from_value::<Vec<ArrowFieldSpec>>(value.clone()).map_err(|error| {
                calc_flow::CalcFlowError::InvalidArgument {
                    field: "schema".into(),
                    message: format!("schema must be a field list: {error}"),
                }
            })
        }
    }
}

/// The finite file snapshot source.
pub struct FileSource {
    config: FileSourceConfig,
    capabilities: SourceCapabilities,
    files: Vec<PathBuf>,
    file_index: usize,
    row_offset: u64,
    line_cache: Option<JsonLineCache>,
    history: Option<calc_flow::SourceHistoryContext>,
    sequence: u64,
}

enum FileBytes {
    Local(Vec<u8>),
    History(calc_flow::SourceHistoryBytes),
}

impl FileBytes {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Local(bytes) => bytes,
            Self::History(bytes) => bytes.bytes(),
        }
    }
}

struct JsonLineCache {
    bytes: FileBytes,
    offset: usize,
}

impl JsonLineCache {
    fn next_line(&self, offset: usize) -> Option<(usize, &[u8])> {
        let bytes = self.bytes.bytes();
        let mut offset = offset;
        while offset < bytes.len() {
            let end = bytes[offset..]
                .iter()
                .position(|byte| *byte == b'\n')
                .map_or(bytes.len(), |end| offset + end);
            let next = end.saturating_add(1).min(bytes.len());
            if end > offset {
                return Some((next, &bytes[offset..end]));
            }
            offset = next;
        }
        None
    }
}

impl FileSource {
    /// Builds the source and freezes its preflight capabilities.
    ///
    /// # Errors
    ///
    /// Returns the configuration error for invalid bounds or schema
    /// specifications.
    pub fn new(config: FileSourceConfig) -> Result<Self> {
        let schema = if config.schema.is_empty() {
            SourceSchema::DynamicOrUnknown
        } else {
            SourceSchema::Exact(schema_from_spec(&config.schema)?)
        };
        let bounds = config.bounds()?;
        let capabilities = SourceCapabilities {
            replay_positioning: calc_flow::ReplayPositioning::ExactPauseReportAndSeek,
            delivery: calc_flow::SourceDeliveryCapability::Lossless,
            max_batch_rows: usize::try_from(bounds.max_rows).unwrap_or(usize::MAX),
            max_batch_bytes: usize::try_from(bounds.max_bytes).unwrap_or(usize::MAX),
            schema,
            native_watermarks: calc_flow::NativeWatermarkCapability::NeverEmits,
        };
        Ok(Self {
            config,
            capabilities,
            files: Vec::new(),
            file_index: 0,
            row_offset: 0,
            line_cache: None,
            history: None,
            sequence: 0,
        })
    }

    fn identity() -> ConnectorIdentity {
        codec_connector_identity(
            &calc_flow::FormatIdentity::new("file", "2.0.0")
                .expect("the file connector identity is non-empty"),
        )
    }

    fn fail(operation: &str, path: &Path, detail: &str) -> calc_flow::CalcFlowError {
        calc_flow::CalcFlowError::Connector(ConnectorError::new(
            Self::identity(),
            ConnectorOperation::new(operation).expect("operation name is non-empty"),
            &format!("{}: {detail}", path.display()),
        ))
    }

    async fn discover(&mut self) -> Result<()> {
        let path = self.config.path.clone();
        let expected = self.config.format.expected_extension();
        let discover_path = path.clone();
        let files = tokio::task::spawn_blocking(move || discover_files(&discover_path, expected))
            .await
            .map_err(|error| Self::fail("open", &path, &error.to_string()))??;
        self.files = files;
        Ok(())
    }

    fn cursor_for(&self, file_index: usize, row: u64) -> Result<Cursor> {
        let sequence = self
            .sequence
            .checked_add(1)
            .ok_or_else(|| Self::fail("read", &self.config.path, "batch sequence exhausted"))?;
        let name = self
            .files
            .get(file_index)
            .and_then(|path| path.file_name())
            .and_then(|name| name.to_str())
            .unwrap_or_default()
            .to_string();
        let mut order = Vec::with_capacity(16);
        order.extend_from_slice(&(file_index as u64).to_be_bytes());
        order.extend_from_slice(&row.to_be_bytes());
        let payload = BTreeMap::from([
            ("file".to_string(), Value::String(name)),
            ("row".to_string(), Value::from(row)),
            ("sequence".to_string(), Value::from(sequence)),
        ]);
        Cursor::unbound(order, payload)
    }

    async fn read_file(&self, path: &Path) -> Result<FileBytes> {
        if let Some(history) = &self.history {
            return history
                .load(&frozen::segment_id(self.file_index))
                .await
                .map(FileBytes::History);
        }
        let detail_path = path.to_path_buf();
        let ceiling = self.config.max_file_bytes;
        tokio::task::spawn_blocking(move || {
            use std::io::Read;
            let mut file = std::fs::File::open(&detail_path)
                .map_err(|error| Self::fail("read", &detail_path, &error.to_string()))?;
            let mut bytes = Vec::new();
            let mut probe = vec![0u8; 64 * 1024];
            loop {
                let read = file
                    .read(&mut probe)
                    .map_err(|error| Self::fail("read", &detail_path, &error.to_string()))?;
                if read == 0 {
                    break;
                }
                if u64::try_from(bytes.len() + read).unwrap_or(u64::MAX) > ceiling {
                    return Err(Self::fail(
                        "read",
                        &detail_path,
                        &format!("payload exceeds the {ceiling} byte file ceiling"),
                    ));
                }
                bytes.extend_from_slice(&probe[..read]);
            }
            Ok(bytes)
        })
        .await
        .map_err(|error| Self::fail("read", path, &error.to_string()))?
        .map(FileBytes::Local)
    }

    async fn decode_current_file(&self, path: &Path) -> Result<Batch> {
        let bytes = self.read_file(path).await?;
        let bounds = self.config.bounds()?;
        match &self.config.format {
            FileFormat::Csv { header } => {
                let codec = CsvCodec::new(crate::csv::IDENTITY_VERSION, *header)?;
                codec.decode(bytes.bytes(), &bounds, &self.config.schema)
            }
            FileFormat::Parquet => {
                let codec = crate::parquet::ParquetCodec::new(crate::parquet::IDENTITY_VERSION)?;
                codec.decode(bytes.bytes(), &bounds, &self.config.schema)
            }
            FileFormat::JsonLines => unreachable!("json advances through the line cache"),
        }
    }

    async fn next_csv_or_parquet(&mut self) -> Result<Option<SourceEvent>> {
        loop {
            let Some(path) = self.files.get(self.file_index).cloned() else {
                return Ok(None);
            };
            let batch = self.decode_current_file(&path).await?;
            let rows = u64::try_from(batch.num_rows()).unwrap_or(u64::MAX);
            if rows <= self.row_offset {
                // Replaying past a checkpointed file boundary: the cursor
                // already covers this file, so it stays consumed.
                self.file_index += 1;
                self.row_offset = 0;
                continue;
            }
            let cursor = self.cursor_for(self.file_index, rows)?;
            self.file_index += 1;
            self.row_offset = 0;
            self.sequence += 1;
            let batch = batch.with_metadata(relabel(&path, self.sequence)?);
            return Ok(Some(SourceEvent::Data { batch, cursor }));
        }
    }

    async fn ensure_json_lines(&mut self) -> Result<bool> {
        while self.line_cache.is_none() {
            let Some(path) = self.files.get(self.file_index).cloned() else {
                return Ok(false);
            };
            let bytes = self.read_file(&path).await?;
            let mut cache = JsonLineCache { bytes, offset: 0 };
            if cache.next_line(0).is_none() {
                self.file_index += 1;
                self.row_offset = 0;
                continue;
            }
            for _ in 0..self.row_offset {
                let Some((next, _)) = cache.next_line(cache.offset) else {
                    break;
                };
                cache.offset = next;
            }
            self.line_cache = Some(cache);
        }
        Ok(true)
    }

    fn next_json_chunk(&self, bounds: &DecodeBounds) -> Result<(Vec<u8>, u64, usize)> {
        let cache = self
            .line_cache
            .as_ref()
            .expect("the caller loaded the file");
        let mut offset = cache.offset;
        let mut chunk = Vec::new();
        let mut taken = 0;
        while let Some((next, line)) = cache.next_line(offset) {
            if taken >= bounds.max_rows
                || u64::try_from(chunk.len() + line.len() + 1).unwrap_or(u64::MAX)
                    > bounds.max_bytes
            {
                break;
            }
            chunk.extend_from_slice(line);
            chunk.push(b'\n');
            offset = next;
            taken += 1;
        }
        if chunk.is_empty() {
            return Err(Self::fail(
                "read",
                self.files.get(self.file_index).unwrap_or(&self.config.path),
                "a single line exceeds the batch byte limit",
            ));
        }
        Ok((chunk, taken, offset))
    }

    fn json_file_exhausted(&mut self) -> bool {
        let cache = self
            .line_cache
            .as_ref()
            .expect("the caller loaded the file");
        if cache.next_line(cache.offset).is_some() {
            return false;
        }
        self.line_cache = None;
        self.file_index += 1;
        self.row_offset = 0;
        true
    }

    fn emit_json_chunk(&mut self) -> Result<SourceEvent> {
        let bounds = self.config.bounds()?;
        let (chunk, taken, offset) = self.next_json_chunk(&bounds)?;
        let codec = JsonLinesCodec::new(crate::json_lines::IDENTITY_VERSION)?;
        let batch = codec.decode(&chunk, &bounds, &self.config.schema)?;
        let consumed = self.row_offset + taken;
        let cursor = self.cursor_for(self.file_index, consumed)?;
        self.row_offset = consumed;
        self.sequence += 1;
        self.line_cache
            .as_mut()
            .expect("the caller loaded the file")
            .offset = offset;
        let path = self.files.get(self.file_index).cloned().unwrap_or_default();
        let batch = batch.with_metadata(relabel(&path, self.sequence)?);
        self.json_file_exhausted();
        Ok(SourceEvent::Data { batch, cursor })
    }

    async fn next_json_lines(&mut self) -> Result<Option<SourceEvent>> {
        loop {
            if !self.ensure_json_lines().await? {
                return Ok(None);
            }
            if self.json_file_exhausted() {
                continue;
            }
            return Ok(Some(self.emit_json_chunk()?));
        }
    }
}

fn relabel(path: &Path, sequence: u64) -> Result<calc_flow::BatchMetadata> {
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or("file");
    calc_flow::BatchMetadata::new(
        "file",
        sequence,
        BTreeMap::from([("file".to_string(), Value::String(name.to_string()))]),
    )
}

fn discover_files(path: &Path, expected: &str) -> Result<Vec<PathBuf>> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| FileSource::fail("open", path, &error.to_string()))?;
    if metadata.file_type().is_symlink() {
        return Err(FileSource::fail(
            "open",
            path,
            "symlinks are rejected; name a regular file or directory",
        ));
    }
    if metadata.is_dir() {
        discover_directory(path, expected)
    } else if metadata.is_file() {
        Ok(vec![path.to_path_buf()])
    } else {
        Err(FileSource::fail(
            "open",
            path,
            "path is neither a regular file nor a directory",
        ))
    }
}

fn discover_directory(directory: &Path, expected: &str) -> Result<Vec<PathBuf>> {
    let mut names: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(directory)
        .map_err(|error| FileSource::fail("open", directory, &error.to_string()))?
    {
        let entry =
            entry.map_err(|error| FileSource::fail("open", directory, &error.to_string()))?;
        let file_type = entry
            .file_type()
            .map_err(|error| FileSource::fail("open", directory, &error.to_string()))?;
        let path = entry.path();
        if file_type.is_symlink() {
            return Err(FileSource::fail(
                "open",
                &path,
                "symlinked entries are rejected",
            ));
        }
        if !file_type.is_file() {
            return Err(FileSource::fail(
                "open",
                &path,
                "directory entries must be regular files; subdirectories are rejected",
            ));
        }
        let extension = path
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or_default();
        if !extension.eq_ignore_ascii_case(expected) {
            return Err(FileSource::fail(
                "open",
                &path,
                &format!("unexpected extension; the format requires .{expected} files"),
            ));
        }
        names.push(path);
    }
    names.sort();
    Ok(names)
}

#[async_trait]
impl StreamSource for FileSource {
    fn capabilities(&self) -> SourceCapabilities {
        self.capabilities.clone()
    }

    async fn open(&mut self, cursor: Option<Cursor>) -> Result<()> {
        if self.history.is_none() {
            self.discover().await?;
        }
        self.file_index = 0;
        self.row_offset = 0;
        self.line_cache = None;
        self.sequence = 0;
        if let Some(cursor) = cursor {
            let invalid = |field| {
                Self::fail(
                    "open",
                    &self.config.path,
                    &format!("invalid cursor.{field}"),
                )
            };
            let payload = cursor.payload();
            let file = payload
                .get("file")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid("file"))?;
            let row = payload
                .get("row")
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("row"))?;
            let sequence = payload
                .get("sequence")
                .and_then(Value::as_u64)
                .ok_or_else(|| invalid("sequence"))?;
            let index = self
                .files
                .iter()
                .position(|path| path.file_name().and_then(|name| name.to_str()) == Some(file))
                .ok_or_else(|| {
                    Self::fail(
                        "open",
                        &self.config.path,
                        &format!("cursor names unknown file {file:?}"),
                    )
                })?;
            let mut order = [0; 16];
            order[..8].copy_from_slice(&(index as u64).to_be_bytes());
            order[8..].copy_from_slice(&row.to_be_bytes());
            if payload.len() != 3 || row == 0 || sequence == 0 || cursor.order() != order {
                return Err(Self::fail(
                    "open",
                    &self.config.path,
                    "invalid cursor position",
                ));
            }
            self.file_index = index;
            self.row_offset = row;
            self.sequence = sequence;
        }
        Ok(())
    }

    async fn next(&mut self) -> Result<Option<SourceEvent>> {
        match self.config.format {
            FileFormat::JsonLines => self.next_json_lines().await,
            FileFormat::Csv { .. } | FileFormat::Parquet => self.next_csv_or_parquet().await,
        }
    }

    async fn close(&mut self) -> Result<()> {
        self.line_cache = None;
        Ok(())
    }
}
