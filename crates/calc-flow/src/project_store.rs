use std::{
    fs::File,
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

use async_trait::async_trait;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::sync::Mutex;

use crate::json::{parse_json_value, validate_json_depth, validate_json_depth_at};
use crate::{
    CalcFlowError, OperatorSpec, PROJECT_FORMAT_VERSION, ProjectSpec, Result, ValidationIssue,
    canonical_json,
};

/// Maximum accepted size of one stored or imported project document.
pub const MAX_PROJECT_DOCUMENT_BYTES: usize = 10 * 1024 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum WriteMode {
    Create,
    Replace,
}

/// Asynchronous persistence contract for strict v3 project documents.
#[async_trait]
pub trait ProjectStore: Send + Sync {
    /// Creates a project, returning [`CalcFlowError::Conflict`] when its ID exists.
    async fn create(&self, project: &ProjectSpec) -> Result<()>;
    /// Atomically creates or replaces a project.
    async fn put(&self, project: &ProjectSpec) -> Result<()>;
    /// Returns a project or [`CalcFlowError::NotFound`].
    async fn get(&self, project_id: &str) -> Result<ProjectSpec>;
    /// Lists all canonical project files, sorted by project ID.
    async fn list(&self) -> Result<Vec<ProjectSpec>>;
    /// Deletes a project or returns [`CalcFlowError::NotFound`].
    async fn delete(&self, project_id: &str) -> Result<()>;
}

/// Atomic file-backed v3 project storage rooted in one canonical directory.
#[derive(Clone, Debug)]
pub struct FileProjectStore {
    directory: PathBuf,
    // Serializes create/put/delete through one store instance: Windows
    // rejects a rename over a destination that another in-flight rename is
    // still resolving, so concurrent writers must take turns.
    write_lock: Arc<Mutex<()>>,
}

impl FileProjectStore {
    /// Creates the storage directory and resolves it to a stable canonical root.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Io`] when the directory cannot be created or resolved.
    pub async fn new(directory: impl AsRef<Path>) -> Result<Self> {
        let requested = directory.as_ref().to_owned();
        tokio::fs::create_dir_all(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        let directory = tokio::fs::canonicalize(&requested)
            .await
            .map_err(|source| io_error(&requested, source))?;
        Ok(Self {
            directory,
            write_lock: Arc::new(Mutex::new(())),
        })
    }

    /// Imports a bounded strict v3 JSON document.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Format`] for malformed or oversized input.
    pub fn import_json(&self, document: impl AsRef<[u8]>) -> Result<ProjectSpec> {
        import_project_json(document.as_ref())
    }

    /// Imports a bounded, single-document, data-only v3 YAML document.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Format`] for malformed, tagged, aliased, or oversized input.
    pub fn import_yaml(&self, document: impl AsRef<[u8]>) -> Result<ProjectSpec> {
        import_project_yaml(document.as_ref())
    }

    /// Exports canonical pretty JSON.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Format`] when the project is not a valid v3 document.
    pub fn export_json(&self, project: &ProjectSpec) -> Result<String> {
        export_project_json(project)
    }

    /// Exports data-only YAML after a strict `ProjectSpec` round trip.
    ///
    /// # Errors
    ///
    /// Returns [`CalcFlowError::Format`] when serialization fails.
    pub fn export_yaml(&self, project: &ProjectSpec) -> Result<String> {
        export_project_yaml(project)
    }

    fn path_for(&self, project_id: &str) -> PathBuf {
        hashed_path(&self.directory, project_id)
    }
}

#[async_trait]
impl ProjectStore for FileProjectStore {
    async fn create(&self, project: &ProjectSpec) -> Result<()> {
        validate_project_identity(project)?;
        let bytes = export_project_json(project)?.into_bytes();
        let _write_guard = self.write_lock.lock().await;
        atomic_write(
            self.directory.clone(),
            self.path_for(&project.id),
            bytes,
            WriteMode::Create,
            "project",
            &project.id,
        )
        .await
    }

    async fn put(&self, project: &ProjectSpec) -> Result<()> {
        validate_project_identity(project)?;
        let bytes = export_project_json(project)?.into_bytes();
        let _write_guard = self.write_lock.lock().await;
        atomic_write(
            self.directory.clone(),
            self.path_for(&project.id),
            bytes,
            WriteMode::Replace,
            "project",
            &project.id,
        )
        .await
    }

    async fn get(&self, project_id: &str) -> Result<ProjectSpec> {
        let path = self.path_for(project_id);
        let Some(bytes) = bounded_read(path.clone(), MAX_PROJECT_DOCUMENT_BYTES).await? else {
            return Err(CalcFlowError::NotFound {
                resource: "project".into(),
                key: project_id.into(),
            });
        };
        let project = import_project_json(&bytes)?;
        if project.id != project_id {
            return Err(format_error(format!(
                "stored project ID {:?} does not match key {project_id:?}",
                project.id
            )));
        }
        Ok(project)
    }

    async fn list(&self) -> Result<Vec<ProjectSpec>> {
        let mut reader = tokio::fs::read_dir(&self.directory)
            .await
            .map_err(|source| io_error(&self.directory, source))?;
        let mut paths = Vec::new();
        while let Some(entry) = reader
            .next_entry()
            .await
            .map_err(|source| io_error(&self.directory, source))?
        {
            if is_canonical_json_name(&entry.file_name().to_string_lossy()) {
                paths.push(entry.path());
            }
        }
        paths.sort();

        let mut projects = Vec::with_capacity(paths.len());
        for path in paths {
            let bytes = bounded_read(path.clone(), MAX_PROJECT_DOCUMENT_BYTES)
                .await?
                .ok_or_else(|| {
                    format_error(format!("project file {} disappeared", path.display()))
                })?;
            let project = import_project_json(&bytes)?;
            let expected = file_name_for(&project.id);
            if path
                .file_name()
                .is_none_or(|name| name != expected.as_str())
            {
                return Err(format_error(format!(
                    "stored project ID {:?} does not match file {}",
                    project.id,
                    path.display()
                )));
            }
            projects.push(project);
        }
        projects.sort_by(|left, right| left.id.cmp(&right.id));
        Ok(projects)
    }

    async fn delete(&self, project_id: &str) -> Result<()> {
        let _write_guard = self.write_lock.lock().await;
        delete_file(
            self.directory.clone(),
            self.path_for(project_id),
            false,
            "project",
            project_id,
        )
        .await
    }
}

/// Serializes a strict v3 project to recursively sorted pretty JSON plus one newline.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] if the project cannot round-trip through `ProjectSpec`.
pub fn export_project_json(project: &ProjectSpec) -> Result<String> {
    validate_project_identity(project)?;
    validate_project_json_values(project)?;
    let value = serde_json::to_value(project).map_err(|error| format_error(error.to_string()))?;
    let normalized: ProjectSpec =
        serde_json::from_value(value.clone()).map_err(|error| format_error(error.to_string()))?;
    if normalized != *project {
        return Err(format_error(
            "project contains values that cannot be represented exactly in JSON".into(),
        ));
    }
    let compact = canonical_json(&value)?;
    let sorted: Value =
        serde_json::from_str(&compact).map_err(|error| format_error(error.to_string()))?;
    let mut document =
        serde_json::to_string_pretty(&sorted).map_err(|error| format_error(error.to_string()))?;
    document.push('\n');
    check_document_size(
        document.as_bytes(),
        MAX_PROJECT_DOCUMENT_BYTES,
        "project JSON",
    )?;
    Ok(document)
}

/// Imports a strict v3 project from JSON using the default byte limit.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] for malformed, non-object, or oversized documents.
pub fn import_project_json(document: &[u8]) -> Result<ProjectSpec> {
    import_project_json_with_limit(document, MAX_PROJECT_DOCUMENT_BYTES)
}

/// Imports a strict v3 project from JSON using an explicit inclusive byte limit.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] for malformed, non-object, or oversized documents.
pub fn import_project_json_with_limit(document: &[u8], max_bytes: usize) -> Result<ProjectSpec> {
    check_document_size(document, max_bytes, "project JSON")?;
    let value = parse_json_value(document, "project JSON")?;
    if !value.is_object() {
        return Err(format_error(
            "project document must contain an object".into(),
        ));
    }
    reject_project_version(&value)?;
    let join_issues = collect_raw_stream_join_issues(&value)?;
    if !join_issues.is_empty() {
        return Err(CalcFlowError::ProjectValidation {
            issues: join_issues,
        });
    }
    let project = serde_json::from_value(value).map_err(|error| format_error(error.to_string()))?;
    validate_project_identity(&project)?;
    validate_project_json_values(&project)?;
    Ok(project)
}

/// Serializes a project as data-only YAML after a strict JSON round trip.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] when validation or serialization fails.
pub fn export_project_yaml(project: &ProjectSpec) -> Result<String> {
    let normalized = import_project_json(export_project_json(project)?.as_bytes())?;
    let mut document =
        serde_saphyr::to_string(&normalized).map_err(|error| format_error(error.to_string()))?;
    if !document.ends_with('\n') {
        document.push('\n');
    }
    check_document_size(
        document.as_bytes(),
        MAX_PROJECT_DOCUMENT_BYTES,
        "project YAML",
    )?;
    Ok(document)
}

/// Imports a bounded, single-document, data-only YAML project.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] for malformed, tagged, aliased, or oversized input.
pub fn import_project_yaml(document: &[u8]) -> Result<ProjectSpec> {
    import_project_yaml_with_limit(document, MAX_PROJECT_DOCUMENT_BYTES)
}

/// Imports a data-only YAML project using an explicit inclusive byte limit.
///
/// # Errors
///
/// Returns [`CalcFlowError::Format`] for malformed, tagged, aliased, deeply nested,
/// multi-document, or oversized input.
pub fn import_project_yaml_with_limit(document: &[u8], max_bytes: usize) -> Result<ProjectSpec> {
    check_document_size(document, max_bytes, "project YAML")?;
    let text = std::str::from_utf8(document)
        .map_err(|error| format_error(format!("project YAML is not UTF-8: {error}")))?;
    let project: ProjectSpec =
        match serde_saphyr::from_slice_with_options(document, yaml_options(max_bytes)) {
            Ok(project) => project,
            Err(error) => {
                if let Ok(value) = serde_saphyr::from_slice_with_options::<Value>(
                    document,
                    yaml_options(max_bytes),
                ) {
                    reject_project_version(&value)?;
                    if let Ok(join_issues) = collect_raw_stream_join_issues(&value)
                        && !join_issues.is_empty()
                    {
                        return Err(CalcFlowError::ProjectValidation {
                            issues: join_issues,
                        });
                    }
                }
                return Err(format_error(error.to_string()));
            }
        };
    // Only scan presentation-level tags and document boundaries after the budgeted
    // typed parse has established finite depth, event, node, and alias bounds.
    reject_yaml_tags(text)?;
    validate_project_identity(&project)?;
    validate_project_json_values(&project)?;
    Ok(project)
}

fn validate_project_json_values(project: &ProjectSpec) -> Result<()> {
    for source in &project.data_sources {
        validate_json_depth(
            &source.data,
            &format!("project data source {:?} data", source.id),
        )?;
    }
    for node in &project.graph.nodes {
        if let OperatorSpec::External { options, .. } = &node.operator {
            for (key, value) in options {
                validate_json_depth_at(
                    value,
                    &format!("project node {:?} option {key:?}", node.id),
                    1,
                )?;
            }
        }
    }
    Ok(())
}

fn yaml_options(max_bytes: usize) -> serde_saphyr::Options {
    serde_saphyr::options! {
        budget: serde_saphyr::budget! {
            max_reader_input_bytes: Some(max_bytes),
            max_events: 200_000,
            max_aliases: 0,
            max_anchors: 0,
            max_depth: 32,
            max_inclusion_depth: 0,
            max_documents: 1,
            max_nodes: 100_000,
            max_total_scalar_bytes: max_bytes,
            max_total_comment_bytes: max_bytes,
            max_merge_keys: 0,
        },
        duplicate_keys: serde_saphyr::DuplicateKeyPolicy::Error,
        merge_keys: serde_saphyr::MergeKeyPolicy::Error,
        alias_limits: serde_saphyr::alias_limits! {
            max_total_replayed_events: 0,
            max_replay_stack_depth: 0,
            max_alias_expansions_per_anchor: 0,
        },
        strict_booleans: true,
    }
}

fn reject_yaml_tags(document: &str) -> Result<()> {
    use serde_saphyr::granit_parser::{Event, Parser};

    let mut documents = 0usize;
    for event in Parser::new_from_str(document) {
        let (event, _) = event.map_err(|error| format_error(error.to_string()))?;
        if matches!(event, Event::DocumentStart(..)) {
            documents += 1;
            if documents > 1 {
                return Err(format_error(
                    "project YAML must contain exactly one document".into(),
                ));
            }
        }
        if event.tag().is_some() || matches!(event, Event::Alias(_)) {
            return Err(format_error(
                "project YAML tags, includes, anchors, and aliases are not allowed".into(),
            ));
        }
    }
    Ok(())
}

fn validate_project_identity(project: &ProjectSpec) -> Result<()> {
    if project.format_version != PROJECT_FORMAT_VERSION {
        return Err(CalcFlowError::UnsupportedVersion {
            expected: PROJECT_FORMAT_VERSION,
            found: project.format_version,
        });
    }
    if project.id.is_empty() {
        return Err(CalcFlowError::InvalidArgument {
            field: "project.id".into(),
            message: "must not be empty".into(),
        });
    }
    Ok(())
}

/// Inclusive JSON-safe ceiling shared by every stream Join bound and limit.
const STREAM_JOIN_SAFE_JSON_INTEGER: u64 = 9_007_199_254_740_991;

/// Pre-typed raw stream Join declaration (spec FR58).
///
/// `join_type`, bounds, and limits stay raw scalar values so malformed input
/// reports the frozen field-level issue paths and codes instead of a Serde
/// format error. This DTO never appears in the public crate surface.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    dead_code,
    reason = "typed fields keep unknown-field rejection strict without raw re-validation"
)]
struct RawStreamJoinSpec {
    join_type: Option<Value>,
    #[serde(default)]
    left_keys: Option<Vec<String>>,
    #[serde(default)]
    right_keys: Option<Vec<String>>,
    left_event_time: Option<String>,
    right_event_time: Option<String>,
    bounds: Option<RawJoinTimeBounds>,
    limits: Option<RawJoinStateLimits>,
    left_prefix: Option<String>,
    right_prefix: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct RawJoinTimeBounds {
    before_micros: Option<Value>,
    after_micros: Option<Value>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_field_names,
    reason = "the frozen public JSON field names all use the max_ limit prefix"
)]
struct RawJoinStateLimits {
    max_state_rows_per_side: Option<Value>,
    max_state_bytes_per_side: Option<Value>,
    max_matches_per_input_batch: Option<Value>,
}

/// Collects pre-typed stream Join issues from one parsed project document.
///
/// Unknown fields, duplicate keys, and object/array values where a scalar is
/// required remain strict project-format errors (spec FR58).
fn collect_raw_stream_join_issues(document: &Value) -> Result<Vec<ValidationIssue>> {
    let Some(nodes) = document
        .get("graph")
        .and_then(|graph| graph.get("nodes"))
        .and_then(Value::as_array)
    else {
        return Ok(Vec::new());
    };
    let mut issues = Vec::new();
    for (index, node) in nodes.iter().enumerate() {
        let Some(operator) = node.get("operator") else {
            continue;
        };
        if operator.get("kind").and_then(Value::as_str) != Some("stream_join") {
            continue;
        }
        let Some(spec) = operator.get("spec") else {
            continue;
        };
        let base = format!("graph.nodes[{index}].operator.spec");
        let raw: RawStreamJoinSpec = serde_json::from_value(spec.clone())
            .map_err(|error| format_error(error.to_string()))?;
        validate_raw_join_scalars(&raw, &base, &mut issues)?;
        validate_raw_join_keys(&raw, &base, &mut issues);
        validate_raw_join_prefixes(&raw, node, &base, &mut issues);
    }
    Ok(issues)
}

/// Emits the frozen `join_type`/bounds/limits issues in table field order.
fn validate_raw_join_scalars(
    raw: &RawStreamJoinSpec,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Result<()> {
    match &raw.join_type {
        None => issues.push(raw_issue(
            format!("{base}.join_type"),
            "unsupported_join_type",
            "join_type must be the string inner",
        )),
        Some(value) => {
            require_scalar(value, "join_type")?;
            if value.as_str() != Some("inner") {
                issues.push(raw_issue(
                    format!("{base}.join_type"),
                    "unsupported_join_type",
                    "join_type must be the string inner",
                ));
            }
        }
    }
    let bounds = raw.bounds.as_ref();
    validate_raw_bound(
        bounds.and_then(|bounds| bounds.before_micros.as_ref()),
        "before_micros",
        base,
        issues,
    )?;
    validate_raw_bound(
        bounds.and_then(|bounds| bounds.after_micros.as_ref()),
        "after_micros",
        base,
        issues,
    )?;
    let limits = raw.limits.as_ref();
    validate_raw_limit(
        limits.and_then(|limits| limits.max_state_rows_per_side.as_ref()),
        "max_state_rows_per_side",
        base,
        issues,
    )?;
    validate_raw_limit(
        limits.and_then(|limits| limits.max_state_bytes_per_side.as_ref()),
        "max_state_bytes_per_side",
        base,
        issues,
    )?;
    validate_raw_limit(
        limits.and_then(|limits| limits.max_matches_per_input_batch.as_ref()),
        "max_matches_per_input_batch",
        base,
        issues,
    )
}

/// Emits one `invalid_time_bound` issue unless the raw value is a safe integer.
fn validate_raw_bound(
    raw: Option<&Value>,
    field: &str,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Result<()> {
    if let Some(value) = raw {
        require_scalar(value, field)?;
    }
    let valid = raw.is_some_and(|value| {
        value
            .as_u64()
            .is_some_and(|number| number <= STREAM_JOIN_SAFE_JSON_INTEGER)
    });
    if !valid {
        issues.push(raw_issue(
            format!("{base}.bounds.{field}"),
            "invalid_time_bound",
            format!("{field} must be an integer microsecond count in 0..=9007199254740991"),
        ));
    }
    Ok(())
}

/// Emits one `invalid_join_limit` issue unless the raw value is a positive
/// safe integer.
fn validate_raw_limit(
    raw: Option<&Value>,
    field: &str,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) -> Result<()> {
    if let Some(value) = raw {
        require_scalar(value, field)?;
    }
    let valid = raw.is_some_and(|value| {
        value
            .as_u64()
            .is_some_and(|number| (1..=STREAM_JOIN_SAFE_JSON_INTEGER).contains(&number))
    });
    if !valid {
        issues.push(raw_issue(
            format!("{base}.limits.{field}"),
            "invalid_join_limit",
            format!("{field} must be an integer in 1..=9007199254740991"),
        ));
    }
    Ok(())
}

/// Emits the frozen empty/duplicate/mismatched key-list issues.
fn validate_raw_join_keys(raw: &RawStreamJoinSpec, base: &str, issues: &mut Vec<ValidationIssue>) {
    let unique_non_empty = |keys: &Vec<String>| {
        !keys.is_empty()
            && !keys.iter().any(String::is_empty)
            && keys.iter().collect::<std::collections::BTreeSet<_>>().len() == keys.len()
    };
    let left_valid = raw.left_keys.as_ref().is_some_and(unique_non_empty);
    if raw.left_keys.is_some() && !left_valid {
        issues.push(raw_issue(
            format!("{base}.left_keys"),
            "invalid_join_keys",
            "left_keys must contain at least one unique column name",
        ));
    }
    let right_mismatches = raw.right_keys.as_ref().is_some_and(|right| {
        !unique_non_empty(right) || Some(right.len()) != raw.left_keys.as_ref().map(Vec::len)
    });
    if left_valid && right_mismatches {
        let count = raw.left_keys.as_ref().map_or(0, Vec::len);
        issues.push(raw_issue(
            format!("{base}.right_keys"),
            "invalid_join_keys",
            format!(
                "right_keys must contain exactly {count} unique column names to match left_keys"
            ),
        ));
    }
}

/// Emits the frozen prefix issues and the differ/collision row.
fn validate_raw_join_prefixes(
    raw: &RawStreamJoinSpec,
    node: &Value,
    base: &str,
    issues: &mut Vec<ValidationIssue>,
) {
    for (side, prefix) in [("left", &raw.left_prefix), ("right", &raw.right_prefix)] {
        if let Some(prefix) = prefix
            && !crate::operator::is_portable_identifier(prefix)
        {
            issues.push(raw_issue(
                format!("{base}.{side}_prefix"),
                "invalid_output_prefix",
                format!("{side}_prefix must be a non-empty portable identifier"),
            ));
        }
    }
    let left = raw
        .left_prefix
        .as_deref()
        .unwrap_or(crate::operator::StreamJoinSpec::DEFAULT_LEFT_PREFIX);
    let right = raw
        .right_prefix
        .as_deref()
        .unwrap_or(crate::operator::StreamJoinSpec::DEFAULT_RIGHT_PREFIX);
    if left == right || raw_output_fields_collide(node, left, right) {
        issues.push(raw_issue(
            format!("{base}.right_prefix"),
            "invalid_output_prefix",
            "right_prefix must differ from left_prefix and produce no output-field collision",
        ));
    }
}

/// Detects `<left_prefix>__<name>` output collisions across both input schemas.
fn raw_output_fields_collide(node: &Value, left: &str, right: &str) -> bool {
    let fields = |port: usize| -> Option<Vec<String>> {
        node.get("input_ports")?
            .as_array()?
            .get(port)?
            .get("schema")?
            .as_array()?
            .iter()
            .map(|field| field.get("name")?.as_str().map(str::to_owned))
            .collect()
    };
    let (Some(left_fields), Some(right_fields)) = (fields(0), fields(1)) else {
        return false;
    };
    left_fields.iter().any(|name| {
        let prefixed = format!("{left}__{name}");
        right_fields
            .iter()
            .map(|other| format!("{right}__{other}"))
            .any(|candidate| candidate == prefixed)
    })
}

/// Rejects object/array values in scalar positions as format errors.
fn require_scalar(value: &Value, field: &str) -> Result<()> {
    if value.is_object() || value.is_array() {
        return Err(format_error(format!(
            "stream_join {field} must be a scalar JSON value"
        )));
    }
    Ok(())
}

fn raw_issue(path: String, code: &str, message: impl Into<String>) -> ValidationIssue {
    ValidationIssue {
        path,
        code: code.into(),
        message: message.into(),
    }
}

fn reject_project_version(value: &Value) -> Result<()> {
    let Some(version) = value.get("format_version") else {
        return Ok(());
    };
    let Some(version) = version.as_u64() else {
        return Ok(());
    };
    let version = u32::try_from(version)
        .map_err(|_| format_error("project format version is outside the u32 range".into()))?;
    if version != PROJECT_FORMAT_VERSION {
        return Err(CalcFlowError::UnsupportedVersion {
            expected: PROJECT_FORMAT_VERSION,
            found: version,
        });
    }
    Ok(())
}

fn check_document_size(document: &[u8], max_bytes: usize, label: &str) -> Result<()> {
    if document.len() > max_bytes {
        Err(format_error(format!(
            "{label} exceeds the {max_bytes}-byte limit"
        )))
    } else {
        Ok(())
    }
}

fn hashed_path(directory: &Path, key: &str) -> PathBuf {
    directory.join(file_name_for(key))
}

fn file_name_for(key: &str) -> String {
    format!("{}.json", hex::encode(Sha256::digest(key.as_bytes())))
}

fn is_canonical_json_name(name: &str) -> bool {
    name.len() == 69
        && name.as_bytes().get(64..) == Some(b".json")
        && name[..64]
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

pub(crate) async fn bounded_read(path: PathBuf, max_bytes: usize) -> Result<Option<Vec<u8>>> {
    spawn_blocking(move || {
        let metadata = match std::fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(io_error(&path, source)),
        };
        if metadata.file_type().is_symlink() {
            return Err(format_error(format!(
                "refusing to follow symbolic link {}",
                path.display()
            )));
        }
        if !metadata.is_file() {
            return Err(format_error(format!(
                "stored entry {} is not a regular file",
                path.display()
            )));
        }
        if metadata.len() > max_bytes as u64 {
            return Err(format_error(format!(
                "stored document {} exceeds the {max_bytes}-byte limit",
                path.display()
            )));
        }
        let file = File::open(&path).map_err(|source| io_error(&path, source))?;
        let capacity = usize::try_from(metadata.len())
            .map_err(|_| format_error("stored document size does not fit usize".into()))?;
        let mut bytes = Vec::with_capacity(capacity);
        file.take(max_bytes as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(|source| io_error(&path, source))?;
        if bytes.len() > max_bytes {
            return Err(format_error(format!(
                "stored document {} exceeds the {max_bytes}-byte limit",
                path.display()
            )));
        }
        Ok(Some(bytes))
    })
    .await
}

pub(crate) async fn atomic_write(
    directory: PathBuf,
    destination: PathBuf,
    bytes: Vec<u8>,
    mode: WriteMode,
    resource: &'static str,
    key: &str,
) -> Result<()> {
    let key = key.to_owned();
    spawn_blocking(move || {
        let mut temporary = tempfile::NamedTempFile::new_in(&directory)
            .map_err(|source| io_error(&directory, source))?;
        temporary
            .write_all(&bytes)
            .map_err(|source| io_error(temporary.path(), source))?;
        temporary
            .flush()
            .map_err(|source| io_error(temporary.path(), source))?;
        temporary
            .as_file()
            .sync_all()
            .map_err(|source| io_error(temporary.path(), source))?;

        match mode {
            WriteMode::Create => temporary.persist_noclobber(&destination).map_err(|error| {
                if error.error.kind() == std::io::ErrorKind::AlreadyExists {
                    CalcFlowError::Conflict {
                        resource: resource.into(),
                        key: key.clone(),
                    }
                } else {
                    io_error(&destination, error.error)
                }
            })?,
            WriteMode::Replace => temporary
                .persist(&destination)
                .map_err(|error| io_error(&destination, error.error))?,
        };
        sync_directory(&directory)?;
        Ok(())
    })
    .await
}

pub(crate) async fn delete_file(
    directory: PathBuf,
    path: PathBuf,
    missing_is_ok: bool,
    resource: &'static str,
    key: &str,
) -> Result<()> {
    let key = key.to_owned();
    spawn_blocking(move || match std::fs::remove_file(&path) {
        Ok(()) => sync_directory(&directory),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound && missing_is_ok => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Err(CalcFlowError::NotFound {
                resource: resource.into(),
                key,
            })
        }
        Err(source) => Err(io_error(&path, source)),
    })
    .await
}

async fn spawn_blocking<T: Send + 'static>(
    operation: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    tokio::task::spawn_blocking(operation)
        .await
        .map_err(|error| CalcFlowError::Internal {
            message: format!("filesystem worker failed: {error}"),
        })?
}

#[cfg(unix)]
fn sync_directory(directory: &Path) -> Result<()> {
    File::open(directory)
        .and_then(|file| file.sync_all())
        .map_err(|source| io_error(directory, source))
}

#[cfg(not(unix))]
#[cfg_attr(
    not(unix),
    allow(
        clippy::unnecessary_wraps,
        reason = "non-unix stub keeps the unix fallible signature so call sites stay platform-agnostic"
    )
)]
fn sync_directory(_directory: &Path) -> Result<()> {
    Ok(())
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
