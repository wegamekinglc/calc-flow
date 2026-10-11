use super::engine;
use datafusion::arrow::{
    array::{Int64Array, StringArray, TimestampMicrosecondArray},
    datatypes::{DataType, Field, Schema, TimeUnit},
    record_batch::RecordBatch,
};
use engine::{Batch, BatchMetadata, CalcFlowError, Epoch, Result};
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::Write as _,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::Duration,
};

pub(super) const TARGET_DURATION: Duration = Duration::from_secs(1_200);

pub(super) const SOAK_WINDOW_MICROS: i64 = 1;

pub(super) const SOAK_CHECKPOINT_BATCH_INTERVAL: u64 = 8;

pub(super) const CHECKPOINT_SOAK_CADENCE: Duration = Duration::from_secs(10);

pub(super) const CHECKPOINT_SOAK_SAMPLE_COUNT: usize = 120;

pub(super) const CHECKPOINT_SOAK_RESTART_SAMPLES: [usize; 2] = [39, 79];

pub(super) const MAX_CHECKPOINT_SOAK_STATE_BYTES: u64 = 64 * 1_024 * 1_024;

pub(super) const CHECKPOINT_SOAK_CHILD_ENV: &str = "CALC_FLOW_M5_CHECKPOINT_SOAK_CHILD_PLAN";

pub(super) const CHECKPOINT_SOAK_MALLOC_ARENA_MAX: &str = "1";

pub(super) const CHECKPOINT_SOAK_MALLOC_TRIM_THRESHOLD: &str = "0";

pub(super) const CHECKPOINT_SOAK_MALLOC_TOP_PAD: &str = "0";

pub(super) const CHECKPOINT_SOAK_CHILD_TEST: &str =
    "runtime::streaming::soak::checkpoint_restart_soak_generation_child_process";

pub(super) const CHECKPOINT_SOAK_PROCESS_SCHEMA: &str = "calc-flow.m5-checkpoint-soak-process.v2";

pub(super) const CHECKPOINT_SOAK_PLAN_SCHEMA: &str = "calc-flow.m5-checkpoint-soak-plan.v1";

pub(super) const CHECKPOINT_SOAK_GENERATIONS: usize = 3;

pub(super) const CHECKPOINT_SOAK_RETAINED_EPOCHS: usize = 2;

pub(super) const CHECKPOINT_SOAK_REPORT_LIMIT: u64 = 1 << 20;

pub(super) const CHECKPOINT_SOAK_CADENCE_TOLERANCE: Duration = Duration::from_secs(3);

pub(super) const MAX_CHECKPOINT_SOAK_RESTART_GAP: Duration = Duration::from_secs(60);

pub(super) const MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT: Duration = Duration::from_secs(60);

pub(super) const CHECKPOINT_SOAK_CANCEL_ALIGNMENT_TIMEOUT: Duration = Duration::from_secs(20);

pub(super) const CHECKPOINT_SOAK_PROCESS_OVERHEAD_BUDGET: Duration = Duration::from_secs(10);

pub(super) const CHECKPOINT_SOAK_SMOKE_GENERATION_TIMEOUT: Duration = Duration::from_secs(300);

pub(super) const CHECKPOINT_SOAK_SETTLE_TIMEOUT: Duration = Duration::from_secs(60);

pub(super) const CHECKPOINT_SOAK_CHECKPOINT_TIMEOUT_MILLIS: u64 = 10_000;

pub(super) const CHECKPOINT_SOAK_SMOKE_TARGET_CHECKPOINTS: u64 = 12;

pub(super) const CHECKPOINT_BENCHMARK_COMMAND: &str = "CARGO_TARGET_DIR=<fresh-candidate-target> CARGO_INCREMENTAL=0 CALC_FLOW_M5_CHECKPOINT_BENCHMARK=1 CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID=<unique-run-id> CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT=<candidate-commit> CALC_FLOW_M5_PRIVATE_SOURCE_TREE=<candidate-tree> cargo test --release --locked -p calc-flow --lib runtime::streaming::soak::private_m5_epoch_checkpoint_absolute_benchmark -- --ignored --exact --nocapture";

pub(super) const M5_PRIVATE_BENCHMARK_CASES: [&str; 12] = [
    "m5/private_path/barrier_cut_single_source",
    "m5/private_path/barrier_cut_two_source_fan_out",
    "m5/private_path/pass_through_two_input_alignment",
    "m5/private_path/window_two_input_alignment",
    "m5/private_path/dirty_window_state_stage",
    "m5/private_path/non_empty_manifest_publication",
    "m5/private_path/retained_delta_compacted_base_restore",
    "m5/private_path/single_transactional_sink_commit",
    "m5/private_path/multi_transactional_sink_commit",
    "m5/private_full_path/no_checkpoint",
    "m5/private_full_path/checkpoint_disabled",
    "m5/private_full_path/checkpoint_enabled",
];

pub(super) const CHECKPOINT_BENCHMARK_SAMPLE_SIZE: usize = 10;

pub(super) fn soak_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new(
            "event_time",
            DataType::Timestamp(TimeUnit::Microsecond, Some(Arc::from("UTC"))),
            false,
        ),
        Field::new("group", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

pub(super) fn soak_batch(source_id: &str, sequence: u64) -> Result<Batch> {
    let (event_times, groups, values) = if sequence % 4 == 0 {
        (Vec::new(), Vec::new(), Vec::new())
    } else {
        let value = i64::try_from(sequence)
            .expect("the twenty-minute soak cannot exhaust i64 source sequence space");
        (
            vec![value],
            vec![format!("{source_id}-{:02}", sequence % 64)],
            vec![value],
        )
    };
    let record = RecordBatch::try_new(
        soak_schema(),
        vec![
            Arc::new(TimestampMicrosecondArray::from(event_times).with_timezone("UTC")),
            Arc::new(StringArray::from(groups)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .expect("soak table schema is stable");
    Batch::table(
        vec![record],
        BatchMetadata::new(source_id, sequence, BTreeMap::new())?,
    )
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
pub(super) struct RssSample {
    pub(super) elapsed_seconds: f64,
    pub(super) rss_kib: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub(super) struct CheckpointRssProcessSamples {
    pub(super) generation: usize,
    pub(super) pid: u32,
    pub(super) samples: Vec<RssSample>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(super) struct PrivateBenchmarkProvenance {
    pub(super) commit: String,
    pub(super) tree: String,
    pub(super) clean: bool,
    pub(super) harness_hash: String,
    pub(super) config_hash: String,
    pub(super) executable: PathBuf,
    pub(super) executable_sha256: String,
    pub(super) build_identity_hash: String,
    pub(super) toolchain_hash: String,
    pub(super) environment_hash: String,
}

pub(super) fn is_full_git_oid(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

pub(super) fn benchmark_evidence_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "CALC_FLOW_M5_CHECKPOINT_BENCHMARK".into(),
        message: message.into(),
    }
}

pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

pub(super) fn validate_private_benchmark_source_identity(
    provenance: &PrivateBenchmarkProvenance,
    embedded_commit: Option<&str>,
    embedded_tree: Option<&str>,
) -> Result<()> {
    let embedded_commit = embedded_commit.ok_or_else(|| {
        benchmark_evidence_error("private benchmark binary has no embedded source commit")
    })?;
    let embedded_tree = embedded_tree.ok_or_else(|| {
        benchmark_evidence_error("private benchmark binary has no embedded source tree")
    })?;
    if !matches_private_benchmark_source(provenance, embedded_commit, embedded_tree) {
        return Err(benchmark_evidence_error(
            "private benchmark binary source identity is stale",
        ));
    }
    Ok(())
}

fn matches_private_benchmark_source(
    provenance: &PrivateBenchmarkProvenance,
    embedded_commit: &str,
    embedded_tree: &str,
) -> bool {
    is_full_git_oid(embedded_commit)
        && is_full_git_oid(embedded_tree)
        && embedded_commit == provenance.commit
        && embedded_tree == provenance.tree
}

pub(super) fn private_benchmark_case_directory(case: &str) -> String {
    case.replace('/', "_")
}

pub(super) fn private_benchmark_sample_reference(
    path: &Path,
    target_root: &Path,
) -> Result<String> {
    path.strip_prefix(target_root)
        .map(|relative| relative.to_string_lossy().into_owned())
        .map_err(|_| CalcFlowError::Internal {
            message: format!(
                "private benchmark sample {} escapes target root {}",
                path.display(),
                target_root.display()
            ),
        })
}

pub(super) fn finite_positive_measurement(value: &serde_json::Value, field: &str) -> Result<f64> {
    let value = value
        .as_f64()
        .filter(|value| value.is_finite() && *value > 0.0)
        .ok_or_else(|| CalcFlowError::Internal {
            message: format!("private benchmark {field} must be finite and positive"),
        })?;
    Ok(value)
}

pub(super) fn median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        values[middle - 1].midpoint(values[middle])
    } else {
        values[middle]
    }
}

pub(super) fn bootstrap_median_confidence_interval(samples: &[f64]) -> [f64; 2] {
    const RESAMPLES: usize = 4_096;
    let mut state = 0x4d35_6570_6f63_6842_u64;
    let sample_count = u64::try_from(samples.len()).unwrap();
    let mut medians = Vec::with_capacity(RESAMPLES);
    for _ in 0..RESAMPLES {
        let mut resample = Vec::with_capacity(samples.len());
        for _ in samples {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let index = usize::try_from(state % sample_count).unwrap();
            resample.push(samples[index]);
        }
        medians.push(median(&mut resample));
    }
    medians.sort_by(f64::total_cmp);
    [medians[RESAMPLES / 40], medians[RESAMPLES * 39 / 40]]
}

pub(super) fn private_benchmark_criterion_summary(
    estimates: &serde_json::Value,
) -> Result<(f64, [f64; 2], f64)> {
    let median =
        finite_positive_measurement(&estimates["median"]["point_estimate"], "Criterion median")?;
    let lower = finite_positive_measurement(
        &estimates["median"]["confidence_interval"]["lower_bound"],
        "Criterion lower confidence bound",
    )?;
    let upper = finite_positive_measurement(
        &estimates["median"]["confidence_interval"]["upper_bound"],
        "Criterion upper confidence bound",
    )?;
    let confidence_level = estimates["median"]["confidence_interval"]["confidence_level"]
        .as_f64()
        .filter(|value| value.to_bits() == 0.95_f64.to_bits())
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark Criterion confidence level must be exactly 0.95".into(),
        })?;
    if lower > median || median > upper {
        return Err(CalcFlowError::Internal {
            message: "private benchmark Criterion confidence interval is not ordered".into(),
        });
    }
    Ok((median, [lower, upper], confidence_level))
}

pub(super) fn private_benchmark_absolute_measurement(
    case: &str,
    run_label: &str,
    run_root: &Path,
    target_root: &Path,
) -> Result<serde_json::Value> {
    let case_root = run_root
        .join(private_benchmark_case_directory(case))
        .join(run_label);
    let estimates_path = case_root.join("estimates.json");
    let sample_path = case_root.join("sample.json");
    let (estimates_bytes, sample_bytes, estimates) =
        private_benchmark_documents(&estimates_path, &sample_path)?;
    let (criterion_median, [criterion_lower, criterion_upper], confidence_level) =
        private_benchmark_criterion_summary(&estimates)?;
    let sample = parse_private_benchmark_document(&sample_bytes, &sample_path, "sample")?;
    let statistics = private_benchmark_sample_statistics(&sample)?;
    Ok(json!({
        "case": case,
        "comparison": "none",
        "median_ns": statistics.median,
        "median_confidence_interval_ns": statistics.confidence,
        "confidence_level": confidence_level,
        "sample_count": statistics.count,
        "relative_sample_spread_percent": statistics.spread_percent,
        "criterion": {
            "median_ns": criterion_median,
            "median_confidence_interval_ns": [criterion_lower, criterion_upper],
        },
        "decision": "absolute_only",
        "artifacts": {
            "sample": {
                "path": private_benchmark_sample_reference(&sample_path, target_root)?,
                "sha256": sha256_bytes(&sample_bytes),
            },
            "estimates": {
                "path": private_benchmark_sample_reference(&estimates_path, target_root)?,
                "sha256": sha256_bytes(&estimates_bytes),
            },
        },
    }))
}

fn private_benchmark_documents(
    estimates_path: &Path,
    sample_path: &Path,
) -> Result<(Vec<u8>, Vec<u8>, serde_json::Value)> {
    let estimates_bytes = std::fs::read(estimates_path).map_err(|source| CalcFlowError::Io {
        path: estimates_path.display().to_string(),
        source,
    })?;
    let sample_bytes = std::fs::read(sample_path).map_err(|source| CalcFlowError::Io {
        path: sample_path.display().to_string(),
        source,
    })?;
    let estimates = parse_private_benchmark_document(&estimates_bytes, estimates_path, "estimate")?;
    Ok((estimates_bytes, sample_bytes, estimates))
}

fn parse_private_benchmark_document(
    bytes: &[u8],
    path: &Path,
    kind: &str,
) -> Result<serde_json::Value> {
    serde_json::from_slice(bytes).map_err(|error| CalcFlowError::Internal {
        message: format!(
            "private benchmark {kind} {} is invalid: {error}",
            path.display()
        ),
    })
}

fn normalized_private_benchmark_sample(
    index: usize,
    iterations: &serde_json::Value,
    elapsed: &serde_json::Value,
) -> Result<f64> {
    let iterations =
        finite_positive_measurement(iterations, &format!("sample {index} iteration count"))?;
    let elapsed = finite_positive_measurement(elapsed, &format!("sample {index} elapsed time"))?;
    let normalized = elapsed / iterations;
    if !normalized.is_finite() || normalized <= 0.0 {
        return Err(CalcFlowError::Internal {
            message: format!("private benchmark sample {index} is not finite"),
        });
    }
    Ok(normalized)
}

fn normalized_private_benchmark_samples(sample: &serde_json::Value) -> Result<Vec<f64>> {
    let iterations = sample["iters"]
        .as_array()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark sample iterations are missing".into(),
        })?;
    let times = sample["times"]
        .as_array()
        .ok_or_else(|| CalcFlowError::Internal {
            message: "private benchmark sample times are missing".into(),
        })?;
    if iterations.len() != CHECKPOINT_BENCHMARK_SAMPLE_SIZE || times.len() != iterations.len() {
        return Err(CalcFlowError::Internal {
            message: format!(
                "private benchmark requires exactly {CHECKPOINT_BENCHMARK_SAMPLE_SIZE} matched samples"
            ),
        });
    }
    iterations
        .iter()
        .zip(times)
        .enumerate()
        .map(|(index, (iterations, elapsed))| {
            normalized_private_benchmark_sample(index, iterations, elapsed)
        })
        .collect()
}

struct PrivateBenchmarkSampleStatistics {
    median: f64,
    confidence: [f64; 2],
    count: usize,
    spread_percent: f64,
}

fn private_benchmark_sample_statistics(
    sample: &serde_json::Value,
) -> Result<PrivateBenchmarkSampleStatistics> {
    let mut normalized = normalized_private_benchmark_samples(sample)?;
    let confidence = bootstrap_median_confidence_interval(&normalized);
    let minimum = normalized.iter().copied().min_by(f64::total_cmp).unwrap();
    let maximum = normalized.iter().copied().max_by(f64::total_cmp).unwrap();
    let raw_median = median(&mut normalized);
    if confidence[0] > raw_median || raw_median > confidence[1] {
        return Err(CalcFlowError::Internal {
            message: "private benchmark bootstrap confidence interval is not ordered".into(),
        });
    }
    Ok(PrivateBenchmarkSampleStatistics {
        median: raw_median,
        confidence,
        count: normalized.len(),
        spread_percent: (maximum / minimum - 1.0) * 100.0,
    })
}

pub(super) fn validate_private_benchmark_artifacts(
    measurement: &serde_json::Value,
    target_root: &Path,
) -> Result<()> {
    validate_private_measurement_contract(measurement)?;
    validate_private_measurement_confidence(measurement)?;
    let artifacts = measurement["artifacts"]
        .as_object()
        .filter(|artifacts| artifacts.len() == 2)
        .ok_or_else(|| benchmark_evidence_error("private benchmark artifacts are incomplete"))?;
    let canonical_root = target_root
        .canonicalize()
        .map_err(|source| CalcFlowError::Io {
            path: target_root.display().to_string(),
            source,
        })?;
    validate_private_artifact_entries(artifacts, &canonical_root)
}

fn validate_private_artifact_entries(
    artifacts: &serde_json::Map<String, serde_json::Value>,
    canonical_root: &Path,
) -> Result<()> {
    for name in ["sample", "estimates"] {
        let artifact = artifacts
            .get(name)
            .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact is missing")))?;
        validate_private_benchmark_artifact(name, artifact, canonical_root)?;
    }
    Ok(())
}

fn validate_private_measurement_contract(measurement: &serde_json::Value) -> Result<()> {
    if measurement["comparison"] != "none"
        || measurement["decision"] != "absolute_only"
        || measurement["sample_count"] != CHECKPOINT_BENCHMARK_SAMPLE_SIZE
        || measurement["confidence_level"].as_f64().map(f64::to_bits) != Some(0.95_f64.to_bits())
    {
        return Err(benchmark_evidence_error(
            "private absolute measurement contract is invalid",
        ));
    }
    Ok(())
}

fn validate_private_measurement_confidence(measurement: &serde_json::Value) -> Result<()> {
    let median = finite_positive_measurement(&measurement["median_ns"], "reported median")?;
    let confidence = measurement["median_confidence_interval_ns"]
        .as_array()
        .filter(|confidence| confidence.len() == 2)
        .ok_or_else(|| benchmark_evidence_error("reported confidence interval is invalid"))?;
    let lower = finite_positive_measurement(&confidence[0], "reported lower confidence bound")?;
    let upper = finite_positive_measurement(&confidence[1], "reported upper confidence bound")?;
    if lower > median || median > upper {
        return Err(benchmark_evidence_error(
            "reported confidence interval is not ordered",
        ));
    }
    Ok(())
}

fn private_benchmark_artifact_path(
    name: &str,
    artifact: &serde_json::Value,
    canonical_root: &Path,
) -> Result<PathBuf> {
    let reference = private_benchmark_artifact_reference(name, artifact)?;
    let path = canonical_root.join(reference);
    let metadata = std::fs::symlink_metadata(&path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(benchmark_evidence_error(format!(
            "{name} artifact is not a regular file"
        )));
    }
    let canonical = path.canonicalize().map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if !canonical.starts_with(canonical_root) {
        return Err(benchmark_evidence_error(format!(
            "{name} artifact escapes the benchmark target root"
        )));
    }
    Ok(canonical)
}

fn private_benchmark_artifact_reference<'a>(
    name: &str,
    artifact: &'a serde_json::Value,
) -> Result<&'a Path> {
    let reference = artifact["path"]
        .as_str()
        .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact path is missing")))?;
    let reference = Path::new(reference);
    if reference.is_absolute() {
        return Err(benchmark_evidence_error(format!(
            "{name} artifact path must be relative"
        )));
    }
    Ok(reference)
}

fn validate_private_benchmark_artifact(
    name: &str,
    artifact: &serde_json::Value,
    canonical_root: &Path,
) -> Result<()> {
    let canonical = private_benchmark_artifact_path(name, artifact, canonical_root)?;
    let expected = artifact["sha256"]
        .as_str()
        .filter(|digest| digest.len() == 64 && digest.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| benchmark_evidence_error(format!("{name} artifact hash is invalid")))?;
    let bytes = std::fs::read(&canonical).map_err(|source| CalcFlowError::Io {
        path: canonical.display().to_string(),
        source,
    })?;
    if sha256_bytes(&bytes) != expected {
        return Err(benchmark_evidence_error(format!(
            "{name} artifact hash does not match the report"
        )));
    }
    Ok(())
}

pub(super) fn private_checkpoint_benchmark_metadata(
    provenance: &PrivateBenchmarkProvenance,
    criterion_root: &Path,
    measurements: &[serde_json::Value],
) -> serde_json::Value {
    json!({
        "schema": "calc-flow.m5-checkpoint-absolute-benchmark.v1",
        "commit": provenance.commit,
        "mode": "candidate_absolute",
        "command": CHECKPOINT_BENCHMARK_COMMAND,
        "criterion_run_root": criterion_root,
        "provenance": provenance,
        "comparison": "none",
        "absolute_cases": M5_PRIVATE_BENCHMARK_CASES,
        "measurements": measurements,
        "overall_result": "absolute_only",
        "topology": {
            "sources": 2,
            "operators": ["union", "final_window", "two_forward_branches"],
            "sinks": 2,
            "checkpoint_modes": ["none", "terminal_only", "periodic"],
        },
    })
}

pub(super) fn write_private_checkpoint_benchmark_metadata(
    target_root: &Path,
    commit: &str,
    run_id: &str,
    metadata: &serde_json::Value,
) -> Result<PathBuf> {
    validate_private_benchmark_report_name(commit, run_id)?;
    let artifact_root = target_root.join("m5-checkpoint-benchmark");
    std::fs::create_dir_all(&artifact_root).map_err(|source| CalcFlowError::Io {
        path: artifact_root.display().to_string(),
        source,
    })?;
    let path = artifact_root.join(format!("{commit}-{run_id}.json"));
    let digest_path = artifact_root.join(format!("{commit}-{run_id}.json.sha256"));
    if path.exists() || digest_path.exists() {
        return Err(CalcFlowError::Conflict {
            resource: "private benchmark report".into(),
            key: commit.into(),
        });
    }
    let bytes = serde_json::to_vec_pretty(metadata).unwrap();
    let _file = create_synced_evidence_file(&path, &bytes)?;
    let _digest_file = create_synced_evidence_file(
        &digest_path,
        format!("{}\n", sha256_bytes(&bytes)).as_bytes(),
    )?;
    sync_directory(&artifact_root)?;
    Ok(path)
}

fn validate_private_benchmark_report_name(commit: &str, run_id: &str) -> Result<()> {
    if !is_full_git_oid(commit) {
        return Err(benchmark_evidence_error(
            "benchmark report name requires an exact commit SHA",
        ));
    }
    if run_id.is_empty() || run_id.len() > 64 || !run_id.bytes().all(private_benchmark_run_id_byte)
    {
        return Err(benchmark_evidence_error(
            "benchmark report run ID is invalid",
        ));
    }
    Ok(())
}

fn private_benchmark_run_id_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || b"-_".contains(&byte)
}

fn create_synced_evidence_file(path: &Path, bytes: &[u8]) -> Result<File> {
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })?;
    file.write_all(bytes).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    file.sync_all().map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    Ok(file)
}

// Directory fsync durability is a Unix idiom; plain File::open on a Windows
// directory fails with ERROR_ACCESS_DENIED, so non-Unix platforms skip the
// directory flush after the file-level sync_all calls above.
#[cfg(unix)]
pub(super) fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })
}

#[cfg(not(unix))]
#[cfg_attr(
    not(unix),
    allow(
        clippy::unnecessary_wraps,
        reason = "non-unix stub keeps the unix fallible signature so call sites stay platform-agnostic"
    )
)]
pub(super) fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

#[derive(Clone, Copy)]
pub(super) enum PrivateFullRunnerMode {
    NoCheckpoint,
    CheckpointDisabled,
    CheckpointEnabled,
}

pub(super) fn private_full_runner_sink_delay(mode: PrivateFullRunnerMode) -> Duration {
    if matches!(mode, PrivateFullRunnerMode::NoCheckpoint) {
        Duration::ZERO
    } else {
        Duration::from_millis(5)
    }
}

#[derive(Debug, Default, Eq, PartialEq)]
pub(super) struct CheckpointTemporaryArtifacts {
    pub(super) state: usize,
    pub(super) manifests: usize,
    pub(super) sink_a: usize,
    pub(super) sink_b: usize,
}

pub(super) fn count_checkpoint_artifacts(
    root: &Path,
    is_durable_file: impl Fn(&Path) -> bool,
) -> usize {
    let mut count = 0;
    let mut directories = vec![root.to_path_buf()];
    while let Some(directory) = directories.pop() {
        let entries = match std::fs::read_dir(&directory) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => continue,
            Err(error) => panic!(
                "checkpoint oracle could not inspect {}: {error}",
                directory.display()
            ),
        };
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic!(
                    "checkpoint oracle could not read an entry under {}: {error}",
                    directory.display()
                )
            });
            let path = entry.path();
            let metadata = std::fs::symlink_metadata(&path).unwrap_or_else(|error| {
                panic!(
                    "checkpoint oracle could not inspect {}: {error}",
                    path.display()
                )
            });
            if metadata.is_dir() {
                directories.push(path);
            } else if !is_durable_file(&path) {
                count += 1;
            }
        }
    }
    count
}

pub(super) fn is_canonical_checkpoint_manifest(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(epoch) = name
        .strip_prefix("manifest-")
        .and_then(|name| name.strip_suffix(".json"))
    else {
        return false;
    };
    epoch.len() == 20 && epoch.bytes().all(|byte| byte.is_ascii_digit())
}

pub(super) fn is_canonical_checkpoint_sink_output(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    let Some(epoch) = name
        .strip_prefix("visible-")
        .and_then(|name| name.strip_suffix(".json"))
    else {
        return false;
    };
    epoch.len() == 20 && epoch.bytes().all(|byte| byte.is_ascii_digit())
}

pub(super) fn inspect_checkpoint_temporary_artifacts(
    state_root: &Path,
    manifest_root: &Path,
    sink_root: &Path,
) -> CheckpointTemporaryArtifacts {
    CheckpointTemporaryArtifacts {
        state: count_checkpoint_artifacts(&state_root.join("staging"), |_| false),
        manifests: count_checkpoint_artifacts(manifest_root, is_canonical_checkpoint_manifest),
        sink_a: count_checkpoint_artifacts(
            &sink_root.join("sink-a"),
            is_canonical_checkpoint_sink_output,
        ),
        sink_b: count_checkpoint_artifacts(
            &sink_root.join("sink-b"),
            is_canonical_checkpoint_sink_output,
        ),
    }
}

pub(super) struct CheckpointRestartSoakReport {
    pub(super) restarts: usize,
    pub(super) generation_process_ids: Vec<u32>,
    pub(super) generation_exit_codes: Vec<i32>,
    pub(super) completed_epochs: u64,
    pub(super) compacted_window_epochs: usize,
    pub(super) maximum_manifest_count: usize,
    pub(super) maximum_state_bytes: u64,
    // Populated on every platform; only the Linux /proc soak evidence bundle reads them.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) source_records: usize,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) output_records: usize,
    pub(super) duplicate_records: usize,
    pub(super) missing_records: usize,
    pub(super) temporary_artifacts: usize,
    pub(super) terminal_tasks: usize,
    pub(super) terminal_charged_edges: usize,
    pub(super) terminal_registries: (usize, usize),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum CheckpointSoakProcessMode {
    Smoke,
    Standard,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointSoakProcessPlan {
    pub(super) schema: String,
    pub(super) commit: String,
    pub(super) executable_sha256: String,
    pub(super) run_root: PathBuf,
    pub(super) parent_pid: u32,
    pub(super) parent_launch_offset_micros: u64,
    pub(super) generation: usize,
    pub(super) sample_start: usize,
    pub(super) sample_end: usize,
    pub(super) checkpoint_interval_millis: u64,
    pub(super) checkpoint_timeout_millis: u64,
    pub(super) target_checkpoints: u64,
    pub(super) sink_write_delay_millis: u64,
    pub(super) retained_epochs: usize,
    pub(super) final_generation: bool,
    pub(super) mode: CheckpointSoakProcessMode,
    pub(super) config_hash: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointSoakProcessSample {
    pub(super) index: usize,
    pub(super) elapsed_micros: u64,
    pub(super) rss_kib: u64,
    pub(super) task_count: usize,
    pub(super) maximum_queue_depth: usize,
    pub(super) maximum_charged_rows: usize,
    pub(super) maximum_charged_bytes: usize,
    pub(super) completed_checkpoints: u64,
    pub(super) failed_checkpoints: u64,
    pub(super) manifest_count: usize,
    pub(super) state_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CheckpointSoakProcessReport {
    pub(super) schema: String,
    pub(super) commit: String,
    pub(super) executable_sha256: String,
    pub(super) config_hash: String,
    pub(super) run_root: PathBuf,
    pub(super) parent_pid: u32,
    pub(super) pid: u32,
    pub(super) generation: usize,
    pub(super) sample_start: usize,
    pub(super) sample_end: usize,
    pub(super) generation_started_micros: u64,
    pub(super) sampling_started_micros: u64,
    pub(super) generation_finished_micros: u64,
    pub(super) restored_epoch: Option<u64>,
    pub(super) restored_cursor_orders: BTreeMap<String, Option<String>>,
    pub(super) terminal_epoch: u64,
    pub(super) terminal_cursor_orders: BTreeMap<String, String>,
    pub(super) restored_window_state: bool,
    pub(super) restored_watermarks: bool,
    pub(super) restored_progress: bool,
    pub(super) samples: Vec<CheckpointSoakProcessSample>,
    pub(super) compacted_epochs: Vec<u64>,
    pub(super) maximum_manifest_count: usize,
    pub(super) maximum_state_bytes: u64,
    pub(super) completed_checkpoints: u64,
    pub(super) failed_checkpoints: u64,
    pub(super) source_open_events: usize,
    pub(super) source_close_events: usize,
    pub(super) sink_close_events: usize,
    pub(super) terminal_cause: String,
    pub(super) terminal_tasks: usize,
    pub(super) terminal_charged_edges: usize,
    pub(super) terminal_registries: (usize, usize),
    pub(super) source_records: usize,
    pub(super) output_records: usize,
    pub(super) duplicate_records: usize,
    pub(super) missing_records: usize,
    pub(super) temporary_artifacts: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(super) struct CheckpointSoakParentTiming {
    pub(super) generation: usize,
    pub(super) launch_micros: u64,
    pub(super) finish_micros: u64,
}

pub(super) struct CheckpointSoakProcessEvidence {
    pub(super) report: CheckpointRestartSoakReport,
    // Populated on every platform; only the Linux /proc soak evidence bundle reads them.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) plans: Vec<CheckpointSoakProcessPlan>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) processes: Vec<CheckpointSoakProcessReport>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) parent_timings: Vec<CheckpointSoakParentTiming>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) rss_samples: Vec<RssSample>,
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    pub(super) rss_process_samples: Vec<CheckpointRssProcessSamples>,
}

pub(super) fn checkpoint_soak_process_error(message: impl Into<String>) -> CalcFlowError {
    CalcFlowError::InvalidArgument {
        field: "checkpoint_restart_soak_process".into(),
        message: message.into(),
    }
}

pub(super) fn checkpoint_soak_plan_hash(plan: &CheckpointSoakProcessPlan) -> String {
    sha256_bytes(
        &serde_json::to_vec(&json!({
            "schema": plan.schema,
            "commit": plan.commit,
            "executable_sha256": plan.executable_sha256,
            "run_root": plan.run_root,
            "parent_pid": plan.parent_pid,
            "parent_launch_offset_micros": plan.parent_launch_offset_micros,
            "generation": plan.generation,
            "sample_start": plan.sample_start,
            "sample_end": plan.sample_end,
            "checkpoint_interval_millis": plan.checkpoint_interval_millis,
            "checkpoint_timeout_millis": plan.checkpoint_timeout_millis,
            "target_checkpoints": plan.target_checkpoints,
            "sink_write_delay_millis": plan.sink_write_delay_millis,
            "retained_epochs": plan.retained_epochs,
            "final_generation": plan.final_generation,
            "mode": plan.mode,
        }))
        .unwrap(),
    )
}

pub(super) fn bind_checkpoint_soak_parent_launch(
    plan: &mut CheckpointSoakProcessPlan,
    elapsed: Duration,
) -> Result<()> {
    plan.parent_launch_offset_micros = checkpoint_soak_parent_elapsed_micros(elapsed)?;
    plan.config_hash = checkpoint_soak_plan_hash(plan);
    Ok(())
}

pub(super) fn checkpoint_soak_parent_elapsed_micros(elapsed: Duration) -> Result<u64> {
    u64::try_from(elapsed.as_micros()).map_err(|_| {
        checkpoint_soak_process_error("checkpoint soak parent clock exhausted u64 microseconds")
    })
}

pub(super) fn checkpoint_soak_process_plans(
    run_root: &Path,
    commit: &str,
    executable_sha256: &str,
    mode: CheckpointSoakProcessMode,
) -> Vec<CheckpointSoakProcessPlan> {
    checkpoint_soak_process_plans_with_retention(
        run_root,
        commit,
        executable_sha256,
        mode,
        CHECKPOINT_SOAK_RETAINED_EPOCHS,
    )
}

pub(super) fn checkpoint_soak_process_plans_with_retention(
    run_root: &Path,
    commit: &str,
    executable_sha256: &str,
    mode: CheckpointSoakProcessMode,
    retained_epochs: usize,
) -> Vec<CheckpointSoakProcessPlan> {
    let ranges = match mode {
        CheckpointSoakProcessMode::Smoke => [0..0, 0..0, 0..0],
        CheckpointSoakProcessMode::Standard => [0..40, 40..80, 80..120],
    };
    ranges
        .into_iter()
        .enumerate()
        .map(|(generation, samples)| {
            let mut plan = CheckpointSoakProcessPlan {
                schema: CHECKPOINT_SOAK_PLAN_SCHEMA.into(),
                commit: commit.into(),
                executable_sha256: executable_sha256.into(),
                run_root: run_root.to_path_buf(),
                parent_pid: std::process::id(),
                parent_launch_offset_micros: 0,
                generation,
                sample_start: samples.start,
                sample_end: samples.end,
                checkpoint_interval_millis: match mode {
                    CheckpointSoakProcessMode::Smoke => 50,
                    CheckpointSoakProcessMode::Standard => 5_000,
                },
                checkpoint_timeout_millis: CHECKPOINT_SOAK_CHECKPOINT_TIMEOUT_MILLIS,
                target_checkpoints: match mode {
                    CheckpointSoakProcessMode::Smoke => CHECKPOINT_SOAK_SMOKE_TARGET_CHECKPOINTS,
                    CheckpointSoakProcessMode::Standard => 0,
                },
                sink_write_delay_millis: match mode {
                    CheckpointSoakProcessMode::Smoke => 0,
                    CheckpointSoakProcessMode::Standard => 20,
                },
                retained_epochs,
                final_generation: generation + 1 == CHECKPOINT_SOAK_GENERATIONS,
                mode,
                config_hash: String::new(),
            };
            plan.config_hash = checkpoint_soak_plan_hash(&plan);
            plan
        })
        .collect()
}

pub(super) fn write_checkpoint_soak_document<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let parent = prepare_checkpoint_soak_destination(path)?;
    let bytes = serde_json::to_vec(value).map_err(|error| CalcFlowError::Internal {
        message: format!("checkpoint soak document encoding failed: {error}"),
    })?;
    let temporary = path.with_extension(format!("json.tmp-{}", std::process::id()));
    let _file = create_synced_evidence_file(&temporary, &bytes)?;
    std::fs::rename(&temporary, path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    sync_checkpoint_soak_directory(parent)
}

fn prepare_checkpoint_soak_destination(path: &Path) -> Result<&Path> {
    let parent = path.parent().ok_or_else(|| {
        checkpoint_soak_process_error("checkpoint soak document has no parent directory")
    })?;
    std::fs::create_dir_all(parent).map_err(|source| CalcFlowError::Io {
        path: parent.display().to_string(),
        source,
    })?;
    if path.exists() {
        return Err(CalcFlowError::Conflict {
            resource: "checkpoint soak evidence".into(),
            key: path.display().to_string(),
        });
    }
    Ok(parent)
}

#[cfg(unix)]
pub(super) fn sync_checkpoint_soak_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| CalcFlowError::Io {
            path: path.display().to_string(),
            source,
        })
}

#[cfg(not(unix))]
#[cfg_attr(
    not(unix),
    allow(
        clippy::unnecessary_wraps,
        reason = "non-unix stub keeps the unix fallible signature so call sites stay platform-agnostic"
    )
)]
pub(super) fn sync_checkpoint_soak_directory(_path: &Path) -> Result<()> {
    Ok(())
}

pub(super) fn read_checkpoint_soak_document<T>(path: &Path) -> Result<T>
where
    T: DeserializeOwned + Serialize,
{
    let bytes = read_bounded_checkpoint_soak_document(path)?;
    let value: T = serde_json::from_slice(&bytes).map_err(|error| {
        checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is malformed: {error}",
            path.display()
        ))
    })?;
    let canonical = serde_json::to_vec(&value).unwrap();
    validate_checkpoint_soak_document_bytes(path, &bytes, &canonical)?;
    Ok(value)
}

fn read_bounded_checkpoint_soak_document(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })?;
    if metadata.file_type().is_symlink()
        || !metadata.is_file()
        || metadata.len() > CHECKPOINT_SOAK_REPORT_LIMIT
    {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is not a bounded regular file",
            path.display()
        )));
    }
    std::fs::read(path).map_err(|source| CalcFlowError::Io {
        path: path.display().to_string(),
        source,
    })
}

fn validate_checkpoint_soak_document_bytes(
    path: &Path,
    bytes: &[u8],
    canonical: &[u8],
) -> Result<()> {
    if canonical != bytes {
        let mismatch = canonical
            .iter()
            .zip(bytes)
            .position(|(canonical, stored)| canonical != stored)
            .unwrap_or(canonical.len().min(bytes.len()));
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak document {} is not canonical at byte {mismatch} (stored {}, canonical {})",
            path.display(),
            bytes.len(),
            canonical.len(),
        )));
    }
    Ok(())
}

pub(super) fn checkpoint_soak_fresh_checkpoint_idle(
    status: &engine::CheckpointStatus,
    previous_completed: Option<Epoch>,
) -> bool {
    status.current_epoch.is_none() && status.last_completed_epoch > previous_completed
}

pub(super) async fn read_checkpoint_manifest_document(
    path: &Path,
) -> Option<engine::CheckpointManifest> {
    let bytes = match tokio::fs::read(path).await {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return None,
        Err(error) => panic!("checkpoint soak could not read manifest: {error}"),
    };
    Some(engine::CheckpointManifest::from_bytes(&bytes).unwrap())
}

pub(super) fn checkpoint_soak_cursor_order(value: &str) -> Result<u64> {
    let bytes = hex::decode(value).map_err(|error| {
        checkpoint_soak_process_error(format!("checkpoint soak cursor is not hex: {error}"))
    })?;
    let order: [u8; 8] = bytes.try_into().map_err(|_| {
        checkpoint_soak_process_error("checkpoint soak cursor is not an exact u64 order")
    })?;
    Ok(u64::from_be_bytes(order))
}

pub(super) fn validate_checkpoint_soak_samples(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> Result<()> {
    let expected = (plan.sample_start..plan.sample_end).collect::<Vec<_>>();
    let observed = report
        .samples
        .iter()
        .map(|sample| sample.index)
        .collect::<Vec<_>>();
    if observed != expected {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child sample indices are invalid: observed {observed:?}, expected {expected:?}"
        )));
    }
    let cadence_micros = u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap();
    let tolerance_micros = u64::try_from(CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_micros()).unwrap();
    if let Some(issue) = report
        .samples
        .iter()
        .enumerate()
        .find_map(|(local_index, sample)| {
            checkpoint_soak_sample_issue(
                report,
                local_index,
                sample,
                cadence_micros,
                tolerance_micros,
            )
        })
    {
        return Err(checkpoint_soak_process_error(issue));
    }
    Ok(())
}

pub(super) fn checkpoint_soak_sample_issue(
    report: &CheckpointSoakProcessReport,
    local_index: usize,
    sample: &CheckpointSoakProcessSample,
    cadence_micros: u64,
    tolerance_micros: u64,
) -> Option<String> {
    checkpoint_soak_sample_timing_issue(
        report,
        local_index,
        sample,
        cadence_micros,
        tolerance_micros,
    )
    .or_else(|| checkpoint_soak_sample_resource_issue(local_index, sample))
}

fn checkpoint_soak_sample_timing_issue(
    report: &CheckpointSoakProcessReport,
    local_index: usize,
    sample: &CheckpointSoakProcessSample,
    cadence_micros: u64,
    tolerance_micros: u64,
) -> Option<String> {
    let target = cadence_micros * u64::try_from(local_index + 1).unwrap();
    let Some(observed) = sample
        .elapsed_micros
        .checked_sub(report.sampling_started_micros)
    else {
        return Some(format!(
            "checkpoint soak generation {} sample {} precedes sampling start",
            report.generation, sample.index
        ));
    };
    if observed < target - tolerance_micros || observed > target + tolerance_micros {
        return Some(format!(
            "checkpoint soak generation {} sample {} elapsed {observed}us outside target {target}us +/-{tolerance_micros}us",
            report.generation, sample.index
        ));
    }
    if sample.elapsed_micros > report.generation_finished_micros {
        return Some(format!(
            "checkpoint soak sample {local_index} finishes after its generation"
        ));
    }
    None
}

fn checkpoint_soak_sample_resource_issue(
    local_index: usize,
    sample: &CheckpointSoakProcessSample,
) -> Option<String> {
    if sample.rss_kib == 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} has no RSS evidence"
        ));
    }
    if sample.task_count == 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} has no live tasks"
        ));
    }
    if sample.failed_checkpoints != 0 {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} failed checkpoints",
            sample.failed_checkpoints
        ));
    }
    if sample.manifest_count > 3 {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} manifests",
            sample.manifest_count
        ));
    }
    if sample.state_bytes > MAX_CHECKPOINT_SOAK_STATE_BYTES {
        return Some(format!(
            "checkpoint soak sample {local_index} records {} state bytes above {}",
            sample.state_bytes, MAX_CHECKPOINT_SOAK_STATE_BYTES
        ));
    }
    None
}

pub(super) fn validate_checkpoint_soak_advance(report: &CheckpointSoakProcessReport) -> Result<()> {
    let expected_sources = BTreeSet::from(["left", "right"]);
    let restored_sources = report
        .restored_cursor_orders
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let terminal_sources = report
        .terminal_cursor_orders
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if restored_sources != expected_sources || terminal_sources != expected_sources {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child source identity set is invalid",
        ));
    }
    for source_id in expected_sources {
        validate_checkpoint_soak_cursor_advance(report, source_id)?;
    }
    if report
        .restored_epoch
        .is_some_and(|restored| report.terminal_epoch <= restored)
        || report.restored_epoch.is_none() && report.terminal_epoch == 0
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak child did not advance the durable epoch",
        ));
    }
    Ok(())
}

fn validate_checkpoint_soak_cursor_advance(
    report: &CheckpointSoakProcessReport,
    source_id: &str,
) -> Result<()> {
    let terminal = checkpoint_soak_cursor_order(&report.terminal_cursor_orders[source_id])?;
    match &report.restored_cursor_orders[source_id] {
        Some(restored) if terminal <= checkpoint_soak_cursor_order(restored)? => {
            return Err(checkpoint_soak_process_error(format!(
                "checkpoint soak child did not advance source {source_id:?}"
            )));
        }
        None if report.generation != 0 || terminal == 0 => {
            return Err(checkpoint_soak_process_error(format!(
                "checkpoint soak child has invalid initial cursor for {source_id:?}"
            )));
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn validate_checkpoint_soak_report(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
    exit_code: i32,
) -> Result<()> {
    validate_checkpoint_soak_report_bounds(plan, report, exit_code)?;
    if plan.final_generation && report.temporary_artifacts != 0 {
        return Err(checkpoint_soak_process_error(format!(
            "final checkpoint soak child retained {} temporary artifacts",
            report.temporary_artifacts
        )));
    }
    validate_checkpoint_soak_restore_evidence(plan, report)?;
    validate_checkpoint_soak_samples(plan, report)?;
    validate_checkpoint_soak_advance(report)
}

fn checkpoint_soak_build_identity_matches(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> bool {
    report.schema == CHECKPOINT_SOAK_PROCESS_SCHEMA
        && report.commit == plan.commit
        && report.executable_sha256 == plan.executable_sha256
        && report.config_hash == plan.config_hash
        && report.run_root == plan.run_root
}

fn checkpoint_soak_process_identity_matches(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> bool {
    report.parent_pid == plan.parent_pid
        && report.pid != 0
        && report.pid != plan.parent_pid
        && report.generation == plan.generation
        && report.sample_start == plan.sample_start
        && report.sample_end == plan.sample_end
}

fn checkpoint_soak_report_timing_matches(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
    preflight_micros: Option<u64>,
    maximum_preflight_micros: u64,
) -> bool {
    report.generation_started_micros == plan.parent_launch_offset_micros
        && preflight_micros.is_some_and(|elapsed| elapsed <= maximum_preflight_micros)
        && report.generation_finished_micros > report.sampling_started_micros
}

fn checkpoint_soak_source_lifecycle_matches(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> bool {
    let expected_cause = if plan.final_generation {
        "natural_end"
    } else {
        "explicit_cancel"
    };
    report.terminal_cause == expected_cause
        && report.source_open_events == 2
        && report.source_close_events == 2
        && report.sink_close_events == 2
}

fn checkpoint_soak_terminal_bounds_match(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> bool {
    report.failed_checkpoints == 0
        && report.maximum_manifest_count <= plan.retained_epochs + 1
        && report.maximum_state_bytes <= MAX_CHECKPOINT_SOAK_STATE_BYTES
        && report.terminal_tasks == 0
        && report.terminal_charged_edges == 0
        && report.terminal_registries == (0, 0)
}

fn checkpoint_soak_maximum_preflight_micros(mode: CheckpointSoakProcessMode) -> u64 {
    let maximum = match mode {
        CheckpointSoakProcessMode::Smoke => CHECKPOINT_SOAK_SMOKE_GENERATION_TIMEOUT,
        CheckpointSoakProcessMode::Standard => MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT,
    };
    u64::try_from(maximum.as_micros()).unwrap()
}

fn validate_checkpoint_soak_report_bounds(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
    exit_code: i32,
) -> Result<()> {
    let preflight_micros = report
        .sampling_started_micros
        .checked_sub(report.generation_started_micros);
    let maximum_preflight_micros = checkpoint_soak_maximum_preflight_micros(plan.mode);
    if exit_code != 0
        || !checkpoint_soak_build_identity_matches(plan, report)
        || !checkpoint_soak_process_identity_matches(plan, report)
        || !checkpoint_soak_report_timing_matches(
            plan,
            report,
            preflight_micros,
            maximum_preflight_micros,
        )
        || !checkpoint_soak_source_lifecycle_matches(plan, report)
        || !checkpoint_soak_terminal_bounds_match(plan, report)
    {
        return Err(checkpoint_soak_process_error(format!(
            "checkpoint soak child report identity, exit, or terminal bounds are invalid: \
             exit={exit_code}, generation={}, cause={}, opens={}, source_closes={}, \
             sink_closes={}, failed={}, manifests={}, state_bytes={}, tasks={}, charged_edges={}, \
             registries={:?}, mode={:?}, preflight_micros={preflight_micros:?}, \
             maximum_preflight_micros={maximum_preflight_micros}",
            report.generation,
            report.terminal_cause,
            report.source_open_events,
            report.source_close_events,
            report.sink_close_events,
            report.failed_checkpoints,
            report.maximum_manifest_count,
            report.maximum_state_bytes,
            report.terminal_tasks,
            report.terminal_charged_edges,
            report.terminal_registries,
            plan.mode,
        )));
    }
    Ok(())
}

fn checkpoint_soak_restored_state_complete(report: &CheckpointSoakProcessReport) -> bool {
    report.restored_epoch.is_some()
        && report.restored_cursor_orders.values().all(Option::is_some)
        && report.restored_window_state
        && report.restored_watermarks
        && report.restored_progress
}

fn validate_checkpoint_soak_restore_evidence(
    plan: &CheckpointSoakProcessPlan,
    report: &CheckpointSoakProcessReport,
) -> Result<()> {
    if plan.generation == 0 {
        if report.restored_epoch.is_some()
            || report.restored_cursor_orders.values().any(Option::is_some)
        {
            return Err(checkpoint_soak_process_error(
                "initial checkpoint soak child unexpectedly restored in-memory continuity",
            ));
        }
    } else if !checkpoint_soak_restored_state_complete(report) {
        return Err(checkpoint_soak_process_error(
            "restarted checkpoint soak child lacks durable restore evidence",
        ));
    }
    Ok(())
}

pub(super) fn checkpoint_soak_child_command(executable: &Path, plan_path: &Path) -> Command {
    let mut command = Command::new(executable);
    command
        .arg(CHECKPOINT_SOAK_CHILD_TEST)
        .arg("--exact")
        .arg("--nocapture")
        .arg("--test-threads=1")
        .env(CHECKPOINT_SOAK_CHILD_ENV, plan_path)
        .env("MALLOC_ARENA_MAX", CHECKPOINT_SOAK_MALLOC_ARENA_MAX)
        .env(
            "MALLOC_TRIM_THRESHOLD_",
            CHECKPOINT_SOAK_MALLOC_TRIM_THRESHOLD,
        )
        .env("MALLOC_TOP_PAD_", CHECKPOINT_SOAK_MALLOC_TOP_PAD);
    command
}

pub(super) fn checkpoint_soak_process_timeout(plan: &CheckpointSoakProcessPlan) -> Duration {
    match plan.mode {
        CheckpointSoakProcessMode::Smoke => CHECKPOINT_SOAK_SMOKE_GENERATION_TIMEOUT,
        CheckpointSoakProcessMode::Standard => {
            let samples = u64::try_from(plan.sample_end - plan.sample_start).unwrap();
            let alignment = if plan.final_generation {
                Duration::ZERO
            } else {
                CHECKPOINT_SOAK_CANCEL_ALIGNMENT_TIMEOUT
            };
            CHECKPOINT_SOAK_CADENCE * u32::try_from(samples).unwrap()
                + MAX_CHECKPOINT_SOAK_CHILD_PREFLIGHT
                + alignment
                + CHECKPOINT_SOAK_SETTLE_TIMEOUT
                + CHECKPOINT_SOAK_PROCESS_OVERHEAD_BUDGET
        }
    }
}

pub(super) fn validate_checkpoint_soak_process_set(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exit_codes: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    validate_checkpoint_soak_executions(plans, reports, exit_codes, parent_timings)?;
    validate_checkpoint_soak_continuity(reports)?;
    validate_checkpoint_soak_sample_set(plans[0].mode, reports)?;
    let final_report = reports.last().unwrap();
    if final_report.duplicate_records != 0 || final_report.missing_records != 0 {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak transactional output is not exactly once",
        ));
    }
    Ok(())
}

fn validate_checkpoint_soak_process_counts(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exit_codes: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    if plans.len() != CHECKPOINT_SOAK_GENERATIONS
        || reports.len() != plans.len()
        || exit_codes.len() != plans.len()
        || parent_timings.len() != plans.len()
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process set is incomplete",
        ));
    }
    Ok(())
}

fn validate_checkpoint_soak_executions(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    exit_codes: &[i32],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    validate_checkpoint_soak_process_counts(plans, reports, exit_codes, parent_timings)?;
    let pids = reports
        .iter()
        .map(|report| report.pid)
        .collect::<BTreeSet<_>>();
    if pids.len() != CHECKPOINT_SOAK_GENERATIONS {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process IDs are not distinct",
        ));
    }
    for ((plan, report), exit_code) in plans.iter().zip(reports).zip(exit_codes) {
        validate_checkpoint_soak_report(plan, report, *exit_code)?;
    }
    validate_checkpoint_soak_timeline(plans, reports, parent_timings)
}

fn validate_checkpoint_soak_continuity(reports: &[CheckpointSoakProcessReport]) -> Result<()> {
    for pair in reports.windows(2) {
        let previous = &pair[0];
        let restarted = &pair[1];
        let previous_cursors = previous
            .terminal_cursor_orders
            .iter()
            .map(|(source_id, order)| (source_id.clone(), Some(order.clone())))
            .collect::<BTreeMap<_, _>>();
        if restarted.restored_epoch != Some(previous.terminal_epoch)
            || restarted.restored_cursor_orders != previous_cursors
        {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak child did not continue from prior filesystem evidence",
            ));
        }
    }
    Ok(())
}

fn validate_checkpoint_soak_sample_set(
    mode: CheckpointSoakProcessMode,
    reports: &[CheckpointSoakProcessReport],
) -> Result<()> {
    let observed_samples = reports
        .iter()
        .flat_map(|report| report.samples.iter().map(|sample| sample.index))
        .collect::<Vec<_>>();
    let expected_samples = match mode {
        CheckpointSoakProcessMode::Smoke => Vec::new(),
        CheckpointSoakProcessMode::Standard => (0..CHECKPOINT_SOAK_SAMPLE_COUNT).collect(),
    };
    if observed_samples != expected_samples {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak process samples contain a gap or duplicate",
        ));
    }
    Ok(())
}

pub(super) fn validate_checkpoint_soak_timeline(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    let maximum_restart_gap = u64::try_from(MAX_CHECKPOINT_SOAK_RESTART_GAP.as_micros()).unwrap();
    validate_checkpoint_soak_parent_timings(plans, reports, parent_timings)?;
    validate_checkpoint_soak_restart_gaps(parent_timings, maximum_restart_gap)?;
    if plans[0].mode == CheckpointSoakProcessMode::Smoke {
        return Ok(());
    }
    let samples = reports
        .iter()
        .flat_map(|report| report.samples.iter())
        .collect::<Vec<_>>();
    let tolerance = u64::try_from(CHECKPOINT_SOAK_CADENCE_TOLERANCE.as_micros()).unwrap();
    let cadence = u64::try_from(CHECKPOINT_SOAK_CADENCE.as_micros()).unwrap();
    validate_checkpoint_soak_sample_cadence(&samples, cadence, tolerance, maximum_restart_gap)?;
    validate_checkpoint_soak_target_span(
        &samples,
        reports[0].sampling_started_micros,
        cadence,
        tolerance,
        maximum_restart_gap,
    )
}

fn validate_checkpoint_soak_parent_timings(
    plans: &[CheckpointSoakProcessPlan],
    reports: &[CheckpointSoakProcessReport],
    parent_timings: &[CheckpointSoakParentTiming],
) -> Result<()> {
    let maximum_process_overhead =
        u64::try_from(CHECKPOINT_SOAK_PROCESS_OVERHEAD_BUDGET.as_micros()).unwrap();
    for ((plan, report), timing) in plans.iter().zip(reports).zip(parent_timings) {
        if timing.generation != plan.generation
            || timing.launch_micros != report.generation_started_micros
            || timing
                .finish_micros
                .checked_sub(report.generation_finished_micros)
                .is_none_or(|overhead| overhead > maximum_process_overhead)
        {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak parent timing does not bound the child report",
            ));
        }
    }
    Ok(())
}

fn validate_checkpoint_soak_restart_gaps(
    parent_timings: &[CheckpointSoakParentTiming],
    maximum_restart_gap: u64,
) -> Result<()> {
    for pair in parent_timings.windows(2) {
        let Some(gap) = pair[1].launch_micros.checked_sub(pair[0].finish_micros) else {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak process restart gap is invalid",
            ));
        };
        if gap > maximum_restart_gap {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak process restart gap is invalid",
            ));
        }
    }
    Ok(())
}

fn validate_checkpoint_soak_sample_cadence(
    samples: &[&CheckpointSoakProcessSample],
    cadence: u64,
    tolerance: u64,
    maximum_restart_gap: u64,
) -> Result<()> {
    for pair in samples.windows(2) {
        let Some(gap) = pair[1].elapsed_micros.checked_sub(pair[0].elapsed_micros) else {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak observed cadence or restart gap is invalid",
            ));
        };
        let restart_boundary = CHECKPOINT_SOAK_RESTART_SAMPLES.contains(&pair[0].index);
        let maximum = if restart_boundary {
            cadence + maximum_restart_gap
        } else {
            cadence + tolerance
        };
        if gap < cadence - tolerance || gap > maximum {
            return Err(checkpoint_soak_process_error(
                "checkpoint soak observed cadence or restart gap is invalid",
            ));
        }
    }
    Ok(())
}

fn validate_checkpoint_soak_target_span(
    samples: &[&CheckpointSoakProcessSample],
    sampling_started_micros: u64,
    cadence: u64,
    tolerance: u64,
    maximum_restart_gap: u64,
) -> Result<()> {
    let Some(first) = samples.first() else {
        return Err(checkpoint_soak_process_error(
            "standard checkpoint soak produced no real samples",
        ));
    };
    let last = samples.last().unwrap();
    let maximum_final_elapsed =
        u64::try_from(TARGET_DURATION.as_micros()).unwrap() + 2 * maximum_restart_gap + tolerance;
    let first_elapsed = first.elapsed_micros.checked_sub(sampling_started_micros);
    if !first_elapsed
        .is_some_and(|elapsed| elapsed >= cadence - tolerance && elapsed <= cadence + tolerance)
        || last.elapsed_micros < u64::try_from(TARGET_DURATION.as_micros()).unwrap()
        || last.elapsed_micros > maximum_final_elapsed
    {
        return Err(checkpoint_soak_process_error(
            "checkpoint soak observed timeline does not span the real target duration",
        ));
    }
    Ok(())
}

pub(super) fn aggregate_checkpoint_soak_processes(
    plans: Vec<CheckpointSoakProcessPlan>,
    reports: Vec<CheckpointSoakProcessReport>,
    exit_codes: Vec<i32>,
    parent_timings: Vec<CheckpointSoakParentTiming>,
) -> CheckpointSoakProcessEvidence {
    let final_report = reports.last().unwrap();
    let compacted_epochs = reports
        .iter()
        .flat_map(|report| report.compacted_epochs.iter().copied())
        .collect::<BTreeSet<_>>();
    let rss_process_samples = reports
        .iter()
        .map(|report| CheckpointRssProcessSamples {
            generation: report.generation,
            pid: report.pid,
            samples: report
                .samples
                .iter()
                .map(|sample| RssSample {
                    elapsed_seconds: Duration::from_micros(sample.elapsed_micros).as_secs_f64(),
                    rss_kib: sample.rss_kib,
                })
                .collect(),
        })
        .collect::<Vec<_>>();
    let rss_samples = rss_process_samples
        .iter()
        .flat_map(|process| process.samples.iter().copied())
        .collect();
    let report = CheckpointRestartSoakReport {
        restarts: reports.len().saturating_sub(1),
        generation_process_ids: reports.iter().map(|report| report.pid).collect(),
        generation_exit_codes: exit_codes,
        completed_epochs: final_report.terminal_epoch,
        compacted_window_epochs: compacted_epochs.len(),
        maximum_manifest_count: reports
            .iter()
            .map(|report| report.maximum_manifest_count)
            .max()
            .unwrap_or(0),
        maximum_state_bytes: reports
            .iter()
            .map(|report| report.maximum_state_bytes)
            .max()
            .unwrap_or(0),
        source_records: final_report.source_records,
        output_records: final_report.output_records,
        duplicate_records: final_report.duplicate_records,
        missing_records: final_report.missing_records,
        temporary_artifacts: final_report.temporary_artifacts,
        terminal_tasks: reports.iter().map(|report| report.terminal_tasks).sum(),
        terminal_charged_edges: reports
            .iter()
            .map(|report| report.terminal_charged_edges)
            .sum(),
        terminal_registries: reports.iter().fold((0, 0), |total, report| {
            (
                total.0 + report.terminal_registries.0,
                total.1 + report.terminal_registries.1,
            )
        }),
    };
    CheckpointSoakProcessEvidence {
        report,
        plans,
        processes: reports,
        parent_timings,
        rss_samples,
        rss_process_samples,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ManifestParentSyncIoSupport {
    Supported,
    Unsupported(&'static str),
}

pub(super) const fn classify_manifest_parent_sync_io_support(
    is_unix: bool,
    permission_denied: bool,
) -> ManifestParentSyncIoSupport {
    if !is_unix {
        ManifestParentSyncIoSupport::Unsupported("requires Unix directory permission semantics")
    } else if !permission_denied {
        ManifestParentSyncIoSupport::Unsupported(
            "this process bypasses mode 0111 directory restrictions",
        )
    } else {
        ManifestParentSyncIoSupport::Supported
    }
}

impl CheckpointTemporaryArtifacts {
    pub(super) const fn total(&self) -> usize {
        self.state + self.manifests + self.sink_a + self.sink_b
    }
}
