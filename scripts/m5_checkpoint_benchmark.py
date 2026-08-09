from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess  # nosec B404 - fixed allowlisted executables only
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

COMMON_CASE = "m5/common/stream_channel_data_roundtrip"
COMMON_SAMPLE_COUNT = 30
PRIVATE_SAMPLE_COUNT = 10
REGRESSION_THRESHOLD_PERCENT = 5.0
RUN_ORDER = ("B1", "C1", "B2", "C2")
PRIVATE_RUN_ORDER = ("P1", "P2")
MAX_PRIVATE_REPEAT_SPREAD_PERCENT = 10.0
PRIVATE_BUILD_IDENTITY_SCHEMA = "calc-flow.m5-private-build-identity.v1"
M5_ABSOLUTE_CASES = (
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
)
PRIVATE_TEST = (
    "runtime::streaming::soak::private_m5_epoch_checkpoint_absolute_benchmark"
)
SCRIPT_ROOT = Path(__file__).resolve().parent
COMMON_HARNESS_ROOT = SCRIPT_ROOT / "m5_checkpoint_benchmark_harness"
COMMON_HARNESS_SOURCE = COMMON_HARNESS_ROOT / "src" / "main.rs"
COMMON_HARNESS_MANIFEST = COMMON_HARNESS_ROOT / "Cargo.toml"
COMMON_HARNESS_FILES = (Path("Cargo.toml"), Path("src/main.rs"))
SOURCE_CONTRACT_FILES = (
    Path("Cargo.toml"),
    Path("Cargo.lock"),
    Path("scripts/m5_checkpoint_benchmark.py"),
    Path("scripts/m5_checkpoint_benchmark_harness/Cargo.toml"),
    Path("scripts/m5_checkpoint_benchmark_harness/src/main.rs"),
    Path("crates/calc-flow/src/runtime/streaming/operator_task.rs"),
    Path("crates/calc-flow/src/runtime/streaming/soak.rs"),
)


@dataclass(frozen=True, slots=True)
class RefSnapshot:
    role: str
    commit: str
    tree: str
    worktree: Path


@dataclass(frozen=True, slots=True)
class CommonRunPlan:
    label: str
    snapshot: RefSnapshot
    cwd: Path
    harness_root: Path
    target_dir: Path
    evidence_root: Path
    report_path: Path


@dataclass(frozen=True, slots=True)
class PrivateRunPlan:
    label: str
    candidate: RefSnapshot
    workspace: Path
    target_dir: Path
    evidence_root: Path


@dataclass(frozen=True, slots=True)
class TrustedCargoCommand:
    executable: Path
    arguments: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.executable != _trusted_system_executable("cargo"):
            raise ValueError("trusted cargo executable does not match the fixed tool")
        if any("\0" in argument for argument in self.arguments):
            raise ValueError("trusted cargo argument contains NUL")

    def argv(self) -> list[str]:
        return [str(self.executable), *self.arguments]


def build_run_plan(
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
) -> list[CommonRunPlan]:
    snapshots = (baseline, candidate, baseline, candidate)
    result = []
    for label, snapshot in zip(RUN_ORDER, snapshots, strict=True):
        run_root = snapshot.worktree / "target" / "m5-checkpoint-common" / label
        evidence_root = output_root / "common-runs" / label
        result.append(
            CommonRunPlan(
                label=label,
                snapshot=snapshot,
                cwd=snapshot.worktree,
                harness_root=run_root / "harness",
                target_dir=run_root / "cargo-target",
                evidence_root=evidence_root,
                report_path=evidence_root / "measurement.json",
            )
        )
    return result


def build_private_run_plan(
    output_root: Path, candidate: RefSnapshot
) -> list[PrivateRunPlan]:
    return [
        PrivateRunPlan(
            label=label,
            candidate=candidate,
            workspace=output_root / "worktrees" / f"private-{label}",
            target_dir=(
                output_root
                / "worktrees"
                / f"private-{label}"
                / "target"
                / "m5-private-absolute"
            ),
            evidence_root=output_root / "private-runs" / label,
        )
        for label in PRIVATE_RUN_ORDER
    ]


def _reproducible_build_environment(
    workspace: Path, *, harness_root: Path | None = None
) -> dict[str, str]:
    remaps = []
    if harness_root is not None:
        remaps.append(
            f"--remap-path-prefix={harness_root.resolve()}=/calc-flow/harness"
        )
    remaps.append(f"--remap-path-prefix={workspace.resolve()}=/calc-flow/source")
    return {
        "CARGO_INCREMENTAL": "0",
        "CARGO_PROFILE_RELEASE_CODEGEN_UNITS": "1",
        "CARGO_PROFILE_RELEASE_DEBUG": "0",
        "CARGO_PROFILE_RELEASE_INCREMENTAL": "false",
        "CARGO_ENCODED_RUSTFLAGS": "\x1f".join(remaps),
        "SOURCE_DATE_EPOCH": "0",
    }


def common_build_environment(
    plan: CommonRunPlan, harness_sha256: str
) -> dict[str, str]:
    return {
        **_reproducible_build_environment(
            plan.snapshot.worktree, harness_root=plan.harness_root
        ),
        "CALC_FLOW_M5_SOURCE_COMMIT": plan.snapshot.commit,
        "CALC_FLOW_M5_SOURCE_TREE": plan.snapshot.tree,
        "CALC_FLOW_M5_HARNESS_SHA256": harness_sha256,
        "CARGO_TARGET_DIR": str(plan.target_dir),
    }


def common_run_environment(
    plan: CommonRunPlan,
    build_environment: dict[str, str],
    *,
    executable_sha256: str,
) -> dict[str, str]:
    return {
        **build_environment,
        "CALC_FLOW_M5_RUN_LABEL": plan.label,
        "CALC_FLOW_M5_COMMON_OUTPUT": str(plan.report_path.resolve()),
        "CALC_FLOW_M5_COMMON_EXECUTABLE": str(_common_executable(plan).resolve()),
        "CALC_FLOW_M5_COMMON_EXECUTABLE_SHA256": _full_sha256(
            executable_sha256, "common executable hash"
        ),
    }


def _common_executable(plan: CommonRunPlan) -> Path:
    executable = plan.target_dir / "release" / "calc-flow-m5-common-benchmark"
    return executable.with_suffix(".exe") if sys.platform == "win32" else executable


def hash_harness_files(root: Path) -> str:
    digest = hashlib.sha256()
    for relative in COMMON_HARNESS_FILES:
        data = (root / relative).read_bytes()
        digest.update(str(relative).encode())
        digest.update(len(data).to_bytes(8, "big"))
        digest.update(data)
    return digest.hexdigest()


def materialize_common_harness(destination: Path) -> Path:
    if destination.exists():
        raise FileExistsError(
            f"common harness destination already exists: {destination}"
        )
    shutil.copytree(COMMON_HARNESS_ROOT, destination)
    return destination


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_hashed_json(path: Path, payload: dict[str, object]) -> None:
    bytes_ = (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    digest_path = Path(f"{path}.sha256")
    if path.exists() or digest_path.exists():
        raise FileExistsError(f"immutable benchmark artifact already exists: {path}")
    _atomic_create(path, bytes_)
    _atomic_create(
        digest_path, f"{hashlib.sha256(bytes_).hexdigest()}\n".encode("ascii")
    )
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _atomic_create(path: Path, bytes_: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(bytes_)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_hashed_json(path: Path) -> dict[str, object]:
    digest_path = Path(f"{path}.sha256")
    _require_regular_file(path, "report")
    _require_regular_file(digest_path, "digest")
    bytes_ = path.read_bytes()
    expected = digest_path.read_text(encoding="utf-8").strip()
    if not _matches_sha256(bytes_, expected):
        raise ValueError(f"benchmark report hash does not match: {path}")
    value = json.loads(bytes_)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark report must be an object: {path}")
    return value


def write_artifact_manifest(root: Path, artifacts: Sequence[Path]) -> Path:
    canonical_root = root.resolve()
    entries = []
    seen = set()
    for artifact in artifacts:
        _require_regular_file(artifact, "artifact")
        canonical = artifact.resolve()
        if not canonical.is_relative_to(canonical_root):
            raise ValueError(f"benchmark artifact escapes evidence root: {artifact}")
        relative = canonical.relative_to(canonical_root)
        if relative in seen:
            raise ValueError(f"benchmark artifact is repeated: {relative}")
        seen.add(relative)
        entries.append(
            {
                "path": relative.as_posix(),
                "size": canonical.stat().st_size,
                "sha256": sha256_file(canonical),
            }
        )
    entries.sort(key=lambda entry: str(entry["path"]))
    manifest = root / "artifact-manifest.json"
    write_hashed_json(
        manifest,
        {
            "schema": "calc-flow.m5-checkpoint-artifact-manifest.v1",
            "root": str(canonical_root),
            "artifacts": entries,
        },
    )
    return manifest


def validate_artifact_manifest(root: Path, manifest: Path) -> dict[str, object]:
    canonical_root = root.resolve()
    payload = load_hashed_json(manifest)
    if payload.get("schema") != "calc-flow.m5-checkpoint-artifact-manifest.v1":
        raise ValueError("benchmark artifact manifest schema is invalid")
    if payload.get("root") != str(canonical_root):
        raise ValueError("benchmark artifact manifest root is invalid")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("benchmark artifact manifest entries are missing")
    observed_paths = [
        _validate_artifact_entry(entry, canonical_root) for entry in artifacts
    ]
    if observed_paths != sorted(observed_paths) or len(observed_paths) != len(
        set(observed_paths)
    ):
        raise ValueError("benchmark artifact manifest order is invalid")
    return payload


def _validate_artifact_entry(entry: object, canonical_root: Path) -> str:
    if not isinstance(entry, dict):
        raise ValueError("benchmark artifact manifest entry is invalid")
    relative_value = entry.get("path")
    if not isinstance(relative_value, str):
        raise ValueError("benchmark artifact path is invalid")
    relative = Path(relative_value)
    if relative.is_absolute() or relative.as_posix() != relative_value:
        raise ValueError("benchmark artifact path is not canonical")
    artifact = canonical_root / relative
    _require_regular_file(artifact, "artifact")
    canonical = artifact.resolve()
    if not canonical.is_relative_to(canonical_root):
        raise ValueError("benchmark artifact path escapes evidence root")
    expected = _full_sha256(entry.get("sha256"), "artifact hash")
    if sha256_file(canonical) != expected:
        raise ValueError(f"benchmark artifact hash does not match: {relative}")
    if entry.get("size") != canonical.stat().st_size:
        raise ValueError(f"benchmark artifact size does not match: {relative}")
    return relative_value


def build_source_contract(
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    *,
    source_files: Sequence[Path] = SOURCE_CONTRACT_FILES,
) -> dict[str, object]:
    files = _source_file_entries(candidate.worktree, source_files)
    return {
        "schema": "calc-flow.m5-checkpoint-source-contract.v1",
        "baseline": {
            "commit": baseline.commit,
            "tree": baseline.tree,
            "cargo_lock_sha256": sha256_file(baseline.worktree / "Cargo.lock"),
        },
        "candidate": {
            "commit": candidate.commit,
            "tree": candidate.tree,
            "cargo_lock_sha256": sha256_file(candidate.worktree / "Cargo.lock"),
        },
        "source_files": files,
        "source_files_sha256": _canonical_hash(files),
        "workload_contract_sha256": _canonical_hash(
            {
                "common_case": COMMON_CASE,
                "common_samples": COMMON_SAMPLE_COUNT,
                "private_cases": M5_ABSOLUTE_CASES,
                "private_samples": PRIVATE_SAMPLE_COUNT,
                "threshold_percent": REGRESSION_THRESHOLD_PERCENT,
            }
        ),
    }


def validate_source_contract(
    contract: dict[str, object],
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    *,
    source_files: Sequence[Path] = SOURCE_CONTRACT_FILES,
) -> None:
    expected = build_source_contract(baseline, candidate, source_files=source_files)
    if contract != expected:
        raise ValueError("benchmark source bytes or ref identity do not match")


def _source_file_entries(
    worktree: Path, source_files: Sequence[Path]
) -> list[dict[str, object]]:
    entries = []
    canonical_root = worktree.resolve()
    for relative in source_files:
        if relative.is_absolute() or relative.as_posix().startswith("../"):
            raise ValueError("benchmark source path is not canonical")
        path = canonical_root / relative
        _require_regular_file(path, "source")
        canonical = path.resolve()
        if not canonical.is_relative_to(canonical_root):
            raise ValueError("benchmark source path escapes its worktree")
        entries.append(
            {
                "path": relative.as_posix(),
                "size": canonical.stat().st_size,
                "sha256": sha256_file(canonical),
            }
        )
    return entries


def _require_regular_file(path: Path, kind: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark {kind} must be a regular file: {path}")


def validate_rooted_directory(root: Path, path: Path, label: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"benchmark {label} directory is a symbolic link")
    if not path.is_dir():
        raise ValueError(f"benchmark {label} directory is missing")
    canonical_root = root.resolve()
    canonical = path.resolve()
    if not canonical.is_relative_to(canonical_root):
        raise ValueError(f"benchmark {label} directory escapes evidence root")
    return canonical


def _matches_sha256(bytes_: bytes, expected: str) -> bool:
    return (
        len(expected) == 64
        and all(character in "0123456789abcdef" for character in expected)
        and hashlib.sha256(bytes_).hexdigest() == expected
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2.0
    return ordered[middle]


def common_statistics(samples: Sequence[float]) -> dict[str, object]:
    if len(samples) != COMMON_SAMPLE_COUNT:
        raise ValueError(f"common benchmark requires {COMMON_SAMPLE_COUNT} raw samples")
    normalized = [
        _finite_positive(sample, f"raw sample {index}")
        for index, sample in enumerate(samples)
    ]
    state = 0x4231_4331_4232_4332
    medians = []
    for _ in range(8_192):
        resample = []
        for _ in normalized:
            state = (state * 6_364_136_223_846_793_005 + 1_442_695_040_888_963_407) & (
                (1 << 64) - 1
            )
            resample.append(normalized[state % len(normalized)])
        medians.append(_median(resample))
    medians.sort()
    median = _median(normalized)
    confidence = [medians[len(medians) // 40], medians[len(medians) * 39 // 40]]
    if confidence[0] > median or median > confidence[1]:
        raise ValueError("recomputed common confidence interval is not ordered")
    return {
        "median_ns": median,
        "median_confidence_interval_ns": confidence,
    }


def _full_sha256(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{field} must be a SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def validate_common_run_report(
    path: Path,
    *,
    label: str,
    commit: str,
    tree: str,
    harness_sha256: str,
) -> dict[str, object]:
    report = load_hashed_json(path)
    _validate_common_run_identity(report, commit, tree, harness_sha256)
    validated = _validated_run(report, label)
    _validate_common_samples(report, validated)
    _validate_reported_executable(report, "common benchmark", None)
    return {
        **validated,
        "report_sha256": sha256_file(path),
    }


def _validate_common_run_identity(
    report: dict[str, object], commit: str, tree: str, harness_sha256: str
) -> None:
    if report.get("schema") != "calc-flow.m5-common-benchmark-run.v1":
        raise ValueError("common benchmark run schema is invalid")
    if report.get("source_commit") != commit or report.get("source_tree") != tree:
        raise ValueError(
            "common benchmark embedded source identity does not match runtime ref"
        )
    if report.get("harness_sha256") != harness_sha256:
        raise ValueError("common benchmark harness hash does not match frozen bytes")
    _full_sha256(report.get("workload_sha256"), "workload hash")


def _validate_common_samples(
    report: dict[str, object], validated: dict[str, object]
) -> None:
    samples = report.get("raw_samples_ns")
    if not isinstance(samples, list):
        raise ValueError("common benchmark raw samples are missing")
    recomputed = common_statistics(samples)
    if (
        validated["median_ns"] != recomputed["median_ns"]
        or validated["median_confidence_interval_ns"]
        != recomputed["median_confidence_interval_ns"]
    ):
        raise ValueError(
            "common benchmark reported statistics do not match recomputed raw data"
        )


def _validate_reported_executable(
    report: dict[str, object], label: str, target_root: Path | None
) -> Path:
    executable_value = report.get("executable")
    if not isinstance(executable_value, str):
        raise ValueError(f"{label} executable path is missing")
    executable = Path(executable_value)
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
    ):
        raise ValueError(f"{label} executable path is invalid")
    canonical = executable.resolve()
    if target_root is not None and not canonical.is_relative_to(target_root.resolve()):
        raise ValueError(f"{label} executable escapes its target root")
    executable_hash = _full_sha256(report.get("executable_sha256"), "executable hash")
    if sha256_file(canonical) != executable_hash:
        raise ValueError(f"{label} executable hash does not match")
    return canonical


def _full_git_oid(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 40:
        raise ValueError(f"{field} must be an exact Git object ID")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} must be a lowercase Git object ID")
    return value


def validate_reference_contract(
    baseline_commit: str,
    candidate_commit: str,
    merge_base: str,
) -> None:
    baseline_commit = _full_git_oid(baseline_commit, "baseline commit")
    candidate_commit = _full_git_oid(candidate_commit, "candidate commit")
    merge_base = _full_git_oid(merge_base, "merge base")
    if baseline_commit == candidate_commit:
        raise ValueError("baseline and candidate commits must be distinct")
    if merge_base != baseline_commit:
        raise ValueError("baseline commit must be the exact candidate ancestor")


def _one_value(runs: Sequence[dict[str, object]], field: str) -> object:
    values = {str(run.get(field)) for run in runs}
    if len(values) != 1:
        raise ValueError(f"common benchmark {field} differs between runs")
    return next(iter(values))


def validate_matrix_provenance(
    runs: Sequence[dict[str, object]],
    *,
    baseline_commit: str,
    baseline_tree: str,
    candidate_commit: str,
    candidate_tree: str,
) -> None:
    if len(runs) != 4 or tuple(run.get("label") for run in runs) != RUN_ORDER:
        raise ValueError(f"common benchmark run order must be {RUN_ORDER}")
    expected_refs = (
        (baseline_commit, baseline_tree),
        (candidate_commit, candidate_tree),
        (baseline_commit, baseline_tree),
        (candidate_commit, candidate_tree),
    )
    for run, expected in zip(runs, expected_refs, strict=True):
        _validate_matrix_run_ref(run, expected)
    _validate_distinct_run_fields(runs)
    _validate_shared_run_hashes(runs)
    _validate_reproducible_executable_hashes(runs)


def _validate_matrix_run_ref(run: dict[str, object], expected: tuple[str, str]) -> None:
    commit, tree = expected
    if run.get("source_commit") != commit or run.get("source_tree") != tree:
        raise ValueError("common benchmark run does not match its declared ref")
    if run.get("git_status_short") != "":
        raise ValueError("common benchmark worktree was not clean")


def _validate_distinct_run_fields(runs: Sequence[dict[str, object]]) -> None:
    for field, description in (
        ("target_dir", "fresh target"),
        ("evidence_root", "fresh evidence root"),
        ("executable", "distinct executable path"),
    ):
        values = [str(run.get(field)) for run in runs]
        if len(set(values)) != len(values):
            raise ValueError(f"every run requires a {description}")


def _validate_shared_run_hashes(runs: Sequence[dict[str, object]]) -> None:
    for field in (
        "harness_sha256",
        "workload_sha256",
        "harness_cargo_lock_sha256",
        "toolchain_sha256",
        "machine_sha256",
        "environment_sha256",
    ):
        _full_sha256(_one_value(runs, field), field)
    for field in (
        "source_cargo_lock_sha256",
        "dependency_graph_sha256",
        "build_environment_sha256",
    ):
        _validate_same_ref_hash(runs, field)


def _validate_same_ref_hash(runs: Sequence[dict[str, object]], field: str) -> None:
    baseline = {_full_sha256(runs[index].get(field), field) for index in (0, 2)}
    candidate = {_full_sha256(runs[index].get(field), field) for index in (1, 3)}
    if len(baseline) != 1 or len(candidate) != 1:
        raise ValueError(f"same-ref common benchmark {field} differs")


def _validate_reproducible_executable_hashes(
    runs: Sequence[dict[str, object]],
) -> None:
    baseline_executables = {
        _full_sha256(runs[index].get("executable_sha256"), "baseline executable hash")
        for index in (0, 2)
    }
    candidate_executables = {
        _full_sha256(runs[index].get("executable_sha256"), "candidate executable hash")
        for index in (1, 3)
    }
    if len(baseline_executables) != 1 or len(candidate_executables) != 1:
        raise ValueError("same-ref benchmark executables must be byte-identical")
    if baseline_executables & candidate_executables:
        raise ValueError("baseline and candidate require distinct executable hashes")


def _finite_positive(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _finite_nonnegative(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be finite and nonnegative")
    return result


def _validated_run(run: dict[str, object], expected_label: str) -> dict[str, object]:
    _validate_run_identity(run, expected_label)
    confidence_level = _finite_positive(run.get("confidence_level"), "confidence level")
    if confidence_level != 0.95:
        raise ValueError("common benchmark confidence level must be exactly 0.95")
    median = _finite_positive(run.get("median_ns"), "median")
    lower, upper = _validated_interval(
        run.get("median_confidence_interval_ns"), median, "common benchmark"
    )
    return {
        **run,
        "median_ns": median,
        "median_confidence_interval_ns": [lower, upper],
    }


def _validate_run_identity(run: dict[str, object], expected_label: str) -> None:
    if run.get("label") != expected_label:
        raise ValueError(f"common benchmark run order must be {RUN_ORDER}")
    if run.get("case") != COMMON_CASE:
        raise ValueError("common benchmark case is incorrect")
    if run.get("sample_count") != COMMON_SAMPLE_COUNT:
        raise ValueError(f"common benchmark requires {COMMON_SAMPLE_COUNT} samples")


def _validated_interval(
    interval: object, median: float, label: str
) -> tuple[float, float]:
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{label} confidence interval is invalid")
    lower = _finite_positive(interval[0], f"{label} lower confidence bound")
    upper = _finite_positive(interval[1], f"{label} upper confidence bound")
    if lower > median or median > upper:
        raise ValueError(f"{label} confidence interval is not ordered")
    return lower, upper


def _regression_percent(baseline: float, candidate: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _pairing(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_median = float(baseline["median_ns"])
    candidate_median = float(candidate["median_ns"])
    baseline_interval = baseline["median_confidence_interval_ns"]
    candidate_interval = candidate["median_confidence_interval_ns"]
    if not isinstance(baseline_interval, list) or not isinstance(
        candidate_interval, list
    ):
        raise ValueError("validated common benchmark confidence interval is invalid")
    lower = _regression_percent(
        float(baseline_interval[1]), float(candidate_interval[0])
    )
    upper = _regression_percent(
        float(baseline_interval[0]), float(candidate_interval[1])
    )
    point = _regression_percent(baseline_median, candidate_median)
    return {
        "baseline": baseline["label"],
        "candidate": candidate["label"],
        "regression_percent": point,
        "regression_confidence_interval_percent": [lower, upper],
        "crosses_threshold": lower <= REGRESSION_THRESHOLD_PERCENT < upper,
    }


def evaluate_common_case(
    runs: Sequence[dict[str, object]], *, host_stable: bool = True
) -> dict[str, object]:
    if len(runs) != len(RUN_ORDER):
        raise ValueError("common benchmark requires exactly B1, C1, B2, C2")
    validated = [
        _validated_run(run, label) for run, label in zip(runs, RUN_ORDER, strict=True)
    ]
    baseline_runs = [validated[0], validated[2]]
    candidate_runs = [validated[1], validated[3]]
    baseline_medians = [float(run["median_ns"]) for run in baseline_runs]
    candidate_medians = [float(run["median_ns"]) for run in candidate_runs]
    baseline_minimum = min(baseline_medians)
    candidate_minimum = min(candidate_medians)
    candidate_min_regression = _regression_percent(baseline_minimum, candidate_minimum)
    baseline_spread = _regression_percent(min(baseline_medians), max(baseline_medians))
    candidate_spread = _regression_percent(
        min(candidate_medians), max(candidate_medians)
    )
    same_ref_spread = max(baseline_spread, candidate_spread)
    pairings = [
        _pairing(validated[0], validated[1]),
        _pairing(validated[2], validated[3]),
    ]
    sustained = all(
        float(pairing["regression_percent"]) > REGRESSION_THRESHOLD_PERCENT
        for pairing in pairings
    )
    exceeds_noise = candidate_min_regression > 2.0 * same_ref_spread
    confidently_above = all(
        float(pairing["regression_confidence_interval_percent"][0])
        > REGRESSION_THRESHOLD_PERCENT
        for pairing in pairings
    )
    confidently_below = all(
        float(pairing["regression_confidence_interval_percent"][1])
        <= REGRESSION_THRESHOLD_PERCENT
        for pairing in pairings
    )
    decision = _common_decision(
        candidate_min_regression,
        exceeds_noise=exceeds_noise,
        sustained=sustained,
        confidently_above=confidently_above,
        confidently_below=confidently_below,
    )
    if not host_stable:
        decision = "inconclusive"
    return {
        "case": COMMON_CASE,
        "threshold_percent": REGRESSION_THRESHOLD_PERCENT,
        "baseline_min_median_ns": baseline_minimum,
        "candidate_min_median_ns": candidate_minimum,
        "candidate_min_regression_percent": candidate_min_regression,
        "baseline_same_ref_spread_percent": baseline_spread,
        "candidate_same_ref_spread_percent": candidate_spread,
        "maximum_same_ref_spread_percent": same_ref_spread,
        "exceeds_twice_baseline_spread": exceeds_noise,
        "sustained_in_both_pairings": sustained,
        "pairings": pairings,
        "host_stable": host_stable,
        "decision": decision,
    }


def _common_decision(
    candidate_min_regression: float,
    *,
    exceeds_noise: bool,
    sustained: bool,
    confidently_above: bool,
    confidently_below: bool,
) -> str:
    if (
        candidate_min_regression > REGRESSION_THRESHOLD_PERCENT
        and exceeds_noise
        and sustained
        and confidently_above
    ):
        return "regression"
    if candidate_min_regression <= REGRESSION_THRESHOLD_PERCENT and confidently_below:
        return "pass"
    return "inconclusive"


def evaluate_private_repeats(
    reports: Sequence[dict[str, object]], *, host_stable: bool
) -> dict[str, object]:
    if len(reports) != 2 or tuple(report.get("label") for report in reports) != (
        PRIVATE_RUN_ORDER
    ):
        raise ValueError(f"private benchmark run order must be {PRIVATE_RUN_ORDER}")
    executable_hashes = {_private_executable_hash(report) for report in reports}
    if len(executable_hashes) != 1:
        raise ValueError("same-ref private executables must be byte-identical")
    measurements = [_private_medians(report) for report in reports]
    cases = []
    for case in M5_ABSOLUTE_CASES:
        first = measurements[0][case]
        second = measurements[1][case]
        spread = _regression_percent(min(first, second), max(first, second))
        cases.append(
            {
                "case": case,
                "run_medians_ns": {"P1": first, "P2": second},
                "same_ref_spread_percent": spread,
                "decision": "absolute_only",
            }
        )
    maximum_spread = max(float(case["same_ref_spread_percent"]) for case in cases)
    stable = host_stable and maximum_spread <= MAX_PRIVATE_REPEAT_SPREAD_PERCENT
    return {
        "decision": "absolute_only",
        "evidence_quality": "stable" if stable else "inconclusive",
        "host_stable": host_stable,
        "maximum_same_ref_spread_percent": maximum_spread,
        "spread_limit_percent": MAX_PRIVATE_REPEAT_SPREAD_PERCENT,
        "executable_sha256": next(iter(executable_hashes)),
        "cases": cases,
    }


def _private_executable_hash(report: dict[str, object]) -> str:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("private repeat provenance is missing")
    return _full_sha256(
        provenance.get("executable_sha256"), "private repeat executable hash"
    )


def _private_medians(report: dict[str, object]) -> dict[str, float]:
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(
        M5_ABSOLUTE_CASES
    ):
        raise ValueError("private repeat measurements are incomplete")
    result = {}
    for measurement, case in zip(measurements, M5_ABSOLUTE_CASES, strict=True):
        if not isinstance(measurement, dict) or measurement.get("case") != case:
            raise ValueError("private repeat measurement order is invalid")
        result[case] = _finite_positive(
            measurement.get("median_ns"), f"private repeat {case} median"
        )
    return result


def evaluate_candidate_self_overhead(
    reports: Sequence[dict[str, object]],
    *,
    repeatability: dict[str, object],
    host_stable: bool,
) -> dict[str, object]:
    if len(reports) != 2 or tuple(report.get("label") for report in reports) != (
        PRIVATE_RUN_ORDER
    ):
        raise ValueError(f"private benchmark run order must be {PRIVATE_RUN_ORDER}")
    deltas = {}
    for report in reports:
        medians = _private_medians(report)
        disabled = medians["m5/private_full_path/checkpoint_disabled"]
        enabled = medians["m5/private_full_path/checkpoint_enabled"]
        deltas[str(report["label"])] = _regression_percent(disabled, enabled)
    noise = _finite_nonnegative(
        repeatability.get("maximum_same_ref_spread_percent"),
        "private repeatability spread",
    )
    stable = host_stable and repeatability.get("evidence_quality") == "stable"
    decision = _self_overhead_decision(tuple(deltas.values()), noise, stable=stable)
    return {
        "scope": "candidate_self_overhead_not_main_regression",
        "comparison": "candidate_checkpoint_enabled_vs_disabled_same_ref",
        "threshold_percent": REGRESSION_THRESHOLD_PERCENT,
        "run_overhead_percent": deltas,
        "maximum_same_ref_noise_percent": noise,
        "host_stable": host_stable,
        "decision": decision,
    }


def _self_overhead_decision(
    deltas: Sequence[float], noise: float, *, stable: bool
) -> str:
    if not stable:
        return "inconclusive"
    if all(delta <= REGRESSION_THRESHOLD_PERCENT for delta in deltas):
        return "pass"
    if (
        all(delta > REGRESSION_THRESHOLD_PERCENT for delta in deltas)
        and min(deltas) > 2.0 * noise
    ):
        return "regression"
    return "inconclusive"


def _trusted_system_executable(name: str) -> Path:
    if name not in {
        "cargo",
        "git",
        "lscpu",
        "powerprofilesctl",
        "rustc",
        "systemd-detect-virt",
    }:
        raise ValueError(f"executable is not allowlisted: {name}")
    discovered = shutil.which(name)
    if discovered is None:
        raise FileNotFoundError(f"required executable is unavailable: {name}")
    executable = Path(discovered)
    if not executable.is_absolute():
        raise ValueError(f"trusted executable path is not absolute: {name}")
    if not executable.is_file():
        raise ValueError(f"trusted executable is not a regular file: {name}")
    return executable


def _validated_cargo_executable(cargo: str) -> str:
    if cargo != "cargo":
        raise ValueError("cargo executable must be the fixed 'cargo' tool")
    return str(_trusted_system_executable(cargo))


def _run_cargo_command(
    command: TrustedCargoCommand,
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str:
    merged_environment = dict(os.environ)
    if environment is not None:
        merged_environment.update(environment)
    displayed_argv = command.argv()
    print(
        f"+ (cd {shlex.quote(str(cwd))} && {shlex.join(displayed_argv)})",
        flush=True,
    )
    # The literal argv selects Cargo; executable pins the validated absolute shim.
    result = subprocess.run(  # nosec B603, B607 - validated absolute Cargo executable
        ["cargo", *command.arguments],
        executable=str(command.executable),
        shell=False,
        cwd=cwd,
        env=merged_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    return _completed_stdout(result, displayed_argv)


def _completed_stdout(
    result: subprocess.CompletedProcess[str], displayed_argv: Sequence[str]
) -> str:
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        command = shlex.join(displayed_argv)
        raise RuntimeError(f"command failed with status {result.returncode}: {command}")
    return result.stdout.strip()


def _git(repository: Path, arguments: Sequence[str]) -> str:
    executable = _trusted_system_executable("git")
    displayed_argv = [str(executable), *arguments]
    print(
        f"+ (cd {shlex.quote(str(repository))} && {shlex.join(displayed_argv)})",
        flush=True,
    )
    # The literal argv selects Git; executable pins the validated absolute binary.
    result = subprocess.run(  # nosec B603, B607 - validated absolute Git executable
        ["git", *arguments],
        executable=str(executable),
        shell=False,
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return _completed_stdout(result, displayed_argv)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
    ).hexdigest()


def private_build_identity_hash(provenance: dict[str, object]) -> str:
    return _canonical_hash(
        [
            PRIVATE_BUILD_IDENTITY_SCHEMA,
            _full_git_oid(provenance.get("commit"), "private commit"),
            _full_git_oid(provenance.get("tree"), "private tree"),
            _full_sha256(
                provenance.get("executable_sha256"), "private executable hash"
            ),
            _full_sha256(provenance.get("toolchain_hash"), "private toolchain hash"),
            _full_sha256(provenance.get("harness_hash"), "private harness hash"),
            _full_sha256(provenance.get("config_hash"), "private config hash"),
            _full_sha256(
                provenance.get("environment_hash"), "private environment hash"
            ),
        ]
    )


def _optional_command_result(
    requested: tuple[str, ...],
) -> subprocess.CompletedProcess[str] | None:
    if requested == ("cargo", "-Vv"):
        return subprocess.run(  # nosec B603, B607 - validated absolute Cargo executable
            ["cargo", "-Vv"],
            executable=str(_trusted_system_executable("cargo")),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    if requested == ("lscpu",):
        return subprocess.run(  # nosec B603, B607 - validated absolute lscpu executable
            ["lscpu"],
            executable=str(_trusted_system_executable("lscpu")),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    if requested == ("powerprofilesctl", "get"):
        return subprocess.run(  # nosec B603, B607 - validated absolute powerprofilesctl executable
            ["powerprofilesctl", "get"],
            executable=str(_trusted_system_executable("powerprofilesctl")),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    if requested == ("rustc", "-vV"):
        return subprocess.run(  # nosec B603, B607 - validated absolute rustc executable
            ["rustc", "-vV"],
            executable=str(_trusted_system_executable("rustc")),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    if requested == ("systemd-detect-virt",):
        return subprocess.run(  # nosec B603, B607 - validated absolute systemd executable
            ["systemd-detect-virt"],
            executable=str(_trusted_system_executable("systemd-detect-virt")),
            shell=False,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    return None


def _optional_command(command: Sequence[str]) -> str:
    try:
        result = _optional_command_result(tuple(command))
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
    if result is None:
        return "unavailable: command is not allowlisted"
    output = (result.stdout or result.stderr).strip()
    return output if result.returncode == 0 else f"unavailable: {output}"


def _execution_context() -> dict[str, object]:
    governors = {}
    for path in sorted(
        Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
    ):
        try:
            governors[path.parent.parent.name] = path.read_text(
                encoding="utf-8"
            ).strip()
        except OSError as error:
            governors[path.parent.parent.name] = f"unavailable: {error}"
    cpuinfo = Path("/proc/cpuinfo")
    cpuinfo_sha256 = sha256_file(cpuinfo) if cpuinfo.is_file() else "unavailable"
    toolchain = {
        "rustc": _optional_command(["rustc", "-vV"]),
        "cargo": _optional_command(["cargo", "-Vv"]),
    }
    machine = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpuinfo_sha256": cpuinfo_sha256,
        "lscpu": _optional_command(["lscpu"]),
        "virtualization": _optional_command(["systemd-detect-virt"]),
        "power_profile": _optional_command(["powerprofilesctl", "get"]),
        "governors": governors,
        "wsl": (
            "microsoft" in platform.release().lower()
            or bool(os.environ.get("WSL_INTEROP"))
        ),
    }
    relevant_environment = {
        name: os.environ.get(name, "")
        for name in (
            "CARGO_BUILD_TARGET",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
            "CARGO_PROFILE_RELEASE_LTO",
            "CARGO_ENCODED_RUSTFLAGS",
            "LD_LIBRARY_PATH",
            "RUSTFLAGS",
        )
    }
    return {
        "toolchain": toolchain,
        "toolchain_sha256": _canonical_hash(toolchain),
        "machine": machine,
        "machine_sha256": _canonical_hash(machine),
        "environment": relevant_environment,
        "environment_sha256": _canonical_hash(relevant_environment),
    }


def host_stability(
    before: dict[str, object], after: dict[str, object]
) -> dict[str, object]:
    reasons = []
    for field in ("toolchain_sha256", "machine_sha256", "environment_sha256"):
        if before.get(field) != after.get(field):
            reasons.append(f"{field} changed during the benchmark matrix")
    machine = before.get("machine")
    if not isinstance(machine, dict):
        reasons.append("machine fingerprint is missing")
    else:
        if machine.get("wsl") is True:
            reasons.append("WSL host cannot yield a confident pass")
        virtualization = str(machine.get("virtualization", ""))
        if (
            virtualization
            and virtualization != "none"
            and not virtualization.startswith("unavailable:")
        ):
            reasons.append(f"virtualized host reported {virtualization!r}")
        governors = machine.get("governors")
        if not isinstance(governors, dict) or not governors:
            reasons.append("CPU governor evidence is unavailable")
        elif set(governors.values()) != {"performance"}:
            reasons.append("CPU governors are not uniformly performance")
        if machine.get("power_profile") != "performance":
            reasons.append("power profile is not performance")
    return {
        "stable": not reasons,
        "reasons": reasons,
        "before_hash": _canonical_hash(before),
        "after_hash": _canonical_hash(after),
    }


def validate_execution_context(context: dict[str, object]) -> None:
    for name in ("toolchain", "machine", "environment"):
        value = context.get(name)
        if not isinstance(value, dict):
            raise ValueError(f"benchmark {name} fingerprint is missing")
        expected = _full_sha256(
            context.get(f"{name}_sha256"), f"{name} fingerprint hash"
        )
        if _canonical_hash(value) != expected:
            raise ValueError(f"benchmark {name} fingerprint does not match")


def _normalized_dependency_graph(metadata: dict[str, object]) -> dict[str, object]:
    packages_value = metadata.get("packages")
    if not isinstance(packages_value, list):
        raise ValueError("cargo metadata packages are missing")
    packages = []
    for package_value in packages_value:
        if not isinstance(package_value, dict):
            raise ValueError("cargo metadata package is invalid")
        dependencies_value = package_value.get("dependencies")
        if not isinstance(dependencies_value, list):
            raise ValueError("cargo metadata dependency list is invalid")
        dependencies = []
        for dependency_value in dependencies_value:
            if not isinstance(dependency_value, dict):
                raise ValueError("cargo metadata dependency is invalid")
            dependencies.append(
                {
                    field: dependency_value.get(field)
                    for field in (
                        "default_features",
                        "features",
                        "kind",
                        "name",
                        "optional",
                        "rename",
                        "req",
                        "target",
                    )
                }
            )
        dependencies.sort(key=lambda dependency: json.dumps(dependency, sort_keys=True))
        packages.append(
            {
                "name": package_value.get("name"),
                "version": package_value.get("version"),
                "source": package_value.get("source"),
                "features": package_value.get("features"),
                "dependencies": dependencies,
            }
        )
    packages.sort(key=lambda package: (str(package["name"]), str(package["version"])))
    return {"packages": packages}


def _resolve_ref(
    repository: Path, role: str, reference: str, worktree: Path
) -> RefSnapshot:
    commit = _git(
        repository,
        ["rev-parse", "--verify", "--end-of-options", f"{reference}^{{commit}}"],
    )
    tree = _git(repository, ["rev-parse", f"{commit}^{{tree}}"])
    _full_git_oid(commit, f"{role} commit")
    _full_git_oid(tree, f"{role} tree")
    return RefSnapshot(role=role, commit=commit, tree=tree, worktree=worktree)


def _prepare_ref_worktrees(
    repository: Path,
    output_root: Path,
    baseline_reference: str,
    candidate_reference: str,
) -> tuple[RefSnapshot, RefSnapshot, str]:
    if _git(repository, ["status", "--porcelain=v1", "--untracked-files=all"]):
        raise ValueError("benchmark orchestration requires a clean source worktree")
    worktrees = output_root / "worktrees"
    baseline = _resolve_ref(
        repository, "baseline", baseline_reference, worktrees / "baseline"
    )
    candidate = _resolve_ref(
        repository, "candidate", candidate_reference, worktrees / "candidate"
    )
    current = _git(repository, ["rev-parse", "HEAD"])
    if candidate.commit != current:
        raise ValueError(
            "candidate ref must equal the clean checkout running the frozen harness"
        )
    merge_base = _git(repository, ["merge-base", baseline.commit, candidate.commit])
    validate_reference_contract(baseline.commit, candidate.commit, merge_base)
    output_root.mkdir(parents=True)
    worktrees.mkdir()
    _git(
        repository,
        ["worktree", "add", "--detach", str(baseline.worktree), baseline.commit],
    )
    _git(
        repository,
        [
            "worktree",
            "add",
            "--detach",
            str(candidate.worktree),
            candidate.commit,
        ],
    )
    for snapshot in (baseline, candidate):
        if _git(snapshot.worktree, ["rev-parse", "HEAD"]) != snapshot.commit:
            raise ValueError(f"{snapshot.role} worktree ref does not match")
        if _git(snapshot.worktree, ["rev-parse", "HEAD^{tree}"]) != snapshot.tree:
            raise ValueError(f"{snapshot.role} worktree tree does not match")
        if _git(
            snapshot.worktree, ["status", "--porcelain=v1", "--untracked-files=all"]
        ):
            raise ValueError(f"{snapshot.role} worktree is not clean")
    return baseline, candidate, merge_base


def _prepare_private_worktrees(
    repository: Path,
    plans: Sequence[PrivateRunPlan],
) -> None:
    for plan in plans:
        if plan.workspace.exists() or plan.evidence_root.exists():
            raise FileExistsError(f"private benchmark run root exists: {plan.label}")
        _git(
            repository,
            [
                "worktree",
                "add",
                "--detach",
                str(plan.workspace),
                plan.candidate.commit,
            ],
        )
        if _git(plan.workspace, ["rev-parse", "HEAD"]) != plan.candidate.commit:
            raise ValueError(f"private {plan.label} worktree ref does not match")
        if _git(plan.workspace, ["rev-parse", "HEAD^{tree}"]) != plan.candidate.tree:
            raise ValueError(f"private {plan.label} worktree tree does not match")
        if _git(
            plan.workspace,
            ["status", "--porcelain=v1", "--untracked-files=all"],
        ):
            raise ValueError(f"private {plan.label} worktree is not clean")
        plan.evidence_root.mkdir(parents=True)


def _run_common_matrix(
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    cargo: Path,
    context: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path]]:
    harness_sha256 = hash_harness_files(COMMON_HARNESS_ROOT)
    common_lock: bytes | None = None
    commands = []
    runs = []
    artifacts = []
    for plan in build_run_plan(output_root, baseline, candidate):
        if plan.harness_root.parent.exists() or plan.evidence_root.exists():
            raise FileExistsError(f"common benchmark run root exists: {plan.label}")
        plan.harness_root.parent.mkdir(parents=True)
        plan.evidence_root.mkdir(parents=True)
        materialize_common_harness(plan.harness_root)
        lock_path = plan.harness_root / "Cargo.lock"
        if common_lock is None:
            lock_command = TrustedCargoCommand(
                cargo, ("generate-lockfile", "--offline")
            )
            _run_cargo_command(lock_command, cwd=plan.harness_root)
            commands.append({"label": plan.label, "command": lock_command.argv()})
            common_lock = lock_path.read_bytes()
        else:
            lock_path.write_bytes(common_lock)
        harness_lock_sha256 = sha256_file(lock_path)
        metadata_command = TrustedCargoCommand(
            cargo, ("metadata", "--locked", "--format-version", "1")
        )
        metadata = json.loads(
            _run_cargo_command(metadata_command, cwd=plan.harness_root)
        )
        if not isinstance(metadata, dict):
            raise ValueError("cargo metadata output is invalid")
        metadata_path = plan.evidence_root / "cargo-metadata.json"
        write_hashed_json(metadata_path, metadata)
        dependency_graph_sha256 = _canonical_hash(
            _normalized_dependency_graph(metadata)
        )
        executable = _common_executable(plan).resolve()
        build_environment = common_build_environment(plan, harness_sha256)
        build_command = TrustedCargoCommand(cargo, ("build", "--release", "--locked"))
        _run_cargo_command(
            build_command,
            cwd=plan.harness_root,
            environment=build_environment,
        )
        if not executable.is_file():
            raise ValueError(f"common benchmark executable is missing: {executable}")
        executable_sha256 = sha256_file(executable)
        run_environment = common_run_environment(
            plan,
            build_environment,
            executable_sha256=executable_sha256,
        )
        run_command = TrustedCargoCommand(
            cargo,
            (
                "run",
                "--release",
                "--locked",
                "--quiet",
                "--bin",
                "calc-flow-m5-common-benchmark",
            ),
        )
        _run_cargo_command(
            run_command,
            cwd=plan.harness_root,
            environment=run_environment,
        )
        validated = validate_common_run_report(
            plan.report_path,
            label=plan.label,
            commit=plan.snapshot.commit,
            tree=plan.snapshot.tree,
            harness_sha256=harness_sha256,
        )
        source_lock = plan.snapshot.worktree / "Cargo.lock"
        run = {
            **validated,
            "artifact_path": str(plan.report_path.relative_to(output_root)),
            "target_dir": str(plan.target_dir.resolve()),
            "evidence_root": str(plan.evidence_root.resolve()),
            "source_cargo_lock_sha256": sha256_file(source_lock),
            "harness_cargo_lock_sha256": harness_lock_sha256,
            "dependency_graph_sha256": dependency_graph_sha256,
            "dependency_metadata_path": str(metadata_path.resolve()),
            "dependency_metadata_sha256": sha256_file(metadata_path),
            "toolchain_sha256": context["toolchain_sha256"],
            "machine_sha256": context["machine_sha256"],
            "environment_sha256": context["environment_sha256"],
            "build_environment_sha256": _canonical_hash(
                _normalized_build_environment(build_environment)
            ),
            "git_status_short": _git(
                plan.snapshot.worktree,
                ["status", "--porcelain=v1", "--untracked-files=all"],
            ),
        }
        run_evidence_path = plan.evidence_root / "run-evidence.json"
        write_hashed_json(
            run_evidence_path,
            {
                "schema": "calc-flow.m5-common-run-evidence.v1",
                "label": plan.label,
                "source_commit": plan.snapshot.commit,
                "source_tree": plan.snapshot.tree,
                "workspace": str(plan.snapshot.worktree.resolve()),
                "harness_root": str(plan.harness_root.resolve()),
                "target_dir": str(plan.target_dir.resolve()),
                "evidence_root": str(plan.evidence_root.resolve()),
                "report_path": str(plan.report_path.relative_to(output_root)),
                "report_sha256": validated["report_sha256"],
                "executable_sha256": executable_sha256,
                "source_cargo_lock_sha256": run["source_cargo_lock_sha256"],
                "harness_cargo_lock_sha256": harness_lock_sha256,
                "dependency_graph_sha256": dependency_graph_sha256,
                "build_environment": build_environment,
                "build_environment_sha256": run["build_environment_sha256"],
                "run_environment": run_environment,
                "commands": {
                    "metadata": metadata_command.argv(),
                    "build": build_command.argv(),
                    "run": run_command.argv(),
                },
                "git_status_short": run["git_status_short"],
            },
        )
        runs.append(run)
        artifacts.extend(
            [
                plan.report_path,
                Path(f"{plan.report_path}.sha256"),
                executable,
                lock_path,
                metadata_path,
                Path(f"{metadata_path}.sha256"),
                run_evidence_path,
                Path(f"{run_evidence_path}.sha256"),
                source_lock,
            ]
        )
        commands.extend(
            [
                {"label": plan.label, "command": metadata_command.argv()},
                {
                    "label": plan.label,
                    "command": build_command.argv(),
                    "environment": build_environment,
                },
                {
                    "label": plan.label,
                    "command": run_command.argv(),
                    "cwd": str(plan.harness_root),
                    "environment": run_environment,
                },
            ]
        )
    validate_matrix_provenance(
        runs,
        baseline_commit=baseline.commit,
        baseline_tree=baseline.tree,
        candidate_commit=candidate.commit,
        candidate_tree=candidate.tree,
    )
    return runs, commands, artifacts


def _normalized_build_environment(environment: dict[str, str]) -> dict[str, object]:
    encoded = environment.get("CARGO_ENCODED_RUSTFLAGS", "")
    remap_destinations = []
    for flag in encoded.split("\x1f"):
        if flag.startswith("--remap-path-prefix=") and "=" in flag:
            remap_destinations.append(flag.rsplit("=", maxsplit=1)[1])
        elif flag:
            remap_destinations.append(flag)
    return {
        key: value
        for key, value in environment.items()
        if key not in {"CARGO_TARGET_DIR", "CARGO_ENCODED_RUSTFLAGS"}
    } | {"cargo_encoded_rustflags": remap_destinations}


def _validate_absolute_artifact(
    artifact: object,
    *,
    name: str,
    target_root: Path,
) -> None:
    if not isinstance(artifact, dict):
        raise ValueError(f"private {name} artifact is missing")
    reference = artifact.get("path")
    expected = _full_sha256(artifact.get("sha256"), f"private {name} hash")
    if not isinstance(reference, str) or Path(reference).is_absolute():
        raise ValueError(f"private {name} path must be relative")
    path = target_root / reference
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"private {name} artifact is not a regular file")
    canonical_root = target_root.resolve()
    canonical = path.resolve()
    if not canonical.is_relative_to(canonical_root):
        raise ValueError(f"private {name} artifact escapes its target root")
    if sha256_file(canonical) != expected:
        raise ValueError(f"private {name} artifact hash does not match")


def validate_private_absolute_report(
    path: Path,
    *,
    target_root: Path,
    candidate: RefSnapshot,
) -> dict[str, object]:
    report = load_hashed_json(path)
    _validate_private_report_header(report, candidate)
    _validate_private_provenance(report, target_root, candidate)
    _validate_private_measurements(report, target_root)
    return {**report, "report_sha256": sha256_file(path)}


def _validate_private_report_header(
    report: dict[str, object], candidate: RefSnapshot
) -> None:
    if report.get("schema") != "calc-flow.m5-checkpoint-absolute-benchmark.v1":
        raise ValueError("private absolute report schema is invalid")
    if report.get("commit") != candidate.commit:
        raise ValueError("private absolute report commit is invalid")
    if (
        report.get("comparison") != "none"
        or report.get("overall_result") != "absolute_only"
    ):
        raise ValueError("private absolute report makes a comparative claim")
    if report.get("absolute_cases") != list(M5_ABSOLUTE_CASES):
        raise ValueError("private absolute report case set is invalid")


def _validate_private_provenance(
    report: dict[str, object], target_root: Path, candidate: RefSnapshot
) -> None:
    provenance = report.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("private absolute provenance is missing")
    if (
        provenance.get("commit") != candidate.commit
        or provenance.get("tree") != candidate.tree
    ):
        raise ValueError("private absolute provenance ref is invalid")
    if provenance.get("clean") is not True:
        raise ValueError("private absolute provenance is dirty")
    reported_build_identity = _full_sha256(
        provenance.get("build_identity_hash"), "private build identity"
    )
    if reported_build_identity != private_build_identity_hash(provenance):
        raise ValueError("private absolute build identity is invalid")
    _validate_reported_executable(provenance, "private absolute", target_root)


def _validate_private_measurements(
    report: dict[str, object], target_root: Path
) -> None:
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(
        M5_ABSOLUTE_CASES
    ):
        raise ValueError("private absolute measurements are incomplete")
    for measurement, case in zip(measurements, M5_ABSOLUTE_CASES, strict=True):
        _validate_private_measurement(measurement, case, target_root)


def _validate_private_measurement(
    measurement: object, case: str, target_root: Path
) -> None:
    if not isinstance(measurement, dict) or measurement.get("case") != case:
        raise ValueError("private absolute measurement order is invalid")
    _validate_private_measurement_claim(measurement)
    if measurement.get("sample_count") != PRIVATE_SAMPLE_COUNT:
        raise ValueError("private absolute sample count is invalid")
    if measurement.get("confidence_level") != 0.95:
        raise ValueError("private absolute confidence level is invalid")
    median = _finite_positive(measurement.get("median_ns"), "private median")
    _validated_interval(
        measurement.get("median_confidence_interval_ns"), median, "private absolute"
    )
    _validate_private_artifacts(measurement.get("artifacts"), target_root)


def _validate_private_measurement_claim(measurement: dict[str, object]) -> None:
    if (
        measurement.get("comparison") != "none"
        or measurement.get("decision") != "absolute_only"
    ):
        raise ValueError("private absolute measurement makes a comparative claim")
    if "baseline_median_ns" in measurement or "regression_percent" in measurement:
        raise ValueError("private absolute measurement contains a fake baseline")


def _validate_private_artifacts(artifacts: object, target_root: Path) -> None:
    if not isinstance(artifacts, dict) or set(artifacts) != {"sample", "estimates"}:
        raise ValueError("private absolute artifacts are incomplete")
    for name in ("sample", "estimates"):
        _validate_absolute_artifact(
            artifacts.get(name), name=name, target_root=target_root
        )


def private_run_environment(
    candidate: RefSnapshot,
    *,
    run_id: str,
    target_root: Path,
    workspace: Path | None = None,
) -> dict[str, str]:
    return {
        **_reproducible_build_environment(workspace or candidate.worktree),
        "CALC_FLOW_M5_CHECKPOINT_BENCHMARK": "1",
        "CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID": run_id,
        "CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT": candidate.commit,
        "CALC_FLOW_M5_PRIVATE_SOURCE_TREE": candidate.tree,
        "CARGO_TARGET_DIR": str(target_root),
    }


def private_benchmark_command(cargo: Path) -> TrustedCargoCommand:
    return TrustedCargoCommand(
        cargo,
        (
            "test",
            "--release",
            "--locked",
            "-p",
            "calc-flow",
            "--lib",
            PRIVATE_TEST,
            "--",
            "--ignored",
            "--exact",
            "--nocapture",
        ),
    )


def _run_private_absolute(
    output_root: Path,
    plan: PrivateRunPlan,
    cargo: Path,
    run_id: str,
    context: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], list[Path]]:
    if plan.target_dir.exists():
        raise FileExistsError(f"private benchmark target exists: {plan.target_dir}")
    command = private_benchmark_command(cargo)
    private_run_id = f"{run_id}-{plan.label}"
    environment = private_run_environment(
        plan.candidate,
        run_id=private_run_id,
        target_root=plan.target_dir,
        workspace=plan.workspace,
    )
    metadata_command = TrustedCargoCommand(
        cargo, ("metadata", "--locked", "--format-version", "1")
    )
    metadata = json.loads(_run_cargo_command(metadata_command, cwd=plan.workspace))
    if not isinstance(metadata, dict):
        raise ValueError("private cargo metadata output is invalid")
    metadata_path = plan.evidence_root / "cargo-metadata.json"
    write_hashed_json(metadata_path, metadata)
    dependency_graph_sha256 = _canonical_hash(_normalized_dependency_graph(metadata))
    _run_cargo_command(command, cwd=plan.workspace, environment=environment)
    report_path = (
        plan.target_dir
        / "m5-checkpoint-benchmark"
        / f"{plan.candidate.commit}-{private_run_id}.json"
    )
    candidate = RefSnapshot(
        role=plan.candidate.role,
        commit=plan.candidate.commit,
        tree=plan.candidate.tree,
        worktree=plan.workspace,
    )
    report = validate_private_absolute_report(
        report_path,
        target_root=plan.target_dir,
        candidate=candidate,
    )
    reference = report_path.relative_to(output_root)
    report["label"] = plan.label
    report["artifact_path"] = str(reference)
    report.update(
        {
            "workspace": str(plan.workspace.resolve()),
            "target_dir": str(plan.target_dir.resolve()),
            "evidence_root": str(plan.evidence_root.resolve()),
            "source_cargo_lock_sha256": sha256_file(plan.workspace / "Cargo.lock"),
            "dependency_graph_sha256": dependency_graph_sha256,
            "toolchain_sha256": context["toolchain_sha256"],
            "machine_sha256": context["machine_sha256"],
            "environment_sha256": context["environment_sha256"],
            "build_environment_sha256": _canonical_hash(
                _normalized_build_environment(
                    _reproducible_build_environment(plan.workspace)
                )
            ),
            "git_status_short": _git(
                plan.workspace,
                ["status", "--porcelain=v1", "--untracked-files=all"],
            ),
        }
    )
    run_evidence_path = plan.evidence_root / "run-evidence.json"
    run_evidence = {
        "schema": "calc-flow.m5-private-run-evidence.v1",
        "label": plan.label,
        "source_commit": plan.candidate.commit,
        "source_tree": plan.candidate.tree,
        "workspace": str(plan.workspace.resolve()),
        "target_dir": str(plan.target_dir.resolve()),
        "evidence_root": str(plan.evidence_root.resolve()),
        "report_path": str(reference),
        "report_sha256": report["report_sha256"],
        "source_cargo_lock_sha256": report["source_cargo_lock_sha256"],
        "dependency_graph_sha256": dependency_graph_sha256,
        "toolchain_sha256": report["toolchain_sha256"],
        "machine_sha256": report["machine_sha256"],
        "environment_sha256": report["environment_sha256"],
        "build_environment_sha256": report["build_environment_sha256"],
        "dependency_metadata_path": str(metadata_path.relative_to(output_root)),
        "dependency_metadata_sha256": sha256_file(metadata_path),
        "command": command.argv(),
        "environment": environment,
        "git_status_short": report["git_status_short"],
    }
    write_hashed_json(run_evidence_path, run_evidence)
    artifacts = _private_report_artifacts(report_path, report, plan.target_dir)
    artifacts.extend(
        [
            metadata_path,
            Path(f"{metadata_path}.sha256"),
            run_evidence_path,
            Path(f"{run_evidence_path}.sha256"),
            plan.workspace / "Cargo.lock",
        ]
    )
    return (
        report,
        {
            "label": plan.label,
            "command": command.argv(),
            "environment": environment,
            "cwd": str(plan.workspace),
        },
        artifacts,
    )


def _private_report_artifacts(
    report_path: Path,
    report: dict[str, object],
    target_root: Path,
) -> list[Path]:
    provenance = report.get("provenance")
    measurements = report.get("measurements")
    if not isinstance(provenance, dict) or not isinstance(measurements, list):
        raise ValueError("private report artifact inventory is incomplete")
    executable = Path(str(provenance.get("executable")))
    artifacts = [report_path, Path(f"{report_path}.sha256"), executable]
    for measurement in measurements:
        if not isinstance(measurement, dict):
            raise ValueError("private report measurement artifact is invalid")
        references = measurement.get("artifacts")
        if not isinstance(references, dict):
            raise ValueError("private report measurement artifacts are missing")
        for name in ("sample", "estimates"):
            artifact = references.get(name)
            if not isinstance(artifact, dict) or not isinstance(
                artifact.get("path"), str
            ):
                raise ValueError(f"private {name} artifact path is missing")
            artifacts.append(target_root / str(artifact["path"]))
    return artifacts


def _valid_run_id(run_id: str) -> bool:
    return 0 < len(run_id) <= 64 and all(
        character.isascii() and (character.isalnum() or character in "-_")
        for character in run_id
    )


def assemble_benchmark_report(
    *,
    run_id: str,
    output_root: Path,
    baseline: dict[str, object],
    candidate: dict[str, object],
    merge_base: str,
    common_runs: Sequence[dict[str, object]],
    shared_edge_result: dict[str, object],
    private_runs: Sequence[dict[str, object]],
    private_repeatability: dict[str, object],
    candidate_self_overhead: dict[str, object],
    host: dict[str, object],
    contexts: dict[str, object],
    source_contract: dict[str, object],
    commands: Sequence[dict[str, object]],
    artifact_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "calc-flow.m5-checkpoint-benchmark-evidence.v2",
        "scope": "benchmark_evidence_only_not_m5_acceptance",
        "run_id": run_id,
        "evidence_root": str(output_root.resolve()),
        "baseline": baseline,
        "candidate": candidate,
        "merge_base": merge_base,
        "common_run_order": list(RUN_ORDER),
        "private_run_order": list(PRIVATE_RUN_ORDER),
        "shared_edge_result": {
            **shared_edge_result,
            "scope": "shared_edge_result",
            "harness_source": (
                "frozen public edge-channel data roundtrip compiled on both refs"
            ),
            "runs": list(common_runs),
        },
        "m5_private_absolute": {
            "decision": "absolute_only",
            "runs": list(private_runs),
            "repeatability": private_repeatability,
            "candidate_self_overhead": candidate_self_overhead,
        },
        "host_stability": host,
        "execution_context": contexts,
        "source_contract": source_contract,
        "commands": list(commands),
        "artifact_manifest": artifact_manifest,
    }


def _run_private_matrix(
    repository: Path,
    output_root: Path,
    candidate: RefSnapshot,
    cargo: Path,
    run_id: str,
    context: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[Path]]:
    plans = build_private_run_plan(output_root, candidate)
    _prepare_private_worktrees(repository, plans)
    reports = []
    commands = []
    artifacts = []
    for plan in plans:
        report, command, run_artifacts = _run_private_absolute(
            output_root, plan, cargo, run_id, context
        )
        reports.append(report)
        commands.append(command)
        artifacts.extend(run_artifacts)
    return reports, commands, artifacts


def _benchmark_evidence_status(
    shared_edge: dict[str, object],
    private_repeatability: dict[str, object],
    self_overhead: dict[str, object],
    host: dict[str, object],
) -> str:
    decisions = (shared_edge.get("decision"), self_overhead.get("decision"))
    if "regression" in decisions:
        return "regression"
    if (
        host.get("stable") is not True
        or private_repeatability.get("evidence_quality") != "stable"
        or "inconclusive" in decisions
    ):
        return "inconclusive"
    return "pass"


def _unique_artifacts(artifacts: Sequence[Path]) -> list[Path]:
    result = []
    seen = set()
    for artifact in artifacts:
        canonical = artifact.resolve()
        if canonical not in seen:
            seen.add(canonical)
            result.append(artifact)
    return result


def _source_contract_artifacts(
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    contract: dict[str, object],
) -> tuple[Path, list[Path]]:
    path = output_root / "source-contract.json"
    write_hashed_json(path, contract)
    artifacts = [
        path,
        Path(f"{path}.sha256"),
        baseline.worktree / "Cargo.lock",
        *(candidate.worktree / relative for relative in SOURCE_CONTRACT_FILES),
    ]
    return path, artifacts


def _artifact_manifest_reference(
    output_root: Path, artifacts: Sequence[Path]
) -> dict[str, object]:
    path = write_artifact_manifest(output_root, _unique_artifacts(artifacts))
    validate_artifact_manifest(output_root, path)
    digest_path = Path(f"{path}.sha256")
    return {
        "path": str(path.relative_to(output_root)),
        "sha256": sha256_file(path),
        "digest_sha256": sha256_file(digest_path),
    }


def validate_benchmark_evidence(
    path: Path,
    *,
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
) -> dict[str, object]:
    report = load_hashed_json(path)
    if report.get("schema") != "calc-flow.m5-checkpoint-benchmark-evidence.v2":
        raise ValueError("benchmark evidence schema is invalid")
    if report.get("evidence_root") != str(output_root.resolve()):
        raise ValueError("benchmark evidence root does not match")
    if report.get("baseline") != {"commit": baseline.commit, "tree": baseline.tree}:
        raise ValueError("benchmark baseline identity does not match")
    if report.get("candidate") != {"commit": candidate.commit, "tree": candidate.tree}:
        raise ValueError("benchmark candidate identity does not match")
    source_contract = report.get("source_contract")
    if not isinstance(source_contract, dict):
        raise ValueError("benchmark source contract is missing")
    validate_source_contract(source_contract, baseline, candidate)
    recorded_source = load_hashed_json(output_root / "source-contract.json")
    if recorded_source != source_contract:
        raise ValueError("recorded benchmark source contract does not match")
    contexts = _validated_context_pair(report)
    host = host_stability(contexts["before"], contexts["after"])
    if report.get("host_stability") != host:
        raise ValueError("benchmark host stability decision does not match")
    _validate_recorded_decisions(
        report, output_root, baseline, candidate, stable=host["stable"] is True
    )
    _validate_outer_manifest_reference(report, output_root)
    return report


def _validated_context_pair(report: dict[str, object]) -> dict[str, dict[str, object]]:
    contexts = report.get("execution_context")
    if not isinstance(contexts, dict):
        raise ValueError("benchmark execution context is missing")
    result = {}
    for label in ("before", "after"):
        context = contexts.get(label)
        if not isinstance(context, dict):
            raise ValueError(f"benchmark {label} context is missing")
        validate_execution_context(context)
        result[label] = context
    return result


def _validate_recorded_decisions(
    report: dict[str, object],
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    *,
    stable: bool,
) -> None:
    shared = report.get("shared_edge_result")
    private = report.get("m5_private_absolute")
    if not isinstance(shared, dict) or not isinstance(private, dict):
        raise ValueError("benchmark scoped results are missing")
    common_runs = shared.get("runs")
    private_runs = private.get("runs")
    if not isinstance(common_runs, list) or not isinstance(private_runs, list):
        raise ValueError("benchmark run evidence is missing")
    _validate_run_roots(output_root, common_runs, private_runs)
    _validate_private_matrix_provenance(private_runs, candidate)
    validate_matrix_provenance(
        common_runs,
        baseline_commit=baseline.commit,
        baseline_tree=baseline.tree,
        candidate_commit=candidate.commit,
        candidate_tree=candidate.tree,
    )
    expected_shared = evaluate_common_case(common_runs, host_stable=stable)
    if any(shared.get(key) != value for key, value in expected_shared.items()):
        raise ValueError("shared edge decision does not match run evidence")
    repeatability = evaluate_private_repeats(private_runs, host_stable=stable)
    if private.get("repeatability") != repeatability:
        raise ValueError("private repeatability does not match run evidence")
    self_overhead = evaluate_candidate_self_overhead(
        private_runs, repeatability=repeatability, host_stable=stable
    )
    if private.get("candidate_self_overhead") != self_overhead:
        raise ValueError("candidate self-overhead does not match run evidence")


def _validate_run_roots(
    output_root: Path,
    common_runs: Sequence[dict[str, object]],
    private_runs: Sequence[dict[str, object]],
) -> None:
    for run in common_runs:
        validate_rooted_directory(
            output_root, Path(str(run.get("target_dir"))), "target"
        )
        validate_rooted_directory(
            output_root, Path(str(run.get("evidence_root"))), "evidence"
        )
    for run in private_runs:
        for field in ("workspace", "target_dir", "evidence_root"):
            validate_rooted_directory(
                output_root, Path(str(run.get(field))), f"private {field}"
            )


def _validate_private_matrix_provenance(
    runs: Sequence[dict[str, object]], candidate: RefSnapshot
) -> None:
    if len(runs) != 2 or tuple(run.get("label") for run in runs) != PRIVATE_RUN_ORDER:
        raise ValueError(f"private benchmark run order must be {PRIVATE_RUN_ORDER}")
    for run in runs:
        provenance = run.get("provenance")
        if (
            run.get("commit") != candidate.commit
            or not isinstance(provenance, dict)
            or provenance.get("tree") != candidate.tree
            or run.get("git_status_short") != ""
        ):
            raise ValueError(
                "private benchmark run does not match its clean candidate ref"
            )
    for field in (
        "source_cargo_lock_sha256",
        "dependency_graph_sha256",
        "toolchain_sha256",
        "machine_sha256",
        "environment_sha256",
        "build_environment_sha256",
    ):
        _full_sha256(_one_value(runs, field), f"private {field}")


def _validate_outer_manifest_reference(
    report: dict[str, object], output_root: Path
) -> None:
    reference = report.get("artifact_manifest")
    if (
        not isinstance(reference, dict)
        or reference.get("path") != "artifact-manifest.json"
    ):
        raise ValueError("benchmark artifact manifest reference is invalid")
    manifest = output_root / "artifact-manifest.json"
    digest = Path(f"{manifest}.sha256")
    if sha256_file(manifest) != _full_sha256(
        reference.get("sha256"), "artifact manifest hash"
    ):
        raise ValueError("benchmark artifact manifest hash does not match")
    if sha256_file(digest) != _full_sha256(
        reference.get("digest_sha256"), "artifact manifest digest hash"
    ):
        raise ValueError("benchmark artifact manifest digest does not match")
    validate_artifact_manifest(output_root, manifest)


def run_benchmark_evidence(
    repository: Path,
    *,
    baseline_reference: str,
    candidate_reference: str,
    run_id: str,
    cargo: str,
) -> tuple[Path, str]:
    repository = repository.resolve()
    cargo_executable = Path(_validated_cargo_executable(cargo))
    if not _valid_run_id(run_id):
        raise ValueError("run ID must contain only ASCII letters, digits, '-' or '_'")
    output_root = repository / "target" / "m5-checkpoint-benchmark-evidence" / run_id
    if output_root.exists():
        raise FileExistsError(f"immutable evidence root already exists: {output_root}")
    baseline, candidate, merge_base = _prepare_ref_worktrees(
        repository,
        output_root,
        baseline_reference,
        candidate_reference,
    )
    source_contract = build_source_contract(baseline, candidate)
    _, source_artifacts = _source_contract_artifacts(
        output_root, baseline, candidate, source_contract
    )
    context_before = _execution_context()
    common_runs, commands, common_artifacts = _run_common_matrix(
        output_root, baseline, candidate, cargo_executable, context_before
    )
    private_runs, private_commands, private_artifacts = _run_private_matrix(
        repository, output_root, candidate, cargo_executable, run_id, context_before
    )
    commands.extend(private_commands)
    context_after = _execution_context()
    host = host_stability(context_before, context_after)
    stable = host["stable"] is True
    shared_edge = evaluate_common_case(common_runs, host_stable=stable)
    private_repeatability = evaluate_private_repeats(private_runs, host_stable=stable)
    self_overhead = evaluate_candidate_self_overhead(
        private_runs, repeatability=private_repeatability, host_stable=stable
    )
    manifest = _artifact_manifest_reference(
        output_root, source_artifacts + common_artifacts + private_artifacts
    )
    report = assemble_benchmark_report(
        run_id=run_id,
        output_root=output_root,
        baseline={"commit": baseline.commit, "tree": baseline.tree},
        candidate={"commit": candidate.commit, "tree": candidate.tree},
        merge_base=merge_base,
        common_runs=common_runs,
        shared_edge_result=shared_edge,
        private_runs=private_runs,
        private_repeatability=private_repeatability,
        candidate_self_overhead=self_overhead,
        host=host,
        contexts={"before": context_before, "after": context_after},
        source_contract=source_contract,
        commands=commands,
        artifact_manifest=manifest,
    )
    report_path = output_root / "benchmark-evidence.json"
    write_hashed_json(report_path, report)
    validate_benchmark_evidence(
        report_path,
        output_root=output_root,
        baseline=baseline,
        candidate=candidate,
    )
    status = _benchmark_evidence_status(
        shared_edge, private_repeatability, self_overhead, host
    )
    return report_path, status


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce exact-ref M5 common and private benchmark evidence."
    )
    parser.add_argument("--baseline", required=True, help="exact ancestor commit")
    parser.add_argument(
        "--candidate", required=True, help="exact final candidate commit"
    )
    parser.add_argument(
        "--run-id", required=True, help="unique immutable evidence label"
    )
    parser.add_argument("--cargo", default="cargo", help=argparse.SUPPRESS)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    repository = Path(_git(Path.cwd(), ["rev-parse", "--show-toplevel"]))
    report, result = run_benchmark_evidence(
        repository,
        baseline_reference=options.baseline,
        candidate_reference=options.candidate,
        run_id=options.run_id,
        cargo=options.cargo,
    )
    print(report)
    return 0 if result == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
