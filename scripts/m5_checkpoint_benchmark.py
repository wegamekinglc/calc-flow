from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

COMMON_CASE = "m5/common/stream_channel_data_roundtrip"
COMMON_SAMPLE_COUNT = 30
PRIVATE_SAMPLE_COUNT = 10
REGRESSION_THRESHOLD_PERCENT = 5.0
RUN_ORDER = ("B1", "C1", "B2", "C2")
PRIVATE_BUILD_IDENTITY_SCHEMA = "calc-flow.m5-private-build-identity.v1"
M5_ABSOLUTE_CASES = (
    "m5/private_path/barrier_cut_fan_out",
    "m5/private_path/two_input_alignment",
    "m5/private_path/dirty_window_state_stage",
    "m5/private_path/production_manifest_publication",
    "m5/private_path/cold_restore",
    "m5/private_path/transactional_sink_commit",
    "m5/private_full_path/periodic_checkpoint_restart",
)
PRIVATE_TEST = (
    "runtime::streaming::soak::private_m5_epoch_checkpoint_absolute_benchmark"
)
SCRIPT_ROOT = Path(__file__).resolve().parent
COMMON_HARNESS_ROOT = SCRIPT_ROOT / "m5_checkpoint_benchmark_harness"
COMMON_HARNESS_SOURCE = COMMON_HARNESS_ROOT / "src" / "main.rs"
COMMON_HARNESS_MANIFEST = COMMON_HARNESS_ROOT / "Cargo.toml"
COMMON_HARNESS_FILES = (Path("Cargo.toml"), Path("src/main.rs"))


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
    with path.open("xb") as report:
        report.write(bytes_)
        report.flush()
        os.fsync(report.fileno())
    with digest_path.open("x", encoding="utf-8") as digest:
        digest.write(f"{hashlib.sha256(bytes_).hexdigest()}\n")
        digest.flush()
        os.fsync(digest.fileno())
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def load_hashed_json(path: Path) -> dict[str, object]:
    digest_path = Path(f"{path}.sha256")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"benchmark report must be a regular file: {path}")
    if digest_path.is_symlink() or not digest_path.is_file():
        raise ValueError(f"benchmark digest must be a regular file: {digest_path}")
    bytes_ = path.read_bytes()
    expected = digest_path.read_text(encoding="utf-8").strip()
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
        or hashlib.sha256(bytes_).hexdigest() != expected
    ):
        raise ValueError(f"benchmark report hash does not match: {path}")
    value = json.loads(bytes_)
    if not isinstance(value, dict):
        raise ValueError(f"benchmark report must be an object: {path}")
    return value


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
    if report.get("schema") != "calc-flow.m5-common-benchmark-run.v1":
        raise ValueError("common benchmark run schema is invalid")
    if report.get("source_commit") != commit or report.get("source_tree") != tree:
        raise ValueError(
            "common benchmark embedded source identity does not match runtime ref"
        )
    if report.get("harness_sha256") != harness_sha256:
        raise ValueError("common benchmark harness hash does not match frozen bytes")
    _full_sha256(report.get("workload_sha256"), "workload hash")
    validated = _validated_run(report, label)
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
    executable_value = report.get("executable")
    if not isinstance(executable_value, str):
        raise ValueError("common benchmark executable path is missing")
    executable = Path(executable_value)
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
    ):
        raise ValueError("common benchmark executable path is invalid")
    executable_hash = _full_sha256(report.get("executable_sha256"), "executable hash")
    if sha256_file(executable) != executable_hash:
        raise ValueError("common benchmark executable hash does not match")
    return {
        **validated,
        "report_sha256": sha256_file(path),
    }


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
    for run, (commit, tree) in zip(runs, expected_refs, strict=True):
        if run.get("source_commit") != commit or run.get("source_tree") != tree:
            raise ValueError("common benchmark run does not match its declared ref")
        if run.get("git_status_short") != "":
            raise ValueError("common benchmark worktree was not clean")
    for field, description in [
        ("target_dir", "fresh target"),
        ("evidence_root", "fresh evidence root"),
        ("executable", "distinct executable path"),
    ]:
        values = [str(run.get(field)) for run in runs]
        if len(set(values)) != len(values):
            raise ValueError(f"every run requires a {description}")
    for field in [
        "harness_sha256",
        "workload_sha256",
        "source_cargo_lock_sha256",
        "harness_cargo_lock_sha256",
        "dependency_graph_sha256",
        "toolchain_sha256",
        "machine_sha256",
        "environment_sha256",
    ]:
        _full_sha256(_one_value(runs, field), field)
    baseline_executables = {
        _full_sha256(runs[index].get("executable_sha256"), "baseline executable hash")
        for index in (0, 2)
    }
    candidate_executables = {
        _full_sha256(runs[index].get("executable_sha256"), "candidate executable hash")
        for index in (1, 3)
    }
    if baseline_executables & candidate_executables:
        raise ValueError("baseline and candidate require distinct executable hashes")


def _finite_positive(value: object, field: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field} must be finite and positive")
    return result


def _validated_run(run: dict[str, object], expected_label: str) -> dict[str, object]:
    if run.get("label") != expected_label:
        raise ValueError(f"common benchmark run order must be {RUN_ORDER}")
    if run.get("case") != COMMON_CASE:
        raise ValueError("common benchmark case is incorrect")
    if run.get("sample_count") != COMMON_SAMPLE_COUNT:
        raise ValueError(f"common benchmark requires {COMMON_SAMPLE_COUNT} samples")
    confidence_level = _finite_positive(run.get("confidence_level"), "confidence level")
    if confidence_level != 0.95:
        raise ValueError("common benchmark confidence level must be exactly 0.95")
    median = _finite_positive(run.get("median_ns"), "median")
    interval = run.get("median_confidence_interval_ns")
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError("common benchmark confidence interval is invalid")
    lower = _finite_positive(interval[0], "lower confidence bound")
    upper = _finite_positive(interval[1], "upper confidence bound")
    if lower > median or median > upper:
        raise ValueError("common benchmark confidence interval is not ordered")
    return {
        **run,
        "median_ns": median,
        "median_confidence_interval_ns": [lower, upper],
    }


def _regression_percent(baseline: float, candidate: float) -> float:
    return (candidate / baseline - 1.0) * 100.0


def _pairing(
    baseline: dict[str, object], candidate: dict[str, object]
) -> dict[str, object]:
    baseline_median = float(baseline["median_ns"])
    candidate_median = float(candidate["median_ns"])
    baseline_interval = baseline["median_confidence_interval_ns"]
    candidate_interval = candidate["median_confidence_interval_ns"]
    assert isinstance(baseline_interval, list)
    assert isinstance(candidate_interval, list)
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


def evaluate_common_case(runs: Sequence[dict[str, object]]) -> dict[str, object]:
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
    pairings = [
        _pairing(validated[0], validated[1]),
        _pairing(validated[2], validated[3]),
    ]
    sustained = all(
        float(pairing["regression_percent"]) > REGRESSION_THRESHOLD_PERCENT
        for pairing in pairings
    )
    exceeds_noise = candidate_min_regression > 2.0 * baseline_spread
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
    if (
        candidate_min_regression > REGRESSION_THRESHOLD_PERCENT
        and exceeds_noise
        and sustained
        and confidently_above
    ):
        decision = "regression"
    elif candidate_min_regression <= REGRESSION_THRESHOLD_PERCENT and confidently_below:
        decision = "pass"
    else:
        decision = "inconclusive"
    return {
        "case": COMMON_CASE,
        "threshold_percent": REGRESSION_THRESHOLD_PERCENT,
        "baseline_min_median_ns": baseline_minimum,
        "candidate_min_median_ns": candidate_minimum,
        "candidate_min_regression_percent": candidate_min_regression,
        "baseline_same_ref_spread_percent": baseline_spread,
        "exceeds_twice_baseline_spread": exceeds_noise,
        "sustained_in_both_pairings": sustained,
        "pairings": pairings,
        "decision": decision,
    }


def _run_command(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str:
    merged_environment = dict(os.environ)
    if environment is not None:
        merged_environment.update(environment)
    print(f"+ (cd {shlex.quote(str(cwd))} && {shlex.join(command)})", flush=True)
    result = subprocess.run(
        command,
        cwd=cwd,
        env=merged_environment,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        raise RuntimeError(
            f"command failed with status {result.returncode}: {shlex.join(command)}"
        )
    return result.stdout.strip()


def _git(repository: Path, arguments: Sequence[str]) -> str:
    return _run_command(["git", *arguments], cwd=repository)


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


def _optional_command(command: Sequence[str]) -> str:
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"
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
    }
    relevant_environment = {
        name: os.environ.get(name, "")
        for name in (
            "CARGO_BUILD_TARGET",
            "CARGO_PROFILE_RELEASE_CODEGEN_UNITS",
            "CARGO_PROFILE_RELEASE_LTO",
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
    commit = _git(repository, ["rev-parse", f"{reference}^{{commit}}"])
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
    _run_command(
        ["git", "worktree", "add", "--detach", str(baseline.worktree), baseline.commit],
        cwd=repository,
    )
    _run_command(
        [
            "git",
            "worktree",
            "add",
            "--detach",
            str(candidate.worktree),
            candidate.commit,
        ],
        cwd=repository,
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


def _run_common_matrix(
    output_root: Path,
    baseline: RefSnapshot,
    candidate: RefSnapshot,
    cargo: str,
    context: dict[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    harness_sha256 = hash_harness_files(COMMON_HARNESS_ROOT)
    common_lock: bytes | None = None
    commands = []
    runs = []
    for plan in build_run_plan(output_root, baseline, candidate):
        if plan.harness_root.parent.exists() or plan.evidence_root.exists():
            raise FileExistsError(f"common benchmark run root exists: {plan.label}")
        plan.harness_root.parent.mkdir(parents=True)
        plan.evidence_root.mkdir(parents=True)
        materialize_common_harness(plan.harness_root)
        lock_path = plan.harness_root / "Cargo.lock"
        if common_lock is None:
            lock_command = [cargo, "generate-lockfile", "--offline"]
            _run_command(lock_command, cwd=plan.harness_root)
            commands.append({"label": plan.label, "command": lock_command})
            common_lock = lock_path.read_bytes()
        else:
            lock_path.write_bytes(common_lock)
        harness_lock_sha256 = sha256_file(lock_path)
        metadata_command = [cargo, "metadata", "--locked", "--format-version", "1"]
        metadata = json.loads(_run_command(metadata_command, cwd=plan.harness_root))
        if not isinstance(metadata, dict):
            raise ValueError("cargo metadata output is invalid")
        metadata_path = plan.evidence_root / "cargo-metadata.json"
        write_hashed_json(metadata_path, metadata)
        dependency_graph_sha256 = _canonical_hash(
            _normalized_dependency_graph(metadata)
        )
        build_environment = {
            "CALC_FLOW_M5_RUN_LABEL": plan.label,
            "CALC_FLOW_M5_SOURCE_COMMIT": plan.snapshot.commit,
            "CALC_FLOW_M5_SOURCE_TREE": plan.snapshot.tree,
            "CALC_FLOW_M5_HARNESS_SHA256": harness_sha256,
            "CARGO_TARGET_DIR": str(plan.target_dir),
        }
        build_command = [cargo, "build", "--release", "--locked"]
        _run_command(
            build_command,
            cwd=plan.harness_root,
            environment=build_environment,
        )
        executable = plan.target_dir / "release" / "calc-flow-m5-common-benchmark"
        if sys.platform == "win32":
            executable = executable.with_suffix(".exe")
        executable = executable.resolve()
        if not executable.is_file():
            raise ValueError(f"common benchmark executable is missing: {executable}")
        run_environment = {
            **build_environment,
            "CALC_FLOW_M5_COMMON_OUTPUT": str(plan.report_path.resolve()),
        }
        run_command = [str(executable)]
        _run_command(run_command, cwd=plan.cwd, environment=run_environment)
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
            "git_status_short": _git(
                plan.snapshot.worktree,
                ["status", "--porcelain=v1", "--untracked-files=all"],
            ),
        }
        runs.append(run)
        commands.extend(
            [
                {"label": plan.label, "command": metadata_command},
                {
                    "label": plan.label,
                    "command": build_command,
                    "environment": build_environment,
                },
                {"label": plan.label, "command": run_command, "cwd": str(plan.cwd)},
            ]
        )
    validate_matrix_provenance(
        runs,
        baseline_commit=baseline.commit,
        baseline_tree=baseline.tree,
        candidate_commit=candidate.commit,
        candidate_tree=candidate.tree,
    )
    return runs, commands


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
    executable_value = provenance.get("executable")
    if not isinstance(executable_value, str):
        raise ValueError("private absolute executable path is missing")
    executable = Path(executable_value)
    if (
        not executable.is_absolute()
        or executable.is_symlink()
        or not executable.is_file()
    ):
        raise ValueError("private absolute executable path is invalid")
    canonical_target = target_root.resolve()
    canonical_executable = executable.resolve()
    if not canonical_executable.is_relative_to(canonical_target):
        raise ValueError("private absolute executable escapes its target root")
    expected_executable_hash = _full_sha256(
        provenance.get("executable_sha256"), "private executable hash"
    )
    if sha256_file(canonical_executable) != expected_executable_hash:
        raise ValueError("private absolute executable hash does not match")
    measurements = report.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != len(
        M5_ABSOLUTE_CASES
    ):
        raise ValueError("private absolute measurements are incomplete")
    for measurement, case in zip(measurements, M5_ABSOLUTE_CASES, strict=True):
        if not isinstance(measurement, dict) or measurement.get("case") != case:
            raise ValueError("private absolute measurement order is invalid")
        if (
            measurement.get("comparison") != "none"
            or measurement.get("decision") != "absolute_only"
        ):
            raise ValueError("private absolute measurement makes a comparative claim")
        if "baseline_median_ns" in measurement or "regression_percent" in measurement:
            raise ValueError("private absolute measurement contains a fake baseline")
        if measurement.get("sample_count") != PRIVATE_SAMPLE_COUNT:
            raise ValueError("private absolute sample count is invalid")
        if measurement.get("confidence_level") != 0.95:
            raise ValueError("private absolute confidence level is invalid")
        median = _finite_positive(measurement.get("median_ns"), "private median")
        confidence = measurement.get("median_confidence_interval_ns")
        if not isinstance(confidence, list) or len(confidence) != 2:
            raise ValueError("private absolute confidence interval is invalid")
        lower = _finite_positive(confidence[0], "private lower confidence bound")
        upper = _finite_positive(confidence[1], "private upper confidence bound")
        if lower > median or median > upper:
            raise ValueError("private absolute confidence interval is not ordered")
        artifacts = measurement.get("artifacts")
        if not isinstance(artifacts, dict) or set(artifacts) != {"sample", "estimates"}:
            raise ValueError("private absolute artifacts are incomplete")
        for name in ("sample", "estimates"):
            _validate_absolute_artifact(
                artifacts.get(name), name=name, target_root=target_root
            )
    return {**report, "report_sha256": sha256_file(path)}


def private_run_environment(
    candidate: RefSnapshot,
    *,
    run_id: str,
    target_root: Path,
) -> dict[str, str]:
    return {
        "CALC_FLOW_M5_CHECKPOINT_BENCHMARK": "1",
        "CALC_FLOW_M5_CHECKPOINT_BENCHMARK_RUN_ID": run_id,
        "CALC_FLOW_M5_PRIVATE_SOURCE_COMMIT": candidate.commit,
        "CALC_FLOW_M5_PRIVATE_SOURCE_TREE": candidate.tree,
        "CARGO_TARGET_DIR": str(target_root),
    }


def _run_private_absolute(
    output_root: Path,
    candidate: RefSnapshot,
    cargo: str,
    run_id: str,
) -> tuple[dict[str, object], dict[str, object]]:
    target_root = candidate.worktree / "target" / "m5-private-absolute" / run_id
    if target_root.exists():
        raise FileExistsError(f"private benchmark target exists: {target_root}")
    command = [
        cargo,
        "test",
        "-p",
        "calc-flow",
        "--lib",
        PRIVATE_TEST,
        "--",
        "--ignored",
        "--exact",
        "--nocapture",
    ]
    environment = private_run_environment(
        candidate,
        run_id=run_id,
        target_root=target_root,
    )
    _run_command(command, cwd=candidate.worktree, environment=environment)
    report_path = (
        target_root / "m5-checkpoint-benchmark" / f"{candidate.commit}-{run_id}.json"
    )
    report = validate_private_absolute_report(
        report_path,
        target_root=target_root,
        candidate=candidate,
    )
    reference = report_path.relative_to(output_root, walk_up=True)
    report["artifact_path"] = str(reference)
    return report, {"command": command, "environment": environment}


def _valid_run_id(run_id: str) -> bool:
    return 0 < len(run_id) <= 64 and all(
        character.isascii() and (character.isalnum() or character in "-_")
        for character in run_id
    )


def run_benchmark_evidence(
    repository: Path,
    *,
    baseline_reference: str,
    candidate_reference: str,
    run_id: str,
    cargo: str,
) -> tuple[Path, str]:
    repository = repository.resolve()
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
    context = _execution_context()
    common_runs, commands = _run_common_matrix(
        output_root, baseline, candidate, cargo, context
    )
    common_decision = evaluate_common_case(common_runs)
    private_report, private_command = _run_private_absolute(
        output_root, candidate, cargo, run_id
    )
    commands.append(private_command)
    overall_result = common_decision["decision"]
    report = {
        "schema": "calc-flow.m5-checkpoint-benchmark-evidence.v1",
        "run_id": run_id,
        "baseline": {"commit": baseline.commit, "tree": baseline.tree},
        "candidate": {"commit": candidate.commit, "tree": candidate.tree},
        "merge_base": merge_base,
        "run_order": list(RUN_ORDER),
        "common_comparison": {
            "harness_source": (
                "frozen public edge-channel data roundtrip compiled on both refs"
            ),
            "runs": common_runs,
            "decision": common_decision,
        },
        "m5_private_absolute": private_report,
        "execution_context": context,
        "commands": commands,
        "overall_result": overall_result,
        "overall_pass": overall_result == "pass",
    }
    report_path = output_root / "benchmark-evidence.json"
    write_hashed_json(report_path, report)
    load_hashed_json(report_path)
    return report_path, str(overall_result)


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
    repository = Path(
        _run_command(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd())
    )
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
