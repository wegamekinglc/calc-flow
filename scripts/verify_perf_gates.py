"""M7-01 paired Python and Rust benchmark-result gate.

Validates saved pytest-benchmark baseline and candidate results with a 5%
paired regression gate and saved Criterion estimates with their reported 95%
confidence intervals. Benchmark invocation, exact-ref provenance, and soak
execution remain release-workflow responsibilities.

Usage:
    CALC_FLOW_BENCHMARK_SCALE=overhead \\
    JAX_PLATFORMS=cpu \\
    uv run --extra benchmark \\
    python scripts/verify_perf_gates.py \
      --baseline-dir .benchmarks/base \
      --candidate-dir .benchmarks/candidate \
      --criterion-dir target/criterion \
      --criterion-baseline exact-baseline \
      --criterion-candidate exact-candidate
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import TypedDict

REGRESSION_THRESHOLD = 0.05
CONFIDENCE_Z = 1.96
PROVENANCE_FILE = "provenance.json"
CRITERION_PROVENANCE_FILE = "criterion-provenance.json"
GIT_SHA = re.compile(r"[0-9a-f]{40}")
FINGERPRINT = re.compile(r"[0-9a-f]{64}")


class BenchResult(TypedDict):
    name: str
    mean_seconds: float
    std_dev: float
    rounds: int


class CriterionResult(TypedDict):
    name: str
    mean_seconds: float
    lower_seconds: float
    upper_seconds: float


class BenchmarkProvenance(TypedDict):
    role: str
    git_sha: str


class StreamLifecycleResult(TypedDict):
    checkpoint_bytes: int
    checkpoint_bytes_p50: int
    checkpoint_bytes_p95: int
    checkpoint_duration_p50_seconds: float
    checkpoint_duration_p95_seconds: float
    recovery_duration_p50_seconds: float
    recovery_duration_p95_seconds: float
    machine_fingerprint: str
    dependency_fingerprint: str
    workload_fingerprint: str


def load_provenance(path: Path, role: str) -> BenchmarkProvenance:
    """Loads exact-commit provenance for one benchmark artifact directory."""
    raw = json.loads(path.joinpath(PROVENANCE_FILE).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or set(raw) != {"role", "git_sha"}:
        raise ValueError(f"{role} provenance must contain only role and git_sha")
    if raw["role"] != role:
        raise ValueError(f"{role} provenance has role {raw['role']!r}")
    git_sha = raw["git_sha"]
    if not isinstance(git_sha, str) or GIT_SHA.fullmatch(git_sha) is None:
        raise ValueError(f"{role} provenance requires a lowercase full git SHA")
    return BenchmarkProvenance(role=role, git_sha=git_sha)


def load_criterion_provenance(
    path: Path,
    baseline: str,
    candidate: str,
) -> tuple[str, str]:
    """Loads the exact commits assigned to the two named Criterion baselines."""
    raw = json.loads(
        path.joinpath(CRITERION_PROVENANCE_FILE).read_text(encoding="utf-8")
    )
    expected = {baseline, candidate}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("Criterion provenance must name exactly both baselines")
    shas = tuple(raw[name] for name in (baseline, candidate))
    if any(not isinstance(sha, str) or GIT_SHA.fullmatch(sha) is None for sha in shas):
        raise ValueError("Criterion provenance requires lowercase full git SHAs")
    return shas


def load_baseline(path: Path) -> dict[str, BenchResult]:
    """Loads the baseline results from a pytest-benchmark savedir."""
    results: dict[str, BenchResult] = {}
    for json_file in sorted(path.glob("*.json")):
        data = json.loads(json_file.read_text(encoding="utf-8"))
        for bench in data.get("benchmarks", []):
            stats = bench.get("stats", {})
            results[bench["name"]] = BenchResult(
                name=bench["name"],
                mean_seconds=stats.get("mean", 0.0),
                std_dev=stats.get("stddev", 0.0),
                rounds=stats.get("rounds", 0),
            )
    return results


def _stream_metadata(benchmark: object) -> dict[str, object] | None:
    if not isinstance(benchmark, dict):
        return None
    extra = benchmark.get("extra_info")
    if not isinstance(extra, dict):
        return None
    if extra.get("scenario") != "symbolic_stream_window_checkpoint":
        return None
    return extra


def _stream_metadata_from_file(path: Path) -> list[dict[str, object]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return []
    benchmarks = document.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        return []
    matching = []
    for benchmark in benchmarks:
        extra = _stream_metadata(benchmark)
        if extra is not None:
            matching.append(extra)
    return matching


def _stream_lifecycle_metadata(path: Path) -> dict[str, object]:
    matching = []
    for json_file in sorted(path.glob("*.json")):
        matching.extend(_stream_metadata_from_file(json_file))
    if len(matching) != 1:
        raise ValueError("paired evidence requires one isolated stream lifecycle case")
    return matching[0]


def _stream_byte_values(extra: dict[str, object]) -> dict[str, int]:
    byte_values: dict[str, int] = {}
    for field in ("checkpoint_bytes", "checkpoint_bytes_p50", "checkpoint_bytes_p95"):
        value = extra.get(field)
        if type(value) is not int or value <= 0:
            raise ValueError(f"stream lifecycle {field} must be positive")
        byte_values[field] = value
    if (
        byte_values["checkpoint_bytes"] != byte_values["checkpoint_bytes_p50"]
        or byte_values["checkpoint_bytes_p95"] < byte_values["checkpoint_bytes_p50"]
    ):
        raise ValueError("stream lifecycle checkpoint byte quantiles are inconsistent")
    return byte_values


def _positive_duration(extra: dict[str, object], field: str) -> float:
    value = extra.get(field)
    if isinstance(value, bool):
        raise ValueError(f"stream lifecycle {field} must be positive")
    if not isinstance(value, int | float):
        raise ValueError(f"stream lifecycle {field} must be positive")
    duration = float(value)
    if not math.isfinite(duration):
        raise ValueError(f"stream lifecycle {field} must be positive")
    if duration <= 0:
        raise ValueError(f"stream lifecycle {field} must be positive")
    return duration


def _duration_quantiles(
    extra: dict[str, object], phase: str
) -> tuple[str, float, str, float]:
    p50_field = f"{phase}_duration_p50_seconds"
    p95_field = f"{phase}_duration_p95_seconds"
    p50 = _positive_duration(extra, p50_field)
    p95 = _positive_duration(extra, p95_field)
    if p95 < p50:
        raise ValueError(f"stream lifecycle {p95_field} must be at least p50")
    return p50_field, p50, p95_field, p95


def _stream_duration_values(extra: dict[str, object]) -> dict[str, float]:
    checkpoint = _duration_quantiles(extra, "checkpoint")
    recovery = _duration_quantiles(extra, "recovery")
    return {
        checkpoint[0]: checkpoint[1],
        checkpoint[2]: checkpoint[3],
        recovery[0]: recovery[1],
        recovery[2]: recovery[3],
    }


def _stream_fingerprints(extra: dict[str, object]) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for field in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        value = extra.get(field)
        if not isinstance(value, str) or FINGERPRINT.fullmatch(value) is None:
            raise ValueError(f"stream lifecycle {field} must be a SHA-256 digest")
        fingerprints[field] = value
    return fingerprints


def load_stream_lifecycle(path: Path) -> StreamLifecycleResult:
    """Load the one isolated stream lifecycle diagnostic result."""
    extra = _stream_lifecycle_metadata(path)
    byte_values = _stream_byte_values(extra)
    durations = _stream_duration_values(extra)
    fingerprints = _stream_fingerprints(extra)
    return StreamLifecycleResult(**byte_values, **durations, **fingerprints)


def check_stream_lifecycle_regression(
    baseline: StreamLifecycleResult,
    candidate: StreamLifecycleResult,
    threshold: float = REGRESSION_THRESHOLD,
) -> list[tuple[str, float]]:
    """Return phase/size regressions supported by the recorded quantiles."""
    for field in (
        "machine_fingerprint",
        "dependency_fingerprint",
        "workload_fingerprint",
    ):
        if baseline[field] != candidate[field]:
            raise ValueError(f"stream lifecycle {field} does not match")
    regressions = []
    byte_delta = (
        candidate["checkpoint_bytes_p50"] - baseline["checkpoint_bytes_p50"]
    ) / baseline["checkpoint_bytes_p50"]
    if candidate["checkpoint_bytes_p50"] > baseline["checkpoint_bytes_p95"] * (
        1.0 + threshold
    ):
        regressions.append(("checkpoint_bytes", byte_delta))
    for phase, p50_field, p95_field in (
        (
            "checkpoint",
            "checkpoint_duration_p50_seconds",
            "checkpoint_duration_p95_seconds",
        ),
        (
            "recovery",
            "recovery_duration_p50_seconds",
            "recovery_duration_p95_seconds",
        ),
    ):
        baseline_p50 = baseline[p50_field]
        baseline_p95 = baseline[p95_field]
        candidate_p50 = candidate[p50_field]
        delta = (candidate_p50 - baseline_p50) / baseline_p50
        if candidate_p50 > baseline_p95 * (1.0 + threshold):
            regressions.append((f"{phase}_duration", delta))
    return regressions


def check_regression(
    baseline: dict[str, BenchResult],
    candidate: dict[str, BenchResult],
    threshold: float = REGRESSION_THRESHOLD,
) -> list[tuple[str, float]]:
    """Returns statistically supported regressions for complete pairs.

    A regression fails only when the candidate's 95% lower confidence bound is
    more than ``threshold`` above the baseline's 95% upper confidence bound.
    Missing pairs and unusable statistics fail closed.
    """
    missing = sorted(set(baseline) - set(candidate))
    if missing:
        raise ValueError(f"missing candidate benchmarks: {', '.join(missing)}")
    regressions: list[tuple[str, float]] = []
    for name, base in baseline.items():
        cand = candidate[name]
        _validate_result(base, role="baseline")
        _validate_result(cand, role="candidate")
        delta = (cand["mean_seconds"] - base["mean_seconds"]) / base["mean_seconds"]
        baseline_upper = _confidence_bound(base, direction=1)
        candidate_lower = _confidence_bound(cand, direction=-1)
        if candidate_lower > baseline_upper * (1.0 + threshold):
            regressions.append((name, delta))
    return regressions


def load_criterion(path: Path, baseline: str) -> dict[str, CriterionResult]:
    """Load one named Criterion baseline from a report tree."""
    results: dict[str, CriterionResult] = {}
    for estimates_file in sorted(path.glob(f"**/{baseline}/estimates.json")):
        relative = estimates_file.parent.parent.relative_to(path).as_posix()
        data = json.loads(estimates_file.read_text(encoding="utf-8"))
        mean = data.get("mean", {})
        interval = mean.get("confidence_interval", {})
        result = CriterionResult(
            name=relative,
            mean_seconds=float(mean.get("point_estimate", 0.0)) / 1_000_000_000,
            lower_seconds=float(interval.get("lower_bound", 0.0)) / 1_000_000_000,
            upper_seconds=float(interval.get("upper_bound", 0.0)) / 1_000_000_000,
        )
        _validate_criterion_result(result, role=baseline)
        if relative in results:
            raise ValueError(f"duplicate Criterion benchmark: {relative}")
        results[relative] = result
    return results


def check_criterion_regression(
    baseline: dict[str, CriterionResult],
    candidate: dict[str, CriterionResult],
    threshold: float = REGRESSION_THRESHOLD,
) -> list[tuple[str, float]]:
    """Return Criterion regressions supported by non-overlapping intervals."""
    missing = sorted(set(baseline) - set(candidate))
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"missing candidate Criterion benchmarks: {missing_names}")
    regressions: list[tuple[str, float]] = []
    for name, base in baseline.items():
        cand = candidate[name]
        _validate_criterion_result(base, role="baseline")
        _validate_criterion_result(cand, role="candidate")
        delta = (cand["mean_seconds"] - base["mean_seconds"]) / base["mean_seconds"]
        if cand["lower_seconds"] > base["upper_seconds"] * (1.0 + threshold):
            regressions.append((name, delta))
    return regressions


def _validate_result(result: BenchResult, *, role: str) -> None:
    mean = result["mean_seconds"]
    deviation = result["std_dev"]
    rounds = result["rounds"]
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(
            f"{role} benchmark {result['name']!r} requires a positive mean"
        )
    if not math.isfinite(deviation) or deviation < 0.0:
        raise ValueError(
            f"{role} benchmark {result['name']!r} requires a non-negative stddev"
        )
    if type(rounds) is not int or rounds < 2:
        raise ValueError(
            f"{role} benchmark {result['name']!r} requires at least two rounds"
        )


def _validate_criterion_result(result: CriterionResult, *, role: str) -> None:
    values = (
        result["lower_seconds"],
        result["mean_seconds"],
        result["upper_seconds"],
    )
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError(
            f"{role} Criterion benchmark {result['name']!r} "
            "requires positive finite estimates"
        )
    if values != tuple(sorted(values)):
        raise ValueError(
            f"{role} Criterion benchmark {result['name']!r} "
            "has an invalid confidence interval"
        )


def _confidence_bound(result: BenchResult, *, direction: int) -> float:
    standard_error = result["std_dev"] / math.sqrt(result["rounds"])
    return result["mean_seconds"] + direction * CONFIDENCE_Z * standard_error


def _parse_options() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        required=True,
        help="Directory containing baseline pytest-benchmark JSON files",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        required=True,
        help="Directory containing candidate results from the paired run",
    )
    parser.add_argument(
        "--criterion-dir",
        type=Path,
        required=True,
        help="Criterion report root containing both named baselines",
    )
    parser.add_argument("--criterion-baseline", required=True)
    parser.add_argument("--criterion-candidate", required=True)
    parser.add_argument(
        "--require-stream-lifecycle",
        action="store_true",
        help="Require and compare isolated checkpoint/recovery phase evidence",
    )
    return parser.parse_args()


def _validate_exact_ref_provenance(options: argparse.Namespace) -> None:
    baseline_provenance = load_provenance(options.baseline_dir, "baseline")
    candidate_provenance = load_provenance(options.candidate_dir, "candidate")
    criterion_shas = load_criterion_provenance(
        options.criterion_dir,
        options.criterion_baseline,
        options.criterion_candidate,
    )
    benchmark_shas = (
        baseline_provenance["git_sha"],
        candidate_provenance["git_sha"],
    )
    if benchmark_shas[0] == benchmark_shas[1]:
        raise ValueError("baseline and candidate git SHAs must be distinct")
    if criterion_shas != benchmark_shas:
        raise ValueError(
            "Criterion provenance does not match the Python benchmark commits"
        )


def _load_python_regressions(
    options: argparse.Namespace,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]], int]:
    baseline = load_baseline(options.baseline_dir)
    if not baseline:
        raise ValueError(f"no baseline benchmarks found in {options.baseline_dir}")
    if options.baseline_dir.resolve() == options.candidate_dir.resolve():
        raise ValueError(
            "baseline and candidate directories must be distinct paired artifacts"
        )
    candidate = load_baseline(options.candidate_dir)
    if not candidate:
        raise ValueError(f"no candidate benchmarks found in {options.candidate_dir}")
    regressions = check_regression(baseline, candidate)
    stream_regressions = (
        check_stream_lifecycle_regression(
            load_stream_lifecycle(options.baseline_dir),
            load_stream_lifecycle(options.candidate_dir),
        )
        if options.require_stream_lifecycle
        else []
    )
    return regressions, stream_regressions, len(baseline)


def _load_criterion_regressions(
    options: argparse.Namespace,
) -> tuple[list[tuple[str, float]], int]:
    criterion_baseline = load_criterion(
        options.criterion_dir, options.criterion_baseline
    )
    criterion_candidate = load_criterion(
        options.criterion_dir, options.criterion_candidate
    )
    if not criterion_baseline or not criterion_candidate:
        raise ValueError("paired Criterion baseline artifacts are empty")
    return (
        check_criterion_regression(criterion_baseline, criterion_candidate),
        len(criterion_baseline),
    )


def _report_gate_result(
    regressions: list[tuple[str, float]],
    matched: int,
) -> int:
    gate_pct = f"{REGRESSION_THRESHOLD:.0%}"
    if regressions:
        print(f"FAIL: {len(regressions)} benchmark(s) exceed the {gate_pct} gate:")
        for name, delta in sorted(regressions):
            print(f"  {name}: +{delta:.1%}")
        return 1
    print(
        f"PASS: {matched} paired Python/Criterion benchmark(s) "
        f"within the {gate_pct} gate"
    )
    print()
    print("Opt-in soak verification (run manually before release):")
    print("  1. Two-source backpressure soak (1200s):")
    print("     CALC_FLOW_STREAM_SOAK=1 cargo test -p calc-flow --lib \\")
    print("       runtime::streaming::soak::twenty_minute_two_source_slow_sink \\")
    print("       -- --ignored --exact --nocapture")
    print("  2. Checkpoint restart soak (1200s):")
    print("     CALC_FLOW_M5_CHECKPOINT_SOAK=1 cargo test -p calc-flow --lib \\")
    print("       runtime::streaming::soak::twenty_minute_epoch_checkpoint_restart \\")
    print("       -- --ignored --exact --nocapture")
    return 0


def _run_gate(options: argparse.Namespace) -> int:
    try:
        _validate_exact_ref_provenance(options)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid exact-ref provenance: {error}", file=sys.stderr)
        return 1
    try:
        python_regressions, stream_regressions, python_count = _load_python_regressions(
            options
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid paired benchmark evidence: {error}", file=sys.stderr)
        return 1
    try:
        criterion_regressions, criterion_count = _load_criterion_regressions(options)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid paired Criterion evidence: {error}", file=sys.stderr)
        return 1
    regressions = (
        [(f"python/{name}", delta) for name, delta in python_regressions]
        + [(f"stream/{name}", delta) for name, delta in stream_regressions]
        + [(f"rust/{name}", delta) for name, delta in criterion_regressions]
    )
    matched = python_count + criterion_count + int(options.require_stream_lifecycle)
    return _report_gate_result(regressions, matched)


def main() -> int:
    return _run_gate(_parse_options())


if __name__ == "__main__":
    raise SystemExit(main())
