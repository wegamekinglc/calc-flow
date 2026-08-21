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


def main() -> int:
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
    options = parser.parse_args()

    try:
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
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid exact-ref provenance: {error}", file=sys.stderr)
        return 1

    baseline = load_baseline(options.baseline_dir)
    if not baseline:
        print(
            f"No baseline benchmarks found in {options.baseline_dir}", file=sys.stderr
        )
        return 1

    candidate_dir = options.candidate_dir
    if options.baseline_dir.resolve() == candidate_dir.resolve():
        print(
            "Baseline and candidate directories must be distinct paired artifacts",
            file=sys.stderr,
        )
        return 1
    candidate = load_baseline(candidate_dir)

    if not candidate:
        print(f"No candidate benchmarks found in {candidate_dir}", file=sys.stderr)
        return 1

    try:
        regressions = check_regression(baseline, candidate)
    except ValueError as error:
        print(f"Invalid paired benchmark evidence: {error}", file=sys.stderr)
        return 1
    try:
        criterion_baseline = load_criterion(
            options.criterion_dir, options.criterion_baseline
        )
        criterion_candidate = load_criterion(
            options.criterion_dir, options.criterion_candidate
        )
        if not criterion_baseline or not criterion_candidate:
            raise ValueError("paired Criterion baseline artifacts are empty")
        criterion_regressions = check_criterion_regression(
            criterion_baseline, criterion_candidate
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"Invalid paired Criterion evidence: {error}", file=sys.stderr)
        return 1
    all_regressions = [(f"python/{name}", delta) for name, delta in regressions] + [
        (f"rust/{name}", delta) for name, delta in criterion_regressions
    ]
    if all_regressions:
        gate_pct = f"{REGRESSION_THRESHOLD:.0%}"
        print(f"FAIL: {len(all_regressions)} benchmark(s) exceed the {gate_pct} gate:")
        for name, delta in sorted(all_regressions):
            print(f"  {name}: +{delta:.1%}")
        return 1

    matched = len(baseline) + len(criterion_baseline)
    gate_pct = f"{REGRESSION_THRESHOLD:.0%}"
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


if __name__ == "__main__":
    raise SystemExit(main())
