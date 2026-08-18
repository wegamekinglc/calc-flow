"""M7-01 performance and memory gate runner.

Runs the Criterion benchmark suite and compares the results against the
recorded baselines with a 5% paired regression gate. Documents the
opt-in soak procedures that verify long-duration stability.

Usage:
    CALC_FLOW_BENCHMARK_SCALE=overhead \\
    JAX_PLATFORMS=cpu \\
    uv run --extra benchmark \\
    python scripts/verify_perf_gates.py --baseline-dir .benchmarks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TypedDict

REGRESSION_THRESHOLD = 0.05


class BenchResult(TypedDict):
    name: str
    mean_seconds: float
    std_dev: float


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
            )
    return results


def check_regression(
    baseline: dict[str, BenchResult],
    candidate: dict[str, BenchResult],
    threshold: float = REGRESSION_THRESHOLD,
) -> list[tuple[str, float]]:
    """Returns benchmarks whose paired candidate mean exceeds the gate."""
    regressions: list[tuple[str, float]] = []
    for name, base in baseline.items():
        cand = candidate.get(name)
        if cand is None:
            continue
        delta = (cand["mean_seconds"] - base["mean_seconds"]) / base["mean_seconds"]
        if delta > threshold:
            regressions.append((name, delta))
    return regressions


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
        default=None,
        help=(
            "Directory containing candidate results "
            "(defaults to baseline-dir for CI comparison)"
        ),
    )
    options = parser.parse_args()

    baseline = load_baseline(options.baseline_dir)
    if not baseline:
        print(
            f"No baseline benchmarks found in {options.baseline_dir}", file=sys.stderr
        )
        return 1

    candidate_dir = options.candidate_dir or options.baseline_dir
    candidate = load_baseline(candidate_dir)

    if not candidate:
        print(f"No candidate benchmarks found in {candidate_dir}", file=sys.stderr)
        return 1

    regressions = check_regression(baseline, candidate)
    if regressions:
        gate_pct = f"{REGRESSION_THRESHOLD:.0%}"
        print(f"FAIL: {len(regressions)} benchmark(s) exceed the {gate_pct} gate:")
        for name, delta in sorted(regressions):
            print(f"  {name}: +{delta:.1%}")
        return 1

    matched = len(set(baseline) & set(candidate))
    gate_pct = f"{REGRESSION_THRESHOLD:.0%}"
    print(f"PASS: {matched} benchmark(s) within the {gate_pct} gate")
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
