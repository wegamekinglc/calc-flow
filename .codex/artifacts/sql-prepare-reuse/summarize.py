"""Recompute the fixed two-round comparison using maintained paired rules."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.benchmark_suite.report import _paired_verdict  # noqa: E402
from scripts.benchmark_suite.statistics import paired_round  # noqa: E402


def phases(worker):
    spans = worker["plan_nested_ns"]
    metrics = worker["last_datafusion_metrics"]
    values = {
        name.removesuffix("_ns") + "_ms": sum(end - start for start, end in entries)
        / 1e6
        for name, entries in spans.items()
    }
    values["api_ms"] = (worker["api_ready_ns"][1] - worker["api_ready_ns"][0]) / 1e6
    values["output_ipc_ms"] = (
        worker["output_ipc_ns"][1] - worker["output_ipc_ns"][0]
    ) / 1e6
    values["last_query_runtime_acquire_ms"] = metrics[0]["runtime_acquire_ns"] / 1e6
    for key in (
        "session_state_create_ns",
        "logical_planning_ns",
        "physical_planning_ns",
        "execution_ns",
    ):
        values["last_query_" + key.removesuffix("_ns") + "_ms"] = (
            sum(entry[key] for entry in metrics) / 1e6
        )
    values["last_query_physical_planning_count"] = sum(
        entry["physical_planning_count"] for entry in metrics
    )
    values["cold_prepare_and_execute_ms"] = (
        worker["warmup_ns"][1] - worker["warmup_ns"][0]
    ) / 1e6
    values["declaration_and_precompile_ms"] = (
        worker["declaration_and_precompile_ns"][1]
        - worker["declaration_and_precompile_ns"][0]
    ) / 1e6
    return values


def summarize(samples):
    output = []
    for case in ("transform-full", "transform-512"):
        rounds = []
        for round_number in range(2):
            times = {
                side: [
                    next(
                        sample["consumable_ms"]
                        for sample in samples
                        if (
                            sample["case"],
                            sample["round"],
                            sample["pair"],
                            sample["side"],
                        )
                        == (case, round_number, pair, side)
                    )
                    for pair in range(10)
                ]
                for side in ("base", "head")
            }
            rounds.append(paired_round(times["base"], times["head"]))
        arms = {}
        for side in ("base", "head"):
            selected = [s for s in samples if s["case"] == case and s["side"] == side]
            if len(selected) != 20:
                raise ValueError(
                    "the predeclared 20 observations per case/side are required"
                )
            observed = [phases(s["worker"]) for s in selected]
            arms[side] = {
                "consumable_p50_ms": statistics.median(
                    s["consumable_ms"] for s in selected
                ),
                "consumable_p95_ms": sorted(s["consumable_ms"] for s in selected)[18],
                "input_rows_s_median": statistics.median(
                    s["input_rows_s"] for s in selected
                ),
                "phases_median": {
                    key: statistics.median(s[key] for s in observed)
                    for key in observed[0]
                },
            }
        output.append(
            {
                "case": case,
                "arms": arms,
                "rounds": rounds,
                "verdict": _paired_verdict(rounds),
            }
        )
    return output


if __name__ == "__main__":
    samples = json.loads(Path(sys.argv[1]).read_text())
    if len(samples) != 80:
        raise ValueError("the predeclared 80 observations are required")
    print(json.dumps(summarize(samples), indent=2))
