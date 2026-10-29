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
    return {
        name.removesuffix("_ns") + "_ms": (worker[name][1] - worker[name][0]) / 1e6
        for name in (
            "input_ipc_ns",
            "caller_table_views_ns",
            "declaration_ns",
            "start_ns",
            "warmup_ns",
            "execute_ns",
            "concat_ns",
            "output_ipc_ns",
            "close_ns",
        )
    }


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
