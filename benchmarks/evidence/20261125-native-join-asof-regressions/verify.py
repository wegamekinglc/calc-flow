"""Recompute the registered result from raw observations and resource logs."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.benchmark_suite.report import _paired_verdict
from scripts.benchmark_suite.statistics import paired_round
from scripts.toolkit import linear_percentile


def read(path):
    return json.loads(path.read_text())


def records(path):
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


def verify(root):
    result = read(root / "outcome.json")
    identity = read(root / "identity.json")
    case = identity["case"]
    expected = {"join": 10_000, "asof": 100_000}[case]
    obs = records(root / "observations.jsonl")
    resources = records(root / "resources.jsonl")
    for pos, row in enumerate(obs):
        rnd, pos_in_round = divmod(pos, 22)
        pair, arm = divmod(pos_in_round, 2)
        order = ("base", "head") if pair == 0 or pair % 2 else ("head", "base")
        assert (row["round"], row["pair"], row["side"]) == (rnd + 1, pair, order[arm])
        data = row["result"]
        assert (
            data["validated_all_rows"] is True
            and data["output_rows"] == expected
            and data["seconds"] > 0
        )
        if case == "asof":
            assert data["retained_right_rows"] + data["evicted_right_rows"] == expected
    for side in ("base", "head"):
        receipt = read(root / (side + ".receipt.json"))
        assert (
            receipt["commit"] == identity[side]
            and receipt["harness_sha256"] == identity["adapter_sha256"]
        )
    if result["state"] != "complete":
        return {
            "state": "stopped",
            "case": case,
            "observations": len(obs),
            "error": result["error"],
        }
    assert (
        len(obs) == 44
        and resources[0]["preflight"]
        and all(r["suitable"] for r in resources)
    )

    def vals(s, r=None):
        return [
            x["result"]["seconds"] * 1000
            for x in obs
            if x["pair"] and x["side"] == s and (r is None or r == x["round"])
        ]

    intervals = [paired_round(vals("base", r), vals("head", r)) for r in (1, 2)]
    assert (
        intervals == result["rounds"]
        and _paired_verdict(intervals) == result["verdict"]
    )
    assert {
        s: {"p50": statistics.median(vals(s)), "p95": linear_percentile(vals(s), 0.95)}
        for s in ("base", "head")
    } == result["pooled_ms"]
    return {
        "case": case,
        "state": "complete",
        "observations": len(obs),
        "resource_intervals": len(resources),
        "verdict": result["verdict"],
    }


if __name__ == "__main__":
    if not __debug__:
        raise SystemExit("Run with Python assertions enabled for all oracle checks.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    print(json.dumps([verify(p) for p in parser.parse_args().directories], indent=2))
