"""Render the primary and secondary clocks from validated native paired observations."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from scripts.toolkit import linear_percentile
from verify import verify


def summary(path):
    checked = verify(path)
    outcome = json.loads((path / "outcome.json").read_text())
    result = {**checked, "wall_seconds": outcome["wall_seconds"]}
    if checked["state"] == "complete":
        rows = [
            json.loads(line) for line in (path / "observations.jsonl").read_text().splitlines()
        ]
        tail = {}
        for side in ("base", "head"):
            ms = [
                r["result"]["watermark_seconds"] * 1000 for r in rows
                if r["pair"] and r["side"] == side
            ]
            tail[side] = {"p50": statistics.median(ms), "p95": linear_percentile(ms, 0.95)}
        result.update(admission=outcome["pooled_ms"], watermark=tail, rounds=outcome["rounds"])
    return result


if __name__ == "__main__":
    print(json.dumps({p: summary(Path(p)) for p in sys.argv[1:]}, indent=2))
