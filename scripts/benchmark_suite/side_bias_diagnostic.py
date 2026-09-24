"""Focused, read-only GitHub runner experiment for benchmark side bias."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import time
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite import measure
from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.process import Worker, install
from scripts.benchmark_suite.release import load_release

CASE_ID = "engines/100000/calc-flow-stream/group_by"


def condition_matrix() -> tuple[tuple[str, str, str, str], ...]:
    """Name each baseline/candidate physical slot and worker start side."""
    return (
        ("aa_same_site", "A0", "A0", "baseline"),
        ("aa_separate", "A0", "A1", "baseline"),
        ("aa_swapped", "A1", "A0", "baseline"),
        ("ab", "A0", "B1", "baseline"),
        ("ba", "B1", "A0", "baseline"),
        ("aa_reversed_start", "A0", "A1", "candidate"),
        ("bb_separate", "B0", "B1", "baseline"),
    )


def _affinity(pid: int) -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(pid))
    except (AttributeError, OSError):
        return None


class ObservedWorker:
    """Record physical worker and request order around the existing harness."""

    journal: list[dict] | None = None

    def __init__(self, worker: Worker, side: str, journal: list[dict]) -> None:
        self.worker = worker
        self.side = side
        self.journal = journal

    @classmethod
    async def start(
        cls, site: Path, root: Path, journal: list[dict] | None = None
    ) -> ObservedWorker:
        started_ns = time.monotonic_ns()
        worker = await Worker.start(site, root)
        pid = worker.process.pid
        events = journal if journal is not None else cls.journal
        if events is None:
            events = []
        events.append(
            {
                "event": "start",
                "side": root.name,
                "site": str(site.resolve()),
                "pid": pid,
                "affinity": _affinity(pid),
                "started_ns": started_ns,
                "finished_ns": time.monotonic_ns(),
            }
        )
        return cls(worker, root.name, events)

    async def request(self, **message: object) -> dict:
        started_ns = time.monotonic_ns()
        response = await self.worker.request(**message)
        operation = str(message["operation"])
        event = {
            "event": operation,
            "side": self.side,
            "started_ns": started_ns,
            "finished_ns": time.monotonic_ns(),
        }
        if operation == "prepare":
            event["warmup_seconds"] = response["warmup"]["seconds"]
        elif operation == "sample":
            event["sample_seconds"] = response["seconds"]
            event["start_row"] = response.get("start_row")
        self.journal.append(event)
        return response

    async def close(self) -> None:
        await self.worker.close()
        self.journal.append({"event": "close", "side": self.side})


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


async def run(baseline_path: Path, candidate_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_release(baseline_path)
    candidate = load_release(candidate_path)
    releases = {"A": baseline, "B": candidate}
    case = next(
        case
        for case in shard_cases(get_shard("engines-100000"))
        if case["id"] == CASE_ID
    )
    slots = {
        slot: await install(releases[slot[0]], output / "install" / slot)
        for slot in ("A0", "A1", "B0", "B1")
    }
    host = {
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "affinity": _affinity(0),
        "load_average": os.getloadavg(),
        "baseline_native_sha256": baseline["native_sha256"],
        "candidate_native_sha256": candidate["native_sha256"],
    }
    _write_json(output / "host.json", host)

    for label, baseline_slot, candidate_slot, first in condition_matrix():
        events: list[dict] = []

        class BoundWorker(ObservedWorker):
            journal = events

        sites = {
            side: slots[slot]
            for side, slot in (
                (("baseline", baseline_slot), ("candidate", candidate_slot))
                if first == "baseline"
                else (("candidate", candidate_slot), ("baseline", baseline_slot))
            )
        }
        trial_releases = {
            "baseline": releases[baseline_slot[0]],
            "candidate": releases[candidate_slot[0]],
        }
        with patch.object(measure, "Worker", BoundWorker):
            row = await measure.measure_case(
                case, "interleaved", sites, trial_releases, output / label
            )
        record = {
            "label": label,
            "baseline_slot": baseline_slot,
            "candidate_slot": candidate_slot,
            "first_worker": first,
            "sites": {side: str(site.resolve()) for side, site in sites.items()},
            "events": events,
            "measurement": row,
        }
        _write_json(output / f"{label}.json", record)
        result = row.get("result", {})
        print(
            json.dumps(
                {
                    "condition": label,
                    "status": row["status"],
                    "error": row.get("error"),
                    "round_changes": result.get("round_changes"),
                    "round_intervals": result.get("round_intervals"),
                    "verdict": result.get("verdict"),
                }
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.baseline, arguments.candidate, arguments.output))


if __name__ == "__main__":
    main()
