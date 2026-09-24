"""Focused, read-only GitHub runner experiment for benchmark side bias."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
import time
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite import measure
from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.process import Worker, install
from scripts.benchmark_suite.release import load_release

CASE_ID = "engines/100000/calc-flow-stream/group_by"


def condition_matrix() -> tuple[tuple[str, str, str, str, str | None], ...]:
    """Run exact-slot controls immediately after a biased A/B probe."""
    return (
        ("aa_exact", "A0", "B1", "baseline", "clone_a"),
        ("aa_exact_reversed", "B1", "A0", "baseline", None),
        ("ba_original", "B1", "A0", "baseline", "restore_b"),
        ("ab_crossed", "B1", "A0", "baseline", "swap"),
        ("ba_crossed", "A0", "B1", "baseline", None),
        ("ab_repeat", "A0", "B1", "candidate", "swap_back"),
        ("ba_repeat", "B1", "A0", "candidate", None),
    )


def bias_detected(row: dict) -> bool:
    """Select only stable, material probes for causal follow-up."""
    changes = row.get("result", {}).get("round_changes", [])
    return (
        row.get("status") == "ok"
        and len(changes) == 2
        and (
            all(change > 5 for change in changes)
            or all(change < -5 for change in changes)
        )
    )


def swap_site_contents(left: Path, right: Path, backups: Path) -> None:
    """Exchange owned install trees while retaining their absolute site paths."""
    backups.mkdir(parents=True)
    left.rename(backups / "left")
    right.rename(backups / "right")
    shutil.copytree(backups / "right", left)
    shutil.copytree(backups / "left", right)


def clone_site_contents(source: Path, destination: Path, backup: Path) -> None:
    """Temporarily put the same wheel in both owned physical slots."""
    backup.parent.mkdir(parents=True, exist_ok=True)
    destination.rename(backup)
    shutil.copytree(source, destination)


def restore_site_contents(destination: Path, backup: Path) -> None:
    shutil.rmtree(destination)
    backup.rename(destination)


def _record_sha256(site: Path) -> str:
    record = next(site.glob("*.dist-info/RECORD"))
    return hashlib.sha256(record.read_bytes()).hexdigest()


def _affinity(pid: int) -> list[int] | None:
    try:
        return sorted(os.sched_getaffinity(pid))
    except (AttributeError, OSError):
        return None


def _scheduler_snapshot(pid: int) -> dict:
    snapshot = {"pid": pid}
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        snapshot["last_cpu"] = int(fields[36])
        snapshot["schedstat"] = [
            int(value) for value in Path(f"/proc/{pid}/schedstat").read_text().split()
        ]
        snapshot["policy"] = os.sched_getscheduler(pid)
    except (IndexError, OSError, ValueError):
        snapshot["unavailable"] = True
    return snapshot


def build_raw_blocks(row: dict, events: list[dict]) -> list[list[dict]]:
    """Keep paired real seconds and actual request order for both rounds."""
    blocks = []
    for round_index, evidence in enumerate(row.get("evidence", [])):
        samples = evidence.get("samples", {})
        if "baseline" not in samples or "candidate" not in samples:
            continue
        sample_order = [
            event["side"]
            for event in events
            if event["event"] == "sample" and event["round"] == round_index
        ]
        blocks.append(
            [
                {
                    "index": index,
                    "baseline_seconds": baseline["seconds"],
                    "candidate_seconds": candidate["seconds"],
                    "baseline_start_row": baseline.get("start_row"),
                    "candidate_start_row": candidate.get("start_row"),
                    "order": sample_order[2 * index : 2 * index + 2],
                }
                for index, (baseline, candidate) in enumerate(
                    zip(samples["baseline"], samples["candidate"], strict=False)
                )
            ]
        )
    return blocks


class ObservedWorker:
    """Record physical worker and request order around the existing harness."""

    journal: list[dict] | None = None

    def __init__(self, worker: Worker, side: str, journal: list[dict]) -> None:
        self.worker = worker
        self.side = side
        self.journal = journal
        self.round_index: int | None = None

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
        observed = cls(worker, root.name, events)
        observed.round_index = int(root.parent.name.removeprefix("round-"))
        events[-1]["round"] = observed.round_index
        events[-1]["scheduler"] = _scheduler_snapshot(pid)
        return observed

    async def request(self, **message: object) -> dict:
        started_ns = time.monotonic_ns()
        pid = self.worker.process.pid
        scheduler_before = _scheduler_snapshot(pid)
        response = await self.worker.request(**message)
        operation = str(message["operation"])
        event = {
            "event": operation,
            "side": self.side,
            "round": self.round_index,
            "pid": pid,
            "scheduler_before": scheduler_before,
            "scheduler_after": _scheduler_snapshot(pid),
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
        self.journal.append(
            {"event": "close", "side": self.side, "round": self.round_index}
        )


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
        for slot in ("A0", "B1")
    }
    evidence_id = "-".join(
        (
            os.environ.get("GITHUB_RUN_ID", "local"),
            os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
            baseline["wheel_sha256"][:12],
            candidate["wheel_sha256"][:12],
        )
    )
    host = {
        "evidence_id": evidence_id,
        "github_sha": os.environ.get("GITHUB_SHA"),
        "runner_name": os.environ.get("RUNNER_NAME"),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "affinity": _affinity(0),
        "load_average": os.getloadavg(),
        "baseline_native_sha256": baseline["native_sha256"],
        "candidate_native_sha256": candidate["native_sha256"],
        "baseline_wheel_sha256": baseline["wheel_sha256"],
        "candidate_wheel_sha256": candidate["wheel_sha256"],
    }
    _write_json(output / "host.json", host)

    wheel_by_slot = {slot: slot[0] for slot in slots}

    async def measure_condition(
        label: str,
        baseline_slot: str,
        candidate_slot: str,
        first: str,
        action: str | None = None,
    ) -> dict:
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
            "baseline": releases[wheel_by_slot[baseline_slot]],
            "candidate": releases[wheel_by_slot[candidate_slot]],
        }
        with patch.object(measure, "Worker", BoundWorker):
            load_before = os.getloadavg()
            row = await measure.measure_case(
                case, "interleaved", sites, trial_releases, output / label
            )
            load_after = os.getloadavg()
        record = {
            "evidence_id": f"{evidence_id}:{label}",
            "label": label,
            "action": action,
            "baseline_slot": baseline_slot,
            "candidate_slot": candidate_slot,
            "baseline_wheel": wheel_by_slot[baseline_slot],
            "candidate_wheel": wheel_by_slot[candidate_slot],
            "first_worker": first,
            "sites": {side: str(site.resolve()) for side, site in sites.items()},
            "site_record_sha256": {
                side: _record_sha256(site) for side, site in sites.items()
            },
            "load_average_before": load_before,
            "load_average_after": load_after,
            "events": events,
            "raw_blocks": build_raw_blocks(row, events),
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
        if row["status"] != "ok":
            raise RuntimeError(
                f"diagnostic condition {label} failed: {row.get('error')}"
            )
        return row

    triggered = None
    for index in range(3):
        label = f"ab_probe_{index}"
        row = await measure_condition(label, "A0", "B1", "baseline")
        if bias_detected(row):
            triggered = label
            break
    _write_json(
        output / "trigger.json",
        {
            "evidence_id": evidence_id,
            "bias_detected": triggered is not None,
            "trigger_condition": triggered,
            "probe_limit": 3,
        },
    )
    if triggered is None:
        return

    original_b = output / "site-backups" / "B1-original"
    for label, baseline_slot, candidate_slot, first, action in condition_matrix():
        if action == "clone_a":
            clone_site_contents(slots["A0"], slots["B1"], original_b)
            wheel_by_slot = {"A0": "A", "B1": "A"}
        elif action == "restore_b":
            restore_site_contents(slots["B1"], original_b)
            wheel_by_slot = {"A0": "A", "B1": "B"}
        elif action in {"swap", "swap_back"}:
            swap_site_contents(
                slots["A0"], slots["B1"], output / "site-backups" / action
            )
            wheel_by_slot = {
                slot: ("B" if wheel == "A" else "A")
                for slot, wheel in wheel_by_slot.items()
            }
        await measure_condition(label, baseline_slot, candidate_slot, first, action)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.baseline, arguments.candidate, arguments.output))


if __name__ == "__main__":
    main()
