"""Focused same-path B-tree identity experiment for benchmark issue #316."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import shutil
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite import measure
from scripts.benchmark_suite.catalog import get_shard, shard_cases
from scripts.benchmark_suite.process import install
from scripts.benchmark_suite.release import load_release
from scripts.benchmark_suite.side_bias_diagnostic import (
    CASE_ID,
    ObservedWorker,
    _affinity,
    _write_json,
    bias_detected,
    build_raw_blocks,
)


def identity_plan() -> tuple[str, ...]:
    return ("original_1", "copy_1", "copy_2", "original_2")


def activate_tree(active: Path, parked_active: Path, parked_next: Path) -> None:
    """Switch tree identity by rename while the active absolute path stays fixed."""
    active.rename(parked_active)
    parked_next.rename(active)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"dev": stat.st_dev, "inode": stat.st_ino}


def _native_path(site: Path) -> Path:
    native = list(site.glob("calc_flow/_native*.so"))
    if len(native) != 1:
        raise ValueError(f"expected one native extension in {site}")
    return native[0]


def tree_fingerprint(site: Path) -> dict:
    files = {}
    for path in sorted(site.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"symlink in diagnostic site: {path}")
        if path.is_file():
            files[str(path.relative_to(site))] = _sha256(path)
    return {
        "files": files,
        "root": _identity(site),
        "native": _identity(_native_path(site)),
    }


def maps_match_native(lines: list[str], site: Path, inode: int) -> bool:
    native_path = (
        _native_path(site) if site.exists() else site / "calc_flow" / "_native.abi3.so"
    )
    native = str(native_path.resolve())
    return any(
        len(fields) >= 6 and fields[4] == str(inode) and fields[5] == native
        for fields in (line.split(maxsplit=5) for line in lines)
    )


def _thread_schedule(pid: int) -> list[dict]:
    observed = []
    try:
        tasks = sorted(Path(f"/proc/{pid}/task").iterdir())
    except OSError:
        return observed
    for task in tasks:
        try:
            fields = (task / "stat").read_text().rsplit(") ", 1)[1].split()
            observed.append(
                {
                    "tid": int(task.name),
                    "name": (task / "comm").read_text().strip(),
                    "last_cpu": int(fields[36]),
                    "schedstat": [
                        int(value) for value in (task / "schedstat").read_text().split()
                    ],
                }
            )
        except (IndexError, OSError, ValueError):
            continue
    return observed


def _cpu_mhz() -> dict[str, float]:
    frequency = {}
    for cpu in _affinity(0) or []:
        path = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq")
        try:
            frequency[str(cpu)] = int(path.read_text()) / 1000
        except (OSError, ValueError):
            continue
    if frequency:
        return frequency
    try:
        for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
            fields = dict(
                line.split(":", 1) for line in block.splitlines() if ":" in line
            )
            if "processor" in fields and "cpu MHz" in fields:
                frequency[fields["processor"].strip()] = float(fields["cpu MHz"])
    except (OSError, ValueError):
        pass
    return frequency


class TreeObservedWorker(ObservedWorker):
    @classmethod
    async def start(
        cls, site: Path, root: Path, journal: list[dict] | None = None
    ) -> TreeObservedWorker:
        observed = await super().start(site, root, journal)
        observed.site = site
        return observed

    async def request(self, **message: object) -> dict:
        operation = str(message["operation"])
        pid = self.worker.process.pid
        if operation in {"prepare", "sample"}:
            threads_before = _thread_schedule(pid)
            frequency_before = _cpu_mhz()
        response = await super().request(**message)
        if operation in {"prepare", "sample"}:
            event = self.journal[-1]
            event["threads_before"] = threads_before
            event["threads_after"] = _thread_schedule(pid)
            event["cpu_mhz_before"] = frequency_before
            event["cpu_mhz_after"] = _cpu_mhz()
        if operation == "prepare":
            lines = Path(f"/proc/{pid}/maps").read_text().splitlines()
            event["maps"] = lines
            event["maps_sha256"] = hashlib.sha256(
                ("\n".join(lines) + "\n").encode()
            ).hexdigest()
            event["native_map_matches_site"] = maps_match_native(
                lines, self.site, _identity(_native_path(self.site))["inode"]
            )
            if not event["native_map_matches_site"]:
                raise ValueError("worker mapped a different native tree identity")
        return response


async def run(baseline_path: Path, candidate_path: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    baseline = load_release(baseline_path)
    candidate = load_release(candidate_path)
    releases = {"baseline": baseline, "candidate": candidate}
    case = next(
        case
        for case in shard_cases(get_shard("engines-100000"))
        if case["id"] == CASE_ID
    )
    sites = {
        "baseline": await install(baseline, output / "install" / "A0"),
        "candidate": await install(candidate, output / "install" / "B1"),
    }
    active = sites["candidate"]
    evidence_id = "-".join(
        (
            os.environ.get("GITHUB_RUN_ID", "local"),
            os.environ.get("GITHUB_RUN_ATTEMPT", "1"),
            baseline["wheel_sha256"][:12],
            candidate["wheel_sha256"][:12],
        )
    )
    _write_json(
        output / "host.json",
        {
            "evidence_id": evidence_id,
            "github_sha": os.environ.get("GITHUB_SHA"),
            "runner_name": os.environ.get("RUNNER_NAME"),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "affinity": _affinity(0),
            "load_average": os.getloadavg(),
            "baseline_wheel_sha256": baseline["wheel_sha256"],
            "candidate_wheel_sha256": candidate["wheel_sha256"],
            "baseline_native_sha256": baseline["native_sha256"],
            "candidate_native_sha256": candidate["native_sha256"],
        },
    )

    async def measure_condition(label: str, tree: str) -> dict:
        events: list[dict] = []

        class BoundWorker(TreeObservedWorker):
            journal = events

        identity_before = {
            side: {
                "root": _identity(site),
                "native": _identity(_native_path(site)),
            }
            for side, site in sites.items()
        }
        load_before = os.getloadavg()
        with patch.object(measure, "Worker", BoundWorker):
            row = await measure.measure_case(
                case, "interleaved", sites, releases, output / label
            )
        record = {
            "evidence_id": f"{evidence_id}:{label}",
            "label": label,
            "tree": tree,
            "sites": {side: str(site.resolve()) for side, site in sites.items()},
            "identity_before": identity_before,
            "identity_after": {
                side: {
                    "root": _identity(site),
                    "native": _identity(_native_path(site)),
                }
                for side, site in sites.items()
            },
            "load_average_before": load_before,
            "load_average_after": os.getloadavg(),
            "events": events,
            "raw_blocks": build_raw_blocks(row, events),
            "measurement": row,
        }
        _write_json(output / f"{label}.json", record)
        print(
            json.dumps(
                {
                    "condition": label,
                    "status": row["status"],
                    "round_changes": row.get("result", {}).get("round_changes"),
                    "error": row.get("error"),
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
        row = await measure_condition(label, "original")
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

    parked = output / "parked"
    parked.mkdir()
    copied = parked / "copy"
    original = parked / "original"
    shutil.copytree(active, copied)
    original_fingerprint = tree_fingerprint(active)
    copy_fingerprint = tree_fingerprint(copied)
    if (
        original_fingerprint["files"] != copy_fingerprint["files"]
        or original_fingerprint["root"] == copy_fingerprint["root"]
        or original_fingerprint["native"] == copy_fingerprint["native"]
    ):
        raise ValueError("B trees are not byte-identical copies with distinct identity")
    _write_json(
        output / "trees-before.json",
        {"original": original_fingerprint, "copy": copy_fingerprint},
    )

    for label in identity_plan():
        if label == "copy_1":
            activate_tree(active, original, copied)
        elif label == "original_2":
            activate_tree(active, copied, original)
        await measure_condition(
            label, "copy" if label.startswith("copy") else "original"
        )

    final_original = tree_fingerprint(active)
    final_copy = tree_fingerprint(copied)
    _write_json(
        output / "trees-after.json",
        {"original": final_original, "copy": final_copy},
    )
    if (
        final_original["files"] != original_fingerprint["files"]
        or final_copy["files"] != copy_fingerprint["files"]
        or final_original["root"] != original_fingerprint["root"]
        or final_copy["root"] != copy_fingerprint["root"]
        or final_original["native"] != original_fingerprint["native"]
        or final_copy["native"] != copy_fingerprint["native"]
    ):
        raise ValueError("B tree contents or identities changed during the experiment")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.baseline, arguments.candidate, arguments.output))


if __name__ == "__main__":
    main()
