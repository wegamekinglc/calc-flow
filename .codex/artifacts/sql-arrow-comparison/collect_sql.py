"""Fixed interleaved SQL/Arroyo observations within one 600-second ledger."""

from __future__ import annotations

import argparse
import datetime
import json
import signal
import subprocess
import time
from pathlib import Path

from common import (
    ARROYO,
    CASES,
    ROWS,
    check_rows,
    digest,
    encode,
    fixture,
    identity,
    inventory,
    milliseconds,
    native_sample,
    read,
    save,
)
from sql_transport import cf_sample


def expired(*_):
    raise TimeoutError("one engine invocation exceeded 45s; collecting owned cleanup")


def observation(context, entry, payload):
    root, binary, daemon = context
    case_name, group, arm = entry
    case = next(case for case in CASES[:2] if case.name == case_name)
    directory = root / f"{case_name}-{group}-{arm}"
    directory.mkdir()
    start, utc = time.perf_counter_ns(), datetime.datetime.now(datetime.UTC).isoformat()
    signal.setitimer(signal.ITIMER_REAL, 45)
    try:
        if arm == "arroyo":
            table, observer = native_sample(case, directory, payload, (binary, daemon))
        else:
            table, observer = cf_sample(case, arm, directory, (payload, daemon))
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    check_rows(case, table)
    worker = read(directory / "worker.json")
    engine_ms = (
        milliseconds(worker["api_ready_ns"])
        if arm != "arroyo"
        else (worker["core_seconds"] + worker["output_binding_seconds"]) * 1000
    )
    consumed_ms = milliseconds(observer["observer_ns"])
    result = {
        "case": case_name,
        "group": group,
        "arm": arm,
        "started_utc": utc,
        "invocation_ns": [start, time.perf_counter_ns()],
        "observer": observer,
        "consumable_ms": consumed_ms,
        "engine_ready_ms": engine_ms,
        "consumable_input_rows_s": ROWS * 1000 / consumed_ms,
        "engine_ready_input_rows_s": ROWS * 1000 / engine_ms,
        "output_rows": table.num_rows,
        "input_sha256": digest(directory / "input.arrow"),
        "output_sha256": digest(directory / "output.arrow"),
        "rss_peak_bytes": max(
            row["rss_bytes"] for row in read(directory / "memory.json")
        ),
    }
    save(directory / "sample.json", result)
    print(
        {key: result[key] for key in ("case", "group", "arm", "consumable_ms")},
        flush=True,
    )
    return result


def collect(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    budget_file = root.parent / "budget.json"
    budget = (
        read(budget_file)
        if budget_file.exists()
        else {"limit_s": 600, "spent_s": 0, "runs": []}
    )
    save(budget_file, budget)
    started, samples = time.monotonic(), []
    signal.signal(signal.SIGALRM, expired)
    try:
        provenance = identity(args.binary)
        if provenance["arroyo_binary_sha256"] != ARROYO:
            raise ValueError("Arroyo binary differs from reviewed #511 release")
        provenance["sql_harnesses"] = {
            path.name: digest(path) for path in Path(__file__).parent.glob("*.py")
        }
        save(root / "identity.json", provenance)
        daemon = subprocess.check_output(
            ["multica", "daemon", "status", "--output", "json"], text=True, timeout=20
        )
        context = root, args.binary, json.loads(daemon)["pid"]
        payloads = {case.name: encode(fixture(case)[1]) for case in CASES[:2]}
        for name, payload in payloads.items():
            (root / f"{name}-input.arrow").write_bytes(payload)
        for entry in inventory(args.preflight):
            if (
                budget["limit_s"] - budget["spent_s"] - (time.monotonic() - started)
                < 60
            ):
                raise TimeoutError(
                    "remaining budget lacks invocation and cleanup reserve"
                )
            samples.append(observation(context, entry, payloads[entry[0]]))
            save(root / "samples.json", samples)
    finally:
        elapsed = time.monotonic() - started
        budget["spent_s"] += elapsed
        budget["runs"].append(
            {
                "output": str(root),
                "elapsed_s": elapsed,
                "samples": len(samples),
                "preflight": args.preflight,
            }
        )
        save(budget_file, budget)
        save(
            root / "completion.json",
            {
                "elapsed_s": elapsed,
                "samples": len(samples),
                "preflight": args.preflight,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    arguments.binary = arguments.binary.resolve()
    collect(arguments)
