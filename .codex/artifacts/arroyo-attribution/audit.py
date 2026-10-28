"""Audit the frozen follow-up observations without loading calc-flow native."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from common import CASES, HISTORY, arms, check, legacy, prepared


def read(path):
    return json.loads(path.read_text())


def audit_live(directory, case, sample, frames, expected):
    arrivals = read(directory / "arrivals.json")
    check([row["row"] for row in arrivals], expected)
    sent = read(directory / "send_times.json")
    if (
        len(sent) != len(frames)
        or sent != sorted(sent)
        or not sample["output_before_eof"]
    ):
        raise ValueError("live ingress/EOF evidence mismatch")
    latencies = []
    for item in arrivals:
        row = item["row"]
        index = (
            row["id"] // case.batch_rows
            if not case.window
            else ((row["ws"] - 1767225600000000) // 4000000 * 2 + 1)
        )
        latencies.append((item["at"] - sent[index]) * 1000)
    ordered = sorted(latencies)
    values = {
        "p50": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
    }
    if values != sample["latency_ms"]:
        raise ValueError("latency quantiles do not match raw send/arrival records")
    return arrivals[-1]["at"] - sent[0]


def audit_sample(root, case, arm, pair, identity):
    directory = root / f"{case.name}-{pair}-{arm}"
    sample = read(directory / "sample.json")
    frames, _, expected = prepared(case)
    wire = frames if case.transport == "live" else frames[:-1]
    expected_hash = hashlib.sha256(b"".join(wire)).hexdigest()
    if sample["input_sha256"] != expected_hash or sample["profiled"]:
        raise ValueError("sample workload/clock identity mismatch")
    if (
        arm != "arroyo"
        and read(directory / "loaded-native.json")["sha256"]
        != identity["native_sha256"]
    ):
        raise ValueError("worker loaded the wrong native")
    if case.transport == "live":
        elapsed = audit_live(directory, case, sample, frames, expected)
    else:
        check(read(directory / "arrow-outputs.json"), expected)
        elapsed = read(directory / "worker-result.json")["elapsed_s"]
    if not math.isclose(elapsed, sample["elapsed_s"], rel_tol=1e-12):
        raise ValueError("timing boundary mismatch")
    if not math.isclose(sample["rows_per_second"], 32768 / elapsed, rel_tol=1e-12):
        raise ValueError("throughput mismatch")
    if read(directory / "command.json")["argv"][:3] != ["taskset", "-c", "4-7"]:
        raise ValueError("CPU resource mismatch")
    expected_exit = -2 if case.transport == "live" and arm != "arroyo" else 0
    if read(directory / "exit.json")["exit_code"] != expected_exit:
        raise ValueError("unexpected worker exit")
    memory = read(directory / "memory.json")
    if sample["rss_peak_bytes"] not in {row["rss_bytes"] for row in memory}:
        raise ValueError("pre-stop RSS peak is absent from process-tree samples")
    if sample["output_rows"] != len(expected):
        raise ValueError("output count does not match the oracle")
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root / "results"
    identity = read(root / "identity.json")
    samples = read(root / "samples.json")
    expected_order = []
    recorded_samples = []
    for case in CASES:
        for pair in range(5):
            order = arms(case) if pair % 2 == 0 else arms(case)[::-1]
            for arm in order:
                sample = audit_sample(root, case, arm, pair, identity)
                recorded_samples.append({"pair": pair, **sample})
                expected_order.append((case.name, pair, arm))
    if [(s["case"], s["pair"], s["arm"]) for s in samples] != expected_order:
        raise ValueError("alternating invocation order/inventory mismatch")
    if samples != recorded_samples:
        raise ValueError("summary inventory differs from the per-run samples")
    if legacy.digest(HISTORY) != identity["historical_harness_sha256"]:
        raise ValueError("historical fixture/oracle harness changed")
    for name, checksum in identity["harnesses"].items():
        if legacy.digest(Path(__file__).with_name(name)) != checksum:
            raise ValueError(f"measured harness changed: {name}")
    if len(list(root.glob("profile-*/profile.json"))) != 8:
        raise ValueError("diagnostic inventory mismatch")
    print(
        json.dumps(
            {
                "audit": "passed",
                "samples": len(samples),
                "profiles": 8,
                "outputs": sum(s["output_rows"] for s in samples),
            }
        )
    )


if __name__ == "__main__":
    main()
