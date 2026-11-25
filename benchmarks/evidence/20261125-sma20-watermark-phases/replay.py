"""Verify an archived SMA20 diagnostic using only the Python standard library."""

from __future__ import annotations

import hashlib
import io
import json
import math
import statistics
import sys
import tarfile
from pathlib import Path

STAGES = (
    "input_validation",
    "ordering_proof",
    "entity_resolution",
    "state_preparation",
    "numeric_update",
    "history_maintenance",
    "arrow_output",
    "budget_preparation",
    "send_wait",
    "other",
)
CALLBACKS = ("data", "watermark", "end")
ARMS = ("baseline", "candidate")
ROWS = 10_000_000


def require(test, message):
    if not test:
        raise ValueError(message)


def p95(values):
    values = sorted(values)
    pos = 0.95 * (len(values) - 1)
    whole = math.floor(pos)
    return values[whole] + (pos - whole) * (values[math.ceil(pos)] - values[whole])


def validate(job):
    require(
        job["state"] == "completed" and job["terminal_cause"] == "natural_end",
        "terminal",
    )
    require(job["task_errors"] == 0 and not job["metrics_overflowed"], "job metrics")
    (node,) = job["rolling_metrics"].values()
    require(not node["overflowed"], "rolling metrics overflowed")
    for name in CALLBACKS:
        cb = node[name]
        require(cb["succeeded"] == cb["started"], "pending callback")
        require(
            not any(cb[key] for key in ("failed", "interrupted", "cancelled")),
            "callback error",
        )
        require(
            cb["callback_duration_ns"] == sum(cb[f"{s}_duration_ns"] for s in STAGES),
            "stage partition",
        )
    return node


def reconstruct(read):
    protocol = read("protocol.json")
    qualified = read("observed/qualification.json")
    require(qualified["case"]["id"] == protocol["case"], "wrong case")
    resources = read("observed/resources.jsonl")
    require(
        resources
        and resources[0]["preflight"]
        and all(x["suitable"] for x in resources),
        "resource rejection",
    )
    observations, oracles = [], []
    work = {}
    clocks = {side: [] for side in ARMS}
    stages = {}
    results = read("observed/samples.json")
    require(results["status"] == "ok", "comparison failed")
    for r in range(2):
        round_seconds = {}
        for side in ARMS:
            registered = protocol["arms"][side]
            build = qualified["arms"][side]["build"]
            require(build["tree"] == registered["tree"], "tree")
            for key in ("native_sha256", "source_sha256", "wheel_sha256", "git_sha"):
                require(build["release"][key] == registered[key], key)
            location = f"observed/measurement/round-{r}/{side}/request-"
            hello = read(location + "0001.completed.json")["response"]
            require(
                hello["native_sha256"] == registered["native_sha256"]
                and hello["tokio_worker_threads"] == "1",
                "loaded module/threads",
            )
            values = []
            for req in range(2, 13):
                form = read(location + f"{req:04}.completed.json")
                warm = req == 2
                sample = form["response"]["warmup"] if warm else form["response"]
                require(
                    form["operation"] == ("prepare" if warm else "sample"),
                    "unexpected worker operation",
                )
                before = sample["before_eof"]
                require(
                    all(
                        before["checkpoint"][key] is None
                        for key in ("current_epoch", "last_completed_epoch")
                    ),
                    "nonterminal epoch",
                )
                node = validate(sample["terminal"])
                oracle = sample["correctness"]
                require(
                    oracle["passed"]
                    and oracle["rows"] == ROWS
                    and oracle["finite_rows"] == ROWS - 64 * 19,
                    "oracle",
                )
                oracles.append(oracle["max_abs_error"])
                if warm:
                    continue
                require(
                    sample == results["evidence"][r]["samples"][side][req - 3],
                    "raw requests differ from summary inputs",
                )
                values.append(sample["seconds"])
                for cb in CALLBACKS:
                    metrics = node[cb]
                    for name, v in metrics.items():
                        if name.endswith("_duration_ns"):
                            key = (r, cb, name.removesuffix("_duration_ns"))
                            stages.setdefault(key, {}).setdefault(side, []).append(
                                v / 1e6
                            )
                        elif isinstance(v, int):
                            work.setdefault((side, cb, name), set()).add(v)
            finish = read(location + "0013.completed.json")
            require(
                finish["response"]["state"] == "completed" and len(values) == 10,
                "incomplete round",
            )
            clocks[side].extend(values)
            round_seconds[side] = values
        changes = sorted(
            100 * (b / a - 1)
            for a, b in zip(
                round_seconds["baseline"], round_seconds["candidate"], strict=True
            )
        )
        interval = {
            "median": statistics.median(changes),
            "low": changes[1],
            "high": changes[-2],
        }
        for key in interval:
            require(
                math.isclose(
                    interval[key],
                    results["result"]["round_intervals"][r][key],
                    abs_tol=1e-12,
                ),
                "paired interval",
            )
        observations.append(interval)
    for side, key in (("baseline", "base_p50"), ("candidate", "head_p50")):
        require(
            math.isclose(statistics.median(clocks[side]), results["result"][key]),
            "median",
        )
    verdict = (
        "regression"
        if all(i["low"] > 5 + 1e-12 for i in observations)
        else "inconclusive"
        if any(i["high"] > 5 + 1e-12 for i in observations)
        else "improved"
        if all(i["high"] < -5 - 1e-12 for i in observations)
        else "no-confirmed-regression"
    )
    require(verdict == results["result"]["verdict"], "public verdict")
    stage_summary = {}
    archived_summary = read("observed/summary.json")
    for (r, cb, stage), arms in stages.items():
        first, second = (arms[side] for side in ARMS)
        delta = [b - a for a, b in zip(first, second, strict=True)]
        row = {
            "baseline_median_ms": statistics.median(first),
            "candidate_median_ms": statistics.median(second),
            "paired_delta_median_ms": statistics.median(delta),
            "paired_delta_mean_ms": statistics.mean(delta),
        }
        require(
            row == archived_summary["terminal_callbacks"][cb][stage][r], "stage summary"
        )
        stage_summary.setdefault(cb, {}).setdefault(stage, []).append(row)
    return {
        "verified_oracles": len(oracles),
        "max_abs_error": max(oracles),
        "public_ms": {
            s: {
                "p50": statistics.median(clocks[s]) * 1000,
                "p95": p95(clocks[s]) * 1000,
            }
            for s in ARMS
        },
        "rounds": observations,
        "verdict": verdict,
        "terminal": stage_summary,
        "work_counter_ranges": {
            s: {
                cb: {
                    name: [min(vals), max(vals)]
                    for (side, category, name), vals in work.items()
                    if side == s and category == cb
                }
                for cb in CALLBACKS
            }
            for s in ARMS
        },
        "resources": {
            "intervals": len(resources),
            "cpu": resources[0]["cpu"],
            "min_available_bytes": min(s["available_bytes"] for s in resources),
            "max_sibling_busy": max(s["sibling_busy"] for s in resources),
        },
    }


def unpack(path):
    with tarfile.open(path, "r:gz") as archive:
        return {
            m.name: archive.extractfile(m).read()
            for m in archive.getmembers()
            if m.isfile()
        }


if __name__ == "__main__":
    data = unpack(Path(sys.argv[1]))
    for name, digest in json.loads(data["hashes.json"]).items():
        require(hashlib.sha256(data[name]).hexdigest() == digest, name)

    def reader(name):
        if name.endswith(".jsonl"):
            return [json.loads(line) for line in io.BytesIO(data[name])]
        return json.loads(data[name])

    print(json.dumps(reconstruct(reader), indent=2, allow_nan=False))
