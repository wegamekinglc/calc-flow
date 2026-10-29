"""New adjacent AB/BA CF/Arroyo observations; never pool historical samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import time
from pathlib import Path

from extension_transport import cf_sample, native_sample
from fixtures import (
    CASES,
    NATIVE,
    ROWS,
    digest,
    encode,
    fixture,
    oracle,
    save,
    validate,
)


def binary_digest(path):
    digestor = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digestor.update(chunk)
    return digestor.hexdigest()


def identity(binary):
    changed = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "b5ae227e48ac9bb8a81583a23485e2c6064cee52",
            "HEAD",
        ],
        text=True,
    ).splitlines()
    if any(not path.startswith(".codex/artifacts/") for path in changed):
        raise ValueError("CF product differs from the fixed baseline")
    subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "python",
            "crates",
            "Cargo.lock",
            "pyproject.toml",
        ],
        check=True,
    )
    return {
        "cf_source": "b5ae227e48ac9bb8a81583a23485e2c6064cee52",
        "cf_native": NATIVE,
        "arroyo_ref": "4ef15f6336a596651f10cf27c1ee29c29bf81577",
        "arroyo_binary_sha256": binary_digest(binary),
        "arroyo_binary_bytes": binary.stat().st_size,
        "arroyo_patch_sha256": digest(
            Path(__file__).with_name("arroyo-ipc-observed.patch")
        ),
        "harnesses": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "boundary": "ready native Batch-to-Batch, input/output bindings separate",
        "ipc_comparability": "CF pipes vs Arroyo files; no observer/IPC speed ranking",
        "worker_affinity": "4-7",
        "controller_affinity": "0-1",
        "tokio_threads": 4,
    }


def check_transform(case, table):
    validate(case, table)
    expected = oracle(case)
    if table.column_names != ["id", "y"] or any(
        not table[name].equals(expected[name]) for name in expected.column_names
    ):
        raise ValueError(
            "transform columns, values or input-relative output order differ"
        )


def one(context, case, arm, pair, payload):
    root, binary, daemon = context
    directory = root / f"{case.name}-{pair}-{arm}"
    directory.mkdir()
    started = time.perf_counter_ns()
    if arm == "cf":
        table, clocks = cf_sample(case, directory, payload, daemon)
    else:
        table, clocks = native_sample(case, directory, payload, (binary, daemon))
    check_transform(case, table)
    worker = json.loads((directory / "worker.json").read_text())
    core_ms = (
        (worker["core_ns"][1] - worker["core_ns"][0]) / 1e6
        if arm == "cf"
        else worker["core_seconds"] * 1000
    )
    if arm == "arroyo" and (
        worker["source_eof_before_output"] or worker["source_eof_at_consume"]
    ):
        raise ValueError("native output depended on source EOF")
    result = {
        "case": case.name,
        "arm": arm,
        "pair": pair,
        "core_ms": core_ms,
        "rows_per_second": ROWS * 1000 / core_ms,
        "output_rows": table.num_rows,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "output_sha256": digest(directory / "output.arrow"),
        "rss_peak_bytes": max(
            row["rss_bytes"]
            for row in json.loads((directory / "memory.json").read_text())
        ),
        "invocation_ns": [started, time.perf_counter_ns()],
        "observer": clocks,
        "correctness": "exact columns, every value, original order",
    }
    save(directory / "sample.json", result)
    print(
        json.dumps({key: result[key] for key in ("case", "arm", "pair", "core_ms")}),
        flush=True,
    )
    return result


def summarize(samples):
    result = []
    for case, arm in dict.fromkeys((s["case"], s["arm"]) for s in samples):
        rows = [s for s in samples if (s["case"], s["arm"]) == (case, arm)]
        times = [s["core_ms"] for s in rows]
        result.append(
            {
                "case": case,
                "arm": arm,
                "samples": len(rows),
                "core_ms": {
                    "median": statistics.median(times),
                    "min": min(times),
                    "max": max(times),
                },
                "rss_peak_mib": [s["rss_peak_bytes"] / 2**20 for s in rows],
            }
        )
    return result


def collect(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    budget_file = root.parent / "measurement-budget.json"
    budget = json.loads(budget_file.read_text())
    started = time.monotonic()
    samples = []
    try:
        save(root / "identity.json", identity(args.binary))
        daemon = json.loads(
            subprocess.check_output(
                ["multica", "daemon", "status", "--output", "json"], text=True
            )
        )["pid"]
        for case in CASES[:2]:
            _, batches = fixture(case)
            payload = encode(batches)
            (root / f"{case.name}-input.arrow").write_bytes(payload)
            for pair in range(1 if args.preflight else 5):
                for arm in ("cf", "arroyo") if pair % 2 == 0 else ("arroyo", "cf"):
                    if (
                        min(args.budget, budget["limit_s"] - budget["spent_s"])
                        - (time.monotonic() - started)
                        < 65
                    ):
                        raise TimeoutError(
                            "measurement budget lacks room for one bounded invocation"
                        )
                    samples.append(
                        one((root, args.binary, daemon), case, arm, pair, payload)
                    )
                    save(root / "samples.json", samples)
        save(root / "summary.json", summarize(samples))
    finally:
        elapsed = time.monotonic() - started
        budget["spent_s"] += elapsed
        budget["entries"].append(
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
                "samples": len(samples),
                "elapsed_s": elapsed,
                "preflight": args.preflight,
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=float, default=500)
    parser.add_argument("--preflight", action="store_true")
    arguments = parser.parse_args()
    arguments.binary = arguments.binary.resolve()
    collect(arguments)
