"""Collect the predeclared Arrow/IPC cases, with no JSON data adaptation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from fixtures import CASES, NATIVE, REF, ROWS, digest, encode, fixture, save, validate
from transport import observe, worker_command


def identity(native_sha):
    import calc_flow._native as native

    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != REF:
        raise ValueError("source differs from frozen main")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True)
    if digest(native.__file__) != native_sha:
        raise ValueError("incompatible native release")
    return {
        "source_ref": REF,
        "native_sha256": native_sha,
        "python": sys.version,
        "platform": platform.platform(),
        "scripts": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "controller_affinity": "0-1",
        "worker_affinity": "4-7",
        "tokio_threads": 4,
        "contract": "Arrow IPC bounded-input core/API and round-trip; descriptive",
    }


def environment():
    return {
        **os.environ,
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
    }


def sample(context, case, arm, pair, payload):
    root, daemon_pid, native_sha = context
    directory = root / f"{case.name}-{pair}-{arm}"
    directory.mkdir()
    command = worker_command(sys.executable, case, arm, directory, native_sha)
    save(directory / "command.json", command)
    table, wire, clocks = observe(
        command, (directory, environment(), daemon_pid), payload
    )
    (directory / "output.arrow").write_bytes(wire)
    started = time.perf_counter_ns()
    validate(case, table)
    validation_ns = time.perf_counter_ns() - started
    worker = json.loads((directory / "worker.json").read_text())
    core = worker["core_ns"]
    result = {
        "case": case.name,
        "arm": arm,
        "pair": pair,
        **clocks,
        "core_ns": core,
        "core_ms": (core[1] - core[0]) / 1e6,
        "round_trip_ms": (clocks["round_trip_ns"][1] - clocks["round_trip_ns"][0])
        / 1e6,
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "output_sha256": hashlib.sha256(wire).hexdigest(),
        "input_rows": ROWS + (512 if case.window else 0),
        "effective_rows": ROWS,
        "output_rows": table.num_rows,
        "validation_ns": validation_ns,
        "rss_peak_bytes": max(
            row["rss_bytes"]
            for row in json.loads((directory / "memory.json").read_text())
        ),
        "correctness": "every-row Arrow equality",
        "native_sha256": native_sha,
    }
    save(directory / "sample.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("case", "arm", "pair", "core_ms", "round_trip_ms")
            }
        ),
        flush=True,
    )
    return result


def collect(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    save(root / "identity.json", identity(args.native_sha))
    daemon_pid = json.loads(
        subprocess.check_output(
            ["multica", "daemon", "status", "--output", "json"], text=True
        )
    )["pid"]
    samples = []
    start = time.perf_counter()
    for case in CASES:
        prep = time.perf_counter_ns()
        _, batches = fixture(case)
        generated = time.perf_counter_ns()
        payload = encode(batches)
        save(
            root / f"{case.name}-fixture.json",
            {
                "generation_ns": generated - prep,
                "encode_ns": time.perf_counter_ns() - generated,
            },
        )
        (root / f"{case.name}-input.arrow").write_bytes(payload)
        for pair in range(args.pairs):
            order = case.arms if pair % 2 == 0 else case.arms[::-1]
            for arm in order:
                if time.perf_counter() - start > args.budget:
                    raise TimeoutError("predeclared collection budget exhausted")
                context = (root, daemon_pid, args.native_sha)
                samples.append(sample(context, case, arm, pair, payload))
                save(root / "samples.json", samples)
    save(
        root / "completion.json",
        {"samples": len(samples), "elapsed_s": time.perf_counter() - start},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--budget", type=float, default=500)
    parser.add_argument("--native-sha", default=NATIVE)
    collect(parser.parse_args())
