"""Two fixed cases, two rounds of ten adjacent CF base/head AB/BA pairs."""

from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / ".codex/artifacts/sql-arrow-comparison"))
from common import (  # noqa: E402
    CASES,
    ROWS,
    check_rows,
    digest,
    encode,
    fixture,
    read,
    save,
)
from extension_transport import owned, wait_file  # noqa: E402


def expired(*_):
    raise TimeoutError("one observation exceeded the predeclared 30s limit")


def observe(context, case, entry):
    args, seals, payload, daemon = context
    round_number, pair, side = entry
    directory = args.output / f"{case.name}-r{round_number}-p{pair}-{side}"
    directory.mkdir()
    (directory / "input.arrow").write_bytes(payload)
    command = [
        "taskset",
        "-c",
        "4-7",
        seals[side]["python"],
        str(Path(__file__).with_name("stream_worker.py")),
        "--case",
        case.name,
        "--directory",
        str(directory),
        "--native-sha",
        seals[side]["native_sha256"],
    ]
    if args.diagnostic:
        command.append("--diagnostic")
    settings = {
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
        "DAL417_GATED": "1",
    }
    save(directory / "command.json", {"argv": command, "settings": settings})
    signal.setitimer(signal.ITIMER_REAL, 30)
    try:
        with owned(command, (directory, settings, daemon)) as process:
            wait_file(directory / "input.arrow.ready.json", process)
            start = time.perf_counter_ns()
            (directory / "input.arrow.start").touch()
            wait_file(directory / "output.arrow", process)
            output = pa.ipc.open_stream(directory / "output.arrow").read_all()
            finish = time.perf_counter_ns()
            process.wait(timeout=5)
            if process.returncode:
                raise RuntimeError("stream worker failed; inspect retained stderr.log")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
    check_rows(case, output)
    metadata = read(directory / "worker.json")
    elapsed = (finish - start) / 1e6
    sample = {
        "case": case.name,
        "round": round_number,
        "pair": pair,
        "side": side,
        "consumable_ms": elapsed,
        "input_rows_s": ROWS * 1000 / elapsed,
        "observer_ns": [start, finish],
        "worker": metadata,
        "input_sha256": digest(directory / "input.arrow"),
        "output_sha256": digest(directory / "output.arrow"),
    }
    save(directory / "sample.json", sample)
    print(case.name, round_number, pair, side, round(elapsed, 3), flush=True)
    return sample


def verify_product_source(reference):
    product = [
        "crates",
        "vendor",
        ".cargo",
        "python/calc_flow",
        "Cargo.toml",
        "Cargo.lock",
        "pyproject.toml",
        "rust-toolchain.toml",
    ]
    subprocess.run(
        ["git", "diff", "--quiet", reference, "--", *product],
        cwd=ROOT,
        check=True,
    )
    if subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *product],
        cwd=ROOT,
    ):
        raise ValueError("untracked product source is not part of the sealed build")


def collect(args):
    seals, samples = read(args.seals), []
    verify_product_source(seals["head"]["source_ref"])
    args.output.mkdir(parents=True, exist_ok=False)
    budget = read(args.budget)
    started = time.monotonic()
    signal.signal(signal.SIGALRM, expired)
    daemon = json.loads(
        subprocess.check_output(
            ["multica", "daemon", "status", "--output", "json"], text=True, timeout=10
        )
    )["pid"]
    try:
        for case in CASES[:2]:
            payload = encode(fixture(case)[1])
            for round_number in range(1 if args.preflight or args.diagnostic else 2):
                for pair in range(1 if args.preflight or args.diagnostic else 10):
                    order = ("base", "head") if pair % 2 == 0 else ("head", "base")
                    for side in order:
                        if budget["spent_s"] + time.monotonic() - started > 565:
                            raise TimeoutError(
                                "600s ledger lacks a 35s cleanup reserve"
                            )
                        entry = round_number, pair, side
                        samples.append(
                            observe((args, seals, payload, daemon), case, entry)
                        )
                        save(args.output / "samples.json", samples)
    finally:
        elapsed = time.monotonic() - started
        budget["spent_s"] += elapsed
        budget["stages"].append(
            {
                "stage": "diagnostic"
                if args.diagnostic
                else ("preflight" if args.preflight else "formal"),
                "elapsed_s": elapsed,
                "samples": len(samples),
            }
        )
        save(args.budget, budget)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seals", type=Path, required=True)
    parser.add_argument("--budget", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--diagnostic", action="store_true")
    collect(parser.parse_args())
