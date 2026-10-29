"""Collect exactly the approved DAL-475 workload subset in ten minutes."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.benchmark_suite.report import comparison

ROOT = Path(__file__).resolve().parents[3]
CASES = ("arrow", "compute", "numpy", "cpu_matrix", "jax_matrix")
OUTPUT = ROOT / "target/standalone-performance"
OUTPUT.mkdir(parents=True, exist_ok=True)
worker = Path(__file__).with_name("worker.py")
paths = {
    "baseline": ROOT / "target/baseline/python",
    "candidate": ROOT / "target/candidate",
}
start = time.monotonic()
expected = {}
for side, path in paths.items():
    module = next((path / "calc_flow").glob("_native*.so"))
    expected[side] = hashlib.sha256(module.read_bytes()).hexdigest()
affinity = min(os.sched_getaffinity(0))
os.sched_setaffinity(0, {affinity})
raw = []
identity = None
sources = {}
summary = []
error = None
try:
    for case in CASES:
        samples = {side: [[], []] for side in paths}
        for round_index in range(2):
            for pair in range(10):
                for side in (
                    ("baseline", "candidate")
                    if pair % 2 == 0
                    else ("candidate", "baseline")
                ):
                    remaining = 600 - (time.monotonic() - start)
                    if remaining <= 0:
                        raise TimeoutError("600-second measurement budget exhausted")
                    env = {
                        **os.environ,
                        "PYTHONPATH": os.pathsep.join((str(paths[side]), str(ROOT))),
                        "TOKIO_WORKER_THREADS": "1",
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "MKL_NUM_THREADS": "1",
                        "JAX_PLATFORMS": "cpu",
                        "XLA_FLAGS": (
                            "--xla_cpu_multi_thread_eigen=false "
                            "intra_op_parallelism_threads=1"
                        ),
                    }
                    run = subprocess.run(
                        [sys.executable, str(worker), case],
                        cwd=ROOT,
                        env=env,
                        text=True,
                        capture_output=True,
                        timeout=remaining,
                        check=True,
                    )
                    item = json.loads(run.stdout)
                    assert item["native_sha256"] == expected[side]
                    assert item["correctness"] is True
                    if identity is None:
                        identity = item["identity"]
                    assert identity == item["identity"], (
                        "dependency, machine or thread drift"
                    )
                    sources.setdefault(side, item["source_sha256"])
                    assert sources[side] == item["source_sha256"], "Python source drift"
                    samples[side][round_index].append(item["seconds"])
                    raw.append(
                        {
                            **item,
                            "side": side,
                            "round": round_index,
                            "pair": pair,
                            "stderr": run.stderr,
                        }
                    )
                    (OUTPUT / "raw.json").write_text(json.dumps(raw, indent=2) + "\n")
        summary.append(
            {
                "id": case,
                **comparison(
                    {
                        "id": case,
                        "comparison": "interleaved",
                        "correctness": True,
                        **samples,
                    }
                ),
            }
        )
        print(json.dumps(summary[-1]), flush=True)
except (Exception, KeyboardInterrupt) as failure:
    error = str(failure)
    if isinstance(failure, subprocess.CalledProcessError):
        error += "\n" + failure.stderr
finally:
    report = {
        "baseline_sha": "d9e2759fb31bfd0c915d31f0cc534665ece72ee9",
        "candidate_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=ROOT
        ).strip(),
        "native_hashes": expected,
        "sources": sources,
        "identity": identity,
        "harness_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                worker,
                Path(__file__),
                ROOT / "benchmarks/array_support.py",
                ROOT / "scripts/benchmark_suite/report.py",
                ROOT / "scripts/benchmark_suite/statistics.py",
            )
        },
        "elapsed_seconds": time.monotonic() - start,
        "budget_seconds": 600,
        "summary": summary,
        "error": error,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "elapsed_seconds": report["elapsed_seconds"],
                "completed_cases": len(summary),
                "error": error,
            }
        ),
        flush=True,
    )
if error:
    raise SystemExit(1)
