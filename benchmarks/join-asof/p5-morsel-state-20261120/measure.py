from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import time
import zipfile
from importlib import metadata
from pathlib import Path

from benchmarks.engine_comparison import EngineCase
from calc_flow import _native


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def percentile95(values: list[float]) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def native_evidence(wheel: Path) -> dict:
    with zipfile.ZipFile(wheel) as archive:
        names = [
            name
            for name in archive.namelist()
            if name.startswith("calc_flow/_native") and name.endswith(".so")
        ]
        if len(names) != 1:
            raise ValueError("sealed wheel must contain exactly one native module")
        expected = sha256_bytes(archive.read(names[0]))
    actual = sha256_file(Path(_native.__file__))
    if actual != expected:
        raise ValueError("loaded native module differs from the sealed candidate wheel")
    return {
        "wheel": str(wheel),
        "wheel_sha256": sha256_file(wheel),
        "loaded_native": _native.__file__,
        "loaded_native_sha256": actual,
        "distribution_version": metadata.version("calc-flow-python"),
    }


def observe(case: dict, root: Path, timeout: float) -> dict:
    if timeout <= 0:
        raise TimeoutError("performance acquisition deadline reached")
    engine = EngineCase(case, root)
    try:
        result = engine.loop.run_until_complete(
            asyncio.wait_for(engine._stream(), timeout=timeout)
        )
        return {
            "seconds": result[1],
            "correctness": engine.validate(result[0]),
            "completion": engine.finish(),
        }
    finally:
        engine.close()


def measure_case(plan: dict, case_plan: dict, root: Path, deadline: float) -> dict:
    case = {
        "backend": "calc-flow-stream",
        "scenario": case_plan["scenario"],
        "rows": case_plan["rows"],
    }
    root.mkdir(parents=True, exist_ok=True)
    for index in range(plan["execution"]["warmups_per_case"]):
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            raise TimeoutError("performance acquisition deadline reached")
        observe(case, root / f"warmup-{index + 1}", timeout)
    samples = []
    for index in range(plan["execution"]["measured_samples_per_case"]):
        timeout = deadline - time.perf_counter()
        if timeout <= 0:
            raise TimeoutError("performance acquisition deadline reached")
        samples.append(observe(case, root / f"sample-{index + 1}", timeout))
    seconds = [sample["seconds"] for sample in samples]
    p50 = statistics.median(seconds)
    return {
        **case_plan,
        "samples": samples,
        "p50_ms": p50 * 1000,
        "p95_ms": percentile95(seconds) * 1000,
        "threshold_pass": p50 * 1000 <= case_plan["threshold_p50_ms"],
        "output_rows": samples[-1]["correctness"]["rows"],
        "all_oracles_passed": all(
            sample["correctness"]["passed"] for sample in samples
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text())
    deadline = started + plan["execution"]["acquisition_stop_seconds"]
    evidence = native_evidence(args.wheel)
    args.root.mkdir(parents=True, exist_ok=True)
    results = []
    error = None
    try:
        for case in plan["cases"]:
            results.append(measure_case(plan, case, args.root / case["id"], deadline))
    except BaseException as exc:
        error = repr(exc)
    elapsed = time.perf_counter() - started
    report = {
        "schema": "calc-flow.join-p5-performance-result.v1",
        "plan": plan,
        "candidate": evidence,
        "environment": {
            "python": os.sys.version,
            "tokio_worker_threads": os.environ.get("TOKIO_WORKER_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "taskset": os.environ.get("CALC_FLOW_TASKSET"),
            "cpus_allowed": next(
                (
                    line.strip()
                    for line in Path("/proc/self/status").read_text().splitlines()
                    if line.startswith("Cpus_allowed_list:")
                ),
                None,
            ),
        },
        "cases": results,
        "wall_seconds": elapsed,
        "within_budget": elapsed <= plan["execution"]["total_budget_seconds"],
        "error": error,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if error is not None:
        raise RuntimeError(error)
    if elapsed > plan["execution"]["total_budget_seconds"]:
        raise RuntimeError("performance acquisition exceeded its fixed budget")
    if any(not case["threshold_pass"] for case in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
