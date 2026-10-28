"""Run the fixed five-workload DAL-417 attribution experiment."""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from capture import arrow_sample, live_sample
from common import (
    ARROYO_REF,
    ARROYO_SHA,
    CASES,
    CF_REF,
    HISTORY,
    arms,
    case_named,
    legacy,
    save,
)

PROFILES = (
    ("live-512", "streaming"),
    ("live-512", "sql"),
    ("live-8192", "streaming"),
    ("live-8192", "sql"),
    ("arrow-full", "sql_plan"),
    ("arrow-512", "sql_plan"),
    ("arrow-512", "streaming"),
    ("window-live-512", "streaming"),
)


def identity(args):
    import calc_flow._native as native

    actual_ref = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_ref != CF_REF:
        raise RuntimeError("source HEAD differs from the frozen main baseline")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--"], check=True)
    if legacy.digest(args.arroyo) != ARROYO_SHA:
        raise RuntimeError("Arroyo binary does not match the fixed release")
    native_sha = legacy.digest(Path(native.__file__))
    daemon = json.loads(
        subprocess.check_output(
            ["multica", "daemon", "status", "--output", "json"], text=True
        )
    )
    os.environ["DAL417_DAEMON_PID"] = str(daemon["pid"])
    return {
        "source_ref": CF_REF,
        "arroyo_ref": ARROYO_REF,
        "native_sha256": native_sha,
        "arroyo_sha256": ARROYO_SHA,
        "python": sys.version,
        "platform": platform.platform(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("calc-flow-python", "pyarrow", "psutil", "websockets")
        },
        "harnesses": {
            p.name: legacy.digest(p) for p in Path(__file__).parent.glob("*.py")
        },
        "historical_harness_sha256": legacy.digest(HISTORY),
        "resource": {
            "controller_affinity": "0-1",
            "worker_affinity": "4-7",
            "tokio_workers": 4,
            "arroyo_parallelism": 1,
        },
        "pairs": args.pairs,
        "budget_s": args.budget,
        "contract": "DAL-417 current cross-engine attribution; informational",
    }


async def one(args, case, arm, directory, profile=False):
    directory.mkdir(parents=True, exist_ok=False)
    try:
        call = live_sample if case.transport == "live" else arrow_sample
        result = await call(args, case, arm, directory, profile)
    except Exception as error:
        save(directory / "failure.json", {"error": repr(error)})
        raise
    result = {"case": case.name, "arm": arm, "profiled": profile, **result}
    save(directory / "sample.json", result)
    return result


async def collect(args):
    start = time.perf_counter()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    record = identity(args)
    save(root / "identity.json", record)
    config = argparse.Namespace(**{**vars(args), "native_sha": record["native_sha256"]})
    samples = []
    selected = [case_named(name) for name in args.cases] if args.cases else CASES
    for case in selected:
        for pair in range(args.pairs):
            order = arms(case) if pair % 2 == 0 else arms(case)[::-1]
            for arm in order:
                if time.perf_counter() - start >= args.budget:
                    raise TimeoutError("fixed acquisition budget exhausted")
                name = f"{case.name}-{pair}-{arm}"
                print(name, flush=True)
                result = await one(config, case, arm, root / name)
                samples.append({"pair": pair, **result})
                save(root / "samples.json", samples)
                print(json.dumps(result), flush=True)
    if args.profiles:
        for name, arm in PROFILES:
            if time.perf_counter() - start >= args.budget:
                raise TimeoutError("budget exhausted before diagnostic profiles")
            print(f"profile-{name}-{arm}", flush=True)
            await one(
                config, case_named(name), arm, root / f"profile-{name}-{arm}", True
            )
    save(
        root / "completion.json",
        {"samples": len(samples), "elapsed_s": time.perf_counter() - start},
    )


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("operation", choices=("collect", "worker"))
    result.add_argument("--arroyo", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--pairs", type=int, default=5)
    result.add_argument("--budget", type=float, default=600)
    result.add_argument("--cases", nargs="*")
    result.add_argument("--profiles", action="store_true")
    result.add_argument("--case")
    result.add_argument("--arm")
    result.add_argument("--endpoint", default="")
    result.add_argument("--directory", type=Path)
    result.add_argument("--profile", action="store_true")
    result.add_argument("--native-sha")
    return result


def main():
    args = parser().parse_args()
    if args.operation == "worker":
        from engines import worker

        asyncio.run(worker(case_named(args.case), args))
    else:
        args.arroyo = args.arroyo.resolve()
        asyncio.run(collect(args))


if __name__ == "__main__":
    main()
