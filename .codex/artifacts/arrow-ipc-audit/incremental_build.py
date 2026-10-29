"""Resume the pinned Arroyo release build within one cumulative wall budget."""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
import time
from pathlib import Path


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def build(root, frontend=False):
    budget = json.loads((root / "budget.json").read_text())
    remaining = math.floor(budget["budget_seconds"] - budget["spent_seconds"] - 10)
    if remaining <= 0:
        raise TimeoutError("cumulative build budget exhausted")
    number = len(budget["attempts"]) + 1
    database = json.loads((root / "postgres.json").read_text())
    wrapper = Path(__file__).with_name("cmake-compat.sh").resolve()
    config = {
        "CARGO_BUILD_JOBS": "8",
        "CFLAGS": "-std=gnu17",
        "CMAKE": str(wrapper),
        "DATABASE_URL": database["database_url"],
    }
    command = [
        "timeout",
        "--kill-after=10s",
        str(remaining),
        "cargo",
        "build",
        "--release",
        "--locked",
        "--manifest-path",
        "target/arroyo-arrow-source/Cargo.toml",
        "-p",
        "arroyo",
        "--bin",
        "arroyo",
    ]
    if frontend:
        command = [
            "timeout",
            "--kill-after=10s",
            str(remaining),
            "bash",
            str(Path(__file__).with_name("frontend_build.sh")),
        ]
    record = {
        "command": command,
        "config": config,
        "start_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "remaining_budget_s": remaining,
        "setup_reserve_s": 10,
        "exit_code": None,
    }
    path = root / f"attempt-{number}.json"
    write(path, record)
    started = time.monotonic()
    with (root / f"attempt-{number}.log").open("wb") as output:
        result = subprocess.run(
            command,
            stdout=output,
            stderr=subprocess.STDOUT,
            env={**os.environ, **config},
        )
    record.update(
        end_utc=datetime.datetime.now(datetime.UTC).isoformat(),
        elapsed_s=time.monotonic() - started,
        exit_code=result.returncode,
    )
    write(path, record)
    budget["spent_seconds"] += record["elapsed_s"]
    budget["attempts"].append(record)
    write(root / "budget.json", budget)
    print(
        json.dumps(
            {
                "attempt": number,
                "exit_code": result.returncode,
                "elapsed_s": record["elapsed_s"],
                "cumulative_build_s": budget["spent_seconds"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--frontend", action="store_true")
    args = parser.parse_args()
    build(args.root, args.frontend)
