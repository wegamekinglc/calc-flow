"""Try explicit Arrow format declarations against the sealed real Arroyo CLI."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

from fixtures import digest, save
from transport import stop

ARROYO_SHA = "558b28b4c2aa3153500308684fd121dea2f040b5599d9f3342830834c24a319f"


def attempt(binary, source, root, format_name):
    directory = root / format_name
    directory.mkdir()
    query = directory / "query.sql"
    query.write_text(
        "CREATE TABLE events (id BIGINT NOT NULL, k BIGINT NOT NULL, "
        "ts TIMESTAMP NOT NULL, x BIGINT NOT NULL) WITH "
        f"(connector='single_file', path='{source}', type='source', "
        f"format='{format_name}'); SELECT id, x*2+1 AS y FROM events WHERE k<8;\n"
    )
    command = [
        "taskset",
        "-c",
        "4-7",
        str(binary),
        "run",
        "-p",
        "1",
        "--state-dir",
        str(directory / "state"),
        str(query),
    ]
    daemon = json.loads(
        subprocess.check_output(
            ["multica", "daemon", "status", "--output", "json"], text=True
        )
    )["pid"]
    environment = {
        **os.environ,
        "ARROYO__DISABLE_TELEMETRY": "true",
        "ARROYO__ADMIN__HTTP_PORT": "0",
        "ARROYO__DEFAULT_CHECKPOINT_INTERVAL": "1h",
        "TOKIO_WORKER_THREADS": "4",
    }
    started = time.perf_counter()
    with (
        (directory / "stdout.log").open("wb") as output,
        (directory / "stderr.log").open("wb") as errors,
    ):
        process = subprocess.Popen(
            command, stdout=output, stderr=errors, env=environment
        )
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            stop(process, daemon)
        record = {
            "argv": command,
            "format": format_name,
            "exit_code": process.returncode,
            "elapsed_s": time.perf_counter() - started,
        }
    save(directory / "attempt.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if digest(args.binary) != ARROYO_SHA:
        raise ValueError("Arroyo differs from the pinned release")
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    attempts = [
        attempt(
            args.binary.resolve(), args.input.resolve(), args.output.resolve(), name
        )
        for name in ("arrow", "arrow_ipc", "ipc")
    ]
    save(
        args.output / "attempts.json",
        {"elapsed_s": time.perf_counter() - started, "attempts": attempts},
    )
