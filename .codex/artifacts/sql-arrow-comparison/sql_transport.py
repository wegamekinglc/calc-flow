"""Common ready-input/file-IPC result consumption, reusing reviewed ownership."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pyarrow as pa
from common import save
from extension_transport import owned, wait_file


def cf_sample(case, arm, directory, config):
    payload, daemon = config
    (directory / "input.arrow").write_bytes(payload)
    command = [
        "taskset",
        "-c",
        "4-7",
        sys.executable,
        str(Path(__file__).with_name("sql_worker.py")),
        "--case",
        case.name,
        "--arm",
        arm,
        "--directory",
        str(directory),
    ]
    settings = {
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
        "DAL417_GATED": "1",
    }
    save(directory / "command.json", {"argv": command, "environment": settings})
    launch = time.perf_counter_ns()
    with owned(command, (directory, settings, daemon)) as process:
        wait_file(directory / "input.arrow.ready.json", process)
        ready = time.perf_counter_ns()
        start = time.perf_counter_ns()
        (directory / "input.arrow.start").touch()
        wait_file(directory / "output.arrow", process)
        table = pa.ipc.open_stream(directory / "output.arrow").read_all()
        finish = time.perf_counter_ns()
        process.wait(timeout=10)
        if process.returncode:
            raise RuntimeError("CF SQL worker failed")
    return table, {
        "launch_ns": launch,
        "ready_ns": ready,
        "observer_ns": [start, finish],
        "transport": "file IPC; both inputs decoded before ready; parent read_all end",
    }
