"""Owned process observation for adjacent, real Arrow engine executions."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pyarrow as pa
from fixtures import save
from transport import consume, sample_memory, send


def wait_file(path, process, seconds=25):
    deadline = time.monotonic() + seconds
    while not path.exists():
        if process.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError(
                f"engine did not produce {path.name}; inspect stderr.log"
            )
        time.sleep(0.001)


def stop_owned(process, daemon):
    if process.pid == daemon:
        raise RuntimeError("refusing to stop the workspace daemon")
    for sig, timeout in ((signal.SIGINT, 5), (signal.SIGTERM, 2), (signal.SIGKILL, 2)):
        if process.poll() is not None:
            return
        process.send_signal(sig)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    process.wait(timeout=2)


@contextmanager
def owned(command, config):
    directory, settings, daemon = config
    stopped, memory = threading.Event(), []
    with (
        (directory / "stderr.log").open("wb") as errors,
        (directory / "stdout.log").open("wb") as output,
    ):
        process = subprocess.Popen(
            command,
            env={**os.environ, **settings},
            stdin=subprocess.PIPE,
            stdout=output if settings.get("DAL417_GATED") else subprocess.PIPE,
            stderr=errors,
        )
        sampler = threading.Thread(
            target=sample_memory, args=(process, stopped, memory)
        )
        sampler.start()
        try:
            yield process
        finally:
            started = time.perf_counter_ns()
            stop_owned(process, daemon)
            stopped.set()
            sampler.join()
            process.stdin.close()
            if process.stdout is not None:
                process.stdout.close()
            save(directory / "memory.json", memory)
            save(
                directory / "exit.json",
                {
                    "code": process.returncode,
                    "stop_ns": time.perf_counter_ns() - started,
                },
            )


def cf_sample(case, directory, payload, daemon):
    command = [
        "taskset",
        "-c",
        "4-7",
        sys.executable,
        str(Path(__file__).with_name("cf_batch_worker.py")),
        "--case",
        case.name,
        "--directory",
        str(directory),
    ]
    settings = {
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "ARROW_NUM_THREADS": "1",
        "RUST_LOG": "warn",
    }
    save(directory / "command.json", {"argv": command, "environment": settings})
    launch = time.perf_counter_ns()
    with owned(command, (directory, settings, daemon)) as process:
        wait_file(directory / "ready", process)
        ready = time.perf_counter_ns()
        with ThreadPoolExecutor(max_workers=2) as pool:
            try:
                start = time.perf_counter_ns()
                output = pool.submit(consume, process.stdout)
                ingress = pool.submit(send, process.stdin, payload)
                table, wire, finish = output.result(timeout=25)
                send_ns = ingress.result(timeout=5)
            except BaseException:
                stop_owned(process, daemon)
                raise
        (directory / "finish").touch()
        process.wait(timeout=20)
        if process.returncode:
            raise RuntimeError("CF native Batch worker failed")
    (directory / "output.arrow").write_bytes(wire)
    return table, {
        "launch_ns": launch,
        "ready_ns": ready,
        "observer_ns": [start, finish],
        "parent_send_ns": send_ns,
        "transport": "stdin/stdout IPC; input decode inside observer interval",
    }


def query(case, source, output):
    def quoted(path):
        return str(path).replace("'", "''")

    fields = (
        "id BIGINT NOT NULL,k BIGINT NOT NULL,ts TIMESTAMP NOT NULL,x BIGINT NOT NULL"
    )
    watermark = (
        ", WATERMARK FOR ts AS ts - INTERVAL '1 microsecond'" if case.window else ""
    )
    output_fields = (
        "window_start TIMESTAMP NOT NULL,window_end TIMESTAMP NOT NULL,"
        "k BIGINT NOT NULL,total BIGINT,cnt BIGINT NOT NULL"
        if case.window
        else "id BIGINT NOT NULL,y BIGINT NOT NULL"
    )
    calculation = (
        "SELECT w.start AS window_start,w.end AS window_end,k,total,cnt FROM "
        "(SELECT TUMBLE(INTERVAL '4 seconds') AS w,k,SUM(x) AS total,"
        "COUNT(id) AS cnt FROM events WHERE id>=0 GROUP BY 1,2)"
        if case.window
        else "SELECT id,x*2+1 AS y FROM events WHERE id>=0 AND k<8"
    )
    return (
        f"CREATE TABLE events ({fields}{watermark}) WITH "
        f"(connector='single_file',path='{quoted(source)}',"
        "type='source',format='json');\n"
        f"CREATE TABLE results ({output_fields}) WITH "
        f"(connector='single_file',path='{quoted(output)}',"
        "type='sink',format='json');\n"
        f"INSERT INTO results {calculation};\n"
    )


def native_settings(directory, source, expected, window):
    settings = {
        "TOKIO_WORKER_THREADS": "4",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "RUST_LOG": "warn",
        "DAL417_GATED": "1",
        "DAL417_EXPECTED_ROWS": str(expected),
        "DAL417_INPUT_PATH": str(source),
        "ARROYO__CONTROLLER__SCHEDULER": "embedded",
        "ARROYO__DISABLE_TELEMETRY": "true",
        "ARROYO__ADMIN__HTTP_PORT": "0",
        "ARROYO__DEFAULT_CHECKPOINT_INTERVAL": "1h",
    }
    if window:
        settings["DAL417_TRACE_WM"] = str(directory / "watermarks.txt")
    return {
        **settings,
        **{
            f"ARROYO__{name}__BIND_ADDRESS": "127.0.0.1"
            for name in ("ADMIN", "API", "CONTROLLER", "WORKER")
        },
    }


def native_sample(case, directory, payload, config):
    binary, daemon = config
    source, output = directory / "input.arrow", directory / "output.arrow"
    source.write_bytes(payload)
    statement = directory / "query.sql"
    statement.write_text(query(case, source, output))
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
        str(statement),
    ]
    settings = native_settings(
        directory, source, 512 if case.window else 16384, case.window
    )
    save(directory / "command.json", {"argv": command, "environment": settings})
    launch, launch_utc = time.perf_counter_ns(), time.time_ns()
    with owned(command, (directory, settings, daemon)) as process:
        wait_file(Path(str(source) + ".ready.json"), process)
        ready = time.perf_counter_ns()
        start = time.perf_counter_ns()
        Path(str(source) + ".start").touch()
        wait_file(output, process)
        table = pa.ipc.open_stream(output).read_all()
        finish = time.perf_counter_ns()
        eof_at_consume = Path(str(source) + ".eof").exists()
        wait_file(Path(str(output) + ".meta.json"), process)
        metadata = json.loads(Path(str(output) + ".meta.json").read_text())
        metadata["source_eof_at_consume"] = eof_at_consume
        Path(str(source) + ".finish").touch()
        wait_file(Path(str(output) + ".closed"), process, 5)
        save(directory / "worker.json", metadata)
    return table, {
        "launch_ns": launch,
        "launch_utc_ns": launch_utc,
        "ready_ns": ready,
        "observer_ns": [start, finish],
        "transport": "file IPC; input before ready; observer starts at gate",
    }
