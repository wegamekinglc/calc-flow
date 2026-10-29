"""Owned binary IPC observation and bounded worker cleanup."""

from __future__ import annotations

import io
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psutil
import pyarrow as pa
from fixtures import save


class Recorder(io.RawIOBase):
    def __init__(self, stream):
        self.stream = stream
        self.payload = bytearray()

    def readable(self):
        return True

    def read(self, size=-1):
        value = self.stream.read(size)
        self.payload.extend(value)
        return value


def consume(stream):
    recorded = Recorder(stream)
    with pa.ipc.open_stream(recorded) as reader:
        batches = list(reader)
        table = pa.Table.from_batches(batches, schema=reader.schema)
    finished = time.perf_counter_ns()
    return table, bytes(recorded.payload), finished


def send(stream, payload):
    started = time.perf_counter_ns()
    stream.write(payload)
    stream.flush()
    stream.close()
    return [started, time.perf_counter_ns()]


def sample_memory(process, stopped, records):
    owner = psutil.Process(process.pid)
    while not stopped.is_set():
        try:
            total = sum(
                child.memory_info().rss
                for child in [owner, *owner.children(recursive=True)]
                if child.is_running()
            )
            records.append({"at_ns": time.perf_counter_ns(), "rss_bytes": total})
        except psutil.Error:
            pass
        stopped.wait(0.01)


def ready(process, directory):
    deadline = time.perf_counter() + 20
    while not (directory / "ready").exists():
        if process.poll() is not None or time.perf_counter() >= deadline:
            raise RuntimeError("IPC worker did not become ready; inspect stderr.log")
        time.sleep(0.001)


def stop(process, daemon_pid):
    if process.pid == daemon_pid:
        raise RuntimeError("refusing to stop the workspace daemon")
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    process.wait(timeout=5)


def observe(command, config, payload):
    directory, environment, daemon_pid = config
    memory, stopped = [], threading.Event()
    launch = time.perf_counter_ns()
    with (directory / "stderr.log").open("wb") as errors:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=errors,
            env=environment,
        )
        sampler = threading.Thread(
            target=sample_memory, args=(process, stopped, memory)
        )
        sampler.start()
        try:
            ready(process, directory)
            startup_ns = time.perf_counter_ns() - launch
            with ThreadPoolExecutor(max_workers=2) as pool:
                try:
                    start = time.perf_counter_ns()
                    output = pool.submit(consume, process.stdout)
                    ingress = pool.submit(send, process.stdin, payload)
                    table, wire, finished = output.result(timeout=25)
                    send_ns = ingress.result(timeout=5)
                except BaseException:
                    stop(process, daemon_pid)
                    raise
            process.wait(timeout=25)
            if process.returncode:
                raise RuntimeError(f"IPC worker exit={process.returncode}")
            return (
                table,
                wire,
                {
                    "round_trip_ns": [start, finished],
                    "parent_send_ns": send_ns,
                    "process_ready_ns": startup_ns,
                },
            )
        finally:
            stop(process, daemon_pid)
            stopped.set()
            sampler.join()
            save(directory / "memory.json", memory)
            save(directory / "exit.json", {"code": process.returncode})
            process.stdin.close()
            process.stdout.close()


def worker_command(python, case, arm, directory, native_sha):
    return [
        "taskset",
        "-c",
        "4-7",
        str(python),
        str(Path(__file__).with_name("worker.py")),
        "--case",
        case.name,
        "--arm",
        arm,
        "--directory",
        str(directory),
        "--native-sha",
        native_sha,
    ]
