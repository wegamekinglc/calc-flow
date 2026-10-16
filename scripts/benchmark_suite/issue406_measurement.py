"""Phase CPU and event layout evidence for the maintained Join/ASOF harness."""

from __future__ import annotations

import math
import time
from pathlib import Path


def _thread_runtimes() -> dict[str, tuple[int, str]]:
    threads = {}
    for entry in Path("/proc/self/task").iterdir():
        try:
            # A TID can be reused; bind it to the kernel start-time field.
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            runtime = int((entry / "schedstat").read_text().split()[0])
            threads[f"{entry.name}/{fields[19]}"] = (
                runtime,
                (entry / "comm").read_text().strip(),
            )
        except FileNotFoundError:
            continue
    return threads


class PhaseClock:
    """Own one interval; process CPU includes workers that exit before sampling."""

    def __init__(self, scope: str) -> None:
        self.scope = scope
        try:
            self.threads = _thread_runtimes()
            self.error = None
        except (OSError, ValueError, IndexError) as error:
            self.threads, self.error = {}, str(error)
        self.cpu = time.process_time_ns()
        self.started = time.perf_counter_ns()

    def finish(self) -> dict:
        wall = (time.perf_counter_ns() - self.started) / 1e9
        cpu = (time.process_time_ns() - self.cpu) / 1e9
        try:
            current = _thread_runtimes()
            error = self.error
        except (OSError, ValueError, IndexError) as failure:
            current, error = {}, str(failure)
        return {
            "schema": "calc-flow.phase-resources.v1",
            "scope": self.scope,
            "status": "unmeasured" if error else "measured",
            "reason": error,
            "wall_seconds": wall,
            "process_cpu_seconds": cpu,
            "process_cpu_clock": "CLOCK_PROCESS_CPUTIME_ID",
            "process_cpu_resolution_seconds": time.get_clock_info(
                "process_time"
            ).resolution,
            "thread_clock": "linux-task-schedstat-runtime-ns",
            "thread_boundary": "thread snapshots bracket the process CPU interval",
            "threads": [
                {
                    "tid": int(identity.split("/")[0]),
                    "identity": identity,
                    "name": name,
                    "runtime_seconds": (
                        runtime - self.threads.get(identity, (0, ""))[0]
                    )
                    / 1e9,
                    "new_thread": identity not in self.threads,
                }
                for identity, (runtime, name) in sorted(current.items())
            ],
            "departed_threads": sorted(self.threads.keys() - current.keys()),
            "thread_coverage": (
                "surviving-and-new-threads; exited CPU retained in process total"
            ),
        }


def validate_cpu_budget(*, threads: int, budget: float | None) -> float:
    """Require a serial 2x limit or an explicit budget within assigned CPUs."""
    maximum = 2.0 if threads == 1 else float(threads)
    if (
        type(threads) is not int
        or threads <= 0
        or not _finite(budget)
        or not 0 < budget <= maximum
        or (threads == 1 and budget != 2.0)
    ):
        raise ValueError("missing or invalid declared CPU budget")
    return budget


def validate_cpu_evidence(
    evidence: dict, *, threads: int, budget: float | None
) -> None:
    """Reject invalid clocks and excess CPU using the explicitly frozen budget."""
    budget = validate_cpu_budget(threads=threads, budget=budget)
    _validate_cpu_header(evidence)
    _validate_cpu_clocks(evidence)
    _validate_cpu_threads(evidence)
    if evidence["process_cpu_seconds"] > budget * evidence["wall_seconds"]:
        raise ValueError("process CPU budget exceeded")


def _validate_cpu_header(evidence: dict) -> None:
    if (
        not isinstance(evidence, dict)
        or evidence.get("schema")
        not in ("calc-flow.phase-resources.v1", "calc-flow.phase-resources.v2")
        or evidence.get("status") != "measured"
    ):
        raise ValueError("missing measured process CPU/per-thread runtime evidence")


def _validate_cpu_clocks(evidence: dict) -> None:
    _validate_clock_metadata(evidence)
    if (
        not _finite(evidence.get("wall_seconds"))
        or evidence["wall_seconds"] <= 0
        or not _finite(evidence.get("process_cpu_seconds"))
        or evidence["process_cpu_seconds"] < 0
    ):
        raise ValueError("missing measured process CPU/per-thread runtime evidence")


def _validate_clock_metadata(evidence: dict) -> None:
    clock = evidence.get("process_cpu_clock")
    if clock is not None and clock not in (
        "CLOCK_PROCESS_CPUTIME_ID",
        "linux-proc-stat-utime-stime",
    ):
        raise ValueError("unknown process CPU clock")
    if evidence["schema"] == "calc-flow.phase-resources.v2":
        if clock != "CLOCK_PROCESS_CPUTIME_ID":
            raise ValueError("v2 requires a high precision process CPU clock")
        _validate_resolution(evidence.get("process_cpu_resolution_seconds"))
    elif "process_cpu_resolution_seconds" in evidence:
        _validate_resolution(evidence["process_cpu_resolution_seconds"])


def _validate_resolution(resolution: object) -> None:
    if not _finite(resolution) or resolution <= 0:
        raise ValueError("invalid process CPU clock resolution")


def _validate_cpu_threads(evidence: dict) -> None:
    if (
        not isinstance(evidence.get("threads"), list)
        or not evidence["threads"]
        or not isinstance(evidence.get("departed_threads"), list)
    ):
        raise ValueError("missing measured process CPU/per-thread runtime evidence")
    for thread in evidence["threads"]:
        _validate_thread_runtime(thread)


def _validate_thread_runtime(thread: dict) -> None:
    if (
        not isinstance(thread, dict)
        or type(thread.get("tid")) is not int
        or not _finite(thread.get("runtime_seconds"))
        or thread["runtime_seconds"] < 0
    ):
        raise ValueError("invalid per-thread runtime evidence")


def sample_cpu_evidence(sample: dict, *, scope: str) -> dict:
    """Bind the CPU budget to the target's wall clock, excluding setup time."""
    evidence = sample.get("cpu_evidence")
    wall = sample.get(f"{scope}_seconds")
    if (
        not isinstance(evidence, dict)
        or evidence.get("scope") != scope
        or not _finite(wall)
        or wall <= 0
        or not _finite(evidence.get("wall_seconds"))
        or evidence["wall_seconds"] < wall
    ):
        raise ValueError("CPU evidence does not enclose its declared target phase")
    return {**evidence, "wall_seconds": wall}


def cpu_evidence_status(evidence: dict | None, *, threads: int, budget: float) -> str:
    """Keep tick-quantized short intervals inconclusive, including zero ticks."""
    if evidence is None:
        return "unmeasured"
    try:
        validate_cpu_evidence(evidence, threads=threads, budget=budget)
    except ValueError:
        return "invalid"
    resolution = evidence.get("process_cpu_resolution_seconds", 0.0)
    if not _finite(resolution) or resolution < 0:
        return "invalid"
    if (
        evidence["process_cpu_seconds"] + 2 * resolution
        > budget * evidence["wall_seconds"]
    ):
        return "inconclusive"
    return "verified"


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def event_layout(streams: dict) -> dict:
    """Record actual RecordBatch geometry without coalescing caller inputs."""
    events = []
    for name, stream in sorted(streams.items()):
        for index, event in enumerate(stream):
            batch = getattr(event, "batch", None)
            if batch is not None:
                batches = batch.to_pyarrow().to_batches()
                events.append(
                    {
                        "input": name,
                        "event_index": index,
                        "rows": batch.num_rows,
                        "record_batch_rows": [record.num_rows for record in batches],
                        "record_batch_count": len(batches),
                    }
                )
    return {
        "data_events": events,
        "multi_record_events": sum(event["record_batch_count"] > 1 for event in events),
    }
