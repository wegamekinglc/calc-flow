"""Exact-release P7 slow-sink diagnostics; separate timing and RSS processes.

This fixed 64 + 64k burst supplements the unchanged paired application workload.
Physical Data chunks are bounded to 256 rows, with one watermark per logical
append. The first append blocks inside Sink.write until a Native sink edge has
demonstrably backed up, then remains held for another 50 ms. All reported output
latencies include this hold. RSS mode adds sampling and is never a timing gate.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import gc
import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

ROLES = ("candidate", "control")
HISTORY_ROWS = 1_024_000
APPENDS = (64, 64_000)
CHUNK_ROWS = 256
EDGE_BYTES = 1 << 20
HOLD_SECONDS = 0.05
POLL_SECONDS = 0.001
MEMORY_INTERVAL_SECONDS = 0.005
TIMEOUT_SECONDS = 60
THREAD_SETTINGS = {
    "TOKIO_WORKER_THREADS": "32",
    "POLARS_MAX_THREADS": "32",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}
ROLLING_STAGES = (
    "input_validation",
    "ordering_proof",
    "entity_resolution",
    "state_preparation",
    "numeric_update",
    "history_maintenance",
    "arrow_output",
    "budget_preparation",
    "send_wait",
    "other",
)


def thread_settings(environment) -> dict[str, str]:
    for name, expected in THREAD_SETTINGS.items():
        if environment.get(name, expected) != expected:
            raise ValueError(f"predeclared diagnostic requires {name}={expected}")
    return dict(THREAD_SETTINGS)


def _check_release_build(record: dict) -> None:
    if (
        record.get("profile") != "release"
        or record.get("tracked_source_clean") is not True
    ):
        raise ValueError("declared builds must be clean release builds")
    for name, length in (
        ("source_sha", 40),
        ("native_sha256", 64),
        ("wheel_sha256", 64),
        ("cargo_lock_sha256", 64),
    ):
        value = record.get(name)
        if (
            not isinstance(value, str)
            or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None
        ):
            raise ValueError(f"declared build has invalid {name}")


def release_pair(role: str, build: dict, peer: dict) -> dict:
    pair = copy.deepcopy({role: build, ROLES[1 - ROLES.index(role)]: peer})
    for record in pair.values():
        _check_release_build(record)
    first, second = pair["candidate"], pair["control"]
    for name in ("source_sha", "native_sha256", "wheel_sha256"):
        if first[name] == second[name]:
            raise ValueError(f"candidate/control must have distinct {name}")
    for name in ("cargo_lock_sha256", "features", "rustc_verbose"):
        if not first.get(name) or first[name] != second.get(name):
            raise ValueError(f"candidate/control build {name} differs or is missing")
    return pair


def require_release(role: str, identity: dict, declared: dict) -> None:
    expected = declared[role]
    if (
        identity["native_sha256"] != expected["native_sha256"]
        or identity["build"] != expected
    ):
        raise ValueError(f"{role} is not the declared release")


def _check_bounds(status: dict, *, drained: bool = False) -> None:
    if not status["edges"]:
        raise ValueError("Native edge metrics are missing")
    for name, edge in status["edges"].items():
        for current, peak, limit in (
            ("current_envelopes", "high_water_envelopes", "envelope_limit"),
            ("current_rows", "high_water_rows", "row_limit"),
            ("current_bytes", "high_water_bytes", "byte_limit"),
        ):
            if not 0 <= edge[current] <= edge[peak] <= edge[limit]:
                raise ValueError(f"Native edge {name} exceeds {limit}")
            if drained and edge[current] != 0:
                raise ValueError(f"Native edge {name} did not drain {current}")


def _check_node_errors(status: dict) -> None:
    for group in ("sources", "operators", "sinks"):
        for name, item in status[group].items():
            if item["errors"] != 0:
                raise ValueError(f"{group}.{name} reported errors")


def _check_health(status: dict) -> None:
    if (
        status["state"] not in ("running", "draining", "completed")
        or status["task_errors"] != 0
        or status["metrics_overflowed"]
    ):
        raise ValueError("Native job failed or overflowed its metrics")
    _check_node_errors(status)
    if not status["rolling_metrics"]:
        raise ValueError("Native rolling metrics are missing")
    if any(
        node["overflowed"] is not False for node in status["rolling_metrics"].values()
    ):
        raise ValueError("Native rolling metrics overflowed")


def _check_callback_outcomes(callback: dict) -> None:
    if any(type(value) is not int or value < 0 for value in callback.values()):
        raise ValueError("rolling callback metrics require nonnegative integers")
    if callback["succeeded"] > callback["started"] or any(
        callback[outcome] for outcome in ("failed", "cancelled", "interrupted")
    ):
        raise ValueError("rolling callback outcomes are inconsistent")


def _callback_settled(callback: dict) -> bool:
    _check_callback_outcomes(callback)
    if callback["started"] != callback["succeeded"]:
        return False
    if callback["callback_duration_ns"] != sum(
        callback[f"{stage}_duration_ns"] for stage in ROLLING_STAGES
    ):
        raise ValueError("rolling callback stages do not cover its duration")
    return True


def _rolling_settled(status: dict) -> bool:
    settled = True
    for node in status["rolling_metrics"].values():
        for name in ("data", "watermark", "end"):
            if not _callback_settled(node[name]):
                settled = False
    return settled


def backpressure_proof(before: dict, held: dict) -> dict:
    """Require a new blocked send and live full charge on a Native sink edge."""
    _check_bounds(held)
    proof = {}
    for name, edge in held["edges"].items():
        if not name.startswith("sink/"):
            continue
        added = edge["blocked_sends"] - before["edges"][name]["blocked_sends"]
        full = any(
            edge[current] == edge[limit]
            for current, limit in (
                ("current_envelopes", "envelope_limit"),
                ("current_rows", "row_limit"),
                ("current_bytes", "byte_limit"),
            )
        )
        if added > 0 and full:
            proof[name] = {**edge, "new_blocked_sends": added}
    return proof


def _check_ended_nodes(status: dict, rows: int) -> None:
    for group in ("sources", "operators", "sinks"):
        for name, item in status[group].items():
            if item["ended"] is not True:
                raise ValueError(f"{group}.{name} did not end")
            if group == "operators" and item["input_rows"] != rows:
                raise ValueError(f"operator {name} input row count changed")


def check_terminal(status: dict, *, rows: int, batches: int) -> None:
    _check_health(status)
    _check_bounds(status, drained=True)
    if status["state"] != "completed":
        raise ValueError("Native job did not complete")
    if not _rolling_settled(status):
        raise ValueError("terminal rolling callbacks have not settled")
    source = status["sources"]["input"]
    if source["data_rows"] != rows or source["data_batches"] != batches:
        raise ValueError("source Data counts changed")
    if status["sinks"]["profile"]["delivered_rows"] != rows:
        raise ValueError("sink delivered row count changed")
    _check_ended_nodes(status, rows)


class BoundedSource:
    """Own only two queued events and expose actual next-poll readiness."""

    def __init__(self, *, clock=time.perf_counter_ns) -> None:
        self.events = asyncio.Queue(maxsize=2)
        self.waiting = asyncio.Event()
        self.eof_polled_ns = None
        self.clock = clock

    def capabilities(self):
        from calc_flow import (
            NativeWatermarkCapability,
            ReplayPositioning,
            SourceCapabilities,
            SourceDeliveryCapability,
        )

        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=CHUNK_ROWS,
            max_batch_bytes=EDGE_BYTES,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor) -> None:
        if cursor is not None:
            raise ValueError("slow-sink source does not support replay")

    async def next(self):
        self.waiting.set()
        try:
            event = await self.events.get()
        finally:
            self.waiting.clear()
        if event is None:
            self.eof_polled_ns = self.clock()
        return event

    async def push(self, event) -> None:
        await self.events.put(event)

    async def close(self) -> None:
        return None


@dataclass(frozen=True, slots=True)
class ArrowDelivery:
    table: object
    entered_ns: int
    materialized_ns: int


class DelayedSink:
    """Hold the first armed write before Arrow conversion, with one output slot."""

    def __init__(self, *, clock=time.perf_counter_ns) -> None:
        self.tables = asyncio.Queue(maxsize=1)
        self.gate = asyncio.Event()
        self.gate.set()
        self.entered = asyncio.Event()
        self.entered_ns = None
        self.released_ns = None
        self.closed = False
        self.inflight = 0
        self.clock = clock

    def arm(self) -> None:
        if self.inflight or not self.tables.empty() or self.closed:
            raise ValueError("cannot arm a nonempty or closed sink")
        self.entered.clear()
        self.entered_ns = None
        self.released_ns = None
        self.gate.clear()

    def release(self) -> None:
        if not self.gate.is_set():
            self.released_ns = self.clock()
            self.gate.set()

    async def open(self) -> None:
        return None

    async def write(self, batch) -> None:
        if self.closed:
            raise ValueError("write after sink close")
        entered_ns = self.clock()
        self.inflight += 1
        try:
            if not self.gate.is_set() and self.entered_ns is None:
                self.entered_ns = entered_ns
                self.entered.set()
                await self.gate.wait()
            table = batch.to_pyarrow()
            delivery = ArrowDelivery(table, entered_ns, self.clock())
            await self.tables.put(delivery)
        finally:
            self.inflight -= 1

    async def receive(self):
        if self.closed and self.tables.empty():
            return None
        return await self.tables.get()

    async def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self.tables.empty():
                self.tables.put_nowait(None)


@dataclass(frozen=True, slots=True)
class _StreamHandles:
    source: BoundedSource
    sink: DelayedSink
    job: object


async def release_after_backpressure(
    stream: _StreamHandles,
    before: dict,
    *,
    hold_seconds: float = HOLD_SECONDS,
    timeout: float = TIMEOUT_SECONDS,
    on_held=None,
) -> dict:
    sink, job = stream.sink, stream.job
    try:
        async with asyncio.timeout(timeout):
            await sink.entered.wait()
            while True:
                held = job.status()
                _check_health(held)
                proof = backpressure_proof(before, held)
                if proof:
                    observed = time.perf_counter_ns()
                    break
                await asyncio.sleep(POLL_SECONDS)
            if on_held is not None:
                await on_held()
            await asyncio.sleep(hold_seconds)
    finally:
        sink.release()
    return {"observed_ns": observed, "held_status": held, "edges": proof}


async def _drain_terminal(sink) -> dict:
    rows, batches = 0, 0
    while (delivery := await sink.receive()) is not None:
        rows += delivery.table.num_rows
        batches += 1
    return {"rows": rows, "batches": batches}


async def finish(
    stream: _StreamHandles,
    *,
    rows: int,
    batches: int,
    clock=time.perf_counter_ns,
) -> dict:
    source, sink, job = stream.source, stream.sink, stream.job
    drain = asyncio.create_task(_drain_terminal(sink))
    try:
        async with asyncio.timeout(TIMEOUT_SECONDS):
            started = clock()
            await source.push(None)
            queued = clock()
            outcome = await job.wait_async()
            completed = clock()
            extra = await drain
        if outcome.state != "completed":
            raise ValueError(f"Native EOF ended in {outcome.state}: {outcome.errors}")
        if extra != {"rows": 0, "batches": 0}:
            raise ValueError(f"unexpected terminal output: {extra}")
        terminal = job.status()
        check_terminal(terminal, rows=rows, batches=batches)
        if not source.events.empty() or not sink.tables.empty():
            raise ValueError("Python source or sink queue did not drain")
        if source.eof_polled_ns is None:
            raise ValueError("source never polled EOF")
        return {
            "eof_enqueue_started_ns": started,
            "eof_enqueue_completed_ns": queued,
            "eof_source_polled_ns": source.eof_polled_ns,
            "job_completed_ns": completed,
            "eof_enqueue_to_completed_ns": completed - started,
            "eof_poll_to_completed_ns": completed - source.eof_polled_ns,
            "extra_output": extra,
            "terminal_status": terminal,
        }
    finally:
        drain.cancel()
        await asyncio.gather(drain, return_exceptions=True)


def memory_summary(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("no process memory samples were collected")
    ordered = sorted(samples, key=lambda item: item["monotonic_ns"])
    gaps = [
        second["monotonic_ns"] - first["monotonic_ns"]
        for first, second in zip(ordered, ordered[1:], strict=False)
    ]
    return {
        "sample_count": len(samples),
        "sampled_peak_rss_bytes": max(item["rss_bytes"] for item in samples),
        "process_lifetime_high_water_bytes": max(
            item["lifetime_peak_rss_bytes"] for item in samples
        ),
        "first_sample_ns": ordered[0]["monotonic_ns"],
        "last_sample_ns": ordered[-1]["monotonic_ns"],
        "maximum_observed_sample_gap_ns": max(gaps, default=0),
    }


def _rss_sample() -> dict:
    import pyarrow as pa

    from benchmarks.native_memory_profile import read_memory_status

    return {
        "monotonic_ns": time.perf_counter_ns(),
        **read_memory_status(Path("/proc/self/status").read_text()),
        "python_arrow_pool_bytes": pa.total_allocated_bytes(),
    }


class MemoryProbe:
    """Own a bounded RSS sample log only in the separately instrumented process."""

    def __init__(self, *, enabled: bool) -> None:
        self.enabled = enabled
        self.samples = []
        self.phases = []
        self.stopped = asyncio.Event()
        self.task = None

    async def _sample(self) -> dict:
        if len(self.samples) >= 10_000:
            raise ValueError("memory sample log capacity exceeded")
        sample = await asyncio.to_thread(_rss_sample)
        self.samples.append(sample)
        return sample

    async def capture(self, phase: str, job, **ownership) -> None:
        if self.enabled:
            self.phases.append(
                {
                    "phase": phase,
                    **await self._sample(),
                    "status": job.status(),
                    "ownership": ownership,
                }
            )

    def start(self) -> None:
        if self.enabled:
            self.task = asyncio.create_task(self._run())

    async def _run(self) -> None:
        while not self.stopped.is_set():
            try:
                await asyncio.wait_for(
                    self.stopped.wait(), timeout=MEMORY_INTERVAL_SECONDS
                )
            except TimeoutError:
                await self._sample()

    async def stop(self) -> None:
        self.stopped.set()
        if self.task is not None:
            await self.task

    def result(self):
        if not self.enabled:
            return None
        return {
            **memory_summary(self.samples),
            "nominal_interval_seconds": MEMORY_INTERVAL_SECONDS,
            "interval": "ready/preloaded through terminal completion",
            "phases": self.phases,
            "samples": sorted(self.samples, key=lambda item: item["monotonic_ns"]),
            "limits": (
                "Sampled RSS is not an exact timed peak or live rolling heap. "
                "VmHWM is the process lifetime high-water including preload. "
                "RSS includes interpreter, caller inputs, pending rows, state, "
                "queued and collected output, allocator caches, transport logs, "
                "sampler metadata "
                "and checkpoint work. Python Arrow pool bytes exclude many Rust "
                "allocations. GC and reference release do not force RSS reclamation."
            ),
        }


def _prepare(config, start: int, rows: int) -> tuple[tuple, dict]:
    from benchmarks.warm_stream import _prepared_events

    events, chunks = [], []
    for offset in range(0, rows, CHUNK_ROWS):
        count = min(CHUNK_ROWS, rows - offset)
        data, watermark = _prepared_events(config, start + offset, count)
        table = data.batch.to_pyarrow()
        if table.num_rows > CHUNK_ROWS or table.nbytes > EDGE_BYTES:
            raise ValueError("prepared source chunk exceeds the declared budget")
        events.append(data)
        chunks.append(
            {
                "start_row": start + offset,
                "rows": count,
                "logical_arrow_bytes": table.nbytes,
                "cursor_hex": data.cursor.order.hex(),
            }
        )
    events.append(watermark)
    return tuple(events), {
        "start_row": start,
        "rows": rows,
        "chunks": chunks,
        "watermark_micros": table["event_time"][-1].value,
    }


async def _send(source, events: tuple) -> None:
    for event in events:
        await source.push(event)


async def _collect_rows(sink, rows: int) -> tuple[list, list[dict]]:
    tables, blocks, received = [], [], 0
    while received < rows:
        delivery = await sink.receive()
        received_ns = time.perf_counter_ns()
        if delivery is None:
            raise ValueError(f"sink closed after {received} of {rows} expected rows")
        table = delivery.table
        tables.append(table)
        blocks.append(
            {
                "start_offset": received,
                "rows": table.num_rows,
                "logical_arrow_bytes": table.nbytes,
                "sink_entered_ns": delivery.entered_ns,
                "arrow_materialized_ns": delivery.materialized_ns,
                "arrow_received_ns": received_ns,
            }
        )
        received += table.num_rows
        if received > rows:
            raise ValueError(
                "sink emitted more rows than the complete logical workload"
            )
    return tables, blocks


def _native_drain_ready(
    status: dict, delivered: int, rows: int, watermark: int
) -> bool:
    return (
        delivered == rows
        and status["watermark_micros"] is not None
        and status["watermark_micros"] >= watermark
        and all(edge["current_envelopes"] == 0 for edge in status["edges"].values())
    )


def _caller_queues_drained(source, sink) -> bool:
    return (
        source.waiting.is_set()
        and source.events.empty()
        and sink.tables.empty()
        and sink.inflight == 0
    )


async def _wait_drained(job, source, sink, *, rows: int, watermark: int) -> dict:
    async with asyncio.timeout(TIMEOUT_SECONDS):
        while True:
            status = job.status()
            _check_health(status)
            _check_bounds(status)
            delivered = status["sinks"]["profile"]["delivered_rows"]
            if delivered > rows:
                raise ValueError("sink delivered additional rows")
            if (
                _native_drain_ready(status, delivered, rows, watermark)
                and _caller_queues_drained(source, sink)
                and _rolling_settled(status)
            ):
                _check_bounds(status, drained=True)
                return status
            await asyncio.sleep(POLL_SECONDS)


async def _preload(config, source, sink, job) -> dict:
    import pyarrow as pa

    from benchmarks.warm_stream import _validate_output

    schema = None
    maximum_error = 0.0
    transport = []
    async with asyncio.timeout(300):
        for start in range(0, config.history_rows, config.history_segment_rows):
            rows = min(config.history_segment_rows, config.history_rows - start)
            events, descriptor = _prepare(config, start, rows)
            async with asyncio.TaskGroup() as group:
                group.create_task(_send(source, events))
                received = group.create_task(_collect_rows(sink, rows))
            table = pa.concat_tables(received.result()[0])
            if schema is not None and not schema.equals(
                table.schema, check_metadata=True
            ):
                raise ValueError("preload output schema changed between segments")
            schema = table.schema
            checked = _validate_output(table, config, start=start, rows=rows)
            maximum_error = max(maximum_error, checked["max_absolute_error"])
            transport.append(descriptor)
    await _wait_drained(
        job,
        source,
        sink,
        rows=config.history_rows,
        watermark=descriptor["watermark_micros"],
    )
    return {
        "schema": schema,
        "max_absolute_error": maximum_error,
        "transport": transport,
        "physical_data_batches": sum(len(item["chunks"]) for item in transport),
    }


async def _watermark_observation(job, target: int) -> dict:
    while True:
        status = job.status()
        _check_health(status)
        value = status["watermark_micros"]
        if value is not None and value >= target:
            return {
                "observed_ns": time.perf_counter_ns(),
                "observed_watermark_micros": value,
                "target_watermark_micros": target,
                "boundary": (
                    "first polled job progress frontier at or beyond target; "
                    "not an exact per-output watermark callback timestamp"
                ),
            }
        await asyncio.sleep(POLL_SECONDS)


def _write_arrow(table, path: Path) -> None:
    import pyarrow as pa

    with (
        pa.OSFile(str(path), "wb") as destination,
        pa.ipc.new_file(destination, table.schema) as writer,
    ):
        writer.write_table(table)


async def _send_burst(source, prepared: list[tuple]) -> list[dict]:
    marks = []
    for events, descriptor in prepared:
        started = time.perf_counter_ns()
        await _send(source, events)
        marks.append(
            {
                "start_row": descriptor["start_row"],
                "rows": descriptor["rows"],
                "enqueue_started_ns": started,
                "enqueue_completed_ns": time.perf_counter_ns(),
            }
        )
    return marks


def _output_latencies(enqueue: list[dict], blocks: list[dict]) -> list[dict]:
    result, offset = [], 0
    for item in enqueue:
        end = offset + item["rows"]
        selected = [
            block
            for block in blocks
            if block["start_offset"] < end
            and block["start_offset"] + block["rows"] > offset
        ]
        if not selected:
            raise ValueError("logical append has no Arrow delivery")
        started = item["enqueue_started_ns"]
        result.append(
            {
                **item,
                "first_sink_entry_ns": selected[0]["sink_entered_ns"],
                "first_arrow_materialized_ns": selected[0]["arrow_materialized_ns"],
                "first_arrow_received_ns": selected[0]["arrow_received_ns"],
                "complete_arrow_received_ns": selected[-1]["arrow_received_ns"],
                "enqueue_to_first_sink_entry_ns": (
                    selected[0]["sink_entered_ns"] - started
                ),
                "enqueue_to_first_arrow_ns": (
                    selected[0]["arrow_received_ns"] - started
                ),
                "enqueue_to_complete_arrow_ns": (
                    selected[-1]["arrow_received_ns"] - started
                ),
            }
        )
        offset = end
    return result


def _validate_input_ownership(config, prepared: list[tuple]) -> None:
    from benchmarks.warm_stream import SCHEMA, _input_table

    for events, descriptor in prepared:
        for event, chunk in zip(events[:-1], descriptor["chunks"], strict=True):
            actual = event.batch.to_pyarrow()
            expected = _input_table(config, chunk["start_row"], chunk["rows"])
            if not actual.schema.equals(
                SCHEMA, check_metadata=True
            ) or not actual.equals(expected):
                raise ValueError("caller-owned source input changed")


async def _burst(config, stream: _StreamHandles, schema, memory, output: Path) -> dict:
    import pyarrow as pa

    from benchmarks.warm_stream import _validate_output

    source, sink, job = stream.source, stream.sink, stream.job
    prepared, position = [], config.history_rows
    for rows in APPENDS:
        prepared.append(_prepare(config, position, rows))
        position += rows
    inputs = pa.concat_tables(
        [event.batch.to_pyarrow() for events, _ in prepared for event in events[:-1]]
    )
    await asyncio.to_thread(_write_arrow, inputs, output / "input.arrow")
    del inputs
    await memory.capture(
        "append_prepared",
        job,
        caller_prepared_input_rows=sum(APPENDS),
        source_queue_events=source.events.qsize(),
        sink_queue_tables=sink.tables.qsize(),
        collected_output_rows=0,
    )
    before = job.status()

    async def held_memory():
        await memory.capture(
            "sink_held_backlogged",
            job,
            caller_prepared_input_rows=sum(APPENDS),
            source_queue_events=source.events.qsize(),
            sink_queue_tables=sink.tables.qsize(),
            sink_callbacks_inflight=sink.inflight,
            collected_output_rows=0,
        )

    sink.arm()
    async with asyncio.timeout(TIMEOUT_SECONDS):
        async with asyncio.TaskGroup() as group:
            producer = group.create_task(_send_burst(source, prepared))
            consumer = group.create_task(_collect_rows(sink, sum(APPENDS)))
            release = group.create_task(
                release_after_backpressure(stream, before, on_held=held_memory)
            )
            frontier = group.create_task(
                _watermark_observation(job, prepared[-1][1]["watermark_micros"])
            )
    tables, blocks = consumer.result()
    output_done = blocks[-1]["arrow_received_ns"]
    enqueue = producer.result()
    frontier_facts = frontier.result()
    drained = await _wait_drained(
        job,
        source,
        sink,
        rows=position,
        watermark=prepared[-1][1]["watermark_micros"],
    )
    drained_ns = time.perf_counter_ns()
    table = pa.concat_tables(tables)
    if not table.schema.equals(schema, check_metadata=True):
        raise ValueError("append output schema differs from the preloaded stream")
    checked = _validate_output(
        table, config, start=config.history_rows, rows=sum(APPENDS)
    )
    _validate_input_ownership(config, prepared)
    await asyncio.to_thread(_write_arrow, table, output / "output.arrow")
    started = enqueue[0]["enqueue_started_ns"]
    return {
        "before_status": before,
        "backpressure": release.result(),
        "logical_inputs": [descriptor for _, descriptor in prepared],
        "physical_data_batches": sum(len(item[1]["chunks"]) for item in prepared),
        "blocks_in_emitted_order": blocks,
        "appends": _output_latencies(enqueue, blocks),
        "gate_entered_ns": sink.entered_ns,
        "gate_released_ns": sink.released_ns,
        "actual_gate_hold_ns": sink.released_ns - sink.entered_ns,
        "minimum_hold_after_backpressure_ns": int(HOLD_SECONDS * 1e9),
        "burst_enqueue_to_complete_arrow_ns": output_done - started,
        "release_to_complete_arrow_ns": output_done - sink.released_ns,
        "drained_observed_ns": drained_ns,
        "release_to_drained_ns": drained_ns - sink.released_ns,
        "watermark": frontier_facts,
        "enqueue_to_watermark_observed_ns": frontier_facts["observed_ns"] - started,
        "enqueue_to_watermark_and_complete_arrow_ns": (
            max(output_done, frontier_facts["observed_ns"]) - started
        ),
        "drained_status": drained,
        "correctness": checked,
        "source_input_unchanged": True,
    }


def _workload(config) -> dict:
    return {
        "config": asdict(config),
        "logical_append_rows": list(APPENDS),
        "physical_data_chunk_rows": CHUNK_ROWS,
        "watermark_policy": "one final watermark per logical segment or append",
        "source_queue_events": 2,
        "sink_queue_tables": 1,
        "edge_rows": CHUNK_ROWS,
        "edge_bytes": EDGE_BYTES,
        "hold_after_native_backpressure_seconds": HOLD_SECONDS,
        "status_poll_seconds": POLL_SECONDS,
        "finite_rtol": 1e-10,
        "finite_atol": 1e-10,
        "scope": (
            "ready and preloaded; enqueue attempt through complete Arrow receipt; "
            "all gate wait included; fixed bounded transport; supplemental to the "
            "unchanged paired application measurement"
        ),
    }


def _instrument() -> dict:
    from benchmarks import (
        native_checkpoint_compatibility,
        native_memory_profile,
        rolling_indicator_comparison,
        warm_stream,
    )

    paths = [
        Path(__file__),
        *(
            Path(module.__file__)
            for module in (
                native_checkpoint_compatibility,
                native_memory_profile,
                rolling_indicator_comparison,
                warm_stream,
            )
        ),
    ]
    return {
        path.name: {
            "path": str(path.resolve()),
            "sha256": native_checkpoint_compatibility.file_hash(path),
        }
        for path in paths
    }


def _machine() -> dict:
    import numpy as np
    import pyarrow as pa

    return {
        "platform": platform.platform(),
        "hostname": platform.node(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "thread_environment": {name: os.environ.get(name) for name in THREAD_SETTINGS},
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "power_mode": "not observable by this WSL instrument",
    }


async def _write_json(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    await asyncio.to_thread(path.write_text, encoded)


def _preload_record(preload: dict) -> dict:
    return {key: value for key, value in preload.items() if key != "schema"}


async def run(args, release: dict, declared: dict) -> dict:
    from benchmarks.native_checkpoint_compatibility import file_hash
    from benchmarks.rolling_indicator_comparison import _native_program
    from benchmarks.warm_stream import ScenarioConfig
    from calc_flow import (
        EdgeBudget,
        ManagedCheckpointRuntime,
        Runtime,
        SinkBinding,
        SourceBinding,
        SourceProvidedWatermarks,
        StreamingRunner,
        StreamRuntimeConfig,
    )

    output = args.output.resolve()
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    config = ScenarioConfig(history_rows=HISTORY_ROWS)
    runtime = Runtime()
    program = _native_program(
        config.window, indicator=config.indicator, fast_window=config.fast_window
    )
    project = program.to_project(runtime, mode="stream").model_dump(mode="json")
    await _write_json(output / "project.json", project)
    plan = program.compile_stream(runtime)
    fingerprint = plan.fingerprint
    source, sink = BoundedSource(), DelayedSink()
    memory = MemoryProbe(enabled=args.mode == "memory")
    identity = {
        "contract": "calc-flow-p7-slow-sink-v1",
        "role": args.role,
        "mode": args.mode,
        "release": release,
        "declared_builds": declared,
        "instrument": _instrument(),
        "workload": _workload(config),
        "graph_fingerprint": fingerprint,
        "program_fingerprint": program.fingerprint,
        "machine": _machine(),
        "statistical_scope": "one fresh process; raw descriptive diagnostics only",
    }
    await _write_json(output / "declaration.json", identity)
    job = None
    completed = False
    try:
        job = await StreamingRunner(
            plan,
            {
                "input": SourceBinding(
                    source, watermark_policy=SourceProvidedWatermarks()
                )
            },
            {"output": [SinkBinding.ordinary("profile", sink)]},
            ManagedCheckpointRuntime(output / "state"),
            config=StreamRuntimeConfig(
                checkpoint_interval=timedelta(hours=24),
                edge_budget=EdgeBudget(max_rows=CHUNK_ROWS, max_bytes=EDGE_BYTES),
            ),
        ).start_async()
        stream = _StreamHandles(source, sink, job)
        await asyncio.wait_for(source.waiting.wait(), timeout=TIMEOUT_SECONDS)
        preload = await _preload(config, source, sink, job)
        gc.collect()
        await memory.capture(
            "ready_preloaded",
            job,
            caller_prepared_input_rows=0,
            collected_output_rows=0,
            finalized_history_input_rows=config.history_rows,
            configured_entities=config.entities,
            configured_window_rows=config.window,
            diagnostic_metadata="preload transport descriptors remain owned",
        )
        memory.start()
        burst = await _burst(config, stream, preload["schema"], memory, output)
        gc.collect()
        await memory.capture(
            "drained_input_output_released",
            job,
            caller_prepared_input_rows=0,
            collected_output_rows=0,
            source_queue_events=source.events.qsize(),
            sink_queue_tables=sink.tables.qsize(),
            retained_runtime_state="live job; RSS cannot isolate its rolling heap",
        )
        terminal = await finish(
            stream,
            rows=config.history_rows + sum(APPENDS),
            batches=preload["physical_data_batches"] + burst["physical_data_batches"],
        )
        completed = True
        await memory.capture(
            "terminal",
            job,
            caller_prepared_input_rows=0,
            collected_output_rows=0,
            source_queue_events=source.events.qsize(),
            sink_queue_tables=sink.tables.qsize(),
            checkpoint_work="included in EOF-to-completion boundary",
            job_handle="still owned while the terminal status is sampled",
        )
        await memory.stop()
        result = {
            **identity,
            "passed": True,
            "preload": _preload_record(preload),
            "burst": burst,
            "finish": terminal,
            "memory": memory.result(),
            "artifacts": {
                name: await asyncio.to_thread(file_hash, output / name)
                for name in ("project.json", "input.arrow", "output.arrow")
            },
        }
        await _write_json(output / "result.json", result)
        return result
    except BaseException as error:
        await _write_json(
            output / "failure.json",
            {
                **identity,
                "passed": False,
                "error": repr(error),
                "status": None if job is None else job.status(),
            },
        )
        raise
    finally:
        sink.release()
        try:
            if job is not None and not completed:
                await asyncio.wait_for(job.cancel_async(), timeout=TIMEOUT_SECONDS)
        finally:
            await memory.stop()


def _read_result(role: str, directory: Path) -> dict:
    from benchmarks.native_checkpoint_compatibility import file_hash

    result = json.loads((directory / "result.json").read_text())
    if result["role"] != role or result["passed"] is not True:
        raise ValueError("incorrect role or failed diagnostic")
    peer_role = ROLES[1 - ROLES.index(role)]
    declared = release_pair(
        role, result["declared_builds"][role], result["declared_builds"][peer_role]
    )
    require_release(role, result["release"], declared)
    for name, digest in result["artifacts"].items():
        if file_hash(directory / name) != digest:
            raise ValueError(f"diagnostic artifact changed: {role}/{name}")
    return result


def _check_comparison_identity(first: dict, second: dict) -> None:
    for field in (
        "contract",
        "mode",
        "workload",
        "declared_builds",
        "graph_fingerprint",
        "program_fingerprint",
        "machine",
    ):
        if first[field] != second[field]:
            raise ValueError(f"candidate/control diagnostic {field} changed")
    for name in ("project.json", "input.arrow"):
        if first["artifacts"][name] != second["artifacts"][name]:
            raise ValueError(f"candidate/control {name} differs")
    instruments = [
        {name: value["sha256"] for name, value in result["instrument"].items()}
        for result in (first, second)
    ]
    if instruments[0] != instruments[1]:
        raise ValueError("candidate/control instrument source changed")


def compare(candidate: Path, control: Path) -> dict:
    from benchmarks.native_checkpoint_compatibility import file_hash
    from scripts import measure_performance_plan

    records = {}
    for role, directory in (("candidate", candidate), ("control", control)):
        records[role] = _read_result(role, directory)
    first, second = records["candidate"], records["control"]
    _check_comparison_identity(first, second)
    equivalence = measure_performance_plan.compare_outputs(
        str(candidate / "output.arrow"), str(control / "output.arrow")
    )
    return {
        "passed": True,
        "mode": first["mode"],
        "graph_fingerprint": first["graph_fingerprint"],
        "output_equivalence": equivalence,
        "statistical_scope": "single-run descriptive evidence; no regression verdict",
        "comparison_helper_sha256": file_hash(Path(measure_performance_plan.__file__)),
        "runs": {
            role: {
                "result": str(directory.resolve() / "result.json"),
                "result_sha256": file_hash(directory / "result.json"),
                "native_sha256": records[role]["release"]["native_sha256"],
                "appends": records[role]["burst"]["appends"],
                "eof_enqueue_to_completed_ns": records[role]["finish"][
                    "eof_enqueue_to_completed_ns"
                ],
                "memory_summary": (
                    None
                    if records[role]["memory"] is None
                    else memory_summary(records[role]["memory"]["samples"])
                ),
            }
            for role, directory in (("candidate", candidate), ("control", control))
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--role", choices=ROLES, required=True)
    run_parser.add_argument("--build", type=Path, required=True)
    run_parser.add_argument("--peer-build", type=Path, required=True)
    run_parser.add_argument("--mode", choices=("timing", "memory"), required=True)
    run_parser.add_argument("--output", type=Path, required=True)
    compare_parser = commands.add_parser("compare")
    compare_parser.add_argument("--candidate", type=Path, required=True)
    compare_parser.add_argument("--control", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "compare":
        result = compare(args.candidate, args.control)
        with args.output.open("x") as target:
            target.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    else:
        from benchmarks.native_checkpoint_compatibility import release_identity

        os.environ.update(thread_settings(os.environ))
        declared = release_pair(
            args.role,
            json.loads(args.build.read_text()),
            json.loads(args.peer_build.read_text()),
        )
        proof = release_identity(args.build)
        require_release(args.role, proof, declared)
        release = {
            **{
                key: value for key, value in proof.items() if key != "instrument_sha256"
            },
            "release_proof_helper_sha256": proof["instrument_sha256"],
        }
        result = asyncio.run(run(args, release, declared))
    print(json.dumps({"passed": result["passed"], "mode": result["mode"]}))


if __name__ == "__main__":
    main()
