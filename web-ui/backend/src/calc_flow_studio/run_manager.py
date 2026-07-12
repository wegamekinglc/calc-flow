from __future__ import annotations

import base64
import binascii
import json
import math
import multiprocessing
import os
import queue
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from io import BytesIO
from pathlib import Path
from threading import Condition, RLock, Thread
from time import monotonic
from typing import Any
from uuid import uuid4

import cloudpickle
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json
from calc_flow.batch import Batch, BatchKind, BatchMetadata, JSONValue
from calc_flow.config import (
    InputFormat,
    ProjectConfig,
    RunOptions,
    compile_project,
)
from calc_flow.udf import UdfRegistry, UdfRegistrySnapshot

from calc_flow_studio.models import (
    InputPayload,
    RunEvent,
    RunRequest,
    RunResponse,
    RunStatus,
)


class RunManagerError(RuntimeError):
    """Raised when a preview run cannot be prepared or managed."""


def _json_size(value: JSONValue) -> int:
    if isinstance(value, str):
        return len(value.encode())
    return len(json.dumps(value, allow_nan=False, separators=(",", ":")).encode())


def _records_table(value: JSONValue) -> pa.Table:
    if isinstance(value, list):
        if not all(isinstance(record, dict) for record in value):
            msg = "JSON table input must be a list of record objects"
            raise RunManagerError(msg)
        return pa.Table.from_pylist(value)
    if isinstance(value, dict):
        if value and all(isinstance(item, list) for item in value.values()):
            return pa.table(value)
        return pa.Table.from_pylist([value])
    msg = "JSON table input must contain records or a column mapping"
    raise RunManagerError(msg)


def _decode_table(payload: InputPayload, *, max_bytes: int) -> tuple[pa.Table, int]:
    size = _json_size(payload.data)
    if size > max_bytes:
        msg = f"input exceeds the {max_bytes} byte preview limit"
        raise RunManagerError(msg)

    if payload.format is InputFormat.INLINE_JSON:
        return _records_table(payload.data), size
    if payload.format is InputFormat.CSV:
        if not isinstance(payload.data, str):
            msg = "CSV input data must be text"
            raise RunManagerError(msg)
        raw = payload.data.encode()
        return pa_csv.read_csv(pa.BufferReader(raw)), len(raw)
    if payload.format is InputFormat.JSON:
        if not isinstance(payload.data, str):
            return _records_table(payload.data), size
        raw = payload.data.encode()
        try:
            decoded = json.loads(payload.data)
        except json.JSONDecodeError:
            try:
                return pa_json.read_json(pa.BufferReader(raw)), len(raw)
            except pa.ArrowInvalid as error:
                msg = (
                    "JSON input must be an array, object, or newline-delimited records"
                )
                raise RunManagerError(msg) from error
        return _records_table(decoded), len(raw)
    if payload.format is InputFormat.ARROW_IPC:
        if not isinstance(payload.data, str):
            msg = "Arrow IPC input data must be base64 text"
            raise RunManagerError(msg)
        try:
            raw = base64.b64decode(payload.data, validate=True)
        except (binascii.Error, ValueError) as error:
            msg = "Arrow IPC input is not valid base64"
            raise RunManagerError(msg) from error
        if len(raw) > max_bytes:
            msg = f"input exceeds the {max_bytes} byte preview limit"
            raise RunManagerError(msg)
        try:
            reader = pa.ipc.open_stream(BytesIO(raw))
        except pa.ArrowInvalid:
            try:
                reader = pa.ipc.open_file(BytesIO(raw))
            except pa.ArrowInvalid as error:
                msg = "Arrow IPC input is neither a stream nor file"
                raise RunManagerError(msg) from error
        return reader.read_all(), len(raw)
    msg = f"unsupported input format {payload.format.value!r}"
    raise RunManagerError(msg)


def _saved_inputs(project: ProjectConfig) -> dict[str, InputPayload]:
    return {
        source.input_name: InputPayload(
            format=source.format,
            data=source.data,
            source_id=source.source_id or source.id,
        )
        for source in project.data_sources
    }


def prepare_run(
    project: ProjectConfig,
    request: RunRequest,
    *,
    udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
) -> tuple[dict[str, Batch], RunOptions]:
    """Decode bounded request inputs and verify them against the compiled graph."""
    plan = compile_project(project, udf_registry=udf_registry)
    options = request.options or project.run_options
    payloads = request.inputs or _saved_inputs(project)
    if not payloads:
        msg = "run requires request inputs or saved project data sources"
        raise RunManagerError(msg)
    if set(payloads) != set(plan.graph_inputs):
        msg = (
            f"run inputs must be {sorted(plan.graph_inputs)}; "
            f"received {sorted(payloads)}"
        )
        raise RunManagerError(msg)

    batches: dict[str, Batch] = {}
    total_bytes = 0
    for input_name, payload in payloads.items():
        endpoint = plan.graph_inputs[input_name]
        node = next(node for node in plan.nodes if node.node_id == endpoint.node_id)
        port = next(
            port for port in node.operator.input_ports if port.name == endpoint.port
        )
        if port.kind is not BatchKind.TABLE:
            msg = "web preview currently accepts table graph inputs only"
            raise RunManagerError(msg)
        table, size = _decode_table(payload, max_bytes=options.max_input_bytes)
        total_bytes += size
        if total_bytes > options.max_input_bytes:
            msg = f"combined inputs exceed the {options.max_input_bytes} byte limit"
            raise RunManagerError(msg)
        if table.num_rows > options.max_rows:
            msg = f"input exceeds the {options.max_rows} row preview limit"
            raise RunManagerError(msg)
        if port.schema is not None and not table.schema.equals(
            port.schema, check_metadata=True
        ):
            try:
                table = table.cast(port.schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                msg = f"input {input_name!r} does not match its declared Arrow schema"
                raise RunManagerError(msg) from error
        batch = Batch.table(
            table,
            metadata=BatchMetadata(source_id=payload.source_id or input_name),
        )
        port.validate(batch, endpoint=f"graph input {input_name!r}")
        batches[input_name] = batch
    return batches, options


def _json_safe(value: Any) -> JSONValue:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode()
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        return _json_safe(item())
    return str(value)


def _result_payload(result: Any, *, output_rows: int) -> dict[str, JSONValue]:
    outputs: dict[str, JSONValue] = {}
    for name, batch in result.outputs.items():
        if batch.kind is BatchKind.TABLE:
            table = batch.table_payload
            outputs[name] = {
                "kind": "table",
                "total_rows": table.num_rows,
                "truncated": table.num_rows > output_rows,
                "schema": [
                    {
                        "name": field.name,
                        "type": str(field.type),
                        "nullable": field.nullable,
                    }
                    for field in table.schema
                ],
                "rows": _json_safe(table.slice(0, output_rows).to_pylist()),
            }
        else:
            array = batch.array_payload
            shape = getattr(array, "shape", ())
            limited = array[:output_rows] if shape else array
            outputs[name] = {
                "kind": "array",
                "total_rows": batch.num_rows,
                "truncated": batch.num_rows > output_rows,
                "data": _json_safe(limited.tolist()),
            }

    return {
        "outputs": outputs,
        "warnings": list(result.warnings),
        "node_timings": {
            node_id: {
                "duration_ns": timing.duration_ns,
                "input_rows": dict(timing.input_rows),
                "output_rows": dict(timing.output_rows),
            }
            for node_id, timing in result.node_timings.items()
        },
        "datafusion_metrics": [
            {
                "node_id": metric.node_id,
                "planning_ns": metric.planning_ns,
                "execution_ns": metric.execution_ns,
                "output_rows": metric.output_rows,
                "logical_plan": metric.logical_plan,
                "physical_plan": metric.physical_plan,
            }
            for metric in result.datafusion_metrics
        ],
        "metadata": {
            "run_id": result.metadata.run_id,
            "pipeline_name": result.metadata.pipeline_name,
            "pipeline_fingerprint": result.metadata.pipeline_fingerprint,
            "started_at": result.metadata.started_at.isoformat(),
            "finished_at": result.metadata.finished_at.isoformat(),
        },
    }


def _apply_resource_limits(options: RunOptions) -> None:
    try:
        import resource
    except ImportError:
        return
    cpu_seconds = max(1, math.ceil(options.timeout_seconds))
    with suppress(OSError, ValueError):
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))


def _resident_bytes(worker: Any) -> int | None:
    pid = getattr(worker, "pid", None)
    if pid is None:
        return None
    try:
        statm = (Path(f"/proc/{pid}/statm")).read_text(encoding="ascii").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _execute_worker(
    worker_payload: bytes,
    output_queue: Any,
    apply_limits: bool,
) -> None:
    try:
        project_data, batches, options_data, registry = cloudpickle.loads(
            worker_payload
        )
        project = ProjectConfig.model_validate(project_data)
        options = RunOptions.model_validate(options_data)
        if apply_limits:
            _apply_resource_limits(options)
        plan = compile_project(project, udf_registry=registry)
        deadline = datetime.now(UTC) + timedelta(seconds=options.timeout_seconds)
        result = plan.execute(batches, deadline=deadline)
        output_queue.put(
            {
                "ok": True,
                "result": _result_payload(result, output_rows=options.output_rows),
            }
        )
    except BaseException as error:
        output_queue.put(
            {
                "ok": False,
                "error": f"{type(error).__name__}: {error}"[:4000],
            }
        )


@dataclass(slots=True)
class _RunHandle:
    id: str
    project_id: str
    status: RunStatus
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: str | None = None
    result: dict[str, JSONValue] | None = None
    events: list[RunEvent] = field(default_factory=list)
    worker: Any = None
    output_queue: Any = None
    cancel_requested: bool = False


class RunManager:
    """Manage bounded preview workers and retain their local run records."""

    def __init__(
        self,
        *,
        udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
        use_processes: bool = True,
        max_workers: int = 2,
        max_history: int = 100,
    ) -> None:
        if max_workers <= 0 or max_history <= 0:
            msg = "max_workers and max_history must be greater than 0"
            raise ValueError(msg)
        if isinstance(udf_registry, UdfRegistry):
            self._registry = udf_registry.snapshot()
        else:
            self._registry = udf_registry or UdfRegistrySnapshot()
        self._lock = RLock()
        self._event_condition = Condition(self._lock)
        self._runs: dict[str, _RunHandle] = {}
        self._use_processes = use_processes
        self._max_workers = max_workers
        self._max_history = max_history
        if use_processes:
            self._process_context = multiprocessing.get_context("spawn")
        else:
            self._process_context = None

    @property
    def udf_registry(self) -> UdfRegistrySnapshot:
        return self._registry

    def submit(self, project: ProjectConfig, request: RunRequest) -> RunResponse:
        # Reserve the worker slot atomically with the capacity check. If the
        # check and the handle insertion were separate critical sections, two
        # concurrent submissions could both pass the check before either
        # registered its run and exceed max_workers. prepare_run and worker
        # construction run outside the lock; on failure we release the slot.
        with self._lock:
            active = sum(
                handle.status in {RunStatus.PENDING, RunStatus.RUNNING}
                for handle in self._runs.values()
            )
            if active >= self._max_workers:
                msg = "all local preview workers are busy"
                raise RunManagerError(msg)
            run_id = uuid4().hex
            handle = _RunHandle(
                id=run_id,
                project_id=project.id,
                status=RunStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            self._runs[run_id] = handle
            self._event(handle, "created", "Run accepted")
        try:
            batches, options = prepare_run(
                project, request, udf_registry=self._registry
            )
            worker_payload = cloudpickle.dumps(
                (
                    project.model_dump(mode="json", by_alias=True),
                    batches,
                    options.model_dump(mode="json"),
                    self._registry,
                )
            )
            if self._use_processes:
                output_queue = self._process_context.Queue(maxsize=1)
                worker = self._process_context.Process(
                    target=_execute_worker,
                    args=(
                        worker_payload,
                        output_queue,
                        True,
                    ),
                    daemon=True,
                    name=f"calc-flow-{run_id[:8]}",
                )
            else:
                output_queue = queue.Queue(maxsize=1)
                worker = Thread(
                    target=_execute_worker,
                    args=(
                        worker_payload,
                        output_queue,
                        False,
                    ),
                    daemon=True,
                    name=f"calc-flow-{run_id[:8]}",
                )
        except BaseException:
            with self._lock:
                self._runs.pop(run_id, None)
            raise
        with self._lock:
            self._prune_history()
            handle.worker = worker
            handle.output_queue = output_queue
            handle.status = RunStatus.RUNNING
            handle.started_at = datetime.now(UTC)
            self._event(handle, "running", "Worker started")
        worker.start()
        Thread(
            target=self._monitor,
            args=(run_id, options),
            daemon=True,
            name=f"calc-flow-monitor-{run_id[:8]}",
        ).start()
        return self.get(run_id)

    def get(self, run_id: str) -> RunResponse:
        with self._lock:
            handle = self._require(run_id)
            return self._response(handle)

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._require(run_id).events)

    def wait_for_events(
        self,
        run_id: str,
        *,
        after_sequence: int,
        timeout: float,
    ) -> tuple[tuple[RunEvent, ...], RunStatus]:
        terminal = {
            RunStatus.COMPLETED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
            RunStatus.TIMED_OUT,
        }

        def ready() -> bool:
            handle = self._require(run_id)
            return (
                any(event.sequence > after_sequence for event in handle.events)
                or handle.status in terminal
            )

        with self._event_condition:
            self._event_condition.wait_for(ready, timeout=timeout)
            handle = self._require(run_id)
            events = tuple(
                event for event in handle.events if event.sequence > after_sequence
            )
            return events, handle.status

    def cancel(self, run_id: str) -> RunResponse:
        with self._lock:
            handle = self._require(run_id)
            if handle.status in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
            }:
                return self._response(handle)
            handle.cancel_requested = True
            self._stop_worker(handle)
            handle.status = RunStatus.CANCELLED
            handle.finished_at = datetime.now(UTC)
            self._event(handle, "cancelled", "Run cancelled")
            return self._response(handle)

    def shutdown(self) -> None:
        with self._lock:
            for handle in self._runs.values():
                if handle.status in {RunStatus.PENDING, RunStatus.RUNNING}:
                    handle.cancel_requested = True
                    self._stop_worker(handle)
                    handle.status = RunStatus.CANCELLED
                    handle.finished_at = datetime.now(UTC)
                    self._event(handle, "cancelled", "Server shut down")

    def _monitor(self, run_id: str, options: RunOptions) -> None:
        started = monotonic()
        while True:
            with self._lock:
                handle = self._require(run_id)
                if handle.cancel_requested:
                    return
                output_queue = handle.output_queue
                worker = handle.worker
            resident_bytes = _resident_bytes(worker) if self._use_processes else None
            if (
                resident_bytes is not None
                and resident_bytes > options.memory_limit_mb * 1024 * 1024
            ):
                with self._lock:
                    handle = self._require(run_id)
                    if handle.status is RunStatus.RUNNING:
                        self._stop_worker(handle)
                        handle.status = RunStatus.FAILED
                        handle.finished_at = datetime.now(UTC)
                        handle.error = "run exceeded its preview memory limit"
                        self._event(handle, "failed", handle.error)
                return
            remaining = options.timeout_seconds - (monotonic() - started)
            if remaining <= 0:
                with self._lock:
                    handle = self._require(run_id)
                    if handle.status is RunStatus.RUNNING:
                        self._stop_worker(handle)
                        handle.status = RunStatus.TIMED_OUT
                        handle.finished_at = datetime.now(UTC)
                        handle.error = "run exceeded its preview timeout"
                        self._event(handle, "timed_out", handle.error)
                return
            try:
                message = output_queue.get(timeout=min(0.05, remaining))
            except queue.Empty:
                worker = handle.worker
                if not worker.is_alive():
                    with self._lock:
                        handle = self._require(run_id)
                        if handle.status is RunStatus.RUNNING:
                            handle.status = RunStatus.FAILED
                            handle.finished_at = datetime.now(UTC)
                            handle.error = "worker exited without a result"
                            self._event(handle, "failed", handle.error)
                    return
                continue

            with self._lock:
                handle = self._require(run_id)
                if handle.cancel_requested:
                    return
                handle.finished_at = datetime.now(UTC)
                if message.get("ok"):
                    handle.status = RunStatus.COMPLETED
                    handle.result = message["result"]
                    self._event(handle, "completed", "Run completed")
                else:
                    handle.status = RunStatus.FAILED
                    handle.error = str(message.get("error", "run failed"))
                    self._event(handle, "failed", handle.error)
                self._join_worker(handle)
            return

    def _event(self, handle: _RunHandle, event_type: str, message: str) -> None:
        handle.events.append(
            RunEvent(
                sequence=len(handle.events),
                timestamp=datetime.now(UTC),
                type=event_type,
                message=message,
            )
        )
        self._event_condition.notify_all()

    def _prune_history(self) -> None:
        if len(self._runs) < self._max_history:
            return
        terminal = [
            handle
            for handle in self._runs.values()
            if handle.status
            in {
                RunStatus.COMPLETED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
                RunStatus.TIMED_OUT,
            }
        ]
        terminal.sort(key=lambda handle: handle.created_at)
        for handle in terminal[: max(0, len(self._runs) - self._max_history + 1)]:
            self._runs.pop(handle.id, None)

    def _require(self, run_id: str) -> _RunHandle:
        try:
            return self._runs[run_id]
        except KeyError as error:
            raise KeyError(f"run {run_id!r} does not exist") from error

    @staticmethod
    def _response(handle: _RunHandle) -> RunResponse:
        return RunResponse(
            id=handle.id,
            project_id=handle.project_id,
            status=handle.status,
            created_at=handle.created_at,
            started_at=handle.started_at,
            finished_at=handle.finished_at,
            error=handle.error,
            result=handle.result,
        )

    def _stop_worker(self, handle: _RunHandle) -> None:
        worker = handle.worker
        if self._use_processes and worker is not None and worker.is_alive():
            worker.terminate()
            worker.join(timeout=1)

    def _join_worker(self, handle: _RunHandle) -> None:
        worker = handle.worker
        if worker is not None:
            worker.join(timeout=1)
