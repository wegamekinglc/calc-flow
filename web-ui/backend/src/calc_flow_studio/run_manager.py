from __future__ import annotations

import base64
import binascii
import json
import math
import multiprocessing
import os
import pickle
import queue
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from decimal import Decimal
from enum import Enum
from io import BytesIO
from pathlib import Path
from threading import Condition, RLock, Thread, current_thread
from time import monotonic
from typing import Any
from uuid import uuid4

import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.json as pa_json
from calc_flow import Batch, ProjectDocument, Runtime, register_jax, register_numpy

from calc_flow_studio.models import (
    JSONValue,
    RunEvent,
    RunOptions,
    RunRequest,
    RunResponse,
    RunStatus,
)

type PreparedInput = tuple[pa.Table, dict[str, JSONValue]]
type PreparedInputs = dict[str, PreparedInput]


class RunManagerError(RuntimeError):
    """Raised when a preview run cannot be prepared or managed."""


def _json_size(value: JSONValue) -> int:
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    return len(
        json.dumps(value, allow_nan=False, separators=(",", ":")).encode("utf-8")
    )


def _records_table(value: object, *, columns_only: bool = False) -> pa.Table:
    try:
        if isinstance(value, dict):
            if columns_only and all(type(item) is list for item in value.values()):
                return pa.table(value)
            if value and all(type(item) is list for item in value.values()):
                return pa.table(value)
            if columns_only:
                raise RunManagerError("columns input must be an object of column lists")
            return pa.Table.from_pylist([value])
        if not columns_only and isinstance(value, list):
            if not all(isinstance(record, dict) for record in value):
                raise RunManagerError("records input must be a list of record objects")
            return pa.Table.from_pylist(value)
    except (pa.ArrowException, TypeError, ValueError) as error:
        raise RunManagerError(
            f"input could not be converted to an Arrow table: {error}"
        ) from error
    if columns_only:
        raise RunManagerError("columns input must be an object of column lists")
    raise RunManagerError("records input must be a list of record objects")


def _arrow_reader(raw: bytes) -> pa.RecordBatchReader | pa.ipc.RecordBatchFileReader:
    try:
        return pa.ipc.open_stream(BytesIO(raw))
    except pa.ArrowInvalid:
        try:
            return pa.ipc.open_file(BytesIO(raw))
        except pa.ArrowInvalid as error:
            raise RunManagerError(
                "Arrow IPC input is neither a stream nor file"
            ) from error


def _decode_source(
    format: str,
    data: object,
    *,
    max_bytes: int,
) -> tuple[pa.Table, int, int]:
    """Decode one saved/request source and return table and both size domains."""
    if format == "records":
        encoded_size = _json_size(data)  # type: ignore[arg-type]
        if encoded_size > max_bytes:
            raise RunManagerError(f"encoded input exceeds the {max_bytes} byte limit")
        table = _records_table(data)
    elif format == "columns":
        encoded_size = _json_size(data)  # type: ignore[arg-type]
        if encoded_size > max_bytes:
            raise RunManagerError(f"encoded input exceeds the {max_bytes} byte limit")
        table = _records_table(data, columns_only=True)
    elif format == "inline_json":
        encoded_size = _json_size(data)  # type: ignore[arg-type]
        if encoded_size > max_bytes:
            raise RunManagerError(f"encoded input exceeds the {max_bytes} byte limit")
        table = _records_table(data)
    elif format == "csv":
        if not isinstance(data, str):
            raise RunManagerError("CSV input data must be text")
        raw = data.encode("utf-8")
        encoded_size = len(raw)
        if encoded_size > max_bytes:
            raise RunManagerError(f"encoded input exceeds the {max_bytes} byte limit")
        try:
            table = pa_csv.read_csv(pa.BufferReader(raw))
        except pa.ArrowException as error:
            raise RunManagerError(f"CSV input is invalid: {error}") from error
    elif format == "json":
        if not isinstance(data, str):
            encoded_size = _json_size(data)  # type: ignore[arg-type]
            if encoded_size > max_bytes:
                raise RunManagerError(
                    f"encoded input exceeds the {max_bytes} byte limit"
                )
            table = _records_table(data)
        else:
            raw = data.encode("utf-8")
            encoded_size = len(raw)
            if encoded_size > max_bytes:
                raise RunManagerError(
                    f"encoded input exceeds the {max_bytes} byte limit"
                )
            try:
                decoded = json.loads(data)
            except json.JSONDecodeError:
                try:
                    table = pa_json.read_json(pa.BufferReader(raw))
                except pa.ArrowException as error:
                    raise RunManagerError(
                        "JSON input must be an array, object, or "
                        "newline-delimited records"
                    ) from error
            else:
                table = _records_table(decoded)
    elif format == "arrow_ipc":
        if not isinstance(data, str):
            raise RunManagerError("Arrow IPC input data must be base64 text")
        encoded_size = len(data.encode("ascii", errors="ignore"))
        if encoded_size > max_bytes:
            raise RunManagerError(f"encoded input exceeds the {max_bytes} byte limit")
        try:
            raw = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise RunManagerError("Arrow IPC input is not valid base64") from error
        if len(raw) > max_bytes:
            raise RunManagerError(f"decoded input exceeds the {max_bytes} byte limit")
        table = _arrow_reader(raw).read_all()
    else:
        raise RunManagerError(f"unsupported input format {format!r}")

    table = table.combine_chunks()
    decoded_size = table.nbytes
    if decoded_size > max_bytes:
        raise RunManagerError(f"decoded input exceeds the {max_bytes} byte limit")
    return table, encoded_size, decoded_size


_ARROW_TYPES: dict[str, pa.DataType] = {
    "bool": pa.bool_(),
    "date32": pa.date32(),
    "date64": pa.date64(),
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "large_string": pa.large_string(),
    "string": pa.string(),
    "time32[s]": pa.time32("s"),
    "time64[us]": pa.time64("us"),
    "timestamp[ms]": pa.timestamp("ms"),
    "timestamp[us]": pa.timestamp("us"),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
}


def _default_input_ports(node: dict[str, JSONValue]) -> list[dict[str, JSONValue]]:
    configured = node.get("input_ports", [])
    if isinstance(configured, list) and configured:
        return [port for port in configured if isinstance(port, dict)]
    operator = node.get("operator", {})
    if not isinstance(operator, dict):
        return []
    kind = operator.get("kind")
    if kind == "expression":
        return [{"kind": "table", "name": "input", "required": True, "schema": []}]
    if kind == "sql":
        aliases = operator.get("aliases", [])
        if isinstance(aliases, list):
            return [
                {"kind": "table", "name": alias, "required": True, "schema": []}
                for alias in aliases
                if isinstance(alias, str)
            ]
    return []


def _input_contracts(project: ProjectDocument) -> dict[str, dict[str, JSONValue]]:
    root = project.root
    pipeline = root["pipeline"]
    assert isinstance(pipeline, dict)
    nodes = pipeline["nodes"]
    edges = pipeline.get("edges", [])
    assert isinstance(nodes, list) and isinstance(edges, list)
    connected = {
        (edge["target_node"], edge.get("target_port", "input"))
        for edge in edges
        if isinstance(edge, dict)
    }
    endpoints: list[tuple[str, str, dict[str, JSONValue]]] = []
    for node in nodes:
        assert isinstance(node, dict)
        node_id = node["id"]
        assert isinstance(node_id, str)
        for port in _default_input_ports(node):
            name = port.get("name")
            if isinstance(name, str) and (node_id, name) not in connected:
                endpoints.append((node_id, name, port))
    counts: dict[str, int] = {}
    for _, port_name, _ in endpoints:
        counts[port_name] = counts.get(port_name, 0) + 1
    return {
        port_name if counts[port_name] == 1 else f"{node_id}.{port_name}": port
        for node_id, port_name, port in endpoints
    }


def _declared_schema(port: dict[str, JSONValue]) -> pa.Schema | None:
    configured = port.get("schema", [])
    if not isinstance(configured, list) or not configured:
        return None
    fields = []
    for field in configured:
        assert isinstance(field, dict)
        fields.append(
            pa.field(
                str(field["name"]),
                _ARROW_TYPES[str(field["data_type"])],
                nullable=bool(field.get("nullable", True)),
            )
        )
    return pa.schema(fields)


def _saved_inputs(project: ProjectDocument) -> dict[str, tuple[str, object, str]]:
    sources = project.root.get("data_sources", [])
    assert isinstance(sources, list)
    return {
        str(source["input"]): (
            str(source["format"]),
            source["data"],
            str(source["id"]),
        )
        for source in sources
        if isinstance(source, dict)
    }


def prepare_run(
    project: ProjectDocument,
    request: RunRequest,
) -> tuple[PreparedInputs, RunOptions]:
    """Prepare plain, bounded worker inputs without constructing PyO3 objects."""
    root_options = project.root.get("run_options", {})
    assert isinstance(root_options, dict)
    options = request.options or RunOptions.model_validate(root_options)
    contracts = _input_contracts(project)
    saved = _saved_inputs(project)
    if request.inputs:
        payloads = {
            name: (payload.format, payload.data, payload.source_id or name)
            for name, payload in request.inputs.items()
        }
    else:
        payloads = saved
    if not payloads:
        raise RunManagerError(
            "run requires request inputs or saved project data sources"
        )
    if set(payloads) != set(saved) or set(saved) != set(contracts):
        raise RunManagerError(
            f"run inputs must be {sorted(saved)}; received {sorted(payloads)}"
        )

    prepared: PreparedInputs = {}
    total_encoded = 0
    total_decoded = 0
    total_rows = 0
    for input_name in sorted(payloads):
        port = contracts[input_name]
        if port.get("kind") != "table":
            raise RunManagerError(
                "web preview currently accepts table graph inputs only"
            )
        format, data, source_id = payloads[input_name]
        table, encoded_size, decoded_size = _decode_source(
            format, data, max_bytes=options.max_input_bytes
        )
        total_encoded += encoded_size
        if total_encoded > options.max_input_bytes:
            raise RunManagerError(
                "combined encoded inputs exceed the "
                f"{options.max_input_bytes} byte limit"
            )
        total_rows += table.num_rows
        if total_rows > options.max_rows:
            raise RunManagerError(
                f"combined inputs exceed the {options.max_rows} row preview limit"
            )
        schema = _declared_schema(port)
        if schema is not None and not table.schema.equals(schema, check_metadata=True):
            try:
                table = table.cast(schema)
            except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as error:
                raise RunManagerError(
                    f"input {input_name!r} does not match its declared Arrow schema"
                ) from error
            decoded_size = table.nbytes
            if decoded_size > options.max_input_bytes:
                raise RunManagerError(
                    f"decoded input exceeds the {options.max_input_bytes} byte limit"
                )
        total_decoded += decoded_size
        if total_decoded > options.max_input_bytes:
            raise RunManagerError(
                "combined decoded inputs exceed the "
                f"{options.max_input_bytes} byte limit"
            )
        prepared[input_name] = (table, {"source_id": str(source_id)})
    return prepared, options


def _json_safe(value: Any) -> JSONValue:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime | date | time):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
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
        if batch.kind == "table":
            table = batch.to_pyarrow()
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
                "metadata": _json_safe(batch.metadata),
            }
        else:
            array = batch.array
            limited = array[:output_rows] if batch.num_rows > output_rows else array
            tolist = getattr(limited, "tolist", None)
            data = tolist() if callable(tolist) else limited
            outputs[name] = {
                "backend": batch.backend,
                "kind": "array",
                "total_rows": batch.num_rows,
                "truncated": batch.num_rows > output_rows,
                "data": _json_safe(data),
                "metadata": _json_safe(batch.metadata),
            }
    return {
        "outputs": outputs,
        "node_timings": _json_safe(result.node_timings),
        "datafusion_metrics": _json_safe(result.datafusion_metrics),
        "metadata": _json_safe(result.metadata),
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
        statm = Path(f"/proc/{pid}/statm").read_text(encoding="ascii").split()
        return int(statm[1]) * os.sysconf("SC_PAGE_SIZE")
    except (IndexError, OSError, ValueError):
        return None


def _register_referenced_builtins(runtime: Runtime, project_json: str) -> None:
    project = json.loads(project_json)
    nodes = project["pipeline"]["nodes"]
    references = {
        (
            operator.get("provider"),
            operator.get("name"),
            operator.get("version"),
        )
        for node in nodes
        if (operator := node.get("operator", {})).get("kind") == "external"
    }
    if ("numpy", "expression", "1") in references:
        register_numpy(runtime)
    if ("jax", "expression", "1") in references:
        register_jax(runtime)


def _serialize_worker_payload(
    project: ProjectDocument,
    prepared: PreparedInputs,
    options: RunOptions,
) -> bytes:
    return pickle.dumps(
        (
            project.canonical_json(),
            prepared,
            options.model_dump(mode="json"),
        ),
        protocol=pickle.HIGHEST_PROTOCOL,
    )


def _execute_worker(
    worker_payload: bytes, output_queue: Any, apply_limits: bool
) -> None:
    try:
        project_json, prepared, options_data = pickle.loads(worker_payload)
        options = RunOptions.model_validate(options_data)
        if apply_limits:
            _apply_resource_limits(options)
        runtime = Runtime()
        _register_referenced_builtins(runtime, project_json)
        plan = runtime.compile_project(project_json)
        batches = {
            name: Batch.from_pyarrow(table, metadata=metadata)
            for name, (table, metadata) in prepared.items()
        }
        result = plan.execute(batches)
        output_queue.put(
            {
                "ok": True,
                "result": _result_payload(result, output_rows=options.output_rows),
            }
        )
    except BaseException as error:
        with suppress(BaseException):
            output_queue.put(
                {"ok": False, "error": f"{type(error).__name__}: {error}"[:4000]}
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
    events: tuple[RunEvent, ...] = ()
    worker: Any = None
    output_queue: Any = None
    monitor: Thread | None = None
    cancel_requested: bool = False


class RunManager:
    """Manage bounded Rust preview workers and retain their local run records."""

    def __init__(
        self,
        *,
        use_processes: bool = True,
        max_workers: int = 2,
        max_history: int = 100,
    ) -> None:
        if max_workers <= 0 or max_history <= 0:
            raise ValueError("max_workers and max_history must be greater than 0")
        self._lock = RLock()
        self._event_condition = Condition(self._lock)
        self._runs: dict[str, _RunHandle] = {}
        self._use_processes = use_processes
        self._max_workers = max_workers
        self._max_history = max_history
        self._shutdown = False
        self._process_context = (
            multiprocessing.get_context("spawn") if use_processes else None
        )

    def submit(self, project: ProjectDocument, request: RunRequest) -> RunResponse:
        with self._lock:
            if self._shutdown:
                raise RunManagerError("run manager is shut down")
            active = sum(
                handle.status in {RunStatus.PENDING, RunStatus.RUNNING}
                for handle in self._runs.values()
            )
            if active >= self._max_workers:
                raise RunManagerError("all local preview workers are busy")
            run_id = uuid4().hex
            handle = _RunHandle(
                id=run_id,
                project_id=str(project.root["id"]),
                status=RunStatus.PENDING,
                created_at=datetime.now(UTC),
            )
            self._runs[run_id] = handle
            self._event(run_id, "created", "Run accepted")

        output_queue: Any = None
        worker: Any = None
        try:
            prepared, options = prepare_run(project, request)
            worker_payload = _serialize_worker_payload(project, prepared, options)
            if self._use_processes:
                assert self._process_context is not None
                output_queue = self._process_context.Queue(maxsize=1)
                worker = self._process_context.Process(
                    target=_execute_worker,
                    args=(worker_payload, output_queue, True),
                    daemon=True,
                    name=f"calc-flow-{run_id[:8]}",
                )
            else:
                output_queue = queue.Queue(maxsize=1)
                worker = Thread(
                    target=_execute_worker,
                    args=(worker_payload, output_queue, False),
                    daemon=True,
                    name=f"calc-flow-{run_id[:8]}",
                )
        except BaseException:
            self._cleanup_resources(worker, output_queue, terminate=True)
            with self._lock:
                self._runs.pop(run_id, None)
                self._event_condition.notify_all()
            raise

        start_error: BaseException | None = None
        with self._lock:
            handle = self._require(run_id)
            if self._shutdown or handle.cancel_requested:
                self._runs.pop(run_id, None)
                cleanup_immediately = True
            else:
                try:
                    worker.start()
                except BaseException as error:
                    self._runs.pop(run_id, None)
                    self._event_condition.notify_all()
                    start_error = error
                    cleanup_immediately = True
                else:
                    self._prune_history()
                    handle.worker = worker
                    handle.output_queue = output_queue
                    handle.status = RunStatus.RUNNING
                    handle.started_at = datetime.now(UTC)
                    self._event(run_id, "running", "Worker started")
                    monitor = Thread(
                        target=self._monitor,
                        args=(run_id, options),
                        daemon=True,
                        name=f"calc-flow-monitor-{run_id[:8]}",
                    )
                    handle.monitor = monitor
                    cleanup_immediately = False
        if cleanup_immediately:
            self._cleanup_resources(worker, output_queue, terminate=True)
            if start_error is not None:
                raise start_error
            raise RunManagerError("run manager shut down during submission")
        try:
            monitor.start()
        except BaseException:
            with self._lock:
                handle = self._require(run_id)
                worker, output_queue, _ = self._detach_resources(handle)
                self._runs.pop(run_id, None)
                self._event_condition.notify_all()
            self._cleanup_resources(worker, output_queue, terminate=True)
            raise
        return self.get(run_id)

    def get(self, run_id: str) -> RunResponse:
        with self._lock:
            return self._response(self._require(run_id))

    def events(self, run_id: str) -> tuple[RunEvent, ...]:
        with self._lock:
            return tuple(self._require(run_id).events)

    def wait_for_events(
        self, run_id: str, *, after_sequence: int, timeout: float
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
            return (
                tuple(
                    event for event in handle.events if event.sequence > after_sequence
                ),
                handle.status,
            )

    def cancel(self, run_id: str) -> RunResponse:
        with self._lock:
            handle = self._require(run_id)
            if handle.status not in {RunStatus.PENDING, RunStatus.RUNNING}:
                return self._response(handle)
            handle.cancel_requested = True
            handle.status = RunStatus.CANCELLED
            handle.finished_at = datetime.now(UTC)
            self._event(run_id, "cancelled", "Run cancelled")
            worker, output_queue, monitor = self._detach_resources(handle)
            response = self._response(handle)
        self._cleanup_resources(worker, output_queue, terminate=True)
        self._join_monitor(monitor)
        return response

    def shutdown(self) -> None:
        resources: list[tuple[Any, Any, Thread | None]] = []
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            for handle in self._runs.values():
                if handle.status in {RunStatus.PENDING, RunStatus.RUNNING}:
                    handle.cancel_requested = True
                    handle.status = RunStatus.CANCELLED
                    handle.finished_at = datetime.now(UTC)
                    self._event(handle.id, "cancelled", "Server shut down")
                    resources.append((*self._detach_resources(handle),))
        for worker, output_queue, monitor in resources:
            self._cleanup_resources(worker, output_queue, terminate=True)
            self._join_monitor(monitor)

    def _monitor(self, run_id: str, options: RunOptions) -> None:
        started = monotonic()
        while True:
            with self._lock:
                handle = self._require(run_id)
                if handle.cancel_requested:
                    handle.monitor = None
                    return
                output_queue = handle.output_queue
                worker = handle.worker
            resident_bytes = _resident_bytes(worker) if self._use_processes else None
            if (
                resident_bytes is not None
                and resident_bytes > options.memory_limit_mb * 1024 * 1024
            ):
                self._finish(
                    run_id,
                    RunStatus.FAILED,
                    error="run exceeded its preview memory limit",
                    event_type="failed",
                )
                return
            remaining = options.timeout_seconds - (monotonic() - started)
            if remaining <= 0:
                self._finish(
                    run_id,
                    RunStatus.TIMED_OUT,
                    error="run exceeded its preview timeout",
                    event_type="timed_out",
                )
                return
            try:
                message = output_queue.get(timeout=min(0.05, remaining))
            except (queue.Empty, OSError, ValueError):
                with self._lock:
                    handle = self._require(run_id)
                    if handle.cancel_requested:
                        handle.monitor = None
                        return
                if worker is not None and not worker.is_alive():
                    try:
                        message = output_queue.get(timeout=0.1)
                    except (queue.Empty, OSError, ValueError):
                        self._finish(
                            run_id,
                            RunStatus.FAILED,
                            error="worker exited without a result",
                            event_type="failed",
                        )
                        return
                    else:
                        self._finish_from_message(run_id, message)
                        return
                continue
            self._finish_from_message(run_id, message)
            return

    def _finish_from_message(self, run_id: str, message: object) -> None:
        if not isinstance(message, dict):
            self._finish(
                run_id,
                RunStatus.FAILED,
                error="worker returned an invalid result",
                event_type="failed",
            )
        elif message.get("ok") is True and isinstance(message.get("result"), dict):
            self._finish(
                run_id,
                RunStatus.COMPLETED,
                result=message["result"],
                event_type="completed",
            )
        else:
            self._finish(
                run_id,
                RunStatus.FAILED,
                error=str(message.get("error", "run failed"))[:4000],
                event_type="failed",
            )

    def _finish(
        self,
        run_id: str,
        status: RunStatus,
        *,
        event_type: str,
        error: str | None = None,
        result: dict[str, JSONValue] | None = None,
    ) -> None:
        with self._lock:
            handle = self._require(run_id)
            if handle.status is not RunStatus.RUNNING:
                return
            handle.status = status
            handle.finished_at = datetime.now(UTC)
            handle.error = error
            handle.result = result
            message = (
                "Run completed"
                if status is RunStatus.COMPLETED
                else error or "run failed"
            )
            self._event(run_id, event_type, message)
            worker, output_queue, _ = self._detach_resources(handle)
        self._cleanup_resources(
            worker,
            output_queue,
            terminate=status is not RunStatus.COMPLETED,
        )

    def _event(self, run_id: str, event_type: str, message: str) -> None:
        handle = self._require(run_id)
        handle.events = (
            *handle.events,
            RunEvent(
                sequence=len(handle.events),
                timestamp=datetime.now(UTC),
                type=event_type,
                message=message[:4000],
            ),
        )
        self._event_condition.notify_all()

    def _prune_history(self) -> None:
        if len(self._runs) < self._max_history:
            return
        terminal = sorted(
            (
                handle
                for handle in self._runs.values()
                if handle.status
                in {
                    RunStatus.COMPLETED,
                    RunStatus.FAILED,
                    RunStatus.CANCELLED,
                    RunStatus.TIMED_OUT,
                }
            ),
            key=lambda handle: handle.created_at,
        )
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

    @staticmethod
    def _detach_resources(handle: _RunHandle) -> tuple[Any, Any, Thread | None]:
        worker, output_queue, monitor = (
            handle.worker,
            handle.output_queue,
            handle.monitor,
        )
        handle.worker = None
        handle.output_queue = None
        handle.monitor = None
        return worker, output_queue, monitor

    def _cleanup_resources(
        self, worker: Any, output_queue: Any, *, terminate: bool
    ) -> None:
        if worker is not None:
            if self._use_processes and terminate and worker.is_alive():
                with suppress(BaseException):
                    worker.terminate()
            with suppress(BaseException):
                worker.join(timeout=1)
            if self._use_processes and worker.is_alive():
                with suppress(BaseException):
                    worker.kill()
                    worker.join(timeout=1)
        if self._use_processes and output_queue is not None:
            with suppress(BaseException):
                output_queue.close()
            with suppress(BaseException):
                output_queue.join_thread()

    @staticmethod
    def _join_monitor(monitor: Thread | None) -> None:
        if monitor is not None and monitor is not current_thread():
            monitor.join(timeout=1)
