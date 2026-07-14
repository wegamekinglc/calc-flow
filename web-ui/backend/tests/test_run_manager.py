from __future__ import annotations

import base64
import json
import pickle
import threading
import time
from datetime import date
from decimal import Decimal
from enum import Enum

import pyarrow as pa
import pytest
from calc_flow import ProjectDocument

import calc_flow_studio.run_manager as run_manager_module
from calc_flow_studio.models import InputPayload, RunOptions, RunRequest, RunStatus
from calc_flow_studio.run_manager import (
    RunManager,
    RunManagerError,
    _decode_source,
    _json_safe,
    _register_referenced_builtins,
    _result_payload,
    _serialize_worker_payload,
    prepare_run,
)


def _source(
    *,
    input_name: str = "input",
    source_id: str = "sample",
    format: str = "inline_json",
    data: object | None = None,
) -> dict[str, object]:
    return {
        "id": source_id,
        "input": input_name,
        "format": format,
        "data": [{"value": 1}, {"value": 2}] if data is None else data,
    }


def _project(
    *,
    sources: list[dict[str, object]] | None = None,
    run_options: dict[str, int] | None = None,
    input_ports: list[dict[str, object]] | None = None,
) -> ProjectDocument:
    node: dict[str, object] = {
        "id": "calculate",
        "operator": {"kind": "expression", "expression": "result = value + 1"},
    }
    if input_ports is not None:
        node["input_ports"] = input_ports
    return ProjectDocument.model_validate(
        {
            "format_version": 2,
            "id": "demo",
            "name": "Demo",
            "pipeline": {"name": "Main", "nodes": [node]},
            "data_sources": sources if sources is not None else [_source()],
            **({"run_options": run_options} if run_options is not None else {}),
        }
    )


def _two_input_project(*, run_options: dict[str, int] | None = None) -> ProjectDocument:
    return ProjectDocument.model_validate(
        {
            "format_version": 2,
            "id": "two_inputs",
            "name": "Two inputs",
            "pipeline": {
                "name": "Two inputs",
                "nodes": [
                    {
                        "id": "join",
                        "operator": {
                            "kind": "sql",
                            "query": (
                                "SELECT left_input.value + right_input.value AS total "
                                "FROM left_input CROSS JOIN right_input"
                            ),
                            "aliases": ["left_input", "right_input"],
                        },
                    }
                ],
            },
            "data_sources": [
                _source(input_name="left_input", source_id="left", data=[]),
                _source(input_name="right_input", source_id="right", data=[]),
            ],
            **({"run_options": run_options} if run_options is not None else {}),
        }
    )


def _ipc_data(*, file: bool = False) -> str:
    sink = pa.BufferOutputStream()
    table = pa.table({"value": [1, 2]})
    factory = pa.ipc.new_file if file else pa.ipc.new_stream
    with factory(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")


def _compressed_ipc_data() -> str:
    sink = pa.BufferOutputStream()
    table = pa.table({"value": ["x" * 1000] * 100})
    options = pa.ipc.IpcWriteOptions(compression="zstd")
    with pa.ipc.new_stream(sink, table.schema, options=options) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode("ascii")


def _wait(manager: RunManager, run_id: str, timeout: float = 10) -> RunStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.get(run_id).status
        if status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return status
        time.sleep(0.01)
    raise AssertionError("run did not finish")


@pytest.mark.parametrize(
    "payload",
    [
        InputPayload(format="records", data=[{"value": 1}, {"value": 2}]),
        InputPayload(format="columns", data={"value": [1, 2]}),
        InputPayload(format="arrow_ipc", data=_ipc_data()),
        InputPayload(format="arrow_ipc", data=_ipc_data(file=True)),
    ],
)
def test_prepare_run_decodes_request_table_formats(payload: InputPayload) -> None:
    prepared, options = prepare_run(_project(), RunRequest(inputs={"input": payload}))

    table, metadata = prepared["input"]
    assert table.to_pylist() == [{"value": 1}, {"value": 2}]
    assert metadata == {"source_id": "input"}
    assert options == RunOptions()


def test_prepare_run_accepts_an_empty_columns_mapping() -> None:
    prepared, _ = prepare_run(
        _project(),
        RunRequest(inputs={"input": InputPayload(format="columns", data={})}),
    )

    table, _ = prepared["input"]
    assert table.num_rows == 0
    assert table.num_columns == 0


@pytest.mark.parametrize(
    ("format", "data"),
    [
        ("inline_json", [{"value": 1}, {"value": 2}]),
        ("inline_json", {"value": [1, 2]}),
        ("csv", "value\n1\n2\n"),
        ("json", '[{"value":1},{"value":2}]'),
        ("json", '{"value":1}\n{"value":2}\n'),
        ("arrow_ipc", _ipc_data()),
        ("arrow_ipc", _ipc_data(file=True)),
    ],
)
def test_prepare_run_decodes_all_saved_source_formats(
    format: str, data: object
) -> None:
    prepared, _ = prepare_run(
        _project(sources=[_source(format=format, data=data)]), RunRequest()
    )

    table, metadata = prepared["input"]
    assert table.to_pylist() == [{"value": 1}, {"value": 2}]
    assert metadata == {"source_id": "sample"}


def test_request_inputs_override_saved_sources_as_a_complete_set() -> None:
    project = _two_input_project()
    with pytest.raises(RunManagerError, match="run inputs must be"):
        prepare_run(
            project,
            RunRequest(
                inputs={
                    "left_input": InputPayload(format="records", data=[{"value": 1}])
                }
            ),
        )
    with pytest.raises(RunManagerError, match="run inputs must be"):
        prepare_run(
            project,
            RunRequest(
                inputs={
                    "left_input": InputPayload(format="records", data=[]),
                    "right_input": InputPayload(format="records", data=[]),
                    "unknown": InputPayload(format="records", data=[]),
                }
            ),
        )


@pytest.mark.parametrize(
    ("format", "data", "message"),
    [
        ("records", [1], "record objects"),
        ("columns", {"value": [1], "bad": 2}, "column lists"),
        ("arrow_ipc", "not-base64!", "valid base64"),
        ("arrow_ipc", base64.b64encode(b"not arrow").decode(), "stream nor file"),
        ("csv", [], "must be text"),
        ("json", "not-json", "array, object"),
    ],
)
def test_decode_source_rejects_malformed_values(
    format: str, data: object, message: str
) -> None:
    with pytest.raises(RunManagerError, match=message):
        _decode_source(format, data, max_bytes=10_000)


def test_prepare_run_enforces_combined_rows_and_exact_limit() -> None:
    request = RunRequest(
        inputs={
            "left_input": InputPayload(format="records", data=[{"value": 1}]),
            "right_input": InputPayload(format="records", data=[{"value": 2}]),
        },
        options=RunOptions(max_rows=2),
    )
    prepared, _ = prepare_run(_two_input_project(), request)
    assert sum(table.num_rows for table, _ in prepared.values()) == 2

    with pytest.raises(RunManagerError, match="combined inputs exceed the 1 row"):
        prepare_run(
            _two_input_project(),
            request.model_copy(update={"options": RunOptions(max_rows=1)}),
        )


def test_prepare_run_enforces_combined_encoded_and_decoded_bytes() -> None:
    left = [{"value": 1}]
    right = [{"value": 2}]
    encoded = len(json.dumps(left, separators=(",", ":")).encode()) + len(
        json.dumps(right, separators=(",", ":")).encode()
    )
    request = RunRequest(
        inputs={
            "left_input": InputPayload(format="records", data=left),
            "right_input": InputPayload(format="records", data=right),
        },
        options=RunOptions(max_input_bytes=encoded),
    )
    prepare_run(_two_input_project(), request)
    with pytest.raises(RunManagerError, match="combined encoded inputs"):
        prepare_run(
            _two_input_project(),
            request.model_copy(
                update={"options": RunOptions(max_input_bytes=encoded - 1)}
            ),
        )

    expanding = InputPayload(format="columns", data={"value": [1, 2, 3, 4]})
    exact_decoded, _ = prepare_run(
        _project(),
        RunRequest(
            inputs={"input": expanding},
            options=RunOptions(max_input_bytes=32),
        ),
    )
    assert exact_decoded["input"][0].nbytes == 32
    with pytest.raises(RunManagerError, match="combined decoded inputs"):
        prepare_run(
            _two_input_project(),
            RunRequest(
                inputs={"left_input": expanding, "right_input": expanding},
                options=RunOptions(max_input_bytes=40),
            ),
        )


def test_prepare_run_casts_declared_schema_and_copies_metadata() -> None:
    source = _source(data=[{"value": "1"}])
    project = _project(
        sources=[source],
        input_ports=[
            {
                "name": "input",
                "kind": "table",
                "required": True,
                "schema": [{"name": "value", "data_type": "int64", "nullable": True}],
            }
        ],
    )
    prepared, _ = prepare_run(project, RunRequest())
    table, metadata = prepared["input"]
    source["id"] = "mutated"

    assert table.schema == pa.schema([pa.field("value", pa.int64())])
    assert table.to_pylist() == [{"value": 1}]
    assert metadata == {"source_id": "sample"}


def test_prepare_run_rejects_array_external_inputs_before_worker_creation() -> None:
    project = ProjectDocument.model_validate(
        {
            "format_version": 2,
            "id": "array",
            "name": "Array",
            "pipeline": {
                "name": "Array",
                "nodes": [
                    {
                        "id": "array",
                        "operator": {
                            "kind": "external",
                            "provider": "numpy",
                            "name": "expression",
                            "version": "1",
                            "options": {"expression": "x + 1"},
                        },
                        "input_ports": [
                            {"name": "input", "kind": "array", "required": True}
                        ],
                        "output_ports": [
                            {"name": "output", "kind": "array", "required": True}
                        ],
                    }
                ],
            },
            "data_sources": [_source()],
        }
    )
    with pytest.raises(RunManagerError, match="table graph inputs only"):
        prepare_run(project, RunRequest())


def test_parent_payload_contains_no_calc_flow_extension_objects() -> None:
    project = _project()
    prepared, options = prepare_run(project, RunRequest())
    payload = _serialize_worker_payload(project, prepared, options)
    project_json, serialized_inputs, options_data = pickle.loads(payload)

    assert isinstance(project_json, str)
    assert isinstance(options_data, dict)
    assert all(isinstance(value[0], pa.Table) for value in serialized_inputs.values())

    def contains_pyo3(value: object) -> bool:
        module = type(value).__module__
        if module.startswith("calc_flow"):
            return True
        if isinstance(value, dict):
            return any(
                contains_pyo3(key) or contains_pyo3(item) for key, item in value.items()
            )
        if isinstance(value, (list, tuple)):
            return any(contains_pyo3(item) for item in value)
        return False

    assert contains_pyo3((project_json, serialized_inputs, options_data)) is False


def test_worker_registers_only_exact_referenced_builtin_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object]] = []
    runtime = object()
    monkeypatch.setattr(
        run_manager_module,
        "register_numpy",
        lambda selected: calls.append(("numpy", selected)),
    )
    monkeypatch.setattr(
        run_manager_module,
        "register_jax",
        lambda selected: calls.append(("jax", selected)),
    )
    project_json = json.dumps(
        {
            "pipeline": {
                "nodes": [
                    {
                        "operator": {
                            "kind": "external",
                            "provider": "numpy",
                            "name": "expression",
                            "version": "1",
                        }
                    },
                    {
                        "operator": {
                            "kind": "external",
                            "provider": "jax",
                            "name": "other",
                            "version": "1",
                        }
                    },
                ]
            }
        }
    )

    _register_referenced_builtins(runtime, project_json)  # type: ignore[arg-type]

    assert calls == [("numpy", runtime)]


def test_thread_worker_executes_rust_plan_with_bounded_result() -> None:
    manager = RunManager(use_processes=False)
    run = manager.submit(
        _project(),
        RunRequest(
            inputs={
                "input": InputPayload(
                    format="records", data=[{"value": 1}, {"value": 2}]
                )
            },
            options=RunOptions(output_rows=1),
        ),
    )
    assert _wait(manager, run.id) is RunStatus.COMPLETED

    result = manager.get(run.id).result
    assert result is not None
    output = result["outputs"]["output"]
    assert output["rows"] == [{"value": 1, "result": 2}]
    assert output["total_rows"] == 2
    assert output["truncated"] is True
    assert result["node_timings"]["calculate"]["input_rows"] == {"input": 2}
    assert result["metadata"]["pipeline_name"] == "Main"
    assert result["metadata"]["pipeline_fingerprint"]
    assert result["metadata"]["run_id"]
    assert result["datafusion_metrics"][0]["query_id"] > 0
    assert result["datafusion_metrics"][0]["node_id"] == "calculate"
    assert result["datafusion_metrics"][0]["logical_plan"]
    assert result["datafusion_metrics"][0]["physical_plan"]
    assert [event.type for event in manager.events(run.id)] == [
        "created",
        "running",
        "completed",
    ]
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_spawned_worker_executes_rust_plan_and_cleans_resources() -> None:
    manager = RunManager()
    run = manager.submit(
        _project(),
        RunRequest(
            inputs={"input": InputPayload(format="records", data=[{"value": 4}])}
        ),
    )

    assert _wait(manager, run.id, timeout=20) is RunStatus.COMPLETED
    result = manager.get(run.id).result
    assert result is not None
    assert result["outputs"]["output"]["rows"] == [{"value": 4, "result": 5}]
    assert result["datafusion_metrics"][0]["logical_plan"]
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_preparation_failure_leaves_no_run_or_capacity_reservation() -> None:
    manager = RunManager(use_processes=False, max_workers=1)
    with pytest.raises(RunManagerError, match="run inputs must be"):
        manager.submit(
            _project(),
            RunRequest(inputs={"wrong": InputPayload(format="records", data=[])}),
        )
    assert manager._runs == {}

    with pytest.raises(RunManagerError, match="encoded input"):
        manager.submit(
            _project(),
            RunRequest(
                inputs={
                    "input": InputPayload(
                        format="records", data=[{"value": "oversized"}]
                    )
                },
                options=RunOptions(max_input_bytes=1),
            ),
        )
    assert manager._runs == {}

    run = manager.submit(_project(), RunRequest())
    assert _wait(manager, run.id) is RunStatus.COMPLETED
    manager.shutdown()


def test_worker_compile_failure_does_not_prevent_immediate_reuse() -> None:
    invalid = _project()
    invalid.root["pipeline"]["nodes"][0]["operator"]["expression"] = "bad("
    manager = RunManager(use_processes=False, max_workers=1)
    failed = manager.submit(invalid, RunRequest())
    assert _wait(manager, failed.id) is RunStatus.FAILED
    assert "Error" in (manager.get(failed.id).error or "")

    succeeded = manager.submit(_project(), RunRequest())
    assert _wait(manager, succeeded.id) is RunStatus.COMPLETED
    manager.shutdown()


def test_submit_caps_concurrent_preparation_without_leaking_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate = threading.Event()
    entered = threading.Event()
    real_prepare = run_manager_module.prepare_run

    def blocking_prepare(*args: object, **kwargs: object) -> object:
        entered.set()
        gate.wait(timeout=5)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(run_manager_module, "prepare_run", blocking_prepare)
    manager = RunManager(use_processes=False, max_workers=1)
    outcomes: dict[str, object] = {}

    def submit_one(label: str) -> None:
        try:
            outcomes[label] = manager.submit(_project(), RunRequest()).id
        except RunManagerError as error:
            outcomes[label] = error

    first = threading.Thread(target=submit_one, args=("first",))
    first.start()
    assert entered.wait(timeout=2)
    submit_one("second")
    gate.set()
    first.join(timeout=5)

    assert isinstance(outcomes["first"], str)
    assert isinstance(outcomes["second"], RunManagerError)
    assert _wait(manager, outcomes["first"]) is RunStatus.COMPLETED
    manager.shutdown()


def test_abnormal_worker_exit_fails_run_and_releases_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(run_manager_module, "_execute_worker", lambda *_: None)
    manager = RunManager(use_processes=False)
    run = manager.submit(_project(), RunRequest())

    assert _wait(manager, run.id) is RunStatus.FAILED
    assert manager.get(run.id).error == "worker exited without a result"
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_monitor_start_failure_terminates_worker_and_releases_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingMonitor:
        def __init__(self, **_: object) -> None:
            return None

        def start(self) -> None:
            raise RuntimeError("monitor unavailable")

    manager = RunManager()
    monkeypatch.setattr(run_manager_module, "Thread", FailingMonitor)

    with pytest.raises(RuntimeError, match="monitor unavailable"):
        manager.submit(_project(), RunRequest())

    assert manager._runs == {}
    manager.shutdown()


def test_cancel_and_shutdown_release_processes_and_queues() -> None:
    manager = RunManager()
    # Holding the manager lock prevents the monitor from winning the race even
    # if the tiny Rust plan completes before cancellation reaches the process.
    with manager._lock:
        run = manager.submit(_project(), RunRequest())
        assert manager.get(run.id).status is RunStatus.RUNNING
        cancelled = manager.cancel(run.id)

    assert cancelled.status is RunStatus.CANCELLED
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_shutdown_cancels_an_active_run_and_is_idempotent() -> None:
    manager = RunManager()
    with manager._lock:
        run = manager.submit(_project(), RunRequest())
        assert manager.get(run.id).status is RunStatus.RUNNING
        manager.shutdown()

    assert manager.get(run.id).status is RunStatus.CANCELLED
    assert manager.events(run.id)[-1].message == "Server shut down"
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()
    with pytest.raises(RunManagerError, match="shut down"):
        manager.submit(_project(), RunRequest())


def test_shutdown_handles_queue_closed_after_monitor_poll() -> None:
    entered = threading.Event()
    closed = threading.Event()

    class ClosingQueue:
        def get(self, *, timeout: float) -> object:
            assert timeout > 0
            entered.set()
            closed.wait(timeout=2)
            raise ValueError("queue is closed")

        def close(self) -> None:
            closed.set()

        def join_thread(self) -> None:
            return None

    class BlockedProcess:
        pid = 1

        def __init__(self) -> None:
            self.alive = False

        def start(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def terminate(self) -> None:
            self.alive = False

        def kill(self) -> None:
            self.alive = False

        def join(self, *, timeout: float) -> None:
            assert timeout > 0

    class BlockingContext:
        queue = ClosingQueue()
        process = BlockedProcess()

        def Queue(self, *, maxsize: int) -> ClosingQueue:
            assert maxsize == 1
            return self.queue

        def Process(self, **kwargs: object) -> BlockedProcess:
            assert kwargs["target"] is run_manager_module._execute_worker
            return self.process

    manager = RunManager()
    context = BlockingContext()
    manager._process_context = context
    run = manager.submit(_project(), RunRequest())
    assert entered.wait(timeout=2)

    manager.shutdown()

    assert manager.get(run.id).status is RunStatus.CANCELLED
    assert manager._runs[run.id].monitor is None


def test_shutdown_during_preparation_never_starts_a_late_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    worker_started = threading.Event()
    real_prepare = run_manager_module.prepare_run

    def blocking_prepare(*args: object, **kwargs: object) -> object:
        entered.set()
        release.wait(timeout=2)
        return real_prepare(*args, **kwargs)

    def record_worker_start(*_: object) -> None:
        worker_started.set()

    monkeypatch.setattr(run_manager_module, "prepare_run", blocking_prepare)
    monkeypatch.setattr(run_manager_module, "_execute_worker", record_worker_start)
    manager = RunManager(use_processes=False)
    outcome: list[BaseException] = []

    def submit() -> None:
        try:
            manager.submit(_project(), RunRequest())
        except BaseException as error:
            outcome.append(error)

    submitting = threading.Thread(target=submit)
    submitting.start()
    assert entered.wait(timeout=2)
    manager.shutdown()
    release.set()
    submitting.join(timeout=2)

    assert worker_started.is_set() is False
    assert len(outcome) == 1
    assert isinstance(outcome[0], RunManagerError)
    assert manager._runs == {}


def test_timeout_releases_worker_and_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    def blocked_worker(*_: object) -> None:
        time.sleep(2)

    monkeypatch.setattr(run_manager_module, "_execute_worker", blocked_worker)
    manager = RunManager(use_processes=False)
    run = manager.submit(_project(), RunRequest(options=RunOptions(timeout_seconds=1)))

    assert _wait(manager, run.id, timeout=3) is RunStatus.TIMED_OUT
    assert "timeout" in (manager.get(run.id).error or "")
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_parent_rss_failure_terminates_worker_and_closes_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        run_manager_module,
        "_resident_bytes",
        lambda _: 65 * 1024 * 1024,
    )
    manager = RunManager()
    run = manager.submit(_project(), RunRequest(options=RunOptions(memory_limit_mb=64)))

    assert _wait(manager, run.id) is RunStatus.FAILED
    assert "memory" in (manager.get(run.id).error or "")
    assert manager._runs[run.id].worker is None
    assert manager._runs[run.id].output_queue is None
    manager.shutdown()


def test_wait_for_events_replays_and_history_prunes_terminal_runs() -> None:
    manager = RunManager(use_processes=False, max_history=1)
    first = manager.submit(_project(), RunRequest())
    assert _wait(manager, first.id) is RunStatus.COMPLETED
    events, status = manager.wait_for_events(first.id, after_sequence=0, timeout=0.01)
    assert [event.type for event in events] == ["running", "completed"]
    assert status is RunStatus.COMPLETED
    assert manager.wait_for_events(
        first.id, after_sequence=events[-1].sequence, timeout=0.01
    ) == ((), RunStatus.COMPLETED)

    second = manager.submit(_project(), RunRequest())
    assert _wait(manager, second.id) is RunStatus.COMPLETED
    with pytest.raises(KeyError, match="does not exist"):
        manager.get(first.id)
    manager.shutdown()


def test_json_safe_and_result_payload_normalize_transport_values() -> None:
    class Value(Enum):
        ITEM = "item"

    class Scalar:
        def item(self) -> Decimal:
            return Decimal("1.25")

    assert _json_safe(float("inf")) is None
    assert _json_safe(Decimal("2.50")) == "2.50"
    assert _json_safe(date(2026, 1, 2)) == "2026-01-02"
    assert _json_safe(b"data") == "ZGF0YQ=="
    assert _json_safe(Value.ITEM) == "item"
    assert _json_safe({1: (Scalar(),)}) == {"1": ["1.25"]}

    class ArrayBatch:
        kind = "array"
        backend = "numpy"
        num_rows = 4
        metadata = {"decimal": Decimal("1.2")}
        array = pa.array([1, 2, 3, 4]).to_numpy()

    class Result:
        outputs = {"output": ArrayBatch()}
        node_timings = {}
        datafusion_metrics = []
        metadata = {"run_id": "run", "value": float("nan")}

    payload = _result_payload(Result(), output_rows=2)
    assert payload["outputs"]["output"] == {
        "backend": "numpy",
        "kind": "array",
        "total_rows": 4,
        "truncated": True,
        "data": [1, 2],
        "metadata": {"decimal": "1.2"},
    }
    assert payload["metadata"] == {"run_id": "run", "value": None}


def test_decode_arrow_checks_encoded_and_decoded_bounds() -> None:
    encoded = _ipc_data()
    with pytest.raises(RunManagerError, match="encoded input"):
        _decode_source("arrow_ipc", encoded, max_bytes=len(encoded.encode()) - 1)

    compressed = _compressed_ipc_data()
    assert len(compressed.encode()) < 100 * 1000
    with pytest.raises(RunManagerError, match="decoded input"):
        _decode_source("arrow_ipc", compressed, max_bytes=len(compressed.encode()))


def test_run_manager_rejects_invalid_configuration_and_unknown_run() -> None:
    with pytest.raises(ValueError, match="greater than 0"):
        RunManager(max_workers=0)
    manager = RunManager(use_processes=False)
    with pytest.raises(KeyError, match="does not exist"):
        manager.get("missing")
    manager.shutdown()
