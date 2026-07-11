from __future__ import annotations

import base64
import threading
import time
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from calc_flow.batch import Batch
from calc_flow.config import (
    DataSourceConfig,
    NodeConfig,
    PipelineConfig,
    ProjectConfig,
    RunOptions,
    UdfReferenceConfig,
)
from calc_flow.pipeline import RunMetadata, RunResult
from calc_flow.udf import UdfRegistry
from calc_flow.web.models import InputPayload, RunRequest, RunStatus
from calc_flow.web.run_manager import (
    RunManager,
    RunManagerError,
    _result_payload,
    prepare_run,
)


def _project(
    *,
    data_sources: tuple[DataSourceConfig, ...] = (),
    options: RunOptions | None = None,
) -> ProjectConfig:
    return ProjectConfig(
        id="demo",
        name="Demo",
        pipeline=PipelineConfig(
            id="main",
            name="Main",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="result = value + 1",
                ),
            ),
        ),
        data_sources=data_sources,
        run_options=options or RunOptions(),
    )


def _wait(manager: RunManager, run_id: str, timeout: float = 5) -> RunStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = manager.get(run_id).status
        if status not in {RunStatus.PENDING, RunStatus.RUNNING}:
            return status
        time.sleep(0.01)
    raise AssertionError("run did not finish")


def _ipc_data() -> str:
    sink = pa.BufferOutputStream()
    table = pa.table({"value": [1, 2]})
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return base64.b64encode(sink.getvalue().to_pybytes()).decode()


@pytest.mark.parametrize(
    "payload",
    [
        InputPayload(format="inline_json", data=[{"value": 1}, {"value": 2}]),
        InputPayload(format="json", data='[{"value":1},{"value":2}]'),
        InputPayload(format="json", data='{"value":1}\n{"value":2}\n'),
        InputPayload(format="csv", data="value\n1\n2\n"),
        InputPayload(format="arrow_ipc", data=_ipc_data()),
    ],
)
def test_prepare_run_decodes_supported_table_inputs(payload: InputPayload) -> None:
    batches, options = prepare_run(_project(), RunRequest(inputs={"input": payload}))

    assert batches["input"].table_payload["value"].to_pylist() == [1, 2]
    assert options.max_rows == 100_000


def test_prepare_run_uses_saved_source_when_request_inputs_are_empty() -> None:
    project = _project(
        data_sources=(
            DataSourceConfig(
                id="sample",
                input_name="input",
                format="inline_json",
                data=[{"value": 3}],
            ),
        )
    )

    batches, _ = prepare_run(project, RunRequest())

    assert batches["input"].table_payload.to_pylist() == [{"value": 3}]


def test_prepare_run_enforces_names_rows_and_bytes() -> None:
    with pytest.raises(RunManagerError, match="must be"):
        prepare_run(
            _project(),
            RunRequest(inputs={"wrong": InputPayload(format="inline_json", data=[])}),
        )

    options = RunOptions(max_rows=1, max_input_bytes=100)
    with pytest.raises(RunManagerError, match="row preview"):
        prepare_run(
            _project(),
            RunRequest(
                inputs={
                    "input": InputPayload(
                        format="inline_json", data=[{"value": 1}, {"value": 2}]
                    )
                },
                options=options,
            ),
        )

    with pytest.raises(RunManagerError, match="byte"):
        prepare_run(
            _project(),
            RunRequest(
                inputs={
                    "input": InputPayload(
                        format="csv", data="value\n" + "1234567890\n" * 20
                    )
                },
                options=RunOptions(max_input_bytes=100),
            ),
        )


def test_thread_run_manager_returns_results_plans_and_metrics() -> None:
    manager = RunManager(use_processes=False)
    run = manager.submit(
        _project(),
        RunRequest(
            inputs={
                "input": InputPayload(
                    format="inline_json", data=[{"value": 1}, {"value": 2}]
                )
            }
        ),
    )

    assert _wait(manager, run.id) is RunStatus.COMPLETED
    completed = manager.get(run.id)
    assert completed.result is not None
    output = completed.result["outputs"]["output"]
    assert output["rows"] == [
        {"value": 1, "result": 2},
        {"value": 2, "result": 3},
    ]
    assert completed.result["datafusion_metrics"][0]["logical_plan"]
    assert [event.type for event in manager.events(run.id)] == [
        "created",
        "running",
        "completed",
    ]
    manager.shutdown()


def test_process_run_manager_executes_registered_udf() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="double_value",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
    )
    def double_value(values):
        return pc.multiply(values, 2)

    project = _project().model_copy(
        update={
            "pipeline": PipelineConfig(
                id="main",
                name="Main",
                nodes=(
                    NodeConfig(
                        id="calculate",
                        kind="expression",
                        expression="result = double_value(value)",
                        udfs=(UdfReferenceConfig(name="double_value", version="1"),),
                    ),
                ),
            )
        }
    )
    manager = RunManager(udf_registry=registry)
    run = manager.submit(
        project,
        RunRequest(
            inputs={"input": InputPayload(format="inline_json", data=[{"value": 4}])}
        ),
    )

    assert _wait(manager, run.id) is RunStatus.COMPLETED
    result = manager.get(run.id).result
    assert result is not None
    assert result["outputs"]["output"]["rows"] == [{"value": 4, "result": 8}]
    manager.shutdown()


def test_process_run_manager_can_cancel_run() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="slow_value",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
        volatility="volatile",
    )
    def slow_value(values):
        time.sleep(1)
        return values

    project = _project().model_copy(
        update={
            "pipeline": PipelineConfig(
                id="main",
                name="Main",
                nodes=(
                    NodeConfig(
                        id="calculate",
                        kind="expression",
                        expression="result = slow_value(value)",
                        udfs=(UdfReferenceConfig(name="slow_value", version="1"),),
                    ),
                ),
            )
        }
    )
    manager = RunManager(udf_registry=registry)
    run = manager.submit(
        project,
        RunRequest(
            inputs={"input": InputPayload(format="inline_json", data=[{"value": 1}])}
        ),
    )

    cancelled = manager.cancel(run.id)

    assert cancelled.status is RunStatus.CANCELLED
    assert manager.events(run.id)[-1].type == "cancelled"
    manager.shutdown()


def test_process_run_manager_enforces_timeout() -> None:
    registry = UdfRegistry()

    @registry.datafusion_scalar(
        name="slow_value",
        version="1",
        input_fields=[pa.int64()],
        return_field=pa.int64(),
        volatility="volatile",
    )
    def slow_value(values):
        time.sleep(1)
        return values

    project = ProjectConfig(
        id="timeout",
        name="Timeout",
        pipeline=PipelineConfig(
            id="main",
            name="Main",
            nodes=(
                NodeConfig(
                    id="calculate",
                    kind="expression",
                    expression="result = slow_value(value)",
                    udfs=(UdfReferenceConfig(name="slow_value", version="1"),),
                ),
            ),
        ),
        run_options=RunOptions(timeout_seconds=0.1),
    )
    manager = RunManager(udf_registry=registry)
    run = manager.submit(
        project,
        RunRequest(
            inputs={"input": InputPayload(format="inline_json", data=[{"value": 1}])}
        ),
    )

    assert _wait(manager, run.id) is RunStatus.TIMED_OUT
    assert "timeout" in (manager.get(run.id).error or "")
    manager.shutdown()


def test_run_manager_rejects_unknown_run() -> None:
    manager = RunManager(use_processes=False)
    with pytest.raises(KeyError, match="does not exist"):
        manager.get("missing")


class _FakeArray:
    """Minimal Array API surface that _result_payload and Batch touch."""

    def __init__(self, values: list[int]) -> None:
        self._values = list(values)

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self._values),)

    def __array_namespace__(self) -> None:
        return None

    def __getitem__(self, key: slice) -> _FakeArray:
        return _FakeArray(self._values[key])

    def tolist(self) -> list[int]:
        return list(self._values)


def test_result_payload_truncates_array_outputs_to_output_rows() -> None:
    batch = Batch.array(_FakeArray(list(range(10))))
    result = RunResult(
        outputs={"output": batch},
        warnings=(),
        node_timings={},
        datafusion_metrics=(),
        metadata=RunMetadata(
            run_id="rid",
            pipeline_name="Main",
            pipeline_fingerprint="fp",
            started_at=datetime(2024, 1, 1, tzinfo=UTC),
            finished_at=datetime(2024, 1, 1, tzinfo=UTC),
        ),
    )

    payload = _result_payload(result, output_rows=3)
    output = payload["outputs"]["output"]

    assert output["kind"] == "array"
    assert output["total_rows"] == 10
    assert output["truncated"] is True
    assert output["data"] == [0, 1, 2]


def test_submit_caps_concurrent_submissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A submission racing with one already mid-prepare must still honor max_workers."""
    import calc_flow.web.run_manager as module

    gate = threading.Event()
    entered = threading.Event()
    real_prepare = module.prepare_run

    def blocking_prepare(*args: object, **kwargs: object) -> object:
        entered.set()
        gate.wait(timeout=5)
        return real_prepare(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_run", blocking_prepare)

    manager = RunManager(use_processes=False, max_workers=1)
    request = RunRequest(
        inputs={"input": InputPayload(format="inline_json", data=[{"value": 1}])}
    )
    outcomes: dict[str, object] = {}

    def submit_one(label: str) -> None:
        try:
            run = manager.submit(_project(), request)
        except RunManagerError as error:
            outcomes[label] = error
        else:
            outcomes[label] = run.id

    first = threading.Thread(target=submit_one, args=("first",))
    first.start()
    assert entered.wait(timeout=2)

    # The second submission runs on this thread while the first is still
    # blocked inside prepare_run (lock released). With the slot reserved
    # atomically, the capacity check must reject it before prepare_run.
    submit_one("second")
    gate.set()
    first.join(timeout=5)

    first_id = outcomes["first"]
    assert isinstance(first_id, str)
    assert isinstance(outcomes["second"], RunManagerError)
    _wait(manager, first_id)
    manager.shutdown()
