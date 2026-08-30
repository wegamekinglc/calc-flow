"""SCE-11 immutable static stream input adapter tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pyarrow
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    ReplayPositioning,
    Runtime,
    SinkBinding,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamingRunner,
    register_numpy,
)
from calc_flow.pipeline import PipelineBuilder, _canonical


class _Source:
    def __init__(self, batch: Batch | None = None) -> None:
        self._batch = batch

    def capabilities(self) -> SourceCapabilities:
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=2,
            max_batch_bytes=1024,
            native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: object) -> None:
        return None

    async def next(self) -> Data | None:
        if self._batch is None:
            return None
        batch = self._batch
        self._batch = None
        return Data(batch, Cursor(b"1", {}))

    async def close(self) -> None:
        return None


class _Sink:
    def __init__(self) -> None:
        self.batches: list[Batch] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: Batch) -> None:
        self.batches.append(batch)

    async def close(self) -> None:
        return None


def _project(declaration: dict[str, object]) -> dict[str, object]:
    nodes = (
        [
            {
                "id": "matrix",
                "operator": {
                    "kind": "external",
                    "provider": "numpy",
                    "name": "symbolic_matrix",
                    "version": "1",
                    "options": {
                        "columns": ["x"],
                        "expression": {
                            "left": {"op": "input"},
                            "op": "matmul",
                            "right": {"op": "weights"},
                        },
                        "names": ["score"],
                    },
                },
                "input_ports": [
                    {"name": "input", "kind": "table"},
                    {"name": "weights", "kind": "array"},
                ],
                "output_ports": [{"name": "output", "kind": "table"}],
            }
        ]
        if declaration["kind"] == "array"
        else [
            {
                "id": "merge",
                "operator": {"kind": "union"},
                "input_ports": [
                    {"name": "a", "kind": "table"},
                    {"name": "w", "kind": "table"},
                ],
            }
        ]
    )
    return {
        "format_version": 3,
        "id": "static-inputs",
        "name": "Static inputs",
        "runtime": {"mode": "stream", "options": {}},
        "graph": {
            "name": "static-inputs",
            "nodes": nodes,
        },
        "static_inputs": [declaration],
    }


def _table_declaration() -> dict[str, object]:
    return {
        "kind": "table",
        "name": "w",
        "mutability": "static",
        "schema": [{"name": "factor", "data_type": "float64", "nullable": False}],
    }


def _array_declaration() -> dict[str, object]:
    return {
        "kind": "array",
        "name": "weights",
        "mutability": "static",
        "backend": "numpy",
        "dtype": "float64",
        "shape": [1, 1],
    }


def _plan(declaration: dict[str, object]) -> object:
    builder = PipelineBuilder._from_json(_canonical(_project(declaration)))
    if declaration["kind"] != "array":
        return builder.compile_stream()
    runtime = Runtime()
    register_numpy(runtime)
    return builder.compile_stream(runtime=runtime)


def _weights_table() -> Batch:
    schema = pyarrow.schema(
        [pyarrow.field("factor", pyarrow.float64(), nullable=False)]
    )
    return Batch.from_pyarrow(pyarrow.table({"factor": [1.0, 2.0, 3.0]}, schema=schema))


def _runner(
    plan: object,
    tmp_path: Path,
    static_inputs: Mapping[str, Batch] | None,
    *,
    source: _Source | None = None,
    sink: _Sink | None = None,
) -> StreamingRunner:
    selected_source = _Source() if source is None else source
    selected_sink = _Sink() if sink is None else sink
    return StreamingRunner(
        plan,
        {
            plan.source_binding_ids[0]: SourceBinding(
                selected_source,
                watermark_policy=DisabledWatermarks(),
            )
        },
        {"output": [SinkBinding.ordinary("archive", selected_sink)]},
        ManagedCheckpointRuntime(tmp_path / "managed"),
        static_inputs=static_inputs,
    )


def test_plan_exposes_static_input_ids_and_excludes_them_from_sources() -> None:
    plan = _plan(_table_declaration())
    assert plan.static_input_ids == ("w",)
    assert plan.source_binding_ids == ("a",)


def test_static_input_mapping_is_validated_before_native_construction(
    tmp_path: Path,
) -> None:
    plan = _plan(_table_declaration())
    with pytest.raises(TypeError, match="static_inputs must be a mapping"):
        _runner(plan, tmp_path, [("w", None)])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="static_inputs must be a mapping"):
        _runner(plan, tmp_path, {1: _weights_table()})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="static_inputs must be a mapping"):
        _runner(plan, tmp_path, {"w": object()})  # type: ignore[dict-item]


def test_static_inputs_are_exempt_from_the_project_plan_rejection(
    tmp_path: Path,
) -> None:
    """Static payload values stay caller-supplied; only the four
    connector-owned arguments keep their rejection semantics."""
    runner = _runner(_plan(_table_declaration()), tmp_path, {"w": _weights_table()})
    assert runner is not None


def test_static_input_mapping_is_defensively_copied(tmp_path: Path) -> None:
    import asyncio

    plan = _plan(_table_declaration())
    supplied = {"w": _weights_table()}
    runner = _runner(plan, tmp_path, supplied)
    supplied.clear()

    async def exercise() -> None:
        job = await runner.start_async()
        outcome = await job.shutdown_async()
        assert outcome is not None

    asyncio.run(exercise())


def test_latched_array_values_survive_caller_mutation(tmp_path: Path) -> None:
    """Open Question 2 acceptance: NumPy-backed values snapshot at the
    latch seam, so caller-side mutation afterwards cannot reach the job."""
    import asyncio

    values = np.array([[4.0]], dtype=np.float64)
    weights = Batch.from_array(values, backend="numpy")
    source = _Source(
        Batch.from_pyarrow(
            pyarrow.table({"x": pyarrow.array([2.0, 3.0], type=pyarrow.float64())})
        )
    )
    sink = _Sink()
    runner = _runner(
        _plan(_array_declaration()),
        tmp_path,
        {"weights": weights},
        source=source,
        sink=sink,
    )
    values[:] = 99.0

    async def exercise() -> None:
        job = await runner.start_async()
        outcome = await job.wait_async()
        assert outcome.state == "completed"

    asyncio.run(exercise())

    assert len(sink.batches) == 1
    observed = sink.batches[0].to_pyarrow().to_pydict()
    assert observed == {"x": [2.0, 3.0], "score": [8.0, 12.0]}
    assert 99.0 not in observed["score"]


def test_missing_static_input_fails_before_sources_open(tmp_path: Path) -> None:
    import asyncio

    from calc_flow import StreamingRuntimeError

    runner = _runner(_plan(_table_declaration()), tmp_path, None)

    async def exercise() -> StreamingRuntimeError:
        return await runner.start_async()  # type: ignore[return-value]

    with pytest.raises(StreamingRuntimeError, match="static_inputs.w") as error:
        asyncio.run(exercise())
    assert "required static input is missing" in str(error.value)


def test_project_document_accepts_and_omits_static_declarations() -> None:
    from calc_flow.config import ProjectDocument

    validated = ProjectDocument.model_validate(_project(_table_declaration()))
    canonical = json.loads(validated.canonical_json())
    assert canonical["static_inputs"][0]["mutability"] == "static"

    without = _project(_table_declaration())
    without["static_inputs"] = []
    canonical_without = json.loads(
        ProjectDocument.model_validate(without).canonical_json()
    )
    assert "static_inputs" not in canonical_without, (
        "empty declaration lists must stay omitted for byte compatibility"
    )


def test_static_inputs_type_is_mapping_protocol(tmp_path: Path) -> None:
    class LazyMapping(Mapping):
        def __iter__(self):
            return iter(())

        def __len__(self) -> int:
            return 0

        def __getitem__(self, key):
            raise KeyError(key)

    runner = _runner(_plan(_table_declaration()), tmp_path, LazyMapping())
    assert runner is not None
