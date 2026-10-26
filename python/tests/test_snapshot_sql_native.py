from __future__ import annotations

import json

import pyarrow as pa
import pytest

import calc_flow as cf


def test_snapshot_budget_uses_independent_native_config_fields() -> None:
    config = cf.StreamRuntimeConfig(
        sql_snapshot_state_budget=cf.StateBudget(7, 1 << 20)
    )
    native = config._native()
    assert native["sql_snapshot_state_max_rows"] == 7
    assert native["sql_snapshot_state_max_bytes"] == 1 << 20
    assert "sql_state_max_rows" not in native
    assert "sql_compact_state_max_groups" not in native
    assert "sql_snapshot_state_max_rows" not in cf.StreamRuntimeConfig()._native()
    with pytest.raises(TypeError, match="sql_snapshot_state_budget"):
        cf.StreamRuntimeConfig(sql_snapshot_state_budget=object())._native()


def _snapshot_plan() -> cf.StreamExecutionPlan:
    project = (
        cf.PipelineBuilder("native-snapshot").expression("snapshot", "value").project
    )
    fields = [{"name": "value", "data_type": "int64", "nullable": True}]
    node = project["graph"]["nodes"][0]
    node["operator"] = {
        "kind": "snapshot_sql",
        "query": "SELECT * FROM p",
        "aliases": ["p"],
    }
    node["input_ports"] = [
        {"name": "p", "kind": "table", "required": True, "schema": fields}
    ]
    node["output_ports"] = [
        {"name": "output", "kind": "table", "required": True, "schema": fields}
    ]
    project["data_sources"] = []
    project["runtime"] = {"mode": "stream", "options": {}}
    return cf.Runtime()._compile_stream_graph_project(json.dumps(project))


def test_explicit_arrow_source_cannot_slice_a_snapshot_ingress() -> None:
    class Sink:
        async def open(self) -> None:
            pass

        async def write(self, batch: cf.Batch) -> None:
            raise AssertionError("rejected binding must not deliver data")

        async def close(self) -> None:
            pass

    binding = cf.SourceBinding.from_arrow(
        pa.table({"value": [1, 2]}), budget=cf.EdgeBudget(1, 1024)
    )
    with pytest.raises(ValueError, match="snapshot_sql.*ArrowSource slices relations"):
        cf.StreamingRunner(
            _snapshot_plan(),
            {"p": binding},
            {"output": [cf.SinkBinding.ordinary("sink", Sink())]},
            config=cf.StreamRuntimeConfig(checkpointing=False),
        )
