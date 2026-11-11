from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pyarrow as pa

import calc_flow as cf
from calc_flow._compat import zip


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: cf.Batch) -> None:
        self.tables.append(batch.to_pyarrow())

    async def close(self) -> None:
        return None


def _program(data: pa.Table, aggregate: str, *, replacement: bool) -> cf.Program:
    source = cf.table_input("events", schema=data.schema)
    if replacement:
        source = source.sql("SELECT key, SUM(value) AS value FROM input GROUP BY key")
    program = cf.Program(
        "sql-checkpoint-layout-inventory",
        engine="streaming",
        outputs={
            "totals": source.sql(
                f"SELECT key, {aggregate} AS total FROM input GROUP BY key"
            )
        },
    )
    return program.with_update_semantics("latest_relation") if replacement else program


def _captured_entry(manifest: dict[str, object], layout: int) -> dict[str, object]:
    owners = [entry for entry in manifest["operators"].values() if entry["segments"]]
    if layout == 6:
        assert len(owners) == 2
        assert {entry["inline_metadata"]["state_layout"] for entry in owners} == {3, 6}
        entry = next(
            entry
            for entry in owners
            if entry["inline_metadata"]["__calc_flow_operator_state_version"] == 3
        )
        assert entry["inline_metadata"]["state_kind"] == "sql_replacement"
        return entry
    assert len(owners) == 1
    assert owners[0]["inline_metadata"]["__calc_flow_operator_state_version"] == 2
    return owners[0]


def _capture(
    root: Path, aggregate: str, *, layout: int
) -> tuple[dict[str, object], dict[str | None, int | None]]:
    data = pa.table(
        {
            "key": ["a", "a", "b", None],
            "value": pa.array([2, 5, -1, None], type=pa.int64()),
            "seq": pa.array([1, 2, 1, None], type=pa.int64()),
        }
    )
    program = _program(data, aggregate, replacement=layout == 6)
    sink = _CollectSink()
    plan = program.compile_stream()
    assert len(plan.source_binding_ids) == len(plan.sink_binding_ids) == 1
    runner = cf.StreamingRunner(
        plan,
        {plan.source_binding_ids[0]: cf.SourceBinding.from_arrow(data)},
        {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("archive", sink)]},
        cf.ManagedCheckpointRuntime(root),
        config=cf.StreamRuntimeConfig(
            sql_replacement_state_budget=cf.StateBudget(100, 1 << 20)
        )
        if layout == 6
        else None,
    )

    async def run() -> None:
        job = await runner.start_async()
        try:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            assert outcome.state == "completed", outcome
        finally:
            if job.status()["state"] == "running":
                await job.cancel_async()

    asyncio.run(run())
    assert len(sink.tables) == 1
    actual = dict(
        zip(
            sink.tables[0]["key"].to_pylist(),
            sink.tables[0]["total"].to_pylist(),
            strict=True,
        )
    )
    assert sink.tables[0].num_rows == len(actual)
    paths = list((root / "manifests").glob("manifest-*.json"))
    assert len(paths) == 1
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    entry = _captured_entry(manifest, layout)
    segments = {}
    for handle in entry["segments"]:
        payload = (root / "state" / handle["relative_path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == handle["sha256"]
        segments[handle["segment_id"]] = payload
    control = json.loads(segments["control"])
    assert control["state_layout"] == entry["inline_metadata"]["state_layout"]
    assert control["datafusion_version"] == "55.0.0"
    return control, actual


def test_sql_layout_inventory_covers_native_captures(tmp_path: Path) -> None:
    observed = set()
    for aggregate, layout, semantics, expected in [
        ("SUM(value)", 3, 1, {"a": 7, "b": -1, None: None}),
        ("SUM(abs(value))", 4, None, {"a": 7, "b": 1, None: None}),
        (
            "LAST_VALUE(value ORDER BY seq)",
            5,
            2,
            {"a": 5, "b": -1, None: None},
        ),
        ("SUM(value)", 6, None, {"a": 7, "b": -1, None: None}),
    ]:
        control, actual = _capture(tmp_path / str(layout), aggregate, layout=layout)
        assert actual == expected
        assert control["state_layout"] == layout
        assert control.get("native_semantics") == semantics
        observed.add(control["state_layout"])

    sql = next(op for op in cf.Runtime().capabilities().operators if op.kind == "sql")
    assert sql.state_version == 2
    assert set(sql.state_layouts) == observed
