from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pyarrow as pa

import calc_flow as cf


class _CollectSink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []

    async def open(self) -> None:
        return None

    async def write(self, batch: cf.Batch) -> None:
        self.tables.append(batch.to_pyarrow())

    async def close(self) -> None:
        return None


def _capture(
    root: Path, aggregate: str
) -> tuple[dict[str, object], dict[str | None, int | None]]:
    data = pa.table(
        {
            "key": ["a", "a", "b", None],
            "value": pa.array([2, 5, -1, None], type=pa.int64()),
            "seq": pa.array([1, 2, 1, None], type=pa.int64()),
        }
    )
    source = cf.table_input("events", schema=data.schema)
    program = cf.Program(
        "sql-checkpoint-layout-inventory",
        engine="streaming",
        outputs={
            "totals": source.sql(
                f"SELECT key, {aggregate} AS total FROM input GROUP BY key"
            )
        },
    )
    sink = _CollectSink()
    plan = program.compile_stream()
    assert len(plan.source_binding_ids) == len(plan.sink_binding_ids) == 1
    runner = cf.StreamingRunner(
        plan,
        {plan.source_binding_ids[0]: cf.SourceBinding.from_arrow(data)},
        {plan.sink_binding_ids[0]: [cf.SinkBinding.ordinary("archive", sink)]},
        cf.ManagedCheckpointRuntime(root),
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
    paths = list((root / "manifests").glob("manifest-*.json"))
    assert len(paths) == 1
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    owners = [entry for entry in manifest["operators"].values() if entry["segments"]]
    assert len(owners) == 1
    entry = owners[0]
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
    ]:
        control, actual = _capture(tmp_path / str(layout), aggregate)
        assert actual == expected
        assert control["state_layout"] == layout
        assert control.get("native_semantics") == semantics
        observed.add(control["state_layout"])

    sql = next(op for op in cf.Runtime().capabilities().operators if op.kind == "sql")
    assert sql.state_version == 2
    assert set(sql.state_layouts) == observed
