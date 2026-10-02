from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pyarrow as pa

import calc_flow as cf


def _schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("key", pa.string(), metadata={b"role": b"group"}),
            pa.field("value", pa.int64(), metadata={b"unit": b"cents"}),
            *(pa.field(f"unused{i}", pa.string()) for i in range(6)),
        ],
        metadata={b"owner": b"projected-recovery"},
    )


def _table(keys: list[str], values: list[int]) -> pa.Table:
    return pa.Table.from_pydict(
        {
            "key": keys,
            "value": values,
            **{f"unused{i}": [str(i) * 8192] * len(keys) for i in range(6)},
        },
        schema=_schema(),
    )


class _Source:
    def __init__(self, pause: bool, offsets: list[int]) -> None:
        self.pause = pause
        self.offsets = offsets
        self.offset = 0
        self.paused = asyncio.Event()
        self.release = asyncio.Event()

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=3,
            max_batch_bytes=1 << 20,
            schema=_schema(),
            native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        self.offset = 0 if cursor is None else int(cursor.payload["offset"])
        self.offsets.append(self.offset)

    async def next(self) -> cf.Data | None:
        if self.pause and self.offset == 1:
            self.paused.set()
            await self.release.wait()
            return None
        if self.offset == 2:
            return None
        table = (
            _table(["a", "b", "a"], [1, 2, 3])
            if self.offset == 0
            else _table(["b", "c"], [5, 7])
        )
        attributes = {"batch": self.offset, "description": "x" * 8192}
        self.offset += 1
        return cf.Data(
            cf.Batch.from_pyarrow(table, metadata=attributes),
            cf.Cursor(str(self.offset).encode(), {"offset": self.offset}),
        )

    async def close(self) -> None:
        self.release.set()


class _Sink:
    def __init__(self) -> None:
        self.tables: list[pa.Table] = []
        self.metadata: list[dict[str, object]] = []
        self.written = asyncio.Event()

    async def open(self) -> None:
        return None

    async def write(self, batch: cf.Batch) -> None:
        self.tables.append(batch.to_pyarrow())
        self.metadata.append(batch.metadata)
        self.written.set()

    async def close(self) -> None:
        return None


def _rows(table: pa.Table) -> dict[str, list[object]]:
    return table.sort_by([("key", "ascending")]).to_pydict()


def test_sql_projected_checkpoint_restores_and_continues(tmp_path: Path) -> None:
    offsets: list[int] = []
    sink = _Sink()

    def runner(source: _Source) -> cf.StreamingRunner:
        plan = (
            cf.PipelineBuilder("sql-projected-recovery")
            .sql("totals", "SELECT key, SUM(value) AS total FROM input GROUP BY key")
            .compile_stream()
        )
        assert plan.source_binding_ids == ("input",)
        return cf.StreamingRunner(
            plan,
            {
                "input": cf.SourceBinding(
                    source, watermark_policy=cf.DisabledWatermarks()
                )
            },
            {"output": [cf.SinkBinding.ordinary("archive", sink)]},
            cf.ManagedCheckpointRuntime(tmp_path),
        )

    async def exercise() -> None:
        source = _Source(True, offsets)
        first_runner = runner(source)
        first = await asyncio.wait_for(first_runner.start_async(), 30)
        try:
            await asyncio.wait_for(source.paused.wait(), 30)
            await asyncio.wait_for(sink.written.wait(), 30)
            assert _rows(sink.tables[0]) == {"key": ["a", "b"], "total": [4, 2]}
            assert sink.metadata[0] == {"batch": 0, "description": "x" * 8192}
            assert await asyncio.wait_for(first.trigger_checkpoint_async(), 30) == 1
        finally:
            await asyncio.wait_for(first.cancel_async(), 30)

        manifests = sorted((tmp_path / "manifests").glob("manifest-*.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        entries = [
            entry
            for entry in manifest["operators"].values()
            if "query_sha256" in entry["inline_metadata"]
        ]
        assert len(entries) == 1
        entry = entries[0]
        metadata = entry["inline_metadata"]
        assert metadata.get("state_layout", 1) == 2
        assert metadata["state_accounting"] == 2
        assert metadata["retained_ordinals"] == [0, 1]
        assert metadata["rows"] == 3
        segments = {handle["segment_id"]: handle for handle in entry["segments"]}
        assert set(segments) == {"input-projected", "logical-schema", "batch-metadata"}

        def body(name: str) -> bytes:
            handle = segments[name]
            raw = (tmp_path / "state" / handle["relative_path"]).read_bytes()
            assert len(raw) == handle["byte_len"]
            assert hashlib.sha256(raw).hexdigest() == handle["sha256"]
            return raw

        projected = pa.ipc.open_file(
            pa.BufferReader(body("input-projected"))
        ).read_all()
        assert projected.schema.equals(
            pa.schema(list(_schema())[:2], metadata=_schema().metadata),
            check_metadata=True,
        )
        assert projected.to_pydict() == {"key": ["a", "b", "a"], "value": [1, 2, 3]}
        assert projected.nbytes < _table(["a", "b", "a"], [1, 2, 3]).nbytes / 100
        logical = pa.ipc.open_file(pa.BufferReader(body("logical-schema")))
        assert logical.schema.equals(_schema(), check_metadata=True)
        assert logical.num_record_batches == 0
        batch_metadata = json.loads(body("batch-metadata"))
        assert batch_metadata == {
            "source": "input",
            "sequence": 0,
            "attributes": {"batch": 0, "description": "x" * 8192},
        }

        second_runner = runner(_Source(False, offsets))
        second = await asyncio.wait_for(second_runner.start_async(), 30)
        try:
            assert (
                await asyncio.wait_for(second.wait_async(), 30)
            ).state == "completed"
        finally:
            await asyncio.wait_for(second.cancel_async(), 30)

    asyncio.run(exercise())
    assert offsets == [0, 1]
    assert len(sink.tables) == 2
    assert sink.tables[1].schema.equals(sink.tables[0].schema, check_metadata=True)
    assert _rows(sink.tables[1]) == {"key": ["a", "b", "c"], "total": [4, 7, 7]}
    assert sink.metadata[1] == {"batch": 1, "description": "x" * 8192}
