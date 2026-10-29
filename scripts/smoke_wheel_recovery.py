"""Check native SQL checkpoint recovery without third-party Python packages."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import calc_flow as cf


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


class _Source:
    def __init__(self, batch: cf.Batch, *, pause: bool) -> None:
        self.batch = batch
        self.pause = pause
        self.offset = 0
        self.resumed = None
        self.opened = False
        self.closed = False
        self.paused = asyncio.Event()
        self.release = asyncio.Event()

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=2,
            max_batch_bytes=1024,
            native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: cf.Cursor | None) -> None:
        self.opened = True
        self.offset = 0 if cursor is None else cursor.payload["offset"]
        self.resumed = self.offset

    async def next(self) -> cf.Data | None:
        if self.pause and self.offset == 1:
            self.paused.set()
            await self.release.wait()
            return None
        if self.offset == 2:
            return None
        self.offset += 1
        return cf.Data(
            self.batch,
            cf.Cursor(str(self.offset).encode(), {"offset": self.offset}),
        )

    async def close(self) -> None:
        self.closed = True
        self.release.set()


class _Sink:
    def __init__(self) -> None:
        self.batches: list[cf.Batch] = []
        self.written = asyncio.Event()
        self.closed = False

    async def open(self) -> None:
        pass

    async def write(self, batch: cf.Batch) -> None:
        self.batches.append(batch)
        self.written.set()

    async def close(self) -> None:
        self.closed = True


def _runner(root: Path, source: _Source, sink: _Sink) -> cf.StreamingRunner:
    plan = (
        cf.PipelineBuilder("standalone-recovery")
        .sql("totals", "SELECT SUM(a) AS total FROM input")
        .compile_stream()
    )
    return cf.StreamingRunner(
        plan,
        {"input": cf.SourceBinding(source, watermark_policy=cf.DisabledWatermarks())},
        {"output": [cf.SinkBinding.ordinary("archive", sink)]},
        cf.ManagedCheckpointRuntime(root),
    )


async def _check_total(batches: list[cf.Batch], expected: int) -> None:
    _check(len(batches) == 1, "SQL recovery emitted the wrong number of batches")
    batch = batches[0]
    invalid = await cf.compute_async(
        batch, lambda t: t.filter(t["total"] != expected), output_format="batch"
    )
    _check(batch.num_rows == 1, "SQL recovery emitted the wrong row count")
    _check(invalid.num_rows == 0, "SQL recovery lost or duplicated aggregate state")


async def _exercise(root: Path, batch: cf.Batch) -> None:
    source = _Source(batch, pause=True)
    sink = _Sink()
    first = await _runner(root, source, sink).start_async()
    try:
        await source.paused.wait()
        await sink.written.wait()
        await _check_total(sink.batches, 3)
        _check(await first.trigger_checkpoint_async() == 1, "checkpoint epoch changed")
    finally:
        await first.cancel_async()
    _check(first.status()["task_count"] == 0, "cancelled checkpoint tasks leaked")
    _check(source.closed and sink.closed, "cancelled runner did not close bindings")

    manifest = next((root / "manifests").glob("manifest-*.json"))
    saved = json.loads(manifest.read_text())
    _check(saved["format_version"] == 3, "checkpoint format changed")
    invalid_root = root.parent / "unknown-version"
    shutil.copytree(root, invalid_root)
    invalid_manifest = invalid_root / manifest.relative_to(root)
    invalid_manifest.write_text(json.dumps({**saved, "format_version": 999}))
    unopened = _Source(batch, pause=False)
    try:
        await _runner(invalid_root, unopened, _Sink()).start_async()
    except cf.StreamingRuntimeError as error:
        _check(error.category == "checkpoint_mismatch", "wrong recovery error category")
        _check(not unopened.opened, "unknown checkpoint version opened the source")
    else:
        raise AssertionError("unknown checkpoint version accepted")

    resumed = _Source(batch, pause=False)
    restored_sink = _Sink()
    second = await _runner(root, resumed, restored_sink).start_async()
    try:
        _check((await second.wait_async()).state == "completed", "restore failed")
    finally:
        await second.cancel_async()
    _check(resumed.resumed == 1, "checkpoint source cursor was not restored")
    await _check_total(restored_sink.batches, 6)
    _check(second.status()["task_count"] == 0, "restored checkpoint tasks leaked")
    _check(resumed.closed and restored_sink.closed, "restored bindings were not closed")


def check_recovery(batch: cf.Batch) -> None:
    Path("target").mkdir(exist_ok=True)
    with TemporaryDirectory(prefix="wheel-recovery-", dir="target") as temporary:
        asyncio.run(
            asyncio.wait_for(_exercise(Path(temporary) / "state", batch), timeout=30)
        )
