from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Awaitable, Mapping
from pathlib import Path

import pyarrow as pa

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DeliveryGuarantee,
    DisabledWatermarks,
    Idle,
    ManagedCheckpointRuntime,
    NativeWatermarkCapability,
    PipelineBuilder,
    ReplayPositioning,
    SinkBinding,
    SinkRecovery,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    StreamExecutionPlan,
    StreamingJob,
    StreamingRunner,
    StreamRequirements,
)

WORKER_OPERATION_TIMEOUT_SECONDS = 30.0
WORKER_CLEANUP_TIMEOUT_SECONDS = 5.0


def _vector() -> dict[str, object]:
    fixture = (
        Path(__file__).parents[2]
        / "tests"
        / "fixtures"
        / "a6"
        / "continuous_restart_vectors.json"
    )
    return json.loads(fixture.read_text(encoding="utf-8"))


def _commit_epoch(root: Path, epoch: int, values: list[int]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"visible-{epoch:020}.json"
    if target.exists():
        assert json.loads(target.read_text(encoding="utf-8")) == values
        return
    temporary = root / f".tmp-{epoch:020}.json"
    temporary.write_text(json.dumps(values), encoding="utf-8")
    temporary.replace(target)


def _visible_values(root: Path) -> list[int]:
    values: list[int] = []
    for path in sorted(root.glob("visible-*.json")):
        values.extend(json.loads(path.read_text(encoding="utf-8")))
    return values


def _temporary_artifacts(root: Path) -> int:
    return sum(
        path.is_file() and (".tmp" in path.name or path.name.startswith("tmp"))
        for path in root.rglob("*")
    )


async def _wait_for_worker_operation[T](
    job: StreamingJob,
    mode: str,
    operation: str,
    awaitable: Awaitable[T],
) -> T:
    try:
        async with asyncio.timeout(WORKER_OPERATION_TIMEOUT_SECONDS):
            return await awaitable
    except TimeoutError as timeout:
        status = job.status()
        try:
            async with asyncio.timeout(WORKER_CLEANUP_TIMEOUT_SECONDS):
                cleanup = await job.cancel_async()
        except BaseException as cleanup_error:
            cleanup = f"failed: {cleanup_error!r}"
        raise RuntimeError(
            f"Python cross-surface {mode} worker {operation} exceeded "
            f"{WORKER_OPERATION_TIMEOUT_SECONDS:.0f}s; status before cleanup: "
            f"{status!r}; cancellation cleanup: {cleanup!r}"
        ) from timeout


async def _run(mode: str, managed_root: Path, sink_root: Path) -> dict[str, object]:
    vector = _vector()
    plan_vector = vector["plan"]
    records = vector["records"]
    expected = vector["expected"]
    assert isinstance(plan_vector, dict)
    assert isinstance(records, list)
    assert isinstance(expected, dict)
    checkpoint_after = vector["checkpoint_after"]
    assert isinstance(checkpoint_after, int)
    writes_changed = asyncio.Condition()
    opened_offsets: list[int] = []
    lifecycle = {"source_closes": 0, "sink_closes": 0, "writes": 0}

    class Source:
        def __init__(self) -> None:
            self.offset = 0

        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities(
                ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                SourceDeliveryCapability.LOSSLESS,
                max_batch_rows=1,
                max_batch_bytes=1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, cursor: Cursor | None) -> None:
            self.offset = 0 if cursor is None else int(cursor.payload["offset"])
            opened_offsets.append(self.offset)

        async def next(self) -> Data | Idle | None:
            if mode == "stage" and self.offset == checkpoint_after:
                await asyncio.sleep(0)
                return Idle()
            if self.offset >= len(records):
                return None
            record = records[self.offset]
            assert isinstance(record, dict)
            assert record["offset"] == self.offset
            value = record["value"]
            assert isinstance(value, int)
            self.offset += 1
            return Data(
                Batch.from_pyarrow(pa.table({"value": [value]})),
                Cursor(self.offset.to_bytes(8, "big"), {"offset": self.offset}),
            )

        async def close(self) -> None:
            lifecycle["source_closes"] += 1

    class Sink:
        def __init__(self) -> None:
            self.pending: list[int] = []

        async def open(self) -> None:
            return None

        async def begin_epoch(self, epoch: int) -> None:
            self.pending.clear()

        async def write(self, batch: Batch) -> None:
            values = batch.to_pyarrow()["doubled"].to_pylist()
            self.pending.extend(values)
            async with writes_changed:
                lifecycle["writes"] += len(values)
                writes_changed.notify_all()

        async def pre_commit(self, epoch: int) -> dict[str, object]:
            return {"values": list(self.pending)}

        async def commit(self, epoch: int, pre_commit: Mapping[str, object]) -> None:
            raw_values = pre_commit["values"]
            assert isinstance(raw_values, list)
            await asyncio.to_thread(
                _commit_epoch, sink_root, epoch, [int(value) for value in raw_values]
            )

        async def abort(
            self, epoch: int, pre_commit: Mapping[str, object] | None
        ) -> None:
            self.pending.clear()

        async def recover(self, recovery: SinkRecovery) -> None:
            raw_values = recovery.pre_commit["values"]
            assert isinstance(raw_values, list)
            await asyncio.to_thread(
                _commit_epoch,
                sink_root,
                recovery.epoch,
                [int(value) for value in raw_values],
            )

        async def close(self) -> None:
            lifecycle["sink_closes"] += 1

    plan: StreamExecutionPlan = (
        PipelineBuilder(str(plan_vector["name"]))
        .expression(str(plan_vector["operator_id"]), str(plan_vector["expression"]))
        .compile_stream(
            requirements=StreamRequirements(
                {
                    str(plan_vector["output_id"]): DeliveryGuarantee.EXACTLY_ONCE,
                }
            )
        )
    )
    plan_fingerprint = plan.fingerprint
    job = await StreamingRunner(
        plan,
        {
            str(plan_vector["source_id"]): SourceBinding(
                Source(), watermark_policy=DisabledWatermarks()
            )
        },
        {
            str(plan_vector["output_id"]): [
                SinkBinding.transactional(str(plan_vector["sink_id"]), Sink())
            ]
        },
        ManagedCheckpointRuntime(managed_root),
    ).start_async()
    if mode == "stage":

        async def wait_for_staged_writes() -> None:
            async with writes_changed:
                await writes_changed.wait_for(
                    lambda: lifecycle["writes"] >= checkpoint_after
                )

        await _wait_for_worker_operation(
            job, mode, "wait for staged sink writes", wait_for_staged_writes()
        )
        epoch = await _wait_for_worker_operation(
            job, mode, "manual checkpoint", job.trigger_checkpoint_async()
        )
        assert epoch == expected["checkpoint_epoch"]
        outcome = await _wait_for_worker_operation(
            job, mode, "cancellation cleanup", job.cancel_async()
        )
    else:
        outcome = await _wait_for_worker_operation(
            job, mode, "terminal completion", job.wait_async()
        )
    status = job.status()
    charged_edges = sum(
        edge["current_envelopes"] != 0
        or edge["current_rows"] != 0
        or edge["current_bytes"] != 0
        for edge in status["edges"].values()
    )
    return {
        "surface": "python",
        "mode": mode,
        "plan_fingerprint": plan_fingerprint,
        "opened_offset": opened_offsets[0],
        "outcome_state": outcome.state,
        "completed_epoch": outcome.completed_epoch,
        "task_count": status["task_count"],
        "charged_edges": charged_edges,
        "source_closes": lifecycle["source_closes"],
        "sink_closes": lifecycle["sink_closes"],
        "visible_values": await asyncio.to_thread(_visible_values, sink_root),
        "temporary_artifacts": await asyncio.to_thread(
            _temporary_artifacts, managed_root.parent
        ),
    }


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(
            "usage: a6_cross_surface_worker.py <stage|resume> <managed-root> "
            "<sink-root> <report>"
        )
    mode, managed_root, sink_root, report = sys.argv[1:]
    if mode not in ("stage", "resume"):
        raise SystemExit("worker mode must be stage or resume")
    result = asyncio.run(_run(mode, Path(managed_root), Path(sink_root)))
    Path(report).write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
