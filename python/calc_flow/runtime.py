from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, NoReturn, Protocol

from calc_flow import _native
from calc_flow.store import FileCheckpointStore, _copy_json_value, _run_blocking


class Source(Protocol):
    def open(self, cursor: object) -> object: ...

    def next(self) -> object: ...


async def _resolve(value: object) -> object:
    if inspect.isawaitable(value):
        return await value
    return value


async def _raise_after_cancellation_cleanup(
    cleanup: Awaitable[object], cancellation: asyncio.CancelledError
) -> NoReturn:
    async def run_cleanup() -> None:
        await cleanup

    cleanup_task = asyncio.create_task(run_cleanup())
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            continue
    try:
        cleanup_task.result()
    except BaseException as cleanup_error:
        raise cleanup_error from cancellation
    raise cancellation


class _SourceAdapter:
    __slots__ = ("_source",)

    def __init__(self, source: Source) -> None:
        for method in ("open", "next"):
            if not callable(getattr(source, method, None)):
                raise TypeError(f"source must provide callable {method}()")
        self._source = source

    async def open(self, cursor: object) -> None:
        copied = _copy_json_value(cursor, root_mapping=False, label="source cursor")
        await _resolve(self._source.open(copied))

    async def next(self) -> tuple[_native.Batch, object, int] | None:
        item = await _resolve(self._source.next())
        if item is None:
            return None
        if type(item) is not tuple or len(item) != 3:
            raise TypeError(
                "source next() must return None or exactly (Batch, cursor, sequence)"
            )
        batch, cursor, sequence = item
        if not isinstance(batch, _native.Batch):
            raise TypeError("source item 0 must be a calc_flow.Batch")
        copied_cursor = _copy_json_value(
            cursor, root_mapping=False, label="source cursor"
        )
        if type(sequence) is not int or not 0 <= sequence <= 2**64 - 1:
            raise TypeError("source sequence must be a non-negative u64 integer")
        return batch, copied_cursor, sequence


def _sink_mapping(
    sinks: Mapping[str, Sequence[Callable[[_native.Batch], object]]] | None,
) -> dict[str, list[Callable[[_native.Batch], object]]]:
    if sinks is None:
        return {}
    if not isinstance(sinks, Mapping):
        raise TypeError("sinks must be a mapping of output names to callback sequences")
    copied: dict[str, list[Callable[[_native.Batch], object]]] = {}
    for output, callbacks in sinks.items():
        if not isinstance(output, str):
            raise TypeError("sink output names must be strings")
        if isinstance(callbacks, (str, bytes)) or not isinstance(callbacks, Sequence):
            raise TypeError(f"sinks[{output!r}] must be a sequence of callables")
        copied_callbacks = list(callbacks)
        if not all(callable(callback) for callback in copied_callbacks):
            raise TypeError(f"sinks[{output!r}] must contain only callables")
        copied[output] = copied_callbacks
    return copied


class MicroBatchRunner:
    __slots__ = ("_inner", "__weakref__")

    def __init__(
        self,
        plan: Any,
        source: Source,
        checkpoints: FileCheckpointStore,
        *,
        sinks: Mapping[str, Sequence[Callable[[_native.Batch], object]]] | None = None,
        checkpoint_every: int = 100,
    ) -> None:
        from calc_flow.pipeline import ExecutionPlan

        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be a calc_flow.ExecutionPlan")
        if not isinstance(checkpoints, FileCheckpointStore):
            raise TypeError("checkpoints must be a calc_flow.FileCheckpointStore")
        if type(checkpoint_every) is not int or checkpoint_every <= 0:
            raise ValueError("checkpoint_every must be a positive integer")
        adapter = _SourceAdapter(source)
        copied_sinks = _sink_mapping(sinks)
        self._inner = _native._MicroBatchRunner(
            plan._inner,
            adapter,
            checkpoints._inner,
            copied_sinks,
            checkpoint_every,
        )

    async def next_async(self) -> _native.RunResult | None:
        try:
            return await self._inner.next_async()
        except asyncio.CancelledError as cancellation:
            await _raise_after_cancellation_cleanup(
                self._inner.wait_idle_async(), cancellation
            )

    async def reset_async(self) -> None:
        await self._inner.reset_async()

    async def plan_snapshot_async(self) -> dict[str, Any]:
        state = await self._inner.plan_snapshot_async()
        return _copy_json_value(state, root_mapping=True, label="plan state")

    def next(self) -> _native.RunResult | None:
        return _run_blocking(self.next_async, "next_async")

    def reset(self) -> None:
        return _run_blocking(self.reset_async, "reset_async")

    def plan_snapshot(self) -> dict[str, Any]:
        return _run_blocking(self.plan_snapshot_async, "plan_snapshot_async")


class StreamingRunner:
    __slots__ = ("_inner", "__weakref__")

    def __init__(self, plan: Any, checkpoints: FileCheckpointStore) -> None:
        from calc_flow.pipeline import ExecutionPlan

        if not isinstance(plan, ExecutionPlan):
            raise TypeError("plan must be a calc_flow.ExecutionPlan")
        if not isinstance(checkpoints, FileCheckpointStore):
            raise TypeError("checkpoints must be a calc_flow.FileCheckpointStore")
        self._inner = _native._StreamingRunner(plan._inner, checkpoints._inner)

    def step_async(
        self,
        batch: _native.Batch,
        *,
        sinks: Mapping[str, Sequence[Callable[[_native.Batch], object]]] | None = None,
    ) -> Awaitable[_native.RunResult]:
        if not isinstance(batch, _native.Batch):
            raise TypeError("batch must be a calc_flow.Batch")
        copied_sinks = _sink_mapping(sinks)

        async def step() -> _native.RunResult:
            try:
                return await self._inner.step_async(batch, copied_sinks)
            except asyncio.CancelledError as cancellation:
                await _raise_after_cancellation_cleanup(
                    self._inner.wait_idle_async(), cancellation
                )

        return step()

    async def reset_async(self) -> None:
        await self._inner.reset_async()

    async def plan_snapshot_async(self) -> dict[str, Any]:
        state = await self._inner.plan_snapshot_async()
        return _copy_json_value(state, root_mapping=True, label="plan state")

    def step(
        self,
        batch: _native.Batch,
        *,
        sinks: Mapping[str, Sequence[Callable[[_native.Batch], object]]] | None = None,
    ) -> _native.RunResult:
        return _run_blocking(lambda: self.step_async(batch, sinks=sinks), "step_async")

    def reset(self) -> None:
        return _run_blocking(self.reset_async, "reset_async")

    def plan_snapshot(self) -> dict[str, Any]:
        return _run_blocking(self.plan_snapshot_async, "plan_snapshot_async")
