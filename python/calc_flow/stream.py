"""Owned asynchronous expression results from the native streaming runtime."""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, fields, replace
from types import TracebackType
from typing import cast

import pyarrow as pa

from calc_flow._native import Batch, StreamingRuntimeError
from calc_flow._stream_inputs import (
    StreamInput as StreamInput,
)
from calc_flow._stream_inputs import (
    _capture_watermarks,
    _IterableSource,
    _selected_policies,
    _source_binding,
)
from calc_flow.compute import TableData, _input_batch
from calc_flow.pipeline import Runtime, StreamExecutionPlan, _canonical
from calc_flow.runtime import (
    EdgeBudget,
    JobOutcome,
    ManagedCheckpointRuntime,
    SinkBinding,
    SourceBinding,
    StreamingJob,
    StreamingRunner,
    StreamRuntimeConfig,
    WatermarkPolicy,
    _finish_cleanup,
    _runner_config,
)
from calc_flow.symbolic.expr import Parameter, TableExpr
from calc_flow.symbolic.lower.bindings import _BatchBindings
from calc_flow.symbolic.program import Program, _node_name, _selected_runtime


@dataclass(frozen=True, slots=True)
class StreamOutput:
    """One logically named output; independent outputs need not arrive together."""

    name: str
    table: pa.Table


@dataclass(frozen=True, slots=True)
class _StreamRequest:
    program: Program
    inputs: dict[str, StreamInput | TableData]
    runtime: Runtime | None
    config: StreamRuntimeConfig | None
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None


class _QueueSink:
    def __init__(
        self,
        name: str,
        queue: asyncio.Queue[StreamOutput],
        ready: asyncio.Event,
    ) -> None:
        self._name = name
        self._queue = queue
        self._ready = ready

    async def open(self) -> None:
        pass

    async def write(self, batch: Batch) -> None:
        await self._queue.put(StreamOutput(self._name, batch.to_pyarrow()))
        self._ready.set()

    async def close(self) -> None:
        pass


def _validate_request(request: _StreamRequest) -> dict[str, TableExpr | Parameter]:
    expected = {_node_name(value._node): value for value in request.program.inputs}
    for name in expected:
        if name not in request.inputs:
            raise ValueError(f"stream.inputs.{name}: missing input")
    for name in request.inputs:
        if name not in expected:
            raise ValueError(f"stream.inputs.{name}: unexpected input name")
    for name, output in request.program.outputs:
        if not isinstance(output, TableExpr):
            raise TypeError(f"stream.outputs.{name}: expected TableExpr")
    return expected


def _compile_stream(
    request: _StreamRequest,
) -> tuple[StreamExecutionPlan, dict[str, tuple[str, ...]], dict[str, tuple[str, str]]]:
    from calc_flow.symbolic.lower.program import lower_program_document

    selected = _selected_runtime(request.runtime)
    bindings = _BatchBindings()
    document = lower_program_document(
        request.program, selected, "stream", _bindings=bindings
    )
    names, outputs = bindings.names()
    reserved = {node["id"] for node in document["graph"]["nodes"]}
    sinks = {}
    for index, (logical, physical) in enumerate(outputs.items()):
        sink_id = f"stream_sink_{index}"
        while sink_id in reserved:
            sink_id += "_"
        reserved.add(sink_id)
        sinks[logical] = (physical, sink_id)
    return selected._compile_stream_graph_project(_canonical(document)), names, sinks


def _outcome_error(
    outcome: JobOutcome,
    failure: str | None,
) -> StreamingRuntimeError:
    if not outcome.errors:
        return StreamingRuntimeError(f"stream: native job ended in {outcome.state}")
    detail = outcome.errors[0]
    if failure is not None:
        detail = replace(detail, message=failure)
    error = StreamingRuntimeError(detail.message)
    error._calc_flow_native_safe_fields = tuple(
        getattr(detail, field.name) for field in fields(detail)
    )
    return error


class StreamResults[T]:
    """One-shot stream owned by ``async with``; iterate only inside its context.

    Ordinary iterables have best-effort delivery and temporary checkpoints, with
    no restart guarantee. Use an explicit runner for durable recovery or sinks.
    """

    def __init__(self, request: _StreamRequest, *, table_output: bool) -> None:
        self._request = request
        self._table_output = table_output
        self._entered = False
        self._closed = False
        self._next_busy = False
        self._job: StreamingJob | None = None
        self._starting: asyncio.Task[None] | None = None
        self._waiter: asyncio.Task[JobOutcome] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[StreamOutput] = asyncio.Queue(maxsize=1)
        self._ready = asyncio.Event()
        self._adapters: list[_IterableSource] = []
        self._root: str | None = None

    @property
    def job(self) -> StreamingJob:
        """The owned native job, available after successful context entry."""
        if self._job is None:
            raise RuntimeError("stream.job: enter with async with before accessing job")
        return self._job

    async def __aenter__(self) -> StreamResults[T]:
        if self._entered or self._closed:
            raise RuntimeError("stream: a result context may be entered only once")
        self._entered = True
        self._starting = asyncio.create_task(self._start())
        try:
            await asyncio.shield(self._starting)
            if self._closed:
                raise asyncio.CancelledError("stream: closed during context entry")
        except BaseException:
            await _finish_cleanup(self.aclose())
            raise
        return self

    async def _start(self) -> None:
        expected = _validate_request(self._request)
        policies = _selected_policies(
            self._request.inputs, expected, self._request.watermarks
        )
        config = _runner_config(self._request.config)
        config._native()
        plan, names, outputs = _compile_stream(self._request)
        sources, static = self._bindings(expected, names, config.edge_budget, policies)
        sinks = {
            outputs[name][0]: [
                SinkBinding.ordinary(
                    outputs[name][1], _QueueSink(name, self._queue, self._ready)
                )
            ]
            for name, _ in self._request.program.outputs
        }
        await self._create_root()
        checkpoints = ManagedCheckpointRuntime(self._root)
        runner = StreamingRunner(
            plan, sources, sinks, checkpoints, config=config, static_inputs=static
        )
        self._job = await runner.start_async()
        self._waiter = asyncio.create_task(self._job.wait_async())
        self._waiter.add_done_callback(lambda _: self._ready.set())

    def _bindings(
        self,
        expected: dict[str, TableExpr | Parameter],
        names: dict[str, tuple[str, ...]],
        budget: EdgeBudget,
        policies: Mapping[str, WatermarkPolicy],
    ) -> tuple[dict[str, SourceBinding], dict[str, Batch]]:
        sources: dict[str, SourceBinding] = {}
        static: dict[str, Batch] = {}
        for name, value in expected.items():
            data = self._request.inputs[name]
            if isinstance(value, Parameter):
                static[name] = _input_batch(data, value, f"stream.inputs.{name}")
            else:
                binding, adapter = _source_binding(
                    data, name, value, budget, policies.get(name)
                )
                ports = names.get(name, ())
                if len(ports) != 1:
                    raise ValueError(
                        f"stream.inputs.{name}: expected one native ingress"
                    )
                sources[ports[0]] = binding
                if adapter is not None:
                    self._adapters.append(adapter)
        return sources, static

    async def _create_root(self) -> None:
        task = asyncio.create_task(
            asyncio.to_thread(tempfile.mkdtemp, prefix="calc-flow-")
        )
        try:
            self._root = await asyncio.shield(task)
        except asyncio.CancelledError:

            async def capture() -> None:
                self._root = await task

            await _finish_cleanup(capture())
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            await self.aclose()
        except BaseException:
            if exc is None:
                raise

    def __aiter__(self) -> AsyncIterator[T]:
        self._require_entered()
        return self

    def _require_entered(self) -> None:
        if self._job is None:
            raise RuntimeError("stream: enter with async with before iteration")

    async def __anext__(self) -> T:
        self._require_entered()
        if self._next_busy:
            raise RuntimeError("stream: concurrent iteration is not supported")
        if self._closed:
            raise StopAsyncIteration
        self._next_busy = True
        try:
            output = await self._next_output()
            return cast(T, output.table if self._table_output else output)
        except asyncio.CancelledError:
            await _finish_cleanup(self.aclose())
            raise
        finally:
            self._next_busy = False

    async def _next_output(self) -> StreamOutput:
        if self._waiter is None:
            raise RuntimeError("stream: enter with async with before iteration")
        while self._queue.empty():
            if self._waiter.done():
                await self.aclose()
                raise StopAsyncIteration
            await self._ready.wait()
            self._ready.clear()
        return self._queue.get_nowait()

    def _raise_failure(self, outcome: JobOutcome | None) -> None:
        if outcome is not None and outcome.state in ("failed", "recovery_required"):
            failure = next(
                (adapter.failure for adapter in self._adapters if adapter.failure), None
            )
            raise _outcome_error(outcome, failure)

    async def aclose(self) -> None:
        """Cancel live work and await source, task and temporary-state cleanup."""
        if self._closing is None:
            self._closed = True
            self._closing = asyncio.create_task(self._close())
        try:
            await asyncio.shield(self._closing)
        except asyncio.CancelledError:
            await _finish_cleanup(self._closing)
            raise

    async def _close(self) -> None:
        try:
            await self._stop_start()
        finally:
            try:
                outcome = await self._stop_job()
            finally:
                await self._close_sources()
        self._raise_failure(outcome)

    async def _stop_start(self) -> None:
        if self._starting is None:
            return
        if not self._starting.done():
            self._starting.cancel()
        # Context entry reports startup errors; close waits for its owned work.
        await asyncio.gather(self._starting, return_exceptions=True)

    async def _stop_job(self) -> JobOutcome | None:
        if self._job is None:
            return None
        if self._waiter is None:
            return await self._job.cancel_async()
        if not self._waiter.done():
            await self._job.cancel_async()
        return await self._waiter

    async def _close_sources(self) -> None:
        try:
            for adapter in self._adapters:
                await adapter.close()
        finally:
            if self._root is not None:
                await asyncio.to_thread(shutil.rmtree, self._root)


def _stream_program(
    program: Program,
    inputs: Mapping[str, StreamInput | TableData],
    runtime: Runtime | None,
    config: StreamRuntimeConfig | None,
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
) -> StreamResults[StreamOutput]:
    if not isinstance(inputs, Mapping):
        raise TypeError("stream.inputs: expected a mapping by declared input name")
    return StreamResults(
        _StreamRequest(
            program, dict(inputs), runtime, config, _capture_watermarks(watermarks)
        ),
        table_output=False,
    )


def _stream_table(
    table: TableExpr,
    inputs: StreamInput | Mapping[str, StreamInput | TableData],
    runtime: Runtime | None,
    config: StreamRuntimeConfig | None,
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
) -> StreamResults[pa.Table]:
    program = Program("stream", outputs={"result": table})
    if not isinstance(inputs, Mapping):
        if len(program.inputs) != 1 or not isinstance(program.inputs[0], TableExpr):
            raise ValueError("stream: multiple or static inputs require a name mapping")
        inputs = {_node_name(program.inputs[0]._node): inputs}
    return StreamResults(
        _StreamRequest(
            program, dict(inputs), runtime, config, _capture_watermarks(watermarks)
        ),
        table_output=True,
    )
