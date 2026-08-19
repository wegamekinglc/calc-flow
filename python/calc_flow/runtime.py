from __future__ import annotations

import asyncio
import concurrent.futures
import inspect
import os
import threading
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, NoReturn, Protocol, TypedDict

from calc_flow import _native
from calc_flow.store import _copy_json_value

if TYPE_CHECKING:
    from calc_flow.pipeline import StreamExecutionPlan

type JSONValue = (
    None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]
)


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


async def _finish_cleanup(cleanup: Awaitable[object]) -> None:
    """Finish already-linearized terminal cleanup despite observer cancellation."""

    owner = asyncio.current_task()

    async def run_cleanup() -> None:
        await cleanup

    cleanup_task = asyncio.create_task(run_cleanup())
    while not cleanup_task.done():
        try:
            await asyncio.shield(cleanup_task)
        except asyncio.CancelledError:
            if owner is not None:
                while owner.cancelling():
                    owner.uncancel()
            continue
    cleanup_task.result()


class ReplayPositioning(StrEnum):
    """Replay protocol implemented by a stream source."""

    EXACT_PAUSE_REPORT_AND_SEEK = "exact_pause_report_and_seek"
    UNSUPPORTED = "unsupported"


class NativeWatermarkCapability(StrEnum):
    """Native watermark behavior declared by a stream source."""

    NEVER_EMITS = "never_emits"
    EMITS_NATIVE = "emits_native"
    RUNTIME_TOGGLEABLE = "runtime_toggleable"
    UNKNOWN = "unknown"


class SourceDeliveryCapability(StrEnum):
    """Whether admitted source events can be lost before observation."""

    LOSSLESS = "lossless"
    LOSSY = "lossy"


@dataclass(frozen=True, slots=True)
class SourceProvidedWatermarks:
    pass


@dataclass(frozen=True, slots=True)
class BoundedOutOfOrderness:
    event_time_column: str
    max_out_of_orderness: timedelta
    emit_interval: timedelta
    idle_timeout: timedelta | None = None


@dataclass(frozen=True, slots=True)
class DisabledWatermarks:
    idle_timeout: timedelta | None = None


type WatermarkPolicy = (
    SourceProvidedWatermarks | BoundedOutOfOrderness | DisabledWatermarks
)


@dataclass(frozen=True, slots=True)
class OrdinaryDelivery:
    pass


@dataclass(frozen=True, slots=True)
class EpochIdempotentDelivery:
    mechanism: str
    retention: Literal["bounded", "unbounded"]


@dataclass(frozen=True, slots=True)
class TransactionalDelivery:
    pass


type SinkDelivery = OrdinaryDelivery | EpochIdempotentDelivery | TransactionalDelivery


def _frozen_json_mapping(value: Mapping[str, object], label: str) -> Mapping[str, Any]:
    copied = _copy_json_value(dict(value), root_mapping=True, label=label)
    return MappingProxyType(copied)


@dataclass(frozen=True, slots=True)
class Cursor:
    """Owned or unbound immutable source replay position."""

    order: bytes
    payload: Mapping[str, Any]
    source_id: str | None = None

    def __post_init__(self) -> None:
        if type(self.order) is not bytes:
            raise TypeError("cursor order must be bytes")
        if not self.order:
            raise ValueError("cursor order must not be empty")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id
        ):
            raise TypeError("cursor source_id must be a non-empty string or None")
        object.__setattr__(
            self, "payload", _frozen_json_mapping(self.payload, "cursor payload")
        )


@dataclass(frozen=True, slots=True)
class SourceCapabilities:
    """Source capability descriptor sampled once before connector open."""

    replay_positioning: ReplayPositioning
    delivery: SourceDeliveryCapability
    max_batch_rows: int
    max_batch_bytes: int
    schema: object | None = None
    native_watermarks: NativeWatermarkCapability = (
        NativeWatermarkCapability.EMITS_NATIVE
    )

    def __post_init__(self) -> None:
        if not isinstance(self.replay_positioning, ReplayPositioning):
            raise TypeError("replay_positioning must be a ReplayPositioning value")
        if not isinstance(self.delivery, SourceDeliveryCapability):
            raise TypeError("delivery must be a SourceDeliveryCapability value")
        for name in ("max_batch_rows", "max_batch_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.native_watermarks, NativeWatermarkCapability):
            raise TypeError(
                "native_watermarks must be a NativeWatermarkCapability value"
            )


@dataclass(frozen=True, slots=True)
class Data:
    """One source batch paired with its replay cursor."""

    batch: _native.Batch
    cursor: Cursor

    def __post_init__(self) -> None:
        if not isinstance(self.batch, _native.Batch):
            raise TypeError("data batch must be a calc_flow.Batch")
        if not isinstance(self.cursor, Cursor):
            raise TypeError("data cursor must be a calc_flow.Cursor")


@dataclass(frozen=True, slots=True)
class Watermark:
    """Timezone-aware source-native event-time watermark."""

    at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ValueError("watermark datetime must be timezone-aware")
        object.__setattr__(self, "at", self.at.astimezone(UTC))


@dataclass(frozen=True, slots=True)
class Idle:
    pass


type SourceEvent = Data | Watermark | Idle


@dataclass(frozen=True, slots=True)
class SinkRecovery:
    epoch: int
    terminal: bool
    delivery: SinkDelivery
    pre_commit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pre_commit",
            _frozen_json_mapping(self.pre_commit, "sink recovery pre_commit"),
        )


def _require_async_methods(
    connector: object, kind: str, methods: Sequence[str]
) -> None:
    for method in methods:
        callback = getattr(connector, method, None)
        if callback is None or not inspect.iscoroutinefunction(callback):
            raise TypeError(f"{kind}.{method} must be declared with async def")


def _duration_micros(value: timedelta, name: str) -> int:
    if not isinstance(value, timedelta):
        raise TypeError(f"{name} must be a datetime.timedelta")
    micros = (
        value.days * 86_400_000_000 + value.seconds * 1_000_000 + value.microseconds
    )
    if micros < 0:
        raise ValueError(f"{name} must not be negative")
    return micros


def _datetime_micros(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - datetime(1970, 1, 1, tzinfo=UTC)
    return delta.days * 86_400_000_000 + delta.seconds * 1_000_000 + delta.microseconds


class StreamSource(Protocol):
    """Async-only source connector consumed by a continuous runner."""

    def capabilities(self) -> SourceCapabilities: ...
    async def open(self, cursor: Cursor | None) -> None: ...
    async def next(self) -> Data | Watermark | Idle | None: ...
    async def close(self) -> None: ...


class StreamSink(Protocol):
    """Async-only ordinary at-least-once sink connector."""

    async def open(self) -> None: ...
    async def write(self, batch: _native.Batch) -> None: ...
    async def close(self) -> None: ...


class TransactionalStreamSink(Protocol):
    """Async-only transactional or epoch-idempotent sink connector."""

    async def open(self) -> None: ...
    async def begin_epoch(self, epoch: int) -> None: ...
    async def write(self, batch: _native.Batch) -> None: ...
    async def pre_commit(self, epoch: int) -> Mapping[str, JSONValue]: ...
    async def commit(self, epoch: int, pre_commit: Mapping[str, JSONValue]) -> None: ...
    async def abort(
        self, epoch: int, pre_commit: Mapping[str, JSONValue] | None
    ) -> None: ...
    async def recover(self, recovery: SinkRecovery) -> None: ...
    async def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceBinding:
    """Own a source connector and its immutable watermark policy."""

    source: StreamSource = field(repr=False)
    watermark_policy: WatermarkPolicy = field(default_factory=SourceProvidedWatermarks)

    def __init__(
        self, source: StreamSource, *, watermark_policy: WatermarkPolicy | None = None
    ) -> None:
        if not callable(getattr(source, "capabilities", None)):
            raise TypeError("source.capabilities must be callable")
        _require_async_methods(source, "source", ("open", "next", "close"))
        selected = (
            SourceProvidedWatermarks() if watermark_policy is None else watermark_policy
        )
        if not isinstance(
            selected,
            (SourceProvidedWatermarks, BoundedOutOfOrderness, DisabledWatermarks),
        ):
            raise TypeError("watermark_policy is not a supported watermark policy")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "watermark_policy", selected)

    def _native_capabilities(self) -> dict[str, object]:
        value = self.source.capabilities()
        if not isinstance(value, SourceCapabilities):
            raise TypeError("source.capabilities() must return SourceCapabilities")
        return {
            "replay_positioning": value.replay_positioning.value,
            "delivery": value.delivery.value,
            "max_batch_rows": value.max_batch_rows,
            "max_batch_bytes": value.max_batch_bytes,
            "schema": value.schema,
            "native_watermarks": value.native_watermarks.value,
        }

    def _native_policy(self) -> dict[str, object]:
        policy = self.watermark_policy
        if isinstance(policy, SourceProvidedWatermarks):
            return {"kind": "source_provided"}
        if isinstance(policy, BoundedOutOfOrderness):
            if not policy.event_time_column:
                raise ValueError("event_time_column must not be empty")
            return {
                "kind": "bounded_out_of_orderness",
                "event_time_column": policy.event_time_column,
                "max_out_of_orderness_micros": _duration_micros(
                    policy.max_out_of_orderness, "max_out_of_orderness"
                ),
                "emit_interval_micros": _duration_micros(
                    policy.emit_interval, "emit_interval"
                ),
                "idle_timeout_micros": (
                    None
                    if policy.idle_timeout is None
                    else _duration_micros(policy.idle_timeout, "idle_timeout")
                ),
            }
        return {
            "kind": "disabled",
            "idle_timeout_micros": (
                None
                if policy.idle_timeout is None
                else _duration_micros(policy.idle_timeout, "idle_timeout")
            ),
        }

    async def _native_open(
        self,
        source_id: str | None,
        order: bytes | None,
        payload: Mapping[str, object] | None,
    ) -> None:
        cursor = (
            None
            if order is None
            else Cursor(order, {} if payload is None else payload, source_id)
        )
        await self.source.open(cursor)

    async def _native_next(self) -> tuple[object, ...] | None:
        value = await self.source.next()
        if value is None:
            return None
        if isinstance(value, Data):
            return (
                "data",
                value.batch,
                value.cursor.source_id,
                value.cursor.order,
                dict(value.cursor.payload),
            )
        if isinstance(value, Watermark):
            return ("watermark", _datetime_micros(value.at))
        if isinstance(value, Idle):
            return ("idle",)
        raise TypeError("source.next() must return Data, Watermark, Idle, or None")

    async def _native_close(self) -> None:
        await self.source.close()


@dataclass(frozen=True, slots=True)
class SinkBinding:
    """Own a named sink connector and frozen delivery evidence."""

    sink_id: str
    sink: object = field(repr=False)
    delivery: SinkDelivery

    @classmethod
    def ordinary(cls, sink_id: str, sink: StreamSink) -> SinkBinding:
        _require_async_methods(sink, "sink", ("open", "write", "close"))
        return cls(sink_id, sink, OrdinaryDelivery())

    @classmethod
    def transactional(cls, sink_id: str, sink: TransactionalStreamSink) -> SinkBinding:
        _require_async_methods(
            sink,
            "sink",
            (
                "open",
                "begin_epoch",
                "write",
                "pre_commit",
                "commit",
                "abort",
                "recover",
                "close",
            ),
        )
        return cls(sink_id, sink, TransactionalDelivery())

    @classmethod
    def epoch_idempotent(
        cls,
        sink_id: str,
        sink: TransactionalStreamSink,
        *,
        mechanism: str,
        retention: Literal["bounded", "unbounded"],
    ) -> SinkBinding:
        binding = cls.transactional(sink_id, sink)
        if not mechanism:
            raise ValueError("mechanism must not be empty")
        if retention not in ("bounded", "unbounded"):
            raise ValueError("retention must be 'bounded' or 'unbounded'")
        return cls(
            binding.sink_id,
            binding.sink,
            EpochIdempotentDelivery(mechanism, retention),
        )

    def __post_init__(self) -> None:
        if not isinstance(self.sink_id, str) or not self.sink_id:
            raise TypeError("sink_id must be a non-empty string")

    def _native_descriptor(self) -> dict[str, object]:
        if isinstance(self.delivery, OrdinaryDelivery):
            return {"kind": "ordinary"}
        if isinstance(self.delivery, TransactionalDelivery):
            return {"kind": "transactional"}
        return {
            "kind": "epoch_idempotent",
            "mechanism": self.delivery.mechanism,
            "retention": self.delivery.retention,
        }

    async def _native_open(self) -> None:
        await self.sink.open()

    async def _native_write(self, batch: _native.Batch) -> None:
        await self.sink.write(batch)

    async def _native_begin_epoch(self, epoch: int) -> None:
        await self.sink.begin_epoch(epoch)

    async def _native_pre_commit(self, epoch: int) -> dict[str, Any]:
        value = await self.sink.pre_commit(epoch)
        return dict(_frozen_json_mapping(value, "sink pre_commit"))

    async def _native_commit(
        self, epoch: int, pre_commit: Mapping[str, object]
    ) -> None:
        await self.sink.commit(
            epoch, _frozen_json_mapping(pre_commit, "sink pre_commit")
        )

    async def _native_abort(
        self, epoch: int, pre_commit: Mapping[str, object] | None
    ) -> None:
        copied = (
            None
            if pre_commit is None
            else _frozen_json_mapping(pre_commit, "sink pre_commit")
        )
        await self.sink.abort(epoch, copied)

    async def _native_recover(
        self,
        epoch: int,
        terminal: bool,
        delivery: Mapping[str, object],
        pre_commit: Mapping[str, object],
    ) -> None:
        kind = delivery["kind"]
        if kind == "ordinary":
            selected: SinkDelivery = OrdinaryDelivery()
        elif kind == "transactional":
            selected = TransactionalDelivery()
        else:
            selected = EpochIdempotentDelivery(
                str(delivery["mechanism"]),
                delivery["retention"],  # type: ignore[arg-type]
            )
        await self.sink.recover(SinkRecovery(epoch, terminal, selected, pre_commit))

    async def _native_close(self) -> None:
        await self.sink.close()


@dataclass(frozen=True, slots=True)
class EdgeBudget:
    """Row and byte bounds applied independently to every stream edge."""

    max_rows: int = 10_000
    max_bytes: int = 64 << 20

    def __post_init__(self) -> None:
        for name in ("max_rows", "max_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class StreamRuntimeConfig:
    """Immutable runtime tuning excluded from the plan fingerprint."""

    checkpoint_interval: timedelta = timedelta(seconds=60)
    checkpoint_timeout: timedelta = timedelta(minutes=10)
    edge_budget: EdgeBudget = EdgeBudget()
    retained_epochs: int = 2

    def _native(self) -> dict[str, int]:
        if not isinstance(self.edge_budget, EdgeBudget):
            raise TypeError("edge_budget must be a calc_flow.EdgeBudget")
        if type(self.retained_epochs) is not int or self.retained_epochs <= 0:
            raise ValueError("retained_epochs must be a positive integer")
        return {
            "checkpoint_interval_micros": _duration_micros(
                self.checkpoint_interval, "checkpoint_interval"
            ),
            "checkpoint_timeout_micros": _duration_micros(
                self.checkpoint_timeout, "checkpoint_timeout"
            ),
            "edge_max_rows": self.edge_budget.max_rows,
            "edge_max_bytes": self.edge_budget.max_bytes,
            "retained_epochs": self.retained_epochs,
        }


class ManagedCheckpointRuntime:
    """Capture one local root for managed state and manifest storage."""

    __slots__ = ("_inner",)

    def __init__(self, directory: os.PathLike[str] | str, /) -> None:
        path = os.fspath(directory)
        if not isinstance(path, str):
            raise TypeError("checkpoint directory must resolve to a string path")
        self._inner = _native._ManagedCheckpointRuntime(path)


@dataclass(frozen=True, slots=True)
class StreamingError:
    """Payload-safe structured terminal error projection."""

    category: str
    message: str
    job_id: int | None
    epoch: int | None
    checkpoint_phase: str | None
    component_kind: str | None
    component_id: str | None
    diagnostic_id: int | None
    position: int


@dataclass(frozen=True, slots=True)
class JobOutcome:
    """Immutable terminal outcome returned by job lifecycle methods."""

    state: str
    cause: str
    completed_epoch: int | None
    errors: tuple[StreamingError, ...]


class OutputDeliveryStatus(TypedDict):
    requested: Literal["best_effort", "at_least_once", "exactly_once"]
    effective: Literal["best_effort", "at_least_once", "exactly_once"]


class JobStatus(TypedDict):
    job_id: int
    state: Literal[
        "running",
        "draining",
        "completed",
        "cancelled",
        "failed",
        "recovery_required",
    ]
    terminal_cause: str | None
    delivery: dict[str, OutputDeliveryStatus]
    task_count: int
    task_errors: int
    metrics_overflowed: bool
    watermark_micros: int | None
    edges: dict[str, dict[str, object]]
    sources: dict[str, dict[str, object]]
    operators: dict[str, dict[str, object]]
    sinks: dict[str, dict[str, object]]
    checkpoint: dict[str, object]


def _outcome(value: Mapping[str, object]) -> JobOutcome:
    errors = tuple(StreamingError(**error) for error in value["errors"])  # type: ignore[arg-type]
    return JobOutcome(
        state=str(value["state"]),
        cause=str(value["cause"]),
        completed_epoch=value["completed_epoch"],  # type: ignore[arg-type]
        errors=errors,
    )


def _reject_active_loop(method: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{method}() cannot run inside an event loop; use {method}_async()"
    )


class _BlockingEventLoop:
    __slots__ = ("_closed", "_loop", "_ready", "_thread")

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._closed = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="calc-flow-continuous",
            daemon=True,
        )
        self._thread.start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()
            self._closed.set()

    async def _invoke[T](self, factory: Callable[[], Awaitable[T]]) -> T:
        return await factory()

    def _submit[T](
        self, factory: Callable[[], Awaitable[T]]
    ) -> concurrent.futures.Future[T]:
        if self._closed.is_set():
            raise RuntimeError("the calc-flow continuous event loop is closed")
        return asyncio.run_coroutine_threadsafe(self._invoke(factory), self._loop)

    def run[T](self, factory: Callable[[], Awaitable[T]]) -> T:
        return self._submit(factory).result()

    async def run_async[T](self, factory: Callable[[], Awaitable[T]]) -> T:
        if self.owns_current_thread():
            return await factory()
        return await asyncio.wrap_future(self._submit(factory))

    def close_after(
        self,
        factory: Callable[[], Awaitable[object]],
        release: Callable[[], object],
    ) -> None:
        async def invoke() -> object:
            try:
                await factory()
            finally:
                release()

        try:
            future = self._submit(invoke)
        except RuntimeError:
            release()
            return
        future.add_done_callback(lambda _future: self._request_stop())

    def owns_current_thread(self) -> bool:
        return threading.current_thread() is self._thread

    def _request_stop(self) -> None:
        if self._closed.is_set():
            return
        try:
            if self.owns_current_thread():
                self._loop.stop()
            else:
                self._loop.call_soon_threadsafe(self._loop.stop)
        except RuntimeError:
            if not self._closed.is_set():
                raise

    def close(self) -> None:
        self._request_stop()
        if not self.owns_current_thread():
            self._thread.join()

    async def close_async(self) -> None:
        if self.owns_current_thread():
            self._request_stop()
            return
        await asyncio.to_thread(self.close)


class StreamingJob:
    """Sole lifecycle owner returned after a streaming runner starts."""

    __slots__ = ("_blocking_loop", "_inner", "__weakref__")

    def __init__(
        self,
        inner: _native._StreamingJob,
        blocking_loop: _BlockingEventLoop | None = None,
    ) -> None:
        self._inner = inner
        self._blocking_loop = blocking_loop

    def _run_blocking[T](
        self,
        factory: Callable[[], Awaitable[T]],
        method: str,
        *,
        terminal: bool = False,
    ) -> T:
        _reject_active_loop(method)
        loop = self._blocking_loop
        if loop is None:
            return asyncio.run(factory())
        try:
            return loop.run(factory)
        finally:
            if terminal:
                loop.close()
                self._blocking_loop = None

    async def _run_terminal_async(
        self, factory: Callable[[], Awaitable[Mapping[str, object]]]
    ) -> JobOutcome:
        loop = self._blocking_loop
        if loop is None or loop.owns_current_thread():
            value = await factory()
        else:
            try:
                value = await loop.run_async(factory)
            except asyncio.CancelledError:
                loop.close_after(self._inner.wait_async, self._inner._release_roots)
                raise
        self._inner._release_roots()
        if loop is not None and not loop.owns_current_thread():
            self._blocking_loop = None
            await _finish_cleanup(loop.close_async())
        return _outcome(value)

    @property
    def id(self) -> int:
        return self._inner.id

    def status(self) -> JobStatus:
        """Return a fresh CPU-local status snapshot; safe inside an event loop."""
        return self._inner.status()

    async def trigger_checkpoint_async(self) -> int:
        """Request and await one durable checkpoint epoch."""
        return await self._inner.trigger_checkpoint_async()

    def trigger_checkpoint(self) -> int:
        return self._run_blocking(self.trigger_checkpoint_async, "trigger_checkpoint")

    async def shutdown_async(self) -> JobOutcome:
        """Drain the job, publish terminal progress, and await cleanup."""
        return await self._run_terminal_async(self._inner.shutdown_async)

    def shutdown(self) -> JobOutcome:
        return self._run_blocking(self.shutdown_async, "shutdown", terminal=True)

    async def cancel_async(self) -> JobOutcome:
        """Cancel the job and await bounded connector cleanup."""
        return await self._run_terminal_async(self._inner.cancel_async)

    def cancel(self) -> JobOutcome:
        return self._run_blocking(self.cancel_async, "cancel", terminal=True)

    async def wait_async(self) -> JobOutcome:
        """Observe terminal completion without changing job state."""
        return await self._run_terminal_async(self._inner.wait_async)

    def wait(self) -> JobOutcome:
        return self._run_blocking(self.wait_async, "wait", terminal=True)

    def __del__(self) -> None:
        loop = self._blocking_loop
        if loop is None:
            return
        self._blocking_loop = None
        try:
            loop.close_after(self._inner.cancel_async, self._inner._release_roots)
        except BaseException:
            loop.close()


def _runner_sources(
    sources: Mapping[str, SourceBinding],
) -> dict[str, SourceBinding]:
    if not isinstance(sources, Mapping):
        raise TypeError("sources must be a mapping of source bindings")
    copied = dict(sources)
    if not all(
        isinstance(name, str) and isinstance(binding, SourceBinding)
        for name, binding in copied.items()
    ):
        raise TypeError("sources must map strings to SourceBinding values")
    return copied


def _runner_sinks(
    sinks: Mapping[str, Sequence[SinkBinding]],
) -> dict[str, tuple[SinkBinding, ...]]:
    if not isinstance(sinks, Mapping):
        raise TypeError("sinks must be a mapping of sink bindings")
    copied = {output: tuple(bindings) for output, bindings in sinks.items()}
    if not all(
        isinstance(output, str)
        and all(isinstance(binding, SinkBinding) for binding in bindings)
        for output, bindings in copied.items()
    ):
        raise TypeError("sinks must map strings to SinkBinding sequences")
    return copied


def _runner_config(config: StreamRuntimeConfig | None) -> StreamRuntimeConfig:
    selected = StreamRuntimeConfig() if config is None else config
    if not isinstance(selected, StreamRuntimeConfig):
        raise TypeError("config must be a calc_flow.StreamRuntimeConfig or None")
    return selected


class StreamingRunner:
    """One-shot source-driven continuous runner owning all bindings."""

    __slots__ = ("_inner", "__weakref__")

    def __init__(
        self,
        plan: StreamExecutionPlan,
        sources: Mapping[str, SourceBinding] | None = None,
        sinks: Mapping[str, Sequence[SinkBinding]] | None = None,
        checkpoints: ManagedCheckpointRuntime | None = None,
        *,
        config: StreamRuntimeConfig | None = None,
    ) -> None:
        from calc_flow.pipeline import StreamExecutionPlan

        if not isinstance(plan, StreamExecutionPlan):
            raise TypeError("plan must be a calc_flow.StreamExecutionPlan")
        settings = plan._project_settings
        if settings is not None:
            if any(
                value is not None for value in (sources, sinks, checkpoints, config)
            ):
                raise TypeError(
                    "connector-backed project plans own sources, sinks, checkpoints, "
                    "and runtime config"
                )
            sources = {}
            sinks = {}
            checkpoints = ManagedCheckpointRuntime(settings.state_root)
            config = StreamRuntimeConfig(
                checkpoint_interval=timedelta(
                    milliseconds=settings.checkpoint_interval_ms
                ),
                edge_budget=EdgeBudget(
                    settings.max_batch_rows, settings.max_batch_bytes
                ),
                retained_epochs=settings.retained_epochs,
            )
        if not isinstance(sources, Mapping):
            raise TypeError("sources must be a mapping of source bindings")
        if not isinstance(sinks, Mapping):
            raise TypeError("sinks must be a mapping of sink bindings")
        if not isinstance(checkpoints, ManagedCheckpointRuntime):
            raise TypeError("checkpoints must be a calc_flow.ManagedCheckpointRuntime")
        self._inner = _native._StreamingRunner(
            plan._inner,
            _runner_sources(sources),
            _runner_sinks(sinks),
            checkpoints._inner,
            _runner_config(config)._native(),
        )

    async def start_async(self) -> StreamingJob:
        """Consume this runner and asynchronously launch one owning job."""
        try:
            try:
                return StreamingJob(await self._inner.start_async())
            except asyncio.CancelledError as cancellation:
                await _raise_after_cancellation_cleanup(
                    self._inner._wait_start_cleanup_async(), cancellation
                )
        finally:
            self._inner._release_roots()

    def start(self) -> StreamingJob:
        """Start outside an event loop using the guarded blocking facade."""
        _reject_active_loop("start")
        loop = _BlockingEventLoop()
        try:
            job = loop.run(self.start_async)
        except BaseException:
            loop.close()
            raise
        job._blocking_loop = loop
        return job
