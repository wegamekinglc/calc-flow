"""Synchronous caller-thread sessions over one compiled native stream plan."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from types import TracebackType
from typing import Generic, TypeVar, Union, cast

import pyarrow as pa

from calc_flow._compat import TypeAliasType, dataclass
from calc_flow._native import Batch, _StreamSession
from calc_flow.compute import TableData, _arrow_table, _require_blocking, _table_batch
from calc_flow.pipeline import Runtime
from calc_flow.runtime import (
    BoundedOutOfOrderness,
    DisabledWatermarks,
    ManagedCheckpointRuntime,
    SourceProvidedWatermarks,
    StreamRuntimeConfig,
    Watermark,
    WatermarkPolicy,
    _datetime_micros,
    _duration_micros,
    _runner_config,
)
from calc_flow.symbolic.analyzer import _schema_fields
from calc_flow.symbolic.expr import Parameter, TableExpr
from calc_flow.symbolic.lower.schema import _arrow_schema
from calc_flow.symbolic.nodes import CStr
from calc_flow.symbolic.program import Program, _node_name

T = TypeVar("T")

WatermarkValue = TypeAliasType("WatermarkValue", Union[datetime, Watermark])

_POLICIES = (SourceProvidedWatermarks, BoundedOutOfOrderness, DisabledWatermarks)


@dataclass(frozen=True, slots=True)
class _SessionInput:
    name: str
    physical: str
    schema: pa.Schema


class StreamSession(Generic[T]):
    """Synchronous pushes into one compiled stream plan on the calling thread.

    Create sessions with ``TableExpr.session`` or ``Program.session``. Each
    ``push`` runs its batch and optional watermark through every operator and
    returns the rows that call emitted; ``finish`` flushes and ends the
    session. Sessions created without ``checkpoints`` are in-memory. Supplying
    managed storage enables durable checkpoint and highest-epoch restore through
    ``checkpoint()`` and ``restore=True``; ``new`` keeps its original behavior.

    ``push``, ``advance``, ``finish`` and ``close`` may run inside a running
    event loop. They execute admitted operators on the calling thread with no
    I/O or cross-thread waits, so a large push blocks the loop for its
    duration. Late rows dropped by ``late_policy="drop"`` are not observable.
    A rejected input (``ValueError`` for schema, ``ConfigError`` for input,
    watermark or ordering) leaves the session usable; an operator failure
    fails it. Calls on a failed, finished, closed or busy session raise
    ``RuntimeError``.
    """

    __slots__ = (
        "_native",
        "_input",
        "_outputs",
        "_empty",
        "_table_output",
        "_copy_empty_program",
    )

    def __init__(
        self,
        native: _StreamSession,
        session_input: _SessionInput,
        outputs: tuple[tuple[str, str], ...],
        empty: Mapping[str, pa.Table],
        *,
        table_output: bool,
    ) -> None:
        self._native = native
        self._input = session_input
        self._outputs = outputs
        self._empty = dict(empty)
        self._table_output = table_output
        self._copy_empty_program = not table_output and tuple(self._empty) == tuple(
            name for name, _ in outputs
        )

    def push(
        self,
        inputs: TableData | Mapping[str, TableData],
        /,
        *,
        watermark: WatermarkValue | Mapping[str, WatermarkValue] | None = None,
    ) -> T:
        """Run one input batch, then an optional completeness watermark.

        Table sessions take one ``pa.Table``, ``pa.RecordBatch`` or table
        ``Batch``; program sessions take a mapping with their dynamic input.
        ``watermark`` is a timezone-aware ``datetime`` or ``Watermark`` (a
        one-entry mapping for program sessions).
        """
        data = self._data(inputs)
        micros = None if watermark is None else self._watermark_micros(watermark)
        path = f"session.inputs.{self._input.name}"
        batch = _session_batch(data, self._input.schema, path)
        try:
            emitted = self._native.push(self._input.physical, batch, micros)
        except ValueError:
            # Native validation rejected a Batch before any operator ran.
            if not isinstance(data, Batch):
                raise
            normalized = _table_batch(data, path)
            if normalized is data or not _same_schema(normalized, self._input.schema):
                raise ValueError(
                    f"{path}.schema: expected declared Arrow schema"
                ) from None
            emitted = self._native.push(self._input.physical, normalized, micros)
        return self._result(emitted)

    def advance(self, watermark: WatermarkValue | Mapping[str, WatermarkValue]) -> T:
        """Advance the input watermark without data; returns the rows it closes."""
        micros = self._watermark_micros(watermark)
        return self._result(self._native.advance(self._input.physical, micros))

    def checkpoint(self) -> int:
        """Durably persist the current session and return its epoch."""
        _require_blocking("session.checkpoint")
        return self._native.checkpoint()

    def finish(self) -> T:
        """End the input, flush every operator and finish the session."""
        return self._result(self._native.finish())

    def close(self) -> None:
        """Release the session without flushing; repeated calls do nothing."""
        self._native.close()

    def __enter__(self) -> StreamSession[T]:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _data(self, inputs: TableData | Mapping[str, TableData]) -> TableData:
        if self._table_output:
            return cast(TableData, inputs)
        if type(inputs) is dict and len(inputs) == 1:
            data = inputs.get(self._input.name)
            if data is not None:
                return data
        if not isinstance(inputs, Mapping):
            raise TypeError("session.inputs: expected a mapping by declared input name")
        names = set(inputs)
        if names != {self._input.name}:
            unexpected = sorted(names - {self._input.name})
            if unexpected:
                raise ValueError(
                    f"session.inputs.{unexpected[0]}: unexpected input name"
                )
            raise ValueError(f"session.inputs.{self._input.name}: missing table input")
        return inputs[self._input.name]

    def _watermark_micros(
        self, watermark: WatermarkValue | Mapping[str, WatermarkValue]
    ) -> int:
        path = f"session.watermark.{self._input.name}"
        if not self._table_output and isinstance(watermark, Mapping):
            if set(watermark) != {self._input.name}:
                raise ValueError(f"{path}: expected one entry for the dynamic input")
            watermark = watermark[self._input.name]
        if isinstance(watermark, Watermark):
            return _datetime_micros(watermark.at)
        if not isinstance(watermark, datetime):
            raise TypeError(f"{path}: expected a timezone-aware datetime or Watermark")
        if watermark.utcoffset() is None:
            raise ValueError(f"{path}: datetime must be timezone-aware")
        return _datetime_micros(watermark)

    def _result(self, emitted: Mapping[str, list[pa.RecordBatch]]) -> T:
        if len(self._outputs) == 1 and (
            not self._table_output or self._outputs[0][0] == "result"
        ):
            name, physical = self._outputs[0]
            batches = emitted.get(physical)
            table = pa.Table.from_batches(batches) if batches else self._empty[name]
            return cast(T, table if self._table_output else {name: table})

        if not emitted and self._copy_empty_program:
            return cast(T, self._empty.copy())

        tables: dict[str, pa.Table] = {}
        by_physical: dict[str, pa.Table] = {}
        for name, physical in self._outputs:
            batches = emitted.get(physical)
            if not batches:
                tables[name] = self._empty[name]
                continue
            if physical not in by_physical:
                by_physical[physical] = pa.Table.from_batches(batches)
            tables[name] = by_physical[physical]
        if self._table_output:
            return cast(T, tables["result"])
        return cast(T, tables)


def _same_schema(batch: Batch, schema: pa.Schema) -> bool:
    return batch.to_pyarrow().schema.equals(schema)


def _session_batch(
    data: TableData, schema: pa.Schema, path: str
) -> pa.Table | pa.RecordBatch | Batch:
    if isinstance(data, Batch):
        if data.kind != "table":
            raise TypeError(f"{path}: expected table input, got {data.kind} Batch")
        # The native session validates the exact schema without a PyArrow copy.
        return data
    table = _arrow_table(data, path)
    actual = table.schema
    if actual.equals(schema, check_metadata=True):
        return table
    if not actual.equals(schema):
        raise ValueError(f"{path}.schema: expected declared Arrow schema")
    if actual.metadata is not None or any(
        field.metadata is not None for field in actual
    ):
        return _table_batch(table, path)
    return table


def _session_policy(name: str, value: TableExpr, policy: object) -> dict[str, object]:
    path = f"session.watermarks.{name}"
    if policy is None:
        event_time = value._node.attr("event_time")
        if isinstance(event_time, CStr):
            return {"kind": "ordered", "event_time_column": event_time.value}
        return {"kind": "disabled"}
    if isinstance(policy, SourceProvidedWatermarks):
        return {"kind": "explicit"}
    if isinstance(policy, BoundedOutOfOrderness):
        if policy.idle_timeout is not None:
            raise ValueError(f"{path}.idle_timeout: sessions have no idle timers")
        return {
            "kind": "bounded_out_of_orderness",
            "event_time_column": policy.event_time_column,
            "max_out_of_orderness_micros": _duration_micros(
                policy.max_out_of_orderness, f"{path}.max_out_of_orderness"
            ),
        }
    if isinstance(policy, DisabledWatermarks):
        if policy.idle_timeout is not None:
            raise ValueError(f"{path}.idle_timeout: sessions have no idle timers")
        return {"kind": "disabled"}
    raise TypeError(f"{path}: expected supported watermark policy")


def _session_policies(
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
    dynamic: Mapping[str, TableExpr],
) -> dict[str, dict[str, object]]:
    if watermarks is None:
        selected: Mapping[str, object] = {}
    elif isinstance(watermarks, Mapping):
        selected = dict(watermarks)
    elif isinstance(watermarks, _POLICIES):
        if len(dynamic) != 1:
            raise ValueError(
                "session.watermarks: multiple inputs require a name mapping"
            )
        selected = {next(iter(dynamic)): watermarks}
    else:
        raise TypeError("session.watermarks: expected supported watermark policy")
    for name in selected:
        if name not in dynamic:
            raise ValueError(f"session.watermarks.{name}: expected dynamic input name")
    return {
        name: _session_policy(name, value, selected.get(name))
        for name, value in dynamic.items()
    }


def _session_dynamic_inputs(program: Program) -> dict[str, TableExpr]:
    expected = {_node_name(value._node): value for value in program.inputs}
    dynamic: dict[str, TableExpr] = {}
    for name, value in expected.items():
        if isinstance(value, Parameter):
            raise ValueError(
                f"session.inputs.{name}: static parameters are not supported; "
                "use stream()"
            )
        if not isinstance(value, TableExpr):
            raise ValueError(
                f"session.inputs.{name}: native array inputs are not supported; "
                "use stream()"
            )
        dynamic[name] = value
    return dynamic


def _check_session_outputs(program: Program) -> None:
    for name, output in program.outputs:
        if not isinstance(output, TableExpr):
            raise TypeError(f"session.outputs.{name}: expected TableExpr")


def _open_session(
    program: Program,
    runtime: Runtime | None,
    config: StreamRuntimeConfig | None,
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
    *,
    options: tuple[bool, ManagedCheckpointRuntime | None, bool],
) -> StreamSession[object]:
    from calc_flow.stream import _compile_stream, _StreamRequest

    table_output, checkpoints, restore = options
    _require_blocking("session")
    dynamic = _session_dynamic_inputs(program)
    _check_session_outputs(program)
    policies = _session_policies(watermarks, dynamic)
    native_config = _runner_config(config)._native()
    plan, names, sinks, _ = _compile_stream(
        _StreamRequest(program, {}, runtime, config, watermarks)
    )
    schemas = program.collect_schema(runtime)
    physical: dict[str, str] = {}
    for name in dynamic:
        ports = names.get(name, ())
        if len(ports) != 1:
            raise ValueError(f"session.inputs.{name}: expected one native ingress")
        physical[name] = ports[0]
    # Native admission rejects graphs other than one dynamic input (ConfigError).
    native = _StreamSession(
        plan._inner,
        native_config,
        {physical[name]: policy for name, policy in policies.items()},
        checkpoints=None if checkpoints is None else checkpoints._inner,
        restore=restore,
    )
    name, value = next(iter(dynamic.items()))
    session_input = _SessionInput(
        name, physical[name], _arrow_schema(_schema_fields(value._node.attr("schema")))
    )
    outputs = tuple((output, sinks[output][0]) for output, _ in program.outputs)
    empty = {output: schemas[output].empty_table() for output, _ in program.outputs}
    return StreamSession(
        native, session_input, outputs, empty, table_output=table_output
    )
