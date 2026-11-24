"""Logical input policies and owned iterable adapters for expression streams."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping, Sequence
from datetime import timedelta
from typing import NoReturn, Union

import pyarrow as pa

from calc_flow._compat import TypeAliasType, dataclass
from calc_flow._native import Batch, NativeArray, _ordered_event_time
from calc_flow.compute import TableData, _arrow_table, _normalized_table_batch
from calc_flow.runtime import (
    BoundedOutOfOrderness,
    Cursor,
    Data,
    DisabledWatermarks,
    EdgeBudget,
    NativeWatermarkCapability,
    ReplayPositioning,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    SourceProvidedWatermarks,
    Watermark,
    WatermarkPolicy,
    _arrow_source_binding,
)
from calc_flow.symbolic.analyzer import _schema_fields
from calc_flow.symbolic.expr import Parameter, TableExpr
from calc_flow.symbolic.lower.schema import _arrow_schema
from calc_flow.symbolic.nodes import CStr

StreamInput = TypeAliasType(
    "StreamInput",
    Union[
        AsyncIterable[Union[TableData, Watermark]],
        SourceBinding,
        TableData,
        Sequence[TableData],
    ],
)

_MATERIALIZED_ITEMS = (pa.Table, pa.RecordBatch, Batch)

_POLICIES = (SourceProvidedWatermarks, BoundedOutOfOrderness, DisabledWatermarks)


def _capture_stream_inputs(
    inputs: Mapping[str, StreamInput | TableData | NativeArray],
) -> dict[str, StreamInput | TableData | NativeArray]:
    return {
        name: tuple(value)
        if isinstance(value, Sequence) and not isinstance(value, _MATERIALIZED_ITEMS)
        else value
        for name, value in inputs.items()
    }


def _capture_watermarks(
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
) -> WatermarkPolicy | dict[str, WatermarkPolicy] | None:
    return dict(watermarks) if isinstance(watermarks, Mapping) else watermarks


def _policy_mapping(
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
    dynamic: set[str],
) -> Mapping[str, WatermarkPolicy]:
    if watermarks is None:
        return {}
    if isinstance(watermarks, Mapping):
        return watermarks
    if not isinstance(watermarks, _POLICIES):
        raise TypeError("stream.watermarks: expected supported watermark policy")
    if len(dynamic) != 1:
        raise ValueError("stream.watermarks: multiple inputs require a name mapping")
    return {next(iter(dynamic)): watermarks}


def _selected_policies(
    inputs: Mapping[str, StreamInput | TableData | NativeArray],
    expected: Mapping[str, TableExpr | Parameter],
    watermarks: WatermarkPolicy | Mapping[str, WatermarkPolicy] | None,
) -> Mapping[str, WatermarkPolicy]:
    dynamic = {name for name, value in expected.items() if isinstance(value, TableExpr)}
    selected = _policy_mapping(watermarks, dynamic)
    for name, policy in selected.items():
        if name not in dynamic:
            raise ValueError(f"stream.watermarks.{name}: expected dynamic input name")
        if not isinstance(policy, _POLICIES):
            raise TypeError(
                f"stream.watermarks.{name}: expected supported watermark policy"
            )
        if isinstance(inputs[name], SourceBinding):
            raise ValueError(f"stream.watermarks.{name}: cannot override SourceBinding")
    return selected


@dataclass(frozen=True, slots=True)
class _IterableConfig:
    name: str
    schema: pa.Schema
    budget: EdgeBudget
    policy: WatermarkPolicy
    ordered_column: str | None


def _source_config(
    name: str,
    value: TableExpr,
    budget: EdgeBudget,
    policy: WatermarkPolicy | None,
) -> _IterableConfig:
    schema = _arrow_schema(_schema_fields(value._node.attr("schema")))
    event_time = value._node.attr("event_time")
    ordered_column = None
    if policy is None:
        policy = DisabledWatermarks()
        if isinstance(event_time, CStr):
            ordered_column = event_time.value
            # A zero emit interval advances the watermark after every data batch.
            policy = BoundedOutOfOrderness(
                ordered_column, timedelta(microseconds=1), timedelta(0)
            )
    return _IterableConfig(name, schema, budget, policy, ordered_column)


class _IterableSource:
    def __init__(
        self,
        source: AsyncIterable[TableData | Watermark],
        config: _IterableConfig,
    ) -> None:
        self._source: AsyncIterable[TableData | Watermark] | None = source
        self._config = config
        self._path = f"stream.inputs.{config.name}"
        self._iterator: AsyncIterator[TableData | Watermark] | None = None
        self._closed = False
        self._position = 0
        self._last_time: int | None = None
        self.failure: str | None = None

    def capabilities(self) -> SourceCapabilities:
        watermarks = NativeWatermarkCapability.NEVER_EMITS
        if isinstance(self._config.policy, SourceProvidedWatermarks):
            watermarks = NativeWatermarkCapability.EMITS_NATIVE
        return SourceCapabilities(
            ReplayPositioning.UNSUPPORTED,
            SourceDeliveryCapability.LOSSY,
            max_batch_rows=self._config.budget.max_rows,
            max_batch_bytes=self._config.budget.max_bytes,
            schema=self._config.schema,
            native_watermarks=watermarks,
        )

    async def open(self, cursor: Cursor | None) -> None:
        if self._source is None:
            raise RuntimeError(f"{self._path}: source is closed")
        self._iterator = self._source.__aiter__()

    async def next(self) -> Data | Watermark | None:
        if self._iterator is None:
            raise RuntimeError(f"{self._path}: source is not open")
        try:
            data = await self._iterator.__anext__()
        except StopAsyncIteration:
            return None
        if isinstance(data, Watermark):
            return self._watermark(data)
        table = self._arrow_input(data)
        self._validate_structure(table)
        column = self._config.ordered_column
        try:
            batch = _normalized_table_batch(data, table)
        except Exception:
            if column is not None:
                # An invalid event time precedes an unrelated table-import failure.
                times = pa.Table.from_arrays([table[column]], names=[column])
                self._validate_time(Batch.from_pyarrow(times), column)
            raise
        if column is not None:
            self._validate_time(batch, column)
        self._position += 1
        return Data(batch, Cursor(self._position.to_bytes(16, "big"), {}))

    def _arrow_input(self, data: TableData) -> pa.Table | pa.RecordBatch:
        try:
            return _arrow_table(data, self._path)
        except TypeError:
            self._reject(": expected Arrow Table, RecordBatch or table Batch")

    def _watermark(self, value: Watermark) -> Watermark:
        if not isinstance(self._config.policy, SourceProvidedWatermarks):
            self._reject(": Watermark requires SourceProvidedWatermarks")
        return value

    def _validate_structure(self, table: pa.Table | pa.RecordBatch) -> None:
        if not table.schema.equals(self._config.schema):
            self._reject(".schema: expected declared Arrow schema")
        if table.num_rows > self._config.budget.max_rows:
            self._reject(": batch exceeds edge_budget.max_rows")
        if table.nbytes > self._config.budget.max_bytes:
            self._reject(": batch exceeds edge_budget.max_bytes")

    def _validate_time(self, batch: Batch, column: str) -> None:
        try:
            self._last_time = _ordered_event_time(batch, column, self._last_time)
        except ValueError as error:
            self._reject(f".event_time: {error}")

    def _reject(self, message: str) -> NoReturn:
        self.failure = self._path + message
        raise ValueError(self.failure)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            close = getattr(self._iterator, "aclose", None)
            if close is not None:
                await close()
        finally:
            self._source = None
            self._iterator = None


def _source_binding(
    data: StreamInput,
    logical_input: tuple[str, TableExpr],
    budget: EdgeBudget,
    policy: WatermarkPolicy | None,
    *,
    complete_snapshot: bool = False,
) -> tuple[SourceBinding, _IterableSource | None]:
    if isinstance(data, SourceBinding):
        return data, None
    name, value = logical_input
    config = _source_config(name, value, budget, policy)
    if isinstance(data, _MATERIALIZED_ITEMS):
        return _materialized_binding(data, config, complete_snapshot)
    if isinstance(data, Sequence):
        return _materialized_sequence_binding(data, config, complete_snapshot)
    if not isinstance(data, AsyncIterable):
        raise TypeError(
            f"stream.inputs.{name}: expected async iterable, materialized Arrow "
            "input or SourceBinding"
        )
    source = _IterableSource(data, config)
    return SourceBinding(source, watermark_policy=config.policy), source


def _materialized_binding(
    data: TableData, config: _IterableConfig, complete_snapshot: bool
) -> tuple[SourceBinding, _IterableSource | None]:
    path = f"stream.inputs.{config.name}"
    table = _arrow_table(data, path)
    if not table.schema.equals(config.schema):
        raise ValueError(f"{path}.schema: expected declared Arrow schema")
    if complete_snapshot:
        if isinstance(config.policy, SourceProvidedWatermarks):
            raise ValueError(
                f"{path}: materialized input cannot emit native watermarks"
            )
        source = _IterableSource(_whole_table(data), config)
        source._validate_structure(table)
        return SourceBinding(source, watermark_policy=config.policy), source
    return _arrow_source_binding(
        _normalized_table_batch(data, table),
        config.budget,
        config.policy,
        config.ordered_column,
    ), None


def _sequence_tables(data: Sequence[TableData], path: str) -> list[pa.Table]:
    tables = []
    for item in data:
        table = _arrow_table(item, path)
        if isinstance(table, pa.RecordBatch):
            table = pa.Table.from_batches([table])
        tables.append(table)
    return tables


def _validate_materialized_sequence(data: Sequence[TableData], path: str) -> None:
    if not data:
        raise ValueError(f"{path}: materialized sequence must not be empty")
    if not all(isinstance(item, _MATERIALIZED_ITEMS) for item in data):
        raise TypeError(
            f"{path}: materialized sequence must contain Arrow Table, "
            "RecordBatch or table Batch values"
        )


def _sequence_batch(tables: list[pa.Table], schema: pa.Schema) -> Batch:
    records = [
        record
        for table in tables
        for record in (
            table.to_batches()
            or [
                pa.RecordBatch.from_arrays(
                    [column.combine_chunks() for column in table.columns],
                    schema=schema,
                )
            ]
        )
    ]
    return Batch.from_pyarrow(pa.RecordBatchReader.from_batches(schema, records))


def _materialized_sequence_binding(
    data: Sequence[TableData], config: _IterableConfig, complete_snapshot: bool
) -> tuple[SourceBinding, _IterableSource | None]:
    path = f"stream.inputs.{config.name}"
    _validate_materialized_sequence(data, path)
    if complete_snapshot:
        raise ValueError(
            f"{path}: materialized sequence is not supported for snapshot_sql; "
            "supply one complete table per input"
        )
    tables = _sequence_tables(data, path)
    for table in tables:
        if not table.schema.equals(config.schema):
            raise ValueError(f"{path}.schema: expected declared Arrow schema")
    if not config.schema or any(table.num_rows == 0 for table in tables):
        # Table concatenation loses zero-column rows and trailing empty records.
        return _materialized_binding(
            _sequence_batch(tables, config.schema), config, complete_snapshot
        )
    merged = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    return _materialized_binding(merged, config, complete_snapshot)


async def _whole_table(data: TableData) -> AsyncIterator[TableData]:
    yield data
