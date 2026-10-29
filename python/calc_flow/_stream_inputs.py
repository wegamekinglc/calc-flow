"""Logical input policies and owned iterable adapters for expression streams."""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator, Mapping
from datetime import timedelta
from typing import NoReturn, Union

from calc_flow import _native
from calc_flow._compat import TypeAliasType, dataclass
from calc_flow._native import Batch
from calc_flow.compute import TableData, _arrow_table, _table_batch
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
    Union[AsyncIterable[Union[TableData, Watermark]], SourceBinding, TableData],
)

_POLICIES = (SourceProvidedWatermarks, BoundedOutOfOrderness, DisabledWatermarks)


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
    inputs: Mapping[str, StreamInput | TableData],
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
    schema: _native._ArrowSchema
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
            policy = BoundedOutOfOrderness(
                ordered_column, timedelta(microseconds=1), timedelta(milliseconds=100)
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
        batch = self._arrow_input(data)
        self._validate_batch(batch)
        batch = _table_batch(batch, self._path)
        self._position += 1
        return Data(batch, Cursor(self._position.to_bytes(16, "big"), {}))

    def _arrow_input(self, data: TableData) -> Batch:
        if not (
            isinstance(data, Batch)
            or hasattr(data, "__arrow_c_stream__")
            or hasattr(data, "__arrow_c_array__")
        ):
            self._reject(": expected Arrow Table, RecordBatch or table Batch")
        if isinstance(data, Batch):
            return _arrow_table(data, self._path)
        try:
            return _native._stream_input_batch(data, self._config.ordered_column)
        except _native._ArrowValidationError as error:
            self._reject(str(error))

    def _watermark(self, value: Watermark) -> Watermark:
        if not isinstance(self._config.policy, SourceProvidedWatermarks):
            self._reject(": Watermark requires SourceProvidedWatermarks")
        return value

    def _validate_batch(self, table: Batch) -> None:
        self._validate_structure(table)
        if self._config.ordered_column is not None:
            try:
                self._last_time = table._validate_event_time(
                    self._config.ordered_column, self._last_time
                )
            except ValueError as error:
                self._reject(".event_time: " + str(error))

    def _validate_structure(self, table: Batch) -> None:
        if not table._schema.equals(self._config.schema):
            self._reject(".schema: expected declared Arrow schema")
        if table.num_rows > self._config.budget.max_rows:
            self._reject(": batch exceeds edge_budget.max_rows")
        if table._nbytes > self._config.budget.max_bytes:
            self._reject(": batch exceeds edge_budget.max_bytes")

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
    if (
        isinstance(data, Batch)
        or hasattr(data, "__arrow_c_stream__")
        or hasattr(data, "__arrow_c_array__")
    ):
        return _materialized_binding(data, config, complete_snapshot)
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
    if not table._schema.equals(config.schema):
        raise ValueError(f"{path}.schema: expected declared Arrow schema")
    if complete_snapshot:
        if isinstance(config.policy, SourceProvidedWatermarks):
            raise ValueError(
                f"{path}: materialized input cannot emit native watermarks"
            )
        source = _IterableSource(_whole_table(table), config)
        source._validate_structure(table)
        return SourceBinding(source, watermark_policy=config.policy), source
    return _arrow_source_binding(
        _table_batch(table, path),
        config.budget,
        config.policy,
        config.ordered_column,
    ), None


async def _whole_table(data: Batch) -> AsyncIterator[Batch]:
    yield data
