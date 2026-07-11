from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter_ns

import pyarrow as pa

from calc_flow.batch import Batch, BatchKind, BatchMetadata
from calc_flow.engine.base import Engine
from calc_flow.expression import sql_projection
from calc_flow.udf import (
    DataFusionScalarUdf,
    UdfReference,
    UdfRegistry,
    UdfRegistrySnapshot,
)

_QUERY_START_RE = re.compile(r"^(select|with)\b", re.IGNORECASE)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SQL_QUOTED_OR_COMMENT_RE = re.compile(
    r"'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\"|--[^\n]*|/\*.*?\*/",
    re.DOTALL,
)
_FORBIDDEN_SQL_RE = re.compile(
    r"\b(alter|copy|create|delete|drop|execute|insert|install|load|set|truncate|update)\b",
    re.IGNORECASE,
)


class DataFusionExecutionError(RuntimeError):
    """Raised when a planned DataFusion query fails during execution."""


@dataclass(frozen=True, slots=True)
class DataFusionConfig:
    """Run-scoped DataFusion execution settings."""

    batch_size: int = 8192
    target_partitions: int = 1
    repartition_aggregations: bool = True
    repartition_joins: bool = True
    repartition_sorts: bool = True
    repartition_windows: bool = True

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            msg = "DataFusion batch_size must be greater than 0"
            raise ValueError(msg)
        if self.target_partitions <= 0:
            msg = "DataFusion target_partitions must be greater than 0"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class DataFusionQueryMetrics:
    """Planning and execution observations for one DataFusion query."""

    node_id: str | None
    planning_ns: int
    execution_ns: int
    output_rows: int
    logical_plan: str
    physical_plan: str


def _record_batches(table: pa.Table) -> list[pa.RecordBatch]:
    batches = table.to_batches()
    if batches:
        return batches
    arrays = [table.column(i).combine_chunks() for i in range(table.num_columns)]
    return [pa.RecordBatch.from_arrays(arrays, schema=table.schema)]


def validate_datafusion_query(query: str) -> str:
    """Validate and normalize one read-only DataFusion SQL query."""
    normalized = query.strip()
    if normalized.endswith(";"):
        normalized = normalized[:-1].rstrip()
    if not normalized or not _QUERY_START_RE.match(normalized):
        msg = "DataFusion queries must be a SELECT statement or CTE"
        raise ValueError(msg)
    if ";" in normalized:
        msg = "DataFusion queries must contain exactly one statement"
        raise ValueError(msg)
    unquoted = _SQL_QUOTED_OR_COMMENT_RE.sub(" ", normalized)
    forbidden = _FORBIDDEN_SQL_RE.search(unquoted)
    if forbidden is not None:
        msg = f"DataFusion query contains prohibited {forbidden.group(1).upper()} SQL"
        raise ValueError(msg)
    return normalized


def _validate_identifier(name: str) -> None:
    if not _IDENTIFIER_RE.fullmatch(name):
        msg = f"invalid DataFusion input alias {name!r}"
        raise ValueError(msg)


def _output_metadata(tables: Mapping[str, Batch]) -> BatchMetadata:
    inputs = list(tables.values())
    if len(inputs) == 1:
        return inputs[0].metadata
    return BatchMetadata(
        attributes={"input_batch_ids": [batch.metadata.batch_id for batch in inputs]}
    )


class DataFusionRuntime:
    """One DataFusion session and its metrics for a single execution run."""

    def __init__(
        self,
        config: DataFusionConfig | None = None,
        *,
        udfs: Iterable[DataFusionScalarUdf] = (),
    ) -> None:
        import datafusion

        self.config = config or DataFusionConfig()
        session_config = (
            datafusion.SessionConfig()
            .with_batch_size(self.config.batch_size)
            .with_target_partitions(self.config.target_partitions)
            .with_repartition_aggregations(self.config.repartition_aggregations)
            .with_repartition_joins(self.config.repartition_joins)
            .with_repartition_sorts(self.config.repartition_sorts)
            .with_repartition_windows(self.config.repartition_windows)
        )
        self._context = datafusion.SessionContext(session_config)
        self._sql_options = (
            datafusion.SQLOptions()
            .with_allow_ddl(False)
            .with_allow_dml(False)
            .with_allow_statements(False)
        )
        self._metrics: list[DataFusionQueryMetrics] = []
        self._udf_names: list[str] = []
        self._closed = False
        for specification in udfs:
            scalar_udf = datafusion.ScalarUDF(
                name=specification.name,
                func=specification.invoke,
                input_fields=list(specification.input_fields),
                return_field=specification.return_field,
                volatility=specification.volatility.value,
            )
            self._context.register_udf(scalar_udf)
            self._udf_names.append(specification.name)

    @property
    def metrics(self) -> tuple[DataFusionQueryMetrics, ...]:
        return tuple(self._metrics)

    @property
    def session_id(self) -> str:
        return self._context.session_id()

    def evaluate(
        self,
        expression: str,
        data: Batch,
        *,
        node_id: str | None = None,
    ) -> Batch:
        if data.kind is not BatchKind.TABLE:
            msg = "DataFusionRuntime requires table batches"
            raise TypeError(msg)
        query = sql_projection(expression, "__input__")
        return self.sql(query, {"__input__": data}, node_id=node_id)

    def sql(
        self,
        query: str,
        tables: Mapping[str, Batch],
        *,
        node_id: str | None = None,
    ) -> Batch:
        if self._closed:
            msg = "DataFusionRuntime is closed"
            raise RuntimeError(msg)
        if not tables:
            msg = "at least one input table is required"
            raise ValueError(msg)

        normalized_query = validate_datafusion_query(query)
        aliases: list[str] = []
        try:
            for name, batch in tables.items():
                _validate_identifier(name)
                if batch.kind is not BatchKind.TABLE:
                    msg = "DataFusionRuntime requires table batches"
                    raise TypeError(msg)
                if self._context.table_exist(name):
                    msg = f"DataFusion input alias {name!r} is already registered"
                    raise ValueError(msg)
                self._context.register_record_batches(
                    name, [_record_batches(batch.table_payload)]
                )
                aliases.append(name)

            planning_started = perf_counter_ns()
            frame = self._context.sql(normalized_query, options=self._sql_options)
            logical_plan = frame.optimized_logical_plan().display_indent()
            physical_plan = frame.execution_plan().display_indent()
            planning_ns = perf_counter_ns() - planning_started

            execution_started = perf_counter_ns()
            try:
                result = frame.to_arrow_table()
            except Exception as error:
                msg = f"DataFusion query execution failed: {error}"
                raise DataFusionExecutionError(msg) from error
            execution_ns = perf_counter_ns() - execution_started
            self._metrics.append(
                DataFusionQueryMetrics(
                    node_id=node_id,
                    planning_ns=planning_ns,
                    execution_ns=execution_ns,
                    output_rows=result.num_rows,
                    logical_plan=logical_plan,
                    physical_plan=physical_plan,
                )
            )
            return Batch.table(result, metadata=_output_metadata(tables))
        finally:
            for name in aliases:
                self._context.deregister_table(name)

    def close(self) -> None:
        if self._closed:
            return
        for name in self._udf_names:
            self._context.deregister_udf(name)
        self._closed = True

    def __enter__(self) -> DataFusionRuntime:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class DataFusionEngine(Engine):
    """Standalone convenience API for the sole tabular calculation engine."""

    input_kind = BatchKind.TABLE

    def __init__(
        self,
        *,
        udf_registry: UdfRegistry | UdfRegistrySnapshot | None = None,
        udfs: Iterable[UdfReference] = (),
    ) -> None:
        if isinstance(udf_registry, UdfRegistry):
            registry = udf_registry.snapshot()
        else:
            registry = udf_registry or UdfRegistrySnapshot()
        self._udfs = registry.select(datafusion=tuple(udfs)).datafusion_specs

    def evaluate(self, expression: str, data: Batch) -> Batch:
        self._require_kind(data)
        with DataFusionRuntime(udfs=self._udfs) as runtime:
            return runtime.evaluate(expression, data)

    def sql(self, query: str, tables: Mapping[str, Batch]) -> Batch:
        with DataFusionRuntime(udfs=self._udfs) as runtime:
            return runtime.sql(query, tables)
