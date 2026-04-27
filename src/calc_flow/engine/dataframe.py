from __future__ import annotations

from typing import Any

import pyarrow as pa

from calc_flow.batch import Batch
from calc_flow.engine.base import Engine
from calc_flow.expression import sql_projection


class DataFrameEngine(Engine):
    """Base class for dataframe-backed engines (pandas, polars, datafusion)."""

    def sql(self, query: str, tables: dict[str, Batch]) -> Batch:
        msg = f"{type(self).__name__} does not provide a SQL execution layer"
        raise NotImplementedError(msg)


def _record_batches(table: pa.Table) -> list[pa.RecordBatch]:
    batches = table.to_batches()
    if batches:
        return batches
    arrays = [table.column(i).combine_chunks() for i in range(table.num_columns)]
    return [pa.RecordBatch.from_arrays(arrays, schema=table.schema)]


class PandasEngine(DataFrameEngine):
    """Pandas-backed computation engine."""

    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch:
        df = batch.to_pandas()
        result = df.eval(expression, **kwargs)
        if result is None:
            return Batch.from_pandas(df)
        if hasattr(result, "to_frame") and not hasattr(result, "columns"):
            return Batch.from_pandas(result.to_frame(name=result.name or "result"))
        return Batch.from_pandas(result)


class PolarsEngine(DataFrameEngine):
    """Polars-backed computation engine."""

    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch:
        import polars as pl

        ctx = pl.SQLContext(__input__=batch.to_polars())
        result = ctx.execute(sql_projection(expression, "__input__"), eager=True)
        return Batch.from_polars(result)

    def sql(self, query: str, tables: dict[str, Batch]) -> Batch:
        import polars as pl

        frames = {name: batch.to_polars() for name, batch in tables.items()}
        ctx = pl.SQLContext(**frames)
        result = ctx.execute(query, eager=True)
        return Batch.from_polars(result)


class DataFusionEngine(DataFrameEngine):
    """Apache DataFusion-backed computation engine with full SQL support."""

    def evaluate(self, expression: str, batch: Batch, **kwargs: Any) -> Batch:
        return self.sql(sql_projection(expression, "__input__"), {"__input__": batch})

    def sql(self, query: str, tables: dict[str, Batch]) -> Batch:
        import datafusion

        ctx = datafusion.SessionContext()
        for name, batch in tables.items():
            ctx.register_record_batches(name, [_record_batches(batch.table)])
        return Batch(ctx.sql(query).to_arrow_table())
