from __future__ import annotations

from typing import Any

import pyarrow as pa

from calc_flow.engine.base import Engine
from calc_flow.expression import sql_projection


class DataFrameEngine(Engine):
    """Base class for dataframe-backed engines (pandas, polars, datafusion)."""

    def sql(self, query: str, tables: dict[str, pa.Table]) -> pa.Table:
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

    def evaluate(self, expression: str, data: pa.Table, **kwargs: Any) -> pa.Table:
        df = data.to_pandas()
        result = df.eval(expression, **kwargs)
        if result is None:
            return pa.Table.from_pandas(df)
        if hasattr(result, "to_frame") and not hasattr(result, "columns"):
            return pa.Table.from_pandas(result.to_frame(name=result.name or "result"))
        return pa.Table.from_pandas(result)


class PolarsEngine(DataFrameEngine):
    """Polars-backed computation engine."""

    def evaluate(self, expression: str, data: pa.Table, **kwargs: Any) -> pa.Table:
        import polars as pl

        ctx = pl.SQLContext(__input__=pl.from_arrow(data))
        result = ctx.execute(sql_projection(expression, "__input__"), eager=True)
        return result.to_arrow()

    def sql(self, query: str, tables: dict[str, pa.Table]) -> pa.Table:
        import polars as pl

        frames = {name: pl.from_arrow(table) for name, table in tables.items()}
        ctx = pl.SQLContext(**frames)
        result = ctx.execute(query, eager=True)
        return result.to_arrow()


class DataFusionEngine(DataFrameEngine):
    """Apache DataFusion-backed computation engine with full SQL support."""

    def evaluate(self, expression: str, data: pa.Table, **kwargs: Any) -> pa.Table:
        return self.sql(sql_projection(expression, "__input__"), {"__input__": data})

    def sql(self, query: str, tables: dict[str, pa.Table]) -> pa.Table:
        import datafusion

        ctx = datafusion.SessionContext()
        for name, table in tables.items():
            ctx.register_record_batches(name, [_record_batches(table)])
        return ctx.sql(query).to_arrow_table()
