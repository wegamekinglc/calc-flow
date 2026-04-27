from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl


class Batch:
    """A batch of data backed by Arrow dataframe data or an Array API array.

    Dataframe batches use Arrow as their internal representation. Array batches
    preserve the provided Array API object for array engines.
    """

    def __init__(self, data: pa.RecordBatch | pa.Table | Any) -> None:
        if isinstance(data, (pa.RecordBatch, pa.Table)):
            self._data = data
            self._kind = "dataframe"
            return
        if _is_array_api_array(data):
            self._data = data
            self._kind = "array"
            return

        msg = f"Expected RecordBatch, Table, or Array API array, got {type(data)}"
        raise TypeError(msg)

    @classmethod
    def from_array(cls, array: Any) -> Batch:
        if not _is_array_api_array(array):
            msg = f"Expected Array API array, got {type(array)}"
            raise TypeError(msg)
        return cls(array)

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> Batch:
        return cls(pa.RecordBatch.from_pandas(df))

    @classmethod
    def from_polars(cls, df: pl.DataFrame) -> Batch:
        return cls(df.to_arrow())

    @classmethod
    def from_pylist(cls, rows: list[dict], schema: pa.Schema | None = None) -> Batch:
        table = pa.Table.from_pylist(rows, schema=schema)
        return cls(table)

    @classmethod
    def from_arrays(cls, columns: Mapping[str, object]) -> Batch:
        """Build an Arrow dataframe batch from array-like columns.

        ``pyarrow.table`` accepts Arrow arrays, NumPy arrays, pandas series,
        and plain Python sequences, which keeps user-facing inputs broad while
        preserving Arrow as the internal representation.
        """
        return cls(pa.table(dict(columns)))

    @property
    def arrow(self) -> pa.RecordBatch | pa.Table:
        self._require_dataframe("arrow")
        return self._data

    @property
    def array(self) -> Any:
        if self._kind != "array":
            msg = "array is only available for Array API batches"
            raise TypeError(msg)
        return self._data

    @property
    def is_dataframe(self) -> bool:
        return self._kind == "dataframe"

    @property
    def is_array(self) -> bool:
        return self._kind == "array"

    @property
    def table(self) -> pa.Table:
        self._require_dataframe("table")
        if isinstance(self._data, pa.Table):
            return self._data
        return pa.Table.from_batches([self._data])

    @property
    def schema(self) -> pa.Schema:
        self._require_dataframe("schema")
        return self._data.schema

    @property
    def num_rows(self) -> int:
        if self._kind == "array":
            return _array_rows(self._data)
        return self._data.num_rows

    @property
    def num_columns(self) -> int:
        if self._kind == "array":
            return _array_columns(self._data)
        return self._data.num_columns

    def to_pandas(self) -> pd.DataFrame:
        self._require_dataframe("to_pandas")
        return self._data.to_pandas()

    def to_polars(self) -> pl.DataFrame:
        self._require_dataframe("to_polars")
        import polars as pl

        return pl.from_arrow(self._data)

    def to_pylist(self) -> list[dict] | list:
        if self._kind == "array":
            if hasattr(self._data, "tolist"):
                return self._data.tolist()
            return list(self._data)
        return self._data.to_pylist()

    def slice(self, offset: int, length: int) -> Batch:
        if self._kind == "array":
            return Batch(self._data[offset : offset + length])
        return Batch(self._data.slice(offset, length))

    def __len__(self) -> int:
        return self.num_rows

    def __repr__(self) -> str:
        return (
            f"Batch(kind={self._kind!r}, rows={self.num_rows}, cols={self.num_columns})"
        )

    def _require_dataframe(self, operation: str) -> None:
        if self._kind == "dataframe":
            return
        msg = f"{operation} is only available for Arrow-backed dataframe batches"
        raise TypeError(msg)


def _is_array_api_array(value: Any) -> bool:
    return hasattr(value, "__array_namespace__")


def _array_rows(array: Any) -> int:
    shape = getattr(array, "shape", ())
    if not shape:
        return 1
    return int(shape[0])


def _array_columns(array: Any) -> int:
    shape = getattr(array, "shape", ())
    if len(shape) < 2:
        return 1
    return int(shape[1])
