from __future__ import annotations

import pyarrow as pa
import pytest

from calc_flow.batch import Batch


class TestBatch:
    def test_from_pylist(self) -> None:
        b = Batch.from_pylist([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
        assert b.num_rows == 2
        assert b.num_columns == 2

    def test_from_arrays(self) -> None:
        b = Batch.from_arrays({"a": [1, 2], "b": [3, 4]})
        assert b.to_pylist() == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]

    def test_from_array(self) -> None:
        import numpy as np

        b = Batch.from_array(np.asarray([[1, 2], [3, 4]]))

        assert b.is_array
        assert not b.is_dataframe
        assert b.num_rows == 2
        assert b.num_columns == 2
        assert b.to_pylist() == [[1, 2], [3, 4]]

    def test_from_pandas(self) -> None:
        import pandas as pd

        df = pd.DataFrame({"x": [1, 2, 3]})
        b = Batch.from_pandas(df)
        assert b.num_rows == 3
        assert b.schema.names == ["x"]

    def test_from_polars(self) -> None:
        import polars as pl

        df = pl.DataFrame({"y": [4, 5]})
        b = Batch.from_polars(df)
        assert b.num_rows == 2
        assert b.schema.names == ["y"]

    def test_slice(self) -> None:
        b = Batch.from_pylist([{"n": i} for i in range(100)])
        assert len(b.slice(10, 5)) == 5

    def test_to_pandas_roundtrip(self) -> None:
        import pandas as pd

        orig = pd.DataFrame({"k": [1.0, 2.0]})
        result = Batch.from_pandas(orig).to_pandas()
        pd.testing.assert_frame_equal(orig, result)

    def test_to_polars_roundtrip(self) -> None:
        import polars as pl

        orig = pl.DataFrame({"k": [1, 2]})
        result = Batch.from_polars(orig).to_polars()
        assert orig.to_dict(as_series=False) == result.to_dict(as_series=False)

    def test_rejects_non_arrow(self) -> None:
        with pytest.raises(TypeError):
            Batch([1, 2, 3])  # type: ignore[arg-type]

    def test_rejects_non_array_in_from_array(self) -> None:
        with pytest.raises(TypeError):
            Batch.from_array([1, 2, 3])

    def test_arrow_property(self) -> None:
        table = pa.table({"c": [1, 2]})
        b = Batch(table)
        assert b.arrow is table

    def test_table_property_normalizes_record_batch(self) -> None:
        record_batch = pa.record_batch({"c": [1, 2]})
        b = Batch(record_batch)
        assert isinstance(b.table, pa.Table)
        assert b.table.to_pylist() == [{"c": 1}, {"c": 2}]

    def test_dataframe_operations_reject_array_batch(self) -> None:
        import numpy as np

        b = Batch.from_array(np.asarray([1, 2, 3]))

        with pytest.raises(TypeError, match="Arrow-backed"):
            _ = b.table

        with pytest.raises(TypeError, match="Arrow-backed"):
            b.to_pandas()

    def test_array_slice(self) -> None:
        import numpy as np

        b = Batch.from_array(np.asarray([1, 2, 3, 4]))

        assert b.slice(1, 2).to_pylist() == [2, 3]
