from __future__ import annotations

import pyarrow as pa
import pytest

from benchmarks.engine_comparison import EngineCase


@pytest.mark.parametrize("backend", ["calc-flow-stream", "calc-flow-stream-1t"])
def test_stream_asof_oracle_rejects_reordered_output(tmp_path, backend):
    engine = EngineCase(
        {"backend": backend, "scenario": "asof_join", "rows": 8}, tmp_path
    )
    try:
        batches = engine.expected.to_batches(max_chunksize=4)
        assert engine.validate(pa.Table.from_batches(batches))["passed"]
        with pytest.raises(ValueError, match="sequence"):
            engine.validate(pa.Table.from_batches(batches[::-1]))
    finally:
        engine.close()


def test_sql_asof_oracle_compares_rows_by_sequence(tmp_path):
    engine = EngineCase(
        {"backend": "calc-flow-sql", "scenario": "asof_join", "rows": 8}, tmp_path
    )
    try:
        batches = engine.expected.to_batches(max_chunksize=4)
        assert engine.validate(pa.Table.from_batches(batches[::-1]))["passed"]
    finally:
        engine.close()
