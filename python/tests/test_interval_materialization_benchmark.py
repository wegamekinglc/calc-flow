from __future__ import annotations

import pyarrow as pa

from benchmarks.interval_materialization import gather_projection, prepare_gathers


def test_projection_uses_physical_positions_and_right_ingress_boundaries():
    source = pa.table(
        {"sequence": pa.array([8, 3, 5], type=pa.uint64()), "price": [2.0, 3.0, 5.0]}
    )
    # The disposable oracle deliberately orders pairs differently from ingress.
    pairs = pa.table(
        {
            "sequence": pa.array([3, 5, 8, 8], type=pa.uint64()),
            "right_sequence": pa.array([5, 8, 3, 8], type=pa.uint64()),
        }
    )
    columns, groups = prepare_gathers(source, pairs, ingress_rows=2)
    result = gather_projection(columns, groups)
    assert [len(batch) for batch in result.to_batches()] == [3, 1]
    assert result.to_pydict() == {
        "sequence": [8, 5, 8, 3],
        "right_sequence": [8, 8, 3, 5],
        "value": [4.0, 10.0, 6.0, 15.0],
    }
