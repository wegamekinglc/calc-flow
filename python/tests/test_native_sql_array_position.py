from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def test_native_sql_plans_and_executes_rolling_array_position():
    original = {
        "sequence": list(range(7)),
        "price": [3.0, 5.0, 5.0, 2.0, 1.0, 8.0, 1.0],
    }
    table = pa.table(original)
    query = (
        "SELECT sequence, CAST(COUNT(*) OVER trailing - "
        "CAST(array_position(array_agg(price) OVER trailing, "
        "MAX(price) OVER trailing) AS BIGINT) AS DOUBLE) AS lag FROM input "
        "WINDOW trailing AS (ORDER BY sequence "
        "ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)"
    )
    result = cf.compute(table, lambda source: source.sql(query)).sort_by("sequence")
    assert result["lag"].type == pa.float64()
    assert result["lag"].to_pylist() == [0.0, 0.0, 1.0, 2.0, 2.0, 0.0, 1.0]
    assert table.to_pydict() == original
