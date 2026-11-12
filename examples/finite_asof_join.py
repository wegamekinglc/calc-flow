"""Complete-input backward ASOF through the DataFrame and SQL frontends."""

from __future__ import annotations

import json
from datetime import timedelta

import pyarrow as pa

import calc_flow as cf


def main():
    schema = pa.schema(
        [
            pa.field("k", pa.int64(), False),
            pa.field("ts", pa.timestamp("us", tz="UTC"), False),
            pa.field("seq", pa.uint64(), False),
        ]
    )
    a, b = [
        cf.table_input(
            name, schema=schema, entity_by=["k"], event_time="ts", sequence_by=["seq"]
        )
        for name in ("a", "b")
    ]
    df = a.stream_asof_join(
        b, tolerance=timedelta(microseconds=2), limits=cf.AsofStateLimits(100, 1048576)
    )
    side = dict(keys=["k"], event_time="ts", sequence_by=["seq"])
    config = json.dumps(
        dict(
            spec=dict(
                left=dict(side, prefix="left"),
                right=dict(side, prefix="right"),
                tolerance_micros=2,
                limits=dict(max_state_rows=100, max_state_bytes=1048576),
                late_policy="error",
            )
        )
    )
    sql = cf.sql(f"SELECT * FROM CF_ASOF_JOIN('a', 'b', '{config}')", a=a, b=b)

    def data(times, sequences):
        return pa.Table.from_arrays(
            [
                pa.array([1] * len(times)),
                pa.array(times, type=schema.field("ts").type),
                pa.array(sequences, type=pa.uint64()),
            ],
            schema=schema,
        )

    p = cf.Program("historical", mode="batch", outputs={"df": df, "sql": sql})
    for table in p.collect(
        {"a": data([10, 20], [1, 2]), "b": data([8, 9, 9], [1, 2, 10])}
    ).values():
        assert table["right__seq"].to_pylist() == [10, None]
    print("DataFrame and SQL ASOF: typed tie selects 10; unmatched row remains")


if __name__ == "__main__":
    main()
