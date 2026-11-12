"""Bounded inner matching through DataFrame and SQL, with independent finite runs."""

from __future__ import annotations

from datetime import timedelta

import pyarrow as pa

import calc_flow as cf

JOIN_QUERY = """SELECT * FROM CF_INNER_JOIN('a', 'b', '{
    "spec": {
        "join_type": "inner", "left_keys": ["k"], "right_keys": ["k"],
        "left_event_time": "ts", "right_event_time": "ts",
        "bounds": {"before_micros": 2, "after_micros": 3},
        "limits": {"max_state_rows_per_side": 100,
            "max_state_bytes_per_side": 1048576, "max_matches_per_input_batch": 1000}
    }
}')"""


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
    df = cf.table.stream_join(
        a,
        b,
        left_keys=["k"],
        right_keys=["k"],
        left_event_time="ts",
        right_event_time="ts",
        bounds=cf.JoinTimeBounds(timedelta(microseconds=2), timedelta(microseconds=3)),
        limits=cf.JoinStateLimits(100, 1048576, 1000),
    )
    sql = cf.sql(JOIN_QUERY, a=a, b=b)
    program = cf.Program("pairs", mode="batch", outputs={"df": df, "sql": sql})

    def data(times):
        return pa.Table.from_arrays(
            [
                pa.array([1] * len(times), type=pa.int64()),
                pa.array(times, type=schema.field("ts").type),
                pa.array(range(len(times)), type=pa.uint64()),
            ],
            schema=schema,
        )

    expected = [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]
    for result in program.collect(
        {"a": data([10, 10]), "b": data([7, 8, 8, 13, 14])}
    ).values():
        actual = sorted(
            zip(result["left__seq"].to_pylist(), result["right__seq"].to_pylist())
        )
        if actual != expected:
            raise RuntimeError("finite inner join pair oracle mismatch")
    program.for_mode("stream").compile_stream()
    print("DataFrame and SQL six-pair oracles passed; both lifecycles compile")


if __name__ == "__main__":
    main()
