"""Complete-history top-k through both public Python frontends."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf


def main() -> None:
    schema = pa.schema(
        [pa.field("x", pa.int64()), pa.field("seq", pa.uint64(), nullable=False)]
    )
    events = cf.table_input("events", schema=schema)
    data = pa.Table.from_pydict({"x": [3, 1, 2], "seq": [10, 11, 12]}, schema)
    spec = cf.SortSpec.bounded_source(
        cf.StateBudget(1000, 8 << 20), sequence_by=["seq"]
    )
    declarations = (
        events.top_k(1, [cf.SortKey("x")], spec=spec),
        cf.sql(
            "SELECT * FROM CF_ORDER_BOUNDED('a',1000,8388608,ARRAY['seq']) "
            "ORDER BY x LIMIT 1",
            a=events,
        ),
    )
    for out in declarations:
        program = cf.Program("smallest", mode="batch", outputs={"out": out})
        result = program.collect({"events": data})["out"]
        if result.to_pydict() != {"x": [1], "seq": [11]}:
            raise RuntimeError("complete-history top-1 returned an unexpected relation")
        print(result.to_pydict())


if __name__ == "__main__":
    main()
