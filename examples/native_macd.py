"""Compute immutable array owners and shared Table MACD component columns."""

from __future__ import annotations

import pyarrow as pa

import calc_flow as cf
import calc_flow.array as cfa


def _array_values() -> dict[str, list[float | None]]:
    values = cfa.from_arrow(pa.array([10.0, 14.0, None, None]))
    parts = cfa.macd_components(values, fast_span=1, slow_span=3, signal_span=3)
    return {
        "line": parts.line.to_arrow().to_pylist(),
        "signal": parts.signal.to_arrow().to_pylist(),
        "histogram": parts.histogram.to_arrow().to_pylist(),
    }


def _table_values() -> dict[str, list[float | None]]:
    schema = pa.schema(
        [
            pa.field("entity", pa.string(), nullable=False),
            pa.field("time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("seq", pa.uint64(), nullable=False),
            pa.field("price", pa.uint64(), nullable=True),
        ]
    )
    data = pa.table(
        {
            "entity": ["a"] * 4,
            "time": [1, 2, 3, 4],
            "seq": [1, 2, 3, 4],
            "price": [10, 14, None, None],
        },
        schema=schema,
    )
    source = cf.table_input(
        "prices",
        schema=schema,
        entity_by=("entity",),
        event_time="time",
        sequence_by=("seq",),
    )
    parts = cf.ts.macd_components(
        source["price"], fast_span=1, slow_span=3, signal_span=3
    )
    output = source.select(
        line=parts.line, signal=parts.signal, histogram=parts.histogram
    )
    return (
        cf.Program("table-macd", engine="sql", outputs={"out": output})
        .collect({"prices": data})["out"]
        .to_pydict()
    )


def main() -> None:
    expected = {
        "line": [0.0, 2.0, 2.0, 2.0],
        "signal": [0.0, 1.0, 1.5, 1.75],
        "histogram": [0.0, 1.0, 0.5, 0.25],
    }
    for actual in (_array_values(), _table_values()):
        if actual != expected:
            raise AssertionError("MACD components differ from the expected example")
    print("MACD signal:", expected["signal"])


if __name__ == "__main__":
    main()
