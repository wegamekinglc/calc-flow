from __future__ import annotations

from datetime import timedelta

import pyarrow as pa
import pytest

import calc_flow as cf
from calc_flow._stream_inputs import _source_binding

_SCHEMA = pa.schema(
    [
        pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
    ]
)


def _rows() -> pa.Table:
    return pa.table({"ts": [1, 2], "price": [1.0, 2.0]}, schema=_SCHEMA)


async def _feed():
    yield _rows()


@pytest.mark.parametrize("kind", ["iterable", "materialized", "sequence"])
def test_default_ordered_policy_emits_watermarks_after_every_batch(kind: str) -> None:
    source = cf.table_input("quotes", schema=_SCHEMA, event_time="ts")
    data = {"iterable": _feed(), "materialized": _rows(), "sequence": [_rows()]}[kind]

    binding, _ = _source_binding(data, ("quotes", source), cf.EdgeBudget(), None)

    assert binding.watermark_policy == cf.BoundedOutOfOrderness(
        "ts", timedelta(microseconds=1), timedelta(0)
    )


def test_default_unordered_policy_stays_disabled() -> None:
    source = cf.table_input("values", schema=pa.schema([("value", pa.int64())]))

    binding, _ = _source_binding(
        pa.table({"value": [1]}), ("values", source), cf.EdgeBudget(), None
    )

    assert binding.watermark_policy == cf.DisabledWatermarks()
