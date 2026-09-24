from __future__ import annotations

import asyncio
import sys
from dataclasses import FrozenInstanceError, fields, replace

import pyarrow as pa
import pytest

from calc_flow import Field, ReplayPositioning, table_input
from calc_flow.runtime import WatermarkPolicy


def test_public_field_keeps_frozen_slotted_dataclass_behavior() -> None:
    field = Field("price", "float64", nullable=False)

    assert not hasattr(field, "__dict__")
    assert tuple(item.name for item in fields(field)) == (
        "name",
        "data_type",
        "nullable",
    )
    assert replace(field, name="volume") == Field("volume", "float64", False)
    with pytest.raises(FrozenInstanceError):
        field.name = "changed"  # type: ignore[misc]


def test_public_string_enum_keeps_string_behavior() -> None:
    value = ReplayPositioning.UNSUPPORTED

    assert isinstance(value, str)
    assert str(value) == "unsupported"
    assert value == "unsupported"


def test_modern_type_alias_retains_stdlib_identity() -> None:
    if sys.version_info >= (3, 12):
        from typing import TypeAliasType

        assert isinstance(WatermarkPolicy, TypeAliasType)


def test_async_stream_keeps_owned_result_lifecycle() -> None:
    source = table_input("events", schema=pa.schema([("value", pa.int64())]))

    async def feed():
        yield pa.table({"value": [1, 2]})
        yield pa.table({"value": [3]})

    results = source.select(doubled=source["value"] * 2).stream(feed())

    async def run():
        async with results:
            tables = [table async for table in results]
        assert pa.concat_tables(tables).to_pydict() == {"doubled": [2, 4, 6]}
        assert results.job.status()["state"] == "completed"
        assert results.job.status()["task_count"] == 0

    asyncio.run(asyncio.wait_for(run(), timeout=5))
