from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import FrozenInstanceError, fields, replace

import pyarrow as pa
import pytest
from typing_extensions import TypeAliasType

from calc_flow import Field, ReplayPositioning, _compat, table_input
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


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires native strict zip")
def test_modern_strict_zip_uses_builtin_fast_path() -> None:
    assert _compat.zip is zip
    assert list(_compat.zip([1, 2], ["a", "b"], strict=True)) == [
        (1, "a"),
        (2, "b"),
    ]


def test_python39_compatibility_paths_keep_dataclass_enum_and_zip_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "calc_flow._compat_python39_test", _compat.__file__
    )
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "version_info", (3, 9, 0))
        spec.loader.exec_module(legacy)

    assert legacy.TypeAliasType is TypeAliasType

    class Mode(legacy.StrEnum):
        LIVE = "live"

    assert isinstance(Mode.LIVE, str)
    assert str(Mode.LIVE) == "live"

    @legacy.dataclass(slots=True, frozen=True)
    class Record:
        name: str
        count: int

    record = Record("events", 2)
    assert not hasattr(record, "__dict__")
    assert record.__getstate__() == ["events", 2]
    restored = object.__new__(Record)
    restored.__setstate__(["events", 2])
    assert restored == record
    with pytest.raises(FrozenInstanceError):
        record.count = 3  # type: ignore[misc]

    class Parent:
        __slots__ = "name"

    @legacy.dataclass(slots=True)
    class Child(Parent):
        name: str
        count: int

    assert Child.__slots__ == ("count",)
    assert Child("events", 2).name == "events"

    @legacy.dataclass()
    class Unslotted:
        value: int

    assert hasattr(Unslotted(1), "__dict__")

    with pytest.raises(TypeError, match="already specifies __slots__"):

        @legacy.dataclass(slots=True)
        class OwnSlots:
            __slots__ = ("value",)
            value: int

    assert list(legacy.zip([1, 2], ["a", "b"], strict=True)) == [
        (1, "a"),
        (2, "b"),
    ]
    assert list(legacy.zip([1], ["a"])) == [(1, "a")]
    with pytest.raises(ValueError, match="different lengths"):
        list(legacy.zip([1], ["a", "b"], strict=True))


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
