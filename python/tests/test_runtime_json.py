from __future__ import annotations

import asyncio
import json
import sys
from collections import UserDict
from collections.abc import Callable, Mapping
from dataclasses import FrozenInstanceError, replace
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    Cursor,
    Data,
    DisabledWatermarks,
    NativeWatermarkCapability,
    ReplayPositioning,
    SinkBinding,
    SinkRecovery,
    SourceBinding,
    SourceCapabilities,
    SourceDeliveryCapability,
    TransactionalDelivery,
)


@pytest.fixture(params=("cursor", "recovery"))
def owner_factory(request) -> Callable[[Mapping[str, Any]], Cursor | SinkRecovery]:
    if request.param == "cursor":
        return lambda payload: Cursor(b"1", payload)
    return lambda payload: SinkRecovery(1, False, TransactionalDelivery(), payload)


def _mapping(owner: Cursor | SinkRecovery) -> Mapping[str, Any]:
    return owner.payload if isinstance(owner, Cursor) else owner.pre_commit


@pytest.mark.parametrize("access", ("index", "get", "items", "values", "dict"))
def test_nested_json_reads_are_isolated(owner_factory, access: str) -> None:
    original = {"position": {"offsets": [1]}}
    owner = owner_factory(UserDict(original))
    mapping = _mapping(owner)
    original["position"]["offsets"].append(2)
    assert mapping == {"position": {"offsets": [1]}}

    def read() -> dict[str, Any]:
        if access == "index":
            return mapping["position"]
        if access == "get":
            return mapping.get("position")
        if access == "items":
            return next(iter(mapping.items()))[1]
        if access == "values":
            return next(iter(mapping.values()))
        return dict(mapping)["position"]

    first, second = read(), read()
    assert type(first) is dict
    assert type(first["offsets"]) is list
    first["offsets"].append(99)
    first["added"] = True
    assert second == read() == {"offsets": [1]}
    assert _mapping(owner) == {"position": {"offsets": [1]}}
    with pytest.raises(TypeError):
        mapping["position"] = {}  # type: ignore[index]
    with pytest.raises(KeyError, match="missing"):
        mapping["missing"]
    default: list[int] = []
    assert mapping.get("missing", default) is default
    assert json.loads(json.dumps(dict(mapping))) == mapping


@pytest.mark.parametrize("operation", ("copy", "left_union", "right_union"))
def test_proxy_snapshots_preserve_union_precedence_and_isolation(
    owner_factory, operation: str
) -> None:
    mapping = _mapping(owner_factory({"position": {"offsets": [1]}, "z": 0}))
    assert isinstance(mapping, MappingProxyType)
    if operation == "copy":
        snapshot = mapping.copy()
        assert snapshot["z"] == 0
    elif operation == "left_union":
        snapshot = mapping | {"z": 2}
        assert snapshot["z"] == 2
    else:
        snapshot = {"z": 2} | mapping
        assert snapshot["z"] == 0
    assert type(snapshot) is dict
    snapshot["position"]["offsets"].append(99)
    assert mapping["position"] == {"offsets": [1]}
    assert list(reversed(mapping)) == ["z", "position"]
    with pytest.raises(TypeError):
        mapping | []
    with pytest.raises(TypeError):
        [] | mapping


@pytest.mark.parametrize("view_name", ("keys", "items", "values"))
@pytest.mark.parametrize("empty", (False, True))
def test_json_views_preserve_forward_and_reverse_order(
    owner_factory, view_name: str, empty: bool
) -> None:
    payload = {} if empty else {"z": {"offsets": [2]}, "a": {"offsets": [1]}}
    mapping = _mapping(owner_factory(payload))
    view = getattr(mapping, view_name)()
    expected = getattr(dict(mapping), view_name)()
    assert len(view) == len(expected)
    assert list(view) == list(expected)
    assert list(reversed(view)) == list(reversed(expected))


@pytest.mark.parametrize(
    "item",
    (
        None,
        1,
        "missing",
        ["offset", 1],
        {"offset": 1},
        (),
        ("offset",),
        ("offset", 1, 2),
        ("offset", 1),
        ("offset", 2),
        ("missing", 1),
        ("position", {"offsets": [1]}),
    ),
)
def test_json_items_membership_matches_dict_items(owner_factory, item: object) -> None:
    payload = {"offset": 1, "position": {"offsets": [1]}}
    view = _mapping(owner_factory(payload)).items()
    assert (item in view) == (item in payload.items())


def test_json_items_membership_uses_tuple_contents(owner_factory) -> None:
    class Item(tuple):
        def __len__(self):
            raise AssertionError("membership must use the tuple contents")

        def __iter__(self):
            raise AssertionError("membership must use the tuple contents")

        def __getitem__(self, key):
            raise AssertionError("membership must use the tuple contents")

    payload = {"offset": 1}
    view = _mapping(owner_factory(payload)).items()
    item = Item(("offset", 1))
    assert (item in view) == (item in payload.items())


def test_json_items_membership_preserves_unhashable_key_errors(owner_factory) -> None:
    payload = {"offset": 1}
    view = _mapping(owner_factory(payload)).items()
    item = ([], 1)
    with pytest.raises(TypeError) as expected:
        assert item in payload.items()
    with pytest.raises(TypeError) as actual:
        assert item in view
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    "items",
    (
        {"missing"},
        {("offset", 1)},
        {"missing", ("offset", 1)},
        {("offset", 2)},
        {("offset",)},
        {None},
    ),
)
def test_json_items_set_operations_match_dict_items(owner_factory, items) -> None:
    payload = {"offset": 1}
    view = _mapping(owner_factory(payload)).items()
    expected = payload.items()
    assert view.isdisjoint(items) == expected.isdisjoint(items)
    assert view & items == expected & items
    assert items & view == items & expected


@pytest.mark.parametrize("view_name", ("items", "values"))
@pytest.mark.parametrize(
    "directions", ((iter, iter), (reversed, reversed), (iter, reversed))
)
def test_same_json_view_keeps_independent_readers_isolated(
    owner_factory, view_name: str, directions
) -> None:
    mapping = _mapping(owner_factory({"a": {"offsets": [1]}, "b": {"offsets": [2]}}))
    view = getattr(mapping, view_name)()
    first_direction, other_direction = directions
    first = list(first_direction(view))
    second = list(other_direction(view))
    pending = other_direction(view)
    # Compare the same key even when the readers traverse in opposite directions.
    if first_direction is not other_direction:
        second.reverse()
    first_value = first[0][1] if view_name == "items" else first[0]
    second_value = second[0][1] if view_name == "items" else second[0]
    expected = {"offsets": list(second_value["offsets"])}
    first_value["offsets"].append(99)
    first_value["changed"] = True
    assert second_value == expected
    assert list(pending) == list(other_direction(view))
    assert list(view) == list(getattr(dict(mapping), view_name)())
    assert mapping == {"a": {"offsets": [1]}, "b": {"offsets": [2]}}


@pytest.mark.parametrize("view_name", ("keys", "items", "values"))
def test_json_view_mapping_and_lifetime_are_safe(owner_factory, view_name: str) -> None:
    owner = owner_factory({"position": {"offsets": [1]}})
    view = getattr(_mapping(owner), view_name)()
    del owner
    assert len(list(view)) == 1
    if sys.version_info < (3, 10):
        assert not hasattr(view, "mapping")
        return
    mapping = view.mapping
    assert isinstance(mapping, MappingProxyType)
    assert list(reversed(mapping)) == ["position"]
    assert list(reversed(mapping.items())) == [("position", {"offsets": [1]})]
    reader = mapping["position"]
    for value in (
        mapping["position"],
        mapping.get("position"),
        next(iter(mapping.items()))[1],
        next(iter(mapping.values())),
        dict(mapping)["position"],
    ):
        value["offsets"].append(99)
    assert reader == mapping["position"] == {"offsets": [1]}
    with pytest.raises(TypeError):
        mapping["position"] = {}
    with pytest.raises(AttributeError):
        view.mapping = {}


def test_json_owners_preserve_dataclass_behavior(owner_factory) -> None:
    owner = owner_factory({"position": {"offsets": [1]}})
    assert owner == owner_factory({"position": {"offsets": [1]}})
    assert owner != owner_factory({"position": {"offsets": [2]}})
    assert "mappingproxy({'position': {'offsets': [1]}})" in repr(owner)
    assert "_Isolated" not in repr(owner)
    assert replace(owner) == owner
    if isinstance(owner, Cursor):
        assert owner == Cursor(order=b"1", payload=_mapping(owner), source_id=None)
        changed = replace(owner, source_id="input")
        assert changed.source_id == "input"
        assert owner.source_id is None
        field = "payload"
    else:
        assert owner == SinkRecovery(
            epoch=1,
            terminal=False,
            delivery=TransactionalDelivery(),
            pre_commit=_mapping(owner),
        )
        assert replace(owner, epoch=2).epoch == 2
        assert owner.epoch == 1
        field = "pre_commit"
    with pytest.raises(FrozenInstanceError):
        setattr(owner, field, {})


@pytest.mark.parametrize(
    ("payload", "message"),
    (
        ({"bad": (1,)}, "project contains a non-JSON value of type tuple"),
        ({"bad": float("nan")}, "project JSON numbers must be finite"),
        ({"bad": 2**64}, "project integer is outside the portable JSON range"),
        ({1: None}, "project JSON object keys must be strings"),
        (
            {"first": float("nan"), "last": (1,)},
            "project contains a non-JSON value of type tuple",
        ),
    ),
)
def test_json_owner_validation_preserves_errors(
    owner_factory, payload, message
) -> None:
    with pytest.raises(ValueError) as caught:
        owner_factory(payload)
    assert str(caught.value) == message


def test_cursor_validates_identity_before_payload() -> None:
    for order, source_id, error_type, message in (
        (1, None, TypeError, "cursor order must be bytes"),
        (b"", None, ValueError, "cursor order must not be empty"),
        (b"1", "", TypeError, "cursor source_id must be a non-empty string or None"),
    ):
        with pytest.raises(error_type) as caught:
            Cursor(order, {"bad": (1,)}, source_id)
        assert str(caught.value) == message


def test_source_adapter_isolates_nested_cursor_json() -> None:
    cursor = Cursor(b"1", {"position": {"offsets": [1]}}, "input")
    batch = Batch.from_pyarrow(pa.table({"value": [1]}))
    opened: list[Cursor | None] = []

    class Source:
        def capabilities(self) -> SourceCapabilities:
            return SourceCapabilities(
                ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
                SourceDeliveryCapability.LOSSLESS,
                1,
                1024,
                native_watermarks=NativeWatermarkCapability.NEVER_EMITS,
            )

        async def open(self, value: Cursor | None) -> None:
            opened.append(value)

        async def next(self) -> Data:
            return Data(batch, cursor)

        async def close(self) -> None:
            pass

    async def exercise() -> None:
        binding = SourceBinding(Source(), watermark_policy=DisabledWatermarks())
        await binding._native_open(None, None, None)
        assert opened == [None]
        payload = {"position": {"offsets": [1]}}
        await binding._native_open("input", b"1", payload)
        payload["position"]["offsets"].append(99)
        assert opened[1] == cursor
        result = await binding._native_next()
        assert result[:4] == ("data", batch, "input", b"1")
        output = result[4]
        assert type(output) is dict
        assert json.loads(json.dumps(output)) == dict(cursor.payload)
        output["position"]["offsets"].append(99)
        assert cursor.payload["position"] == {"offsets": [1]}
        assert (await binding._native_next())[4] == dict(cursor.payload)

    asyncio.run(exercise())


@pytest.fixture
def sink_lifecycle_state():
    original = {"position": {"offsets": [1]}}
    received: list[Mapping[str, Any] | None] = []
    recoveries: list[SinkRecovery] = []

    class Sink:
        async def open(self) -> None:
            pass

        async def begin_epoch(self, epoch: int) -> None:
            pass

        async def write(self, batch: Batch) -> None:
            pass

        async def pre_commit(self, epoch: int) -> Mapping[str, Any]:
            return original

        async def commit(self, epoch: int, evidence: Mapping[str, Any]) -> None:
            received.append(evidence)

        async def abort(self, epoch: int, evidence: Mapping[str, Any] | None) -> None:
            received.append(evidence)

        async def recover(self, recovery: SinkRecovery) -> None:
            recoveries.append(recovery)

        async def close(self) -> None:
            pass

    return Sink(), original, received, recoveries


def test_sink_adapter_isolates_nested_evidence_at_each_lifecycle_boundary(
    sink_lifecycle_state,
) -> None:
    sink, original, received, recoveries = sink_lifecycle_state

    async def exercise() -> None:
        binding = SinkBinding.transactional("archive", sink)
        output = await binding._native_pre_commit(1)
        assert type(output) is dict
        assert type(output["position"]["offsets"]) is list
        assert json.loads(json.dumps(output)) == original
        output["position"]["offsets"].append(99)
        assert original == {"position": {"offsets": [1]}}
        await binding._native_commit(1, original)
        await binding._native_abort(1, original)
        await binding._native_abort(1, None)
        await binding._native_recover(1, False, {"kind": "transactional"}, original)
        recovery = recoveries[0]
        assert (recovery.epoch, recovery.terminal, recovery.delivery) == (
            1,
            False,
            TransactionalDelivery(),
        )
        original["position"]["offsets"].append(2)
        for evidence in (*received[:2], recovery.pre_commit):
            reader = evidence["position"]
            evidence["position"]["offsets"].append(99)
            assert reader == evidence["position"] == {"offsets": [1]}
        assert received[2] is None
        assert original == {"position": {"offsets": [1, 2]}}

    asyncio.run(exercise())
