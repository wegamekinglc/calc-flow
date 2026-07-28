from __future__ import annotations

import asyncio
import gc
import inspect
import json
import math
import threading
import time
import weakref
from collections import UserDict
from collections.abc import Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone, tzinfo

import numpy as np
import pyarrow as pa
import pytest

import calc_flow
from calc_flow import (
    Batch,
    CancelledError,
    ExecutionOptions,
    PipelineBuilder,
    ProviderContext,
    ProviderError,
    Runtime,
)


def _nested_settings(depth: int) -> dict[str, object]:
    root: dict[str, object] = {}
    current = root
    for _ in range(depth):
        child: dict[str, object] = {}
        current["value"] = child
        current = child
    return root


class _OnePassMapping(Mapping[str, object]):
    def __init__(self, pairs: list[tuple[object, object]]) -> None:
        self._pairs = pairs
        self.items_calls = 0
        self.iterations = 0

    def __getitem__(self, key: str) -> object:
        raise AssertionError("strict settings must not call __getitem__")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("strict settings must not call __iter__")

    def __len__(self) -> int:
        raise AssertionError("strict settings must not call __len__")

    def items(self) -> Iterator[tuple[object, object]]:  # type: ignore[override]
        self.items_calls += 1

        def iterate() -> Iterator[tuple[object, object]]:
            self.iterations += 1
            yield from self._pairs

        return iterate()


class _CallableMapping(_OnePassMapping):
    def __call__(self) -> None:
        raise AssertionError("strict settings must retain data, not invoke the mapping")


class _SecretMappingError(RuntimeError):
    pass


class _FailingMapping(_OnePassMapping):
    def items(self) -> Iterator[tuple[object, object]]:  # type: ignore[override]
        raise _SecretMappingError("secret-key=https://example.invalid/token")


def test_execution_options_defaults_identity_and_frozen_fields() -> None:
    options = ExecutionOptions()

    assert calc_flow.ExecutionOptions is calc_flow._native.ExecutionOptions
    assert options.settings == {}
    assert options.deadline is None
    with pytest.raises(AttributeError):
        options.settings = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        options.deadline = datetime.now(UTC)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        options.extra = True  # type: ignore[attr-defined]


def test_native_execution_and_provider_signatures_are_canonical() -> None:
    assert str(inspect.signature(ExecutionOptions)) == "(settings={}, deadline=None)"

    runtime = calc_flow._native.Runtime()
    assert str(inspect.signature(runtime.register_provider)) == (
        "(provider, name, version, callback, *, accepts_context=False)"
    )
    assert str(inspect.signature(runtime._register_mapping_provider)) == (
        "(provider, name, version, callback, *, input_ports, output_ports, "
        "accepts_context=False)"
    )
    assert (
        inspect.signature(runtime.register_provider)
        .parameters["accepts_context"]
        .default
        is False
    )
    assert (
        inspect.signature(runtime._register_mapping_provider)
        .parameters["accepts_context"]
        .default
        is False
    )


def test_execution_options_snapshots_settings_and_returns_fresh_copies() -> None:
    nested = {"list": [1, {"enabled": True}]}
    source = {"nested": nested}
    options = ExecutionOptions(source)

    source["added"] = "later"
    nested["list"].append("later")
    observed = options.settings
    observed["nested"]["list"].append("observer")  # type: ignore[index, union-attr]

    assert options.settings == {"nested": {"list": [1, {"enabled": True}]}}
    assert options.settings is not observed


def test_execution_options_accepts_one_root_mapping_materialization() -> None:
    class CountingMapping(UserDict[str, object]):
        copies = 0

        def items(self):
            self.copies += 1
            return super().items()

    source = CountingMapping({"nested": [None, False, 1, 2**64 - 1, 1.25, "x"]})
    options = ExecutionOptions(source)

    assert source.copies == 1
    assert options.settings == {"nested": [None, False, 1, 2**64 - 1, 1.25, "x"]}


def test_execution_options_accepts_explicit_none_as_empty_settings() -> None:
    assert ExecutionOptions(settings=None).settings == {}
    assert ExecutionOptions(None).settings == {}


def test_execution_options_snapshots_nested_mapping_once_without_other_hooks() -> None:
    nested = _CallableMapping(
        [
            ("enabled", True),
            ("items", [_OnePassMapping([("value", 7)])]),
        ]
    )
    options = ExecutionOptions(_OnePassMapping([("nested", nested)]))

    assert options.settings == {"nested": {"enabled": True, "items": [{"value": 7}]}}
    assert nested.items_calls == 1
    assert nested.iterations == 1


def test_execution_options_rejects_duplicate_custom_mapping_keys() -> None:
    source = _OnePassMapping([("secret-key", 1), ("secret-key", 2)])

    with pytest.raises(ValueError) as caught:
        ExecutionOptions(source)

    assert str(caught.value) == "settings at $ contains duplicate object keys"
    assert "secret-key" not in str(caught.value)
    assert source.items_calls == 1
    assert source.iterations == 1


@pytest.mark.parametrize(
    "source",
    [
        _FailingMapping([]),
        _OnePassMapping([("valid", 1), ("malformed",)]),  # type: ignore[list-item]
    ],
)
def test_execution_options_redacts_mapping_hook_and_pair_failures(
    source: Mapping[str, object],
) -> None:
    with pytest.raises(ValueError) as caught:
        ExecutionOptions(source)

    assert str(caught.value) == "settings could not be copied as strict JSON data"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "secret" not in str(caught.value)
    assert "_SecretMappingError" not in str(caught.value)


@pytest.mark.parametrize(
    "settings",
    [
        [],
        (),
        "value",
        1,
    ],
)
def test_execution_options_rejects_non_mapping_settings(settings: object) -> None:
    with pytest.raises(TypeError, match="^settings must be a mapping or None$"):
        ExecutionOptions(settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        (1,),
        {1},
        b"bytes",
        object(),
        datetime.now(UTC),
        math.nan,
        math.inf,
        -math.inf,
        -(2**63) - 1,
        2**64,
    ],
)
def test_execution_options_rejects_non_strict_json_values(value: object) -> None:
    with pytest.raises(ValueError, match="settings"):
        ExecutionOptions({"value": value})  # type: ignore[dict-item]


def test_execution_options_rejects_non_string_and_subclassed_keys() -> None:
    class StringSubclass(str):
        pass

    for key in (1, StringSubclass("value")):
        with pytest.raises(ValueError, match="settings"):
            ExecutionOptions({key: True})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "value",
    [
        type("IntSubclass", (int,), {})(1),
        type("FloatSubclass", (float,), {})(1.0),
        type("StringSubclass", (str,), {})("value"),
        type("ListSubclass", (list,), {})([1]),
    ],
)
def test_execution_options_rejects_json_type_subclasses(value: object) -> None:
    with pytest.raises(ValueError, match="settings"):
        ExecutionOptions({"value": value})  # type: ignore[dict-item]


def test_execution_options_accepts_mapping_subclasses_but_not_list_subclasses() -> None:
    class DictSubclass(dict[str, object]):
        pass

    class ListSubclass(list[object]):
        pass

    assert ExecutionOptions({"value": DictSubclass(nested=1)}).settings == {
        "value": {"nested": 1}
    }
    with pytest.raises(
        ValueError,
        match=r"^settings at \$\.\* contains a non-JSON value$",
    ):
        ExecutionOptions({"value": ListSubclass([1])})


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        (
            {"value": object()},
            "settings at $.* contains a non-JSON value",
        ),
        (
            {1: True},
            "settings at $ contains a non-string object key",
        ),
        (
            {"value": -(2**63) - 1},
            "settings at $.* contains an integer outside the portable JSON range",
        ),
        (
            {"value": 2**64},
            "settings at $.* contains an integer outside the portable JSON range",
        ),
        (
            {"value": math.inf},
            "settings at $.* contains a non-finite JSON number",
        ),
        (
            {"value": "\ud800"},
            "settings at $.* contains a non-portable Unicode string",
        ),
        (
            {"value": "\ud83d\ude00"},
            "settings at $.* contains a non-portable Unicode string",
        ),
        (
            {"\udfff": True},
            "settings at $ contains a non-portable Unicode string",
        ),
    ],
)
def test_execution_options_uses_stable_redacted_settings_errors(
    settings: dict[object, object],
    message: str,
) -> None:
    with pytest.raises(ValueError) as caught:
        ExecutionOptions(settings)  # type: ignore[arg-type]

    assert str(caught.value) == message
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_execution_options_enforces_depth_and_cycles_without_rejecting_aliases() -> (
    None
):
    ExecutionOptions(_nested_settings(32))
    with pytest.raises(ValueError) as too_deep:
        ExecutionOptions(_nested_settings(33))
    assert str(too_deep.value) == (
        "settings exceeds the maximum JSON depth of 32 at "
        "$.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*.*"
    )

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError) as cyclic:
        ExecutionOptions({"value": cycle})
    assert str(cyclic.value) == "settings at $.*[0] contains a cycle"

    shared = [1, 2]
    options = ExecutionOptions({"left": shared, "right": shared})
    observed = options.settings
    assert observed == {"left": [1, 2], "right": [1, 2]}
    assert observed["left"] is not observed["right"]


class _NoneOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> None:
        return None


class _BadOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> object:  # type: ignore[override]
        return object()


class _RaisingOffset(tzinfo):
    def utcoffset(self, value: datetime | None) -> timedelta:
        raise RuntimeError("must be translated")


@pytest.mark.parametrize(
    ("offset", "microsecond", "expected"),
    [
        (
            timedelta(0),
            0,
            datetime(2027, 4, 5, 6, 7, 8, 0, tzinfo=UTC),
        ),
        (
            timedelta(hours=8, minutes=30),
            1,
            datetime(2027, 4, 4, 21, 37, 8, 1, tzinfo=UTC),
        ),
        (
            -timedelta(hours=7, minutes=45),
            999_999,
            datetime(2027, 4, 5, 13, 52, 8, 999_999, tzinfo=UTC),
        ),
    ],
)
def test_execution_options_normalizes_any_aware_deadline_to_exact_utc(
    offset: timedelta,
    microsecond: int,
    expected: datetime,
) -> None:
    deadline = datetime(
        2027,
        4,
        5,
        6,
        7,
        8,
        microsecond,
        tzinfo=timezone(offset, "SECRET ZONE"),
    )

    options = ExecutionOptions(deadline=deadline)

    assert options.deadline == expected
    assert options.deadline is not deadline
    assert type(options.deadline) is datetime
    assert options.deadline is not None
    assert options.deadline.tzinfo is UTC


@pytest.mark.parametrize(
    "deadline",
    [
        datetime(2027, 4, 5, tzinfo=_BadOffset()),
        datetime(2027, 4, 5, tzinfo=_RaisingOffset()),
        datetime.min.replace(tzinfo=timezone(timedelta(hours=1))),
        datetime.max.replace(tzinfo=timezone(-timedelta(hours=1))),
    ],
)
def test_execution_options_redacts_invalid_deadline_behavior(
    deadline: datetime,
) -> None:
    with pytest.raises(ValueError) as caught:
        ExecutionOptions(deadline=deadline)

    assert str(caught.value) == (
        "deadline must be a valid timezone-aware datetime representable in UTC"
    )
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert "SECRET" not in str(caught.value)
    assert "translated" not in str(caught.value)


@pytest.mark.parametrize(
    "deadline",
    [
        datetime(2027, 4, 5),
        datetime(2027, 4, 5, tzinfo=_NoneOffset()),
    ],
)
def test_execution_options_rejects_naive_deadline_with_stable_error(
    deadline: datetime,
) -> None:
    with pytest.raises(ValueError) as caught:
        ExecutionOptions(deadline=deadline)

    assert str(caught.value) == "deadline must be timezone-aware"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_execution_options_rejects_non_datetime_deadline() -> None:
    with pytest.raises(TypeError) as caught:
        ExecutionOptions(deadline="later")  # type: ignore[arg-type]

    assert str(caught.value) == "deadline must be a datetime or None"
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_execution_options_constructor_preserves_normal_argument_errors() -> None:
    with pytest.raises(TypeError):
        ExecutionOptions({}, None, "extra")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ExecutionOptions({}, settings={})  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        ExecutionOptions(unexpected=True)  # type: ignore[call-arg]


def _table_batch(value: int = 1) -> Batch:
    return Batch.from_pyarrow(pa.table({"value": [value]}))


def test_execution_options_are_keyword_only_and_type_checked_before_execution() -> None:
    plan = PipelineBuilder("options-shape").expression("calc", "out = value").compile()
    inputs = {"input": _table_batch()}

    with pytest.raises(TypeError):
        plan.execute(inputs, ExecutionOptions())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        plan.execute(inputs, options={})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        plan.execute_async(inputs, ExecutionOptions())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        plan.execute_async(inputs, options={})  # type: ignore[arg-type]


def test_sync_event_loop_error_precedes_inputs_and_options_validation() -> None:
    plan = (
        PipelineBuilder("loop-precedence").expression("calc", "out = value").compile()
    )

    async def exercise() -> None:
        with pytest.raises(
            RuntimeError,
            match=(
                r"^execute\(\) cannot run inside an event loop; "
                r"use execute_async\(\)$"
            ),
        ):
            plan.execute(object(), options=object())  # type: ignore[arg-type]

    asyncio.run(exercise())


def test_already_expired_deadline_cancels_sync_before_provider_entry() -> None:
    calls = 0

    def callback(batch: Batch, _options: dict[str, object]) -> Batch:
        nonlocal calls
        calls += 1
        return batch

    runtime = Runtime()
    runtime.register_provider("test", "identity", "1", callback)
    plan = (
        PipelineBuilder("expired-sync")
        .external("identity", "test", "identity", "1", {})
        .compile(runtime)
    )

    with pytest.raises(CancelledError):
        plan.execute(
            {"input": Batch.from_array(np.array([1]), backend="test")},
            options=ExecutionOptions(deadline=datetime(2000, 1, 1, tzinfo=UTC)),
        )

    assert calls == 0
    assert plan.snapshot() == {"identity": None}


def test_already_expired_deadline_cancels_async_before_provider_entry() -> None:
    calls = 0

    def callback(batch: Batch, _options: dict[str, object]) -> Batch:
        nonlocal calls
        calls += 1
        return batch

    async def exercise() -> None:
        runtime = Runtime()
        runtime.register_provider("test", "identity", "1", callback)
        plan = (
            PipelineBuilder("expired-async")
            .external("identity", "test", "identity", "1", {})
            .compile(runtime)
        )
        with pytest.raises(CancelledError):
            await plan.execute_async(
                {"input": Batch.from_array(np.array([1]), backend="test")},
                options=ExecutionOptions(deadline=datetime(2000, 1, 1, tzinfo=UTC)),
            )
        assert await plan.snapshot_async() == {"identity": None}

    asyncio.run(exercise())
    assert calls == 0


def test_context_aware_single_provider_observes_sync_and_async_run_options() -> None:
    observed: list[tuple[dict[str, object], datetime | None]] = []
    retained: list[ProviderContext] = []

    def callback(
        batch: Batch,
        provider_options: dict[str, object],
        context: ProviderContext,
    ) -> Batch:
        assert provider_options == {"compiled": True}
        first = context.settings
        first["nested"]["values"].append("local")  # type: ignore[index, union-attr]
        observed.append((context.settings, context.deadline))
        retained.append(context)
        return batch

    runtime = Runtime()
    runtime.register_provider(
        "test",
        "context",
        "1",
        callback,
        accepts_context=True,
    )
    plan = (
        PipelineBuilder("context-single")
        .external("context", "test", "context", "1", {"compiled": True})
        .compile(runtime)
    )
    source = {"nested": {"values": [1, 2]}}
    deadline = datetime(2030, 5, 6, 7, 8, 9, 123456, tzinfo=UTC)
    options = ExecutionOptions(source, deadline)
    source["nested"]["values"].append("caller")
    inputs = {"input": Batch.from_array(np.array([1]), backend="test")}

    sync = plan.execute(inputs, options=options)
    asynchronous = asyncio.run(plan.execute_async(inputs, options=options))

    expected = {"nested": {"values": [1, 2]}}
    assert observed == [(expected, deadline), (expected, deadline)]
    assert sync.metadata["run_id"] != asynchronous.metadata["run_id"]
    assert options.settings == expected
    assert retained[0].settings is not retained[0].settings
    with pytest.raises(TypeError):
        ProviderContext()  # type: ignore[call-arg]
    with pytest.raises(AttributeError):
        retained[0].settings = {}  # type: ignore[misc]
    with pytest.raises(AttributeError):
        retained[0].extra = True  # type: ignore[attr-defined]


def test_context_aware_mapping_provider_observes_authoritative_run_options() -> None:
    observed: list[tuple[dict[str, object], datetime | None]] = []

    def callback(
        inputs: dict[str, Batch],
        provider_options: dict[str, object],
        context: ProviderContext,
    ) -> dict[str, Batch]:
        assert sorted(inputs) == ["table", "weights"]
        assert provider_options == {"columns": ["value"]}
        observed.append((context.settings, context.deadline))
        return {"output": inputs["weights"]}

    runtime = Runtime()
    runtime._register_mapping_provider(
        "test",
        "mapping",
        "1",
        callback,
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
        accepts_context=True,
    )
    project = (
        PipelineBuilder("context-mapping")
        .table_matmul("mapping", backend="numpy", columns=("value",))
        .project
    )
    project["pipeline"]["nodes"][0]["operator"].update(
        {"provider": "test", "name": "mapping"}
    )
    plan = runtime.compile_project(json.dumps(project))
    deadline = datetime(2030, 1, 2, 3, 4, 5, 600_007, tzinfo=UTC)

    result = asyncio.run(
        plan.execute_async(
            {
                "table": _table_batch(),
                "weights": Batch.from_array(np.array([[2.0]]), backend="numpy"),
            },
            options=ExecutionOptions({"request": {"id": 7}}, deadline),
        )
    )

    assert result.outputs["output"].array.tolist() == [[2.0]]
    assert observed == [({"request": {"id": 7}}, deadline)]


def test_legacy_and_context_provider_abis_are_explicit_and_one_shot() -> None:
    legacy_calls = 0
    failing_calls = 0

    def legacy(batch: Batch, provider_options: dict[str, object]) -> Batch:
        nonlocal legacy_calls
        legacy_calls += 1
        assert provider_options == {}
        return batch

    def raises_type_error(
        batch: Batch,
        provider_options: dict[str, object],
        context: ProviderContext,
    ) -> Batch:
        del batch, provider_options, context
        nonlocal failing_calls
        failing_calls += 1
        raise TypeError("application failure")

    runtime = Runtime()
    runtime.register_provider("test", "legacy", "1", legacy)
    runtime.register_provider(
        "test",
        "failing",
        "1",
        raises_type_error,
        accepts_context=True,
    )
    batch = Batch.from_array(np.array([1]), backend="test")
    legacy_plan = (
        PipelineBuilder("legacy")
        .external("provider", "test", "legacy", "1", {})
        .compile(runtime)
    )
    failing_plan = (
        PipelineBuilder("failing")
        .external("provider", "test", "failing", "1", {})
        .compile(runtime)
    )

    legacy_plan.execute({"input": batch})
    with pytest.raises(ProviderError, match="application failure"):
        failing_plan.execute({"input": batch}, options=ExecutionOptions())

    assert legacy_calls == 1
    assert failing_calls == 1


@pytest.mark.parametrize("accepts_context", [None, 0, 1, "yes", object()])
def test_accepts_context_requires_an_exact_bool_without_partial_registration(
    accepts_context: object,
) -> None:
    runtime = Runtime()
    before = runtime._registration_snapshot()

    with pytest.raises(TypeError, match="accepts_context"):
        runtime.register_provider(
            "test",
            "invalid",
            "1",
            lambda batch, options: batch,
            accepts_context=accepts_context,  # type: ignore[arg-type]
        )

    assert runtime._registration_snapshot() == before


def test_direct_native_registration_rejects_non_bool_context_flag() -> None:
    runtime = Runtime()

    for method, kwargs in (
        (
            runtime._inner.register_provider,
            {},
        ),
        (
            runtime._inner._register_mapping_provider,
            {
                "input_ports": (("input", "array"),),
                "output_ports": (("output", "array"),),
            },
        ),
    ):
        with pytest.raises(TypeError):
            method(
                "test",
                f"invalid-{len(kwargs)}",
                "1",
                lambda batch, options: batch,
                accepts_context=1,
                **kwargs,
            )


def test_selected_provider_arity_mismatch_is_a_provider_error() -> None:
    runtime = Runtime()
    runtime.register_provider(
        "test",
        "missing-context",
        "1",
        lambda batch, options: batch,
        accepts_context=True,
    )
    runtime.register_provider(
        "test",
        "unexpected-context",
        "1",
        lambda batch, options, context: batch,
    )
    batch = Batch.from_array(np.array([1]), backend="test")

    for name in ("missing-context", "unexpected-context"):
        plan = (
            PipelineBuilder(name)
            .external("provider", "test", name, "1", {})
            .compile(runtime)
        )
        with pytest.raises(ProviderError):
            plan.execute({"input": batch})


def test_concurrent_registrations_keep_their_context_flags_isolated() -> None:
    runtime = Runtime()

    def register(index: int) -> None:
        runtime.register_provider(
            "test",
            f"provider-{index}",
            "1",
            lambda batch, options, *context: batch,
            accepts_context=index % 2 == 0,
        )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(register, range(12)))

    snapshot = runtime._registration_snapshot()
    assert len(snapshot) == 12
    assert {
        registration["name"]: registration.get("accepts_context", False)
        for registration in snapshot
    } == {f"provider-{index}": index % 2 == 0 for index in range(12)}


def test_one_options_value_is_isolated_across_concurrent_independent_plans() -> None:
    contexts: list[dict[str, object]] = []

    def callback(
        batch: Batch,
        _provider_options: dict[str, object],
        context: ProviderContext,
    ) -> Batch:
        settings = context.settings
        settings["local"] = len(contexts)
        contexts.append(settings)
        return batch

    runtime = Runtime()
    runtime.register_provider(
        "test",
        "concurrent",
        "1",
        callback,
        accepts_context=True,
    )
    builder = PipelineBuilder("concurrent-options").external(
        "provider", "test", "concurrent", "1", {}
    )
    first_plan = builder.compile(runtime)
    second_plan = builder.compile(runtime)
    options = ExecutionOptions({"shared": [1, 2]})
    inputs = {"input": Batch.from_array(np.array([1]), backend="test")}

    async def exercise():
        return await asyncio.gather(
            first_plan.execute_async(inputs, options=options),
            second_plan.execute_async(inputs, options=options),
        )

    first, second = asyncio.run(exercise())

    assert first.metadata["run_id"] != second.metadata["run_id"]
    assert options.settings == {"shared": [1, 2]}
    assert contexts == [
        {"shared": [1, 2], "local": 0},
        {"shared": [1, 2], "local": 1},
    ] or contexts == [
        {"shared": [1, 2], "local": 1},
        {"shared": [1, 2], "local": 0},
    ]


def test_same_plan_queued_runs_keep_context_snapshots_isolated() -> None:
    started = threading.Event()
    release = threading.Event()
    observed: list[tuple[str, datetime | None]] = []

    def callback(
        batch: Batch,
        _provider_options: dict[str, object],
        context: ProviderContext,
    ) -> Batch:
        run = context.settings["run"]
        assert isinstance(run, str)
        observed.append((run, context.deadline))
        if run == "A":
            started.set()
            assert release.wait(timeout=5)
        return batch

    runtime = Runtime()
    runtime.register_provider(
        "test",
        "same-plan-context",
        "1",
        callback,
        accepts_context=True,
    )
    plan = (
        PipelineBuilder("same-plan-context")
        .external("provider", "test", "same-plan-context", "1", {})
        .compile(runtime)
    )
    batch = Batch.from_array(np.array([1]), backend="test")
    first_deadline = datetime(2030, 1, 1, tzinfo=UTC)
    second_deadline = datetime(2031, 2, 3, 4, 5, 6, 7, tzinfo=UTC)

    async def exercise() -> None:
        first = asyncio.create_task(
            plan.execute_async(
                {"input": batch},
                options=ExecutionOptions({"run": "A"}, first_deadline),
            )
        )
        assert await asyncio.to_thread(started.wait, 5)

        second_submitted = asyncio.Event()

        async def run_second():
            execution = plan.execute_async(
                {"input": batch},
                options=ExecutionOptions({"run": "B"}, second_deadline),
            )
            second_submitted.set()
            return await execution

        second = asyncio.create_task(run_second())
        await second_submitted.wait()
        assert not second.done()
        release.set()
        await asyncio.gather(first, second)

    asyncio.run(exercise())

    assert observed == [("A", first_deadline), ("B", second_deadline)]


def test_same_plan_deadline_expires_while_waiting_without_second_provider_call() -> (
    None
):
    started = threading.Event()
    release = threading.Event()
    observed: list[str] = []

    def callback(
        batch: Batch,
        _provider_options: dict[str, object],
        context: ProviderContext,
    ) -> Batch:
        run = context.settings["run"]
        assert isinstance(run, str)
        observed.append(run)
        if run == "A":
            started.set()
            assert release.wait(timeout=5)
        return batch

    runtime = Runtime()
    runtime.register_provider(
        "test",
        "queued-deadline",
        "1",
        callback,
        accepts_context=True,
    )
    plan = (
        PipelineBuilder("queued-deadline")
        .external("provider", "test", "queued-deadline", "1", {})
        .compile(runtime)
    )
    batch = Batch.from_array(np.array([1]), backend="test")

    async def exercise() -> None:
        first = asyncio.create_task(
            plan.execute_async(
                {"input": batch},
                options=ExecutionOptions({"run": "A"}),
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        deadline = datetime.now(UTC) + timedelta(milliseconds=100)
        second = asyncio.create_task(
            plan.execute_async(
                {"input": batch},
                options=ExecutionOptions({"run": "B"}, deadline),
            )
        )
        await asyncio.sleep(
            max(0.0, (deadline - datetime.now(UTC)).total_seconds()) + 0.02
        )
        assert not second.done()
        release.set()
        await first
        with pytest.raises(CancelledError):
            await second

    asyncio.run(exercise())

    assert observed == ["A"]
    assert plan.snapshot() == {"provider": None}


def _deadline_pipeline(
    started: threading.Event,
    release: threading.Event,
    calls: dict[str, int],
):
    def gate(batch: Batch, _options: dict[str, object]) -> Batch:
        calls["gate"] += 1
        if calls["gate"] == 1:
            started.set()
            assert release.wait(timeout=5)
        return batch

    def downstream(batch: Batch, _options: dict[str, object]) -> Batch:
        calls["downstream"] += 1
        return batch

    runtime = Runtime()
    runtime.register_provider("test", "gate", "1", gate)
    runtime.register_provider("test", "downstream", "1", downstream)
    return (
        PipelineBuilder("deadline-crossing")
        .external("gate", "test", "gate", "1", {})
        .external("downstream", "test", "downstream", "1", {})
        .connect("gate", "downstream")
        .compile(runtime)
    )


def test_sync_deadline_crossed_in_provider_rolls_back_and_recovers() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = {"gate": 0, "downstream": 0}
    plan = _deadline_pipeline(started, release, calls)
    batch = Batch.from_array(np.array([1]), backend="test")
    deadline = datetime.now(UTC) + timedelta(milliseconds=100)

    with ThreadPoolExecutor(max_workers=1) as executor:
        execution = executor.submit(
            plan.execute,
            {"input": batch},
            options=ExecutionOptions(deadline=deadline),
        )
        assert started.wait(timeout=5)
        time.sleep(max(0.0, (deadline - datetime.now(UTC)).total_seconds()) + 0.02)
        release.set()
        with pytest.raises(CancelledError):
            execution.result(timeout=5)

    assert calls == {"gate": 1, "downstream": 0}
    assert plan.snapshot() == {"downstream": None, "gate": None}
    recovered = plan.execute({"input": batch})
    assert recovered.outputs["output"].array.tolist() == [1]
    assert calls == {"gate": 2, "downstream": 1}


def test_async_deadline_crossed_in_provider_rolls_back_and_recovers() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = {"gate": 0, "downstream": 0}
    plan = _deadline_pipeline(started, release, calls)
    batch = Batch.from_array(np.array([1]), backend="test")

    async def exercise() -> None:
        deadline = datetime.now(UTC) + timedelta(milliseconds=100)
        execution = asyncio.create_task(
            plan.execute_async(
                {"input": batch},
                options=ExecutionOptions(deadline=deadline),
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(
            max(0.0, (deadline - datetime.now(UTC)).total_seconds()) + 0.02
        )
        release.set()
        with pytest.raises(CancelledError):
            await execution

        assert calls == {"gate": 1, "downstream": 0}
        assert await plan.snapshot_async() == {"downstream": None, "gate": None}
        recovered = await plan.execute_async({"input": batch})
        assert recovered.outputs["output"].array.tolist() == [1]

    asyncio.run(exercise())
    assert calls == {"gate": 2, "downstream": 1}


def test_provider_error_after_deadline_is_reported_as_cancellation() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def callback(batch: Batch, _options: dict[str, object]) -> Batch:
        del batch
        nonlocal calls
        calls += 1
        started.set()
        assert release.wait(timeout=5)
        raise RuntimeError("provider failure must lose to deadline")

    runtime = Runtime()
    runtime.register_provider("test", "deadline-error", "1", callback)
    plan = (
        PipelineBuilder("deadline-error")
        .external("provider", "test", "deadline-error", "1", {})
        .compile(runtime)
    )
    batch = Batch.from_array(np.array([1]), backend="test")

    async def exercise() -> None:
        deadline = datetime.now(UTC) + timedelta(milliseconds=100)
        execution = asyncio.create_task(
            plan.execute_async(
                {"input": batch},
                options=ExecutionOptions(deadline=deadline),
            )
        )
        assert await asyncio.to_thread(started.wait, 5)
        await asyncio.sleep(
            max(0.0, (deadline - datetime.now(UTC)).total_seconds()) + 0.02
        )
        release.set()
        with pytest.raises(CancelledError):
            await execution

    asyncio.run(exercise())

    assert calls == 1
    assert plan.snapshot() == {"provider": None}


def test_async_task_cancellation_waits_for_cleanup_and_reuses_options() -> None:
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def callback(batch: Batch, _options: dict[str, object]) -> Batch:
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=5)
        return batch

    runtime = Runtime()
    runtime.register_provider("test", "cancel", "1", callback)
    plan = (
        PipelineBuilder("task-cancel")
        .external("provider", "test", "cancel", "1", {})
        .compile(runtime)
    )
    options = ExecutionOptions({"request": 7})

    async def exercise() -> weakref.ReferenceType[np.ndarray]:
        payload = np.array([1])
        payload_ref = weakref.ref(payload)
        batch = Batch.from_array(payload, backend="test")
        execution = asyncio.create_task(
            plan.execute_async({"input": batch}, options=options)
        )
        assert await asyncio.to_thread(started.wait, 5)
        del batch, payload
        execution.cancel()

        async def observe_cancellation(task) -> None:
            await task

        observer = asyncio.create_task(observe_cancellation(execution))
        await asyncio.sleep(0.02)
        assert not observer.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await observer

        recovered = await plan.execute_async(
            {"input": Batch.from_array(np.array([2]), backend="test")},
            options=options,
        )
        assert recovered.outputs["output"].array.tolist() == [2]
        del execution, observer
        return payload_ref

    payload_ref = asyncio.run(exercise())
    gc.collect()

    assert calls == 2
    assert payload_ref() is None
    assert options.settings == {"request": 7}


class _RecordingCancellation:
    def __init__(
        self,
        events: list[str],
        entered: asyncio.Event | None = None,
    ) -> None:
        self.events = events
        self.entered = entered
        self.calls = 0

    def cancel(self) -> None:
        self.events.append("cancel")
        self.calls += 1
        if self.entered is not None:
            self.entered.set()


class _SuppressedCallbackFuture(asyncio.Future[object]):
    def __init__(self, events: list[str]) -> None:
        super().__init__()
        self.events = events

    def add_done_callback(  # type: ignore[override]
        self,
        fn: object,
        *,
        context: object | None = None,
    ) -> None:
        del fn, context

    def done(self) -> bool:
        self.events.append("done")
        return super().done()

    def result(self) -> object:
        self.events.append("result")
        return super().result()


class _CleanupFuture(asyncio.Future[object]):
    def __init__(
        self,
        events: list[str],
        cancellation_entered: asyncio.Event,
        cleanup_checks: asyncio.Queue[None],
    ) -> None:
        super().__init__()
        self.events = events
        self.cancellation_entered = cancellation_entered
        self.cleanup_checks = cleanup_checks

    def done(self) -> bool:
        self.events.append("done")
        if self.cancellation_entered.is_set():
            self.cleanup_checks.put_nowait(None)
        return super().done()

    def result(self) -> object:
        self.events.append("result")
        return super().result()


class _FakeExecutionPlanInner:
    def __init__(
        self,
        future: asyncio.Future[object],
        cancellation: _RecordingCancellation,
        entered: asyncio.Event,
    ) -> None:
        self.future = future
        self.cancellation = cancellation
        self.entered = entered

    def _execute_async_cancellable(
        self,
        _inputs: dict[str, Batch],
        *,
        options: object,
    ) -> tuple[asyncio.Future[object], _RecordingCancellation]:
        del options
        self.entered.set()
        return self.future, self.cancellation


@pytest.mark.parametrize("native_outcome", ["result", "exception"])
def test_native_terminal_state_wins_at_async_cancellation_handler_entry(
    native_outcome: str,
) -> None:
    async def exercise() -> None:
        events: list[str] = []
        entered = asyncio.Event()
        future = _SuppressedCallbackFuture(events)
        cancellation = _RecordingCancellation(events)
        inner = _FakeExecutionPlanInner(future, cancellation, entered)
        plan = calc_flow.pipeline.ExecutionPlan(inner)  # type: ignore[arg-type]
        execution = asyncio.create_task(plan.execute_async({}))
        await entered.wait()
        events.clear()

        def complete_and_cancel() -> None:
            if native_outcome == "result":
                future.set_result("core-result")
            else:
                future.set_exception(RuntimeError("native failure"))
            events.clear()
            assert execution.cancel()

        asyncio.get_running_loop().call_soon(complete_and_cancel)
        if native_outcome == "result":
            assert await execution == "core-result"
        else:
            with pytest.raises(RuntimeError, match="^native failure$"):
                await execution

        assert events[-2:] == ["done", "result"]
        assert events.count("result") == 1
        assert "cancel" not in events
        assert cancellation.calls == 0

    asyncio.run(exercise())


def test_async_cancellation_linearizes_once_and_tolerates_repeated_cancellation() -> (
    None
):
    async def exercise() -> None:
        events: list[str] = []
        entered = asyncio.Event()
        cancellation_entered = asyncio.Event()
        cleanup_checks: asyncio.Queue[None] = asyncio.Queue()
        future = _CleanupFuture(events, cancellation_entered, cleanup_checks)
        cancellation = _RecordingCancellation(events, cancellation_entered)
        inner = _FakeExecutionPlanInner(future, cancellation, entered)
        plan = calc_flow.pipeline.ExecutionPlan(inner)  # type: ignore[arg-type]
        execution = asyncio.create_task(plan.execute_async({}))
        await entered.wait()
        events.clear()

        assert execution.cancel()
        await cancellation_entered.wait()
        await cleanup_checks.get()

        assert execution.cancel()
        assert execution.cancel()
        assert cancellation.calls == 1

        future.set_result("late core result")
        with pytest.raises(asyncio.CancelledError):
            await execution

        cancel_index = events.index("cancel")
        assert "done" in events[:cancel_index]
        assert cancellation.calls == 1
        assert events.count("cancel") == 1

    asyncio.run(exercise())
