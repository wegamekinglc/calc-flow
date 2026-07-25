from __future__ import annotations

import asyncio
import gc
import json
import math
import threading
import time
import weakref
from collections import UserDict
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

        def keys(self):
            self.copies += 1
            return super().keys()

    source = CountingMapping({"nested": [None, False, 1, 2**64 - 1, 1.25, "x"]})
    options = ExecutionOptions(source)

    assert source.copies == 1
    assert options.settings == {"nested": [None, False, 1, 2**64 - 1, 1.25, "x"]}


@pytest.mark.parametrize(
    "settings",
    [
        None,
        [],
        (),
        "value",
        1,
    ],
)
def test_execution_options_rejects_non_mapping_settings(settings: object) -> None:
    with pytest.raises(TypeError, match="execution settings"):
        ExecutionOptions(settings)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value",
    [
        (1,),
        {1},
        b"bytes",
        object(),
        datetime.now(UTC),
        UserDict({"nested": True}),
        math.nan,
        math.inf,
        -math.inf,
        -(2**63) - 1,
        2**64,
    ],
)
def test_execution_options_rejects_non_strict_json_values(value: object) -> None:
    with pytest.raises(ValueError, match="execution settings"):
        ExecutionOptions({"value": value})  # type: ignore[dict-item]


def test_execution_options_rejects_non_string_and_subclassed_keys() -> None:
    class StringSubclass(str):
        pass

    for key in (1, StringSubclass("value")):
        with pytest.raises(ValueError, match="execution settings"):
            ExecutionOptions({key: True})  # type: ignore[dict-item]


@pytest.mark.parametrize(
    "value",
    [
        type("IntSubclass", (int,), {})(1),
        type("FloatSubclass", (float,), {})(1.0),
        type("StringSubclass", (str,), {})("value"),
        type("ListSubclass", (list,), {})([1]),
        type("DictSubclass", (dict,), {})({"value": 1}),
    ],
)
def test_execution_options_rejects_json_type_subclasses(value: object) -> None:
    with pytest.raises(ValueError, match="execution settings"):
        ExecutionOptions({"value": value})  # type: ignore[dict-item]


def test_execution_options_enforces_depth_and_cycles_without_rejecting_aliases() -> (
    None
):
    ExecutionOptions(_nested_settings(32))
    with pytest.raises(ValueError, match="execution settings"):
        ExecutionOptions(_nested_settings(33))

    cycle: list[object] = []
    cycle.append(cycle)
    with pytest.raises(ValueError, match="execution settings"):
        ExecutionOptions({"value": cycle})

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


def test_execution_options_normalizes_utc_deadline_and_preserves_microseconds() -> None:
    deadline = datetime(
        2027,
        4,
        5,
        6,
        7,
        8,
        654321,
        tzinfo=timezone(timedelta(0), "ZERO"),
    )

    options = ExecutionOptions(deadline=deadline)

    assert options.deadline == datetime(2027, 4, 5, 6, 7, 8, 654321, tzinfo=UTC)
    assert options.deadline is not deadline
    assert options.deadline is not None
    assert options.deadline.tzinfo is UTC


@pytest.mark.parametrize(
    "deadline",
    [
        datetime(2027, 4, 5),
        datetime(2027, 4, 5, tzinfo=timezone(timedelta(hours=1))),
        datetime(2027, 4, 5, tzinfo=_NoneOffset()),
        datetime(2027, 4, 5, tzinfo=_BadOffset()),
        datetime(2027, 4, 5, tzinfo=_RaisingOffset()),
    ],
)
def test_execution_options_rejects_invalid_deadline_timezones(
    deadline: datetime,
) -> None:
    with pytest.raises(ValueError, match="deadline"):
        ExecutionOptions(deadline=deadline)


def test_execution_options_rejects_non_datetime_deadline() -> None:
    with pytest.raises(TypeError, match="deadline"):
        ExecutionOptions(deadline="later")  # type: ignore[arg-type]


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
