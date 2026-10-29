"""Compilation preparation is cached without sharing owning stream plans."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pyarrow as pa
import pytest

import calc_flow as cf
import calc_flow.pipeline as pipeline
import calc_flow.symbolic.lower.program as lowering


def _program(engine: str, name: str = "cached") -> cf.Program:
    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    return cf.Program(
        name,
        engine=engine,
        outputs={"result": source.select(incremented=source["value"] + 1)},
    )


def _compile(program: cf.Program, runtime: cf.Runtime):
    return (
        program.compile_batch(runtime)
        if program.engine == "sql"
        else program.compile_stream(runtime)
    )


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_warm_compile_skips_analysis_and_capability_reconstruction(monkeypatch, engine):
    program = _program(engine)
    runtime = cf.Runtime()
    first = _compile(program, runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm compilation repeated declaration preparation")

    monkeypatch.setattr(lowering, "_run", forbidden)
    monkeypatch.setattr(cf.Runtime, "capabilities", forbidden)
    second = _compile(program, runtime)
    if engine == "sql":
        assert second is first
    else:
        assert second is not first
        assert second.fingerprint == first.fingerprint


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_cold_compile_uses_one_capability_snapshot(monkeypatch, engine):
    snapshots = []
    original = cf.Runtime.capabilities

    def counted(runtime):
        snapshot = original(runtime)
        snapshots.append(snapshot)
        return snapshot

    monkeypatch.setattr(cf.Runtime, "capabilities", counted)
    _compile(_program(engine), cf.Runtime())
    assert len(snapshots) == 1


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_registration_invalidates_prepared_compilation(monkeypatch, engine):
    calls = []
    original = lowering._run

    def counted(*args, **kwargs):
        value = original(*args, **kwargs)
        calls.append(value[1].scope.revision)
        return value

    monkeypatch.setattr(lowering, "_run", counted)
    runtime = cf.Runtime()
    program = _program(engine)
    first = _compile(program, runtime)
    runtime.register_provider("test", "unused", "1", lambda _inputs, _options: {})
    second = _compile(program, runtime)
    _compile(program, runtime)
    assert calls == [0, 1]
    assert first is not second


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_failed_native_compilation_does_not_cache_preparation(monkeypatch, engine):
    method = (
        "compile_batch_project" if engine == "sql" else "_compile_stream_graph_project"
    )
    original = getattr(cf.Runtime, method)
    failures = []

    def fail_once(runtime, *args, **kwargs):
        if not failures:
            failures.append(True)
            raise ValueError("native compilation rejected")
        return original(runtime, *args, **kwargs)

    monkeypatch.setattr(cf.Runtime, method, fail_once)
    program = _program(engine)
    runtime = cf.Runtime()
    with pytest.raises(ValueError, match="native compilation rejected"):
        _compile(program, runtime)
    assert not runtime._symbolic_compile_cache
    assert _compile(program, runtime).name == program.name


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_preparation_cache_is_bounded_and_evicted_program_recompiles(
    monkeypatch, engine
):
    monkeypatch.setattr(pipeline, "_SYMBOLIC_COMPILE_CACHE_MAX_ENTRIES", 2)
    calls = []
    original = lowering._run

    def counted(program, *args, **kwargs):
        calls.append(program.name)
        return original(program, *args, **kwargs)

    monkeypatch.setattr(lowering, "_run", counted)
    runtime = cf.Runtime()
    for name in ("first", "second", "third", "first"):
        _compile(_program(engine, name), runtime)
    assert calls == ["first", "second", "third", "first"]
    assert len(runtime._symbolic_compile_cache) == 2


@pytest.mark.parametrize("engine", ["sql", "streaming"])
def test_registry_is_locked_across_analysis_and_native_compilation(monkeypatch, engine):
    original = lowering._run

    def analyzed(program, runtime, mode):
        def can_register():
            acquired = runtime._registration_lock.acquire(blocking=False)
            if acquired:
                runtime._registration_lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(can_register).result(timeout=5) is False
        return original(program, runtime, mode)

    monkeypatch.setattr(lowering, "_run", analyzed)
    _compile(_program(engine), cf.Runtime())


def test_same_declaration_keeps_batch_and_stream_plan_ownership_separate():
    batch = _program("sql")
    stream = _program("streaming")
    assert batch.fingerprint == stream.fingerprint
    runtime = cf.Runtime()
    assert isinstance(batch.compile_batch(runtime), cf.BatchExecutionPlan)
    first = stream.compile_stream(runtime)
    assert isinstance(first, cf.StreamExecutionPlan)
    assert stream.compile_stream(runtime) is not first


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize(
    "arguments, exception, message",
    [
        ({"allowed_lateness_micros": True}, TypeError, "exact int"),
        ({"allowed_lateness_micros": -1}, ValueError, "non-negative"),
        ({"late_policy": "side_output"}, ValueError, "late_policy"),
    ],
)
def test_compile_stream_validates_row_local_lateness_before_cache_lookup(
    warm, arguments, exception, message
):
    program = _program("streaming")
    runtime = cf.Runtime()
    if warm:
        program.compile_stream(runtime)
    with pytest.raises(exception, match=message):
        program.compile_stream(runtime, **arguments)


def test_public_lowering_keeps_graph_error_before_invalid_lateness():
    source = cf.table_input("events", schema=pa.schema([("value", pa.int64())]))
    program = cf.Program(
        "invalid", engine="streaming", outputs={"result": source.select("missing")}
    )
    with pytest.raises(cf.CompileError, match="missing"):
        lowering.lower_program_document(
            program, cf.Runtime(), "stream", allowed_lateness_micros=-1
        )


def test_stream_preparation_distinguishes_lateness_options(monkeypatch):
    calls = []
    original = lowering._run

    def counted(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(lowering, "_run", counted)
    program = _program("streaming")
    runtime = cf.Runtime()
    for lateness in (0, 1, 0):
        program.compile_stream(runtime, allowed_lateness_micros=lateness)
    assert len(calls) == 2


def test_collect_keeps_lowering_revision_locked_until_native_plan_is_ready(monkeypatch):
    original = cf.Runtime.compile_batch_project

    def compile_project(runtime, project):
        def can_register():
            acquired = runtime._registration_lock.acquire(blocking=False)
            if acquired:
                runtime._registration_lock.release()
            return acquired

        with ThreadPoolExecutor(max_workers=1) as executor:
            assert executor.submit(can_register).result(timeout=5) is False
        return original(runtime, project)

    monkeypatch.setattr(cf.Runtime, "compile_batch_project", compile_project)
    program = _program("sql")
    result = program.collect(
        {"events": pa.table({"value": [1, 2]})}, runtime=cf.Runtime()
    )
    assert result["result"].to_pydict() == {"incremented": [2, 3]}


@pytest.mark.parametrize(
    "engine, warm", [("sql", False), ("streaming", False), ("streaming", True)]
)
@pytest.mark.parametrize("mutation", ["provider", "lifecycle"])
def test_reentrant_provider_registration_does_not_repopulate_old_preparation(
    monkeypatch, engine, mutation, warm
):
    import calc_flow.array as arrays

    runtime = cf.Runtime()
    cf.register_numpy(runtime)

    def callback(batch, options):
        return batch

    if mutation == "lifecycle":
        runtime.register_provider("test", "during_compile", "1", callback)
    source = cf.table_input("events", schema=pa.schema([("value", pa.float64())]))
    weights = cf.parameter(
        "weights", kind="array", backend="numpy", dtype="float64", shape=(1, 1)
    )
    output = cf.table.attach_columns(
        source,
        cf.linalg.matmul(
            cf.linalg.from_columns(source, columns=("value",), backend="numpy"), weights
        ),
        names=("score",),
    )
    program = cf.Program("reentrant", engine=engine, outputs={"result": output})
    hook = "validate" if engine == "sql" else "validate_stream"
    original_validate = getattr(arrays._SymbolicMatrixProvider, hook)
    original_run = lowering._run
    preparations = []
    mutations = []
    armed = [not warm]

    def analyzed(*args, **kwargs):
        preparations.append(True)
        return original_run(*args, **kwargs)

    def validate(provider, options):
        original_validate(provider, options)
        if mutations or not armed[0]:
            return
        mutations.append(True)
        if mutation == "provider":
            runtime.register_provider("test", "during_compile", "1", callback)
        else:
            runtime._register_stateless_stream_provider(
                "test",
                "during_compile",
                "1",
                callback,
                microbatch_invariant=True,
                deterministic=True,
                replay_safe=True,
                supports_static_inputs=False,
                array_rules=arrays._NUMPY_STREAM_ARRAY_RULES,
            )

    monkeypatch.setattr(lowering, "_run", analyzed)
    monkeypatch.setattr(arrays._SymbolicMatrixProvider, hook, validate)
    if warm:
        _compile(program, runtime)
        armed[0] = True
    first = _compile(program, runtime)
    second = _compile(program, runtime)
    third = _compile(program, runtime)
    assert len(preparations) == 2
    assert mutations == [True]
    if engine == "sql":
        assert first is not second
        assert third is second


def test_collect_does_not_repopulate_template_invalidated_during_lowering(monkeypatch):
    import importlib

    compute = importlib.import_module("calc_flow.compute")
    runtime = cf.Runtime()
    original = compute._lower_collect
    preparations = []

    def lowered(program, selected):
        value = original(program, selected)
        preparations.append(True)
        if len(preparations) == 1:
            selected.register_provider(
                "test", "during_lower", "1", lambda batch, opts: batch
            )
        return value

    monkeypatch.setattr(compute, "_lower_collect", lowered)
    program = _program("sql")
    for _ in range(3):
        result = program.collect({"events": pa.table({"value": [1]})}, runtime=runtime)
        assert result["result"].to_pydict() == {"incremented": [2]}
    assert len(preparations) == 2
