from __future__ import annotations

import asyncio
import inspect
import json
import re
from typing import Any

import numpy as np
import pyarrow as pa
import pytest

from calc_flow import (
    Batch,
    ConfigError,
    PipelineBuilder,
    ProviderOption,
    ProviderOptionsSchema,
    Runtime,
    RuntimeCapabilities,
    register_numpy,
)

_PUBLIC_REGISTRATION_METHODS = (
    "register_mapping_provider",
    "registration_snapshot",
    "capability_registration_snapshot",
)
_MAPPING_PORTS: dict[str, Any] = {
    "input_ports": (("table", "table"), ("weights", "array")),
    "output_ports": (("output", "array"),),
}
_PROVIDER_RECORD_KEYS = frozenset(
    {
        "kind",
        "provider",
        "name",
        "version",
        "callback",
        "options_schema",
        "accepts_context",
        "provider_mode",
        "input_ports",
        "output_ports",
    }
)


def _batch(value: int) -> Batch:
    return Batch.from_pyarrow(pa.table({"value": [value]}))


def _plan(name: str = "stream"):
    return (
        PipelineBuilder(name).expression("calc", "result = value + 1").compile_batch()
    )


def _single_provider(batch: Batch, _options: dict[str, object]) -> Batch:
    return batch


def _context_provider(
    batch: Batch, _options: dict[str, object], _context: object
) -> Batch:
    return batch


def _mapping_provider(
    inputs: dict[str, Batch], _options: dict[str, object]
) -> dict[str, Batch]:
    return {"output": inputs["weights"]}


def _scalar_identity(value: pa.Array) -> pa.Array:
    return value


def _register_scalar_identity(runtime: Runtime) -> None:
    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=["int64"],
        return_type="int64",
        volatility="immutable",
        function=_scalar_identity,
    )


def _assert_capabilities_describe_records(
    capabilities: RuntimeCapabilities, records: tuple[dict[str, Any], ...]
) -> None:
    for described, kind in (
        (capabilities.providers, "provider"),
        (capabilities.udfs, "scalar_udf"),
    ):
        assert sorted(
            (item.provider, item.name, item.version) for item in described
        ) == sorted(
            (record["provider"], record["name"], record["version"])
            for record in records
            if record["kind"] == kind
        )


def _replayed(records: tuple[dict[str, Any], ...]) -> Runtime:
    """Rebuild a runtime from the listed record keys through public methods only."""
    runtime = Runtime()
    for record in records:
        kind = record["kind"]
        if kind == "scalar_udf":
            runtime.register_scalar_udf(
                provider=record["provider"],
                name=record["name"],
                version=record["version"],
                input_types=record["input_types"],
                return_type=record["return_type"],
                volatility=record["volatility"],
                function=record["function"],
            )
            continue
        assert kind == "provider", f"cannot replay a {kind} registration"
        identity = (record["provider"], record["name"], record["version"])
        options = {
            "options_schema": record["options_schema"],
            "accepts_context": record.get("accepts_context", False),
        }
        if record.get("provider_mode") == "mapping":
            runtime.register_mapping_provider(
                *identity,
                record["callback"],
                input_ports=record["input_ports"],
                output_ports=record["output_ports"],
                **options,
            )
        else:
            runtime.register_provider(*identity, record["callback"], **options)
    return runtime


def test_kafka_decoder_registration_validates_inputs() -> None:
    runtime = Runtime()
    with pytest.raises(TypeError, match="callable"):
        runtime.register_kafka_decoder(name="pipe", version="1", function=42)
    with pytest.raises(ValueError, match="name"):
        runtime.register_kafka_decoder(name="", version="1", function=lambda p: p)
    with pytest.raises(ValueError, match="version"):
        runtime.register_kafka_decoder(name="pipe", version="", function=lambda p: p)


def test_kafka_decoder_registration_follows_the_compiled_connectors() -> None:
    runtime = Runtime()
    available = {item.name for item in runtime.capabilities().connectors}

    def decode(payload: bytes) -> object:
        return payload

    if "kafka" not in available:
        with pytest.raises(RuntimeError, match="connector-kafka"):
            runtime.register_kafka_decoder(
                name="pipe-orders", version="1", function=decode
            )
        return
    runtime.register_kafka_decoder(name="pipe-orders", version="1", function=decode)
    with pytest.raises(ConfigError):
        runtime.register_kafka_decoder(name="pipe-orders", version="1", function=decode)
    assert any(
        registration["kind"] == "kafka_decoder"
        and registration["name"] == "pipe-orders"
        for registration in runtime.registration_snapshot()
    )


def test_registration_snapshot_is_success_only_and_defensive() -> None:
    runtime = Runtime()

    def provider(batch: Batch, _options: dict[str, object]) -> Batch:
        return batch

    def identity(value: pa.Array) -> pa.Array:
        return value

    runtime.register_provider("test", "identity", "1", provider)
    runtime.register_scalar_udf(
        provider="python",
        name="identity",
        version="1",
        input_types=("int64",),
        return_type="int64",
        volatility="immutable",
        function=identity,
    )
    snapshot = runtime.registration_snapshot()
    assert [(record["kind"], record["name"]) for record in snapshot] == [
        ("provider", "identity"),
        ("scalar_udf", "identity"),
    ]
    assert snapshot[0] == {
        "kind": "provider",
        "provider": "test",
        "name": "identity",
        "version": "1",
        "callback": provider,
        "options_schema": None,
    }
    snapshot[0]["name"] = "mutated"
    assert runtime.registration_snapshot()[0]["name"] == "identity"

    with pytest.raises(ConfigError, match="duplicate"):
        runtime.register_provider("test", "identity", "1", provider)
    with pytest.raises(ConfigError, match="duplicate"):
        runtime.register_scalar_udf(
            provider="python",
            name="identity",
            version="1",
            input_types=("int64",),
            return_type="int64",
            volatility="immutable",
            function=identity,
        )
    assert len(runtime.registration_snapshot()) == 2


def test_mapping_provider_executes_mixed_named_inputs() -> None:
    runtime = Runtime()

    def callback(
        inputs: dict[str, Batch],
        options: dict[str, object],
    ) -> dict[str, Batch]:
        assert sorted(inputs) == ["table", "weights"]
        assert options == {"columns": ["value"]}
        assert (
            inputs["table"].to_pyarrow()["value"].chunk(0).buffers()[1].address
            == table.to_pyarrow()["value"].chunk(0).buffers()[1].address
        )
        assert inputs["weights"].array is weights.array
        return {"output": inputs["weights"]}

    runtime.register_mapping_provider(
        "test",
        "table_matmul",
        "1",
        callback,
        input_ports=(("table", "table"), ("weights", "array")),
        output_ports=(("output", "array"),),
    )
    project = (
        PipelineBuilder("mapping")
        .table_matmul("multiply", backend="numpy", columns=("value",))
        .project
    )
    project["graph"]["nodes"][0]["operator"]["provider"] = "test"
    plan = runtime.compile_project(json.dumps(project))
    table = _batch(3)
    weights = Batch.from_array(np.array([[2.0]]), backend="numpy")

    result = plan.execute({"table": table, "weights": weights})

    assert result.outputs["output"].array.tolist() == [[2.0]]


def test_mapping_provider_registration_validates_arguments() -> None:
    runtime = Runtime()

    with pytest.raises(TypeError, match="provider callback must be callable"):
        runtime.register_mapping_provider(
            "test",
            "mapping",
            "1",
            object(),
            input_ports=(("input", "array"),),
            output_ports=(("output", "array"),),
        )
    with pytest.raises(
        ValueError,
        match="mapping provider port kind must be 'table' or 'array'",
    ):
        runtime.register_mapping_provider(
            "test",
            "mapping",
            "1",
            lambda inputs, options: inputs,
            input_ports=(("input", "scalar"),),
            output_ports=(("output", "array"),),
        )


def test_mapping_registration_snapshot_preserves_copied_contracts() -> None:
    runtime = Runtime()
    input_ports = [("table", "table"), ("weights", "array")]
    output_ports = [("output", "array")]

    runtime.register_mapping_provider(
        "test",
        "mapping",
        "1",
        lambda inputs, options: {"output": inputs["weights"]},
        input_ports=input_ports,
        output_ports=output_ports,
    )
    input_ports.clear()
    output_ports.clear()

    snapshot = runtime.registration_snapshot()

    assert snapshot[0]["provider_mode"] == "mapping"
    assert snapshot[0]["input_ports"] == (
        ("table", "table"),
        ("weights", "array"),
    )
    assert snapshot[0]["output_ports"] == (("output", "array"),)
    snapshot[0]["input_ports"] = ()
    snapshot[0]["output_ports"] = ()
    assert runtime.registration_snapshot()[0]["input_ports"] == (
        ("table", "table"),
        ("weights", "array"),
    )
    assert runtime.registration_snapshot()[0]["output_ports"] == (("output", "array"),)


def test_register_mapping_provider_has_the_published_signature() -> None:
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword_only = inspect.Parameter.KEYWORD_ONLY
    required = inspect.Parameter.empty

    parameters = inspect.signature(Runtime.register_mapping_provider).parameters

    assert [
        (parameter.name, parameter.kind, parameter.default)
        for parameter in parameters.values()
    ] == [
        ("self", positional, required),
        ("provider", positional, required),
        ("name", positional, required),
        ("version", positional, required),
        ("callback", positional, required),
        ("input_ports", keyword_only, required),
        ("output_ports", keyword_only, required),
        ("options_schema", keyword_only, None),
        ("accepts_context", keyword_only, False),
    ]
    assert parameters["accepts_context"].default is False
    assert (
        Runtime().register_mapping_provider(
            "test", "mapping", "1", _mapping_provider, **_MAPPING_PORTS
        )
        is None
    )


@pytest.mark.parametrize("method", _PUBLIC_REGISTRATION_METHODS)
def test_public_registration_methods_are_documented(method: str) -> None:
    assert inspect.getdoc(getattr(Runtime, method))


@pytest.mark.parametrize("method", _PUBLIC_REGISTRATION_METHODS)
def test_runtime_keeps_no_underscore_registration_aliases(method: str) -> None:
    assert not hasattr(Runtime, f"_{method}")
    assert not hasattr(Runtime(), f"_{method}")


def test_registration_records_match_the_published_contract() -> None:
    # Whole-record equality pins today's keys as a regression check; the
    # published contract guarantees only the listed keys.
    runtime = Runtime()
    runtime.register_provider("test", "single", "1", _single_provider)
    runtime.register_provider(
        "test", "contextual", "1", _context_provider, accepts_context=True
    )
    runtime.register_mapping_provider(
        "test",
        "mapping",
        "1",
        _mapping_provider,
        input_ports=[["table", "table"], ["weights", "array"]],
        output_ports=[("output", "array")],
    )
    _register_scalar_identity(runtime)

    single, contextual, mapping, scalar = runtime.registration_snapshot()

    assert single == {
        "kind": "provider",
        "provider": "test",
        "name": "single",
        "version": "1",
        "callback": _single_provider,
        "options_schema": None,
    }
    assert contextual == {
        "kind": "provider",
        "provider": "test",
        "name": "contextual",
        "version": "1",
        "callback": _context_provider,
        "options_schema": None,
        "accepts_context": True,
    }
    assert contextual["accepts_context"] is True
    assert mapping == {
        "kind": "provider",
        "provider": "test",
        "name": "mapping",
        "version": "1",
        "callback": _mapping_provider,
        "options_schema": None,
        "provider_mode": "mapping",
        "input_ports": (("table", "table"), ("weights", "array")),
        "output_ports": (("output", "array"),),
    }
    assert scalar == {
        "kind": "scalar_udf",
        "provider": "python",
        "name": "identity",
        "version": "1",
        "input_types": ("int64",),
        "return_type": "int64",
        "volatility": "immutable",
        "function": _scalar_identity,
    }


def test_kafka_decoder_record_matches_the_published_contract() -> None:
    runtime = Runtime()
    if "kafka" not in {item.name for item in runtime.capabilities().connectors}:
        pytest.skip("requires connector-kafka")

    def decode(payload: bytes) -> object:
        return payload

    runtime.register_kafka_decoder(name="pipe-orders", version="1", function=decode)

    (record,) = runtime.registration_snapshot()
    assert record == {
        "kind": "kafka_decoder",
        "name": "pipe-orders",
        "version": "1",
        "function": decode,
    }
    assert "provider" not in record


def test_registration_snapshots_are_fresh_records_sharing_registered_values() -> None:
    runtime = Runtime()
    schema = ProviderOptionsSchema(
        fields=(ProviderOption("expression", "string", required=True),)
    )
    runtime.register_provider(
        "test", "single", "1", _single_provider, options_schema=schema
    )
    _register_scalar_identity(runtime)

    first = runtime.registration_snapshot()
    second = runtime.registration_snapshot()

    assert first == second
    assert first is not second
    assert all(left is not right for left, right in zip(first, second))
    for provider_record, udf_record in (first, second):
        assert provider_record["callback"] is _single_provider
        assert provider_record["options_schema"] is schema
        assert udf_record["function"] is _scalar_identity

    first[0]["name"] = "mutated"
    first[0]["unlisted"] = object()
    del first[1]["function"]

    assert runtime.registration_snapshot() == second


@pytest.mark.parametrize(
    ("arguments", "keywords", "error", "message"),
    [
        pytest.param(
            ("test", "candidate", "1", object()),
            _MAPPING_PORTS,
            TypeError,
            "provider callback must be callable",
            id="callback-not-callable",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider),
            {**_MAPPING_PORTS, "input_ports": (("table", "scalar"),)},
            ValueError,
            "mapping provider port kind must be 'table' or 'array'",
            id="unknown-port-kind",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider),
            {**_MAPPING_PORTS, "input_ports": (("table",),)},
            ValueError,
            "not enough values to unpack",
            id="port-entry-not-a-pair",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider),
            {**_MAPPING_PORTS, "input_ports": ((1, "table"),)},
            TypeError,
            "argument 'input_ports'",
            id="port-name-not-a-string",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider),
            {**_MAPPING_PORTS, "accepts_context": 1},
            TypeError,
            "accepts_context must be an exact bool; found int",
            id="accepts-context-not-a-bool",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider),
            {**_MAPPING_PORTS, "options_schema": {}},
            TypeError,
            "options_schema must be a ProviderOptionsSchema or None; found dict",
            id="options-schema-of-another-type",
        ),
        pytest.param(
            ("test", "registered", "1", _mapping_provider),
            _MAPPING_PORTS,
            ConfigError,
            "invalid provider: duplicate provider/name/version",
            id="duplicate-identity",
        ),
        pytest.param(
            ("test", "candidate", "1", _mapping_provider, *_MAPPING_PORTS.values()),
            {},
            TypeError,
            "positional argument",
            id="ports-passed-positionally",
        ),
    ],
)
def test_failed_mapping_registration_leaves_no_record_and_keeps_the_revision(
    arguments: tuple[object, ...],
    keywords: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    runtime = Runtime()
    runtime.register_mapping_provider(
        "test", "registered", "1", _mapping_provider, **_MAPPING_PORTS
    )
    before = runtime.registration_snapshot()
    revision = runtime.capabilities().scope.revision

    with pytest.raises(error, match=re.escape(message)):
        runtime.register_mapping_provider(*arguments, **keywords)

    assert runtime.registration_snapshot() == before
    assert runtime.capabilities().scope.revision == revision


def test_registration_records_replay_onto_a_fresh_runtime() -> None:
    source = Runtime()
    source.register_provider(
        "test",
        "contextual",
        "1",
        _context_provider,
        options_schema=ProviderOptionsSchema(
            fields=(ProviderOption("expression", "string", required=True),)
        ),
        accepts_context=True,
    )
    source.register_mapping_provider(
        "test", "mapping", "1", _mapping_provider, **_MAPPING_PORTS
    )
    _register_scalar_identity(source)
    records = source.registration_snapshot()

    assert [record["kind"] for record in records] == [
        "provider",
        "provider",
        "scalar_udf",
    ]

    worker = _replayed(records)

    assert worker.registration_snapshot() == records
    assert worker.capabilities().providers == source.capabilities().providers
    assert worker.capabilities().udfs == source.capabilities().udfs


def test_capability_registration_snapshot_pairs_capabilities_with_records() -> None:
    runtime = Runtime()

    empty_capabilities, empty_records = runtime.capability_registration_snapshot()

    assert empty_records == ()
    assert empty_capabilities.providers == ()
    assert empty_capabilities.udfs == ()

    runtime.register_provider("test", "single", "1", _single_provider)
    runtime.register_mapping_provider(
        "test", "mapping", "1", _mapping_provider, **_MAPPING_PORTS
    )
    earlier_capabilities, earlier_records = runtime.capability_registration_snapshot()
    _register_scalar_identity(runtime)

    capabilities, records = runtime.capability_registration_snapshot()

    assert [record["name"] for record in records] == ["single", "mapping", "identity"]
    _assert_capabilities_describe_records(capabilities, records)
    assert (capabilities, records) == (
        runtime.capabilities(),
        runtime.registration_snapshot(),
    )
    assert capabilities.scope.session_id == earlier_capabilities.scope.session_id
    assert capabilities.scope.revision > earlier_capabilities.scope.revision
    assert [record["name"] for record in earlier_records] == ["single", "mapping"]
    assert earlier_capabilities.udfs == ()
    _assert_capabilities_describe_records(earlier_capabilities, earlier_records)


def test_register_numpy_records_stay_within_the_published_contract() -> None:
    runtime = Runtime()
    register_numpy(runtime)

    records = runtime.registration_snapshot()

    assert [
        (record["kind"], record["provider"], record["name"], record["version"])
        for record in records
    ] == [
        ("provider", "numpy", "expression", "1"),
        ("provider", "numpy", "table_matmul", "1"),
        ("provider", "numpy", "symbolic_matrix", "1"),
    ]
    unlisted = [set(record) - _PROVIDER_RECORD_KEYS for record in records]
    assert unlisted[1] == set()
    assert unlisted[0] <= {"stream_lifecycle"}
    assert unlisted[2] <= {"stream_lifecycle"}


def test_replayed_register_numpy_records_are_batch_only() -> None:
    def listed(record: dict[str, Any]) -> dict[str, Any]:
        return {key: record[key] for key in record if key in _PROVIDER_RECORD_KEYS}

    def modes(runtime: Runtime) -> dict[str, tuple[str, ...]]:
        return {item.name: item.modes for item in runtime.capabilities().providers}

    source = Runtime()
    register_numpy(source)
    records = source.registration_snapshot()

    worker = _replayed(records)

    replayed = worker.registration_snapshot()
    assert [listed(record) for record in replayed] == [
        listed(record) for record in records
    ]
    assert all("stream_lifecycle" not in record for record in replayed)
    for name in ("expression", "symbolic_matrix"):
        assert modes(source)[name] == ("batch", "stream")
        assert modes(worker)[name] == ("batch",)


def test_execution_plan_lifecycle_is_async_defensive_and_guarded() -> None:
    plan = _plan("lifecycle")

    async def exercise() -> None:
        state = await plan.snapshot_async()
        state["calc"] = {"changed": True}
        assert await plan.snapshot_async() == {"calc": None}
        await plan.restore_async({"calc": None})
        await plan.reset_async()
        with pytest.raises(RuntimeError, match="await snapshot_async"):
            plan.snapshot()

    asyncio.run(exercise())
    assert plan.snapshot() == {"calc": None}
    plan.restore({"calc": None})
    plan.reset()


def test_stream_join_status_typeddict_matches_the_native_mapping_value() -> None:
    """The node ID is the outer mapping key, not a value field (api note)."""
    from calc_flow.runtime import StreamJoinSideStatus, StreamJoinStatus

    assert set(StreamJoinStatus.__annotations__) == {
        "left",
        "right",
        "emitted_match_rows",
        "state_limit_failures",
        "match_limit_failures",
    }
    assert set(StreamJoinSideStatus.__annotations__) == {
        "retained_rows",
        "retained_bytes",
        "evicted_rows",
        "late_rows",
        "late_affected_batches",
        "max_lateness_micros",
        "null_event_time_rows",
        "null_key_rows",
        "watermark_micros",
        "idle",
        "ended",
    }
