"""P3 warm-input and exact-output guards without native timing acquisition."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np
import pyarrow as pa
import pytest

from benchmarks import warm_stream as warm


def _mean_leaf(size: int) -> dict:
    return {
        "kind": "mean",
        "primitive_version": 1,
        "input": "price",
        "frame": {"kind": "rows", "size": size},
        "min_periods": 1,
    }


def _project(config) -> dict:
    output = _mean_leaf(config.window)
    if config.indicator == "dual_sma_spread":
        output = {
            "kind": "difference",
            "primitive_version": 1,
            "left": _mean_leaf(config.fast_window),
            "right": output,
        }
    output["output"] = config.indicator
    return {
        "runtime": {"mode": "stream"},
        "graph": {
            "datafusion": {
                "batch_size": 8192,
                "target_partitions": 1,
                "parallelism_mode": "fixed",
                "enable_rolling_rewrite": True,
            },
            "nodes": [
                {
                    "operator": {
                        "kind": "rolling",
                        "spec": {
                            "configuration_version": 1,
                            "state_layout_version": 1,
                            "partition_by": ["symbol"],
                            "event_time": "event_time",
                            "sequence_by": ["sequence"],
                            "allowed_lateness_micros": 0,
                            "late_policy": {"kind": "error", "scope": "envelope"},
                            "value_policy": "stateful_numeric_v1",
                            "outputs": [output],
                        },
                    }
                }
            ],
        },
    }


@pytest.mark.parametrize("indicator", ("rolling_mean", "dual_sma_spread"))
def test_p3_plan_preflight_checks_effective_profile_frames_and_runtime(
    indicator,
) -> None:
    config = warm.ScenarioConfig(indicator=indicator, window=60, fast_window=20)
    project = _project(config)
    facts = warm._plan_preflight(project, config)
    assert facts["numerical_profile"] == "stable_v1"
    assert facts["windows"] == ([60] if indicator == "rolling_mean" else [20, 60])
    assert facts["min_periods"] == 1
    assert facts["datafusion"] == project["graph"]["datafusion"]
    changed = deepcopy(project)
    changed["graph"]["nodes"][0]["operator"]["spec"]["numerical_profile"] = "stable_v2"
    with pytest.raises(ValueError, match="preflight"):
        warm._plan_preflight(changed, config)
    changed = deepcopy(project)
    changed["graph"]["datafusion"]["target_partitions"] = 4
    with pytest.raises(ValueError, match="preflight"):
        warm._plan_preflight(changed, config)
    changed = deepcopy(project)
    leaf = changed["graph"]["nodes"][0]["operator"]["spec"]["outputs"][0]
    if indicator == "dual_sma_spread":
        leaf = leaf["left"]
    leaf["min_periods"] = 20
    with pytest.raises(ValueError, match="preflight"):
        warm._plan_preflight(changed, config)


def test_p3_zero_late_guard_rejects_missing_and_nonzero_operator_metrics() -> None:
    warm._validate_no_late_rows({"operators": {"rolling": {"late_rows": 0}}})
    for value in (
        {},
        {"operators": {}},
        {"operators": {"rolling": {}}},
        {"operators": {"rolling": {"late_rows": 1}}},
    ):
        with pytest.raises(RuntimeError, match="late rows"):
            warm._validate_no_late_rows(value)


def test_p3_entity_labels_preserve_numeric_order_at_ten_thousand_entities() -> None:
    labels = warm._segment(0, 10_000, 10_000)["symbol"].to_pylist()
    assert labels == sorted(labels)
    assert labels[0] == "S0000"
    assert labels[-1] == "S9999"
    assert warm._segment(0, 2, 64)["symbol"].to_pylist() == ["S000", "S001"]


def test_p3_config_admits_subtick_segments_without_accepting_partial_history() -> None:
    config = warm.ScenarioConfig(
        history_rows=600_000,
        entities=10_000,
        history_segment_rows=1000,
        append_rows=1000,
        append_entities=1000,
        window=60,
        fast_window=20,
        indicator="dual_sma_spread",
    )
    assert config.history_rows // config.entities == 60
    for values in (
        {"history_rows": 600_001},
        {"history_segment_rows": 3000},
        {"append_entities": None},
    ):
        with pytest.raises(ValueError, match="complete entity ticks"):
            replace(config, **values)


def test_p3_sparse_append_layout_keeps_per_row_time_and_first_active_entities() -> None:
    config = warm.ScenarioConfig(
        history_rows=600_000,
        entities=10_000,
        history_segment_rows=10_000,
        append_rows=1000,
        append_entities=1000,
        window=60,
        fast_window=20,
        indicator="dual_sma_spread",
    )
    first = warm._input_table(config, 600_000, 1000)
    following = warm._input_table(config, 601_000, 1000)
    assert first["symbol"].to_pylist() == [f"S{entity:04d}" for entity in range(1000)]
    assert following["symbol"].equals(first["symbol"])
    ticks = first["event_time"].cast(pa.int64()).to_numpy()
    np.testing.assert_array_equal(
        ticks, warm.BASE_MICROS + np.arange(600_000, 601_000) * 1_000_000
    )
    assert following["event_time"][0].value > first["event_time"][-1].value


def _float_table(bits: list[int], validity: int, *, schema=None) -> pa.Table:
    values = pa.py_buffer(np.asarray(bits, dtype="<u8").tobytes())
    array = pa.Array.from_buffers(
        pa.float64(), len(bits), [pa.py_buffer(bytes([validity])), values]
    )
    return pa.Table.from_arrays(
        [array], schema=schema or pa.schema([pa.field("value", pa.float64())])
    )


def test_p3_float_digest_ignores_null_payload_and_chunk_offsets_only() -> None:
    schema = pa.schema(
        [pa.field("value", pa.float64(), metadata={b"z": b"last", b"a": b"first"})],
        metadata={b"z": b"2", b"a": b"1"},
    )
    table = _float_table(
        [99, 0x8000_0000_0000_0000, 0xDEAD_BEEF, 0x7FF8_0000_0000_1234, 88],
        0b11011,
        schema=schema,
    ).slice(1, 3)
    other = _float_table(
        [0x8000_0000_0000_0000, 0, 0x7FF8_0000_0000_1234], 0b101, schema=schema
    )
    chunks = pa.chunked_array(
        [other[0].chunk(0).slice(0, 1), other[0].chunk(0).slice(1, 2)]
    )
    reordered = pa.schema(
        [pa.field("value", pa.float64(), metadata={b"a": b"first", b"z": b"last"})],
        metadata={b"a": b"1", b"z": b"2"},
    )
    chunked = pa.Table.from_arrays([chunks], schema=reordered)
    assert warm._float64_output_digest(table) == warm._float64_output_digest(chunked)


def test_p3_float_digest_preserves_signed_zero_valid_nan_and_validity() -> None:
    digest = warm._float64_output_digest
    assert digest(_float_table([0], 1)) != digest(
        _float_table([0x8000_0000_0000_0000], 1)
    )
    assert digest(_float_table([0x7FF8_0000_0000_0001], 1)) != digest(
        _float_table([0x7FF8_0000_0000_0002], 1)
    )
    assert digest(_float_table([0], 1)) != digest(_float_table([0], 0))
    first = _float_table([0, 0x7FF8_0000_0000_0001], 3)
    running = warm._Float64Digest()
    running.update(first.slice(0, 1))
    running.update(first.slice(1, 1))
    assert running.hexdigest() == digest(first)


def test_p3_value_oracle_also_returns_the_exact_native_float_digest() -> None:
    config = warm.ScenarioConfig(
        history_rows=1000,
        entities=1,
        append_rows=1024,
        append_entities=1,
        history_segment_rows=1000,
        window=60,
        fast_window=20,
        indicator="dual_sma_spread",
    )
    table = warm._input_table(config, 1000, 1024).append_column(
        "dual_sma_spread", pa.array(warm._expected(config, start=1000, rows=1024))
    )
    result = warm._validate_output(table, config, start=1000, rows=1024)
    assert result["passed"]
    assert result["float64_digest"] == warm._float64_output_digest(table)


class _WarmTickSource:
    def __init__(self, outputs: asyncio.Queue) -> None:
        self.outputs = outputs
        self.opened = asyncio.Event()
        self.opened.set()
        self.data = []
        self.watermarks = []
        self.maximum_pending = 0
        self._pending = []

    def _push_data(self, event) -> None:
        if not self._pending and not self.outputs.empty():
            raise RuntimeError("warm tick outputs were not drained")
        self._pending.append(event)
        self.data.append((event.start, event.rows))
        pending_rows = sum(item.rows for item in self._pending)
        self.maximum_pending = max(self.maximum_pending, pending_rows)

    async def _close_tick(self, event) -> None:
        if sum(item.rows for item in self._pending) != 10_000:
            raise RuntimeError("premature history watermark")
        assert not self.watermarks or event.at > self.watermarks[-1]
        self.watermarks.append(event.at)
        first = self._pending[0].start
        for start in range(first, first + 10_000, 500):
            await self.outputs.put(SimpleNamespace(start=start, num_rows=500))
        self._pending.clear()

    async def push(self, event) -> None:
        if event.kind == "data":
            self._push_data(event)
        else:
            await self._close_tick(event)


def test_p3_subtick_warm_sends_one_complete_tick_before_watermark_and_drains_all_chunks(
    tmp_path,
) -> None:
    async def exercise() -> None:
        outputs: asyncio.Queue = asyncio.Queue()
        source = _WarmTickSource(outputs)

        def events(config, start, rows):
            return (
                SimpleNamespace(kind="data", start=start, rows=rows),
                SimpleNamespace(
                    kind="watermark",
                    at=warm.BASE
                    + warm.timedelta(seconds=(start + rows - 1) // config.entities),
                ),
            )

        def validate(table, config, *, start, rows):
            assert table.start == start
            assert table.num_rows == rows
            return {"passed": True, "float64_digest": "d" * 64}

        job = SimpleNamespace(
            status=lambda: {"operators": {"rolling": {"late_rows": 0}}},
            cancel_async=AsyncMock(),
        )
        runner = SimpleNamespace(start_async=AsyncMock(return_value=job))
        updates = []
        digest = SimpleNamespace(update=updates.append, hexdigest=lambda: "d" * 64)
        config = warm.ScenarioConfig(
            history_rows=600_000,
            entities=10_000,
            history_segment_rows=1000,
            append_rows=1000,
            append_entities=1000,
            window=60,
            fast_window=20,
            indicator="dual_sma_spread",
        )
        with (
            patch.object(warm, "_InteractiveSource", return_value=source) as factory,
            patch.object(
                warm, "_QueueSink", return_value=SimpleNamespace(receive=outputs.get)
            ),
            patch.object(warm, "_prepared_events", events),
            patch.object(warm, "_validate_output", validate),
            patch.object(warm, "_Float64Digest", return_value=digest, create=True),
            patch.object(
                warm,
                "_native_program",
                return_value=SimpleNamespace(
                    fingerprint="d" * 64,
                    compile_stream=lambda _runtime: SimpleNamespace(
                        fingerprint="e" * 64
                    ),
                    to_project=lambda _runtime: SimpleNamespace(
                        model_dump=lambda **_kwargs: _project(config)
                    ),
                ),
            ),
            patch.object(warm, "Runtime", return_value=object()),
            patch.object(warm, "SourceBinding", return_value=object()),
            patch.object(
                warm, "SinkBinding", SimpleNamespace(ordinary=lambda *_args: object())
            ),
            patch.object(warm, "ManagedCheckpointRuntime", return_value=object()),
            patch.object(warm, "StreamingRunner", return_value=runner),
        ):
            scenario = await warm.WarmScenario.start(config, tmp_path)
        assert scenario.position == 600_000
        assert scenario.plan_identity["program_fingerprint"] == "d" * 64
        assert scenario.plan_identity["plan_fingerprint"] == "e" * 64
        assert scenario.plan_identity["preflight"]["windows"] == [20, 60]
        assert len(scenario.plan_identity["project_sha256"]) == 64
        assert scenario.warm_output_digest == "d" * 64
        assert len(updates) == 1200
        assert source.data == [(start, 1000) for start in range(0, 600_000, 1000)]
        assert len(source.watermarks) == 60
        assert source.maximum_pending == 10_000
        factory.assert_called_once_with(max_batch_rows=1000)
        assert outputs.empty()
        job.cancel_async.assert_not_awaited()

    asyncio.run(exercise())


def test_p3_sparse_context_uses_full_warm_history_and_only_prior_active_rows() -> None:
    config = warm.ScenarioConfig(
        history_rows=600_000,
        entities=10_000,
        history_segment_rows=10_000,
        append_rows=1000,
        append_entities=1000,
        window=60,
        fast_window=20,
        indicator="dual_sma_spread",
    )
    np.testing.assert_array_equal(
        warm._entity_context(config, 600_999, 999), np.arange(10_999, 591_000, 10_000)
    )
    np.testing.assert_array_equal(
        warm._entity_context(config, 630_000, 0),
        np.concatenate(
            (np.arange(310_000, 590_001, 10_000), np.arange(600_000, 630_000, 1000))
        ),
    )
