from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from calc_flow.context import CancellationToken, RunCancelledError, RunContext
from calc_flow.engine.datafusion import DataFusionRuntime


def test_run_context_shares_run_services_with_node_context() -> None:
    with DataFusionRuntime() as runtime:
        context = RunContext.create(runtime, settings={"project": "demo"})
        node_context = context.for_node("calculate")

        assert node_context.run_id == context.run_id
        assert node_context.node_id == "calculate"
        assert node_context.datafusion is runtime
        assert node_context.settings["project"] == "demo"


def test_run_context_settings_are_read_only() -> None:
    with DataFusionRuntime() as runtime:
        context = RunContext.create(runtime, settings={"x": 1})

        with pytest.raises(TypeError):
            context.settings["x"] = 2  # type: ignore[index]


def test_run_context_observes_cancellation() -> None:
    token = CancellationToken()
    token.cancel()
    with DataFusionRuntime() as runtime:
        context = RunContext.create(runtime, cancellation=token)

        with pytest.raises(RunCancelledError, match="cancelled"):
            context.check_cancelled()


def test_run_context_observes_deadline() -> None:
    with DataFusionRuntime() as runtime:
        context = RunContext.create(
            runtime, deadline=datetime.now(UTC) - timedelta(seconds=1)
        )

        with pytest.raises(RunCancelledError, match="deadline"):
            context.check_cancelled()


def test_run_context_rejects_naive_deadline() -> None:
    with (
        DataFusionRuntime() as runtime,
        pytest.raises(ValueError, match="timezone"),
    ):
        RunContext.create(runtime, deadline=datetime.now())


def test_run_context_deeply_freezes_settings() -> None:
    with DataFusionRuntime() as runtime:
        context = RunContext.create(
            runtime,
            settings={"nested": {"values": [1, 2]}},
        )

        assert context.settings["nested"]["values"] == (1, 2)
        with pytest.raises(TypeError):
            context.settings["nested"]["new"] = True  # type: ignore[index]


def test_run_context_rejects_non_string_setting_keys() -> None:
    with (
        DataFusionRuntime() as runtime,
        pytest.raises(TypeError, match="keys must be strings"),
    ):
        RunContext.create(runtime, settings={1: "invalid"})  # type: ignore[dict-item]
