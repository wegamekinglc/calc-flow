from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from calc_flow import ConfigError, Runtime

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = (
    ("15_file_source.py", "file"),
    ("16_kafka_source.py", "kafka"),
    ("17_postgresql_source.py", "postgresql"),
    ("18_mysql_source.py", "mysql"),
    ("19_clickhouse_source.py", "clickhouse"),
    ("20_http_source.py", "http"),
    ("21_websocket_source.py", "websocket"),
)


def load_example(filename: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "example", ROOT / "examples" / filename
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("filename", "connector"), EXAMPLES)
def test_source_example_project_compiles_without_opening_service(
    filename: str, connector: str, tmp_path: Path
) -> None:
    example = load_example(filename)
    if connector == "file":
        example.write_input(tmp_path, "csv")
        project = example.build_project(tmp_path, "csv")
    else:
        project = example.build_project(tmp_path)
    runtime = Runtime()
    available = {item.name for item in runtime.capabilities().connectors}
    if connector not in available:
        with pytest.raises(ConfigError, match="missing_connector"):
            runtime.compile_stream_project(project.canonical_json())
    else:
        plan = runtime.compile_stream_project(project.canonical_json())
        assert plan.fingerprint


@pytest.mark.parametrize("format_name", ("csv", "json", "parquet"))
def test_file_source_example_checks_native_results(
    format_name: str, tmp_path: Path
) -> None:
    example = load_example("15_file_source.py")
    asyncio.run(example.run(tmp_path, format_name))
    assert example.read_totals(tmp_path) == [20.0, 60.0]


@pytest.mark.parametrize("bootstrap", (None, "broker.example:19092"))
def test_kafka_entry_point_uses_default_or_configured_broker(
    bootstrap: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = load_example("16_kafka_source.py")
    if bootstrap is None:
        monkeypatch.delenv("CALC_FLOW_KAFKA_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv("CALC_FLOW_KAFKA_BOOTSTRAP", bootstrap)
    runtime = Mock()
    runtime.capabilities.return_value.connectors = [
        SimpleNamespace(name="file"),
        SimpleNamespace(name="kafka"),
    ]
    monkeypatch.setattr(example, "Runtime", Mock(return_value=runtime))
    run = AsyncMock()
    monkeypatch.setattr(example, "run", run)

    example.main()

    run.assert_awaited_once()
    project = example.build_project(run.await_args.args[0])
    assert project.model_dump()["sources"][0]["options"]["bootstrap_servers"] == (
        bootstrap or "127.0.0.1:9092"
    )


@pytest.mark.parametrize(
    "filename", ("16_kafka_source.py", "20_http_source.py", "21_websocket_source.py")
)
def test_live_source_example_cancels_an_idle_job_on_timeout(
    filename: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    example = load_example(filename)
    job = Mock()
    job.status.return_value = {"state": "running", "sinks": {}}
    job.cancel_async = AsyncMock()
    job.shutdown_async = AsyncMock()
    runner = Mock()
    runner.start_async = AsyncMock(return_value=job)
    monkeypatch.setattr(example, "Runtime", Mock())
    monkeypatch.setattr(example, "StreamingRunner", Mock(return_value=runner))

    with pytest.raises(TimeoutError):
        asyncio.run(example.run(tmp_path, timeout=0.01))

    job.cancel_async.assert_awaited_once()
    job.shutdown_async.assert_not_awaited()
