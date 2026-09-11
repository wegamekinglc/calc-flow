"""Named performance diagnostics, kept separate from the engine comparison catalog."""

from __future__ import annotations

import asyncio
import hashlib
import io
import time
from pathlib import Path


def save_output(table, path: Path, *, ordered: bool) -> str:
    import pyarrow as pa

    if not ordered:
        key = "sequence" if "sequence" in table.column_names else "symbol"
        table = table.sort_by(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with (
        pa.OSFile(str(path), "wb") as sink,
        pa.ipc.new_file(sink, table.schema) as writer,
    ):
        writer.write_table(table)
    return str(path)


class EngineDiagnosticCase:
    """Use the original timed adapter and expose its untimed comparison output."""

    def __init__(self, case: dict, root: Path) -> None:
        from benchmarks.engine_comparison import EngineCase

        output = root / "comparison.arrow"
        ordered = case["backend"] == "calc-flow-stream"

        class ComparedEngine(EngineCase):
            def validate(self, result):
                correctness = super().validate(result)
                save_output(result, output, ordered=ordered)
                return correctness

        self.engine = ComparedEngine(case, root / "jobs")
        self.output = output

    def sample(self) -> dict:
        return {**self.engine.sample(), "comparison_output": str(self.output)}

    def finish(self) -> dict:
        return self.engine.finish()

    def close(self) -> None:
        self.engine.close()


FALLBACK_VARIANTS = (
    "map_partition_single_batch",
    "unsupported_count",
    "rewrite_disabled",
)


def _fallback_batch_size(case: dict) -> int:
    """Validate the closed layout/configuration declaration before preparation."""
    if case["diagnostic_variant"] == "map_partition_single_batch":
        batch_size = max(8192, case["rows"])
        if (
            case.get("input_layout") != "single_batch"
            or type(case.get("batch_size")) is not int
            or case["batch_size"] != batch_size
        ):
            raise ValueError("SQL fallback configuration differs from its variant")
        return batch_size
    if "input_layout" in case or "batch_size" in case:
        raise ValueError("SQL fallback configuration differs from its variant")
    return 8192


def diagnostic_sql(scenario: str, variant: str | None = None) -> str:
    """Return a closed query variant without modifying the original workloads."""
    from benchmarks.engine_comparison import sql_query

    if variant is not None:
        if variant not in FALLBACK_VARIANTS or scenario not in ("sma20", "dual_sma"):
            raise ValueError("unsupported SQL diagnostic variant")
        query = sql_query(scenario)
        if variant == "map_partition_single_batch":
            return query.replace("PARTITION BY symbol", "PARTITION BY entity_map")
        if variant == "unsupported_count":
            return query.replace(
                "COUNT(price) OVER slow = 20 ",
                "COUNT(price) OVER slow = 20 "
                "AND COUNT(DISTINCT sequence) OVER slow = 20 ",
            )
        return query
    if scenario == "filter_uint64_modulo":
        return (
            "SELECT sequence, price AS value FROM input "
            "WHERE sequence % CAST(4 AS BIGINT UNSIGNED) "
            "= CAST(0 AS BIGINT UNSIGNED)"
        )
    return sql_query(scenario)


def _map_partition_input(table, entities: int):
    import numpy as np
    import pyarrow as pa

    chunks = [
        pa.MapArray.from_arrays(
            pa.array(np.arange(batch.num_rows + 1, dtype=np.int32)),
            pa.repeat("entity", batch.num_rows),
            pa.array(batch["sequence"].to_numpy() % entities, type=pa.int32()),
        )
        for batch in table.to_batches()
    ]
    maps = pa.chunked_array(chunks)
    return table.append_column(pa.field("entity_map", maps.type, nullable=False), maps)


def _full_window_nulls(table):
    """Derive local entity ranks from the actual fixture's window ordering."""
    import numpy as np
    import pyarrow as pa

    ordered = table.select(["symbol", "event_time", "sequence"]).sort_by(
        [
            ("symbol", "ascending"),
            ("event_time", "ascending"),
            ("sequence", "ascending"),
        ]
    )
    codes = ordered["symbol"].combine_chunks().dictionary_encode().indices.to_numpy()
    positions = np.arange(ordered.num_rows)
    starts = np.concatenate(([True], codes[1:] != codes[:-1]))
    local_positions = positions - np.maximum.accumulate(np.where(starts, positions, 0))
    return pa.table(
        {"sequence": ordered["sequence"], "is_null": local_positions < 19}
    ).sort_by("sequence")["is_null"]


def _sql_fixture(table, entities: int, nulls) -> dict:
    """Fingerprint the same extended Arrow table passed to Batch.from_pyarrow."""
    import pyarrow as pa

    with io.BytesIO() as output:
        with pa.ipc.new_stream(output, table.schema) as writer:
            writer.write_table(table)
        digest = hashlib.sha256(output.getbuffer()).hexdigest()
    return {
        "input_sha256": digest,
        "schema_sha256": hashlib.sha256(table.schema.serialize()).hexdigest(),
        "batch_rows": [batch.num_rows for batch in table.to_batches()],
        "rows": table.num_rows,
        "entities": entities,
        "full_window_rows": table.num_rows - nulls.to_numpy().sum().item(),
    }


def _check_rewrite_setting(case: dict, variant: str | None) -> None:
    if variant is not None and case.get("enable_rolling_rewrite") is not (
        variant != "rewrite_disabled"
    ):
        raise ValueError("SQL fallback variant has a different rewrite setting")


def _diagnostic_table(data, variant: str | None):
    table = data.table
    if variant != "map_partition_single_batch":
        return table
    import pyarrow as pa

    table = _map_partition_input(table, data.entities)
    return pa.Table.from_batches(
        [
            pa.RecordBatch.from_arrays(
                [column.combine_chunks() for column in table.columns],
                schema=table.schema,
            )
        ]
    )


def _native_fixture_facts(batch, table, fixture: dict, case: dict) -> dict:
    variant = case["diagnostic_variant"]
    roundtrip = batch.to_pyarrow()
    native_rows = [batch.num_rows for batch in roundtrip.to_batches()]
    if (
        not roundtrip.equals(table, check_metadata=True)
        or native_rows != fixture["batch_rows"]
        or (variant == "map_partition_single_batch" and native_rows != [case["rows"]])
    ):
        raise ValueError("SQL fallback native input roundtrip changed")
    return {
        "native_batch_rows": native_rows,
        "input_layout": (
            "single_batch"
            if variant == "map_partition_single_batch"
            else "engine_batches"
        ),
    }


class SqlDiagnosticCase:
    """Own a SQL fixture and collect plans outside execute-to-Arrow timing."""

    def __init__(self, case: dict, root: Path) -> None:
        from benchmarks.engine_comparison import EngineCase
        from calc_flow import Batch, PipelineBuilder
        from scripts.benchmark_suite.catalog import THREADS

        scenario = (
            "filter" if case["scenario"] == "filter_uint64_modulo" else case["scenario"]
        )
        variant = case.get("diagnostic_variant")
        self.query = diagnostic_sql(case["scenario"], variant)
        batch_size = _fallback_batch_size(case) if variant else 8192
        _check_rewrite_setting(case, variant)
        self.engine = EngineCase({**case, "scenario": scenario}, root)
        self.output = root / "comparison.arrow"
        aliases = ("input", "dimension") if scenario == "join" else ("input",)
        self.plan = (
            PipelineBuilder("performance-diagnostic-sql")
            .with_datafusion_config(
                target_partitions=THREADS,
                batch_size=batch_size,
                enable_rolling_rewrite=case.get("enable_rolling_rewrite", True),
            )
            .sql("query", self.query, aliases=aliases)
            .compile_batch()
        )
        table = _diagnostic_table(self.engine.data, variant)
        self.full_window_nulls = _full_window_nulls(table) if variant else None
        self.fixture = (
            _sql_fixture(table, self.engine.data.entities, self.full_window_nulls)
            if variant
            else None
        )
        self.inputs = {"input": Batch.from_pyarrow(table)}
        if variant:
            self.fixture.update(
                _native_fixture_facts(self.inputs["input"], table, self.fixture, case),
            )
        if scenario == "join":
            self.inputs["dimension"] = Batch.from_pyarrow(self.engine.data.dimension)

    def sample(self) -> dict:
        started = time.perf_counter_ns()
        result = self.plan.execute(self.inputs)
        table = result.outputs["output"].to_pyarrow()
        seconds = (time.perf_counter_ns() - started) / 1e9
        extra = {}
        if self.full_window_nulls is not None:
            actual = table.select(["sequence", "value"]).sort_by("sequence")
            nulls = actual["value"].is_null()
            if not nulls.equals(self.full_window_nulls):
                raise ValueError("SQL full-window per-entity null bitmap changed")
            extra = {
                "fixture": self.fixture,
                "full_window_validation": {
                    "passed": True,
                    "full_window_rows": self.fixture["full_window_rows"],
                    "null_rows": self.fixture["rows"]
                    - self.fixture["full_window_rows"],
                },
            }
        return {
            "seconds": seconds,
            "correctness": self.engine.validate(table),
            "query": self.query,
            "datafusion_metrics": result.datafusion_metrics,
            "comparison_output": save_output(table, self.output, ordered=False),
            **extra,
        }

    def finish(self) -> dict:
        return {"state": "completed"}

    def close(self) -> None:
        self.engine.close()


class NativeDiagnosticCase:
    """Own a warm stream with explicit dimensions and sparse-append identity."""

    def __init__(self, case: dict, root: Path) -> None:
        from benchmarks.warm_stream import ScenarioConfig, WarmScenario

        config = ScenarioConfig(**case["config"])
        if config.append_rows != case["rows"]:
            raise ValueError("diagnostic row count must match the actual append")
        self.loop = asyncio.new_event_loop()
        self.finished = False

        class ComparedScenario(WarmScenario):
            output: Path | None = None

            def validate(self, table, rows):
                correctness = super().validate(table, rows)
                if self.output is not None:
                    save_output(table, self.output, ordered=True)
                return correctness

        try:
            self.scenario = self.loop.run_until_complete(
                ComparedScenario.start(config, root / "job")
            )
            self.scenario.output = root / "comparison.arrow"
        except BaseException:
            self.loop.close()
            raise

    def sample(self) -> dict:
        return {
            **self.loop.run_until_complete(self.scenario.sample(collect_gc=False)),
            "comparison_output": str(self.scenario.output),
        }

    def finish(self) -> dict:
        result = self.loop.run_until_complete(self.scenario.finish())
        self.finished = True
        if not self.scenario.sink._tables.empty():
            raise ValueError(
                "warm EOF delivered extra output after the measured appends"
            )
        delivery = result["after_status"]["sinks"]["profile"]
        if (
            delivery["delivered_rows"] != self.scenario.position
            or delivery["errors"] != 0
            or delivery["ended"] is not True
        ):
            raise ValueError(
                "warm terminal sink delivery does not match validated rows"
            )
        return {
            **result,
            "delivery_validation": {
                "expected_rows": self.scenario.position,
                "delivered_rows": delivery["delivered_rows"],
                "queue_empty": True,
            },
        }

    def close(self) -> None:
        try:
            if not self.finished:
                self.loop.run_until_complete(self.scenario.job.cancel_async())
            self.loop.run_until_complete(self.loop.shutdown_asyncgens())
        finally:
            self.loop.close()
