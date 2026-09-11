"""Produce an exact-base Native checkpoint and verify it with a candidate wheel."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

BASELINE_SHA = "e1a1228d53e361c3576ce9a250f9843128e3c6d7"
BASE = datetime(2026, 1, 1, tzinfo=UTC)
PREFIX_ROWS = 48
BATCH_SLICES = ((0, 16), (16, 16), (32, 16), (48, 8), (56, 8))
RTOL = 1e-12
ATOL = 1e-12


def require_baseline(build: dict) -> None:
    """Require the approved immutable producer revision."""
    if build.get("source_sha") != BASELINE_SHA:
        raise ValueError("checkpoint producer must be the approved baseline revision")


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "provenance.json"
    }


def verify_artifacts(root: Path, expected: dict[str, str]) -> None:
    if artifact_hashes(root) != expected:
        raise ValueError("checkpoint fixture artifact integrity mismatch")


def copy_checkpoint(fixture: Path, destination: Path) -> dict:
    provenance = json.loads((fixture / "provenance.json").read_text())
    require_baseline(provenance["producer"]["build"])
    verify_artifacts(fixture, provenance["artifact_hashes"])
    if destination.resolve().is_relative_to(fixture.resolve()):
        raise ValueError("recovery output must be outside the immutable fixture")
    shutil.copytree(fixture / "checkpoint", destination)
    return provenance


def _float_equivalent(left, right) -> bool:
    if left is None or right is None:
        return left is right
    if math.isnan(left) or math.isnan(right):
        return math.isnan(left) and math.isnan(right)
    if math.isinf(left) or math.isinf(right):
        return left == right
    return math.isclose(left, right, rel_tol=RTOL, abs_tol=ATOL)


def assert_equivalent(actual, expected) -> None:
    import pyarrow as pa

    if not actual.schema.equals(expected.schema, check_metadata=True):
        raise AssertionError("output Arrow schemas or metadata differ")
    if actual.num_rows != expected.num_rows:
        raise AssertionError("output Arrow row counts differ")
    for field, got, wanted in zip(
        actual.schema, actual.columns, expected.columns, strict=True
    ):
        if not pa.types.is_floating(field.type):
            if not got.equals(wanted):
                raise AssertionError(f"output column {field.name} differs")
            continue
        for row, (left, right) in enumerate(
            zip(got.to_pylist(), wanted.to_pylist(), strict=True)
        ):
            if not _float_equivalent(left, right):
                raise AssertionError(
                    f"output {field.name}[{row}] differs: {left!r} != {right!r}"
                )


def release_identity(build_path: Path) -> dict:
    import calc_flow
    from benchmarks.native_memory_profile import prove_release
    from calc_flow import _native

    build = json.loads(build_path.read_text())
    native = Path(_native.__file__).resolve()
    proof = prove_release(build, native)
    wheel = Path(build["wheel"])
    if file_hash(wheel) != build["wheel_sha256"]:
        raise ValueError("release wheel does not match its build record")
    package = Path(calc_flow.__file__).resolve().parent
    if native.parent != package:
        raise ValueError("Python and native modules came from different packages")
    with zipfile.ZipFile(wheel) as archive:
        for member in archive.namelist():
            if not member.startswith("calc_flow/") or member.endswith("/"):
                continue
            with archive.open(member) as content:
                digest = hashlib.file_digest(content, "sha256").hexdigest()
            if file_hash(package.parent / member) != digest:
                raise ValueError(
                    f"loaded package differs from the release wheel: {member}"
                )
    return {
        **proof,
        "python_executable": sys.executable,
        "python_package": str(package),
        "python_version": platform.python_version(),
        "instrument_sha256": file_hash(Path(__file__)),
    }


def _program():
    from calc_flow.symbolic import FeatureSet, Field, Program, rows, table_input, ts

    quotes = table_input(
        "quotes",
        schema=(
            Field("event_time", "timestamp[us, UTC]", nullable=False),
            Field("sequence", "uint64", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("price", "float64"),
        ),
        entity_by=("symbol",),
        event_time="event_time",
        sequence_by=("sequence",),
    )
    slow = ts.mean(quotes["price"], window=rows(20), min_periods=20)
    fast = ts.mean(quotes["price"], window=rows(5), min_periods=5)
    signals = quotes.with_columns(
        FeatureSet((("slow", slow), ("fast", fast), ("spread", fast - slow)))
    )
    return Program(
        "checkpoint-cross-version", inputs=(quotes,), outputs=(("signals", signals),)
    )


def _compile():
    from calc_flow import Runtime

    runtime = Runtime()
    program = _program()
    document = program.to_project(runtime, mode="stream").model_dump(mode="json")
    project_json = json.dumps(document, sort_keys=True, indent=2) + "\n"
    return program.compile_stream(runtime), project_json


def _input():
    import pyarrow as pa

    special_values = {
        ("a", 3): None,
        ("a", 27): float("nan"),
        ("b", 25): float("inf"),
        ("b", 29): float("-inf"),
    }
    records = []
    for tick in range(1, 33):
        for entity, symbol in enumerate(("a", "b")):
            value = special_values.get((symbol, tick), float(tick + 100 * entity))
            records.append(
                {
                    "event_time": BASE + timedelta(microseconds=tick),
                    "sequence": 2 * (tick - 1) + entity,
                    "symbol": symbol,
                    "price": value,
                }
            )
    schema = pa.schema(
        [
            pa.field("event_time", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("sequence", pa.uint64(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("price", pa.float64()),
        ]
    )
    return pa.Table.from_pylist(records, schema=schema)


class _ReplaySource:
    def __init__(self, table, pause: bool) -> None:
        self.batches = tuple(table.slice(start, size) for start, size in BATCH_SLICES)
        self.pause = pause
        self.index = 0
        self.watermark = None
        self.paused = asyncio.Event()
        self.opened_indices: list[int] = []

    def capabilities(self):
        from calc_flow import (
            NativeWatermarkCapability,
            ReplayPositioning,
            SourceCapabilities,
            SourceDeliveryCapability,
        )

        return SourceCapabilities(
            ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=16,
            max_batch_bytes=16 * 1024 * 1024,
            native_watermarks=NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor) -> None:
        self.index = 0 if cursor is None else int(cursor.payload["next_batch"])
        if not 0 <= self.index <= len(self.batches):
            raise ValueError("restored cursor is outside the frozen input sequence")
        self.opened_indices.append(self.index)

    async def next(self):
        from calc_flow import Batch, Cursor, Data, Idle, Watermark

        if self.watermark is not None:
            current, self.watermark = self.watermark, None
            return Watermark(current)
        if self.pause and self.index == 3:
            self.paused.set()
            await asyncio.sleep(0.001)
            return Idle()
        if self.index == len(self.batches):
            return None
        table = self.batches[self.index]
        self.index += 1
        self.watermark = table["event_time"][-1].as_py()
        return Data(
            Batch.from_pyarrow(table),
            Cursor(self.index.to_bytes(8, "big"), {"next_batch": self.index}),
        )

    async def close(self) -> None:
        return None


class _Sink:
    def __init__(self) -> None:
        self.tables = []
        self.rows = 0
        self.prefix_complete = asyncio.Event()

    async def open(self) -> None:
        return None

    async def write(self, batch) -> None:
        table = batch.to_pyarrow()
        self.tables.append(table)
        self.rows += table.num_rows
        if self.rows >= PREFIX_ROWS:
            self.prefix_complete.set()

    async def close(self) -> None:
        return None


async def _run(plan, table, state_root: Path, *, checkpoint: bool) -> tuple:
    import pyarrow as pa

    from calc_flow import (
        ManagedCheckpointRuntime,
        SinkBinding,
        SourceBinding,
        SourceProvidedWatermarks,
        StreamingRunner,
        StreamRuntimeConfig,
    )

    source, sink = _ReplaySource(table, checkpoint), _Sink()
    job = await StreamingRunner(
        plan,
        {
            plan.source_binding_ids[0]: SourceBinding(
                source, watermark_policy=SourceProvidedWatermarks()
            )
        },
        {plan.sink_binding_ids[0]: [SinkBinding.ordinary("archive", sink)]},
        ManagedCheckpointRuntime(state_root),
        config=StreamRuntimeConfig(checkpoint_interval=timedelta(hours=24)),
    ).start_async()
    try:
        if checkpoint:
            await asyncio.wait_for(source.paused.wait(), timeout=30)
            await asyncio.wait_for(sink.prefix_complete.wait(), timeout=30)
            epoch = await asyncio.wait_for(job.trigger_checkpoint_async(), timeout=30)
            if epoch != 1 or sink.rows != PREFIX_ROWS:
                raise AssertionError(
                    "baseline checkpoint did not capture exactly the committed prefix"
                )
            status = job.status()
            outcome = await asyncio.wait_for(job.cancel_async(), timeout=30)
            expected_state = "cancelled"
        else:
            outcome = await asyncio.wait_for(job.wait_async(), timeout=30)
            status = job.status()
            expected_state = "completed"
        if outcome.state != expected_state or outcome.errors:
            raise AssertionError(f"unexpected checkpoint run outcome: {outcome}")
        return pa.concat_tables(sink.tables), source.opened_indices, status
    finally:
        await asyncio.wait_for(job.cancel_async(), timeout=30)


def _write_arrow(path: Path, table) -> None:
    import pyarrow as pa

    with (
        pa.OSFile(str(path), "wb") as output,
        pa.ipc.new_file(output, table.schema) as writer,
    ):
        writer.write_table(table)


def _read_arrow(path: Path):
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        return pa.ipc.open_file(source).read_all()


def _checkpoint_manifest(root: Path) -> tuple[Path, dict]:
    manifests = sorted((root / "manifests").glob("manifest-*.json"))
    if len(manifests) != 1:
        raise AssertionError("expected exactly one nonterminal baseline manifest")
    manifest = json.loads(manifests[0].read_text())
    if manifest["format_version"] != 3 or manifest["epoch"] != 1:
        raise AssertionError("baseline manifest must be the first v3 checkpoint")
    source = manifest["sources"]["input"]
    if (
        source["ended"]
        or source["cursor"]["payload"] != {"next_batch": 3}
        or source["cursor"]["order"] != "0000000000000003"
    ):
        raise AssertionError("baseline checkpoint has an unexpected replay cursor")
    return manifests[0], manifest


def _checkpoint_state_tables(root: Path, segments):
    state_tables = []
    for segment in segments:
        path = root / "state" / segment["relative_path"]
        if (
            path.stat().st_size != segment["byte_len"]
            or file_hash(path) != segment["sha256"]
        ):
            raise AssertionError("baseline state segment differs from its manifest")
        state_tables.append(_read_arrow(path))
    return state_tables


def _rolling_checkpoint_facts(root: Path, entry: dict) -> dict:
    metadata = entry["inline_metadata"]
    if metadata["ended"] or metadata["state_layout_version"] != 3:
        raise AssertionError("baseline rolling checkpoint must be live writer layout 3")
    state_tables = _checkpoint_state_tables(root, entry["segments"])
    history = Counter()
    buffered = 0
    for table in state_tables:
        for kind, entity in zip(
            table["_state_kind"].to_pylist(),
            table["_entity_id"].to_pylist(),
            strict=True,
        ):
            history[entity] += int(kind == 1)
            buffered += int(kind == 2)
    if sorted(history.values()) != [20, 20] or buffered:
        raise AssertionError(
            f"checkpoint lacks committed per-entity history: {history}, "
            f"buffered={buffered}"
        )
    return {
        "inline_metadata": metadata,
        "history_rows_by_entity": dict(history),
        "buffered_rows": buffered,
    }


def _checkpoint_facts(root: Path) -> dict:
    manifest_path, manifest = _checkpoint_manifest(root)
    operators = {}
    for operator_id, entry in manifest["operators"].items():
        metadata = entry["inline_metadata"]
        if "numerical_profile" not in metadata:
            continue
        operators[operator_id] = _rolling_checkpoint_facts(root, entry)
    if not operators:
        raise AssertionError("baseline checkpoint contains no rolling state")
    return {
        "manifest": str(manifest_path.relative_to(root)),
        "pipeline_fingerprint": manifest["pipeline_fingerprint"],
        "operators": operators,
        "sources": manifest["sources"],
    }


def prepare(args, producer: dict) -> dict:
    require_baseline(producer["build"])
    args.fixture.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), args.fixture / "instrument.py")
    plan, project = _compile()
    fingerprint = plan.fingerprint
    table = _input()
    (args.fixture / "project.json").write_text(project)
    _write_arrow(args.fixture / "input.arrow", table)
    prefix, cursors, status = asyncio.run(
        _run(plan, table, args.fixture / "checkpoint", checkpoint=True)
    )
    _write_arrow(args.fixture / "baseline-prefix.arrow", prefix)
    facts = _checkpoint_facts(args.fixture / "checkpoint")
    if facts["pipeline_fingerprint"] != fingerprint:
        raise AssertionError("baseline manifest does not identify its compiled graph")
    with tempfile.TemporaryDirectory(
        prefix="checkpoint-baseline-oracle-", dir=args.fixture.parent
    ) as directory:
        reference_plan, reference_project = _compile()
        if reference_project != project or reference_plan.fingerprint != fingerprint:
            raise AssertionError("baseline oracle graph differs from checkpoint graph")
        reference, reference_cursors, _ = asyncio.run(
            _run(reference_plan, table, Path(directory), checkpoint=False)
        )
    assert_equivalent(prefix, reference.slice(0, PREFIX_ROWS))
    if reference.num_rows != 64 or cursors != [0] or reference_cursors != [0]:
        raise AssertionError("baseline did not consume the frozen full input sequence")
    _write_arrow(args.fixture / "baseline-full.arrow", reference)
    _write_arrow(args.fixture / "baseline-suffix.arrow", reference.slice(PREFIX_ROWS))
    record = {
        "format": "calc-flow.native-checkpoint-compatibility/1",
        "producer": producer,
        "recorded_utc": datetime.now(UTC).isoformat(),
        "project_sha256": file_hash(args.fixture / "project.json"),
        "graph_fingerprint": fingerprint,
        "input": {
            "rows": 64,
            "prefix_rows": PREFIX_ROWS,
            "batch_slices": BATCH_SLICES,
            "sequence": table["sequence"].to_pylist(),
            "cursor": {"next_batch": 3},
            "opened_indices": cursors,
        },
        "comparison": {
            "rtol": RTOL,
            "atol": ATOL,
            "schema_metadata_validity_nan_infinity": "exact",
        },
        "checkpoint_facts": facts,
        "prefix_status": status,
        "artifact_hashes": artifact_hashes(args.fixture),
        "candidate_verified": False,
    }
    (args.fixture / "provenance.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    return record


def verify(args, candidate: dict) -> dict:
    if candidate["build"]["source_sha"] == BASELINE_SHA:
        raise ValueError(
            "candidate verification requires a different exact source revision"
        )
    if args.output is None:
        raise ValueError(
            "candidate verification needs an independent --output directory"
        )
    args.output.mkdir(parents=True, exist_ok=False)
    provenance = copy_checkpoint(args.fixture, args.output / "checkpoint")
    plan, project = _compile()
    fingerprint = plan.fingerprint
    if (
        project != (args.fixture / "project.json").read_text()
        or fingerprint != provenance["graph_fingerprint"]
    ):
        raise AssertionError("candidate project or native graph fingerprint changed")
    table = _read_arrow(args.fixture / "input.arrow")
    actual, cursors, status = asyncio.run(
        _run(plan, table, args.output / "checkpoint", checkpoint=False)
    )
    if cursors != [3]:
        raise AssertionError(
            f"candidate did not seek to the durable base cursor: {cursors}"
        )
    assert_equivalent(actual, _read_arrow(args.fixture / "baseline-suffix.arrow"))
    verify_artifacts(args.fixture, provenance["artifact_hashes"])
    _write_arrow(args.output / "candidate-suffix.arrow", actual)
    record = {
        "format": provenance["format"],
        "candidate": candidate,
        "producer_source_sha": provenance["producer"]["build"]["source_sha"],
        "producer_provenance_sha256": file_hash(args.fixture / "provenance.json"),
        "producer_graph_fingerprint": provenance["graph_fingerprint"],
        "candidate_graph_fingerprint": fingerprint,
        "candidate_project_sha256": hashlib.sha256(project.encode()).hexdigest(),
        "opened_indices": cursors,
        "output_rows": actual.num_rows,
        "comparison": provenance["comparison"],
        "status": status,
        "decision": "pass",
    }
    (args.output / "verification.json").write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    identity = release_identity(args.build)
    result = (
        prepare(args, identity) if args.mode == "prepare" else verify(args, identity)
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "fixture": str(args.fixture),
                "source_sha": identity["build"]["source_sha"],
                "graph_fingerprint": result.get(
                    "graph_fingerprint", result.get("candidate_graph_fingerprint")
                ),
                "decision": result.get(
                    "decision",
                    "baseline fixture prepared; candidate verification pending",
                ),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
