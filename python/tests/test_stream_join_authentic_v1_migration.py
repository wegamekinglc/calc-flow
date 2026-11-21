"""Reject an immutable legacy Join checkpoint before opening connectors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf

FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/stream-join-authentic-v1"
)
FINGERPRINT = "adf19867ed435dd9e58505a1876ea397b9edf2ce588ff1f82b11312f9f4fdb88"
PROVENANCE_SHA = "9170131030c9c4e0c215b7c877f7ed8714a95f8d858f4786cf212e05fd9c32c7"


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _verify_fixture() -> dict[str, object]:
    assert _sha(FIXTURE / "provenance.json") == PROVENANCE_SHA
    provenance = _json(FIXTURE / "provenance.json")
    assert (
        provenance["selected_member_count"],
        provenance["selected_member_bytes"],
    ) == (
        12,
        39654,
    )
    for member in provenance["members"]:
        path = FIXTURE / member["path"]
        assert (path.stat().st_size, _sha(path)) == (member["size"], member["sha256"])
    return provenance


def _copy_fixture_root(root: Path) -> None:
    prefix = "attempt-005/capture/managed-root/"
    for member in _verify_fixture()["members"]:
        if member["archive_member"].startswith(prefix):
            destination = root / member["archive_member"][len(prefix) :]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(FIXTURE / member["path"], destination)


def _input(side: str) -> pa.Table:
    with pa.ipc.open_file(FIXTURE / "capture" / f"{side}-all-inputs.arrow") as reader:
        return reader.read_all()


def _document_values(column: pa.ChunkedArray) -> list[object]:
    if pa.types.is_timestamp(column.type) and column.type.tz == "UTC":
        return [
            value.replace(tzinfo=timezone.utc).isoformat()
            if value is not None
            else None
            for value in column.cast(pa.timestamp(column.type.unit)).to_pylist()
        ]
    return [
        value.isoformat() if isinstance(value, datetime) else value
        for value in column.to_pylist()
    ]


def _documents(table: pa.Table) -> list[dict[str, object]]:
    columns = [_document_values(column) for column in table.columns]
    return [
        {name: column[index] for name, column in zip(table.column_names, columns)}
        for index in range(table.num_rows)
    ]


def _build_plan(version: str = "1"):
    fields = [
        cf.ArrowFieldSpec("key", "int64"),
        cf.ArrowFieldSpec("at", "timestamp[us, UTC]", False),
        cf.ArrowFieldSpec("sequence", "uint64", False),
        cf.ArrowFieldSpec("value", "int64", False),
        cf.ArrowFieldSpec("note", "string"),
    ]
    builder = (
        cf.PipelineBuilder("authentic-join-v1-release-fixture")
        .stream_join(
            "join",
            left_schema=fields,
            right_schema=fields,
            left_keys=["key"],
            right_keys=["key"],
            left_event_time="at",
            right_event_time="at",
            bounds=cf.JoinTimeBounds(timedelta(seconds=5), timedelta(seconds=5)),
            limits=cf.JoinStateLimits(100, 1 << 20, 1000),
        )
        .expression(
            "proof",
            "",
            select=("fixture_identity(value) AS checked",),
            udfs=(("fixture", "fixture_identity", "1"),),
        )
    )
    assert builder.project == _json(FIXTURE / "capture/project.json")
    runtime = cf.Runtime()
    runtime.register_scalar_udf(
        provider="fixture",
        name="fixture_identity",
        version=version,
        input_types=["int64"],
        return_type="int64",
        volatility="immutable",
        function=lambda value: value,
    )
    plan = builder.compile_stream(runtime=runtime)
    assert plan.fingerprint == FINGERPRINT
    assert plan.source_binding_ids == ("input", "left", "right")
    assert plan.sink_binding_ids == ("join.output", "proof.output")
    return plan


class _ReplaySource:
    def __init__(self, side: str, table: pa.Table, events: list[dict[str, object]]):
        self.side = side
        self.table = table
        self.events = events
        self.offset = 0
        self.continue_allowed = False
        self.end_allowed = False

    def capabilities(self):
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=4,
            max_batch_bytes=1 << 20,
            schema=self.table.schema,
            native_watermarks=cf.NativeWatermarkCapability.EMITS_NATIVE,
        )

    async def open(self, cursor) -> None:
        assert cursor is not None
        assert cursor.order == (4).to_bytes(8, "big")
        assert dict(cursor.payload) == {
            "row_offset": 4,
            "dataset": "authentic-join-v1-v1",
        }
        self.offset = int(cursor.payload["row_offset"])
        self.events.append(
            {
                "kind": "open",
                "side": self.side,
                "row_offset": self.offset,
                "order": cursor.order.hex(),
                "source_id": cursor.source_id,
            }
        )

    async def next(self):
        if self.offset == 4 and self.continue_allowed:
            self.offset = 6
            self.events.append(
                {"kind": "data", "side": self.side, "start": 4, "rows": 2}
            )
            return cf.Data(
                cf.Batch.from_pyarrow(self.table.slice(4, 2)),
                cf.Cursor(
                    (6).to_bytes(8, "big"),
                    {"row_offset": 6, "dataset": "authentic-join-v1-v1"},
                ),
            )
        if self.offset == 6 and self.end_allowed:
            self.events.append({"kind": "eof", "side": self.side})
            return None
        await asyncio.sleep(0.002)
        return cf.Idle()

    async def close(self) -> None:
        self.events.append(
            {"kind": "close", "side": self.side, "row_offset": self.offset}
        )


class _RecordingSink:
    def __init__(self):
        self.rows: list[dict[str, object]] = []
        self.schemas: list[str] = []
        self.lifecycle: list[str] = []

    async def open(self) -> None:
        self.lifecycle.append("open")

    async def write(self, batch) -> None:
        table = batch.to_pyarrow()
        self.rows.extend(_documents(table))
        self.schemas.append(str(table.schema))

    async def close(self) -> None:
        self.lifecycle.append("close")

    def proof(self) -> dict[str, object]:
        return {"rows": self.rows, "schemas": self.schemas, "lifecycle": self.lifecycle}


def _reject_udf_version(observations, events, sink, udf_sink) -> None:
    with pytest.raises(
        cf.ConfigError,
        match=(
            r"graph\.nodes\[1\]\.operator\.udfs\[0\] \[missing_udf\]: "
            "UDF fixture:fixture_identity@1 is unavailable"
        ),
    ) as error:
        _build_plan("2")
    observations["diagnostic"] = str(error.value)
    assert not events and not sink.lifecycle and not udf_sink.lifecycle


def _create_runner(root: Path, sources, sink, udf_sink):
    return cf.StreamingRunner(
        _build_plan(),
        {
            "input" if side == "udf" else side: cf.SourceBinding(
                source, watermark_policy=cf.SourceProvidedWatermarks()
            )
            for side, source in sources.items()
        },
        {
            "join.output": [cf.SinkBinding.ordinary("fixture-archive", sink)],
            "proof.output": [cf.SinkBinding.ordinary("fixture-udf-archive", udf_sink)],
        },
        cf.ManagedCheckpointRuntime(root),
        config=cf.StreamRuntimeConfig(edge_budget=cf.EdgeBudget(10_000, 64 << 20)),
    )


async def _reject_corrupt_state(runner, observations, events, sink, udf_sink) -> None:
    with pytest.raises(
        cf.StreamingRuntimeError,
        match="checkpoint lineage contains invalid recovery data",
    ) as error:
        await asyncio.wait_for(runner.start_async(), 30)
    observations["diagnostic"] = str(error.value)
    observations["category"] = error.value.category
    assert error.value.category == "checkpoint_mismatch"
    assert not events and not sink.lifecycle and not udf_sink.lifecycle


async def _run_worker(mode: str, root: Path, observations, inputs) -> None:
    events: list[dict[str, object]] = []
    observations["source_events"] = events
    sources = {
        side: _ReplaySource(side, inputs["left" if side == "udf" else side], events)
        for side in ("left", "right", "udf")
    }
    sink, udf_sink = _RecordingSink(), _RecordingSink()
    if mode == "udf":
        _reject_udf_version(observations, events, sink, udf_sink)
    elif mode == "corrupt":
        runner = _create_runner(root, sources, sink, udf_sink)
        await _reject_corrupt_state(runner, observations, events, sink, udf_sink)
    else:
        raise ValueError(f"unknown worker mode: {mode}")
    observations["join_sink"], observations["udf_sink"] = sink.proof(), udf_sink.proof()


def _worker_main(
    mode: str, root: Path, output: Path, native: Path, native_sha: str
) -> None:
    loaded = Path(cf._native.__file__).resolve()
    observations = {
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "package": cf.__file__,
        "native": str(loaded),
        "native_sha256": _sha(loaded),
        "mode": mode,
    }
    try:
        assert loaded == native.resolve()
        assert observations["native_sha256"] == native_sha
        inputs = {side: _input(side) for side in ("left", "right")}
        asyncio.run(_run_worker(mode, root, observations, inputs))
        observations["success"] = True
    except BaseException:
        observations["traceback"] = traceback.format_exc()
        raise
    finally:
        _write(output / "observations.json", observations)


def _launch(mode: str, root: Path, output: Path) -> dict[str, object]:
    output.mkdir()
    native = Path(cf._native.__file__).resolve()
    args = [
        str(Path(__file__).resolve()),
        mode,
        str(root),
        str(output),
        str(native),
        _sha(native),
    ]
    bootstrap = (
        "import runpy,sys; "
        f"sys.path.insert(0, {str(Path(cf.__file__).resolve().parent.parent)!r}); "
        f"sys.argv={args!r}; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    command = [sys.executable, "-I", "-c", bootstrap]
    with (
        (output / "stdout.log").open("wb") as stdout,
        (output / "stderr.log").open("wb") as stderr,
    ):
        # The interpreter, worker script, and modes are trusted test inputs.
        # Paths use repr in the fixed Python bootstrap; argv invokes no shell.
        process = subprocess.Popen(  # noqa: E501  # nosemgrep: python.lang.security.audit.dangerous-subprocess-use-audit.dangerous-subprocess-use-audit
            command, stdout=stdout, stderr=stderr
        )
        _write(output / "command.json", {"argv": command, "pid": process.pid})
        try:
            code = process.wait(timeout=120)
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            _write(
                output / "exit.json",
                {
                    "pid": process.pid,
                    "exit_code": process.returncode,
                    "settled": process.poll() is not None,
                },
            )
    assert code == 0, (output / "stderr.log").read_text()
    return _json(output / "observations.json")


def test_authentic_v1_checkpoint_is_rejected_without_fallback(tmp_path):
    _verify_fixture()
    root = tmp_path / "authentic-v1"
    _copy_fixture_root(root)
    before = {
        path.relative_to(root): _sha(path) for path in root.rglob("*") if path.is_file()
    }

    observations = _launch("corrupt", root, tmp_path / "rejection-control")

    assert (
        "checkpoint lineage contains invalid recovery data"
        in observations["diagnostic"]
    )
    assert observations["category"] == "checkpoint_mismatch"
    assert observations["source_events"] == []
    assert observations["join_sink"]["lifecycle"] == []
    assert observations["udf_sink"]["lifecycle"] == []
    assert all(_sha(root / path) == digest for path, digest in before.items())
    _verify_fixture()


def test_authentic_v1_selected_udf_version_fails_before_connectors_open(tmp_path):
    _verify_fixture()
    observations = _launch("udf", tmp_path / "unused-root", tmp_path / "udf-control")
    assert (
        "graph.nodes[1].operator.udfs[0] [missing_udf]: "
        "UDF fixture:fixture_identity@1 is unavailable" in observations["diagnostic"]
    )
    assert observations["source_events"] == []
    assert (
        observations["join_sink"]["lifecycle"]
        == observations["udf_sink"]["lifecycle"]
        == []
    )
    assert not (tmp_path / "unused-root").exists()
    _verify_fixture()


if __name__ == "__main__":
    _worker_main(
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
        sys.argv[5],
    )
