"""Recover an immutable published-wheel cut through the current Join writer."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pytest

import calc_flow as cf

FIXTURE = (
    Path(__file__).resolve().parents[2] / "tests/fixtures/stream-join-authentic-v1"
)
FINGERPRINT = "adf19867ed435dd9e58505a1876ea397b9edf2ce588ff1f82b11312f9f4fdb88"
WATERMARK = 1767225610000000
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


def _manifest(root: Path, epoch: int) -> dict[str, object]:
    return _json(root / "manifests" / f"manifest-{epoch:020d}.json")


def _input(side: str) -> pa.Table:
    with pa.ipc.open_file(FIXTURE / "capture" / f"{side}-all-inputs.arrow") as reader:
        return reader.read_all()


def _documents(table: pa.Table) -> list[dict[str, object]]:
    return [
        {
            name: value.isoformat() if isinstance(value, datetime) else value
            for name, value in row.items()
        }
        for row in table.to_pylist()
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


async def _wait_status(job, predicate):
    async def wait():
        while True:
            status = job.status()
            assert status["state"] == "running", status
            if predicate(status):
                return status
            await asyncio.sleep(0.005)

    return await asyncio.wait_for(wait(), 30)


def _held_status(status: dict[str, object]) -> None:
    expected = _json(FIXTURE / "reference/observations.json")["cut_before_checkpoint"]
    archived = expected["stream_joins"]["join"]
    assert status["stream_joins"]["join"] == {
        **archived,
        **{
            side: {
                **archived[side],
                "watermark_micros": WATERMARK,
                "idle": True,
                "ended": False,
            }
            for side in ("left", "right")
        },
    }
    assert status["watermark_micros"] == WATERMARK
    assert status["task_errors"] == 0


async def _continue(job, sources, sink, udf_sink, observations) -> None:
    sources["left"].continue_allowed = True
    await _wait_status(job, lambda _: len(sink.rows) == 3)
    sources["right"].continue_allowed = True
    await _wait_status(job, lambda _: len(sink.rows) == 8)
    sources["udf"].continue_allowed = True
    status = await _wait_status(
        job,
        lambda status: (
            len(udf_sink.rows) == 2
            and all(
                status["stream_joins"]["join"][side]["idle"]
                for side in ("left", "right")
            )
        ),
    )
    observations["continued_before_eof"] = status
    metrics = status["stream_joins"]["join"]
    assert metrics["emitted_match_rows"] == 13
    for side in ("left", "right"):
        assert metrics[side] == {
            "retained_rows": 5,
            "retained_bytes": 694,
            "evicted_rows": 0,
            "late_rows": 0,
            "late_affected_batches": 0,
            "max_lateness_micros": None,
            "null_event_time_rows": 0,
            "null_key_rows": 1,
            "watermark_micros": WATERMARK,
            "idle": True,
            "ended": False,
        }
    assert metrics["state_limit_failures"] == metrics["match_limit_failures"] == 0
    for source in sources.values():
        source.end_allowed = True
    outcome = await asyncio.wait_for(job.wait_async(), 30)
    assert outcome.state == "completed", outcome
    observations["outcome"] = {
        "state": outcome.state,
        "completed_epoch": outcome.completed_epoch,
    }
    observations["completed_status"] = job.status()


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


def _assert_held_outputs(events, sink, udf_sink) -> None:
    assert all(event["kind"] == "open" for event in events)
    assert sink.rows == udf_sink.rows == []


def _assert_closed_outputs(events, sink, udf_sink, outcome, mode: str) -> None:
    assert sink.lifecycle == udf_sink.lifecycle == ["open", "close"]
    assert sorted(event["side"] for event in events if event["kind"] == "close") == [
        "left",
        "right",
        "udf",
    ]
    assert outcome.state == ("cancelled" if mode == "migrate" else "completed")


async def _run_worker(mode: str, root: Path, observations, inputs) -> None:
    events: list[dict[str, object]] = []
    observations["source_events"] = events
    sources = {
        side: _ReplaySource(side, inputs["left" if side == "udf" else side], events)
        for side in ("left", "right", "udf")
    }
    sink, udf_sink = _RecordingSink(), _RecordingSink()
    observations["join_sink"], observations["udf_sink"] = sink.proof(), udf_sink.proof()
    if mode == "udf":
        _reject_udf_version(observations, events, sink, udf_sink)
        return
    runner = _create_runner(root, sources, sink, udf_sink)
    if mode == "corrupt":
        await _reject_corrupt_state(runner, observations, events, sink, udf_sink)
        return
    job = await asyncio.wait_for(runner.start_async(), 30)
    try:
        status = await _wait_status(job, lambda _: len(events) >= 3)
        observations["held_status"] = status
        _held_status(status)
        _assert_held_outputs(events, sink, udf_sink)
        if mode == "migrate":
            epoch = await asyncio.wait_for(job.trigger_checkpoint_async(), 30)
            observations["published_epoch"] = epoch
            assert epoch == 2
            observations["after_checkpoint"] = job.status()
            _held_status(job.status())
            _assert_held_outputs(events, sink, udf_sink)
        else:
            await _continue(job, sources, sink, udf_sink, observations)
    finally:
        outcome = await asyncio.wait_for(job.cancel_async(), 30)
        observations["cleanup"] = {
            "state": outcome.state,
            "completed_epoch": outcome.completed_epoch,
        }
        observations["join_sink"], observations["udf_sink"] = (
            sink.proof(),
            udf_sink.proof(),
        )
    _assert_closed_outputs(events, sink, udf_sink, outcome, mode)


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


def _segment(root: Path, descriptor) -> bytes:
    path = root / "state" / descriptor["relative_path"]
    data = path.read_bytes()
    assert len(data) == descriptor["byte_len"]
    assert _sha(path) == descriptor["sha256"]
    return data


def _header(data: bytes, magic: bytes, side: int) -> None:
    assert data[:8] == magic
    assert struct.unpack_from("<I", data, 8) == (2,)
    assert data[12:16] == bytes((side, 0, 0, 0))


def _assert_payload(root: Path, descriptors, payload, side: str) -> bytes:
    digest = payload["sha256"]
    data = _segment(root, descriptors[f"{side}-payload-{digest}"])
    assert hashlib.sha256(data).hexdigest() == digest
    _header(data, b"CFJPAY2\0", 0 if side == "left" else 1)
    rows, ipc_bytes = struct.unpack_from("<QQ", data, 16)
    assert rows == payload["rows"] == 3
    assert len(data) == payload["bytes"] == 56 + ipc_bytes
    assert struct.unpack_from("<QQQ", data, 32) == (0, 2, 3)
    with pa.ipc.open_stream(data[56:]) as reader:
        table = reader.read_all()
    original = _input(side)
    assert table.schema.equals(original.schema, check_metadata=True)
    assert _documents(table) == [_documents(original)[index] for index in (0, 2, 3)]
    return bytes.fromhex(digest)


def _assert_index(data: bytes, side: str, digest: bytes) -> None:
    _header(data, b"CFJIDX2\0", 0 if side == "left" else 1)
    assert struct.unpack_from("<QQ", data, 16) == (3, 0)
    original = _input(side)
    micros = original.column("at").cast(pa.int64()).to_pylist()
    keys = original.column("key").to_pylist()
    offset = 32
    for payload_row, (row_id, charge) in enumerate(((0, 140), (2, 140), (3, 134))):
        assert struct.unpack_from("<QqQ", data, offset) == (
            row_id,
            micros[row_id],
            charge,
        )
        assert data[offset + 24 : offset + 56] == digest
        actual_row, key_bytes = struct.unpack_from("<QQ", data, offset + 56)
        key = bytes((5,)) + struct.pack("<IIq", 0, 8, keys[row_id])
        assert (actual_row, key_bytes) == (payload_row, len(key))
        assert data[offset + 72 : offset + 72 + key_bytes] == key
        offset += 72 + key_bytes
    assert offset == len(data)


def _assert_migrated_cut(root: Path) -> None:
    old, new = _manifest(root, 1), _manifest(root, 2)
    assert new["epoch"] == 2
    for key in (
        "pipeline_name",
        "pipeline_fingerprint",
        "runtime_config_hash",
        "sinks",
    ):
        assert new[key] == old[key]
    assert new["sources"] == {
        source_id: {**source, "history": None}
        for source_id, source in old["sources"].items()
    }
    join = new["operators"]["join"]
    metadata = join["inline_metadata"]
    original = old["operators"]["join"]["inline_metadata"]
    assert metadata == {
        **original,
        "epoch": 2,
        "layout_version": 2,
        "v2_inventory": metadata["v2_inventory"],
    }
    assert join["progress"] == old["operators"]["join"]["progress"]
    inventory = metadata["v2_inventory"]
    assert inventory["codec_version"] == 2
    assert inventory["base_epoch"] == 1
    assert inventory["deltas"] == []
    assert [payload["side"] for payload in inventory["payloads"]] == ["left", "right"]
    descriptors = {item["segment_id"]: item for item in join["segments"]}
    assert set(descriptors) == {"left-base", "right-base"} | {
        f"{entry['side']}-payload-{entry['sha256']}" for entry in inventory["payloads"]
    }
    for payload in inventory["payloads"]:
        side = payload["side"]
        digest = _assert_payload(root, descriptors, payload, side)
        _assert_index(_segment(root, descriptors[f"{side}-base"]), side, digest)
    for source in new["sources"].values():
        assert source["sequence"] == 1
        assert source["cursor"]["order"] == "0000000000000004"
        assert source["cursor"]["payload"]["row_offset"] == 4
        assert source["ended"] is False


@pytest.fixture(scope="module")
def migrated_root(tmp_path_factory):
    _verify_fixture()
    output = tmp_path_factory.mktemp("authentic-v1-migration")
    root = output / "process-a-root"
    _copy_fixture_root(root)
    before = {
        path.relative_to(root): _sha(path) for path in root.rglob("*") if path.is_file()
    }
    observations = _launch("migrate", root, output / "process-a")
    assert observations["published_epoch"] == 2
    _assert_migrated_cut(root)
    assert all(_sha(root / path) == digest for path, digest in before.items())
    _verify_fixture()
    return root, output, observations


def test_authentic_v1_migrates_to_v2_and_recovers_on_a_fresh_process(migrated_root):
    root, output, first = migrated_root
    resumed = output / "process-b-root"
    shutil.copytree(root, resumed)
    second = _launch("resume", resumed, output / "process-b")
    assert first["pid"] != second["pid"] != os.getpid()
    reference = _json(FIXTURE / "reference/sink-proof.json")
    assert len(reference["rows"][:5]) == 5
    assert reference["rows"][:5] + second["join_sink"]["rows"] == reference["rows"]
    assert len(reference["rows"]) == 13
    assert second["join_sink"]["schemas"] == [reference["batches"][0]["schema"]] * 2
    udf = _json(FIXTURE / "reference/udf-output/sink-proof.json")["rows"]
    assert udf == [{"checked": value} for value in range(100, 106)]
    assert udf[:4] + second["udf_sink"]["rows"] == udf
    assert second["udf_sink"]["schemas"] == ["checked: int64"]
    _verify_fixture()
    capability = next(
        item
        for item in cf.Runtime().capabilities().operators
        if item.kind == "stream_join"
    )
    assert capability.state_version == 1
    assert capability.state_layouts == (1, 2)


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


def test_new_v2_corruption_rejects_without_falling_back_to_old_v1(migrated_root):
    root, output, _ = migrated_root
    corrupt = output / "corrupt-root"
    shutil.copytree(root, corrupt)
    segment = _manifest(corrupt, 2)["operators"]["join"]["segments"][0]
    path = corrupt / "state" / segment["relative_path"]
    data = path.read_bytes()
    path.write_bytes(bytes((data[0] ^ 1,)) + data[1:])
    assert _sha(path) != segment["sha256"]
    observations = _launch("corrupt", corrupt, output / "corruption-control")
    assert (
        "checkpoint lineage contains invalid recovery data"
        in observations["diagnostic"]
    )
    assert observations["category"] == "checkpoint_mismatch"
    assert observations["source_events"] == []
    assert observations["join_sink"]["rows"] == observations["udf_sink"]["rows"] == []
    assert _manifest(corrupt, 1) == _manifest(root, 1)
    _verify_fixture()


if __name__ == "__main__":
    _worker_main(
        sys.argv[1],
        Path(sys.argv[2]),
        Path(sys.argv[3]),
        Path(sys.argv[4]),
        sys.argv[5],
    )
