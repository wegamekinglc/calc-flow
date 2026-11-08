from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

import calc_flow as cf
import calc_flow.array as cfa


class _EndingSource:
    def __init__(self, opens: dict[str, int]) -> None:
        self.opens = opens

    def capabilities(self) -> cf.SourceCapabilities:
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=1024,
            native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor: object) -> None:
        self.opens["source"] += 1

    async def next(self) -> None:
        return None

    async def close(self) -> None:
        return None


class _Sink:
    def __init__(self, opens: dict[str, int]) -> None:
        self.opens = opens

    async def open(self) -> None:
        self.opens["sink"] += 1

    async def write(self, batch: cf.Batch) -> None:
        return None

    async def close(self) -> None:
        return None


def _runner(root: Path, opens: dict[str, int]) -> cf.StreamingRunner:
    runtime = cf.Runtime()
    cfa.register_native(runtime)
    plan = (
        cf.PipelineBuilder("native-sequence-recovery")
        .external(
            "sequence",
            "native",
            "array_sequence",
            "1",
            {
                "ir_version": 1,
                "input_dtype": "float64",
                "kernel": {"function": "ewma", "span": 3},
            },
        )
        .compile_stream(runtime=runtime)
    )
    return cf.StreamingRunner(
        plan,
        {
            "input": cf.SourceBinding(
                _EndingSource(opens), watermark_policy=cf.DisabledWatermarks()
            )
        },
        {"output": [cf.SinkBinding.ordinary("archive", _Sink(opens))]},
        cf.ManagedCheckpointRuntime(root),
    )


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _corrupt_layout_with_valid_checksums(root: Path) -> None:
    path = root / "manifests" / "manifest-00000000000000000001.json"
    manifest = json.loads(path.read_bytes())
    handle = manifest["operators"]["sequence"]["segments"][0]
    segment_path = root / "state" / handle["relative_path"]
    encoded = bytearray(segment_path.read_bytes())
    assert encoded[:8] == b"CFASEQ\0\0"
    encoded[8] = 2
    segment_path.write_bytes(encoded)
    handle["sha256"] = hashlib.sha256(encoded).hexdigest()
    state = {name: manifest[name] for name in ("sources", "operators", "sinks")}
    manifest["state_checksum"] = hashlib.sha256(_canonical(state)).hexdigest()
    path.write_bytes(_canonical(manifest))


def test_real_native_terminal_bad_codec_preserves_python_error_context_before_open(
    tmp_path: Path,
) -> None:
    opens = {"source": 0, "sink": 0}

    async def first_run() -> None:
        job = await _runner(tmp_path, opens).start_async()
        assert (await job.wait_async()).state == "completed"

    asyncio.run(first_run())
    assert opens == {"source": 1, "sink": 1}
    _corrupt_layout_with_valid_checksums(tmp_path)

    async def restore() -> None:
        with pytest.raises(cf.StreamingRuntimeError) as captured:
            await _runner(tmp_path, opens).start_async()
        error = captured.value
        assert error.category == "checkpoint_mismatch"
        assert error.component_kind == "operator"
        assert error.component_id == "sequence"
        assert "state.layout" in error.message

    asyncio.run(restore())
    assert opens == {"source": 1, "sink": 1}
