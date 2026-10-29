"""Cross-revision recovery using the maintained static-input fixture."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
from test_static_stream_inputs import _array_declaration, _plan

import calc_flow as cf
from calc_flow.symbolic.analyzer import _Analyzer
from calc_flow.symbolic.types import TABLE_FIELD_TYPES

side = sys.argv[1]
root = Path("target/standalone-identity")
root.mkdir(exist_ok=True, parents=True)
state = root / "state"
if side == "candidate":
    state = root / "candidate-state"
    shutil.copytree(
        root / "state",
        state,
        ignore=shutil.ignore_patterns("manifest-00000000000000000002.json"),
    )
source_table = cf.table_input("events", schema=[cf.Field("x", "float64")])
expression = source_table.select(value=source_table["x"] + 1)
program = cf.Program("identity", engine="sql", outputs={"output": expression})
identities = {
    "source_digest": source_table.digest,
    "expression_digest": expression.digest,
    "program_fingerprint": program.fingerprint,
}
plan = _plan(_array_declaration())
identities["stream_fingerprint"] = plan.fingerprint

analyzer = _Analyzer("batch", frozenset({source_table.digest}), TABLE_FIELD_TYPES, True)
identities["source_lineage"] = str(analyzer.table(source_table._node, "input").lineage)
identities["expression_lineage"] = str(
    analyzer.table(expression._node, "output").lineage
)
if side == "probe":
    print(json.dumps(identities))
    raise SystemExit(0)


class Source:
    def __init__(self, pause=False):
        self.pause = pause
        self.offset = 0
        self.opened = False
        self.written = asyncio.Event()

    def capabilities(self):
        return cf.SourceCapabilities(
            cf.ReplayPositioning.EXACT_PAUSE_REPORT_AND_SEEK,
            cf.SourceDeliveryCapability.LOSSLESS,
            max_batch_rows=1,
            max_batch_bytes=1024,
            native_watermarks=cf.NativeWatermarkCapability.NEVER_EMITS,
        )

    async def open(self, cursor):
        self.opened = True
        self.offset = 0 if cursor is None else cursor.payload["offset"]
        self.resumed = self.offset

    async def next(self):
        if self.pause and self.offset == 1:
            await asyncio.sleep(0)
            return cf.Idle()
        if self.offset >= 2:
            return None
        self.offset += 1
        return cf.Data(
            cf.Batch.from_pyarrow(pa.table({"x": [float(self.offset + 1)]})),
            cf.Cursor(self.offset.to_bytes(1, "big"), {"offset": self.offset}),
        )

    async def close(self):
        pass


class Sink:
    def __init__(self, source):
        self.source = source
        self.scores = []

    async def open(self):
        pass

    async def write(self, batch):
        self.scores.extend(batch.to_pyarrow()["score"].to_pylist())
        self.source.written.set()

    async def close(self):
        pass


def runner(source, sink, weight=4.0):
    selected = _plan(_array_declaration())
    return cf.StreamingRunner(
        selected,
        {
            selected.source_binding_ids[0]: cf.SourceBinding(
                source, watermark_policy=cf.DisabledWatermarks()
            )
        },
        {"output": [cf.SinkBinding.ordinary("archive", sink)]},
        cf.ManagedCheckpointRuntime(state),
        static_inputs={
            "weights": cf.Batch.from_array(
                np.array([[weight]], dtype=np.float64), backend="numpy"
            )
        },
    )


async def run():
    source = Source(pause=side == "baseline")
    sink = Sink(source)
    job = await runner(source, sink).start_async()
    if side == "baseline":
        await asyncio.wait_for(source.written.wait(), 5)
        await job.trigger_checkpoint_async()
        await job.cancel_async()
        assert sink.scores == [8.0]
    else:
        assert (await job.wait_async()).state == "completed"
        assert source.resumed == 1
        assert sink.scores == [12.0]
        changed = Source()
        try:
            await runner(changed, Sink(changed), 5.0).start_async()
        except Exception as error:
            assert "static" in str(error).lower(), str(error)
            assert not changed.opened
        else:
            raise AssertionError("changed static weights accepted")
    identities["scores"] = sink.scores
    identities["resumed_offset"] = source.resumed


asyncio.run(run())
manifests = []
for path in state.rglob("*"):
    if not path.is_file():
        continue
    try:
        data = json.loads(path.read_text())
    except (UnicodeError, ValueError):
        continue
    if isinstance(data, dict) and "static_inputs" in data:
        manifests.append(
            {
                "path": str(path.relative_to(state)),
                "static_inputs": data["static_inputs"],
                "format_version": data.get("format_version"),
                "fingerprint": data.get("pipeline_fingerprint"),
            }
        )
identities["manifests"] = manifests
(root / f"{side}.json").write_text(json.dumps(identities, indent=2) + "\n")
if side == "candidate":
    base = json.loads((root / "baseline.json").read_text())
    for key in (
        "source_digest",
        "expression_digest",
        "program_fingerprint",
        "stream_fingerprint",
        "source_lineage",
        "expression_lineage",
    ):
        assert identities[key] == base[key], key
    assert manifests and base["manifests"]
    assert manifests[0]["static_inputs"] == base["manifests"][0]["static_inputs"]
print(json.dumps(identities))
