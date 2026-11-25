"""Bind each managed worker's finalize trace to exact protocol request byte ranges."""
from __future__ import annotations

import json
import os
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from scripts.benchmark_suite.process import Worker


def trace_workers():
    original_start = Worker.start.__func__
    original_exchange = Worker._exchange

    async def start(cls, site, root, **kwargs):
        path = (root / "asof-finalize.jsonl").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb"):
            pass
        with patch.dict(os.environ, {"CALC_FLOW_ASOF_WATERMARK_TRACE": str(path)}):
            worker = await original_start(cls, site, root, **kwargs)
        worker.asof_trace_file = path
        worker.asof_trace_offset = 0
        return worker

    async def exchange(worker, message):
        file = worker.asof_trace_file
        begin = file.stat().st_size
        if begin != worker.asof_trace_offset:
            raise ValueError("unexpected ASOF activity between worker requests")
        result = await original_exchange(worker, message)
        with file.open("rb") as f:
            f.seek(begin)
            lines = f.read()
            end = f.tell()
        worker.asof_trace_offset = end
        recs = [json.loads(line) for line in lines.splitlines()]
        record = {"request_id": message["request_id"], "operation": message["operation"], "pid": worker.process.pid, "trace": file.name, "byte_start": begin, "byte_end": end, "records": len(recs)}
        with (file.parent / "trace-request-ranges.jsonl").open("a") as f:
            f.write(json.dumps(record) + "\n")
        if message["operation"] in ("sample", "prepare"):
            if not recs or not lines.endswith(b"\n"):
                raise ValueError("complete instrumented calculation returned no complete ASOF records")
            from diagnostic_validate import counts
            record["stages"] = counts(recs, worker.process.pid)
            if "warmup" in result:
                result = {**result, "warmup": {**result["warmup"], "trace_range": record}}
            else:
                result = {**result, "trace_range": record}
        elif recs:
            raise ValueError("ASOF finalize occurred outside a workload request")
        return result

    stack = ExitStack()
    stack.enter_context(patch.object(Worker, "start", classmethod(start)))
    stack.enter_context(patch.object(Worker, "_exchange", exchange))
    return stack
