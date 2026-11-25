"""Associate each completed worker request with its immutable trace byte range."""

import json
import os
from unittest.mock import patch

from scripts.benchmark_suite.process import Worker


def enable():
    original = Worker.start.__func__
    exchange = Worker._exchange

    async def start(cls, site, root, **kwargs):
        path = (root / "phases.jsonl").resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=False)
        with patch.dict(
            os.environ,
            {
                key: str(path)
                for key in (
                    "CALC_FLOW_ASOF_ADMISSION_TRACE",
                    "CALC_FLOW_ASOF_WATERMARK_TRACE",
                )
            },
        ):
            return await original(cls, site, root, **kwargs)

    async def bound(worker, message):
        path = worker.root / "phases.jsonl"
        begin = path.stat().st_size
        result = await exchange(worker, message)
        end = path.stat().st_size
        if message["operation"] in ("prepare", "sample"):
            if not end > begin:
                raise ValueError("missing trace for completed workload")
            (
                worker.root / f"trace-request-{message['request_id']:04d}.json"
            ).write_text(json.dumps(dict(start=begin, end=end, pid=worker.process.pid)))
        elif end != begin:
            raise ValueError("trace outside measured worker request")
        return result

    Worker.start = classmethod(start)
    Worker._exchange = bound
