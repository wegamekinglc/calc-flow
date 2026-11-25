"""Pin the compare controller before importing benchmark modules or extensions."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def selected_cores(allowed):
    distinct = {}
    for cpu in sorted(allowed):
        mask = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings")
        distinct.setdefault(mask.read_text().strip(), cpu)
    cores = list(distinct.values())[:4]
    if len(cores) < 2:
        raise RuntimeError("measurement requires at least two distinct physical cores")
    return cores


def pin_and_exec():
    if not sys.flags.no_site or len(list(Path("/proc/self/task").iterdir())) != 1:
        raise RuntimeError("launcher must be single-threaded with site imports disabled")
    allowed = sorted(os.sched_getaffinity(0))
    selected = selected_cores(allowed)
    os.sched_setaffinity(0, set(selected))
    if sorted(os.sched_getaffinity(0)) != selected:
        raise RuntimeError("controller affinity differs from the physical-core selection")
    output = Path(os.environ["GITHUB_WORKSPACE"]) / "target/evidence/early-affinity.json"
    output.write_text(json.dumps(dict(allowed=allowed, selected=selected, pid=os.getpid())))
    os.execv(sys.executable, [sys.executable, *sys.argv[1:]])


if __name__ == "__main__":
    pin_and_exec()
