"""Check benchmark-only cgroup accounting, idle sibling cores, paging and free RAM."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import psutil

from scripts.sparse_changelog_ci import _paging, _siblings, interval


def select_cpus():
    allowed = sorted(os.sched_getaffinity(0))
    groups = {tuple(sorted((cpu, *_siblings(cpu)))) for cpu in allowed}
    cores = sorted(min(set(group).intersection(allowed)) for group in groups)[:4]
    if not 2 <= len(cores) <= 4:
        raise ValueError("expected 2 to 4 independent cores for 4 workers")
    return cores


def owned_cpu():
    name = os.environ["ASOF_CGROUP"]
    if Path("/proc/self/cgroup").read_text().strip() != "0::/" + name:
        raise RuntimeError("measurement is outside its dedicated cgroup")
    stats = (Path("/sys/fs/cgroup") / name / "cpu.stat").read_text()
    return int(dict(line.split() for line in stats.splitlines())["usage_usec"])


def cgroup_masks():
    group = Path("/sys/fs/cgroup") / os.environ["ASOF_CGROUP"]
    masks = set()
    for tid in (group / "cgroup.threads").read_text().split():
        try:
            masks.add(tuple(sorted(os.sched_getaffinity(int(tid)))))
        except ProcessLookupError:
            continue
    return sorted(masks)


async def observe_cpus(output, selected=None):
    preflight = selected is None
    cores = select_cpus() if preflight else selected
    before = _paging()
    psutil.cpu_percent(percpu=True)
    started = time.monotonic()
    owned_before = owned_cpu()
    await asyncio.sleep(3 if preflight else 1)
    busy = psutil.cpu_percent(percpu=True)
    elapsed = time.monotonic() - started
    self_percent = (owned_cpu() - owned_before) / (elapsed * 10000 * len(busy))
    host_percent = sum(busy) / len(busy)
    external = max(0, host_percent - self_percent)
    free = psutil.virtual_memory().available
    paging = (before, _paging())
    rows = []
    for cpu in cores:
        detail = interval(
            paging, busy, free, (cpu, *_siblings(cpu)), preflight=preflight
        )
        if not preflight:
            # The cgroup includes exited sample children. Its selected physical cores
            # are distinct, leaving every SMT sibling outside the measured workload.
            detail["suitable"] = (
                external < 50
                and detail["sibling_busy"] < 50
                and free >= 10 * 1024**3
                and detail["major_faults"] < 256
                and detail["swap_pages"] < 1024
            )
        detail.update(owned_host_percent=self_percent, external_host_percent=external)
        rows.append(detail)
    masks = cgroup_masks()
    with (output / "resources.jsonl").open("a") as stream:
        stream.write(
            json.dumps(dict(epoch=time.time(), preflight=preflight, cores=rows, thread_masks=masks)) + "\n"
        )
    if not masks or any(not set(mask).issubset(cores) for mask in masks):
        raise RuntimeError(f"measuring cgroup has threads outside selected CPUs: {masks}")
    if not all(row["suitable"] for row in rows):
        raise RuntimeError(f"host resource envelope failed: {rows}")
    return cores
