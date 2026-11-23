# Quiet Python program results

This comparison times the public, synchronous Python session interface with 32
declared outputs. It records one quiet watermark advance and one data-bearing
push per iteration. Each fresh process validates both results. Registration pins
the Git trees, uploaded artifacts, session module and measurement sources.

To reproduce on Linux x86-64 with CPython 3.13, PyArrow, psutil, GitHub CLI
access to the two CI artifacts, uv, readelf and taskset, run the following from
the repository root in a fresh checkout.

```sh
PYTHONPATH=. timeout --kill-after=8s 588s python3.13 benchmarks/stream-session/20261124-quiet-program-results/collect.py
```

The collector writes into a new local `acquisition/` directory. The published
run is under `recorded/`. Each JSONL worker record identifies its round, pair,
product and 64 individual calls per workload in nanoseconds. Pair zero is the
excluded warmup. Resource observations cover preflight and the sampling window.
Release manifests identify the independently built and verified wheels.

The registered run stopped after its first warmup pair, during the first formal
baseline worker. Preflight passed. The next interval recorded 302 major faults
on the host, exceeding the maintained limit of 255. Both warmups checked every
Arrow output; neither contributes to the latency estimate. There are zero
completed formal pairs, so neither workload has a p50, p95 or comparison verdict.
