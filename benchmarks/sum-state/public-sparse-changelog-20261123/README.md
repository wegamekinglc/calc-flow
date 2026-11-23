# Public Python sparse changelog baseline

The workload uses the Rust grouped SUM example's sparse Arrow IPC fixture. Its
opening batch seeds 10,000 account/stock groups. The next 20 batches contain
100 input rows each. Both surfaces compute signed trade quantities and sum
them by account and stock. Python uses a fresh public TableExpr.stream with
checkpointing disabled and a compact-state budget of 10,000 groups.

## Registered collection

This is one reference build at main fced5062c1a6f05b6302674e8eda926182bb558a.
The optimized CPython 3.13 ABI3 Linux wheel comes from that commit's Linux CI
run 45069870259, artifact 14773009244. The collector verifies the loaded
Python files and native module against the wheel and checks the fixture
SHA-256 before reading it. The original optimized build took 21m 43s on CI.
Local collection reuses that artifact and an existing sparse.arrow.

The primary clock starts before declaring the public expressions and ends
after the async iterator drains and its context closes. It includes
compilation, native execution, public Arrow delivery, and job cleanup.
All tables survive until their validation after the timer. A separate Python
oracle accumulates signed trades in first-seen key order, checks the exact
Arrow schema, verifies old retract and new upsert values in order, replays the
returned changes, and checks the entire reconstructed state after every input.

Run two rounds of ten measured samples in one process, with one validated
warmup per round. Sample zero identifies each warmup. Report raw nanoseconds
and each round's median. A subsequent implementation comparison needs a fresh
contemporaneous baseline and candidate in alternating AB/BA order, 10 pairs
per round for two rounds, with the maintained paired median interval.

The single-arm collection budget is 600 seconds including resource checks,
loading, warmups, validation, startup, and cleanup. Use CPU 8 with one Tokio
and Rayon worker. Preflight checks CPU 8 and sibling 9 for at least 80% idle
over three seconds. Require at least 10 GiB of available RAM, less than 50%
mean host CPU use, fewer than 256 major faults and fewer than 1,024 total
swap pages in the interval. Independent builds may run on other cores. Record
host and sibling load and paging during collection. Stop on an oracle failure,
a budget overrun, or renewed host pressure. Retain completed samples.

## Reproduce

The benchmark needs Python 3.11 or newer, a compatible installed release
wheel, PyArrow, NumPy and psutil. The build record uses the fields documented
in [the latency diagnostic](../../README.md#open-event-time-stream-latency).
If the sparse fixture is unavailable, use the maintained generator to prepare
it before running the collector.

```bash
cargo run --locked --release -p calc-flow --example grouped_sum_state_benchmark \
  -- --prepare target/sum-fixtures
TOKIO_WORKER_THREADS=1 RAYON_NUM_THREADS=1 taskset -c 8 \
  timeout 600s python -m benchmarks.sparse_changelog \
    --fixture target/sum-fixtures/sparse.arrow --build release.json \
    --output target/sparse-changelog.json
```
