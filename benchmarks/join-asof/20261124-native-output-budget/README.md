# Rust ASOF watermark settlement with 512 pending rows

The final revision `cfce5309` had a pooled P50 of **59.397 µs** against
**65.9175 µs** on the baseline. The observed reduction of **9.89%** fell
short of the 10% goal. The paired result was `no-confirmed-regression`.
Its two intervals reached +2.07% and −4.67% at their upper ends.
[Final revision report and raw observations](final/README.md)

The two independently registered comparisons share the same baseline
executable. Each used 44 fresh workers and its own timing observations.
Borrowing the projection iterator changed the final optimized machine code.
The earlier `476431f3` result below applies to that specific binary.

## Earlier revision 476431f3

The median time to settle a watermark fell from **65.184 to 59.024 µs** on the
registered `balanced_512` workload. The pooled P50 reduction was **9.45%**.
The 10% target required a candidate P50 of at most 58.66515 µs. Both rounds
met the repository’s 5% improvement criterion.

| Native operator latency        | Baseline 8ce0b7c8 | Candidate 476431f3 |
|--------------------------------|------------------:|-------------------:|
| P50, 20 observations per build |         65.184 µs |          59.024 µs |
| P95, 20 observations per build |         76.357 µs |          63.642 µs |

| Round | Baseline P50 | Candidate P50 | Median paired change | Exact median interval |
|-------|-------------:|--------------:|---------------------:|----------------------:|
|     1 |    65.161 µs |     58.047 µs |               −8.56% |     [−25.37%, −7.65%] |
|     2 |    65.413 µs |     59.616 µs |               −9.35% |     [−11.76%, −7.66%] |

Each round had one excluded warmup per arm and ten adjacent pairs, alternating
AB and BA. Each observation launched a fresh process. The intervals use the
second and ninth sorted percentage changes for ten pairs, with 97.8515625%
coverage under the paired-median assumptions. Pooled P95 uses linear
interpolation. The repository classified the comparison as `improved`.

## Work and clock

The Rust bench admitted 512 right rows and 512 left rows for 32 balanced keys.
Both input frontiers then advanced to 2,000,000 µs. The clock covered
`StreamAsofJoinOperator::on_watermark`, including output budget accounting,
eight-column Arrow materialization, delivery to the in-process Rust collector,
committed state and right-side eviction. Fixture construction, admission,
checkpoint captures, process startup and full-row validation were outside the
reported formal latency. The operator used latest-per-key lookup, a ten-second
tolerance and a 10,000-row / 64 MiB output edge budget. The measured change
reused selected row indices across columns when accounting for the output.

Each of the 44 processes first ran a separate `ASOF_READY` workload that
checked all eight field values for 512 ordered output rows and verified the
output sequence. Each then ran one `ASOF_SAMPLE` that checked row count,
chunk bounds and state. All 88 workloads emitted one 512-row, 32,768-byte
batch, matched all 512 left rows, retained 32 right rows, evicted 480 and
reported zero state, workspace and output limit failures. The final output
watermark was 1,999,999 µs in both builds. All 40 formal samples reported
261 to 206 heap allocations and 187,948 to 136,652 cumulative allocated bytes
across builds. The thread allocation counter’s peak was 71,616 bytes in both.

The run covers full-width, all-matched settlement on a single native operator.
Separate cases are needed to measure a full graph, Python delivery, skew,
unmatched rows, restored state or a different projection.

## Host and source

Both binaries ran on Linux x86-64 under WSL2 on an Intel Core i9-13900HX.
The collector pinned workers to logical CPU 4 and monitored SMT sibling 5.
The three-second preflight and one one-second monitoring interval passed.
Sibling utilization was 0% in both intervals. Whole-host utilization was
3.53% and 3.75%, available memory stayed above 29 GiB, and both major faults
and swapped pages were zero. These are interval-level host observations.
The collector recorded 4.158 s, including preflight and the 44 workers, within
its 580 s timeout and the registered 600 s allowance. Cargo reported separate
optimized builds of 10m 41s and 7m 04s using cached dependencies.

The measured commits are `8ce0b7c8303918821fec24654c36f1d99113de21` and
`476431f371eda962471a57bd64eece36f0a8765b`. Both used the locked default
bench profile, Rust 1.94.0, LLVM 21.1.8 and the `benchmark-internals` feature.
The controller used CPython 3.13.9 and psutil 7.2.2. The raw source tree IDs,
binary and Cargo log digests, per-arm build profiles and common source hashes
are in [provenance.json](provenance.json) and [sources.sha256](sources.sha256).

## Reproduction

Each `WORKER` header in [workers.log](workers.log) gives the original filename
and SHA-256 of the two unaltered stdout lines that follow it. The file
contains the original timings, oracle flags, chunk, allocator, status and
checkpoint statistics for every warmup and formal observation. The original
[protocol](protocol.md), [resource observations](resources.jsonl) and
[outcome](outcome.json) accompany it.

To verify the saved observations, set `EVIDENCE_DIR` to this directory and
run the following from the repository root at either measured commit. This
command validates the raw worker logs and reproduces the original statistics.

```bash
PYTHONPATH=. python3 "$EVIDENCE_DIR/verify.py"
```

To repeat the workload, use a separate clean checkout with both revisions
and the Rust toolchain and Python dependency versions above. Keep the
evidence directory outside that disposable checkout. Start at its root.

```bash
git switch --detach 8ce0b7c8303918821fec24654c36f1d99113de21
mkdir -p target/observations
cp "$EVIDENCE_DIR/build.sh" "$EVIDENCE_DIR/measure.py" "$EVIDENCE_DIR/protocol.md" target/observations/
sha256sum -c "$EVIDENCE_DIR/sources.sha256"
sh target/observations/build.sh baseline
git switch --detach 476431f371eda962471a57bd64eece36f0a8765b
sh target/observations/build.sh candidate
PYTHONPATH=. python3.13 target/observations/measure.py 476431f371eda962471a57bd64eece36f0a8765b
```

The collector starts a new `resources.jsonl` and will stop if that file already
exists. Its local `identity.json` includes the host name. The public provenance
retains the original identity file digest and the CPU model and OS fields.
