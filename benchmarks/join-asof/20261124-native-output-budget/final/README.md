# Final Rust ASOF watermark settlement on cfce5309

For 512 pending rows, the final candidate's pooled P50 was **59.397 µs** against
**65.9175 µs** on the baseline, a **9.89% reduction**. The registered 10% goal
required a candidate P50 of at most 59.32575 µs. The maintained paired rule
classified this comparison as `no-confirmed-regression`. A 5% improvement
requires both paired interval upper bounds to fall below −5%.

| Native operator latency | Baseline 8ce0b7c8 | Final cfce5309 |
|-------------------------|------------------:|---------------:|
| P50                     |        65.9175 µs |      59.397 µs |
| P95                     |       76.26855 µs |     69.2786 µs |

| Round | Baseline P50 | Final P50 | Median paired change | Exact median interval |
|------:|-------------:|----------:|---------------------:|----------------------:|
|     1 |    66.940 µs | 59.993 µs |               −9.97% |     [−19.86%, +2.07%] |
|     2 |    65.516 µs | 59.397 µs |               −9.35% |     [−11.16%, −4.67%] |

This registration compared the original sealed baseline executable with a new
build of the final commit. The final source borrows the projection iterator.
The earlier and final binaries have different SHA-256 and ELF `.text` digests,
and their optimized `output_charge` closures are respectively 0x15dd and
0x162d bytes. Each comparison used its own fresh baseline and candidate
observations. The previous revision's paired result is documented in the
[first report](../README.md).

## Measurement and checks

Each round started with one excluded warmup per arm, followed by ten
alternating adjacent AB/BA pairs. All 44 fresh workers ran the full-value
`ASOF_READY` oracle followed by one `ASOF_SAMPLE`. The 20 formal timings per
build cover the direct Rust watermark callback, output budget calculation,
eight-column Arrow result construction, delivery to an in-process collector,
commit and eviction. Source preparation, admission and the checkpoint captures
preceded or followed that clock. Each full-value oracle verified the output
sequence and all eight fields for every one of the 512 rows. Formal samples
retained the count, budget and state checks.

All 88 workloads emitted a single 512-row batch of 32,768 bytes, retained 32
right rows, evicted 480 and reported zero state, workspace or output limit
failures. The output watermark reached 1,999,999 µs. Across the 40 formal
samples, baseline and final allocations were respectively 261 and 206 per
callback. Cumulative allocated bytes were 187,948 and 136,652. Peak allocated
bytes were 71,616 on both builds. These observations cover this full-width,
all-matched, 32-key fixture and this direct Rust operator clock.

Intervals are the second through ninth order statistics of ten per-pair
percentage changes in each round, with 97.8515625% coverage under the
repository's independent-pair assumptions. Pooled P95 uses linear interpolation.

## Provenance and reproduction

The baseline was `8ce0b7c8303918821fec24654c36f1d99113de21`, executable
SHA-256 `59548f6c7c6340cd01f78b66bd97604c260c1488b09c5aab41f3ead2ecdaa31f`.
The final candidate was `cfce5309d11582e9e57d6cc7da779ab279762433`, tree
`78198d3bd7ed6bad10e052667fe4d8961bcb6e3d`, executable SHA-256
`7f1a807eaab735bc3d3f33a68e42f662be865c5f7675875e18fd65b6f2a90a4a`.
Cargo reported 6m 53s for the final optimized build. The original baseline
build took 10m 41s. Both used the locked default Cargo bench profile with
`benchmark-internals`, Rust 1.94.0 and LLVM 21.1.8. CPU, compiler settings,
artifact hashes and the two closure digests are in [provenance.json](provenance.json).

Workers ran on logical CPU 4 of an Intel Core i9-13900HX under Linux WSL2.
SMT sibling 5 showed 0% busy in both resource intervals. The three-second
preflight recorded 0.297% whole-host utilization, six major faults and seven
swapped pages. The one-second collection interval recorded 0.725%, three
major faults and six swapped pages. Available memory exceeded 29 GiB in each
interval and both passed the registered gates. The collector took 4.031 s
under a 500 s outer deadline. [Resource log](resources.jsonl) · [Outcome](outcome.json)

The original [final registration](protocol.md) inherits the
[workload protocol](common-protocol.md). Each `WORKER` header in [workers.log](workers.log)
records the raw filename and SHA-256 of the following two original stdout
lines. To verify this comparison offline, set `EVIDENCE_DIR` to this `final`
directory and run the following from the repository root at either measured
commit. The verifier checks all 88 worker responses, reconstructs the original
partial-observation file digest and reproduces the final comparison statistics.

```bash
PYTHONPATH=. python3 "$EVIDENCE_DIR/verify.py"
```

To reproduce this single registration, use a separate clean checkout
containing both commits and keep the evidence directory outside it. The
collector used CPython 3.13.9 and psutil 7.2.2. Start at the checkout root.

```bash
git switch --detach 8ce0b7c8303918821fec24654c36f1d99113de21
mkdir -p target/observations target/final-observations
cp "$EVIDENCE_DIR/build.sh" "$EVIDENCE_DIR/measure.py" target/observations/
cp "$EVIDENCE_DIR/common-protocol.md" target/observations/protocol.md
sha256sum -c "$EVIDENCE_DIR/sources.sha256"
sh target/observations/build.sh baseline
git switch --detach cfce5309d11582e9e57d6cc7da779ab279762433
sh target/observations/build.sh candidate
cp target/observations/{baseline,candidate}{,.sha,.rustc,.digest} target/final-observations/
cp "$EVIDENCE_DIR/measure.py" target/final-observations/
cp "$EVIDENCE_DIR/protocol.md" target/final-observations/final-protocol.md
PYTHONPATH=. timeout 500 python3.13 target/final-observations/measure.py cfce5309d11582e9e57d6cc7da779ab279762433
```
