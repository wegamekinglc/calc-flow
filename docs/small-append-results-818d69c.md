# Small-append streaming follow-up

The small-row latency breakthrough is **not established**. This follow-up
reduces resident-entity allocations and repeated Python attachment, but most
small-append confidence intervals still include no improvement. Larger
increments benefit more clearly. No default scheduler setting is changed.

## Measured identities and boundaries

Clean release baseline: `6515f4d19317f6b05573f41078177d866cc1eee1`.
Clean release candidate: `818d69ce0ff129828179bc8b46ee24cded92c326`.
The candidate contains the callback/state changes and opt-in diagnostics.
Later harness/documentation commits do not change its engine source.

The machine was WSL2 Linux x86-64, Python 3.13.9, NumPy 2.5.1 and
PyArrow 24.0.0, with 32 visible logical CPUs. Compiler, dependency, wheel,
native-module, affinity and harness fingerprints are retained in the raw data.
No local builds/tests or other benchmark runs overlapped measurements.

Both versions use the actual Python `StreamingRunner`: timing includes
preconstructed Data enqueue, source/task/channel work, rolling, watermark
finalization, projection, sink and `to_pyarrow`. Compilation, startup, history,
input construction, validation, IPC, checkpoint and shutdown are excluded.
Forced pre-sample GC and normal GC remain separate; GC is never disabled.
Each case owns a fresh process pair. Cases are shuffled with a recorded seed,
and samples alternate AB/BA while both runners advance through identical rows.

The initial engine matrices used the harness at `818d69c`; scheduler experiments
used the harness at `6e27ae3`; the declared follow-up used `60dc39d`.
Subsequent portability fixes use integer UTC
timestamps outside timing and an independently owned test directory. The
current seeded permutation uses NumPy; the original artifacts retain their
earlier harness hashes and actual case order. Reproduce historical numbers with
their recorded harness revision, not an unrecorded replacement.

## Engine comparison: original complete-tick workload

At 1,024,000 warm rows, 64 resident entities and forced pre-sample collection:

| Append | SMA P50 before / after ms | Paired speedup, 95% CI | Dual SMA P50 before / after ms | Paired speedup, 95% CI |
| ------ | ------------------------- | ---------------------- | ------------------------------ | ---------------------- |
| 64     | 1.147 / 1.056             | 1.028 (0.988–1.205)    | 1.238 / 1.146                  | 1.027 (0.947–1.120)    |
| 640    | 1.385 / 1.360             | 1.012 (0.983–1.084)    | 1.408 / 1.329                  | 1.018 (0.982–1.078)    |
| 6,400  | 2.263 / 2.029             | 1.111 (1.085–1.147)    | 2.312 / 2.070                  | 1.116 (1.076–1.141)    |
| 64,000 | 9.880 / 8.561             | 1.123 (1.111–1.189)    | 11.315 / 9.940                 | 1.163 (1.132–1.192)    |

Paired speedup is the median of paired ratios, **not** the ratio of the two
independent P50s. Intervals use 20,000 paired bootstrap resamples. These are
per-case, unadjusted intervals, not a family-wide statistical guarantee.

All 28 original matrix cases and 1,680 measured appends pass identity, order
and numerical checks. The maximum absolute error is
`3.694822225952521e-13`. Eight case intervals favor the candidate; none establish
a regression. This does not mean the remaining cases establish equivalence.
The original historical-depth range, normal-GC data, tails, CVs and every
sample are in [the complete raw matrix](../benchmarks/rolling/small-full-tick-818d69c.json).

Absolute latencies differ from the previous two-round report. Do not compute
a speedup by dividing a new number by that report's old median: worker
lifetime, case ordering and measurement period differ.

## Engine comparison: one to 64 sparse appended rows

History still contains 64 entities. The table below cycles increments over a
pool of 64 entities; an N-row batch touches N entities for these values of N.
Each appended row has its own increasing timestamp, making partial ticks and
their watermarks legal. This is separate from the complete-tick workload.
Warm history is 1,024,000 rows; collection is forced before each sample.

| Append | SMA P50 before / after ms | Paired speedup, 95% CI | Dual SMA P50 before / after ms | Paired speedup, 95% CI |
| ------ | ------------------------- | ---------------------- | ------------------------------ | ---------------------- |
| 1      | 0.869 / 0.890             | 0.934 (0.842–1.131)    | 0.882 / 0.861                  | 1.014 (0.943–1.144)    |
| 4      | 1.029 / 0.962             | 1.055 (0.984–1.120)    | 0.951 / 0.956                  | 1.074 (0.944–1.197)    |
| 16     | 0.933 / 0.874             | 1.038 (0.963–1.187)    | 1.079 / 1.022                  | 1.034 (0.993–1.077)    |
| 64     | 1.100 / 1.032             | 1.065 (0.937–1.145)    | 1.138 / 1.033                  | 1.088 (0.958–1.126)    |

None of these eight representative intervals establishes a speedup. In
particular, fewer input rows do not remove a full source/watermark/sink round.

The full sparse matrix also uses entity pools of 1, 4 and 16, both indicators
and both GC modes: 64 cases, 3,840 correct measured appends, maximum absolute
error `5.115907697472721e-13`. Thirteen initial intervals favor the candidate;
one favors the baseline. All cases, including unfavorable results, remain in
[the raw sparse matrix](../benchmarks/rolling/small-sparse-818d69c.json).

### Declared regression follow-up

The first-pass negative signal is dual SMA, N=1, entity pool 16, forced GC:
paired speedup `0.960`, interval `0.919–0.994`. Its P50 changes from 1.020 ms to
1.085 ms, while P95 improves from 1.423 ms to 1.290 ms. Neither statistic is
discarded. An additional 100-pair run is selected for N=1/4, pool 16, both
indicators and both GC modes; it includes neighboring controls and is not a
repeated-until-green single-case test.

The declared follow-up did not reproduce an established regression: the
flagged case measured 0.888 / 0.881 ms P50 and paired speedup
`0.994 (0.954–1.068)`. Of the eight follow-up cases, three normal-GC intervals
favor the candidate; none establishes a regression. All 1,600 appends pass,
with maximum absolute error `3.126388037344441e-13`. This does not prove
equivalence or erase the initial result. See
[the complete follow-up](../benchmarks/rolling/small-recheck-818d69c.json).

| Indicator | Append | GC     | P50 before / after ms | Paired speedup, 95% CI |
| --------- | ------ | ------ | --------------------- | ---------------------- |
| SMA       | 1      | forced | 0.903 / 0.868         | 1.058 (0.990–1.140)    |
| SMA       | 1      | normal | 0.685 / 0.660         | 1.068 (1.019–1.096)    |
| SMA       | 4      | forced | 0.959 / 0.910         | 1.017 (0.971–1.101)    |
| SMA       | 4      | normal | 0.718 / 0.667         | 1.062 (1.034–1.092)    |
| Dual SMA  | 1      | forced | 0.888 / 0.881         | 0.994 (0.954–1.068)    |
| Dual SMA  | 1      | normal | 0.698 / 0.671         | 1.037 (0.997–1.071)    |
| Dual SMA  | 4      | forced | 0.935 / 0.904         | 1.014 (0.953–1.106)    |
| Dual SMA  | 4      | normal | 0.683 / 0.640         | 1.058 (1.029–1.093)    |

## Why the small-row floor remains

The allocation regression proves that resolving 64 resident entity IDs needs
one result allocation instead of seven allocations. Source polling now creates
its callback, checks awaitability and submits it under one Python attachment;
source result decoding remains separate. Sink batch wrapping, invocation and
scheduling preparation also share one attachment. Cancellation, task factories,
context isolation, GC roots, finality and checkpoints retain their semantics.

Those changes do **not** remove the serial cross-thread handshake. Separate
instrumented 30-sample diagnostics at H=1,024,000, N=64 show these request-level
median wall intervals, in microseconds:

| Callback    | Attach | Prepare | Loop dispatch wait | Rust resume / cleanup |
| ----------- | ------ | ------- | ------------------ | --------------------- |
| source.next | 3.33   | 14.40   | 66.45              | 81.96                 |
| sink write  | 1.79   | 13.20   | 54.18              | 49.91                 |

The source callback lifetime itself can contain prearmed input waits, GC and
IPC, so it is intentionally omitted from this table. These per-request medians
are neither additive nor CPU attribution. Instrumentation is off for all
engine/configuration comparisons. See
[full-tick diagnostics](../benchmarks/rolling/callback-full-818d69c.json) and
[one-row diagnostics](../benchmarks/rolling/callback-sparse-818d69c.json).

In the uninstrumented representative complete-tick SMA case, post-warm through
terminal operator counters average about 164.4 µs for rolling watermark work,
27.1 µs for other rolling processing, and 7.2 µs for projection processing per
measured append. These counters include untimed terminal control handling and
are not exact per-sample timings. The timed `to_pyarrow` P50 is 65.2 µs and is
already contained in sink-to-receive; it must not be added twice.

Optimizing projection arithmetic alone cannot explain a millisecond-scale
improvement. Physical projection fusion might also remove a task/channel, but
that extra cost is not isolated by these counters. This follow-up therefore
does not implement fusion or promise its gain without per-edge evidence and
preservation of logical metrics, checkpoint ownership and branched graphs.

## Same-wheel scheduler experiments

These are separate configuration comparisons using the **identical candidate
wheel** on both sides, not additional engine speedups. Each case has H=1,024,000,
N=64 complete ticks and 30 AB/BA pairs. Declared Tokio worker overrides are
checked in child fingerprints; all other environment fields must match.

| Tokio workers | SMA forced-GC paired gain | Dual SMA forced-GC paired gain |
| ------------- | ------------------------- | ------------------------------ |
| 32 → 1        | 1.020 (0.941–1.106)       | 1.106 (1.022–1.186)            |
| 32 → 4        | 1.084 (1.013–1.155)       | 0.979 (0.905–1.036)            |

Normal-GC intervals include no improvement in all four cases. Overall there
is no consistent winner across indicators and modes. All 480 measured appends
are correct. Neither default Tokio workers nor SQL/DataFusion parallelism is
changed. Raw data: [32 versus 1](../benchmarks/rolling/scheduler-32-to-1-818d69c.json),
[32 versus 4](../benchmarks/rolling/scheduler-32-to-4-818d69c.json).

## Follow-on decision

Keep the proven allocation reduction and bounded diagnostic coverage. The
small-append latency goal remains unmet; do not relabel normal-GC numbers or
configuration experiments as its achievement. The next design should target
the complete serialized source/Data/watermark/sink exchange, with per-edge and
cursor-decoding measurements first. Native source protocols, atomic message
grouping or a different runner contract require their own semantic review;
none is smuggled into this performance patch.

For workload semantics and reproduction commands, see
[the benchmark guide](warm-stream-performance.md).

## Integrity and validation

Across the two engine matrices, two scheduler comparisons and declared
follow-up, all 7,600 uninstrumented measured appends pass. Both standalone
diagnostic runs also pass all 60 measured append checks. These counts exclude
historical warmup and do not combine the workloads into one speedup estimate.

Local validation includes the complete Rust harness (core integration/example
targets, connectors and benchmark smoke), 112 PyO3 unit tests in each of three
serial runs, 962 Python/workload tests, and 23 controller tests. Strict
allocation, callback cancellation/context/factory/GC, independent sparse
oracle and timezone-database-free regressions are included. Remote final-head
CI/coverage status is maintained in PR #242; local checks alone do not certify
that external state.

SHA-256 fingerprints of the raw evidence:

```text
small-full-tick-818d69c.json
171c4786b5a6b27d1d8bfc625a79087d1b56a7e7550186fb607cef02f4130c49
small-sparse-818d69c.json
27f1f75bcd1fbb98cfb6fc86c81aea376088ef2635725e3b0e42057533eca34f
small-recheck-818d69c.json
44d38319b851d400ba1eca8415c3f34a2c44c2b25ff50a4236b5f4a537168461
scheduler-32-to-1-818d69c.json
a3cbfbfbf56c757cc949c611b96d05c832fd2aebb99e49265cdac86433c5dc53
scheduler-32-to-4-818d69c.json
09b4bb9f4025fce1747ea635d3c21c40fc16d96f970654444311525d2a4b6972
callback-full-818d69c.json
4b125a2dacb4b3b9594b7a10a94a86f8aff17a7a723d78edabcd6d2a135f625f
callback-sparse-818d69c.json
19b57cf816a4ee2bfaf32d030782e7d490eb2c8abdfa6e641d654ebea9a9c025
```
