# Warm-state optimization results: 39a0c7f

Measured on 2026-09-05, after both optimization rounds. These are measured
results, not the pre-implementation forecasts. This report and its raw data
are added in a report-only follow-up to the measured code commit.

## Provenance and boundary

- Baseline: `9044501f99598b83f38ec37e60756c37b7bf6fb7`.
- Candidate: `39a0c7f640ba9a26cca313f722dea389d6f86d17`.
- Both checkouts were clean when built and checked before measurement.
  Both use release Maturin builds, Rust 1.88.0, the same Cargo.lock, Python
  3.13.9, NumPy 2.5.1, and PyArrow 24.0.0.
- The same WSL2 Linux machine exposes 32 logical CPUs; both workers have
  affinity 0–31. This is not a claim that one incremental rolling stream
  computes on 32 threads in parallel.
- The controller alternates AB/BA over 30 paired samples per case, with
  28 cases: seven history/increment points, two indicators, and two GC modes.
  No concurrent local compilation, tests, or other agent-run benchmarks were
  active during this matrix.
- Each worker owns a persistent real StreamingRunner with 64 entities,
  SMA(20) or SMA(5)-SMA(20), and an advancing row sequence. Timing covers
  prepared Data enqueue through source/task/channel, rolling, watermark
  finalization, projection, sink, and `to_pyarrow`. Compilation, startup,
  history warmup, validation, worker IPC, GC collection, and shutdown are
  outside timing.
- Source and sink are in-process benchmark adapters, not Kafka/database/network
  I/O. Their effective delivery is best effort. Scheduled checkpoints are
  outside the timed interval; this experiment does not benchmark durable
  delivery, checkpoint throughput, or recovery latency.

The [raw JSON](../benchmarks/rolling/warm-stream-39a0c7f.json) contains all
1,680 measured append samples, phase intervals, initial/final runtime metrics,
terminal states, dependency fingerprints, harness hashes, and full wheel/native
build identities. Its SHA-256 is
`02040a5c789647758ce4cb16bebd95fe789730c9c14b19167bf986b92cbfcb5a`.
See [reproduction instructions](warm-stream-performance.md#reproduce).

Every measured append passes strict identity/order and numerical checks
(`rtol=atol=1e-10`, equal NaNs). The maximum absolute error across both builds
and indicators is `3.694822225952521e-13`; all 56 scenarios end as
`completed`. These checks do not substitute for the separate checkpoint and
recovery regression suite.

## Results and interpretation

At 1,024,000 history rows with forced pre-sample collection, SMA P50 falls
from 1.598 to 0.931 ms for 64 appended rows, and from 123.973 to 8.810 ms for
64,000 rows. Dual SMA falls from 1.567 to 0.908 ms and from 127.867 to
9.782 ms respectively. The median paired speedups below are medians of
per-pair ratios, not ratios of independently computed P50s.

All 28 paired-speedup bootstrap intervals favor the candidate; the smallest
lower bound is 1.359x. This is evidence for this matrix on this machine, not a
universal performance guarantee. P95 uses only 30 samples per case and is
less stable than the median. No second-run or cross-machine repeat is claimed.

The large-batch gain is concentrated in the watermark-to-sink interval: for
64,000-row SMA its P50 falls from 123.250 to 8.108 ms. That interval includes
rolling finalization, projection, channels, scheduling, and sink dispatch;
it is not a CPU-only rolling measurement.

For 64-row SMA, the candidate still spends 0.373 ms between source Data and
source watermark callbacks, and 0.444 ms from watermark callback to sink entry.
The separately recorded candidate operator-counter deltas average 18.2 us
for rolling Data handling, 159.6 us for watermark handling, and 5.77 us for
projection handling per append. These are aggregate wall-time means across
the advancing run, not per-sample P50s or disjoint CPU buckets. Older baseline
processing counters omit watermark handling, so their totals are not directly
comparable. No edge blocked-duration increase is recorded for these fixed
1,024,000-row-history cases.

This supports two conclusions: bulk row/state work was the main large-batch
cost, and the remaining small-batch latency is substantially outside the
rolling arithmetic itself. Further CPU-level attribution would need a separate
trace/profile; the callback intervals alone cannot assign exact scheduler,
queue, allocator, and Python bridge percentages.

## Forecast reconciliation and limitations

| Workload, forced GC at H=1,024,000 | Provisional P50 budget | Measured P50 | Outcome             |
|------------------------------------|------------------------|--------------|---------------------|
| Round 1, SMA append 64             | <= 1.3 ms              | 0.931 ms     | Reached             |
| Round 1, SMA append 64,000         | <= 60 ms               | 8.810 ms     | Reached             |
| Round 1, dual SMA append 64,000    | <= 65 ms               | 9.782 ms     | Reached             |
| Full rounds, SMA append 64         | 0.42–0.73 ms           | 0.931 ms     | Missed upper target |
| Full rounds, SMA append 64,000     | 16.4–32.7 ms           | 8.810 ms     | Faster than range   |

The full-round small-batch forecast was too optimistic for this complete
Python StreamingRunner boundary: the measured result is about 28% above
0.73 ms. Both rounds are implemented; this missed forecast is not hidden by
switching to normal GC or a standalone kernel benchmark.

Normal-GC SMA at H=1,024,000 and N=64 is 1.143 -> 0.653 ms. Compare normal-GC
builds with each other, not the candidate's normal-GC result with the
baseline's forced-GC result. Collection happens before timing but changes
the state from which the timed append begins. Normal mode still includes
untimed controller IPC gaps and is not a saturation-throughput experiment.

There are residual costs and local counter-regressions despite the total gain.
For 64-row SMA, sink-to-receive P50 increases from 0.063 to 0.083 ms while
conversion stays near 0.049 ms. The callback boundaries do not establish a
cause for that increase. The candidate is also not history-size invariant:
forced-GC SMA N=64 rises from 0.931 ms at H=1,024,000 to 1.363 ms at
H=10,240,000. Neither effect should be described as eliminated overhead.

The benchmark covers ordered, non-null quote inputs and small fixed windows.
Unordered/lateness/duplicate/null/schema/output-budget/checkpoint cases remain
correctness-test coverage, not measured fast-path performance claims. Real
connectors, backpressure, wider schemas, many more resident entities, and
larger windows can have different bottlenecks.

## Full matrix

| Indicator       | History    | Append | Forced GC | Baseline P50 / P95 ms | Candidate P50 / P95 ms | Paired speedup (95% CI) |
|-----------------|------------|--------|-----------|-----------------------|------------------------|-------------------------|
| rolling_mean    | 10,240     | 64     | True      | 1.573 / 1.977         | 0.822 / 1.046          | 1.79x (1.65–1.98)       |
| rolling_mean    | 10,240     | 64     | False     | 1.109 / 1.336         | 0.611 / 0.790          | 1.79x (1.72–1.87)       |
| rolling_mean    | 102,400    | 64     | True      | 1.518 / 2.243         | 0.822 / 1.054          | 1.88x (1.60–2.14)       |
| rolling_mean    | 102,400    | 64     | False     | 1.061 / 1.210         | 0.590 / 0.683          | 1.91x (1.76–1.97)       |
| rolling_mean    | 1,024,000  | 64     | True      | 1.598 / 1.835         | 0.931 / 1.121          | 1.71x (1.48–1.93)       |
| rolling_mean    | 1,024,000  | 64     | False     | 1.143 / 1.427         | 0.653 / 0.793          | 1.74x (1.67–1.86)       |
| rolling_mean    | 1,024,000  | 640    | True      | 2.524 / 2.695         | 1.140 / 1.305          | 2.17x (2.03–2.33)       |
| rolling_mean    | 1,024,000  | 640    | False     | 2.085 / 2.261         | 0.879 / 1.050          | 2.34x (2.26–2.44)       |
| rolling_mean    | 1,024,000  | 6,400  | True      | 12.927 / 14.023       | 1.950 / 2.179          | 6.70x (6.59–6.88)       |
| rolling_mean    | 1,024,000  | 6,400  | False     | 14.238 / 16.647       | 2.051 / 2.545          | 7.09x (6.67–7.38)       |
| rolling_mean    | 1,024,000  | 64,000 | True      | 123.973 / 128.418     | 8.810 / 9.601          | 14.16x (13.95–14.28)    |
| rolling_mean    | 1,024,000  | 64,000 | False     | 125.039 / 128.874     | 8.753 / 9.238          | 14.31x (14.21–14.51)    |
| rolling_mean    | 10,240,000 | 64     | True      | 2.096 / 2.354         | 1.363 / 1.542          | 1.50x (1.43–1.63)       |
| rolling_mean    | 10,240,000 | 64     | False     | 1.604 / 1.985         | 1.097 / 1.374          | 1.47x (1.42–1.55)       |
| dual_sma_spread | 10,240     | 64     | True      | 1.555 / 1.752         | 0.906 / 1.035          | 1.69x (1.55–1.94)       |
| dual_sma_spread | 10,240     | 64     | False     | 1.101 / 1.277         | 0.609 / 0.845          | 1.82x (1.75–1.88)       |
| dual_sma_spread | 102,400    | 64     | True      | 1.550 / 1.702         | 0.908 / 1.149          | 1.70x (1.54–1.94)       |
| dual_sma_spread | 102,400    | 64     | False     | 1.110 / 1.340         | 0.636 / 0.722          | 1.75x (1.67–1.81)       |
| dual_sma_spread | 1,024,000  | 64     | True      | 1.567 / 1.792         | 0.908 / 1.038          | 1.78x (1.52–1.94)       |
| dual_sma_spread | 1,024,000  | 64     | False     | 1.119 / 1.301         | 0.649 / 0.743          | 1.78x (1.71–1.82)       |
| dual_sma_spread | 1,024,000  | 640    | True      | 2.551 / 2.680         | 1.147 / 1.324          | 2.22x (2.08–2.33)       |
| dual_sma_spread | 1,024,000  | 640    | False     | 2.161 / 2.456         | 0.922 / 1.086          | 2.37x (2.26–2.41)       |
| dual_sma_spread | 1,024,000  | 6,400  | True      | 13.302 / 13.751       | 2.079 / 2.460          | 6.40x (6.09–6.51)       |
| dual_sma_spread | 1,024,000  | 6,400  | False     | 13.033 / 14.000       | 1.994 / 2.131          | 6.76x (6.35–7.03)       |
| dual_sma_spread | 1,024,000  | 64,000 | True      | 127.867 / 134.575     | 9.782 / 10.330         | 13.05x (12.91–13.20)    |
| dual_sma_spread | 1,024,000  | 64,000 | False     | 126.508 / 130.594     | 9.592 / 10.022         | 13.11x (13.02–13.33)    |
| dual_sma_spread | 10,240,000 | 64     | True      | 2.130 / 2.603         | 1.459 / 2.257          | 1.42x (1.36–1.60)       |
| dual_sma_spread | 10,240,000 | 64     | False     | 1.710 / 2.006         | 1.178 / 1.430          | 1.46x (1.36–1.51)       |

## Callback interval breakdown

Representative H=1,024,000 forced-GC cases; independent P50 values in ms.
These wall intervals include scheduling and waiting, not just CPU work.
Conversion is a subset of sink-to-receive and must not be added again.
Independent phase medians need not sum to the total latency median.

| Indicator       | Append | Build     | Enqueue to source | Data to watermark | Watermark to sink | Sink to receive | to_pyarrow subset |
|-----------------|--------|-----------|-------------------|-------------------|-------------------|-----------------|-------------------|
| rolling_mean    | 64     | baseline  | 0.040             | 0.409             | 1.074             | 0.063           | 0.049             |
| rolling_mean    | 64     | candidate | 0.036             | 0.373             | 0.444             | 0.083           | 0.049             |
| rolling_mean    | 640    | baseline  | 0.039             | 0.395             | 2.043             | 0.066           | 0.052             |
| rolling_mean    | 640    | candidate | 0.036             | 0.364             | 0.660             | 0.082           | 0.048             |
| rolling_mean    | 6,400  | baseline  | 0.042             | 0.398             | 12.405            | 0.083           | 0.064             |
| rolling_mean    | 6,400  | candidate | 0.038             | 0.361             | 1.421             | 0.092           | 0.055             |
| rolling_mean    | 64,000 | baseline  | 0.045             | 0.563             | 123.250           | 0.135           | 0.110             |
| rolling_mean    | 64,000 | candidate | 0.042             | 0.538             | 8.108             | 0.108           | 0.062             |
| dual_sma_spread | 64     | baseline  | 0.038             | 0.404             | 1.039             | 0.065           | 0.052             |
| dual_sma_spread | 64     | candidate | 0.038             | 0.365             | 0.407             | 0.083           | 0.051             |
| dual_sma_spread | 640    | baseline  | 0.041             | 0.385             | 2.032             | 0.068           | 0.054             |
| dual_sma_spread | 640    | candidate | 0.037             | 0.362             | 0.656             | 0.082           | 0.050             |
| dual_sma_spread | 6,400  | baseline  | 0.042             | 0.436             | 12.735            | 0.084           | 0.065             |
| dual_sma_spread | 6,400  | candidate | 0.037             | 0.388             | 1.564             | 0.090           | 0.053             |
| dual_sma_spread | 64,000 | baseline  | 0.045             | 0.607             | 127.108           | 0.138           | 0.115             |
| dual_sma_spread | 64,000 | candidate | 0.043             | 0.492             | 9.127             | 0.117           | 0.066             |
