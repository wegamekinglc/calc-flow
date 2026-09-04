# SQL and DataFusion performance controls

Calc Flow executes table expressions and SQL with Apache DataFusion 54. The
public configuration keeps the historical fixed single-partition behavior by
default, exposes an opt-in conservative auto policy, and provides independent
rollback switches for SQL rolling rewrites and diagnostic collection.

## Configuration

`DataFusionConfig` is part of project-v3 and is also available through Rust's
`PipelineBuilder::with_datafusion_config`. Python builders use the immutable
`with_datafusion_config(...)` method.

| Field                    | Default | Meaning                                                        |
| ------------------------ | ------: | -------------------------------------------------------------- |
| `batch_size`             |  `8192` | DataFusion execution batch size                                |
| `target_partitions`      |     `1` | Requested count in `fixed` mode                                |
| `parallelism_mode`       | `fixed` | `fixed` preserves explicit behavior; `auto` uses trusted facts |
| `max_partitions`         |    `32` | Upper bound for `auto`                                         |
| `min_rows_per_partition` | `65536` | Work cap used by both modes                                    |
| `small_rows_threshold`   | `10001` | Smaller auto workloads remain p1                               |
| `enable_rolling_rewrite` |  `true` | Enables the fail-closed bounded `AVG` rewrite                  |
| `collect_diagnostics`    |  `true` | Collects plan strings and physical metric traversal            |

All numeric values must be positive. Existing project documents that omit the
new fields retain their defaults. `parallelism_mode="auto"` is opt-in: the
weekly P3 matrix and two stable candidate repeats must pass before changing the
default.

Auto mode calculates:

```text
requested = min(available_parallelism, max_partitions)
effective = min(requested, ceil(rows / min_rows_per_partition), active_entities)
```

Inputs below `small_rows_threshold`, single-entity inputs, and inputs without a
trusted positive `calc_flow.datafusion.active_entities` metadata value use one
partition. The runtime never scans a table to choose its partition count.

## Telemetry

Each `DataFusionQueryMetric` records configured, requested and effective
partitions; the host capacity; decision inputs and reason; input and output row
counts; per-output-partition rows; spill bytes; physical operator counts; total,
window, and repartition/sort compute; and phase timings. The recorded
`parallelism_mode`, `rolling_rewrite_enabled`, and `diagnostics_collected`
values make rollback state explicit.

Set `collect_diagnostics=false` when plan text and recursive physical metrics
are not required. Planning and execution continue unchanged, while plan strings
are empty and the expensive traversal is skipped. Set
`enable_rolling_rewrite=false` to route every SQL window through DataFusion's
standard executor. Set `parallelism_mode="fixed", target_partitions=1` for the
complete historical fallback.

## Fair benchmark and evidence gates

The comparison benchmark runs Calc Flow and raw DataFusion in the same binary
with identical input batch boundaries, session configuration, and normalized
physical plan. Fair profiles always disable the rolling rewrite.

```bash
cargo bench -p calc-flow --bench sql_datafusion_performance -- \
  --profile matched-adaptive --samples 20 --warmups 1 \
  --output target/sql-datafusion/matched-first.json

python scripts/verify_sql_datafusion_performance.py \
  target/sql-datafusion/matched-first.json \
  --repeat target/sql-datafusion/matched-second.json \
  --serial-control target/sql-datafusion/serial-first.json \
  --minimum-samples 20 --require-stable --require-p1
```

The verifier fails closed on configuration, batch-boundary, plan, correctness,
sample, stability, CV, RSS, or P1 threshold mismatches. It suppresses a speedup
conclusion whenever the two physical plans are not comparable.
The machine-readable contract is
[`sql-datafusion-performance-v1.schema.json`](../schemas/sql-datafusion-performance-v1.schema.json).
Because the measured SQL relation has no outer `ORDER BY`, correctness is
aligned by the unique `(symbol, event_time, sequence)` key outside the timed
envelope; canonical key order, null/NaN masks, and values must then match. This
does not add a global sort to either measured physical plan.

The P4 report assigns the same-binary gap to execution, fixed envelope,
materialization, and the directly timed run/session envelope, and then applies
the P5/P6/P7 evidence thresholds. The analyzer itself requires a clean release
build with at least 20 pairs, so a diagnostic smoke report cannot emit gate
decisions:

```bash
python scripts/analyze_sql_datafusion_attribution.py \
  target/sql-datafusion/attribution.json \
  --output target/sql-datafusion/attribution-analysis.json
```

Weekly CI runs the complete P3 grid for 100k, 1m, and 2.1m rows; 1, 4, 16, and
64 active entities; p1 through p32; batch sizes 4096 through 32768; and both
SMA workloads. Five-pair screening retains the latency/RSS Pareto frontier,
then candidates receive two independent 20-pair runs.

## Rolling rewrite boundary

The rewrite accepts only one or more `AVG(Float64Column)` expressions in a
bounded `ROWS ... PRECEDING AND CURRENT ROW` window with simple entity keys and
ascending, non-null microsecond event-time and sequence columns. Unsupported
expressions, casts, filters, distinctness, null-treatment options, ordering,
frames, types, and aggregates keep DataFusion's plan and record a stable
fallback reason.

The shared kernel matches DataFusion 54 for partial windows, null/all-null
partitions, IEEE NaN and infinities, `W=1`, `W>N`, duplicate timestamps,
out-of-order input after physical sorting, aliases, and multiple entities.
Only each retained-window state transition is amortized `O(1)`; sorting,
partitioning, scanning, and output construction keep the full query `O(N)`.

## Canary and rollback

For a release candidate, retain `fixed` as the default and test auto mode at 5%
of eligible jobs, then 25%, then 100%. Advance only when two nightly reports
have matching fingerprints, `CV <= 10%`, no unexplained spill, correct output,
P1 latency/ratio targets, and RSS within the documented guards.

Rollback in this order:

1. Set `enable_rolling_rewrite=false` if a window semantic or memory signal fires.
2. Set `parallelism_mode="fixed"` and `target_partitions=1` for latency, RSS,
   spill, or skew regressions.
3. Set `collect_diagnostics=false` only to remove observational overhead; this
   does not change query results or physical planning.

Cross-run session or plan caching and partition-preserving DAG envelopes are
not enabled. They remain No-Go unless same-binary evidence independently meets
their thresholds and their isolation/order contracts can be preserved.
