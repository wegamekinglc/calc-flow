# SQL and DataFusion performance controls

[Documentation](README.md) / 5.3 SQL performance

Calc Flow executes table expressions and SQL with Apache DataFusion 54. The
default configuration uses one fixed partition. Applications can opt into
conservative auto parallelism and independently control SQL rolling rewrites
and diagnostic collection.

## Configuration

`DataFusionConfig` is part of project-v3 and is also available through Rust's
`PipelineBuilder::with_datafusion_config`. Python builders use the immutable
`with_datafusion_config(...)` method.

| Field                    | Default | Meaning                                                        |
|--------------------------|--------:|----------------------------------------------------------------|
| `batch_size`             | `8192`  | DataFusion execution batch size                                |
| `target_partitions`      | `1`     | Requested count in `fixed` mode                                |
| `parallelism_mode`       | `fixed` | `fixed` preserves explicit behavior; `auto` uses trusted facts |
| `max_partitions`         | `32`    | Upper bound for `auto`                                         |
| `min_rows_per_partition` | `65536` | Work cap used by both modes                                    |
| `small_rows_threshold`   | `10001` | Smaller auto workloads remain p1                               |
| `enable_rolling_rewrite` | `true`  | Enables bounded `AVG` and compatible paired `COUNT` rewrites   |
| `collect_diagnostics`    | `true`  | Collects plan strings and physical metric traversal            |

The project schema accepts non-negative numeric values for compatibility with
external-only plans, which never create a DataFusion runtime. When a project
does execute a table expression or SQL node, every numeric value is validated
as positive before DataFusion is initialized. Omitted fields use their
defaults. `parallelism_mode="auto"` is opt-in.

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
fixed single-partition execution.

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

## UInt64 filter predicates

The logical optimizer specializes Boolean modulo comparisons when the input
schema proves a `UInt64` column and DataFusion has widened it to
`Decimal128(20, 0)`. A positive, representable divisor and non-negative,
representable comparison constant allow the equivalent UInt64 operation.
All six ordinary comparisons and either comparison orientation are covered;
the full UInt64 domain and null behavior are preserved. Unproved types,
casts, literals and zero divisors retain DataFusion's original expression and
error behavior. Projected arithmetic is not specialized by this filter rule.

This rule is independent of `enable_rolling_rewrite`. The original benchmark
query retains `sequence % 4 = 0`; the separately named
`filter_uint64_modulo` diagnostic uses explicit UInt64 literals. A typed-query
comparison and a same-query revision comparison answer different questions.
Physical-plan matching remains mandatory for the fair benchmark above.

## Rolling rewrite boundary

The rewrite accepts bounded `ROWS ... PRECEDING AND CURRENT ROW` windows
over `AVG(Float64Column)`, with compatible `COUNT` guards for the same value
column and frame. It requires simple partition columns, ascending non-null
microsecond event-time and sequence columns, supported key encoding, and
DataFusion's physical `InputOrderMode::Sorted` guarantee. Adjacent partition
ranges are scanned directly; only the last partition's window state survives
into the next input batch. The route does not infer ordering from input values.

COUNT/AVG queries use SQL sum/count transitions, adding the incoming value
before retracting the outgoing value. Counts are non-null Int64 and include
NaN values while excluding nulls. The route preserves DataFusion's NaN,
infinity and overflow behavior, including their effect on later frames.
DataFusion's optimized `COUNT(1)` is eligible only when a same-frame AVG proves
the corresponding value column non-nullable. Full-window guards are retained:
ten-row SMA(20) queries still have no full-window values.

Queries newly admitted by COUNT use SQL transitions for every rewritten AVG
stage. If any window stage cannot meet the compatibility proof, the entire
query retains DataFusion's window operators. Unsupported expressions, casts,
filters, distinctness, null-treatment options, ordering, frames, aggregates
and key types record a fallback reason instead of partially rewriting the
query. Map keys, including unsupported nested encodings, therefore retain the
ordinary DataFusion path.

AVG-only queries retain the existing shared Native numerical profile and its
West mean and refold behavior. This is not a bitwise sum/count equivalence
promise on arbitrary floating inputs. Native `stable_v1`/`stable_v2` profiles,
checkpoint identities and transition ordering are unchanged. Finite
comparison tolerances and special-value classifications are checked by the
corresponding numerical fixtures.

Retained-window updates are amortized `O(1)` per value. Sorting, partitioning,
scanning and Arrow output construction remain part of full-query cost.

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

Cross-run DataFusion session or plan caching and partition-preserving DAG
envelopes are not enabled. Python's symbolic runtime compile cache is a
separate mechanism; see [symbolic compiler design](symbolic-design.md#compile-cache-and-inspection).
