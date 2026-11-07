# Both-engine core-count sweep on main 9a75a71f

Informational broad measurement, explicitly requested, covering both
engines: the streaming engine (example 30, compact incremental aggregation,
checkpoints disabled) and the SQL engine (`sql_plan_reuse_probe`, the
compile-once state-feedback batch lifecycle with physical-plan template
reuse from #557, fixed partitions = core count). Same workload scale as the
[maintained seven-engine table](seven-engines-2193f8ee.md): 10,000,000
business rows, 100x100 keys, complete cumulative snapshots.

## Results

Wall-time medians of five fresh-process samples per cell, `taskset` pinned
to the first N logical CPUs of the same 32-CPU host. Background host load
during collection was 4.85-7.85 (one-minute average at start); this is a
disclosed concurrent-load observation, not an isolated benchmark.

| Batch size | Engine | 1 CPU | 4 CPU | 8 CPU | 16 CPU | 32 CPU |
|------------|----------|---------|---------|---------|---------|---------|
| 1,000,000  | Streaming | 1.669 s | 1.190 s | 1.199 s | 1.210 s | 1.184 s |
| 1,000,000  | SQL (p=N) | 0.429 s | 0.163 s | 0.149 s | 0.124 s | 0.181 s |
| 100,000    | Streaming | 2.030 s | 1.152 s | 1.159 s | 1.098 s | 1.145 s |
| 100,000    | SQL (p=N) | 0.579 s | 0.384 s | 0.378 s | 0.392 s | 0.549 s |

Child CPU medians: streaming stays within 1.7-2.4 s at every core count;
the SQL engine uses 3.4-5.7 s, trading CPU for wall time as partitions grow.

Timer boundaries differ and are not compared across engines here: the
streaming timer includes the demo source's lazy row generation while the
SQL probe times only its execute loop against prebuilt inputs. The
seven-engine table keeps equal boundaries for cross-engine absolutes; this
sweep's value is each engine's own scaling shape.

## Findings

1. The SQL engine's large-batch aggregation scales 3.5x from one to sixteen
   cores (0.429 to 0.124 s at 1M-row batches) and flattens beyond; at
   100k-row batches it scales only 1.5x and degrades at 32 partitions,
   because 101 per-batch lifecycles bound the parallel share.
2. The streaming engine is flat beyond four cores (1.2 s at 1M-row batches,
   ~1.8 CPU s everywhere): the single compact-state owner caps its
   parallelism, as the multicore investigation found.
3. Consequently the large-batch streaming/SQL gap widens with cores in
   relative terms while streaming stays flat in absolute terms. Under equal
   timer boundaries the seven-engine gap at 32 CPUs was 0.573 vs 0.238 s.

## Implication for the withdrawn delegation route

The SQL engine's own curve is the upper bound for delegating the streaming
operator's per-batch delta aggregation to a partitioned DataFusion
aggregate (template-reused) and merging the ~group-count result into
compact state once. That bound is large at four or more cores (3-7x on the
aggregate-dominated 1M-batch case), and the route has none of the overheads
that measured-rejected the two sharding prototypes: no routing copy of
input rows and no per-fragment accumulator state round-trip - the grouped
output is ~10,000 rows. The cost is CPU: delegation spends the SQL engine's
3-4 CPU s instead of the streaming kernel's 1.8.

Whether that trade is right depends on the deployment target: under the
maintained 1T/4T discipline the streaming kernel remains the efficiency
choice per CPU; on hosts with idle cores and wall-time targets, delegation
at an explicit partition count is the one remaining evidence-backed route
for the large-batch gap.

## Reproduction

```bash
cargo build -p calc-flow --release --example 30_streaming_stock_positions --example sql_plan_reuse_probe
# streaming: taskset -c 0-3 <example30> --rows 10000000 --batch-size 1000000
# sql: PLAN_REUSE_PROBE_ROWS=10000000 PLAN_REUSE_PROBE_BATCHES=1000000 \
#      PLAN_REUSE_PROBE_PARTITIONS=4 taskset -c 0-3 <probe>
```
