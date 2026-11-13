# Bounded analytic OVER

Rust and Python DataFrame and SQL declarations share one native evaluator for
`ROW_NUMBER`, `LAG(column,1)` and checked integer `SUM` over explicit ROWS frames.
The result contains every input column followed by the analytic columns. It is
emitted once, after the accepted history H actually closes. Ordinary SQL `OVER`
keeps its existing per-batch behavior.

```python
import calc_flow as cf

order = cf.OverOrder(["account"], [cf.SortKey("score")], sequence_by=["seq"])
spec = cf.OverSpec.bounded_source(order, state_budget=cf.StateBudget(1000, 8 << 20))
features = events.over(
    [
        cf.AnalyticExpr.row_number("rn"),
        cf.AnalyticExpr.lag("value", name="prev"),
        cf.AnalyticExpr.sum("value", frame=cf.RowsFrame.cumulative(), name="total"),
        cf.AnalyticExpr.sum("value", frame=cf.RowsFrame.preceding(1), name="last_two"),
    ],
    spec=spec,
)
```

Rust exposes `OverOrder::new`, `OverSpec::bounded_source` / `closed_window`,
`OverWindow::new`, `RowsFrame::cumulative` / `preceding`, `AnalyticExpr` and
`TableExpr::over` at the crate root. Python exports the same configurations from
`calc_flow` and `calc_flow.symbolic`. These immutable values contain declarations,
not data or callbacks. See the independently executable
[Rust example](../crates/calc-flow/examples/bounded_analytic_over.rs) and
[Python example](../examples/bounded_analytic_over.py).

## Ordering and functions

Final row order is partition keys ASC NULLS LAST, primary order, then sequence
keys ASC. NULL partition keys compare equal. Primary keys explicitly select
NULL first/last, independently of direction. The mandatory sequence is a
nonnullable integer or UTF-8 tuple, globally unique over the whole accepted H.
It must be an unchanged input field or pure rename; duplicate tuples reject the
entire incoming envelope. Arrival order does not supply a tie breaker.

`ROW_NUMBER` starts at one in each partition and returns nonnullable UInt64.
`LAG` reads the previous partition row, including its NULL, with NULL at the
partition start. It preserves the operand dtype and always permits NULL.
`SUM` skips NULL values, produces NULL for all-NULL frames, and widens signed
integers to Int64 and unsigned integers to UInt64. Every output must fit that
type: overflow fails the entire final, even when a later value could cancel it.
The new evaluator does not change ordinary DataFusion SUM.

Cumulative ROWS includes all preceding partition rows and the current row.
`preceding(N)` includes up to N preceding rows and the current row;
`preceding(0)` reads only the current row. N accepts exact integers in
0..4294967295. LAG and ROW_NUMBER have no frame. ROWS bounds calculation, not
retained history; even N=0 retains complete H until finalization.

All retained columns must be Boolean, integers, Utf8/LargeUtf8, Date32/Date64,
timestamp[us] naive/UTC, or timestamp[ms] naive. SUM operands must be integers.
Project other payload types away before this node. Output names are unique and
cannot overwrite input columns or reference another analytic alias.

## SQL declarations

```sql
SELECT *,
  ROW_NUMBER() OVER (PARTITION BY account ORDER BY score, seq) AS rn,
  LAG(value, 1) OVER (PARTITION BY account ORDER BY score, seq) AS prev,
  SUM(value) OVER (PARTITION BY account ORDER BY score, seq
                  ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS total
FROM CF_OVER_BOUNDED('a', 1000, 8388608, ARRAY['seq'])
```

Bind `a` using Rust `sql(query, tables)` or Python `cf.sql(query, a=events)`.
`CF_OVER_BOUNDED` takes exactly `(alias,max_rows,max_bytes,sequence_array)`.
For a fixed UTC window, use
`CF_OVER_WINDOW('a','ts',0,10,1000,8388608,ARRAY['seq'][,'drop'])`;
the optional local policy defaults to `error`. The interval is `[0,10)` in
signed microseconds, including 0 and excluding 10 and NULL timestamps.

All expressions must share partition and primary order, followed by the exact
sequence suffix ASC NULLS LAST. SUM requires an explicit supported ROWS frame.
Other functions, LAG offsets/defaults, NULL-treatment clauses, RANGE/GROUPS,
FOLLOWING, shorthand frames, other projections, DISTINCT, filtering, joins,
CTEs and outer ORDER/LIMIT are rejected by this explicit dialect. Declare
row-local transformations before or after the node. A malformed macro never
falls back to ordinary SQL.

## Closure, budgets and recovery

Bounded-source H closes at actual normal EOF of a fixed finite cut. Unknown or
unbounded sources are rejected before opening. A window closes at a trustworthy
watermark on the original timestamp coordinate, or bounded EOF. A row beyond
end, a full row budget, Idle, Barrier, or checkpoint does not close H. Before
closure, an in-domain row remains acceptable even when its timestamp is below
the current watermark. Empty H produces exactly one typed empty final.

After watermark closure, local `error` atomically rejects an envelope containing
any in-domain member. Local `drop` ignores those members and persists late row,
affected envelope and maximum `watermark - timestamp` metrics, without growing
H or emitting another final. Global rolling lateness options do not override
this declaration. Window batch execution filters the complete current input;
it does not simulate stream lateness.

Every node requires a StateBudget. `max_rows` counts all accepted rows across
partitions. `max_bytes` limits the node-owned logical peak, including copied H,
candidates, sequence validation, sorting, partition/frame state, all new output
columns, the compact final copy and concurrent captures/restore scratch. The
node synchronously deep copies visible ingress and releases borrowed owners
before its first internal await. Foreign parent buffers, other graph owners
and process RSS are outside that quota. A small final result need not fit a
small peak budget. There is no spill or implicit unlimited budget.

Data, finalization, late metrics and restore install state only after their
fallible preparation succeeds. Failed overflow, schema, budget, cancellation
or emit leaves the previous committed history and closure unchanged. This is
operator state atomicity, not whole-graph rollback or ExactlyOnce delivery.

`bounded_over_v1` has state version 1, accounting version 1, copy_visible_v2,
Error layout 1 and Drop layout 2. Checkpoint identity includes the complete
history/order/functions/frames/names/budget/policy, exact schemas and engine
configuration. Open restore continues H; closed restore does not resend the
final. Incompatible or corrupt captures reject before Source.open. Source cut
and replay obligations still apply; ordinary iterables are not durable replay.

Only one source and deterministic row-local transforms may surround an OVER
node. Nested/stateful finalizers, aggregates, snapshots, multi-source paths,
providers and arrays are rejected by mandatory native admission, including
reloaded projects and cached plans. Batch executions of the same compiled plan
remain independent. Query analysis revision 11 reports the closure and sequence
obligations; final-relation portability compares the same accepted H and cut.

## Budget ledger

The finalizer prepays the existing concat/sort/take reservation and an analytic
increment. The increment accounts separately for the provisional analytic
output's retained input columns, even while those columns share owned backing
with the taken input. The later compact copy acquires its own full reservation.

| Role                      | Prepaid bound and lifetime                                                                                                                                                                                                             |
|---------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Partition boundaries      | N machine-index slots (conservatively eight bytes each) and partition comparator descriptors, held through evaluation.                                                                                                                 |
| Generated columns         | Aligned value/offset/validity buffers and array descriptors for every expression, held together until normalization finishes.                                                                                                          |
| LAG indices               | A full UInt32 index/validity array, in addition to its generated value array.                                                                                                                                                          |
| Nullable string LAG       | Visible owned UTF-8 bytes, offsets, and a separate N-element `(usize,usize)` source-range vector plus its descriptor. Arrow's nullable byte take uses this vector; it is not assumed to fit unrelated scratch.                         |
| Provisional input columns | One additional complete owned input layout, conservatively charged as a separate output role despite shared backing.                                                                                                                   |
| Input schema work         | Four schema quotes: worker output names, temporary owned projection schema, cloned field/result schema, and validation containers. Each quote bounds its named role; new analytic fields and declaration copies have their own quotes. |
| Compact final             | A complete normalization reservation for every retained and generated column, acquired before copying while H, workspace and live captures still exist.                                                                                |
| Worker control            | Per-work credit and first-job worker-generation allocations remain charged through the existing owned worker service. Reusing a worker can change the participating peak, so exact boundary trials use equivalent fresh jobs.          |
| Capture/restore           | IPC, metadata, strict control/identity, decoded candidate, sequence validation, history containers and every still-live capture owner remain charged.                                                                                  |

All layout arithmetic uses checked additions, multiplications and alignment.
String byte counts come from normalized owned H before kernel allocation.
These bounds permit conservative rejection and describe participating logical
bytes, rather than allocator internals or RSS. Exact measured B/B−1 evidence is
recorded with the implementation; it is not a portable fixed byte promise.
