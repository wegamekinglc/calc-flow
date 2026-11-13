# Complete-history sorting and top-k

Declare `sort` or `top_k` with an explicit finite history and `StateBudget`.
Rust and Python DataFrame and SQL declarations lower to one native
`bounded_order` operator. Ordinary SQL `ORDER BY ... LIMIT ...` retains its
existing per-batch stream behavior.

```python
import pyarrow as pa
import calc_flow as cf

schema = pa.schema([
    pa.field("x", pa.int64()),
    pa.field("seq", pa.uint64(), nullable=False),
])
events = cf.table_input("events", schema=schema)
spec = cf.SortSpec.bounded_source(
    cf.StateBudget(1000, 8 << 20), sequence_by=["seq"]
)
out = events.top_k(1, [cf.SortKey("x")], spec=spec)
program = cf.Program("smallest", mode="batch", outputs={"out": out})
data = pa.Table.from_pydict({"x": [3, 1, 2], "seq": [10, 11, 12]}, schema)
assert program.collect({"events": data})["out"].to_pydict() == {
    "x": [1], "seq": [11],
}
```

`top_k` takes the first k rows of the declared comparator order after offset.
Use `SortKey("x", descending=True)` for largest first. Default NULL placement
is ASC last / DESC first; explicit `nulls="first"|"last"` is independent of
direction. All input columns and duplicate rows remain available. `sort`
returns the complete ordered relation after offset.

Stable ties require explicit `sequence_by` columns. They must be original or
purely renamed input fields, nonnullable integers or UTF-8 strings, disjoint
from primary keys. The native node checks their tuple is unique over all H,
including across batches. Sequence comparators are ascending after the primary
keys. Callers must supply the same reproducible row identity at the same source
cut; batch metadata sequence and arrival order do not establish it. Without
this declaration, equal-key order and top-k/offset membership are unspecified.

The equivalent SQL dialect is:

```sql
SELECT * FROM CF_ORDER_BOUNDED('a', 1000, 8388608, ARRAY['seq'])
ORDER BY x ASC NULLS LAST LIMIT 1
```

The sequence ARRAY is optional. A window declaration uses
`SortSpec.closed_window("ts", start_micros=0, end_micros=10,
state_budget=budget, sequence_by=["seq"], late_policy="drop")`, or:

```sql
SELECT * FROM CF_ORDER_WINDOW('a', 'ts', 0, 10, 1000, 8388608, 'drop', ARRAY['seq'])
ORDER BY x LIMIT 1 OFFSET 1
```

The macro subset requires one top-level `SELECT *`, one referenced input,
direct-column ORDER BY and optional decimal-literal LIMIT/OFFSET. Perform
filtering or projection as separate relations before passing the input. Invalid
macros are rejected without falling back to ordinary SQL.

In finite execution, H is the complete input to that execute; repeated calls on
one compiled plan are independent. In a stream, `bounded_source` requires a
proven fixed finite source cut and waits for actual normal EOF. A window accepts
`[start,end)` and closes at a trustworthy watermark at or beyond end, or normal
EOF. Empty input and k=0 still emit exactly one typed final at closure. Empty
Data, idle, barriers, failures and cancellation do not close H. A restored
closed node does not emit again.

Window Error rejects a whole envelope containing post-close members. Local
Drop ignores those members without extending H or emitting another result,
and checkpoints late row/batch/max-lateness counters. NULL-time and out-of-window
rows are outside H. Global temporal options do not override this local policy.

The first supported path has one exact-schema source, deterministic row-local
transforms, one ordering node and row-local output transforms. Stateful,
snapshot, multi-source, nested finalizer and external/array paths are rejected.
Ordering facts after a transform require a physical order proof and retained
comparator fields; dropping keys does not prove declared-order portability.

Keys and retained payload support booleans, signed/unsigned integers, UTF-8
strings, dates, timestamp[us] naive/UTC and timestamp[ms] naive. Other retained
types must be projected away before the node. k and offset are inclusive
0..4294967295, with offset+k at most 4294967295. Row budgets are positive and
at most 4294967295; byte budgets are positive JSON-safe integers.

The implementation retains full H, including rows outside the final top-k.
This is not O(k) storage. The byte limit covers the node-owned participating
peak: copied input/schema/metadata, candidates, sequence validation, sorting,
compact output and live captures/restore work. Raw ingress is borrowed only
during synchronous copying; unknown external backing during that loan, other
graph owners and process RSS are outside this quota. Work is reserved before
target allocation. Even a tiny final result can fail the peak budget.

See the independent [Python example](../examples/bounded_global_ordering.py)
and [Rust core example](../crates/calc-flow/examples/bounded_global_ordering.rs).
