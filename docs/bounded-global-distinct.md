# Final DISTINCT over bounded history

`TableExpr.distinct(columns, spec=...)` and the explicit SQL macros
`CF_DISTINCT_BOUNDED(alias, max_rows, max_bytes)` and
`CF_DISTINCT_WINDOW(alias, time_column, start_micros, end_micros, max_rows, max_bytes)`
declare one final, unordered relation. They share the Rust operator and DataFusion
DISTINCT implementation. Ordinary SQL `SELECT DISTINCT` keeps its per-batch behavior.

Use `DistinctSpec.bounded_source(StateBudget(...))` for a source with an authoritative
fixed finite cut. Streaming validates its descriptor before opening the source or
sink, and emits once at real normal EOF. Unknown and unbounded sources are refused.
Materialized Arrow sources declare boundedness; an arbitrary iterator does not.

`DistinctSpec.closed_window("ts", start_micros=0, end_micros=10, state_budget=...)`
accepts only timestamps in `[0,10)` and ignores NULL times. The time field need not
be a key. A trustworthy watermark at or beyond 10, or normal EOF, closes the relation.
An unbounded/unknown source needs a proved watermark route. Earlier watermarks do
not discard still-open window members. Cancellation, errors and disconnects fail
the job; they do not produce a successful final relation.

The current implementation handles post-close window members with `error`.
Selecting `drop` is rejected before compilation/export because the frozen v2
declaration has no field that can preserve that selection. This contract gap is
tracked on DAL-519; the implementation remains a draft pending its resolution.

`max_rows` counts **accepted history including duplicates**, not unique keys. Two
events `[1]`, `[1]` need two rows of history even though the final relation is `{1}`.
Empty history emits one empty final with the exact declared projected schema.
Supported keys are Boolean, signed/unsigned integers, strings/large strings, dates
and microsecond timestamps (naive or UTC). NULL tuples group together. Result order
is unspecified. Float, dictionary and nested keys are outside this version.

`copy_visible_v2` copies visible scalar values into independent Arrow buffers.
`max_bytes` limits the node's **owned participating peak**: its declarations,
retained history, replacement candidates, tracked query workspace, prepared output,
IPC scratch and live checkpoint captures. Reservations precede those allocations.
The temporary synchronous borrow of external ingress is an explicit exception:
its unknown parent allocation is excluded. Other graph owners, reusable sources,
callers, allocator overhead outside this accounting and process RSS are excluded.
Retained history never uses this exception. Dropping the caller's input may not
release a graph/source owner; the DISTINCT node itself retains only its copy.

The supported path is one exact-schema source through deterministic row-local
DF/SQL operations, one DISTINCT, then row-local operations. Independent branches
and shared outputs are supported. Stateful/snapshot/multi-source/array paths and
nested DISTINCT are rejected. Window time must trace to the input unchanged or
through a pure rename.

Project format 3 exports `kind="distinct"`, ordered keys and a required
`ingress_policy="copy_visible_v2"`. Native identity is `distinct_v2`, state version
1, layout 2, accounting 2. Checkpoints preserve duplicate history, closed state,
watermark and output sequence. Old draft layouts/policies are refused. Captures
remain charged while alive. Finite execution starts fresh on every call; it has
NULL batch snapshots and requires no reset between calls.

Analysis schema 2/revision 9 reports conditional finality and binding obligations;
it does not inspect input rows or promise a runtime budget will suffice.
The capability is `distinct@1`, batch/stream, `group_final_append_only`, checkpoint
layout `(2,)`. It has no unconditional watermark requirement; the selected window
still performs its conditional closure check. Full regression and coverage run in
CI; this feature does not assert a performance guarantee.
