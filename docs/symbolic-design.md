# Symbolic compiler design

[Documentation](README.md) / 4.3 Symbolic compiler design

The expression compiler implements the declarations exported from the Python
application API. Its implementation remains under `calc_flow.symbolic` and turns
immutable expressions into the native project graphs also used by direct builders.
It owns declaration identity, static analysis, and lowering. Native operators and runners own data execution,
event-time progress, and recovery. For usage, read
[expression workflows](symbolic-workflows.md); for accepted declarations, use the
[expression API](symbolic-api.md).

## Declaration and analysis

Expressions form an immutable directed acyclic graph. Typed inputs, ordered
features, and outputs contribute to a runtime-independent program fingerprint.
Structural identity allows shared expressions without evaluating data or
mutating caller declarations.

`Program.analyze` selects a default runtime when omitted and consumes one
immutable capability snapshot from the selected runtime. It checks
types, row lineage, symbolic dimensions, attachment compatibility, ordering,
state requirements, and stream safety. Issues have stable paths and codes.
Analysis and compilation are separate: the declaration catalog includes
aggregate-free event windows and standalone array forms without executable
lowerers. Compilation checks the supported shape and reports an error before
execution.

The implementation lives in
[nodes.py](../python/calc_flow/symbolic/nodes.py),
[program.py](../python/calc_flow/symbolic/program.py), and
[analyzer.py](../python/calc_flow/symbolic/analyzer.py).

## Convenience execution and export

The [compute adapter](../python/calc_flow/compute.py) turns Arrow schema into
existing ordered `Field` declarations without reading values to infer types or
nullability. Named table mappings and keyword expressions normalize to existing
nodes. Omitted Program inputs are discovered deterministically from outputs;
explicit input lists remain authoritative. The canonical IR and fingerprints
are unchanged for equivalent declarations.

The adapter tracks logical declarations through lowering to physical graph
endpoints, allowing collection by declared names even with multiple inputs,
shared stages, or several outputs. It returns Arrow tables in logical output
order. Project export uses `Program.to_project` and serializes only existing
project-v3 graph/input fields; aliases and live data stay in Python.

Every convenience call compiles a fresh native batch plan, including calls with
the same runtime. This isolates state without touching an explicitly cached plan.
Async entry points copy mappings and capture Batch references at call time,
then await the native cancellation-aware execution bridge. Arrow buffers remain
shared and their storage must stay read-only through execution. Schema/field
metadata are omitted only from internal execution schemas; caller metadata and
`Batch.metadata` remain intact.

## Lowering and physical sharing

The [lowering modules](../python/calc_flow/symbolic/lower/) emit strict
project-v3 nodes with registered operator identities. Row-local work becomes
expression stages. Nested temporal expressions are scheduled from the
innermost calculation outward, with deterministic row-local stages between
stateful stages when necessary. Each unique bounded join declaration has one
native join state owner.

Aggregate-bearing event windows lower to the existing native
`WindowAggregateOperator`. Their `@2` declaration identity includes the full
input graph, geometry, grouping order, and aggregate order and output names.
Equal complete declarations share one window state owner; geometry alone does
not establish sharing. Aggregate-free `@1` declarations retain their stable
canonical identity and remain declaration-only.

The [optimizer](../python/calc_flow/symbolic/optimizer.py) shares structurally
identical expressions, compatible rolling state, and compatible cross-section
grouping/sort work across output branches. Prefilter identity, ordering, and
group finality are part of compatibility. Filters do not move across finality
boundaries. Complete-group calculations accumulate across micro-batches;
batch segmentation does not establish group completeness.

Post-join entity, event-time, and sequence metadata proves the order needed by
nested joins and downstream stateful calculations. It does not sort data.
Projection discards ordering facts when it removes the named fields.

## Native event-window state

The [event-window lowerer](../python/calc_flow/symbolic/lower/event_windows.py)
compiles stateless table fragments on either side of each window and connects
them through the existing strict project-v3 `window` variant. Native
compilation validates the exact input and derived output schemas. A window
path contains one event window with only stateless table work before and after
it; independent legal output branches keep their own lowering contracts.

The [schema adapter](../python/calc_flow/symbolic/lower/schema.py) propagates
Arrow schemas through each actual lowered row-local and shared-expression
stage before a window and after its output. The analyzer first validates
types, field references, and stable diagnostic paths, then checks that native
field names and types agree with the frozen declarations. Native planning
supplies nullability; Python does not reproduce DataFusion's CASE or boolean
optimizer rules. Planning failures are analysis failures.

The private `Runtime._infer_symbolic_expression_schema` adapter calls the
private PyO3 `_infer_expression_schema` entry point. The internal Rust adapter
preserves the native stream fast path for column projections. Other
expressions reach DataFusion physical planning over an empty schema-bearing
`MemTable`, and the adapter reads the physical plan's schema without opening
sources, executing user rows, or invoking registered UDFs. This is a schema
planning seam; project-v3, `WindowSpec`, checkpoint layouts, and REST data
contracts retain their existing shapes.

Window output has a distinct typed row origin. It does not inherit the source
row origin, event-time key, entity keys, or sequence keys. Source column and
array attachment cannot prove alignment with the changed row set. The time
column used for assignment must still be an unchanged input column or a pure
rename so that its coordinate agrees with the source watermark. The analyzer
checks this separately from the ordering proof used for rolling and
cross-section state; nullable time and ungrouped event windows are valid.

The native window owns accumulation, late-assignment and null-time counters,
watermark closure, output order, and checkpoint state layout `1`. Tumbling and
hopping close at `watermark >= end` or end-of-input and append final results.
An aligned checkpoint preserves open windows using the existing manifest and
state protocol. Python adds no runtime accumulator or window buffer. Active
window/group counts remain data-dependent; the 1024 hopping-overlap limit is
an assignment bound, not a total memory bound. See the
[event-window API](symbolic-api.md#symbolic-event-time-window-aggregation)
for the type matrix, composition limits, and lateness-option ownership.

## Native rolling state

[RollingOperator](../crates/calc-flow/src/operator/rolling.rs) uses the same
kernel for batch and stream execution. Compatible outputs share retained rows
and accumulators; readout choices such as `min_periods` do not create duplicate
state. EWMA retains its valid count and exact binary64 recurrence value.

Project declarations carry the validated configuration and declaration layout
versions. The current checkpoint writer uses columnar state layout `3`, with
an entity dictionary, projected history, a reorder buffer, recurrence state,
and kernel/numerical fingerprints. Restore also supports the declaration's
layout `1` or `2` state. Writer layout is distinct from the declaration fields.
The [columnar state implementation](../crates/calc-flow/src/operator/rolling/state_v3.rs)
owns this encoding.

The default numerical profile is `stable_v1`. Explicit `stable_v2` enables
shifted compensated sums with periodic rebasing. Profile identity contributes
to configuration and kernel fingerprints and is checked on recovery.
The [Rust runtime reference](rust-api.md#rolling-windows) specifies types, null/NaN handling,
infinity classification, numerical limits, and late-row behavior.

## Static values and matrix placement

A supported symbolic matrix segment fuses table-to-array conversion,
allowlisted elementwise work, one direct static-weight matmul, and result
attachment into a registered `symbolic_matrix@1` provider. Table row lineage,
backend, dimensions, and output names must be proved before lowering.

The runner validates and latches static inputs once before opening sources.
[static_input.rs](../crates/calc-flow/src/static_input.rs) defines the canonical
`calc_flow.static_input.digest.v1` tagged-byte encoding and SHA-256 digest.
The encoding follows logical values independently of batch chunking,
dictionary layout, and array strides, and canonicalizes NaNs per dtype.
The source defines the exact accepted types and byte grammar.

Declarations enter the plan fingerprint; payload digests enter the checkpoint
manifest. Changing weights therefore reaches the same lineage and fails the
digest check before opening a source. Payloads and backing memory are not
exposed through status or diagnostics.

Provider placement runs on a blocking worker and caches only the successfully
placed immutable value after a cancellation check. `static_placement_bytes`
counts logical provider transfer, not peak memory or RSS. During first
placement, snapshot carriers, a Python host list, NumPy storage, and a JAX
device value can coexist. Subsequent micro-batches reuse the placed weights.

## Compile cache and inspection

Each [Runtime](../python/calc_flow/pipeline.py) keys its symbolic compile cache
by program identity, execution mode, declarations, and capability/version facts.
Batch entries retain immutable plans. Stream entries retain successfully
compiled immutable project JSON, and each compile creates a fresh native
owning plan because the runner consumes it. A restart recompiles the same
declaration and creates fresh bindings and a runner.
Convenience `compute`/`collect` calls bypass cached plan instances and do not
reset or replace them. `Program.to_project` exports data without adding a new
plan lifecycle. Registration changes invalidate that runtime's cache. The cache is bounded
and belongs to the runtime instance; it is not a cross-run DataFusion session
cache.

An independent runtime-local schema cache holds at most 128 immutable Arrow
schemas. Its key contains the ordered projection, filter, and exact serialized
input schema. Only successful native planning enters the cache. Successful
registrations clear it alongside the compile cache. It retains planning
metadata and does not retain user rows, running state, or DataFusion sessions.

`Program.explain` reports deterministic physical sharing and state/copy facts.
Studio inspects facts encoded in the lowered project document. Neither is a
measurement of latency, peak memory, or throughput. Execution timings, job
status, and the [benchmark suite](benchmark-suite.md) provide measurements.

The relevant runnable examples are
[09](../examples/09_symbolic_financial_features.py),
[10](../examples/10_symbolic_streaming_recovery.py),
[11](../examples/11_symbolic_static_matrix.py),
[12](../examples/12_symbolic_stream_join.py),
[13](../examples/13_symbolic_relational_dag.py), and
[event-window aggregation](../examples/symbolic_event_window.py).

Next: [verification](verification.md).
