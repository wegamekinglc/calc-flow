# Symbolic compiler design

[Documentation](README.md) / 4.3 Symbolic compiler design

The symbolic layer turns immutable Python declarations into the same native
project graphs used by direct builders. It owns declaration identity, static
analysis, and lowering. Native operators and runners own data execution,
event-time progress, and recovery. For usage, read
[symbolic workflows](symbolic-workflows.md); for accepted declarations, use the
[symbolic API](symbolic-api.md).

## Declaration and analysis

Expressions form an immutable directed acyclic graph. Typed inputs, ordered
features, and outputs contribute to a runtime-independent program fingerprint.
Structural identity allows shared expressions without evaluating data or
mutating caller declarations.

`Program.analyze` consumes one explicit runtime capability snapshot. It checks
types, row lineage, symbolic dimensions, attachment compatibility, ordering,
state requirements, and stream safety. Issues have stable paths and codes.
Analysis and compilation are separate: the declaration catalog includes event
windows and standalone array forms without executable lowerers. Compilation
checks the supported shape and reports an error before execution.

The implementation lives in
[nodes.py](../python/calc_flow/symbolic/nodes.py),
[program.py](../python/calc_flow/symbolic/program.py), and
[analyzer.py](../python/calc_flow/symbolic/analyzer.py).

## Lowering and physical sharing

The [lowering modules](../python/calc_flow/symbolic/lower/) emit strict
project-v3 nodes with registered operator identities. Row-local work becomes
expression stages. Nested temporal expressions are scheduled from the
innermost calculation outward, with deterministic row-local stages between
stateful stages when necessary. Each unique bounded join declaration has one
native join state owner.

The [optimizer](../python/calc_flow/symbolic/optimizer.py) shares structurally
identical expressions, compatible rolling state, and compatible cross-section
grouping/sort work across output branches. Prefilter identity, ordering, and
group finality are part of compatibility. Filters do not move across finality
boundaries. Complete-group calculations accumulate across micro-batches;
batch segmentation does not establish group completeness.

Post-join entity, event-time, and sequence metadata proves the order needed by
nested joins and downstream stateful calculations. It does not sort data.
Projection discards ordering facts when it removes the named fields.

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
The [Rust API](rust-api.md#rolling-windows) specifies types, null/NaN handling,
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

Each [Runtime](../python/calc_flow/pipeline.py) caches immutable plans using
program identity, execution mode, declarations, and capability/version facts.
Registration changes invalidate that runtime's cache. The cache is bounded
and belongs to the runtime instance; it is not a cross-run DataFusion session
cache.

`Program.explain` reports deterministic physical sharing and state/copy facts.
Studio inspects facts encoded in the lowered project document. Neither is a
measurement of latency, peak memory, or throughput. Execution timings, job
status, and the [benchmark suite](benchmark-suite.md) provide measurements.

The relevant runnable examples are
[09](../examples/09_symbolic_financial_features.py),
[10](../examples/10_symbolic_streaming_recovery.py),
[11](../examples/11_symbolic_static_matrix.py),
[12](../examples/12_symbolic_stream_join.py), and
[13](../examples/13_symbolic_relational_dag.py).

Next: [verification](verification.md).
