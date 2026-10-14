# Issue 406: compiled stream expressions

Status: implemented with recorded RED/GREEN and focused boundary controls;
formatting, scoped Clippy and source provenance pass. Final specialist review
is pending. No performance/CI or whole-issue acceptance is claimed.

## Scope

Phase 2.3 removes per-batch execution-plan rebinding for eligible immutable,
single-input ExpressionOperator stream queries. The first batch retains the
existing SQL planning/execution path. Subsequent exact-schema batches reuse
compiled DataFusion PhysicalExprs from the existing bounded query-template
cache. This increment does not claim arithmetic-tail fusion or a speedup.

Batch execution and public DataFusionRuntime query methods retain their current
execution and diagnostic metrics. Stream callback timing/status remains owned by
the runner. The private stream expression runtime records SQL-query metrics only
for batches executing the SQL fallback; direct compiled evaluation does not
pretend to execute a fresh SQL plan. Exact column projections retain their current
path. Custom selected UDFs, mutable functions, non-row-local physical nodes,
fetch/limits and unavailable templates keep the existing SQL path.

## Requirements

1. Capture no input rows, parent Arrow buffers or per-call execution state in
   the compiled expression. The template cache remains bounded to eight entries;
   cloned operators own fresh runtimes. Exact input schema, including metadata,
   remains part of cache identity.
2. Compile only the existing immutable row-local template whitelist. Preserve
   the planned scan projection, physical projection expressions/schema, filter
   predicate, filter projection and filter batch-coalescing size/order. Reject
   unsupported fetch or ordered scans instead of interpreting a broader tree.
3. Evaluate arrays through DataFusion's Projector and filter kernels. Do not
   substitute Python evaluation, reparsing, row callbacks or scalar arithmetic.
   Preserve nullable values, casts, aliases, empty/zero-column row counts,
   multiple input record batches and SQL-derived output schema metadata.
4. Stream output preserves the existing SQL Batch metadata and source-cursor
   behavior. Retain the same SQL visible-slice row/byte collection limits and
   DataFusion error node IDs/messages. Failed evaluation emits no output.
5. Compiled state is immutable. Per-call filter coalescers and output vectors
   are owned by that call, with no rows retained after success/error/cancellation
   or abandonment. Check job cancellation/deadline at record boundaries and
   before emission; yield between multiple input records without adding an
   unconditional scheduler hop to a single-record batch.
6. Focused differential tests compare repeated distinct batches with the SQL
   reference, verify the cached path actually avoids both full planning and
   template rebinding, and cover schema changes, nullable arithmetic/filtering,
   empty and multiple records, source lifetime and errors. Unsupported queries
   must remain on SQL. Reuse an existing libtest registration where necessary.
7. Verify affected Rust formatting/lint and generated-contract no-drift;
   changed/new functions have CCN at most eight. Final specialist review is
   required. Full CI/coverage and measured phase/all-eight acceptance remain
   separate gates; do not repeat stopped historical performance attempts.

## Local verification

Start with the existing stream_computed_and_filtered_projections_keep_sql_semantics
registration augmented by repeated-stream controls. Record expected RED before
product implementation. Expand only to directly affected expression/runtime
controls when a concrete source change or failure justifies it. No performance
measurement or release build is part of this initial correctness increment.
