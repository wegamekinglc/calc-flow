# Issue 406: compiled stream expression evidence

Base source is e884adc362ac8f8d03c1364d0dfdb4e205407696; the Join phase 2.2
review/documentation-only follow-up is carried by d83e68ff. Phase 2.3 is in
progress: implementation and focused verification pass; final specialist review
is pending. No performance claim, remote CI or release build belongs to this
increment. All eight original acceptance targets and other phases remain active.

## Expected regression RED

The existing stream_computed_and_filtered_projections_keep_sql_semantics
registration now runs repeated-stream differential controls for distinct input
values and nullable computed/filter expressions. The original SQL path executes
two plans for two stream batches even though its immutable template is cached;
the requirement is one SQL warmup followed by compiled array evaluation.

Root target/issue406-compiled-expressions-20261014/compiled-expressions-red.log
records the expected failure: observed private runtime query-metric count 2,
required 1. Value/schema/metadata and caller-input comparisons passed before
that assertion. Compilation took 3m11s, test execution 0.02s, runner191.454s,
exit101. This is a production-regression RED, not a fixture compiler error.
Default libtest registration count remains2033.

## Implementation in progress

Compiled state is captured lazily from the existing empty-scan, bounded,
immutable row-local query template. It retains only scan indices, DataFusion
Projectors, filter predicates/projections, schemas and coalescing sizes. Each
call owns fresh filter coalescers, bounded result collection and temporary rows.
The first batch and ineligible plans still use SQL; ordinary public runtime
queries and batch execution continue their existing path. Query-cache identity
continues to include exact input schema. Selected UDFs, non-row-local nodes,
fetch and ordered scans are refused by the direct path.

Filter predicates evaluate before physical projection, followed by Arrow
filtering and the same DataFusion LimitedBatchCoalescer policy. Computed output
uses DataFusion Projector. Collection reuses the exact SQL visible-slice limit
helper. Cancellation/deadline checks occur at record/step/flush boundaries and
before emission; multiple records yield between records. A single-record input
has no unconditional scheduler hop. The first focused GREEN run is pending.

Private stream SQL-query metrics describe batches that actually execute SQL;
compiled evaluation does not invent fresh-plan metrics. Runner callback metrics
and status still cover every stream call. No public DataFusionRuntime metrics
behavior is changed. Remaining controls include exact SQL parity for multi-record
and empty inputs, schema/metadata changes, errors, unsupported functions, input
owner lifetime, cancellation/abandonment, and cached retry, followed by scoped
lint/provenance/documentation/final specialist review.

The initial GREEN passes in compiled-expressions-green.log (2m37s
compilation, 0.02s execution, runner157.126s). Both nullable arithmetic
and computed/filter repeated-stream controls match SQL exactly, with only
one SQL execution per two stream batches. Added edge controls are in progress.
The first edge-control build failed in7.143s on a fixture-only struct-variant
pattern for Cancelled; it is corrected to Cancelled { .. }. No production
behavior changed for that compiler diagnostic.

The corrected edge controls PASS in compiled-expressions-edge-corrected.log:
1m11s compilation,0.14s execution, runner71.543s. Controls cover multiple and
empty records, zero results, SQL chunk/schema/metadata parity, dynamic schema
and Float64-to-Float32 changes, fresh clone runtimes, released nonempty input
buffers (each held raw Buffer has strong_count1), cast-error parity and cached
retry, aggregate/window/stable-function fallbacks, selected-UDF fallbacks both
declared and graph-selected, and actual mid-call cancellation and abandoned
futures after yielding. Both stopped-call controls emit nothing, release
per-call input/coalescer/result owners and retain a usable cache for retry.
Scoped core lib/test Clippy is now running; no passing runtime check is repeated.

The first scoped Clippy run failed in91.286s on one test-only ignored-unit
pattern in tokio::join!; the pattern now explicitly matches (). Product
execution is unchanged. Before final review, one added constant-projection
control exercises SQL/compiled parity with an actual zero-column Arrow input
carrying three rows and a source cursor, including preserved caller ownership
and SQL output cursor behavior. This new coverage justifies another focused
registration run; it is not a repeat of an unchanged passing check.

The final focused registration also PASSes in
compiled-expressions-zero-column-controls.log (1m11s compilation,0.16s
execution, runner71.362s). Constant evaluation on a genuine zero-column,
three-row Arrow input agrees with SQL and preserves the caller's source cursor
while producing the same output cursor behavior as SQL. The eight adjacent
expression controls PASS in compiled-expressions-adjacent.log (0.05s execution,
runner0.282s), excluding the already-passing augmented registration. These
cover the changed shared projection helper, exact schema/buffer/metadata/cursor
semantics, schema inference, invalid names and source-replay eligibility.

Rust formatting, diff whitespace and generated project-v3 schema no-drift pass.
The frozen complexity-final.json covers5Rust files/38changed-new functions,
maximumCCN8, source digest
`c14b9e88904d0d9e501e0e3605d69ac00bd52cb9c6a786edc6af004c021e8116`.
The initial strict omission assertion found two pre-existing Lizard omissions
(physical_plan_statistics and parallelism_decision); both complete top-level
source excerpts match HEAD exactly and their hashes are recorded. No changed
or new function is omitted. Scoped corrected Clippy and final review remain
pending. No performance acquisition, release build or remote action occurred.

Corrected scoped Clippy PASSes in compiled-expressions-clippy-corrected.log
(1m30s tool time, runner90.101s). No unchanged passing behavior test was repeated
for the unit-pattern style correction. All owned compile/formatter/lint commands
are settled. The doc writer reconciled only runtime-envelope#execution-paths,
correcting its stale arithmetic/filter SQL-only claim. Streaming-guide remains
accurate and no qualifying changelog/public-contract change was found. The
doc writer verified27local links/anchors, whitespace/newline/tables and schema
no-drift, with prior phase2.2 wording untouched. Final local specialist review
remains required; remote CI/90%coverage/release integration and all eight
performance targets remain unproven.
