# Issue 406: physical Join output projection

The complete eight-target objective and remaining acceleration phases stay
active. This increment starts from reviewed `90c323e9`, ordinary product
`995b1b9a`. The preceding ASOF comparison stopped before any observation;
its binary and stopped attempt retain their identities and are not retried.

## Problem and intended behavior

The stream compiler pushes exact column projections into ASOF. It currently
materializes every Join output column even when all downstream expression
consumers read only a subset. Issue 406 phase 2.1 explicitly includes consumers
that calculate expressions, rather than only simple selection.

Compile a physical Join output mapping from the union of its consumers' input
dependencies. Resolve names against the validated logical producer schema;
preserve deterministic original-column order. Consumers still calculate their
original projections, aliases, arithmetic and predicates against the narrowed
input. Analyze the stored parsed query without reading rows or reparsing per
batch. Unknown scopes, unresolved references and wildcard consumers retain the
full producer schema. Aggregates, window expressions, selected UDFs and other
batch-sensitive query shapes retain full input: narrower output can change
physical chunk boundaries, so dependency discovery alone cannot prove their
equivalence. Any consumer requiring full output prevents pruning.
An unconsumed Join port remains a full external output.

Function eligibility requires a known immutable scalar implementation from
the shared default DataFusion registry, as well as a row-local syntax shape.
Unknown, Stable and Volatile functions retain full input. A graph selecting
UDFs retains full Join output conservatively, including functions invoked by a
consumer whose own UDF list is empty; selected resources are registered across
the graph after this pass. ASOF keeps its existing exact-column eligibility.

The private projection must reach actual serial and owned parallel column
materialization; gathering everything and dropping columns afterward does not
satisfy this requirement. Output schema metadata, row counts and field order
remain exact, including constant-only expressions and zero-column intermediates.
Output preflight measures the selected physical fields, including nullable,
variable, dictionary and nested columns, without visiting unrelated columns.
Conservative existing single-chunk bounds may still decline to exact preflight.

## Contracts

- Logical graph declarations, public APIs, exported project-v3, logical
  fingerprints, Join matching/order, row IDs, drop/error priority and counters
  remain unchanged. Compilation validates the logical graph before projection.
- Retained input schemas, frozen V1 row/state charges and checkpoint formats
  remain unchanged in this output-only increment. Checkpoint continuation must
  reproduce the full Join's results followed by the same consumer expressions.
  Retained-state projection (phase 2.2) remains separate required work; completing
  this increment must not claim that phase or full issue acceptance.
- Caller buffers remain read-only. Output has the existing independent buffer
  ownership and dictionary compaction guarantees. Empty output still carries
  its exact schema; nonempty zero-column output retains its row count.
- Preserve match/output preflight before emission, edge row/byte limits,
  cancellation, declined worker fallback, credit lifetime and owned cleanup.
  New immutable projection metadata must be included in owned-work funding
  where copied; do not add per-row mappings or worker control injection.
- ASOF's existing physical projection and recovery behavior retain their
  current eligibility in this increment. Shared expression dependency analysis
  must not silently change ASOF retained-state scope.

## Verification and delivery

Start with recorded RED for Join physical schema pruning under exact and
arithmetic consumers. Reuse existing registrations to avoid the pinned Clippy
libtest-table limit. Add a materialized-column work counter proving unselected
kernels are not run, plus differential serial/worker cases covering both incoming
sides, multiple consumers, aliases, filters, constants, duplicate/reordered
internal projections, narrow output budgets, sliced/null/variable/nested/
dictionary payloads, worker refusal, cancellation and checkpoint continuation.
Check logical fingerprint/export stability and caller immutability.

Run focused affected tests and necessary format, lint and module complexity
checks, schema no-drift and whitespace verification, followed by final specialist
review. Full CI/90% coverage remain required remote gates. Any later performance
comparison needs a separately frozen proportional selection, exact ordinary
releases, fixed maintained AB/BA counts and a 600-second inclusive budget;
do not start or retry acquisition while controlled work competes for resources.
