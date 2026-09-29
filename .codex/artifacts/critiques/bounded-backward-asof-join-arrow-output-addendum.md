# Bounded Backward ASOF Join: Arrow Output Addendum

## Scope

This addendum reviews the columnar ASOF performance revision against the
[original critique](bounded-backward-asof-join.md), the revised
[specification](../specs/bounded-backward-asof-join.md), and the revised
[API note](../api-notes/bounded-backward-asof-join.md). It does not revise the
original pre-implementation verdict.

## Changed Premise

The original C2 and Correctness/Performance axis findings assumed that
DataFusion would perform candidate-table equality, association, projection,
and output ordering. The stream-only ASOF operator now selects one candidate
with its native typed index and gathers final rows directly from retained
Arrow batches. DataFusion remains the engine for general table expressions
and SQL. The old DataFusion materialization probe and pool test therefore do
not establish the new output path's semantics or memory bound.

The revision also introduces an ASOF-internal version 2 columnar checkpoint:
one index segment and one IPC segment per retained payload batch. Version 1
snapshots remain readable. The original C3 recovery concern continues to
apply to this representation.

## Retained Acceptance Conditions

- Typed key and sequence comparison, inclusive backward choice, left-row
  order, exact output schema, nullable unmatched right fields, and every
  supported flat payload type must match the independent oracle and the public
  ASOF contract. Direct Arrow gathering may assemble selected rows; it must
  not become a second general table-expression or SQL engine.
- Admission, output gathering, IPC/index encoding, compaction, and restore
  must reserve their transient allocations before use. Persistent state must
  charge shared payload batches once and every operator-retained segment at
  its actual capacity, including a drained checkpoint base until compaction.
  Output edges keep their separate row and byte budgets.
- A failed or cancelled admission or checkpoint preparation must preserve a
  retryable committed state. A failed output send must not commit the pending
  prefix. Managed barriers must await preparation, then capture prepared
  immutable bytes without bulk synchronous work; a preparation failure must
  forward neither checkpoint acknowledgement nor barrier.
- Restore must reject corrupt or incompatible metadata and payloads before
  installation, preserve the managed wrapper's progress contract, and cover
  version 1 migration, version 2 repeated capture, and cancellation after
  accepted output prefixes. Existing inner Join identity and fingerprint
  vectors remain independent.

## Critic Conclusion

The revised division of work is consistent with the single DataFusion engine
boundary because Arrow assembly is confined to this stream-specific operator.
The original C2/C3 evidence gates remain in force under the version 2 accounting
and layout. The implementation review found no remaining blocking contract
issue after the retained drained-base charge, worker-held output reservation,
asynchronous checkpoint digest, and failure-path fixes. Focused local checks
passed: 44 ASOF unit tests, the operator-task preparation-failure test,
`cargo clippy -p calc-flow --lib --tests --no-default-features -- -D warnings`,
`cargo fmt --all --check`, and `git diff --check`. The author also reported
green empty-input and non-null flat-payload integration tests. Full CI,
cross-platform, release, and benchmark gates remain separate evidence; this
addendum does not claim they passed.
