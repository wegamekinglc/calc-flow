# Inline grouped integer SUM state study

The user authorized continued performance research, focused verification, PR
repairs, and merging after the final specialist review and required green CI.
The preceding parallel prototype was rejected; its measurements remain sealed.
The new comparison starts from main
`222f4e2d88f56b5b4ce635abab24eceed732da6d`.

## Bounded implementation

For exactly one admitted built-in grouped Int64 or UInt64 SUM, test whether one
inline scalar can replace nested heap state vectors and the duplicate result.
The result is derived from the same scalar. Keep all other aggregate shapes on
the existing representation. This is an internal optimization with no public
API, project schema, checkpoint policy, or serialized state layout change.

The chosen state is a compact typed nullable integer, with stack views at scalar
export/seeding boundaries. A pinned-compiler layout probe confirms unchanged
Group (96 bytes), Candidate (128 bytes), and map-row (136 bytes) sizes relative
to the old representation. Tests must also include container capacities in
owned-memory accounting; per-group heap counts alone omit those containers.

The implementation must preserve null and wrapping semantics, aggregate and
input filters, sparse updates, first-appearance order, complete snapshots,
checkpoint capture/restore, and atomic transaction admission and commit.
Cancelled or refused work must leave committed state unchanged. Memory
reservations must pay for the actual representation and owned capacities.

Record a focused failing test before implementation, then verify the intended
allocation/storage reduction and affected semantic, memory, and recovery paths.
Use scoped formatting and strict lint checks. Full regression and connector
coverage remain CI responsibilities.

## Measurement and delivery

Reuse the maintained SUM harness and paired-statistics contract. Before running,
freeze exact baseline/candidate sources, identical common harness, selected
cases and rationale, fixed sample counts, provenance, and a 600-second total
acquisition budget. Builds are estimated and reported separately. Select at
most five cases directly affected by this change or necessary controls.

The former `af400` product is incompatible with this baseline's compiled source:
main now includes DataFusion input-partition changes. Build baseline and
candidate afresh with the identical audited harness. Initial build estimate is
20–30 minutes combined; this is separate from the acquisition budget.

The selected cases are dense single-CPU SUM (allocation target), dense four-CPU
SUM (the current main branch's thread convention), and sparse single-CPU SUM
against retained groups (regression control). Run two rounds of ten alternating
pairs per case: 120 formal observations and 12 warmups. Capture one dense
single-CPU profile process per arm, six jobs each with the first job excluded.
Use one DataFusion partition, default input partitioning, and the identical
audited `common657` harness. The eight-CPU prior experiment stays unchanged.

The frozen runtime acceptance requires complete source, correctness, resource,
and profile evidence, an exact maintained `improved` verdict for at least one
dense target, and no `regression` verdict for any selected case. Do not add a
requirement that every other interval exclude the regression threshold;
inconclusive cases still cannot establish equivalence or speedup. Generic
aggregate latency, UInt64 latency, and checkpoint performance remain unmeasured;
focused allocation, correctness, capacity, and recovery tests cover them.

Correctness oracles, resource measurements, processing-window profiles, source
and binary seals, cleanup, and uncertainty reporting remain mandatory. Stop
on unsuitable conditions or budget exhaustion; do not retry for a favorable
verdict or increase samples. Missing evidence does not establish improvement.

Only ship a reviewed implementation supported by the frozen comparison. If
the implementation regresses or cannot establish its claimed benefit, retain
the evidence and report that limitation rather than claim a performance gain.
Any final PR must pass required checks on its exact final head before merge.
