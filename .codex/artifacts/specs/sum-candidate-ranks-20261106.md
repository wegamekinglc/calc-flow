# Grouped SUM candidate/rank study

Continue the authorized test-first, fixed-budget performance study from latest
main `1fcafde11a247b1be275d3cb6ab41a59476d6832`. Preserve the original dirty
stock-comparison checkout and every prior sealed experiment. The user has
already authorized fixing review/CI issues, waiting for final required checks
and merging after independent approval and green CI.

## Hypothesis and scope

Investigate lookup and container costs in the existing grouped integer SUM
transaction. Historical group lookup is already cached per distinct touched
key. Partial groups already carry dense ranks; candidate payloads are indexed
by global integer slot in a HashMap. Determine whether existing ranks can
remove unnecessary slot lookups or whether duplicate hashing/rehash work can
be removed without introducing a second string-key dictionary.

First audit the complete call path and pick one minimal prototype. A dense
candidate design is a hypothesis, not an approved implementation. New
candidate storage must scale with groups touched by the current transaction,
not all historical groups. Preserve the canonical keys, first admission order,
generic paths, state/checkpoint representation, public API and thread policy.
Do not combine expression-filter, batch-parallel or key-interner changes.

Main includes PR #548's empty-accumulator removal, skipped inline finalization
and typed integer output, and PR #549's batch SQL policy. Retain them. The
previous pre-security-fix profiles establish only a research direction;
they do not establish the current baseline's phase costs or improvement.

## Required design and correctness gate

Before production edits, state exact eligibility, storage/rank mapping,
per-phase lookup counts and lifetime, and a capacity-based full owned peak.
Prepay old/new growth overlap, maps, vectors, indices, candidate payloads and
simultaneously retained coordinator state before allocation. Preserve actual
group and byte limits, finite/custom pool behavior and failing field/error
order. A post-allocation size check alone is insufficient.

The conditionally admitted TDD design applies only to the existing
`inline_sum.is_some()` grouped Int64/UInt64 SUM route. Retain a lightweight
global-slot-to-rank map and store candidate payloads in a rank-ordered Vec.
Align payload capacity to the small map's actual capacity, not an independent
power of two. Partial merge accesses candidates by rank; full snapshots still
use the small slot map. Generic/global/float/mixed and ordered aggregate
LAST_VALUE paths remain legacy. Post-aggregation HAVING and output ORDER BY
retain the existing single-SUM admission; they are not ordered aggregates.

After merge, move the existing partial rank-to-slot Vec into the candidate
store for safe owned zip iteration. Do not add an Option payload array,
duplicate the slot Vec, sort the whole touched set, or retain the complete
partial lease. Existing candidate credits must independently cover the moved
slots before partial credits release. Keep all existing fees: actual
old/new map, payload Vec and slot growth peaks must fit them. Do not assume
whole owned memory or allocation count improves from a narrower map entry.

Require a meaningful baseline RED for actual merge slot lookups (a generic
positive control still performs them), independent value checks and full
container/output/teardown allocation accounting. Verify unfavorable capacity
phases in a single bounded growth trace, not only favorable cardinalities.

Observe a focused meaningful RED against the baseline before implementation.
Use actual operation/allocation evidence for the changed path and do not infer
speed from allocation counts. Preserve and test NULL/empty/Unicode key
identity, first appearance, WHERE versus aggregate FILTER admission, signed
and unsigned wrapping/null SUM behavior, sparse updates, historical ranks,
held outputs, cancellation, discard/refusal/retry, recovery and generic
aggregate behavior. Caller inputs remain read-only; failed transactions and
leases must settle without installing state. Keep unsafe code forbidden.

Use focused Rust tests and necessary formatting/strict lint/complexity checks.
Full workspace, connectors, cross-platform, coverage and packaging gates
belong to final CI. Final specialist review remains required.

## Frozen measurement proposal

Before any acquisition, finalize the cases, fixed counts and protocol with
the performance specialist and reviewer. The initial proposal uses five
maintained controls: dense one CPU, dense four CPUs, sparse one CPU,
all-distinct-within-batch one CPU, and two occurrences per key one CPU.
The last case is a moderate-repeat control, not an admission threshold.
Use the existing smallest stock fixtures/common harness and independent
twenty-one-snapshot oracle where compatible; no new general framework.

The proposed fixed count is two rounds of ten alternating pairs per case:
200 formal observations and twenty warmups. Both dense single-CPU products
require six profile jobs, excluding the first, with five accepted processing
windows and maintained timestamp/loss checks. Separate precise path evidence
from generic/inlined symbols; missing anchors are uncertainty, not zero cost.

One 600-second acquisition, prior debit zero, covers fixtures, preflight,
startup, warmup, independent correctness, resources, profiles/decoding,
analysis, local publication, post-seals and cleanup. Estimate and record
builds separately. No additional diagnosis run, fixture generation or profile
is silently excluded. Do not reuse previous studies' residual budget.

Freeze public source and helper/protocol identities before acquisition. Build
the current-main baseline and reviewed public candidate in permanent frozen
trees with one identical audited harness. Require actual fresh core/example
compiler records, dependency proof, public Git-blob equality and source/binary
pre/post seals. Old products cannot be relabeled as current builds.

Run no acquisition until the exact source/build/helper/protocol packet passes
independent review and the parent gives explicit GO. The existing in-budget
quiet guard must succeed, and no unrelated process may be killed. Stop on an
unsuitable environment, failed evidence or deadline; settle owned processes
and preserve incomplete results without retries, case expansion or tuning.

Accept a runtime change only with complete evidence, at least one dense target
having the exact maintained improved verdict, and no confirmed regression on
any selected case. Use the maintained paired rule; inconclusive is neither
equivalence nor improvement. Reject and archive an unproven prototype,
restoring the runtime while preserving reproducible source and evidence.
