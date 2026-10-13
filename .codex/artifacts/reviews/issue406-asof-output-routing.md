# Issue 406: ASOF output routing review

**Branch:** `feature/issue406-asof-output-routing` → `main`
**Base:** `ac5d7dffabe098d33ee9c07c912d3c6c63898552`
**Reviewed branch head:** `8c6096e4ed13360b67b352b8219bf5dba5f850c9`
**Routing implementation:** `c7b964bc`, previously reviewed as `5aa03e54`
**Benchmark-proof implementation:** `9f6158ea`
**Measured candidate:** `6739fa2b5573f4e184266793fd9ee4d4281a5b14`
**Evidence follow-up:** analysis addendum reviewed over `8c6096e4`; no new source

This review concerns the existing output-routing worktree. The older
output-matching increment and its measurements are separate evidence and do not
serve as this branch's comparator. No PR status was queried or remote action
taken. The implementation owner retains the worktree.

## Summary

Eligible monotonic output runs resolve a small chunk dictionary once into a
fixed array of borrowed buckets and local cursors. Runs require at most 64
dictionary slots and at least four selected rows per slot. Generic matching,
its cursor map and reservation, and parallel probe capture remain unchanged.

The routing implementation, pre-EOF checkpoint-proof correction and final
completed/stopped evidence report have no outstanding review blocker. The
correction preserves actual committed manifest bytes before terminal retention
can remove them, while retaining the original processing-completion window.

## Build and test results

- Rust: the existing ASOF log records **341 passed**, zero failed, in 26.27
  seconds. The existing all-feature core/library/test Clippy log completes
  successfully in 1m 26s. These records retain their original implementation
  identity; they are not new runs on the rebased head.
- `git diff 5aa03e54 c7b964bc -- crates/calc-flow/src/operator/asof` is empty.
  The unchanged ASOF tree supports reuse of the prior focused evidence without
  repeating tests or rebuilding native code for review.
- Python benchmark correction: inspected owner logs record **30 tests passed**
  in 0.249 seconds. The recorded RED has five expected missing-helper failures
  and one existing passing control. Owner-reported focused Ruff check/format,
  schema-drift and whitespace checks pass.
- Reviewer verification was read-only: source and focused-test review,
  unchanged-tree comparison, evidence/provenance validation, recomputed paired
  comparisons, schema-drift and whitespace checks. No product test, native
  build or measurement was launched by this reviewer.
- Coverage and current-branch required CI: not certified by this review.

## Blocking issues

None in the reviewed routing implementation, benchmark correction or final
report. Current CI, incomplete ON qualification and unmet issue-wide targets
remain limitations; source approval does not certify them.

## Style issues

None found in the reviewed implementation. The new Rust test scenarios are
called by an existing regression, retaining the pinned toolchain and lint
policy while avoiding the documented generated-libtest array limit.

## Correctness and ownership

- `OutputRun::keyed_rows` uses the same `run.start + offset` range, event times,
  compact key IDs and payload positions as the ordinary row iterator. Its
  selection counts dictionary slots, including retired holes, so cached IDs
  remain bounded by the fixed array.
- Each routed run starts local cursors at its own first event time. Runs retain
  canonical order, inclusive tolerance, sequence dominance and identity-only
  suppression through the unchanged `candidate_monotonic` implementation.
- Routed runs do not advance the generic cursor map. A later generic run has a
  nondecreasing event time and catches up through the existing monotonic
  advance operation. The explicit 201/1/1/1/318 routed/generic transition
  regression covers this boundary.
- `append_monotonic_run` checks cancellation and source funding before selecting
  each candidate, preserving the global 1,024-row yield cadence and output
  reference order. The local route array adds no retained heap state, does not
  mutate caller buffers, and does not escape matching.
- Prefix accounting and checkpoint/status installation remain on their existing
  preflight and collector-acceptance boundaries. Worker ownership and parallel
  probe capture are untouched.
- `measure_acceptance_job` now requires `state_root`; all actual callers supply
  it. ON capture starts after the unchanged processing timer and before EOF,
  and its I/O counts toward full-cycle and acquisition time.
- `_capture_processing_checkpoint_evidence` reads real committed manifest JSON,
  records its exact content and hash, and limits accepted epochs to the captured
  `last_completed` status. Concurrent retirement records missing evidence
  without retry; other read or parse failures propagate.
- `_snapshot_manifests` validates capture bounds and reconstructs all projected
  fields from the recorded JSON, rejecting altered hashes or source progress.
  `_advancing_processing_epochs` still requires nonterminal manifests, positive
  rows and completion inside the original processing window. Snapshot time
  cannot promote a later checkpoint completion into that window.
- `_snapshot_processing_checkpoints` settles its owned `to_thread` reader before
  propagating cancellation. Final retained-file inventory remains separate from
  the pre-EOF proof scope; default retention and native checkpoint semantics
  remain unchanged.

## Test coverage

The counter regression records 1,026 hashes before the change for 1,024 rows
and one key. Differential controls compare complete output plans, payload
positions, null flags, spans, rowwise ownership, projected inventory and journals
with binary search. They cover 1/63/64/65-key dictionaries, retired slots,
partial consumption, canonical string sequences, overlapping chunks, generic
fallbacks, identity-only rows and missing matches.

The benchmark regressions use real manifest files to preserve a pre-EOF proof
after retention removes the file. They reject outside-window completion, empty
source progress, uncommitted and terminal manifests, altered proof content and
invalid capture bounds. They also cover concurrent retirement without retry,
capture-before-EOF ordering and cancellation waiting for the owned reader.
The full directly affected module retains its lifecycle, observer and cleanup
controls. These focused proofs support approval of the correction; they do not
establish a completed ON performance comparison.

## Existing measurement evidence

The immutable attempt under
`target/issue406-routing-current-main-20261013/` compares exact merged main
`ac5d7dff` with the ordinary candidate release from `6739fa2b`.

Four cases completed two rounds of ten adjacent AB/BA pairs, with one warmup
per arm per round: 160 measured observations and sixteen warmups. Stored
classifications are `improved` for OFF/64k/1 CPU, `inconclusive` for
OFF/64k/32 CPUs, and `no-confirmed-regression` for OFF/1024/1 CPU and
LOW/64k/1 CPU. No complete-set performance verdict is justified.

ON additionally retained two successful warmups and one accepted baseline
sample. Its candidate sample has a successful oracle and lifecycle but lacks
qualified advancing-epoch proof: epoch 1 completed during processing, while
the final retained manifests contain only epochs 2 and 3. The strict validator
stopped the attempt, and no ON paired classification exists. All 180 raw
observations remain; only 179 have normalized metrics.

The reviewer independently checked all **400 result-evidence hashes** and
**3,100 frozen input hashes**, validated both ordinary release identities and
their complete **294-compiled-artifact** inventories, and reproduced all four
stored paired comparisons. Baseline/candidate native hashes are respectively
`ac9686cd24c69199c969ff61ed4ed8af6d3d37c849f42d786801e4c7f122f6b6` and
`a131c569adaa5fdb629e868be6af42ca0e5a3e31359694a37a39116930340011`.
The independent audit records successful output oracles, lifecycle and cleanup
for all 180 raw observations, including the unqualified ON candidate.

Inclusive elapsed time is **96.04917656 / 600 seconds**. All owned groups
settled, with zero forced kills, surviving groups or cleanup errors.
Stopped/unqualified ON evidence remains immutable and visible; the new
correction cannot relabel it as a successful five-case acquisition.

## Documentation consistency

The analysis distinguishes the rebased implementation from historical measured
products and preserves all eight targets. It reports the four completed
comparisons, OFF/32 uncertainty and incomplete ON classification accurately.
ASOF OFF medians of 112.49 / 169.66 ms still miss the 100 / 30 ms targets; the
other seven target rows are unmeasured. LOW/ON budgets derive from the OFF goal,
and small-batch ASOF timings do not prove matched-harness overhead.

Processing CPU and thread diagnostics retain their read-boundary and retired
thread caveats and remain distinct from whole-job CPU. The 294-artifact audit
is described as an artifact audit, not test execution. Earlier pending source
handoff statements and the spec's PR #420 planning context are explicitly
historical. The correction's 30-test evidence and separate incomplete ON
diagnostic are reported without repairing either stopped raw run.

## Separate ON diagnostic addendum

The follow-up under `target/issue406-routing-on-proof-20261013/` freezes the
corrected harness at `8c6096e4` with plan SHA-256
`30ea8ef361cb0c387d70487e7b074911c555b93524f5debb36a3b13a435724c6`.
It selects only ASOF 1M/input, 64k, ON/100 ms, one CPU, reusing the same native
products without builds. Its fixed plan retains two rounds of ten adjacent
AB/BA pairs, one warmup per arm per round and a 600-second inclusive budget.
The wrapper requires the actual validated pre-EOF snapshot even when retained
final manifests could independently qualify the sample. Owner evidence records
six passing focused wrapper tests after the two expected guard regressions.

This separate attempt is also **STOPPED**, at candidate round 0, sample 2.
All eight raw observations pass the output oracle, lifecycle and cleanup and
contain a valid pre-EOF snapshot of nonterminal epoch 1 with positive source
progress. Seven qualify: two warmups and five measured observations. Two pairs
are complete; baseline sample 2 is unmatched. No ON paired statistics or p50
comparison are justified or present.

The rejected candidate's processing duration is 136.749381 ms. Its observed
epoch-1 completion interval, relative to processing end, is
**[-7.851084 ms, +0.465852 ms]**. Its real snapshot contains 896,000 rows for
each source. The interval crosses the boundary, so the unchanged whole-window
predicate returns no completed or advancing processing epoch. This evidence
cannot establish whether completion preceded processing end. It is a timing
qualification limitation, distinct from the first attempt's lost manifest.

The reviewer verified the plan identity, all **3,099 frozen input hashes** and
**24 result-evidence hashes**, revalidated all eight snapshots from their raw
JSON, and reproduced completion windows, advancing-epoch results and processing
CPU clock bounds. Inclusive elapsed time is **12.601763956 / 600 seconds**;
all groups settled with zero forced kills, survivors or cleanup errors.
The analysis addendum agrees with these independently checked facts.

The corrected snapshot mechanism works on actual runtime output. Full ON
qualification remains incomplete. Both stopped attempts and every raw sample
remain preserved, with no retries, filtering or sample expansion. No other
case was rerun, and the attempts do not form a completed original five-case
acquisition or eight-target acceptance result. No new source blocker follows
from the strict stop.

## Verdict

**Approve** the reviewed routing source and benchmark correction at `8c6096e4`,
and the completed/stopped evidence report with its reviewed analysis addendum.
No outstanding correctness, style or reporting finding remains. ON fixed-count
qualification is incomplete, OFF/32 is inconclusive, the eight-target objective
remains unmet, and current required CI/coverage is not certified. This approval
does not establish merge readiness.
