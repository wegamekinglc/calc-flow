# P3 same-input numeric Rows shared history — critic critique

Date: 2026-11-08. Source baseline: `8d8be7d907a095dbc0dc2b73dafcfdd842979cdf`.

## Target

- Spec: `.codex/artifacts/specs/ta-arroyo-p3-shared-history.md`.
- Scope: the first Float64 different-width dual-mean/difference sharing path,
  its numerical identity, execution/restore coverage and optional cache credit.
- No public API note, new numerical algorithm or state version is proposed.

## Verdict

**Proceed with caveats.** The proposed ordering admits a direct bitwise
identity argument, and the ownership/fallback model can preserve existing
budget and checkpoint contracts. No new numerical or protocol blocker was
found. Record the concrete cache facility/limit before implementing allocation,
then proceed test-first; the implementation and its real performance remain
unproved. Rebase on then-current main and recheck any merged P2 helpers rather
than assuming either P1a or P2 must merge first.

## Findings

### Blocking issues

None in the reviewed proposal. Clearing independent queues without replacing
all their readers, replaying accumulators during resource fallback, changing
logical fingerprint descriptors or silently disabling the existing parallel
path would violate the design and block the resulting implementation.

### Numerical and window argument

Assume corresponding shared/independent groups start with the same accumulator
bits and canonical rows. On a new row:

1. The input Option value has the same IEEE bits and NaN policy. Appending it
   to an already private, sufficiently reserved tape is a storage action; it
   must not add a new fallible operation inside the arithmetic sequence.
2. Each group executes the original add on that value, then the original
   removal on its own outgoing original-row value, in the same group order.
   The tape retains the widest outgoing position until every group is done.
3. Its repair predicate sees the same accumulator and post-expiry raw row
   length. The oldest-first view supplies exactly the values the original
   queue supplies to refold/rebase. Thus the same operations produce the same
   accumulator bits. The shared maximum length must not determine this test.
4. Output readouts execute in their original order after updates. Trimming
   no-longer-needed tape positions changes none of these readout inputs.

This gives the row-by-row identity required by FR2, including M2-triggered
repair and non-finite behavior, without using D13 tolerance to excuse changed
arithmetic. The same induction applies after reconstruction when the original
restored summaries/counts and window suffixes are preserved. The existing
uninterrupted-versus-restored tolerance remains separate; P3 does not promise
new bitwise identity between two routes that the baseline itself treats only
within tolerance.

### Significant implementation requirements

1. **Do not lose sharing facts when restore clears outputs.**
   `kernel.rs:803–815` clones the plan for `rebuild_history`, retains finite
   groups and clears readouts. `MeanKernel::compile` requires one or two
   readouts, so this replay can take generic fill even for a normal dual-SMA
   plan. Then `seed_restored_state` clones the reconstructed state and calls
   `force_stable_v2_rebase` before installing saved counts
   (`kernel.rs:840–879`).
   - Keep private sharing facts available and correctly indexed through these
     routes; do not rediscover them solely from a now-empty Mean output list.
     Forced rebase must obtain each group's own borrowed view from its entity
     owner. Copying a summary for this read-only repair does not authorize tape
     mutation or require duplicating a shared tape.
   - Initially gate sharing to the intended exclusive dual-mean/difference
     shape. Do not accidentally admit a larger mixed plan merely because it
     contains two matching Numeric groups. Any filtered/reindexed replay
     groups must retain a correct owner/view mapping.

2. **Adapt every direct update and parallel eligibility path.**
   Generic `TypedWindowState::update` and specialized
   `mean.rs:173–221` call numeric updates separately. Parallel lanes do so again
   at `entity_parallel.rs:915–975`, and `parallel_entity_state` currently checks
   the owned `state.samples.frame/len` at lines 546–565. Replacing two queues
   with empty placeholders while changing only one loop is wrong.
   - Route sharing once per entity row, preserve the original per-group
     repair points and readout order, and teach the existing lane predicate
     the valid shared representation. Do not force an otherwise admitted
     Rows5/20 dual-SMA plan to serial execution. Preserve cancellation/error
     hooks and row/group tags; candidates are private if a stop occurs after
     the tape append but before the last group completes.
   - Keep single-window/nonsharing dispatch on its original row loop, with
     sharing admission outside it rather than a new per-row map lookup.

3. **Resource refusal is a representation conversion, not a numerical reset.**
   The proposed shared-to-owned fallback is viable only by copying each existing
   accumulator field verbatim and populating its original window suffix.
   Replaying add operations or forcing a new refold/rebase would change West/M2
   bits. Keep the resident shared owner untouched until normal commit.
   - For owned-to-shared admission, the widest owned queue supplies the common
     suffix; preserve each summary rather than recomputing it. Neither
     conversion changes the entity transition counter or repair phase.
   - Tape coordinates need not be lifetime counters. A bounded suffix index
     can identify outgoing/view positions without introducing earlier overflow.
     Any chosen offset scheme must preserve this property. Allocation-size
     overflow in an optional sharing request should select the old path, not
     turn a previously valid query into a new semantic error.

4. **Keep actual cache credit separate from the logical state budget.**
   Reuse P2's cache-only facility if available, or name a small private
   MemoryPool/RAII facility and finite cap. Do not move logical history/buffer
   charges or checkpoint copies into that pool. `rolling/budget.rs:201–281`
   remains the established logical domain. Keep tape-owner metadata out of
   the generic WindowAccumulator/WindowState if adding it would change their
   sizeof-based history charge for unrelated declarations.
   - Credit each actual tape/container allocation, COW copy and reallocation
     peak before allocation. Shallow aliases share one allocation/credit;
     deep copies require a separate reservation. Credit remains with every
     actual owner until its last reference drops. Two copies are not a proof
     when arbitrary old owners are retained.
   - A refusal may allocate the old independent-queue representation according
     to its existing behavior, as the spec explicitly says; do not claim that
     those old caches are now physically budgeted. Do not clear the whole
     typed plan or rebuild neighboring arithmetic state to implement fallback.
   - Reserve the slide slot in a candidate copy. The existing Rows Clone does
     so at `kernel.rs:2563–2577`; losing it adds a warm append allocation and
     can erase the intended gain. Include added header/dispatch cost in the
     control rather than claiming literally zero global overhead.

5. **Storage views must not become new logical descriptors.**
   Keep TypedGroupPlan's existing Numeric/Rows descriptors, their order and
   readouts unchanged. Do not encode a storage policy as a new TypedFrame
   value in the fingerprint input. Private plan facts can select storage.
   Existing state codec stores logical projected history, not typed queues;
   reconstruct using the same retained rows and existing per-window rebase.
   - A Rows tape can omit event times because the Rows expiry/refold readers
     use positions and values. If conversion recreates old timed samples,
     demonstrate that their private time placeholders cannot reach a Duration
     reader or durable snapshot. All excluded families retain their old path.
   - Use pinned baseline fixtures for old-reader/new-reader continuation and
     compare subsequent values. A new-code round trip alone is insufficient.

### Minor / measurement notes

- Existing diagnostics historically size Float64 queue samples by Option<f64>
  rather than TimedSample<f64> (`kernel.rs:2751–2755`). Keep the old metric's
  basis for nonsharing paths and report real capacity/credit/copy bytes
  separately. Neither the old estimate nor a width ratio establishes RSS.
- Holding an old Arrow output table does not necessarily hold a typed tape
  owner. If an ownership performance case claims retained kernel state, prove
  that actual ownership. Otherwise cover held owners in focused native
  allocation/credit tests and measure the real sparse-append workload only.

## Test and performance assessment

The spec's focused acceptance list is sufficient, with the execution-path
requirements above. Before merge the tests must show accumulator bit equality
under both profiles, correct per-window lengths and wraparound, same-frame and
single-window controls, parallel admission/selected-error compatibility,
failed-emission rollback, held owner credit, fallback conversion, legacy restore
and forced rebase. Current source tests already make the M2 and spare-capacity
properties load-bearing; they must remain passing.

Prefer the first two proposed measured cases: a real warm same-input dual-SMA
append and a matched single-SMA control. Add the sparse/held-state third case
only if the maintained harness supports its real ownership and it answers a
necessary risk. Confirm the exact generated case list before execution; CLI
scale arguments alone must not silently add a broad matrix.

Fix the source/build/native and harness identities, window widths, history,
append/entity shape, CPU/worker configuration, GC mode, exact fixed sample
count and total budget before acquisition. Use one maintained statistical
contract unchanged: standalone warm 30 pairs or release's two rounds of ten
fresh-process AB/BA pairs. If the selected warm tool reports ratios/intervals
rather than an `improved` enum, preregister the actual delivery rule before
measurement; do not relabel a different tool's verdict. All chosen cases total
600 seconds including setup/correctness/cleanup, with builds reported separately.

The user requires real end-to-end improvement; lower copied payload alone
cannot authorize merge. Report actual copy/allocation/credit/peak-memory and
latency results separately. A confirmed control regression or inconclusive
target cannot establish the requested improvement. Do not expand/retry the
case set for a favorable result.

## Axis check

- **Correctness:** the same-value/add/remove/repair sequence supports exact
  identity; every shared-window reader must use the correct group view.
- **Hidden assumptions:** no shared owner crosses entities or concurrent lanes;
  null positions and the global NaN policy remain unchanged. No lifetime tape
  ordinal may introduce a new earlier overflow failure.
- **Missing edge cases:** the proposed tests cover the significant arithmetic,
  wraparound, retry and restore cases. Keep output-cleared replay, forced rebase
  and actual parallel eligibility explicit.
- **Backward compatibility:** viable without a new format or hash mapping if
  logical descriptors remain unchanged. Actual old-source continuation and
  candidate old-reader acceptance remain implementation evidence.
- **Performance:** unmeasured. Reduced duplicate queues/copies can help, but
  header, credit, view and dispatch work can offset it; the control is required.
- **Surface and ergonomics:** OK; no public schema/API/budget change is needed.
- **Test plan:** concrete and testable without a full local matrix. Different
  execution paths must be exercised rather than inferred from one fast path.
- **Risk and scope:** the exclusive dual-mean/difference path is coherent.
  Excluded groups stay excluded; do not silently weaken their behavior to
  accommodate the new representation.

## Verification and handoff

Read the full spec, baseline Numeric/queue/repair code, direct Mean fill,
parallel lane and eligibility code, typed reconstruction/forced rebase, logical
budget and the existing M2/allocation/retry tests. Cited source paths, whitespace
and final newline checks pass. No build, product test, performance run, source
edit or remote mutation was performed for this critique.

Proceed to test-first implementation with the concrete cache facility/limit
recorded and all update/read/restore routes accounted for. Return a discovered
public budget, identity or numerical-order change for review; real speedup,
final specialist review and required CI/Codacy remain merge gates.
