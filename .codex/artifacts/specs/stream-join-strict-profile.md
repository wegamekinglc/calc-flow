# Stream Join: strict comparison and profile-guided optimization

## Scope and frozen inputs

Continue PR #393 in its isolated worktree. Preserve the caller's current
checkout. Compare original #393 `eff2fcad17b8922de6aa83955cc14ef602611a0b`
against #394 `8b9154103614bd3608d13dd30a45b2657af7ea73`, using one maintained
harness and independently sealed release wheels. #394 is a performance
reference; passing selected workload oracles does not resolve its previously
reported capacity-funding, error-precedence, nested-gather, owner-lifetime,
prefix-eviction, or wide-index issues.

The fixed workloads are 1M Lookup Join, 100k Lookup Join, and 100k retained
interval Join, all with the catalog's default batch size. Each comparison uses
two rounds of ten adjacent AB/BA pairs, the full-row oracle on every sample,
and the maintained paired confidence-interval rule. Each sample creates a
fresh runner with empty state. A worker process persists within a round.

Diagnostic and final phases share a 600-second measurement budget, including
startup, fixture creation, warmup, validation, profiling, and process cleanup.
Builds are separate. Stage caps are 250 and 350 seconds, each including its
cleanup reserve. Do not retry or extend sampling according to the result.

## Measurement evidence before implementation

The diagnostic phase completed in 55.047 seconds with no remaining owned
processes. Original #393 versus #394 P50 was 315.595 versus 119.975 ms for 1M,
35.491 versus 13.376 ms for 100k, and 251.730 versus 226.436 ms for retained
interval Join. Both rounds establish an improvement under the maintained rule.
These measurements supersede cross-PR comparisons of historical reports.

CPU diagnostics cover two exact 1M timed windows per version, excluding warmup
and CRC markers. Symbols are bound to module hashes and debug identities;
allocated ELF sections match their sealed native modules except build-id.
The original #393 profile has 2,696 timed CPU samples and 18 lost events over
the complete recording; the lost events cannot be assigned to a particular
window. #394 has 1,091 timed samples and no reported loss. Both have zero
missing stacks or unresolved native frames. Interpret these as directional
CPU evidence, not wall-time attribution or allocation counts.

The source and profiles identify repeated native dictionary lookup in both
pair counting and collection, borrowed-key hashing, and admission scheduling
as optimization candidates. #394 already resolves each distinct probe key to
an opposite dictionary ID once; original #393 resolves full key bytes again
for every window lookup.

## Required implementation invariants

1. Resolve opposite dictionary IDs once per distinct probe key. Use bounded
   IDs only while the opposite index is immutable; no cached ID survives an
   index mutation, eviction, or slot recycling. Preserve time-window
   inclusivity, out-of-order ordering, collision checks, match-limit priority,
   and count-before-output-credit admission. Fund actual added capacities.
2. Before selecting a faster hasher, give stored and borrowed keys the same
   canonical streaming protocol independent of input fragment boundaries.
   Keep one canonical key owner per distinct probe key and map physical rows
   through bounded IDs; clone key ownership only when retained state needs it.
   Do not add per-row heap buffers. Cover empty and long values, block
   boundaries, compound keys, timestamp units/timezones, collisions, restore,
   removal, and reuse. Serialized key bytes remain unchanged.
3. Within the existing generic path, hoist repeated admission work only for
   mask blocks proven fully admitted. A whole-block row-ID range must be
   validated before bulk reservation; retain original sequential handling for
   boundary or exceptional blocks. Append parent references, normalized times,
   retention bits, and physical source positions in bounded synchronous chunks.
   Preserve counter and first-error priority, including multi-record offsets.
4. Keep the existing Quantum thresholds. Any reduction in work charges needs
   an explicit account of work eliminated by bulk validation/reservation; do
   not simply lower the old per-row charge or raise a threshold. If that proof
   is unavailable, batch exact-budget grants without changing yield boundaries.
   Check cancellation at the same documented finite boundaries and commit no
   state or metrics on failure. Do not add a new native budget dependency to
   generic admission.
5. Preserve #393's capacity-aware funding, collision-safe recycling,
   surviving-owner refresh, head-cursor eviction, nested/dictionary output
   fallback, and wide output indices. Do not raise Quantum thresholds or
   weaken resource checks to improve a benchmark.

## CPU-lane investigation and deferred scope

The existing OwnedCpuWork service supports non-serial execution, but a minimal
optional admission hook does not preserve the whole preparation budget. A cold
rejected generation can leave shared Home controls charged. On success, an
additional input/scratch lease can consume the budget later needed for pair,
SQL fallback, or output admission. Fixing both needs a design for the complete
preparation and rollback lifetime, rather than a local submit fallback.

The incoming edge budget is released at dequeue, so a cheap input clone also
cannot substitute for a worker funding lease. Any future lane must cover the
actual shared backing extent through worker retirement and uninstalled results,
while preserving the original generic retained-row funding family.

This task therefore defers CPU-lane production integration. Preserve its
observed failing tests and fixtures as investigation evidence, explicitly
unimplemented; they are not passing acceptance tests. No runtime service or
pool contract is changed by this experiment.

## Verification and acceptance

Record focused failing tests before each behavior change, then run affected
Join tests and property tests, scoped lint and format checks. Obtain final
specialist source and evidence review. Freeze the optimized revision and
rebuild workspace packages with explicit invalidation and actual source-path
compile proof before measuring; preserve all invalid build receipts.

Compare the optimized version independently against both frozen versions on
all three selected workloads, using the same fixed protocol and new adjacent
pairs. Profile the optimized 1M path with the same diagnostic method. Report
per-case confidence intervals, uncertainty, memory observations, CPU sample
coverage, build time, and measurement time. Improvement versus original #393
alone does not establish that the #394 performance gap is closed. The issue's
absolute 60 ms target remains a separate claim requiring measured evidence.

Keep PR #393 draft while required CI checks are unresolved. This focused
experiment does not establish checkpoint, ASOF, full-suite, coverage, or
cross-platform acceptance.

## Second candidate after observed failure

The first candidate `1728cab907ba9eee6913ba7d5de163db2e5ef01d` did not meet
performance acceptance. Fixed paired results versus #394 confirm Lookup
regressions of +109.00% (1M) and +119.04% (100k); interval Join is inconclusive.
Comparisons versus original #393 are all no-confirmed-regression, not a
confirmed greater-than-5% improvement. Its complete failed measurements and
2,580-sample profile are preserved in
[the first-candidate report](../analysis/stream-join-strict-profile.md).

The profile and source comparison show repeated typed Arrow decoding and
framing in borrowed hash and equality. #394 encodes into a reused contiguous
buffer, then performs byte hashing and comparison. The first candidate also
keeps every old yield despite removing scalar classification work.

The second candidate must therefore:

1. Bind typed key columns at parent boundaries and encode canonical V1
   frames into one contiguous batch arena with checked row offsets. Hash and
   compare spans directly; create canonical owners only for distinct keys.
   Include multi-column keys, all supported types/timezones, sliced parents,
   Rowed/Packed/Legacy payloads and physical offsets. Do not add per-row heap
   objects. Keep collision-safe full-byte equality and distinct opposite IDs.
2. Preserve the existing probe reservation exactly:
   `P = 1024 + 12*N + 4*sum(logical_cell_charge + timezone_length + 64)`.
   Compute checked encoded lengths before reservation; allocate arena, offsets
   and typed bindings only after success. Prove P covers their overlap with
   canonical owners, vectors and actual hash-table/reallocation peaks; verify
   with the existing allocation counter. Use exact-size canonical copies and
   drop arena/offset scratch before allocating opposite slots. A previous
   successful budget must not fail merely due to a new scratch charge.
3. Replace microsecond normalization with a bulk copy. When the watermark is
   absent and retention is constant (threshold at either i128 endpoint), derive
   temporal masks from validity bits without per-row scans. Keep the existing
   scalar path and work charge when conversion or temporal scans remain.
4. Count eliminated work explicitly. A fully vectorized mask block may charge
   one bulk-copy visit plus per-key validity-word visits and its unchanged byte
   work. A proven all-admitted append may charge one visit/16 bytes per row,
   matching the existing append_copy_rows metadata-append operation. Mixed or
   ID-boundary blocks retain four visits/16 bytes. Keep Quantum thresholds at
   64 visits/4096 bytes and grants at most 64 physical rows. Reserve each grant's
   IDs only after its awaited work grant; check cancellation at each bounded
   yield boundary. Preserve first-error priority and
   leave state/metrics uncommitted on failure. Carry Quantum across all record
   batches within the uncommitted admission bundle so many small records cannot
   bypass the accumulated work boundary. Yield counts may decrease only
   with this documented removal of scalar operations.

Observe focused RED before these changes. Verify exact framing, allocation
peak/funding refusal, differential masks, bounded cancellation and ID overflow
order, then run the affected Join and property tests and scoped lint. Obtain
specialist approval before freezing the new source.

The second source gets new fixed pairs against both references on the same
three workloads and a new exact-window profile. This is a distinct source
candidate, not a resample of the failed source. Keep the maintained timing and
statistics method byte-identical. Record a separate plan and build seals.
The first diagnostic and failed-candidate phases consumed 180.186 seconds;
charge them to the same 600-second cumulative measurement budget, with a
350-second cap for this candidate. Preserve every result regardless of verdict.

## User scope correction

The user reiterated that delivery must focus on #394 and its affected Join
paths. Finish the current implementation without further exploratory designs
or broad local verification. The final candidate uses only the same three
fixed workloads paired directly against #394; omit the redundant comparison
against original #393. Keep two rounds of ten pairs, the full-row oracle and
the exact-window profile. The maintained harness and sampler dependencies
remain identical; record the reduced comparison selection in the driver.
Existing diagnostic and failed-candidate evidence remains preserved and charged
to the cumulative budget.

## Direct PR #394 row-buffer adoption

Frozen second candidate `118e6212b80bc1933bc400cc871916dacf81ff7b` also failed
the fixed comparison: Lookup 1M +50.78%, Lookup 100k +40.01%, both confirmed
regressions; retained interval is inconclusive. Its exact results and profile
are preserved in [the second-candidate report](../analysis/stream-join-strict-profile-bulk.md).

Follow the user's instruction to adopt #394 directly when no better design is
established. Replace the full-batch key arena and N+1 offsets with #394's
reusable row buffer: clear, encode, immediately lookup/intern. Retain typed
column bindings, D canonical owners, distinct opposite IDs, exact existing P,
actual peak funding, collision equality and V1 bytes. The buffer must cover
one row's maximum frame, including composite and multi-parent keys. Keep
allocation before/after reservation rules and credit refunds. Observe the
existing repeated-owner test fail on transient scratch above 1024 bytes for
129 repeated fixture keys, then pass after this direct transplant. Check only
that test, the existing type/encoding and actual-allocation cases, and forced
collisions; preserve every existing assertion and warning-denying lint.

Do not transplant #394's synchronous generic admission: it lacks bounded
Quantum cancellation and reserves all physical row IDs before timestamp
validation, reintroducing confirmed correctness bugs. This iteration leaves
admission, output and retained indexing unchanged.

Freeze the changed source and compare only the same three #394 workloads,
using the unchanged two rounds of ten paired samples and one warmup plus two
exact-window CPU profiles. The prior phases consumed 232.209295 seconds of
the cumulative 600-second measurement budget; cap this phase at 300 seconds.
Reject identical source/native seals to either failed candidate. Preserve
all outcomes without resampling. Final acceptance still requires performance
matching #394 except necessary bug fixes, PR #393's green required CI, Codacy
and resolved review, a verified merge into main, then closing PR #394.
