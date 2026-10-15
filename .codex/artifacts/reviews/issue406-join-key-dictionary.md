# Issue 406: Join persistent key dictionary review

**Verdict: Approve — final local specialist source review.**

**Feature:** `feature/issue406-join-key-dictionary` at
`dcaef75486c10c3eef91e1f2a47b9f1f00f8d08e`, against
`798a97ee44b701b00bb89196a6483b01c62e16bf`.
**Delivery:** `feature/issue406-key-dictionary-delivery` at
`db4f5ec7bc6be28dc7f2f973d3342aa092710dda`, including main
`dac0eb1a868452be81f18a3dd12f0318d40db379`, documentation commit
`10a9c89f04c1c00ba8444e18a8e77ea3375a691d` and the final restore fix.
**Changed Rust seal:**
`1add6adfb5b2acb0a04a50e2fc4b2ae8f7d031f93201eb2ccc7120a856eaa8e6`.

## Summary

Join now stages persistent key identities shared across both sides, binds typed
columns once per admitted segment, and preserves full V1 logical charges.
Funded dictionary/index acceleration can decline to complete SQL matching;
legacy-row bindings, captured native indices and immutable leases govern
identity lifetime. Recovery derives identities only after historical validation.

This review covers the persistent-dictionary increment, the final specification/
guide/changelog patch, the B1 restore correction and delivery of their exact
source. It reuses the bounded source pre-reads and examines the final differences,
including the in-place admission recovery refactor. It does not reapprove
unrelated main changes or the earlier issue increments.

## Build and Test Results

- Rust at `ca3de78c`: supplied affected run **Passed**, 141 distinct tests, zero failures,
  including 40 persistent-dictionary controls. The raw log's test names exactly
  match `validation-final.json`; compilation was 2m16s and execution 3.65s.
- B1 correction: `duplicate-id-root-green.log` **Passed**, one changed registration
  exercising five restore routes, 1m54s compilation and 0.02s execution
  (114.927s recorded command elapsed). Its exact command was
  `cargo test -p calc-flow --lib --features benchmark-internals operator::join::tests::persistent_dictionary_restore_tests::duplicate_id_tests::test_persistent_dictionary_restore_rejects_duplicate_side_row_ids -- --exact --nocapture --test-threads=1`.
  Two unchanged controls for prior payload-error priority and authentic-layout
  continuation passed in `duplicate-id-resume-green.log`; that acquisition
  remains **2 passed, 1 failed**, with the failed main control superseded by the
  final run. Its three production restore files match the final fix exactly.
- Final Clippy: supplied **Passed**, using
  `cargo clippy -p calc-flow --lib --tests --features benchmark-internals -- -D warnings`
  in 90.590s (`duplicate-id-root-clippy.log`). The earlier 1m46s result belongs
  to `ca3de78c`. Neither is an all-features/workspace result.
- Formatting, project-schema no-drift and whitespace: supplied **Passed**.
  Final `complexity-review-fix-final.json` records 58 changed Rust paths,
  464 changed/new functions, maximum CCN 8,
  no changed-function parser omissions and three exact unchanged exclusions.
- Python: focused validator/preflight and Ruff passes are recorded; some earliest
  outcomes have captured tool output rather than retained raw logs. No full
  Python suite or newly built Python native package is certified.
- Maintained correctness diagnostic at `ca3de78c`: the actual non-`cfg(test)` core dev build
  **Passed**. It records a real dictionary `try_grow` refusal, two available SQL
  grows, retained rows 2/1, full fees 228/114, the sole `a/a` output, and owned
  cleanup with pool zero. The raw JSON equals its provenance observation and
  the executable hash matches
  `5c0080e7e06b61943b96d4921197855584e10b26a00c624e21fdeec5a808d39e`.
  The missing-`--check` invocation fails as required. This is correctness-only
  dev-profile evidence, not release or performance evidence. Its binary and
  input aggregate are not relabeled as a final-fix rebuild; the bounded restore
  correction leaves the diagnostic and `process_data` paths unchanged.
- Reviewer execution: read-only source/artifact inspection and independent
  source/evidence comparisons; no builds, repeated tests, benchmarks or CI calls.
- Unresolved new failures: none. Authentic B1 RED failures and intermediate
  accounting-fixture failures remain recorded separately from final GREEN.
  The closed-home fixture's self-wait interruption is not a completed RED run.
- CI, combined coverage and full regression suites: **Not run by reviewer**.

Evidence is under the repository-root
`target/issue406-join-key-dictionary-20261015/`, principally
`validation-final.json`, `final-focused-green.log`, `clippy-final.log`,
`complexity-final.json` and `diagnostic-native-control-provenance.json` for the
original revision; `validation-review-fix.json`, `duplicate-id-root-green.log`,
`duplicate-id-resume-green.log`, `duplicate-id-root-clippy.log` and
`complexity-review-fix-final.json` identify the bounded correction.

## Blocking Issues

None remaining. B1 is resolved as described below.

## Style Issues

None requiring a separate change. The restored identity invariant is now checked
before sidecar construction. No lint suppression, production boxing for complexity,
dependency change or weakened workspace lint was introduced.

## Test Coverage and Resolved Findings

The recorded focused controls cover all 17 V1 tags, composite/sliced/nullable
keys, long values and forced collisions, physical row-ID gaps, projected nested
payloads, ordered matches, full fees, state/match limits, optional refusals,
reduced-domain exhaustion, checkpoint layouts and history, and cleanup/retry.

The earlier retirement-chain blocker is closed: backlinks make unlink touch at
most the removed entry and its two neighbours, with a discriminating 256-entry
tail-eviction control. Ordinary known-reference preflight uses a batch-touched
list, rather than scanning unrelated dictionary entries.

The earlier V1 capture-refusal publication blocker is closed: metadata/schema/
base/segment/UTF8 decisions retain their actual workspace/descriptor owners,
finish cleanup and check cancellation before synchronous installation. The
actual DerivedWork wait selects cancellation/deadline; its stack capture guard
discards private replacements on failure or dropped futures. New dictionary
construction occurs in the owned worker, without adding managed actor-side
O(D) `block_on` work.

Real Count/Fill input gates and the delivered Fill-result gate exercise the
production ownership chain: effective legacy binding to leased KeyRun, captured
NativeIndex, ProbeInputs, worker and outer result installer. The proofs retain
Weak identities and byte-only checkpoint history, not an extra strong test
lease. Reset, pending mutation, actual release, healthy retry, identity retirement
and independent byte ownership are distinguished. Pair outputs contain row
positions and need no redundant identity lease of their own.

The in-place SQL/native recovery refactor returns admitted ownership before
retrying. Prepared dictionary keys survive native/capture/submission decline,
and non-budget failures preserve their source rather than selecting SQL retry.

B1 is closed. Previously, V1 validation rejected only duplicate
`(event_time, row_id)` pairs, allowing two same-side live rows with one ID and different times
to reach `SideBindings::insert` and panic. Full and projected V1 restoration now
reuse the existing 16-byte pair trees ordered by `(row_id, event_time)`. Each
original row-validation loop latches a per-side duplicate fact through a bounded
same-ID range; there is no extra tree or full-state tail scan, and row order is
unchanged. The typed `CheckpointMismatch` is deferred until the original payload,
metric, limit, writer and historical checks finish, before optional dictionary
construction. V2/V3 history already rejects a second upsert for one row ID.

The authentic V1 control changes only a second live row ID, preserving both
different timestamps, payloads and literal full fees. Direct, managed, projected,
configured-refusal and healthy closed-admission routes all return the typed
error without a panic, preserve old state/writer/history/pending/canonical owner,
continue matching correctly and drain to pool zero. Required validation precedes
optional dictionary work; the invalid refusal-configured candidate records zero
dictionary refusals. Existing valid-candidate controls separately prove actual
grant-refused SQL acceptance.

The final ledger uses actual resident, writer and home/generation/attempt
reservation sizes. Projected writer controls plus fresh segment owners account
for 8,873 bytes independently, deduplicated by reservation pointer. Closed
admission retires exactly 16,384 generation bytes while 15,896 resident bytes
remain unchanged. Both observations satisfy exact pool accounting, and resident/
writer totals remain equal before and after rejection. The test-only writer
getter asserts no prepared owners and requires fresh direct segment owners; it
does not claim to handle arbitrary composed or restored segment owners.

## Documentation Consistency and Caveats

The final specification closes the initially missing healthy-closed-capability
boundary: only managed full-layout V1 optional derived scope/admission may
decline after `job.check_cancelled()` succeeds. It installs the validated new SQL
candidate and records `closed_restore_routes`, without claiming a memory refusal
or incrementing `dictionary_grant_refusals`. Actual job cancellation/deadline,
worker execution errors and other non-budget failures still propagate. The
existing nonempty home-close control checks candidate rows, literal fees and
carried history ownership.

The guide documents persistent cross-side identities, probe-only behaviour,
full fees, incomplete SQL coverage and side-local live row-ID uniqueness. The
changelog identifies the intentional full-V1 historical time/charge validation
fix and layout-1 duplicate-ID rejection. Wire layouts, checkpoint versions,
fingerprints and ASOF behaviour remain outside this change.

Explicit rebuild replaces a constant number of fields, but synchronous teardown
of old indices/directories/bindings can remain O(D). The analysis accurately
retains that cost; the whole publication tail is not constant-time. Existing
legacy Rowed backing and the inventoried legacy framing allocation retain their
prior compatibility funding boundary. No throughput, phase memory or latency
claim follows from these controls.

## Source Provenance and Verdict

The reviewer independently recomputed every one of the 58 final changed Rust
hashes in the feature and delivery worktrees and the sorted-path aggregate above;
all match the frozen manifest. Its path inventory exactly equals the feature's
changed Rust paths. The three new test-helper/control hashes equal the bytes
reviewed before the final run. The reviewed three documentation files are
byte-identical to `10a9c89f` in the final delivery. Delivery provenance is recorded in
`target/issue406-key-dictionary-delivery-20261015/final-source-merge-provenance.json`.

**Approve** for this final local source review at the exact feature/delivery
heads and seal above. B1 and prior pre-read findings are resolved; supplied
focused evidence is scoped to its recorded revisions. Required CI,
cross-platform verification, combined 90% coverage, release/new-native Python
proof, later phases and all eight issue
performance targets remain unproven. This report grants no push, merge or
whole-issue acceptance authority.
