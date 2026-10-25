# DAL-441 / T4: ordered LAST_VALUE recovery acceptance

Baseline: `37d8371e24c76f03ab9409eae21a78673d58371a`, merged T3 / PR #498.
The dedicated DAL-441 checkout starts at the fetched main containing that
commit. This stage adds acceptance coverage; no production defect was found.
The only changes outside test modules propagate a test-only IPC hook's error
inside existing `cfg(test)` blocks. Public APIs, project-v3, manifest-v3,
layout-3/4 encoders, dependencies and runtime behavior are unchanged.

## Coverage mapping

`operator/sql/ordered_last_tests/recovery.rs` exercises 34 successive input
prefixes, each compared with an uninterrupted operator and independent DF55
batch SQL. It restores detached segment bytes after every capture, alternating
sync/async capture. The first 32 sparse captures append delta frames; capture
33 rewrites the base and capture 34 appends to the new generation. Assertions
cover generation numbers, base allocation reuse within each live generation,
old/new/tied/null quotes, new keys, first-seen output slots, null-only IGNORE
NULLS groups, metadata, schema and recapture. A separate full/delta continuation
test covers all four Timestamp units with absent and explicit offset timezones,
including negative timestamps and first-winner ties.

`ordered_last_tests/corruption.rs` rebuilds control digests and, for malformed
Arrow state, the corresponding segment digests and census. It verifies rejection
of unknown layout/accounting/native/log versions and policies; incompatible
query, alias, physical/state/output schemas, retained columns, ordering column
type/index/direction/null placement, null treatment, tie/group-slot/engine model;
missing control fields/segments, unlisted segments, wrong digests, generation
and ledger inconsistencies; invalid winner/flag states, Timestamp unit/timezone
changes, and duplicate keys in both base and delta. Duplicate-key cases reach
the duplicate-key error rather than an earlier census refusal. Every failure
preserves the installed native owner and byte-identical checkpoint metadata
and digests, and refunds the actual engine pool. Unknown outer layout wins over
an invalid control payload before decoding begins.

`ordered_last_tests/faults.rs` covers these distinct boundaries:

- Dropping a pending emit future after a candidate has been built releases its
  new key, winner and metadata without committing the input ledger. A successful
  retry accepts that input once.
- Sync and async native IPC encoding failures preserve the prior recoverable
  capture and dirty delta. Retrying captures/restores the pending winner. The
  hook is scoped to a unique test node and removed by its registration guard.
- A failure after restore preparation and cancellation at every actual restore
  check boundary leave the old snapshot installed. Candidate leases expire and
  pool fees return exactly, including after the final validation boundary.
- Timestamp-bearing state and null-only groups are charged under the current
  group/byte budgets on restore. Reduced limits refuse atomically. A nonempty
  layout-4 ordered checkpoint conflicts with an explicit compact budget while
  preserving the receiver's live compact state.

`runner/tests/sql_recovery_tests/ordered_last.rs` runs real managed jobs with
the existing transactional source/sink, state backend and fault seams. A
StateStage I/O failure sends no operator ACK or sink pre-commit. A ManifestWrite
I/O failure follows the expected operator ACK and sink pre-commit but commits
no new sink epoch and publishes no new manifest. Both preserve the original
manifest bytes and restore its winner. A native capture-prepare failure installs
no candidate, sends no ACK and performs no sink pre-commit. Tests await job
termination and runner shutdown and assert empty registries and connector
cleanup; there is no sleep/retry or new product fault-injection API.

The accepted T2/T3 artifacts remain evidence for unchanged admission/DF55
semantics, group/state accounting, cumulative input quotas, bindings and
retained compatibility. Existing emit-error/cancellation and layout-4 ordered
continuation tests also run with the new tests. The existing SUM/COUNT layout-3
roundtrip is a control for the shared capture hook.

## Validation

Run against the Rust content of this stage, reusing the original checkout's
Cargo target tree with `CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0`.
Only compiled dependencies/artifacts are reused; sources come from this clean
DAL-441 checkout, not the old WIP.

- `cargo test --locked -p calc-flow --lib -- ordered_last_tests compact::direct_async_tests sql_recovery_tests::ordered_last --nocapture`:
  26 passed, including 13 new acceptance tests; 0.85 seconds execution,
  44.49 seconds compilation for the final iteration.
- `cargo test --locked -p calc-flow --lib test_compact_storage_two_roundtrips_continue_and_recapture_every_prefix -- --nocapture`:
  the SUM/COUNT layout-3 control passed, 0.17 seconds execution.
- `cargo fmt --all --check`, changed-module Lizard complexity threshold 8,
  `git diff --check`, and unchanged Cargo.lock/project-schema checks passed.
- `cargo clippy --locked -p calc-flow --lib --tests -- -D warnings` passed.
  Four lint-only fixes add statement semicolons, use a checked conversion of
  the known-positive prefix counter and remove a Copy type's clone. The test
  evidence above is reused after these behavior-preserving changes.

The first run exposed two fixture errors: the encoder hook shared a node name
with other concurrent tests, and a cold restore was incorrectly expected to
reuse the source snapshot's allocation. The final tests use an isolated hook
name and compare base ownership within each live generation. No product
assertion or validation was weakened to address these fixture errors.

Full regression, cross-platform checks and the existing combined Rust 90%
coverage gate remain with CI. This stage does not establish those CI results
or merge readiness; T5/T6 remain outside its scope.

## Preserved work

The original DAL-431 checkout on `agent/cf-implementer/5ceb4eb9b8b4` still has
its 21 tracked changes and four untracked files. It was not edited or cleaned.
Stash `32edcee9716fcd7a4d0cb595d69b86967fdb7c37` and the existing patch/archive
backups remain the previous stages' recovery copies. Its six historical
diagnostics are excluded from this stage's source and validation evidence.
