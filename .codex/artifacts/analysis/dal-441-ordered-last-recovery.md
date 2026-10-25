# DAL-441 / T4: ordered LAST_VALUE recovery acceptance

Baseline: `37d8371e24c76f03ab9409eae21a78673d58371a`, merged T3 / PR #498.
The dedicated DAL-441 checkout starts at the fetched main containing that
commit. This stage adds acceptance coverage and rejects impossible unset
RESPECT NULLS winners during recovery. The initial delivery was
`41f23060c87559f7451b3975b76050ea19e89b60`; the focused review found the
missing null-state boundary described below. Public APIs, project-v3,
manifest-v3, layout-3/4 encoders, identities and dependencies are unchanged.

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

## Focused review correction

The original corruption fixture used IGNORE NULLS and missed the impossible
unset state for default/RESPECT NULLS when the value and entire ordering tuple
are null. The reviewer supplied a production-library probe showing that a
checksum-valid checkpoint with only `is_set` changed from true to false restored
successfully, then replaced the null winner with an earlier quote. Independent
DF55 and uninterrupted execution both retained null.

`ordered_last.rs::result` now rejects unset candidates unless IGNORE NULLS is
active and every candidate scalar is null. Set null winners remain valid under
default/RESPECT NULLS; legitimate unset IGNORE NULLS groups are unchanged.
`corruption/respect_nulls.rs` runs default and explicit RESPECT NULLS with full
and delta checkpoints, changes only the target group's flag, preserves schema
and census, updates digests and removes segment owners. Before the fix both
new tests failed because restore returned Ok. After the fix rejection reaches
the candidate-state consistency error, preserves the native owner, winner/key
census, uncaptured dirty state, quota ledger, metadata and old capture, and
refunds both actual engine and compact-state fees exactly. Legal restore and
continuation with an earlier quote match uninterrupted execution and DF55,
including the null winner; the result can be recaptured/restored again.

## Windows ASOF failure diagnosis and scheduling

The Windows Rust job
[121238214507](https://github.com/wegamekinglc/calc-flow/actions/runs/40216341910/job/121238214507)
reported 2400 passed, three ASOF barrier-ACK timeouts and six ignored cases.
It uses all features, dev debug=0 and parallel libtest execution. All three
failures are at the same five-second wait in `TaskFixture::fence`. That wait
includes native capture admission and durable segment staging, validation and
publication before ACK. The ASOF/gather/state production code and fixture are
byte-identical between main and the initial T4 head; SQL IPC probes are scoped
to unique SQL node names and are not called by ASOF.

On Linux, the unmodified all-features fixture passed 10/10 with four test
threads. A bounded five-run comparison mixed that module, T4 restore/encoder
faults, managed captures and shared async IPC tests on two CPU cores with four
test threads: 22/22 each time (110 total), 0.86–0.95 seconds per run. Thus no
hook leak or native lifecycle defect was reproduced; this does not establish a
native Windows result or prove that all possible resource races are absent.

The existing Windows job already isolates two other cases using the same
durable ASOF fixture to avoid checkpoint deadline contention. The three new
failures now execute in that same exact-name serial stage. Static command
mapping proves all seven cases excluded from the parallel phase execute in
the subsequent required serial phase. No test is removed or ignored overall;
test bodies, five/ten-second waits, assertions and required gates are unchanged.
This addresses scheduling contention as the working diagnosis, not a claimed
production lifecycle fix. The actual Windows confirmation remains with CI.

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

After the review correction, with the same target/debug settings and all
features:

- `cargo test --locked -p calc-flow --lib --all-features -- corruption::respect_nulls ordered_last_tests::corruption test_ordered_last_layout_five_roundtrip_preserves_null_groups_and_ties test_ordered_last_full_delta_preserve_all_timestamp_units_and_timezones test_ordered_last_delta_rewrite_and_every_cold_prefix_match_uninterrupted_and_df55 compact_budget_restores_full_and_delta compact_budget_group_limit test_ordered_last_rejects_inconsistent_winner_state --nocapture`:
  11/11 passed, 0.65 seconds. Covers the two new regressions, actual restore
  rejection/refunds, legitimate IGNORE NULLS, full/delta rewrite and budgets.
- `cargo test --locked -p calc-flow --lib --all-features runtime::streaming::operator_task::tests::asof_tests:: -- --nocapture --test-threads=4`:
  baseline 10/10 passed, 0.05 seconds. The bounded mixed-module comparison
  described above passed 110/110 without source changes between runs.
- Exact serial execution of the three failing ASOF names with
  `cargo test --locked -p calc-flow --lib --all-features -- <three exact names> --exact --test-threads=1 --nocapture`:
  3/3 passed, 0.05 seconds. The exact names are preserved in the workflow.
- Windows workflow YAML parses; its seven-case parallel/serial command sets
  match exactly. No overall test exclusion or deadline change was introduced.
- `cargo test --locked -p calc-flow --lib --all-features test_ordered_last_every_prefix_matches_datafusion_across_batch_splits -- --nocapture`:
  the existing DF55 oracle passed, including default/RESPECT/IGNORE NULLS,
  ASC/DESC, composite order and batch splits; 0.50 seconds.
- `cargo clippy --locked -p calc-flow --lib --tests --all-features -- -D warnings`,
  formatting, changed Rust complexity threshold 8 and diff/schema/lock checks
  passed. The ASOF fixture and production lifecycle code were not modified.

The first run exposed two fixture errors: the encoder hook shared a node name
with other concurrent tests, and a cold restore was incorrectly expected to
reuse the source snapshot's allocation. The final tests use an isolated hook
name and compare base ownership within each live generation. No product
assertion or validation was weakened to address these fixture errors.

Full regression, cross-platform checks and the existing combined Rust 90%
coverage gate remain with CI. This stage does not establish those CI results
or merge readiness; T5/T6 remain outside its scope.

## Workflow expectation synchronization

The follow-up Linux wheel job failed before building the wheel because
`scripts/test_run_rust_tests.py::WINDOWS_CHECKPOINT_REGRESSIONS` still listed
four cases after the workflow moved three more ASOF cases into the serial
stage. The exact failing unittest was reproduced locally before editing. The
independent literal tuple now lists all seven names in workflow order; both
exact list/argument comparisons remain unchanged, including
`--exact --test-threads=1`. No expected data is generated from the YAML and no
subset assertion or exemption replaces the comparisons.

`python -m unittest scripts.test_run_rust_tests.WindowsCheckpointSoakIsolationTests scripts.test_run_rust_tests.RustTestHarnessTests.test_lib_skip_appends_every_filter_to_the_calc_flow_leg_only scripts.test_run_rust_tests.RustTestHarnessTests.test_lib_skip_is_ignored_by_compile_only_runs -v`
passed 7/7 in 0.329 seconds. This expectation-only synchronization does not
change the workflow, Rust behavior, deadlines or gates; the preceding Rust
evidence remains applicable. `uv tool run --from ruff==0.16.0 ruff check scripts/test_run_rust_tests.py`
and `uv tool run --from ruff==0.16.0 ruff format --check scripts/test_run_rust_tests.py`
passed. Native Windows confirmation and the full CI result remain pending.

## Preserved work

The original DAL-431 checkout on `agent/cf-implementer/5ceb4eb9b8b4` still has
its 21 tracked changes and four untracked files. It was not edited or cleaned.
Stash `32edcee9716fcd7a4d0cb595d69b86967fdb7c37` and the existing patch/archive
backups remain the previous stages' recovery copies. Its six historical
diagnostics are excluded from this stage's source and validation evidence.
