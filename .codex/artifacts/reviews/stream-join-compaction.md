## Branch Review: Stream Join asynchronous checkpoint compaction

**Author:** Cheng Li | **Branch:** `feature/stream-join-compaction-main` → `main` | **Files:** 9

**Base:** `eccb26973811bc476f0944b977ddedf8564b0237`.
**Delivered commit:** `cd6ad73a5f8f56cfadd8d3a867791da21aaf69d7`.
**Delivered tree:** `4daf8fa050c5836a8dec01fd5bbd4a5f799273fc`.
Production source is identical to the independently reviewed source seal
`11fa57d6a6de05094bd7bb0c535c4b2e056eced8`; the follow-up changes only its
pre-install funding test and the single analysis artifact.
The independent review is local and precedes PR creation; there are no PR
check runs or prior GitHub reviews to consult. No remote action was performed.

### Summary

J1.6 moves Join layout-1 base rebuilding from data/progress handlers to bounded
native work owned by asynchronous checkpoint preparation. O(1) immutable
retained-vector snapshots, explicit ownership/drop boundaries, and opt-in
cleanup tracking preserve dirty changes, old captures, and funded native work
through cancellation and replacement. The public surface, V1 logical gauges
and limits, semantic capability, graph fingerprint, and wire framing remain
unchanged.

### Build and Test Results

- Rust: **Passed**, from the author's evidence: 57 Join unit tests,
  26 gather tests, and four `stream_join_properties` integration tests on the
  production seal. The delivered follow-up passed its one affected test,
  covering both unpolled and blocked-admission schedules, and lib/tests Clippy.
- Python: **Not touched**.
- Studio backend: **Not touched**.
- Studio frontend: **Not touched**.
- New failures: None in the author's final scoped checks.
- Regressions: None observed in those scoped checks.

All author Cargo commands used the repository root `target/` directory and
`CARGO_BUILD_JOBS=2`:

```bash
cargo test --locked -p calc-flow --lib operator::join::tests:: -- --test-threads=1
cargo test --locked -p calc-flow --lib gather_work:: -- --test-threads=1
cargo test --locked -p calc-flow --test stream_join_properties -- --test-threads=1
cargo test --locked -p calc-flow --lib \
  pre_install_submission_keeps_guard_until_owner_and_credit_drop -- --test-threads=1
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- schemas/project-v3.schema.json web-ui/openapi.json \
  web-ui/src/api/schema.d.ts crates/calc-flow/src/operator/asof tests/fixtures/v1
```

The final affected test returned exit 0 in author session `70625`; final
lib/tests Clippy returned exit 0 in session `31350`. The author also confirmed
format/whitespace/contracts checks and no remaining owned Cargo/test process.
Unchanged passing suites were not repeated for the evidence-only follow-up.

The reviewer read the changed source/tests/artifacts and actual saved RED
transcripts without building, running tests, or measuring performance. The
author's [analysis](../analysis/stream-join-compaction.md) records the observed
commands/results and distinguishes actual runtime RED from existing passing
coverage and structural refactoring. Full CI, the combined Rust 90% line
coverage gate, cross-platform checks, and performance evidence remain unrun;
the local results do not establish those gates.

### Blocking Issues

None. The pre-install funding proof identified at source-seal review is
resolved by `original_credit_paid`, which accounts for actual home/generation
fees plus the original 32 KiB at the gate and after real payload Drop. The
fixture also asserts an uninstalled attempt/observer; its focused verification
passed in both schedules.

### Style Issues

None found. New mutation is confined to the owned operator/attempt lifecycle;
caller Arrow buffers stay read-only, `Arc::make_mut` is absent, and workspace
unsafe/lint policy is unchanged. Tests remain beside the affected source in
focused modules with local fixtures.

### Test Coverage

The reviewed tests demonstrate off-thread encoding with the original retained
vector pointer, four-capture scheduling, handler exclusion, direct incremental
capture/continuation, five byte-identical frozen V1 captures, and preparation
under tight logical limits without an encoded-segment cap. Failure and dropped
futures preserve dirty changes and usable prior captures.

Ownership coverage includes reset/restore without stale installation,
cancellation/deadline errors, native-capacity waits before ticket delivery,
released input Arcs before actual record refund, full-pool retry, exact admitted
attempt control funding, escaped outputs, repeated managed drain, final pool
zero, and custom-Wake notification after actual refund. An expired Weak-home
test is accurately reported as a structural helper contract rather than a
production expired-home scenario. The ordered property suite covers schedules,
checkpoint cuts, counters, and reference multiplicities.

Source inspection confirms `ObservedSubmission` keeps work before credit and
retirement guard in one captured bundle. After funded installation the caller
receives its observer before the first fallible worker lookup/await; existing
ticket abandonment and SlotRelease retain actual attempt credit until native
cleanup. Escaped output owns value before credit before notification/retirement
release, and independent enabled notifications avoid lost wakeups.

The completed bookkeeping boundary is fixed O(1): only a Weak home, attempt
identity, and signal remain after refund. It retains no payload, workspace,
native worker, strong home, or retirement owner. Live work/output controls are
covered by actual attempt credit. Original WorkTicket, WorkOutput, Attempt,
submit_work, execution_fees, and ASOF call paths remain unchanged.

### Documentation Consistency

The single analysis artifact now matches the delivered opt-in submission,
cleanup, funding boundary, and actual TDD history. CHANGELOG records the actual
private behavior without claiming measured gains. Existing
`docs/runtime-envelope.md` already specifies managed async preparation and
worker compaction; no concrete normative mismatch or additional public API
documentation obligation was found. Generated contracts and historical fixtures
are unchanged.

### Verdict

**Approve** for the delivered commit/tree. No blocking source, style, test,
or documentation finding remains. Performance measurement and required CI
gates remain handoff work; this local specialist approval is not a green-CI
or merge-ready claim.
