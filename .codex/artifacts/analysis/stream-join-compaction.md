# Stream Join asynchronous compaction

## Scope and contracts

Issue #363 J1.6 implements FR10 and the shared safety constraints in the
[approved specification](../specs/stream-join-asof-acceleration.md). It starts
from merged main `eccb26973811bc476f0944b977ddedf8564b0237`. The older
`stream-join-safety` worktree is preserved unchanged. Only its Join tests and
implementation were migration inputs; its performance reports are not evidence
for this main-based change. J2 and ASOF algorithm changes are outside this stage.

Matching order, physical row IDs, FlatV1 logical charges and limits, semantic
capability/fingerprint, and layout-1 framing remain unchanged. Five frozen
captures are the exact byte oracle; the final compacted capture also checks
restore and continuation. Base versus carried-delta inventory can differ when
preparation is scheduled differently, so arbitrary historical snapshots are not
claimed byte-identical. Encoded segments receive no new size ceiling.

## Implementation and lifecycle

Four nonempty dirty captures schedule a base rebuild. Data and progress handlers
leave that work to `prepare_checkpoint_async`. An immutable
`Arc<Vec<StoredRow>>` snapshot shares both retained vectors in O(1). Sorting,
Arrow IPC encoding, and hashing run on the bounded native gather worker.
`Arc::get_mut` enforces exclusive mutation; there is no `Arc::make_mut` fallback
or retained-vector clone on Tokio. The input-release receiver remains on the
operator across dropped futures, reset, and restore. Mutable handlers wait for
old snapshot owners before touching retained vectors.

Snapshot release alone does not prove workspace refunds. The private opt-in
`AttemptCleanup` also waits until its attempt leaves Active/Parked/Dropping and
the actual credit-release marker completes. Ordinary `WorkTicket`,
`WorkOutput`, `Attempt`, `execution_fees`, and ASOF call paths keep their layouts
and accounting. Opt-in submission installs its observer into the caller
operator immediately after funded admission, before any fallible worker lookup
or pool-initialization await. The active ticket supplies abandonment RAII.

A single `ObservedSubmission` bundle orders pre-install/unpolled Drop as work
owners, reservation, stop metadata, then retirement guard. Work and credit are
borrowed during admission, so an inner await cannot independently capture their
ownership. After installation those resources belong to the existing attempt;
the caller wrapper alone holds its guard. Home and pool never hold a guard or
observer, avoiding ownership cycles. Guard registration fails closed before
submission; there is no successful output without a guard.

A successful `ObservedOutput` owns value followed by `TrackedCredit`, whose
fields drop actual reservation before release hook/guard. The only value
transfer is a synchronous `install` callback; there is no credit split or
unfunded value extraction API. Installation checks cancellation before updating
base/dirty inventory. It retains paid output until installation returns, then
refunds and clears the completed observer. It never awaits its own credit
release. Error, dropped-future, and abandonment paths retain the observer for a
later mutable handler. Failure leaves dirty operations and prior captures usable;
reset/restore cannot accept stale completion. Both input and credit waits retain
the existing cancellation/deadline failure kind.

The refund signal has independent Notify state, so an expired Weak home cannot
silently imply refund. Both home and refund notifications are enabled before
checking state/Acquire markers. Managed close uses the existing job retirement
guard and native attempt drain, including abandoned output and pre-ticket work.

## Funding boundary

Compaction reserves the unchanged logical input bytes plus two sort descriptors
per retained row. `BaseWork.control_bytes` additionally funds the private cleanup
wrappers, admission control fields, and Arc signal allocation/header/padding.
This is real attempt credit, verified by an exact live funding assertion while
native capacity is unavailable. Generic gather fees and ASOF budgets are
unchanged. Logical retained gauges exclude scratch/control and remain the V1
contract; this is not a whole-process RSS ceiling.

While work is Active/Parked/Dropping or an escaped successful output holds
credit, its new control allocations remain funded. After actual refund the
operator may retain only fixed O(1) completed Signal/Weak bookkeeping until its
next handler or Drop. It contains no payload, workspace, or strong worker/home
owner, and does not require a separate reservation category. The parent accepted
this boundary explicitly. Worker-inclusive RSS is still required in measurement.

## Actual TDD record

All main-based native checks use the root target directory and two build jobs.
Tests were installed before the first production migration. The first Join run
had **44 passing and eight expected failing tests** (151-second build, 0.13-second
runtime). Failures established handler-driven compaction, missing asynchronous
base rebuild, tight logical-limit preparation, absent credit preflight, and
owned-worker lifecycle behavior. The frozen five V1 bytes already passed on the
untouched production base. The migrated implementation then passed 52/52.

Additional actual cycles:

- Reset/restore retirement: two failures before retaining the receiver across
  replacement; reset exposed a new row before old input retirement, while
  restore mutation returned Ready rather than Pending.
- Deadline waiting: the old input wait returned timeout Elapsed instead of the
  existing Cancelled failure. Adding deadline-aware waiting passed the case.
  The resulting Join stage passed 55/55.
- Snapshot-before-credit: a native gate released both snapshot Arcs but retained
  the original attempt. With the real pool filled to 1 GiB, retry returned
  Ready(DataFusion) when requesting 560 bytes. The cleanup observer makes retry
  Pending until actual record credit refunds. Actual transcript:
  `target/issue363-compaction-red/credit-retirement.txt`.
- Weak-home-before-refund: a controlled empty Weak and unset marker made the
  observer incorrectly return Ready(Ok). This is a structural completion test,
  not evidence that a production job expired while still holding a paid output.
  Independent refund notification fixes the helper contract.
  Actual transcript: `target/issue363-compaction-red/weak-home-refund.txt`.
- Pre-ticket tracking: real native-capacity pressure left the paid preparation
  without an observer in the operator. The first RED asserted required tracking;
  the later stronger GREEN also gates a closing thread between owner release and
  credit refund, verifies a live Weak payload and exact admitted control fee,
  fills the real pool, and requires retry Pending. Actual transcript:
  `target/issue363-compaction-red/pre-ticket-tracking.txt`.

The transcripts above preserve actual tool output excerpts, rather than claim to
be original redirected process logs. Test-only gate compilation fixes are not
behavioral RED evidence. The new private observer API initially failed compilation
because it was absent; this is recorded separately from the runtime failures.

Pre-install/unpolled custom-owner Drop already passed both schedules before the
explicit bundle refactor. This is additional safety coverage and a behavior-
preserving structural refactor, not an invented independent RED. Escape-output
coverage checks real credit beyond home/generation charges, a live payload,
managed drain Pending after its first future is dropped, error/success installation,
accurate remaining funding, and final pool zero. Final review strengthened the
pre-install fixture to assert an uninstalled second attempt/observer and actual
funding of home + generation + the original 32 KiB. Its gated owner destructor
explicitly drops the real payload before recording that the original credit
still exceeds those same control charges. Temporary generation references keep
that comparison stable during concurrent native retirement. Both unpolled and
blocked-admission schedules passed the single focused check; this is evidence
strengthening without a production change or invented RED. A custom Wake observer also
verifies that actual reservation refund wakes an expired-home waiter.

## PR371 complexity refactor

Codacy identified `BaseWork::run` at complexity 13 and `prepare_compaction`
at 15 against a limit of eight. The follow-up extracts the test gate, one-side
encoding, workspace reservation, owned snapshot construction, admission error
projection, and synchronous installation into focused helpers. It preserves
the original cancellation checks, left-before-right encoding, funded admission
and observer publication, owner/credit retirement, and install/dirty-clear order.
No data fields, control fee formulas, generic gather code, or wire format change.
This is a behavior-preserving refactor; no independent RED is invented.

Local Lizard reports `run` at six, `prepare_compaction` at four, and the largest
helper (`rebuild_compaction_base`) at eight. Every function in the file meets
the same limit. After the parent's debug-cache grant, the following affected
checks passed with the same root target and two build jobs:

```bash
cargo test --locked -p calc-flow --lib \
  operator::join::tests::checkpoint_compaction_tests:: -- --test-threads=1
# 14 passed; 2m43s build, 0.10s runtime.
cargo test --locked -p calc-flow --lib \
  frozen_v1_checkpoint_fixture_preserves_wire_bytes_and_continuation \
  -- --test-threads=1
# 1 passed, covering five byte identities and final restore/continuation.
cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
# Passed, 1m12s.
lizard -l rust -C 8 -w crates/calc-flow/src/operator/join/checkpoint_compaction.rs
# Exit 0, no function above eight.
```

Format, whitespace, and unchanged-contract checks also passed. The extracted
workspace helper retains the mutable receiver required for the existing lazy
runtime initialization; its initial compile-signature correction is not RED
evidence. Earlier unchanged Join/gather/property suites are not repeated for
this refactor. Remote Codacy and required CI results remain the parent's handoff.

## Local verification and measurement handoff

Observed local checks:

```bash
CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  operator::join::tests:: -- --test-threads=1
# 57 passed, including five frozen byte identities and exact live control funding.

CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow --lib \
  gather_work:: -- --test-threads=1
# 26 passed, including the final whole-bundle capture and custom Wake proof.

CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo test --locked -p calc-flow \
  --test stream_join_properties -- --test-threads=1
# 4 passed: exact schedule order/cuts/counters and reference multiplicities.

CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target \
  CARGO_BUILD_JOBS=2 cargo clippy --locked -p calc-flow --lib --tests -- -D warnings
cargo fmt --all --check
git diff --check
git diff --exit-code -- schemas/project-v3.schema.json web-ui/openapi.json \
  web-ui/src/api/schema.d.ts crates/calc-flow/src/operator/asof tests/fixtures/v1
```

Clippy passed after import placement and equivalent test boolean-conversion
cleanup; no behavior or assertion was weakened. Format, whitespace, generated
contracts, ASOF source, and historical fixture drift checks passed. Original
normal gather types and fees changed only by adding a private module/re-export;
ASOF-specific integration tests were not rerun locally. Full CI, Rust 90% line
coverage, Python/Studio checks, and performance measurement remain unrun here.

After source seal and independent review, the parent owns public-API paired
handler/preparation latency, worker-inclusive RSS, restoration oracle, and PR/CI
handoff. Handler latency and total checkpoint-preparation latency must be
reported separately; scheduling the expensive work is not eliminating it. No
performance gain, remote CI result, or merge-ready claim is made at this seal.
