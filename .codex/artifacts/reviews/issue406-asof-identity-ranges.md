# Issue 406: identity workspace ranges review

**Branch:** `feature/issue406-asof-identity-ranges`
**Parent:** `88922e13932f1426d5e8b57260b75d8e119eae77`
**Ordinary parent product:** `05c13fd983dd1ead625eef2278eb7127dac0c61c`
**Reviewed implementation:** `8d3f6933c53f0dab1a562c79ad355cd9e08c665f`,
limited to `workspace.rs` and `workspace/tests/identity_ranges.rs`
**Evidence addendum:** analysis working diff over that source product
**Review phase:** final implementation, focused verification, sealed diagnosis
and completed paired evidence; phase-wide acceptance remains incomplete

The reviewer owns only this report in the implementation owner's isolated
worktree. No product files were edited and no build, test, profiling, CI query
or remote mutation was performed.

## Summary

The optimization resolves Utf8/LargeUtf8 offsets once per accepted
range, then computes the same historical per-row identity workspace charge.
It reduces repeated Arrow type dispatch and downcasts; it still visits value
lengths to preserve independent row alignment. The specification and diagnostic
scope match the implementation, with no outstanding source review blocker.
The two paired OFF cases retain the maintained `no-confirmed-regression`
classification. Favorable point estimates do not certify an `improved` verdict
or complete issue-wide performance acceptance.

## Build and test results

- RED: `identity-red.log` records the existing range-parity regression failing
  with **4,097 scalar length dispatches for 4,096 rows and two columns**.
- GREEN: `workspace-green.log` records **ten workspace tests passed**, zero
  failed, in 0.04 seconds, including the complexity bound and all new range
  parity controls. Core registration remains 2,033.
- Admission: `admission-tests.log` records **59 passed and four failed**, in
  13.29 seconds. All four errors are `ASOF gather: bounded process
  infrastructure exhausted`. The unchanged process service has a shared
  32 MiB infrastructure cap in `gather_work/process.rs`; this increment changes
  no worker count, service reservation or process lifecycle. The exact four
  tests pass in separate serial test processes in 0.03/0.04/0.01/0.01 seconds
  (`admission-isolated.log`). The unchanged 59 were not repeated. This is not
  a claim that the initial combined admission run passed.
- Scoped all-feature core/library/test Clippy passes in **2m 49s**
  (`clippy.log`). Owner formatting, schema-drift and whitespace checks pass.
- Lizard 1.24.1 recognizes all **twelve changed/new functions**; observed
  maximum CCN is **five**, below the enforced limit of eight. The tool's initial
  nonexistent-venv launch failure remains in `complexity.log`; the corrected
  PYTHONPATH invocation passes in `complexity-fixed.log`. No project dependency
  was added, and unchanged functions are not covered by this certification.
- Python package behavior is unchanged and was not rerun. Required CI and
  workspace coverage were not queried or certified. No unresolved source or
  isolated focused-test failure remains.

The owner's recorded commands run from the implementation worktree root:

```bash
export CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-join-hotpaths-build
export CARGO_BUILD_JOBS=2
cargo test -p calc-flow --lib operator::asof::workspace::tests::admission_ranges_preserve_exact_row_workspace_and_skip_late_rows -- --exact
cargo test -p calc-flow --lib operator::asof::workspace::tests -- --nocapture
cargo test -p calc-flow --lib operator::asof::admission -- --nocapture
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

Each failed admission test runs in its own subsequent process:

```bash
RUST_TEST_THREADS=1 cargo test -p calc-flow --lib operator::asof::admission::identity_tests::columnar::restored_and_disordered_live_maximum_guards_overlap_and_duplicates -- --exact --nocapture
RUST_TEST_THREADS=1 cargo test -p calc-flow --lib operator::asof::admission::identity_tests::columnar::utf8_empty_non_ascii_and_record_cuts_match_legacy -- --exact --nocapture
RUST_TEST_THREADS=1 cargo test -p calc-flow --lib operator::asof::admission::identity_tests::cpu::abandoned_columnar_preparation_retains_owner_until_native_drain -- --exact --nocapture
RUST_TEST_THREADS=1 cargo test -p calc-flow --lib operator::asof::admission::identity_tests::cpu::ordered_scalar_left_admission_avoids_cpu_worker -- --exact --nocapture
```

The corrected static complexity command is:

```bash
PYTHONPATH=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-routing-parallel-profile-20261013/complexity-tool \
python /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-asof-admission-profile-20261014/check_complexity.py
```

## Blocking issues

None in the reviewed source, focused verification or diagnostic report. The
initial admission failures and tool-launch error remain visible with their
diagnoses and passing bounded follow-ups. The declared two-case timing attempt
is complete; complete phase-wide allocation/RSS acceptance is unproved.

## Style issues

None identified. The private pure range transform changes
no public API, dependencies, checkpoint format or caller-owned Arrow buffers.
The helper test module follows the Rust mirrored layout and reuses an existing
registered regression without adding a lint exception.

## Correctness, budgets and ownership

- `identity_column_range_workspace` dispatches/downcasts each string column
  once and borrows its existing offsets. `identity_offset_range_workspace`
  visits adjacent offsets only within the accepted range, without allocating
  or inspecting/mutating the stored values.
- Each row still computes `aligned(column.fixed + adjacent_offset_length)`
  before 33/32 blocked-string scaling and framing. `string_identity_workspace`
  preserves the original formula; lengths are not aggregated before alignment.
- Offset differences preserve nonzero slice origins and physical spans under
  null entries. The implementation neither skips nulls nor replaces their
  stored lengths with zero.
- `ColumnWorkspace::bytes_with_length` extracts the existing checked addition
  without changing its error reason or message. Negative lengths retain the
  existing Format error; subsequent framing and sum operations retain checked
  arithmetic and failing node names.
- Retained sequence copies are still charged after the per-column range total.
  Fixed-width calculation is unchanged, and `identity_range_workspace` returns
  zero for empty ranges before accessing a first value. Typed downcasts rely
  on the same private validated-type invariant as the previous scalar helper.
- Accepted late-row ranges, source selection, memory reservation boundaries,
  worker ownership, cancellation points and durable state/checkpoint contracts
  are unchanged. The range helper introduces no heap state or caller mutation.

## Test coverage

The unchanged scalar `identity_row_workspace` is the independent historical
charge oracle. The new helper compares every selected subrange for Utf8 and
LargeUtf8, including full arrays, slices with nonzero origins, empty slices,
null bitmaps with nonempty underlying spans, variable lengths around alignment
and block boundaries and long strings. It covers both string key and retained
string sequence charges alongside a fixed-width retained sequence.

The synthetic overflow case checks `AsofCounterOverflow`, node `range-node`
and the exact column-workspace error text. The existing range test retains
late-row selection and every short prefix/suffix cut. Admission evidence also
covers non-ASCII keys, record cuts, composite identities, state/workspace
thresholds, caller immutability, refunds, worker abandonment and cold recovery.
The four initially affected infrastructure consumers pass individually without
weakening their lifetime or recovery assertions.

## Sealed diagnostic evidence

Artifacts are under `target/issue406-asof-admission-profile-20261014/`.
The reviewer verified plan SHA-256
`4b89acf1b4d7d5062880896e842dd743c570eb0381df53ece3485afbfc618766`,
all **1,654 frozen input hashes** and **23 result seals**, and all six raw
million-row output oracles, completed lifecycles, cleanups and true OFF
zero-file checks. Product identity is exact `05c13fd9`, native SHA-256
`cbc1a85acd9d3b1d6baf0ec2e5fe0c888a5ec53353ee2176d1ae56281a6c600b`.
All observations use the planned 32-CPU affinity. Parent `88922e13` and
diagnostic product `05c13fd9` have no source difference under `crates`, `python`,
`Cargo.toml` or `Cargo.lock`.

The single case uses 1M rows per input and 64k batches, one warmup and five
fixed windows. The windows match raw processing-clock stamps and contain
549/574/553/536/561 samples, totaling **2,773**, with **2,522 native-frame
samples**. Identity range estimation has 6.64% inclusive samples, 2.70% self;
scalar variable-length dispatch has 3.14% self. These shares overlap and
establish no timing gain. WSL2 host isolation is unverified.

Inclusive acquisition and sealing take **10.381440611 / 180 seconds**. All
owned groups settle with no forced kills or surviving groups. The preserved
launcher failure occurs at the toolkit import before plan freeze/acquisition;
the corrected PYTHONPATH invocation is the sole acquisition. No observation
was retried, removed or substituted.

## Completed paired evidence

The sole fixed attempt under `target/issue406-identity-ranges-paired-20261014/`
compares exact ordinary parent `05c13fd9`, native SHA-256
`cbc1a85acd9d3b1d6baf0ec2e5fe0c888a5ec53353ee2176d1ae56281a6c600b`,
with candidate `8d3f6933`, native SHA-256
`3664ebab40c619a0c4e7b8a9cfe93a684ab2da24e5c3ca52f8cd88c831041d08`.
The frozen plan SHA-256 is
`cdee70dddc4b70fe92d456886a18e6742945a0641d70bf97c9a0e365b66e7859`.
The candidate ordinary build takes **380.894939 seconds**, separate from
acquisition; parent binaries are reused. This comparator isolates the increment
and is not a new merged-main comparison.

The reviewer independently verified all **3,118 frozen input hashes** and
**198 result seals**, and validated both ordinary release audits. Their complete
inventories of **294 compiled artifacts**, normalized feature/profile settings,
environment, interpreter, toolchain and lock identity agree. The acquisition
wrapper differs from the previously reviewed wrapper only in scope and corrected
report labels; it retains the same collection and classification behavior.

Each case completes two rounds of ten adjacent AB/BA pairs and one warmup per
arm per round, totaling **80 measured observations and eight warmups**. All
88 raw results pass recorded million-row oracles, completed lifecycles, owned
cleanup and true OFF zero-file checks. Raw/normalized digests, processing-clock
and CPU bounds, product revisions, affinities and all **eight worker native
identities** agree. Actual observation and pair ordering match the fixed plan.

Recomputed maintained statistics and CPU medians exactly match the stored
objects and independent audits:

- OFF/32: parent p50 **159.888209 ms**, candidate **148.191731 ms**, point
  change **-7.3154%**. Round paired medians are -5.760743% / -7.830057%,
  with intervals [-8.798251%, -4.210052%] / [-10.781846%, -5.764932%].
  Verdict: **no-confirmed-regression**.
- OFF/1: parent p50 **130.997559 ms**, candidate **120.456423 ms**, point
  change **-8.0468%**. Round medians are -7.653816% / -5.916994%, with
  intervals [-9.991739%, -3.347668%] / [-8.486696%, -4.021210%].
  Verdict: **no-confirmed-regression**.

Each ten-pair order-statistic interval has **97.8515625% coverage**. Neither
case meets the maintained requirement that both upper bounds be below -5%
for `improved`. These results prove neither equivalence nor complete absence
of regression. The sealed generated report correctly labels its comparator
`Exact parent p50` and its interval `97.85%`; originals remain unchanged.

Processing CPU medians are 187.59475 → 175.1312 ms at 32 CPUs and
130.7536 → 120.22965 ms at one CPU. Busiest observed thread medians are
59.18415 → 56.7925 ms and 119.0725 → 109.23445 ms respectively. CPU hooks
retain the exact maintained processing stamps. Counter-read uncertainty and
retired-thread visibility prevent exact thread accounting; whole-job CPU is
a separate scope. These diagnostics do not replace paired wall statistics.

The reviewer recomputed all **32 resource snapshots**: maximum observed
aggregate RSS is **839,221,248 bytes**, including controller and workers;
maximum observed controller RSS is 90,902,528 bytes, and largest observed
single-worker RSS is **377,720,832 bytes**. Every snapshot records no competing
controlled work. Sampled acquisition maxima are not exact per-arm or per-phase
peaks and are not allocation counts. Full-phase allocation/peak-RSS acceptance
and WSL2 host isolation remain unverified.

Inclusive acquisition and sealing take **40.763058131 / 600 seconds**. All
owned groups settle with zero forced kills, survivors, pending creations or
cleanup errors. There are no retries, filtering, count increases or added
cases. Candidate p50 values still miss the OFF one/32-CPU targets of
100 / 30 ms. The other seven target rows, ON qualification and earlier
stopped/inconclusive evidence retain their existing status.

## Documentation consistency and limits

The specification preserves all eight issue targets and the earlier stopped
ON and inconclusive results. Any later admission comparison must predeclare
exact ordinary releases, fixed proportional cases/counts and an inclusive
600-second budget, with builds separate. It also requires allocation and RSS
evidence; this profile provides neither that complete evidence nor a paired
performance verdict.

The predeclared `prebuild-selection.json` fixes two directly affected OFF cases
at 32 and one CPU, each 1M rows/input and 64k messages. Both execute this
changed admission estimator. Two rounds of ten adjacent AB/BA pairs, one
warmup per arm per round and the inclusive 600-second budget are recorded
before acquisition; candidate builds are estimated separately at 6–14 minutes.
Preflight/artifact audit, fixtures, startup, oracles, CPU reads and cleanup count
toward acquisition. The exact ordinary parent product is `05c13fd9`, rather
than current merged main. The completed attempt above retains that fixed scope.

Existing candidate admission allocation controls retain their actual counts,
including 95 caller allocations and 26 chunk-worker allocations for 64k rows.
They compare the candidate's fast/legacy paths, not this increment against its
parent, and establish no complete phase-wide allocation or peak-RSS verdict.
The analysis makes that distinction explicitly. These unresolved issue-wide
metrics must remain visible in any later performance handoff.

The reviewed analysis addendum correctly reports exact identities, retained
classifications, interval coverage, sample/order counts, resource scopes and
unmet goals. It does not promote favorable point estimates to speedup claims
or sampled RSS to phase-wide acceptance.

PR #421's independent report correction merged as `1606e516`; its original
during-compaction acceptance metric remains unmeasured. That merge does not
complete the issue-wide acceptance objective. Required CI and coverage remain
separate from this bounded review.

## Verdict

**Approve** source product `8d3f6933`, its specification and reviewed
verification/evidence addendum. No correctness, budget, style, focused-test or
reporting blocker remains. Both paired cases are `no-confirmed-regression`;
phase-wide allocation/RSS, ON qualification and the complete issue objective
remain incomplete. Required CI and coverage are uncertified. This is not a
merge-readiness or complete performance-acceptance claim.
