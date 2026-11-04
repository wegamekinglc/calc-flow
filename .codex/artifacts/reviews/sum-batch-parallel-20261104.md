# Branch Review: Archive the Rejected Batch-Parallel SUM Experiment

**Branch:** `feature/sum-batch-parallel-ci` → `main`

**Delivery base:** `d5e05c31110ebb1e6bfb8fa3d63359f50010f16e`

**Experiment base:** `c9afa829ff025280ab346667582c62dc26a34785`

**Measured prototype:** public
[fe0e86ad220bd88ae3aeb1ae6c384e3a72e0a792](https://github.com/wegamekinglc/calc-flow/commit/fe0e86ad220bd88ae3aeb1ae6c384e3a72e0a792),
tree `e0834050279eba44c48205c84aaa0eb1a600a396`.

**Archive runtime verification checkpoint:** `36cad599f16a4073762411e1bd8c7a0492ee3e4e`, tree
`bc7209cbee98a34925dac3ff7b55fcc85aa8f341`
(local verification identity, before adding this final review).

**Security followup checkpoint:** local
`fd4168a143dc963cc8f834bcef0f3ca25998b672`, tree
`1a8d9bbc96a354bfb8b9989c2f62e615f92f5711`, based on published archive head
`4f3c9b7206def7967ac10d12818b159bc5ff1b30`.

**Upstream artifacts:** [Specification](../specs/sum-batch-parallel-20261104.md)
and [preimplementation critique](../critiques/sum-batch-parallel-20261104.md).
The critique's conditional TDD clearance is `f1d9ce64`. Historical source
review bound to local `2dafaa83`; tool checkpoints are `bc178d1d` and
`f9c1eb50`. Public fe0 contains those reviewed code/tool bytes.

**Files changed against base:** 22. The archive is now
[PR #537](https://github.com/wegamekinglc/calc-flow/pull/537). Its published
4f3c9b72 head has one observed Codacy hash-rule blocker; the narrow local
correction below requires final-head remote verification.
Only the parent agent performs remote publication, CI completion, and merge.

## Summary

**Reject the runtime prototype. Approve the archive-only source/evidence
scope, subject to required remote checks.** The fixed five-case experiment
confirmed dense-stream regressions with four and eight CPUs despite genuine
SQL worker execution. Its performance acceptance is false.

The delivery preserves the current merged target's runtime and API. After
removing the prototype, it merges upstream d5e05c31 without dropping its
changes. It retains the measurement tools, their CI tests, the rejected public
prototype in history, and all sealed negative evidence. It introduces no
parallel compact SUM behavior, public knob, schema, or checkpoint change.

## Build and Test Results

- **Final runtime/API: unchanged against delivery target d5e05c31.** Root's
  `git diff --exit-code origin/main -- crates Cargo.toml Cargo.lock schemas/project-v3.schema.json python/calc_flow python/tests`
  exits zero. Reviewer independently read the same explicit d5e05c31 diff:
  no changed paths. The earlier eleven-path restoration in `7cda7d59`
  removes the prototype; the subsequent merge preserves upstream changes.
  Native suites were not rerun merely to review this restoration and merge.
- **Retained Python tooling: focused checks passed.** Author ran
  `<root>/.venv/bin/python -m unittest scripts.test_sum_batch_parallel`:
  23 passed in 0.023 seconds. Two existing profile-clock controls passed in
  the initial combined run. The later exact async-frame correction ran
  `scripts.test_sum_batch_parallel.WorkerEvidenceTests`: six passed in
  0.001 seconds, comprising five existing tests and one new test. These
  overlapping runs are not 29 unique tests.
- Author Ruff lint/format and focused complexity checks passed. Lizard
  1.23.0 reported maximum CCN 8/7/5 for driver/decoder/tests, maximum NLOC
  91/27/68, at most five parameters, and no new waivers. Root's actual
  `scripts/verify_complexity_gates.py` reported “complexity ratchet holds”.
  Actual Bandit 1.8.6 scans and Opengrep 1.33.1 scans of the two subprocess
  rules reported zero findings/errors in the changed Python scope. That
  Opengrep result did not cover the complete Codacy rule pack. The later
  strings/test-only attribution correction changed no sensitive subprocess call.
- After runtime restoration, root ran the 24 tool tests plus the affected CI
  registration test: **25 passed**. This verifies that retained tooling does
  not depend on the removed prototype module; no new native build was needed.
  Root reran these same 25 affected tests after merging upstream's workflow
  change: **25 passed** again. These are the same tests, not 50 unique tests.
- CI registration and its configuration control place
  `scripts.test_sum_batch_parallel` after the existing development dependency
  sync and complexity gate, using `uv run --no-sync python -m unittest`.
  The bare package job does not run the dependency-using SUM tests.
- **Historical prototype only:** fresh strict Clippy passed in 122 seconds.
  Eighteen new focused concurrency/lifecycle tests and 52 existing affected
  controls passed, 70 unique Rust tests in total. Those results describe
  public fe0's discarded prototype, not a new runtime in the archive tree.
- Reviewer ran no native build, test, profile, timing acquisition, or remote
  operation. Read-only verification covered artifact hashes, every formal
  observation, all warmup/profile jobs, emitted intervals and verdicts,
  process receipts, frozen source/binary seals, and final source restoration.
- **New unresolved local test failures:** none. **Measured regressions:**
  dense four/eight CPUs in the rejected prototype. **Remote CI, Codacy,
  cross-platform checks, and combined coverage:** pending; no green or
  merge-ready claim is made.

The historical Rust commands, from the prototype worktree with
`CARGO_TARGET_DIR=<root>/target CARGO_BUILD_JOBS=2`, were:

```text
cargo clippy --locked -p calc-flow --lib --tests --example grouped_sum_state_benchmark --features benchmark-internals -- -D warnings
cargo test --locked -p calc-flow --lib --features benchmark-internals grouped_parallel -- --nocapture
```

The same fresh `calc_flow-d2ecfee9191de7de` binary ran these filters:

```text
operator::sql::incremental::grouped --skip grouped_parallel --test-threads=2
operator::sql::compact::grouped_float_tests --test-threads=2
operator::sql::incremental_tests::test_sql_incremental_grouped --test-threads=2
```

The three existing-control groups passed 12/38/2 tests in 1.91/9.73/0.10
seconds. Final rustfmt, whitespace, schema-drift, and changed-function
complexity checks passed. The full workspace suites and 90% combined
coverage gate were not run locally or represented as passed.

## Measurement Contract and Acquisition Gate

The final [protocol](../../../benchmarks/sum-state/sum-batch-parallel-20261104.protocol.json)
has SHA-256
`7455ab73636085a84b2dbd983cc831fcfdc23a37b54f76d63fe3efdcc568b72c`.
Its public candidate is fe0, not the earlier null-candidate draft. The
reviewer approved this actual receipt/protocol gate before sampling.

Exactly five cases were selected: dense streams at one/four/eight CPUs,
sparse stream at one CPU, and hot-key stream at eight CPUs. Affinities are
`[0]`, `[0,2,4,6]`, and `[0,1,2,3,4,5,6,7]`, as applicable. All use
DataFusion target partitions one. Eight logical CPUs report four physical
core IDs on this WSL2 host; host power and unrelated load are not observable.

Each case has two rounds of ten alternating baseline/candidate pairs and
one warmup per arm per round: 200 formal observations and twenty warmups.
Baseline and candidate eight-CPU profiles were collected first, six jobs
each with the first excluded. Each accepted profile contains five verified
processing windows; every candidate window must contain at least two
canonical OS SQL-worker TIDs. The serial baseline has no false worker gate.

One 600-second clock covers fixture creation, guards, preflight, startup,
warmups, full correctness validation, both profile captures/decoding,
statistics, publication, and owned cleanup. The acquisition completed once
in **224.693588034017 seconds**, prior debit zero, no retries. The unused
375.306 seconds did not become a second experiment.

The acceptance rule requires all five cases completed, dense four or eight
CPUs `improved`, and no selected `regression`. Under the maintained
two-round/5% rule, a regression requires both interval lower bounds above
+5%; otherwise an upper bound above +5% is inconclusive. An improvement
requires both upper bounds below −5%. Inconclusive timings establish
neither equivalence nor a speedup.

## Provenance, Correctness, and Resource Evidence

The reused baseline is actual public
`af400b38511bc7f6255fd3549518095b51b20e0c`, binary SHA-256
`be89d670e89c7142fdf89849ae86240d3df3b86b2bfa508b52979a525f4c96bd`.
Its compiled source bytes match the merged c9afa base; it is not relabeled
as a fresh merged-base build.

The fresh public fe0 candidate binary is
`08fbff37a2e1c7ac51b08a117172c8b276bb0c0ab146de61836d74e30e43239d`.
Its build receipt SHA-256 is
`f478fd69d7ce5f56acb4845b3444ec1c1103fc9027015ad052c2149a6628fe7e`.
Actual Cargo records show both core and example `fresh=false`, and the core
depfile includes the new prototype module. Candidate source inventories
contain 1,008 files and baseline inventories 1,003. Public Git blobs, live
frozen sources, before/after seals, binary, lockfile, compiler, features,
flags, configuration, vendor, and support compatibility were checked.

Both native trees use the audited common harness SHA-256
`657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`
from public841 as the sole instrumentation overlay. The prototype build
took 631.784 seconds and read-only validation 0.979 seconds, recorded
separately. Frozen native trees remain unchanged by restoring the delivery
tree.

The maintained fixture has a 10,000-row seed and twenty 100,000-row business
inputs. Upstream filtering yields one 80,000-row SQL record per business
input: ten legacy chunks, allowing five units with eight CPUs. This layout
was verified from the frozen source/DF55 coalescer path, rather than assumed
for arbitrary user records.

Hot-key generation maps business rows `row % 100 < 90` to one existing key,
preserving other fields, seed, and ten-percent tail. An independent Python
wrapping-arithmetic manifest and the native independent oracle cover every
prefix. All twenty-one complete 10,000-group snapshots are retained and
validated. Common657 requires non-null input; nullable behavior is covered
by the historical Rust controls, not this performance fixture.

Reviewer independently validated all 200 formal observations and all
twenty warmup plus twelve profile jobs with maintained correctness,
resource, and lifecycle validators: **232 benchmark jobs**. The 223
invocation receipts comprise 200 formal processes, twenty warmups, two
six-job profile processes, and one preparation process; all owned processes
were reaped. Both immutable native source/binary seals still match.

All nine published acquisition artifacts remain byte-identical, including
the eight files hashed by
[the manifest](../../../benchmarks/sum-state/sum-batch-parallel-20261104/evidence-manifest.json).
The manifest SHA-256 is
`3dee7293826afec794ab76babf4c4f8296a1a4d734e7b7167f2b4810f66cf164`;
the complete compressed raw evidence SHA-256 is
`2cd40b52508067421c2649529d7245e7b1e4d93b3b656a6aa6c91d38a6be3b7f`.
Raw stacks, selected windows, per-thread counts, original timestamp text,
clock alignment, commands, snapshots, and process receipts are preserved.
Large original perf files remain at disclosed local paths with hashes and
sizes; they are not claimed as part of the committed public archive.

## Performance Results

Pooled wall medians are descriptive; the verdicts use the paired intervals.

| Case            | Baseline P50 ms | Prototype P50 ms | Change  | Verdict      |
|-----------------|-----------------|------------------|---------|--------------|
| Dense, 1 CPU    | 353.969         | 361.425          | +2.11%  | inconclusive |
| Dense, 4 CPUs   | 376.225         | 435.468          | +15.75% | regression   |
| Dense, 8 CPUs   | 375.070         | 444.834          | +18.60% | regression   |
| Sparse, 1 CPU   | 33.471          | 34.572           | +3.29%  | inconclusive |
| Hot key, 8 CPUs | 112.540         | 127.855          | +13.61% | inconclusive |

Dense four-CPU round intervals are [+11.823%, +24.325%] and
[+9.895%, +28.625%]; eight-CPU intervals are [+11.620%, +28.313%] and
[+16.020%, +24.186%]. Both cases meet the frozen regression rule.
Hot-key intervals are [+10.914%, +24.742%] and [−0.881%, +27.935%];
its larger pooled median is not a confirmed regression under that rule.
No target is improved, so performance acceptance is **false**.

Dense four-CPU process CPU P50 increases from 427.090 to 548.670 ms, with
RSS P50 from 247.27 to 268.45 MiB. At eight CPUs, process CPU increases
428.307→587.846 ms and RSS 247.67→271.14 MiB; median average cores increase
1.152→1.324. These are whole-process measurements, including preload,
retained outputs, and validation, not isolated SQL working-set charges.

### Profile Interpretation and Causal Limits

Baseline has 1,792 selected samples in five windows; prototype has 2,302.
Every prototype window has exactly five canonical OS SQL-worker TIDs.
The exact `ParallelCpuWork::run` frame accounts for 558 samples (24.24%);
baseline has zero such worker samples. The earlier bounded overlap RED/GREEN
test establishes concurrent worker execution; these TID samples complement
that proof. Occupied CPUs do not establish elapsed speedup.

Prototype inclusive samples include `group_rank` 24.72%, fragment destruction
5.13%, finishing 18.68%, and commit 12.21%. Inclusive shares overlap and
cannot be summed into a wall-time decomposition. Actual native symbol
inventory contains the exact worker trait method; helper and coordinator
closure symbols are absent/inlined. The 33.84% Other bucket cannot be called
merge time, and absent symbols do not establish zero coordinator cost.

Both arms have about 29% unresolved leaf samples. All 3,752 baseline and
4,422 prototype raw timestamps match imported clocks, maximum error
2.98e−8 ms. Five windows per side, raw stacks, and timestamps are preserved.
Separate profile cohorts and CPU-sample percentages cannot establish
paired wall-phase deltas, memory-bandwidth saturation, lock contention, or
off-CPU causes.

Additional local grouping, coordinator rank work, partial-result ownership,
and destruction are plausible costs consistent with higher total CPU/RSS.
The evidence rejects this implementation on these cases; it does not prove
that every possible parallel SUM design regresses.

## Blocking Issues

The runtime prototype's performance failure is a blocker to shipping it.
Removing the prototype and preserving the current merged target's runtime
closes that delivery blocker. No remaining source/tool/evidence blocker was
found in the archive-only tree reviewed here.

Required final CI, Codacy, combined Rust coverage, and cross-platform results
remain merge gates. Pending, failed, cancelled, or absent checks cannot be
reported as green. The parent must resolve every required issue before the
already-authorized merge.

Historical prebuild review found and closed six implementation issues with
genuine RED/GREEN evidence: sparse state validity across 8,192-row slices,
exact floor partitions, independently owned record-schema metadata, actual
physical predicate payload ownership, AND/OR whole-record copies, and
simultaneous variable-literal evaluation copies. They remain documented as
prototype findings; the corrected prototype was still rejected by performance.

### Observed Codacy Hash Finding — Narrow Local Fix Approved

PR #537 at 4f3c9b72 reported one critical Opengrep issue for the Git blob
identity calculation in `source_equivalence`, rule
`python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1`.
The call constructs Git's canonical `blob <length>\0<data>` object ID for
comparison with `git ls-tree`, using `usedforsecurity=False`. Build/source
receipts and measurement seals continue to use SHA-256.

Commit fd4168a1 adds only an explanatory comment and a rule-specific
`nosemgrep: insecure-hash-algorithm-sha1` on that call. The blob header,
input bytes, comparison, failure path, and SHA-256 seals are unchanged.
This is an audited Git-format compatibility exception, not a global rule
exclusion or weaker security seal. No test merely mirroring the comment,
native rebuild, or repeated performance run was needed.

Reviewer read the primary-derived rule, qualified rule, verification receipt,
and actual complete-driver before/after scan JSON under
`<root>/target/sum-batch-parallel-20261104/security/`. Actual Opengrep 1.33.1
with the Codacy-qualified ID and `--no-rewrite-rule-ids --error` reported
one finding before (exit 1), zero after (exit 0), with no errors. Its
qualified ID is
`Semgrep_python.lang.security.insecure-hash-algorithms.insecure-hash-algorithm-sha1`;
the actual engine verifies that the annotation's suffix identifies that rule.
Before/after Python AST hashes are identical:
`fb396d2b904e0afed82c2a3c277f588920d7a16d63213fe8e8d8ae180b01a234`.
Author focused Ruff lint/format and diff checks passed; reviewer did not
repeat them. This local result does not establish remote Codacy completion.

The frozen protocol still records the measured public fe0 driver SHA-256
`52d29ece79b9472ae3fa030ed1fa7211a5f703a35e3aeffc4a3400ac8b0cd8bb`.
The annotated archive driver SHA-256 is
`29760a53fb59634ebd2ea9e507d99a7b5be79d7d7e22cb91e808028ce18675c8`.
The latter is a post-measurement source revision, not the acquired driver.
Reproducing the original frozen protocol uses the public fe0 tools; the
historical protocol, driver hash, matcher, statistics, and all sealed evidence
remain unchanged. No new observation or performance claim is made.

## Style Issues

None remain in the reviewed retained tool scope. No dependency, unsafe-code
exception, global complexity/security exclusion, or baseline weakening is
introduced. Subprocesses retain fixed validated argument vectors and owned
cleanup. The sealed generated evidence is preserved as its original output;
the parent analysis document supplies the final rejection and limitations.

## Test Coverage

Retained tool controls cover independent prefix arithmetic, seed/tail
preservation, non-null fixture admission, fixed cases/counts, public source
binding, compatible baseline and fresh candidate receipts, exact compiled
symbols, canonical TID attribution, profile-first gating, deadline ownership,
partial-profile truth, and durable late-publication demotion.

Historical prototype controls cover genuine two-unit overlap, reverse
completion with disjoint first-appearance keys, Int64/UInt64 wrapping/null
prefixes, WHERE versus FILTER, sparse merges, cold/warm bounded fallback,
funded payload/input/result lifetimes, cancellation/panic/drop retirement,
output refusal, global budget rollback, and actual managed Barrier/checkpoint
plus one-CPU recovery. The final archive has no prototype runtime to claim
these as new product behavior. No missing runtime test justifies another
native build for the byte-identical restoration.

## Documentation Consistency

The changelog, streaming guide, benchmark index, experiment analysis, and
specification outcome describe rejection and serial delivery. The original
specification/critique and prebuild review history explain the experiment;
they do not advertise shipped worker admission. Reproduction points to the
public fe0 prototype and common657 harness rather than compiling the restored
archive tree as if it contained parallel SUM.

The parent analysis preserves the actual per-case labels, resource costs,
single-partition/full-snapshot/non-null limitations, WSL2 topology, inlined
coordinator uncertainty, and separate build/measurement clocks. Prior result
directories and all nine new sealed artifact files are unchanged.

The final narrow merge review binds delivery to d5e05c31 and checkpoint
36cad599. The benchmark protocol and all measurement claims remain bound to
the original c9afa/af400 versus public fe0 comparison. The only later archive
documentation edits distinguish those identities. At that merge checkpoint,
measurement tools, tests, protocol, and sealed results had no delta from the
previous review. The subsequent hash-rule annotation is the separate
AST-identical followup above. Against current main, the CI workflow differs
only by the one SUM test-module registration; upstream's disk-space
stabilization is preserved.

## Next Single Optimization Hypothesis

Reduce per-group integer-SUM candidate/state container allocation and
destruction on the accepted serial path before adding workers. The fresh
baseline still attributes 13.06% inclusive samples to candidate construction
and 15.90% to group destruction. A typed fixed-width transactional state/result
representation could remove nested scalar containers while preserving
null/wrapping semantics, rollback, output immutability, and checkpoint encoding.

Those overlapping CPU shares are neither additive wall fractions nor a
speedup estimate. This is an unimplemented hypothesis requiring focused
allocation/correctness tests and a separately frozen paired comparison.
No extra measurement or implementation is authorized by unused time in this
completed experiment.

## Verdict

**Runtime prototype: Request Changes / REJECT for delivery.**
Its measured performance fails the frozen acceptance rule.

**Archive-only tree: Approve / source-and-evidence PASS.**
Runtime restoration removes the rejected feature; the complete negative
evidence, tools, and accurate outcome documentation are suitable to preserve.

This is a specialist source/evidence handoff, not a green-CI or merge-ready
statement. Only the final archive PR with all required CI, Codacy, coverage,
and cross-platform checks green may be merged.
