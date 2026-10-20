# PR #484 complexity remediation and performance controls

The two bounded timing controls are **inconclusive**. They do not establish
statistical non-regression or equivalence. Separate offline evidence shows the
rebuilt candidate and baseline native modules are **byte-identical**, as are
all runtime Python package files. This supports preserving native performance
for the tested default Linux release without interpreting noisy timings as a
regression or claiming a universal performance guarantee.

## Change and exact sources

Production `Selection::append_view` extracts its existing Runs arm into private
`append_runs`. The original traversal, early exit, partial-range clamping,
ordered `append_incoming` calls, and single `extend_from_slice` retain their
order. There is no additional pair loop, allocation, capacity change, per-pair
push, or cancellation/budget logic. Offset is read once in the selected branch.
The other Codacy change extracts test-fixture setup and is absent from release
code. Static comparison verifies the extracted body token-for-token after
whitespace normalization and moving that offset read.

| Arm        | Commit                                     | Tree                                       | Sealed source                                          |
|------------|--------------------------------------------|--------------------------------------------|--------------------------------------------------------|
| Baseline X | `98012d8a91bb3d0847a50cb77d21025f13629864` | `1ae0a8e48818c7464fbf51425c9931347cafa928` | `target/worktrees/join-single-thread-candidate-sealed` |
| Repair R   | `00c313ba0ab144e0f1537c7160ee323b5ccbef18` | `65c98f18e09b50a18245bbbba3c99e95c7aa6683` | `target/worktrees/pr484-remediation-candidate`         |

Baseline reuses X's original clean default-release manifest and artifact at
`target/performance/join-single-thread-performance/releases/candidate/`.
Candidate R was independently rebuilt in 608.593 seconds, outside the acquisition
budget, under `target/performance/pr484-remediation/releases/candidate/`.
The candidate build verifies clean source/HEAD/tree before and after compilation,
cleans the core release artifacts, and records forced core recompilation.
Its build log names core and Python compilation from the sealed repair source;
the independent reviewer checked those entries. Compiler, Cargo.lock, managed
Python, release/default features and flags match X. The subsequent evidence-only
commit S does not become a separately measured native build.

This is **X versus R**, not main `78b75724` versus R. Baseline timing samples
were collected afresh; earlier five-case observations were not reused.

## Exact native and package identity

Actual files from both sealed sites are identical across all **107,360,072
bytes**, SHA-256:
`2740ff5037253d672f52c77193cf7de45afbd29abf49779f42a5b85fea08cc60`.
Both the performance audit and independent reviewer verified the real files;
this is not just equality of manifest strings. Their formal workers loaded
those sealed artifacts and recorded the corresponding native proofs.

The candidate wheel SHA-256 is
`f474336e306489924033a53c572c01af18ebd42f0f17645a4170cf33add5e204`;
baseline wheel SHA-256 is
`5f07d5d94fd1f580971e9fbff6124718e2404f19adc4e74d38c6fedd82c32e85`.
Wheel member comparison finds only CycloneDX SBOM metadata and its RECORD entry
have changed content. There are no added/removed members, and every native and
Python runtime payload is unchanged. The differing wheel archive hashes do not
represent changed runtime instructions.

Both shipped modules are stripped; `nm --defined-only` reports no symbols.
Individual helper inlining cannot be attributed by symbol. Full-module byte
identity instead establishes that all emitted native instructions/data in this
exact default release are unchanged. No native import/execution, profile,
assembly-driven rebuild or extra timing was needed for this proof. Other target
platforms, compiler settings or feature combinations are not covered by it.

## Focused checks and fixed acquisition

The coordinator records 16 passing focused Rust checks: six Selection checks,
one ordered owner-assembly check and nine native gather/window checks. Fresh
scoped Clippy passed in 27.39 seconds; formatting and whitespace passed.
These are directly affected checks, not a full workspace/coverage run.

The pre-execution plan is
`target/performance/pr484-remediation/acquisition-plan.md`. Exactly two maintained
EngineCase interval controls use 1,000,000 rows on each side, one coalesced
physical RecordBatch per source, 64 keys, inclusive +/-5-second bounds and
10,998,080 exact output pairs. Explicit state/edge budgets and exact-cursor
lifecycle match. Product 1M/unlimited-memory defaults and CPU-derived capacities
remain unchanged. Both arms use the same explicit Tokio workers and affinity
(1 or 32 logical CPUs), with BLAS/OpenMP at one.

Each case completed two rounds of ten adjacent alternating AB/BA pairs, with
opposite round leads: **80 formal observations plus eight excluded warmups**.
The full acquisition, including startup, preflight, fixtures, complete oracles,
EOF and owned cleanup, completed in **181.699 seconds** of the fixed 600-second
budget. Requests were bounded at 570 seconds with 30 seconds reserved for
cleanup. No cases/samples were missing; there was no retry, sample expansion,
profile, budget extension, detected competing compiler/benchmark or cleanup error.

Timing remains ready-source enqueue through Arrow concatenation. The maintained
`report.comparison` rule is unchanged: both round lower bounds above +5%
confirm regression; any upper bound above +5% is inconclusive; both upper bounds
below -5% establish improvement; otherwise no-confirmed-regression. Classify
only matching machine/dependency/workload fingerprints within each case.

## Timing results

Times are milliseconds; each arm has 20 formal observations per case. P95 uses
the maintained linear percentile. P50 change is the ratio of aggregate medians;
round intervals concern median aligned-pair changes, a different statistic.

| Logical CPUs | X P50  | R P50  | P50 change | X P95  | R P95  | Maintained verdict |
|--------------|--------|--------|------------|--------|--------|--------------------|
| 1            | 708.60 | 707.46 | -0.16%     | 768.15 | 782.59 | inconclusive       |
| 32           | 520.82 | 538.13 | +3.32%     | 613.14 | 653.32 | inconclusive       |

| Logical CPUs | Round 1 paired median | Round 1 interval  | Round 2 paired median | Round 2 interval |
|--------------|-----------------------|-------------------|-----------------------|------------------|
| 1            | +2.23%                | [-8.12%, +9.40%]  | -0.67%                | [-5.58%, +1.19%] |
| 32           | +0.87%                | [-7.92%, +22.12%] | +2.01%                | [-5.07%, +6.72%] |

The intervals' exact order-statistic coverage is 0.978515625 for ten pairs
(at least 95%). Both cases have upper bounds beyond +5%; their maintained
verdict remains **inconclusive**, not no-confirmed-regression or equivalent.
Neither meets the confirmed-regression condition. The single-CPU candidate also
contains an approximately 1.048-second observation; it was retained without
outlier removal. These observations compare identical runtime payloads, so
their spread cannot be credited to a changed native algorithm.

The host is WSL2 exposing 32 **logical** CPUs, not a claim of 32 physical cores.
The guard saw no known competing workload, but host-wide activity/power remains
unobservable. The larger parallel intervals illustrate the uncertainty. There
is no whole-suite or cross-CPU scaling verdict, no comparison to the earlier
feature timing table, and no Polars or new profiling result.

## Correctness, lifecycle and resources

All 88 warmup/formal invocations passed exact pair identity/multiplicity,
values, schema/nullability and the expected resource/progress checks. Every
formal invocation returned through maintained `run_with_completion`; all eight
directly recorded warmup jobs ended naturally with zero live tasks/errors,
ended sources/sinks, the correct delivered rows and drained edges. Each worker
finished and owned cleanup succeeded. Interval global order hashes remain
**descriptive only**: the 32-CPU samples have two hashes in both arms, consistent
with legitimate cross-input scheduling. The full pair multiset and focused
Rust per-probe/run ordering remain the acceptance contract.

| Logical CPUs | X peak RSS MiB | R peak RSS MiB | X whole-sample CPU s | R whole-sample CPU s |
|--------------|----------------|----------------|----------------------|----------------------|
| 1            | 2,034.62       | 2,029.03       | 1.6912               | 1.6942               |
| 32           | 3,521.39       | 3,737.98       | 2.3685               | 2.3672               |

RSS is the maximum worker-process lifetime high-water mark, including fixtures,
warmup, allocator retention, output and full pair oracles. It is not active
operator/state allocation and has no separate confidence verdict. Whole-sample
CPU is a median including untimed validation/lifecycle work and cannot be
divided by ready wall time to infer native utilization. No memory or CPU
performance conclusion is established by these descriptive counters.

The offline audit recomputed both saved maintained statistics dictionaries
exactly, checked 80/8 sample/order counts, matching paired fingerprints,
frozen harness/Cargo.lock/build/wheel/native hashes and terminal evidence.
It imported no native package and collected no new timing.

## Acceptance advice and evidence

The timing result is **inconclusive** and supplies no statistical non-regression
or equivalence claim. Independent static evidence is stronger for this narrow
repair: the tested default Linux native executable and all Python runtime
payloads are identical, with source-clean forced-recompilation proof and
focused semantic/work-count checks. Accept the performance aspect of this
structural remediation for that exact configuration, subject to independent
review and green required final-head CI. Do not require favorable resampling,
claim universal preservation across other build configurations, or relabel the
timing verdict. Merge authority and final CI/coverage gates remain the
coordinator's responsibility.

The [compact evidence archive](pr484-remediation-evidence.tar.gz) contains
**56 members, 152,142 bytes**, SHA-256:
`cd4100a2ed26c5a4b7704734b7bb3a443f54e77b127784f93b22114856720cd7`.
It preserves raw report/worker observations, frozen maintained harness/helpers,
original X and rebuilt R manifests/logs, focused verification, offline audit
and actual native/wheel-member identity evidence. Every member hash and the
formal report hash were verified. See the
[member index](pr484-remediation-evidence-index.json) and
[verification record](pr484-remediation-evidence-verification.json).
Native/wheel binaries remain at their sealed paths; the archive retains their
hashes. Runtime checkpoint state is excluded; correctness/lifecycle records
remain included.

To reproduce statistics without native code, extract the archive, add its
`harness/` directory to `PYTHONPATH`, load `task/reports/measurement.json`, and
pass each case's saved two A/B sample arrays to archived
`scripts.benchmark_suite.report.comparison` with `comparison="interleaved"`.
The archived workspace audit additionally checks original sealed artifact paths.
No new measurement is needed or authorized by that reproduction.
