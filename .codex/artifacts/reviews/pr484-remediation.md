## PR #484 Review: perf: reduce streaming join bookkeeping and ASOF output overhead

**Author:** wegamekinglc | **Branch:** `feature/join-single-thread-performance` → `main` | **Files:** two source files in this remediation

**Reviewed repair:** `00c313ba0ab144e0f1537c7160ee323b5ccbef18`

**Repair tree:** `65c98f18e09b50a18245bbbba3c99e95c7aa6683`

**Performance baseline:** previously approved implementation
`98012d8a91bb3d0847a50cb77d21025f13629864`, not main.

### Summary

The repair addresses the two reported Codacy complexity findings by extracting
the existing Runs branch into `Selection::append_runs` and moving test-fixture
timestamp construction into `gather_fixture_opposite_times`. The production
loop, slice bounds, coalescing operations and single bulk copy are unchanged.
The fixture produces the same timestamps, duplicate positions and row order.

This review supplements the existing implementation review without modifying
its archived source or performance evidence. The complete source change from
the evidence-only predecessor `31b6aa19` is limited to these two files. No
additional per-pair work, allocation, cancellation check, budget operation,
public API, default or checkpoint-format change was introduced.

### Build and Test Results

- Rust: **Passed**, 16 focused tests: six match-selection tests, one owner
  fragment-assembly test and nine native gather/window tests. The coordinator
  compiled the actual repair worktree using
  `cargo test -p calc-flow --lib --features benchmark-internals --no-run`, then
  ran the three recorded filters against that binary. All passed with no
  ignored tests or failures.
- Scoped Clippy: **Passed**; the log records actual `Checking calc-flow` from
  the repair worktree and completion in 27.39 seconds.
- Formatting, generated project schema and whitespace: **Passed**, as recorded
  by the coordinator. This reviewer independently matched all seven hashes in
  `target/performance/pr484-remediation/verification.json` and read the compile,
  lint and three test logs.
- Python: **Not touched**; no additional Python suite was run for this
  extraction. The previously reviewed installed-candidate tests remain
  historical evidence for implementation `98012d8a`.
- New failures: **None in the focused local checks**. Actual repaired-head
  Codacy and full CI remain pending at this review handoff.
- Regressions: **No source finding or confirmed timing regression**; both new
  timing comparisons are explicitly inconclusive, as detailed below.

No native build, test, benchmark or remote operation was executed by this
reviewer. The coordinator owns those operations and the sealed source checkout.

### Blocking Issues

No remaining source, ownership, ordering, funding, cancellation or focused-test
finding. Required repaired-head CI, coverage, cross-platform checks and Codacy
must be green before merge. The coordinator's final snapshot of the old
`31b6aa19` head had 32 successful checks and only the two Codacy annotations
requiring action; that snapshot does not satisfy the repaired-head merge gate.

### Style Issues

No suppressions, weakened lint limits, dependencies or unrelated changes were
introduced. Extracting the existing branches reduces the two flagged methods'
complexity; the actual repaired-head Codacy result remains a remote gate.

### Test Coverage

Existing selection tests cover sliced runs, identity transitions, sparse
coordinates, run-boundary ordering, cursor parity and compact representations.
The owner-assembly and cached/serial gather tests retain the zero-per-pair-push
assertions. Native gather checks also cover both input sides, missing keys,
inclusive bounds, refusal/refund paths, final-stop publication and cancellation
at the existing 4096-item quantum. The fixture extraction preserves all inputs
to these tests. Additional tests mirroring a private helper extraction would
not add a behavioral guarantee.

### Performance Evidence

The coordinator completed the two predeclared interval controls in **181.70
seconds**, within the fixed 600-second acquisition budget: two rounds of ten
adjacent alternating AB/BA pairs per case, 80 formal observations and eight
excluded warmups. No retry, sample expansion, extra case or profile was added.
The candidate release build took 608.59 seconds, budgeted separately; the
original clean baseline release was reused with its original manifest.

| Logical CPUs | Baseline P50 ms | Repair P50 ms | Change | Baseline P95 ms | Repair P95 ms | Round 1 interval % | Round 2 interval % | Maintained verdict |
|--------------|-----------------|---------------|--------|-----------------|---------------|--------------------|--------------------|--------------------|
| 1            | 708.60          | 707.46        | -0.16% | 768.15          | 782.59        | [-8.12, +9.40]     | [-5.58, +1.19]     | inconclusive       |
| 32           | 520.82          | 538.13        | +3.32% | 613.14          | 653.32        | [-7.92, +22.12]    | [-5.07, +6.72]     | inconclusive       |

The maintained 5% rule classifies both comparisons as **inconclusive**. These
timings establish neither non-regression, equivalence, improvement nor a
confirmed regression. They must not be relabeled as passing timing gates.

There is stronger, separate evidence for preservation of executable work in
this tested configuration: both independently built default-release native
modules are **byte-for-byte identical**, each 107,360,072 bytes with SHA-256
`2740ff5037253d672f52c77193cf7de45afbd29abf49779f42a5b85fea08cc60`.
This reviewer independently hashed and compared both files. The repair build
log explicitly records compilation of `calc-flow` and `calc-flow-python` from
the sealed repair checkout. Thus this extraction adds no executable operations
to this exact build; that fact does not change the statistical verdicts or
constitute a guarantee about every compiler, platform or workload.

This reviewer independently checked the plan and build-manifest hashes,
matching per-round machine/dependency/workload identities, all 88 full-output
correctness and lifecycle records, all eight completed workers, and all eight
empty stderr logs. Each sample produces exactly 10,998,080 pairs, retains 320
rows and evicts 999,680 rows per side, and records zero state/match failures.
The oracle checks exact pair multiplicity, identity, values and schema;
cross-input order hashes remain descriptive only. Focused Rust tests verify
the required per-probe/run order.

The 32-CPU control uses 32 logical CPUs on WSL reporting 16 core groups, not 32
physical cores. Process-lifetime RSS includes fixture/oracle/allocator work;
whole-sample CPU includes untimed work and does not measure ready-window
utilization. Host-wide quiet and power mode cannot be established from WSL.
These controls do not replace the original five-case main-versus-implementation
results, establish a scaling verdict, or measure unmodified ASOF paths.
Raw results and the offline audit are under
`target/performance/pr484-remediation/`.

### Documentation Consistency

No behavior or public documentation contract changes. The remediation analysis
must preserve the inconclusive timing verdicts and distinguish the exact-build
binary identity from timing evidence. The previous implementation review and
archived five-case results remain unchanged and apply to their recorded source.

### Verdict

**Approve** the reviewed source repair. Exact structural preservation, focused
work-count checks and identical rebuilt native bytes leave no justified
implementation or executable-performance blocker in this configuration. The
timing evidence remains inconclusive. Merge still requires the actual final
head's green required CI, coverage, cross-platform and Codacy checks; this local
approval does not report those pending gates as passed.
