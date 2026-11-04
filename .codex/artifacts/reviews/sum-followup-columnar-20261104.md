## Grouped SQL UTF-8 output follow-up review

**Author:** wegamekinglc | **Branch:** feature/sum-followup-columnar → main

Base product: `4f2a6190d20ce18d10750cd87d86da0bec6f778f`.
Measured public candidate: `af400b38511bc7f6255fd3549518095b51b20e0c`.
Reviewed local delivery checkpoint:
`f709529666a4489b8b0615fcf69bc67af2c82023`.
Local delivery tree before this review artifact:
`bcad738f7ff904d7c9196f619c92058f771a874b`.

This is a branch review before the final evidence PR. A detached worktree at
the delivery checkpoint was used and removed after inspection. The local
checkpoint is not presented as an accessible GitHub revision. The reviewer
confirmed that all post-measurement changes relative to the public candidate
are evidence and documentation: `crates/`, Cargo manifests/lock, `.cargo/` and
`scripts/` have no further diff.

### Summary

Fresh profiling of the merged product selected output materialization as a
bounded optimization target. For chunks containing an exact Utf8 group key,
the implementation resolves each output group once, borrows its strings while
constructing a fresh Arrow array, and avoids intermediate owned-string clones.
Other types retain scalar conversion; no persistent output cache, checkpoint
layout change, public configuration or new parallel execution model is added.

The retained-state sparse streaming case establishes a measured improvement.
Dense streaming and finite results retain their more limited classifications.

### Build and Test Results

The reviewer inspected existing evidence rather than rerunning passing checks:

- **Meaningful RED/GREEN:** the old synchronous output boundary performed 558
  allocations; the candidate performed 29. Requested total bytes changed from
  27,641 to 13,337 and peak live bytes from 14,056 to 11,897. This isolated
  256-group/two-string-key probe is not an end-to-end timing or RSS result.
  Earlier zero-test fingerprint runs were rejected as evidence.
- **17 unique focused Rust controls passed:** ten passed in the initial grouped
  run, the ordering/snapshot control passed after its test-owner cleanup was
  corrected, and six fallback/recovery/budget/cancellation controls passed.
  Logs include `target/sum-followup-utf8-functional-fixed.log`,
  `target/sum-followup-utf8-snapshots.log`, and
  `target/sum-followup-utf8-fallback-controls.log` in the original repository.
  The initial 10-pass/one-failure run is not reported as an all-green run.
- **Fresh strict scoped Clippy passed**, recorded in
  `target/sum-followup-utf8-clippy.log`, with a 108-second build/check interval.
  The author also recorded the changed-file rustfmt, schema-drift and whitespace
  checks passing.
- **40 focused Python tests passed:** 13 follow-up protocol/guard tests and 27
  maintained measurement tests. CI registration controls verify that the new
  dependency-requiring tests run after the existing dev dependency setup.
- **Actual repository complexity gate and focused Ruff passed.** New tooling
  functions have maximum CCN 7 and at most five arguments. The changed Rust
  function report in `target/sum-followup-columnar-20261104/rust-complexity.json`
  records 16 changed functions with CCN at most 5 and NLOC at most 80. Existing
  unrelated full-file Lizard warnings are not relabeled as new findings.
- **Bandit 1.8.6 and Opengrep 1.33.1 passed locally** on the changed measurement
  modules, with zero findings/errors in the root-provided `bandit.json` and
  `opengrep.json`. The Opengrep scan uses the corresponding fully qualified
  subprocess rules. No analyzer exclusion or complexity-baseline increase was
  introduced.

Full product, coverage and cross-platform verification belongs to the pending
PR CI. These local results do not certify those required remote checks.

### Blocking Issues

None remain in the reviewed source, measurement evidence or documentation.
Two measurement-protocol issues raised before acquisition were corrected:

- The process guard previously missed directly invoked Rust test/benchmark
  binaries without a Cargo parent. It now inspects executable paths under the
  managed Cargo target roots, while excluding its own collector PID. Focused
  mocked tests cover direct libtests, sealed benchmarks and collector exclusion.
- P0 now requires fresh current-product capture and complete decoding before
  finite comparisons. Tests establish this order, reject missing/invalid
  profiles, reject an old 841 product posing as current main, bind P1 to the
  exact P0 binary, and reject a no-op current revision as a new candidate.
  A separate mocked late-publication test checks durable stopped statuses.

### Correctness, Ownership and Funding

- The fast path admits exact Utf8 key fields. Its first scan verifies scalar
  types and computes checked byte capacity. Unsupported scalar/type shapes use
  the existing conversion path, including LargeUtf8 columns in mixed-key output.
- Utf8 byte capacity is checked against the i32 offset limit before builder
  allocation, avoiding an append overflow panic. Empty result handling remains
  in the existing empty-record path.
- A chunk contains at most 8,192 groups. One temporary `Vec<&Group>` shares
  immutable group lookups across its columns. `StringBuilder` copies borrowed
  string bytes into fresh owned buffers; no group/string reference escapes in
  the returned Arrow array. Held older snapshots remain immutable after later
  commits and plan destruction.
- `output_records` still prepays output and scratch reservations before any
  chunk construction. Charges were not reduced. The reference vector, exact
  offset/data capacities, lazy null bitmap and fixed metadata fit the existing
  conservative per-row/column and encoded-key-byte charges. Arrow 59's builder
  transfers buffers with `mem::take`; there is no growing old/new value-buffer
  overlap. The reviewer inspected that dependency implementation.
- Small all-null and mixed-null/Unicode controls compare actual peak requested
  heap bytes against prepaid output plus the zero-input workspace lower bound.
  They complement the source-level capacity proof; they do not promise a
  process-RSS limit or larger admitted state budgets.
- HAVING, projection, ORDER/LIMIT, schemas, field/schema metadata and Batch
  metadata continue through the existing paths. Controls cover null/empty/NUL
  strings, distinct Unicode encodings, held snapshots and exact ordered output.
- Mixed Utf8/LargeUtf8 public-operator prefixes are checked against an
  independent DataFusion oracle through capture, drop, restore and continuation.
  Existing string-key integer/floating recovery, budget refusal/refund and
  cancellation controls also passed. State representation and checkpoint
  identity/layout remain unchanged.
- The ordering control's remaining 55,040-byte reservation belonged to the
  existing job-owned sort gather lifecycle. Its corrected cleanup follows
  `compact/post_order_tests`: close/drain the gather owner, drop context/job,
  then retain the zero-reservation assertion. This completes the test's owner
  contract rather than suppressing a resource-leak assertion.

### Measurement Protocol and Provenance

The frozen protocol fixes four distinct workloads, two rounds of ten alternating
pairs and one 600-second total budget. P0 completed capture, decoding, analysis,
finite comparisons and publication before implementation selection. It debited
35.703777270 seconds. P1 received exactly the remaining 564.296222730 seconds;
builds and ordinary functional tests were separate.

Every native arm uses common diagnostic harness SHA-256
`657de3e697907e473691c2fd0d1b99967d46577333135c50176801f6faa4e04e`,
overlaid only into permanently retained frozen build trees. Instrumentation
revision 841 is distinguished from product revisions throughout. The dev
example used for CI was not substituted into the measured arms.

The reviewer independently read Cargo compiler-artifact records for both new
builds: the core library and example each have `fresh=false` and source paths
under the correct frozen tree. Current main was newly compiled from 4f2 in
554.210007899 seconds. Its binary bytes happen to equal the earlier 841 binary;
the new compiler records prove that the old artifact was not simply relabeled.
The public candidate was newly compiled in 558.397989215 seconds.

P1's baseline is the exact P0 current binary, SHA-256
`17259e26f471722696cbc9c1af67ebc901d1d3c2c6f67b49f78f77c655137ab2`.
The measured candidate SHA-256 is
`be89d670e89c7142fdf89849ae86240d3df3b86b2bfa508b52979a525f4c96bd`.
Build receipts retain source maps, compiler/feature/lock/support/configuration
identity and compiler-log hashes. The reviewer checked P0's immutable artifact
hashes before admitting P1 and inspected both phases' final manifests and raw
build/observation identities.

### Complete Performance Results

P0 preserves 40 formal original-199d versus merged-4f2 finite observations and
four warmups. Its pooled change is +3.39%, with paired medians +0.657% and
-4.732% and wide intervals. The maintained result is **inconclusive**: this
closes the absence of a measurement, not the uncertainty about equivalence.

P1 preserves 160 formal observations and 16 warmups. Each of four cases has two
complete ten-pair rounds, with independent full snapshot/schema/hash/resource/
lifecycle checks. Results are compared against fresh adjacent main observations,
not selected earlier medians:

| Case               | Main P50 ms | Candidate P50 ms | Pooled change | Maintained verdict      |
|--------------------|-------------|------------------|---------------|-------------------------|
| stock-stream-1cpu  | 280.088     | 266.919          | -4.70%        | no-confirmed-regression |
| stock-stream-8cpu  | 278.658     | 269.054          | -3.45%        | inconclusive            |
| sparse-stream-1cpu | 34.302      | 28.346           | -17.36%       | improved                |
| stock-batch-8cpu   | 57.158      | 54.550           | -4.56%        | inconclusive            |

Only the sparse case establishes a greater-than-5% improvement under the
unchanged maintained rule. Its two paired intervals are
[-20.490%, -14.908%] and [-23.516%, -15.296%], each with 97.8515625% coverage.
The other classifications must not be promoted to confirmed improvement,
equivalence or a general no-regression verdict.

The sparse workload keeps 10,000 groups and emits 21 full snapshots despite
small updates, so output construction remains relevant. CPU/RSS statistics,
every formal CSV row, complete compressed warmup/formal/profile observations,
source seals and accepted stacks are committed under
`benchmarks/sum-state/sum-followup-columnar-20261104/`.

P0's fresh profile has 1,455 selected samples in five windows; P1's candidate
profile has 1,306. All imported per-thread timestamps were matched before exact
window selection. Inclusive `output_chunk` shares are 9.55% and 5.44%,
consistent with the mechanism but from separate cohorts. They are not paired
phase timings or another speedup measurement. Approximately 29% of leaf samples
are unresolved; no source-line/inline, off-CPU, hardware-cache or bandwidth
attribution is established.

P1 completed in 137.326696254 seconds. The final shared debit is
**173.030473524 / 600 seconds**, with zero retries, added samples, extra cases or
stopped stages. Both phase publications remain immutable. Eight logical CPUs
report four core IDs on this WSL2 host; streaming uses one partition. These
results do not establish new parallel scaling or performance of other key
types/checkpoint paths. Whole-process RSS includes preload, retained snapshots
and validation and is not SQL reservation size.

### Documentation and Verdict

The changelog, benchmark index, protocol, result reports and reproduction guide
match the final source and evidence. Public measured product identities are
separate from the older common instrumentation. Prior archived result
directories and their inconclusive verdicts remain unchanged. No post-measurement
production/tool change or unsupported performance claim was found.

**Approve — PASS for the reviewed source and scoped measurement evidence.**

No new PR CI, Codacy or coverage result was available at this review handoff.
The reviewer ran no duplicate native build/test, analyzer, benchmark or remote
status poll. Required remote CI/Codacy/coverage must all pass before the root
merges, as explicitly required by the user. This review alone does not certify
green checks or grant merge authority.
