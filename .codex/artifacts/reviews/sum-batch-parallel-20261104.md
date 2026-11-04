# Branch Review: Bounded Single-Batch Integer SUM Parallelism

**Branch:** `feature/sum-batch-parallel` → `main`

**Base:** `c9afa829ff025280ab346667582c62dc26a34785`

**Specification:** `abac023778b2dccf436b0434b50c6a279ae6735b`

**Reviewed source checkpoint:** `2dafaa83866b5651f5d92f03d0d9e176dbb427c7`
(local verification checkpoint; tree
`1d6427e41b4e4333ad9e17508c74dd24bf98e6bf`). Managed test/helpers are in
ancestor `ba503df4`, runtime documentation in `c70c7a27`.

**Reviewed tool checkpoints:** `bc178d1da9cd36128aafcd8b0ca824571ae50589`
and exact async-attribution followup
`f9c1eb508b94c08320e318a94b3aaa6366f7b521`. CI registration is in ancestor
`1b70d476`. These are local verification identities, not links to an assumed
published candidate.

**Preimplementation review:** [Concurrency critique](../critiques/sum-batch-parallel-20261104.md),
commit `f1d9ce645e2a7119c9da300d8892cd8b7d971221`.

## Summary

This prebuild review approves the frozen bounded integer SUM implementation,
its focused lifecycle/recovery tests, measurement tooling, and documentation
within the scope below. Workers compute owned local partials within one
original Batch; the coordinator preserves global group order, history,
complete snapshot acceptance, and the existing commit boundary.

The driver selects five maintained stream workloads and requires fresh,
SQL-specific multicore profile evidence before collecting paired timings.
No performance result or new remote CI result is available. Source approval
does not establish the required speedup or permit merge.

## Measurement Contract Reviewed

- Fixed cases: dense streams at one, four, and eight CPUs; sparse stream at
  one CPU; hot-key stream at eight CPUs. Every stream uses one DataFusion
  partition. Affinities are `[0]`, `[0,2,4,6]`, and `[0,1,2,3,4,5,6,7]` as
  applicable; eight logical CPUs report four physical core IDs on this host.
- Each case uses two rounds of ten alternating baseline/candidate pairs and
  one warmup per arm per round: 200 formal observations and twenty warmups
  across the complete inventory. There are no timing-driven retries.
- One 600-second clock starts before preparation. Fixture generation,
  preflight, process startup, warmups, full prefix validation, both profile
  captures/decoding, statistics, publication, and owned cleanup are charged
  to it. Native builds are separately timed.
- Baseline identity remains actual public product
  `af400b38511bc7f6255fd3549518095b51b20e0c`, binary SHA256
  `be89d670e89c7142fdf89849ae86240d3df3b86b2bfa508b52979a525f4c96bd`.
  It may be reused only with byte-compatible compiled sources against the
  merged base and the complete compatible receipt; it is not relabeled as a
  fresh c9afa829 build. Both arms use audited common657 instrumentation.
- The candidate must be a new, reviewed public revision with fresh product
  and example compilation. Public product identity, common instrumentation,
  and mutable development-tree identity are distinct.
- Dense input has 10,000 seed rows followed by twenty 100,000-row source
  business batches. Upstream `deltas` filters non-trades before SQL; each
  business SQL input has 80,000 rows, ten legacy 8,192-row chunks, and at most
  five local units at eight CPUs under the selected admission formula.
- The hot-key fixture changes only business account/stock keys for rows
  `row % 100 < 90`, preserving other fields and the seed. An independent
  Python modular arithmetic reference produces every prefix digest; each
  stream still retains and validates all twenty-one complete 10,000-group
  snapshots. Common657 inputs are non-null; nullable SQL semantics require
  the separate Rust controls.
- Full performance acceptance requires all five cases completed, an
  `improved` verdict on dense four/eight CPUs, and no `regression` on any
  selected case under the maintained paired-median/5% rule. Inconclusive
  results establish neither equivalence nor speedup. More occupied CPUs
  alone cannot satisfy acceptance.

## Blocking Issues

None remain in the reviewed source/tool scope. The following initial tooling
findings were closed in the frozen tool checkpoint and reviewed with their
focused regression tests:

1. A stopped acquisition with five complete timing summaries could still
   print performance acceptance as true in its README. `accepted_performance`
   now gates both report and manifest on completed status and the same
   fixed-case verdict predicate; `validate_inventory` checks exact counts.
2. Worker rules used SQL substrings instead of exact frozen function
   boundaries. `SQL_PHASE_FRAMES` now fixes complete local-worker/coordinator
   symbols; `_phase` compares canonical names for equality. `compiled_symbols`
   verifies an actual exact worker symbol before sampling. Prefix collisions,
   unrelated SQL frames, and imported segments of one OS TID cannot satisfy
   the worker gate. Each of the candidate's five windows requires at least
   two distinct canonical OS TIDs; the serial baseline has no false
   multicore requirement.
3. A partial or invalid decode could publish zero accepted profiles with an
   accepted disposition and unconditional claims of two verified profiles.
   Publication now records actual profile disposition and
   `sql_worker_evidence_verified`, demotes a completed collection if the
   required evidence is invalid, and reports actual accepted counts. The
   generic profile paragraph describes the protocol requirement.
4. The initial report described 100,000 SQL rows/thirteen chunks/six units,
   overlooking the upstream trade filter. The driver, draft protocol, and
   generated report now use the corrected 80,000 rows/ten chunks/five units.
5. Candidate source seals must be tied to the reviewed public candidate
   tree, rather than relying only on a receipt's revision string and a
   live-source-versus-receipt hash match. `source_equivalence` now compares
   candidate source blobs to the protocol's revision and baseline blobs to
   c9afa829. The sealed common657 harness is the explicit permitted overlay.
6. `clock_evidence` embeds raw sample timestamp text and clock-alignment text
   with hashes in compressed evidence. The large perf files remain locally
   retained with absolute path, hash, and byte count. Selected stacks,
   processing windows, and imported per-thread attribution remain preserved.
7. Two generated Markdown destinations were split across lines. Keep the
   complete-evidence and profile-target links intact; the regression test
   now checks the corrected destinations.

`finalize_acquisition` rechecks all native source/binary seals after
collection. Late publication durably demotes source acquisition/summary,
published summary/compressed raw/README, and manifest/phase receipt; revised
artifact hashes cover the stopped data. Completed case timings remain
available without a false collection-acceptance claim. Fixture deadline
failure removes only its owned partial destination, never the source.

## Style Issues

None found in the reviewed tool scope. No dependencies, global exclusions,
complexity-baseline relaxations, subprocess shell execution, or new native
executor were introduced. Existing audited subprocess ownership and fixed
argument-vector helpers are reused.

## Build and Test Results

- Reviewer: read-only source review; no Python tests, native builds,
  measurements, or remote status polling run.
- Author command:
  `<root>/.venv/bin/python -m unittest scripts.test_sum_batch_parallel` —
  **23 passed**, 0.023 seconds. Two existing
  `scripts.test_sum_state.ProfileClockTests` passed in the initial combined
  run and were not repeated unchanged.
- The later exact async coordinator-frame correction ran
  `<root>/.venv/bin/python -m unittest scripts.test_sum_batch_parallel.WorkerEvidenceTests`
  — **6 passed**, 0.001 seconds, following a genuine three-symbol RED.
  This includes five existing tests and one new test; it is not 29 unique
  new tests. Exact `merge_fragment`, `resolve_ranks`, and `merge_states`
  closure bodies receive coordinator attribution without receiving worker
  TID credit. The three synchronous worker anchors remain unchanged.
- Author recorded the expected RED observations before each corresponding
  correction, including worker classification, broad/prefix rules, fixture
  deadline, segmented TID, stopped acceptance, candidate source binding,
  partial profile prose, and broken links. The local
  `target/sum-batch-parallel-20261104/tool-red-green.json` is explicitly a
  session-observation ledger, not a claimed raw test log.
- Author Ruff checks/format on all three tool files and focused
  `C901,PLR0912,PLR0913,PLR0915` checks passed. Lizard 1.23.0 reports maximum
  CCN 8 for the driver, 7 for the decoder, and 5 for tests; maximum NLOC is
  91/27/68 respectively, with at most five parameters and no new waivers.
  Author diff/schema-drift checks passed.
- CI registration: reviewed `ci-linux.yml` and
  `test_sum_tool_tests_run_after_installing_dev_dependencies`. The new module
  runs through `uv run --no-sync python -m unittest` in the job with installed
  dev dependencies, after its existing sync and complexity gate; it is absent
  from the bare package job.
- Root ran the maintained
  `<root>/.venv/bin/python scripts/verify_complexity_gates.py` on the latest
  source: **complexity ratchet holds**. Root's actual Bandit 1.8.6 and
  Opengrep 1.33.1 scans reported zero findings/errors on the changed Python
  scope. The later f9c1eb50 strings/test-only delta changes no sensitive
  process call and did not repeat unchanged security scans.

Reviewer independently checked the final three tool files against the
frozen commit with `git diff` (no difference) and their SHA256 values against
the draft protocol. The protocol draft's SHA256 is
`1c4f726f62a92fdb725b0f8e3d6307cec4c64de2f8cbfd09e1c0d451ea314199`
after the reviewed exact async-frame update.
Its `candidate_revision` is intentionally null; it is not yet admissible for
measurement. No ordinary functional-test time or native build time is being
reported as measurement evidence.

## Test Coverage

Focused tests cover immutable wrapping-prefix reference behavior, explicit
non-null fixture limits, unchanged source/seed/tail and twenty-one prefix
digests, deadline cleanup, fixed cases/counts, old-baseline identity and fresh
candidate requirements, public-source binding, exact compiled symbols,
per-window/TID attribution, invalid-profile stop before formal cases,
profile/decode/gate order, owned cancellation finalization, and durable
late-publication statuses/raw timestamp retention. The tool tests do not
prove the future SQL implementation's concurrency or resource ownership.

## Documentation Consistency

Protocol and result prose must distinguish requirements from actual completed
observations, source rows from filtered SQL rows, reused baseline compilation
from fresh candidate compilation, and sampled CPU-thread shares from elapsed
speedup or simultaneous execution. A real bounded two-unit overlap test is
the concurrency proof; multi-TID samples are complementary profile evidence.

Prior result directories remain immutable. The final report must disclose
single-partition/full-snapshot retention, non-null performance fixtures,
unknown/inlined samples, and the WSL host-load limitation.

## Production Prebuild Review — Closed

The independent review found six source issues before the candidate build.
All are corrected in the frozen source checkpoint. The author reproduced the
first five in one genuine focused RED: seventeen tests, twelve passed and
five expected failures. A separate genuine RED demonstrated the final
variable-literal evaluation issue. Initial stale-binary/zero-test runs and
fixture/visibility failures were rejected as behavior RED evidence.

- **`grouped_sum::PartialAccumulator::merge_batch` — sparse validity.** The
  coordinator registers all fragment groups before merging state chunks of
  8,192 rows. DataFusion 55's first all-valid slice can mark every global
  group valid, including unseen groups in later slices. The actual RED
  returned `Int64(0)` rather than NULL for group 8,192. The integer merge
  adapter now performs the existing empty filtered update pattern before
  merging, preserving unseen validity across state slices.
- **`grouped_parallel::partition` — exact contiguous boundaries.** The
  initial front-loaded remainder policy differs from FR3's locked
  `floor(C*i/U)` boundaries. The non-divisible ten-chunk/four-unit RED catches
  that discrepancy. Checked quotient/remainder boundaries now implement the
  exact floor formula and retain the `usize::MAX` boundary proof.
- **`grouped_parallel::record_charge` — distinct record schemas.** A Batch
  requires structurally equal schemas, not shared schema Arc identity.
  Retaining every record can retain multiple independent metadata payloads;
  Arrow array memory size excludes them. The actual RED found 416 charged
  bytes versus 49,449 retained heap bytes for the record-specific ownership
  probe. Each retained record schema is now conservatively funded.
- **`grouped_parallel` predicate ownership — physical payloads.** Query length
  and node-count proxies do not bound actual expanded Column/Literal payloads
  after alias normalization. The actual RED found 1,536 charged bytes versus
  110,923 retained heap bytes. Actual admitted Column names, Literal values
  and field metadata, and recursive node/Arc ownership are now funded in
  addition to the conservative plan allowance.
- **`grouped_parallel` Boolean workspace — whole-record preselection.**
  DataFusion 55 AND/OR short-circuit evaluation can filter the complete
  RecordBatch, copying unused variable-width columns and nesting those
  copies. Key-only and fixed-per-row workspace do not fund that behavior.
  AND/OR now decline to serial before launch; their RED confirmed that the
  initial route accepted the unsupported shape.
- **`grouped_parallel::filter_charge` — simultaneous literal evaluation.**
  DataFusion clones a variable scalar in `Literal::evaluate`, clones it
  again for comparison normalization, and copies it to an Arrow scalar.
  The isolated RED measured 344,488 peak bytes against a 115,433-byte
  once-only fee. Checked `(1 + 4*U)` funding now covers shared retained
  payload plus per-unit temporary copies. The same one-unit assertion passes
  with a 577,165-byte fee. Row-count workspace is separate from this bound.

The strengthened existing overlap test now gives the two held units disjoint
first-appearance keys, deliberately completes unit one first, and verifies
both real overlap and exact output key order. Optional local-lane capacity
proof failures now decline to serial before launch; started failures do not
retry serially.

### Ownership and Transaction Proof

Admission requires exact unbounded scratch, at least 32,768 rows/four original
chunks, actual bound Column keys/arguments, only trusted unordered integer
SUMs, no eager normalization checks, and the fail-closed physical predicate
subset. Existing primitive NativeKeys fast paths remain serial. The unit
bound is `min(8, CPU allowance, floor(C/2))`, with contiguous exact floor
boundaries and unchanged per-record legacy chunks.

Checked scratch covers row/rank/map/key capacity and growth, input references,
each record schema, plan/schema/physical-predicate ownership, and per-unit
evaluation workspace. Variable key slices receive an eightfold envelope;
row/key constants cover encoding markers, offsets, ranks, and map capacity.
Each local `PartialGroups` separately prepays its existing capacity-rounded
native state formula, including three state copies and rounded validity
buffers. Gather execution/home/generation controls retain their existing
real reservations. Coordinator candidates/global partials/captures retain
their original compact-state charges and global distinct-group limits.
Reservations are conservative ownership charges, not an RSS cap.

Without an input backing owner, the package funds retained array memory;
otherwise it retains that owner through its last buffer use. Work fields
release records, physical expressions, and lanes before scratch/backing
credit. Fragment fields release keys/state arrays before their partial and
shared scratch credit. `WorkOutput` remains intact through coordinator merge;
no uncharged detached result escapes. The actual local heap probe measured
3,116,916 peak newly allocated bytes against 27,418,422 prepaid bytes; this
is a component ownership control, not end-to-end RSS or a speedup result.

Direct `WorkTicket::finish` waits for real settlement. Full stop checks occur
between chunks and at 1,024-key intervals; dropped futures transfer unfinished
work to the existing job owner. No per-Batch close of the reusable owner,
new executor, or new production gather primitive was introduced. The three
managed fixture/helper files use cfg(test)-only seams and assert that the
actual runner-owned gather home is cold before isolated service attachment.

Only the coordinator assigns slots and incorporates historical SUM state,
exactly once. WHERE and FILTER retain distinct group-existence semantics.
Integer wrapping addition permits local reduction; float/decimal/mixed and
unproven expression paths remain serial. Output refusal or any earlier failure
discards candidate state and ledger changes. Barrier/EOF/capture/next input
cannot overtake accepted output and commit. State-layout version, project
schema, public APIs, and Python bindings are unchanged.

### Actual Rust Verification

From the development worktree, with
`CARGO_TARGET_DIR=<root>/target CARGO_BUILD_JOBS=2`:

```text
cargo clippy --locked -p calc-flow --lib --tests --example grouped_sum_state_benchmark --features benchmark-internals -- -D warnings
cargo test --locked -p calc-flow --lib --features benchmark-internals grouped_parallel -- --nocapture
```

Genuine fresh strict Clippy passed in 122 seconds. The final test command
compiled the frozen benchmark-feature source and passed **18/18** in 1.13
seconds. Its injected worker panic is expected and caught by the passing
rollback test. The same fresh `calc_flow-d2ecfee9191de7de` binary then ran:

```text
operator::sql::incremental::grouped --skip grouped_parallel --test-threads=2
operator::sql::compact::grouped_float_tests --test-threads=2
operator::sql::incremental_tests::test_sql_incremental_grouped --test-threads=2
```

These controls passed **12 + 38 + 2 = 52**, in 1.91/9.73/0.10 seconds: **70
unique focused Rust tests total**. They include integer/native bulk prefixes,
float/mixed/filter cold recovery, sparse seeds, held output/schema/metadata,
and existing output accounting. Final rustfmt, whitespace, generated-schema
drift, and changed-function Lizard checks passed; maximum new-function CCN
is eight. Full workspace suites and the 90% combined coverage gate were not
run locally or represented as passed.

Reviewer read the actual final GREEN, strict Clippy, three existing-control
logs, valid sparse-merge/funding RED, first genuine overlap RED, and final
literal-copy RED under `<root>/target/sum-batch-parallel-*.log`. The frozen
source/tool paths have no post-commit diff. No native command was run by the
reviewer.

### Documentation and Remaining Gates

The runtime guide describes unbounded-only admission, native CPU availability
at service startup, separate DataFusion partitions, primitive fast-path and
AND/OR serial fallback, original-Batch/full-snapshot semantics, and unchanged
checkpoint compatibility. Root's final docs-only canonical spec alignment
records those same restrictions. No measured speedup is claimed.

The fixed maintained layout supports ten SQL chunks/five units without a new
run: common657 writes and reloads one record per 100,000-row source batch;
`CompiledExpression` filters that whole record. DF55's fresh
`LimitedBatchCoalescer` sets its bypass threshold to half of 8,192. Arrow's
coalescer emits the oversized 80,000-row filtered record intact when its
buffer is empty. The cold MemoryStream/FilterExec route likewise does not
pre-split the source record. Projection and bounded collection retain those
records. This is specific to the frozen maintained graph, not an assumption
about arbitrary user record layouts.

## Verdict

**Approve — prebuild SOURCE/tool PASS** for local source checkpoint 2dafaa83
and reviewed tool followup f9c1eb50. All identified source blockers are closed
with real RED/GREEN evidence. Publishing this reviewed source and building
its exact frozen candidate may proceed.

The actual public candidate, native build receipts, and completed protocol
still require acquisition review; the null-candidate draft cannot authorize
sampling. Performance acceptance, remote CI/Codacy, combined coverage, and
cross-platform results remain pending. No merge readiness or elapsed speedup
is claimed. Final evidence review will update this same artifact, and all
required remote checks must finish green before merge.
