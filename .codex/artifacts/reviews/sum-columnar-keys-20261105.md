# Columnar composite grouping-key review

**Research archive and Arrow security correction: PASS. Runtime prototype:
REJECT. Required final-head CI remains pending; this is not merge approval.**

## Scope and checkpoint

- Branch: `feature/sum-columnar-keys`; base
  `22f4b9a562bdd087a75d810aeeb2161f29a3d6b7`.
- Reviewed delivery checkpoint: local `67fc715786e494daf7fb8e726d69c989510be1b9`,
  tree `fcc25d8a6375f0f72310d044a60c0f89117a9780`, before this report update.
  It includes the post-acquisition security correction
  `5cde59211057b7564892995d032f8ce4616c92db`.
- Historical reviewed prototype: local verification checkpoint
  `6dd9475167928d1534b91d84c66b6dde7ead79f5`, tree
  `9b4fbd788d36580c04d64b43bff8813e2ba0aa64`. The measured public candidate is
  [e492158c](https://github.com/wegamekinglc/calc-flow/commit/e492158c4d7109c779b7b52aac87cf6ffd3e1613),
  tree `441f82dfb4558ce0d70613e50958b4da67e70d23`, exactly matching the
  subsequent source-review tree. Its seven Rust paths are restored/removed
  from the final archive delivery; the prototype remains experimental history.
- This report includes the adversarial design review against the
  [study specification](../specs/sum-columnar-keys-20261105.md), subsequent source
  review, tool review, actual acquisition gate, negative-result review and the
  required dependency-security follow-up for PR #545. The corrected public head
  has not yet completed CI at this checkpoint.
- Review used repository guidance, `$code-style`, current incremental SQL
  source and the pinned DataFusion 55 implementation. No compilation, tests,
  fixture generation or measurements were run by the reviewer.

## Design findings

The factory-selected `GroupValuesColumn<false>` is unsuitable here. Its
vectorized collision handling can reorder first-seen IDs and grow equality
scratch with collision chains; `size()` omits that scratch and collision-list
storage. A post-allocation `size()` check cannot establish prepayment.

The revised proposal uses the public `GroupValuesColumn<true>::try_new`
constructor directly. Its scalar interning assigns IDs in input order and
does not use the vectorized collision-list path. After each original legacy
chunk, only newly assigned IDs are canonicalized, in first-appearance order,
through the existing `RowConverter` and `group_rank` admission. Current row
indices then map through an explicit local-ID-to-partial-rank vector. This
preserves the original boundary for group-limit failures before later chunks;
deferring all admission until transaction end would not.

The proposed route remains private and ephemeral: exactly two actual Utf8
Column keys, one actual trusted built-in integer SUM with a Column argument,
no SQL WHERE or eager input checks, the exact default unbounded execution
memory pool, and the default compact-byte sentinel. Other shapes and pools
retain the existing route. Keep `requires_explicit_grouping` unchanged for the
existing primitive-key path. FILTER-false rows still create groups; WHERE
queries remain on the legacy route under this initial scope.

## Capacity and profitability constraints

The author's checked envelope is a credible basis for focused TDD. It uses
transaction rows `N`, chunk rows `R = min(N, 8192)`, configured group bound `G`,
and dictionary bound `K = min(N, G + R)`. The latter includes groups interned
in the same chunk before canonical admission rejects an excessive group.
Funding only successful retained groups would be insufficient.

The envelope separately covers both Utf8 builders and their old/new growth
overlap, offset and lazy-null buffers, hash buckets/control bytes and rehash
overlap, chunk hashes, local IDs/ranks/first-row scratch, unique-key `take`
arrays and canonical conversion. Per-column source offset spans bound copied
data and must not exceed the i32 offset limit. The dictionary is dropped
before history merging and output. Existing compact reservations continue to
fund candidates, partial states and persistent containers; an engine scratch
reservation funds the additional dictionary work. Actual native `size()` is
only a defensive check.

Before implementation is accepted, retain these testable constraints:

- All arithmetic, rounding, bucket growth and allocation layouts must be
  checked before construction. Overflow, unsupported shape or a failed
  prework proof selects the original path; it must not introduce an earlier
  error or a late serial retry after optional work.
- The census over input RecordBatches must have bounded cancellation/yield
  behavior. The author agreed to a header-only census that checks cancellation
  per record and yields every 128 records, without evaluating expressions.
- Use a declared input-work/group-cap selector, with its cost included in
  the allocation and runtime boundaries. A fixed `N >= 2 * G` floor is the
  accepted design: for successful inputs in this no-WHERE scope it guarantees
  at least half as many canonical rows. Implement it as `G <= N / 2` to avoid
  multiplying `G`. This replaces the initial `N > G` proposal, which could
  save only one row. Do not tune it from subsequent benchmark results.

The author also agreed to bound the complete positive precharge by
`isize::MAX`, with checked layout arithmetic and `K <= i64::MAX`, before any
native builder allocation. A constant two-field native schema avoids
retaining additional caller metadata payload. Optional constructor/funding/
reserve failures must drop optional state and decline before first interning.

On the maintained dense input, chunk-local canonicalization would encode
78,488 of 80,000 SQL rows, a reduction of only 1.89%. Transaction-wide
deduplication can reduce this to the 10,000-key union. Sparse and the proposed
10,000-row/all-distinct batch should remain legacy under the profitability
selector. That fourth measurement is then a pipeline/selection regression
control, not evidence about active-interner all-distinct performance.

## Required implementation evidence

Require an observed failing encoded-row/allocation test before production
edits. Measure total allocations and the complete owned heap peak across
selector/setup, interning, unique-key copying, canonical admission and partial
growth; retained-only counts or maps allocated outside that boundary cannot
support a full-peak claim. Compare the peak with the relevant scratch and
compact reservations, including new groups, late null-bitmap materialization,
long Utf8 values and a hash-capacity transition.

Focused controls must cover tuple equality for null/empty/Unicode/delimiter
values, original first-appearance order, FILTER-only groups, multiple records
and the 8192-row boundary, exact group/byte refusal and cleanup, cancellation,
discard/output refusal, wrapping/null SUM and unchanged checkpoint recovery.
Verify caller arrays and held outputs remain immutable, and that unsupported
budgets, pools, WHERE queries and primitive/generic shapes select the legacy
path. Local IDs must never enter persistent keys or checkpoint descriptors.

## Measurement-tool review

**Tool and protocol-draft PASS** at `5372c5d293781d9eff5b6634d333ddf6972263ae`,
following the original tool commit `9a44dd0e`. The amended specification fixes
five cases before acquisition: dense 1T/4T, sparse 1T, all-distinct 1T and the
first eligible `N = 2G` edge on 1T. Two rounds of ten alternating pairs produce
200 formal observations and twenty warmups. Both dense-1T profiles require
five sampled, clock-verified windows after excluding the first of six jobs.

The distinct/edge fixtures preserve the seed, select one/two complete key
cycles per business batch and set only `flow_type` to `trade`. Their independent
wrapping Python reference builds all 21 snapshot digests. Source inputs remain
immutable; incomplete owned fixtures are removed. Sparse/distinct are selection
controls, and the edge tests the least admitted duplication ratio. Near-below
latency, finite/nullable/UInt64/generic/checkpoint and tight-byte latency remain
unmeasured.

The driver retains the maintained correctness/resource/paired-statistics and
process lifecycle checks. Its single 600-second clock includes setup, fixtures,
profiles, decoding, formal samples, post-seals, analysis, publication and cleanup;
builds are separate. Invalid profiles, helper drift and late publication cannot
retain completed/accepted receipts. Eleven declared active Python helpers are
sealed, with this inventory explicitly distinguished from all transitive/system
dependencies. The CI registration runs the new module after dev dependencies
are installed.

Route evidence matches complete canonical symbols against sealed compiled
anchors. `Dictionary::intern` and `admit_columnar_keys` can support an observed
active route; the common `update_columnar_chunk` dispatcher cannot do so alone.
Prefix collisions receive no credit. Missing, inlined and unsampled work anchors
remain unresolved, with no zero-cost or worker-parallelism claim. Full stacks and
clock evidence remain available. The owner recorded actual prefix/dispatcher
REDs followed by 26 passing focused tests, Ruff/format and complexity checks;
the reviewer did not repeat them.

The actual acquisition gate subsequently passed against public `e492158c` and
protocol SHA256
`1ba3640f929df2964e330a99c4330bbb5a6736d32cf9d8bb8b064ca119d94422`.
The reviewer independently verified both receipt/binary hashes, all 1008/1013
pre/post/live source seals and public Git blobs with only the common657 overlay,
raw fresh-false core/example Cargo records, copied dependency files including
the new module, exact nm inventories and equal compiler/config/vendor/support.
Both exact route inventories were empty before acquisition. This was qualified
as nonseparable attribution, never permission to change matchers or claim a
Release path hit. The fixed five cases and original 600-second budget were
unchanged. No reviewer native workload was run.

## Historical prototype source review

The measured implementation followed the accepted design. It added a cached Copy
plan containing two column indices, an optional boxed transaction dictionary,
and a private module; it leaves `Group`, `Candidate`, persistent indices,
checkpoint descriptors and SUM arithmetic unchanged. The measured test-build
layout is plan 1040 bytes, transaction input 424 versus legacy-equivalent 416,
key plan 16/optional 24, `Group` 96 and `Candidate` 128. The existing 4096-byte
plan/workspace metadata floors cover the added inline fields; no per-group
container is widened.

For clarity, the implemented positive scratch envelope is the checked sum
below, where `w = size_of::<usize>()`, `r64` rounds bytes up to 64, and
`H = next_power_of_two(max(256, 4K))`:

- Per Utf8 builder: `256 + 4*max(8192, r64(B_i)) + 16*(K+5) +
  4*r64(ceil(K/8))`.
- Hash table/control and rehash overlap: `256 + 2*H*(16+1)`.
- Hashes: `4R*size_of::<u64>()`; rank and row scratch: `K*w + 8R*w`.
- Unique-key take/canonical workspace: `4096 + 4*(B_0+B_1) + 128R`.
- Additional object/schema floor: `4096`.

The 8192-byte string minimum follows DataFusion's builder initialization; the
multipliers cover pinned Arrow doubling/64-byte rounding and old/new buffer
overlap. Offset and lazy-validity growth, hash load-factor/control bytes and
rehashing are included independently of DataFusion's approximate `size()`.
Every operation and final positive sum is checked, including the i32 per-column
span and isize allocation bounds. The old compact candidate/partial/container
reservations remain in addition to this engine scratch envelope.

The observed whole-update RED encoded 32,768 rows/6,737,152 bytes. First GREEN
encoded exactly 256 rows/52,634 bytes, matching the independent distinct-key
reference. The final fresh GREEN allocated 1,002,262 bytes versus 8,041,056 in
the RED; heap peak was 672,331 versus 2,070,541 bytes. Total allocation count
increased from 1,679 to 1,726; held allocation count/bytes were unchanged at
1,044/236,224. These are whole-update allocation results using a current-thread
executor, not end-to-end speedups.

The conservative paid bound increased from approximately 9.26 MB to 63.27 MB;
it is not actual heap use or RSS. The probe subtracts pre-existing plan credit
and sums scratch/compact peak reservations as a conservative coverage bound.
The exact-unbounded/default-byte scope prevents this optional reservation from
reducing finite-pool serial admissibility. No charged-capacity improvement is
established.

The final preparation move into `update_records` preserves ordering: the caller
passes actual whole-Batch rows, and preparation/observation precede the first
chunk after unchanged schema/proof/base-workspace setup. Global coalescing has
no eligible two-key plan. `PlanWork::charge` forwards the exact old fee inputs.
The direct grouped-seed test caller now uses the actual 64+96 record rows for
both workspace preparation and `update_records`; `input_rows` does not shadow
the later `rows()` oracle helper.

## Build and test evidence

Author evidence was inspected rather than rerun. These fresh tests and strict
lint cover the historical prototype; failed intermediate compiles and the corrected
WHERE fixture are not counted as passing tests or production RED evidence.

- Historical prototype Rust: **38 passed, zero failed**: ten new focused tests and 28 exact existing
  controls on the same fresh `calc_flow-69fc90115393543c` test binary, SHA256
  `18deb6aa521a528855cef6dc67756b52418bd2dfb3c0463d3b51ab71dabfaa62`.
  The new suite compiled in 204 seconds and ran in 0.31 seconds; existing
  controls ran in 4.03 seconds.
- Strict scoped Clippy passed in 104 seconds, covering the core library, tests
  and the single benchmark example with `benchmark-internals` and warnings
  denied. Author fmt/schema/harness/whitespace checks passed; all 64 audited
  changed/new functions have CCN at most eight. No lint or complexity gate was
  weakened.
- Python: **26 focused tool tests passed**, with the owner's Ruff/format and
  complexity checks. Reviewer source inspection also verified CI registration
  after dependency installation. No package/full Python suite was rerun.
- Coverage, cross-platform and full product regression: **not run locally**;
  final-head required CI remains a merge gate. No new local failure remains.

Evidence logs are under the repository `target/` directory:
`sum-columnar-keys-red.log`, `sum-columnar-keys-ready-green.log`,
`sum-columnar-keys-ready-clippy-2.log` and
`sum-columnar-keys-regression-controls.log`. The last contains the literal
28-filter `--exact` libtest command and binary identity. Tool evidence is under
`target/sum-columnar-keys-20261105/tool-tdd/`.

The Cargo commands below ran in the development worktree with
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target`,
`CARGO_BUILD_JOBS=2` and `CARGO_INCREMENTAL=0`:

```bash
cargo test --locked -p calc-flow --lib --features benchmark-internals operator::sql::incremental::columnar -- --nocapture
cargo clippy --locked -p calc-flow --lib --tests --example grouped_sum_state_benchmark --features benchmark-internals -- -D warnings
```

The focused tool command was the repository managed interpreter running
`python -m unittest scripts.test_sum_columnar_keys`.

## Coverage and consistency

The new tests cover whole-update encoding/allocation/credit and metadata sizes;
overflow/i32/layout limits without giant allocations; one-byte prework refusal;
census and active-update cancellation; signed/unsigned wrapping and null
continuation; tuple identity including null/empty/Unicode/NUL/delimiters;
FILTER-only groups; multi-record input; first-seen keys beyond 8192; changed
local/global order; capture/restore/discard; held outputs and caller immutability;
selected output refusal/retry; exact admission edge and earlier group refusal
before a later filter error; and legacy selection for below-edge, tight bytes,
custom pool, WHERE, mixed aggregates and LargeUtf8. The same-shape 80k-row test
observes ten columnar chunks and only 10k canonical encodings.

Existing controls cover unchanged generic/container fees and one-byte refusal,
native codec corruption/refund/wire rules, mixed and floating aggregates,
ordered LAST_VALUE, capture, finalization cancellation and output metadata.
Checkpoint bytes continue to come from canonical keys and the existing writer;
ephemeral IDs are not serialized. There are no unaddressed source/style blockers.

The specification and five-case protocol agree. Final narrative, changelog,
measurement index and streaming guide now identify this as a rejected experiment
and retain the existing runtime. The narrative states the exact experimental
activation conditions without recommending relaxed business memory limits.
Unconfigured compact budgets, ordinary finite-byte budgets and custom/bounded
engine pools were outside the active prototype route.

## Completed campaign and frozen verdict

The single acquisition completed in **147.604826276 seconds of 600**, with
prior debit zero and no retries, extra cases or observations. This is the final
manifest debit; the earlier summary write excludes the small publication tail.
Fresh baseline/candidate builds took 599.736/641.063 seconds separately. An
external build was recorded during candidate compilation, so those durations
are not a controlled build-performance comparison. Acquisition began only
after the maintained quiet guard passed.

The emitted outcomes below are retained without recomputing statistics:

| Case         | Baseline P50 ms | Candidate P50 ms | Change  | Exact verdict |
|--------------|-----------------|------------------|---------|---------------|
| Dense 1T     | 185.134         | 189.210          | +2.20%  | inconclusive  |
| Dense 4T     | 172.947         | 180.635          | +4.45%  | inconclusive  |
| Sparse 1T    | 22.838          | 22.725           | -0.49%  | inconclusive  |
| Distinct 1T  | 99.598          | 99.617           | +0.02%  | inconclusive  |
| Edge 1T      | 108.790         | 121.376          | +11.57% | inconclusive  |

Neither dense target has the required exact `improved` verdict, so frozen
performance acceptance is **false**. In particular, the edge intervals are
`[4.8841%, 19.5476%]` and `[5.5730%, 14.0848%]`. The first lower endpoint is
at most 5%; the maintained rule does not label this a confirmed regression.
No inconclusive case establishes equivalence. The runtime is rejected for
failing its predeclared acceptance contract, not because every case was proven
slower. No threshold tuning or reuse of the remaining time is approved.

The reviewer verified all nine manifest-listed file hashes/lengths, the exact
protocol copy, 200 CSV/formal rows with 40 per case and two rounds of ten per
arm, twenty warmups and twelve profile jobs. All 232 native samples match every
independent input/schema/value digest and all 21 snapshots: 4872 snapshot checks
with successful validation flags. All 223 embedded invocation receipts confirm
reaping. Native diagnostics report natural completion, no remaining tasks or
task errors, completed actors and empty edges. Helper-after seals match the
frozen eleven-helper inventory; completed finalization includes native source
and binary post-seals. No acquisition/evidence blocker was found.

The accepted profiles contain 876/963 selected user-CPU samples over five
windows each. Baseline window counts are 176/172/168/174/186; candidate counts
are 190/216/198/184/175. All 2319/2503 raw timestamps match imported clocks,
with maximum error below `6e-8` ms. Unresolved leaf counts are 173/174; perf
stderr reports no loss/unwind warning. Both exact route records remain
`nonseparable_or_inlined`, with null counts, as required before acquisition.

Existing emitted inclusive shares show RowConverter at 13.13% versus 2.28%,
candidate byte-key equality at 7.48% and candidate `group_rank` at 22.01%.
Absence from retained baseline top entries does not establish symbol absence
or zero cost. These are overlapping CPU shares from different cohorts, not
phase wall times or an isolated cause of the pooled elapsed changes. Reduced
encoding/allocation did not establish end-to-end acceleration. Resource medians
likewise do not establish a capacity or RSS improvement.

Evidence commit `e373aab2b818868f4a66b1a0354ff530c3b76b76` preserves the protocol
and ten generated files, including the manifest; manifest SHA256 is
`ccd0921e8b661f45d6e9d665333dd2337b6bb4872bed0fd2b976e75ce6262eb1`.
The complete raw gzip SHA256 is
`7cfe9466fe5a4d7176bc44959ccc203b07b8b77e9551f6eadc67eb27bfabe9e5`.
The [parent narrative](../../../benchmarks/sum-state/sum-columnar-keys-20261105.md)
and [sealed results](../../../benchmarks/sum-state/sum-columnar-keys-20261105/README.md)
accurately describe the source identities, timer boundary, resource/profiling
limits and unmeasured paths. Nullable/UInt64/generic/checkpoint/finite,
tight-byte and near-below-edge timings remain unmeasured.

## Final delivery scope

The archive restore returned all seven prototype Rust
paths to `22f4b9a`: the two existing files are restored and five new modules are
removed. That archive checkpoint had no runtime/API/schema diff against the
study baseline. The subsequent security correction changes the dependency pin,
root/fuzz locks and a test-only Batch module; the final delivery is therefore
not byte-identical to `22f4b9a`. The seven SUM prototype paths remain restored.
The preserved 38-test result concerns the experimental revision, not the
security-updated product.

The archive retains the reviewed Python collector/tests, one CI test-module
registration, specification/review, original negative evidence and explanatory
documentation. Its helpers and common native harness remain the measured ones.
Reproduction targets the public baseline and `e492158c` prototype with the sole
common657 overlay, not the archive's restored runtime. Original measured trees,
binaries, protocol and generated evidence remain preserved. The security change
does not reintroduce the prototype or add an API, configuration setting,
checkpoint wire format or thread policy.

## Dependency-security follow-up

The initial PR #545 dependency-policy check reported `RUSTSEC-2026-0442` for
the inherited `arrow-data 59.2.0`. The official
[RustSec advisory](https://raw.githubusercontent.com/RustSec/advisory-db/main/crates/arrow-data/RUSTSEC-2026-0442.md)
identifies improper RunEndEncoded validation and requires at least 59.3.2 on
the 59.x line. The [upstream reproducer](https://github.com/apache/arrow-rs/issues/11587)
demonstrates a parent logical offset/length that exceeds its last run end.
DataFusion 55's Arrow 59 semver requirements permit the patched dependency;
the Arrow 59.3.2 MSRV is below the workspace's Rust 1.94 requirement.

The reviewed five-file correction pins `arrow-data` to `=59.3.2`; root and
fuzz locks update only `arrow-data` 59.2.0 and `arrow-buffer`/`arrow-schema`
59.3.1 to 59.3.2. The latter two are required by patched arrow-data. The fuzz
lock additionally records the already-declared `datafusion-functions-nested`
55 dependency edge, without introducing a new package version. Full locked
metadata for both graphs contains a single 59.3.2 copy of each affected Arrow
crate. Vendor locks, audit exceptions and workspace lints are unchanged.

`batch.rs` adds only `cfg(test)` imports/module wiring. The shared 54-line
repository module constructs valid run ends `[2,3]` and values `[7,9]`, verifies
valid length/slicing, then requires errors for parent length 5 and offset 1
plus length 3. It never reads an invalid array element or uses unsafe code.
The Arrow-only probe imports this exact file by path; it is not a rewritten
surrogate test.

Observed old-dependency RED: two expected failures after the valid controls,
7.51-second build. Patched GREEN: two passes, 4.53-second build. Strict
Arrow-only Clippy passed in 2.43 seconds with all/pedantic warnings denied and
unsafe forbidden. The actual full-workspace policy command
`cargo deny --locked check` passed advisories, bans, licenses and sources;
pre-existing duplicate/yanked warnings remain warnings under unchanged policy.
Locked root/fuzz metadata succeeded. The reviewer inspected these logs and
resolved graphs without rerunning native work.

The probe commands use the manifest at
`target/pr545-arrow-security/Cargo.toml` and the shared repository test source:

```bash
cargo test --locked --manifest-path /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr545-arrow-security/Cargo.toml --test run_end_validation -- --nocapture
cargo clippy --locked --manifest-path /home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr545-arrow-security/Cargo.toml --tests -- -D warnings
cargo deny --locked check
cargo metadata --locked --format-version 1
cargo metadata --locked --manifest-path fuzz/Cargo.toml --format-version 1
```

The RED used the same test command before updating the probe's old dependency
lock, without `--locked`. Logs and resolved graphs are under
`target/pr545-arrow-security/{red,green,clippy,deny}.log` and
`{root,fuzz}-metadata.json`. Test/Clippy used the shared repository target,
two build jobs and `CARGO_INCREMENTAL=0`.

**The full core/DataFusion test graph was not built locally after this
dependency update.** The shared module's integration was reviewed statically;
final required CI must compile/run it along with the full product, coverage and
platform checks. The unchanged CI also runs cargo-audit with the documented
exceptions. Arrow-only GREEN is not reported as full-core GREEN.

The top-level experiment narrative and changelog now disclose the security
correction. Every recorded timing, allocation and profile remains scoped to
the original `22f4b9a` versus `e492158c` sources and frozen pre-correction locks.
The security-updated delivery has not been performance measured. No old source,
binary, protocol or generated evidence was rewritten to imply otherwise.

## Current verdict

**Archive and security-source PASS; runtime prototype REJECT.** Historical
source/tool safety approval did not confer performance acceptance. The delivery
honors the negative result, excludes the prototype, preserves reproducible
evidence and fixes the observed Arrow advisory without suppressing audit policy.
No unaddressed source, evidence, style or documentation issue remains.

Green required checks on the exact final public head remain a merge gate,
including dependency policy, coverage reporting and cross-platform checks. The
prior public head had the now-addressed dependency failure and a Coveralls
upload HTTP 504; the latter was not a unit-test or coverage-floor failure, but
successful final-head reporting is still required. Updated-head CI is pending
at this review snapshot, not green. Only the parent may publish/merge under the
user's existing authority after those checks pass. No further experiment is
authorized by the unused campaign budget.
