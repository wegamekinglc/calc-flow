# Columnar composite grouping-key review

**Source and measurement tools: PASS. Performance, CI and merge approval:
pending.**

## Scope and checkpoint

- Branch: `feature/sum-columnar-keys`; base
  `22f4b9a562bdd087a75d810aeeb2161f29a3d6b7`.
- Reviewed source: local verification checkpoint
  `6dd9475167928d1534b91d84c66b6dde7ead79f5`, tree
  `9b4fbd788d36580c04d64b43bff8813e2ba0aa64`. This is not yet the public
  measured-candidate revision. Seven Rust files include the runtime module,
  integration, focused tests and one necessary existing-test caller adaptation.
- This report includes the adversarial design review against the
  [study specification](../specs/sum-columnar-keys-20261105.md), subsequent source
  review and tool review. There is no PR or final-head CI result at this
  checkpoint.
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

The reviewed draft remains inadmissible for acquisition until the actual public
candidate and fresh baseline/candidate receipts, native/helper/source seals and
final protocol are supplied. No measurement has been approved in this review
checkpoint.

## Frozen production review

The current implementation follows the accepted design. It adds a cached Copy
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

Author evidence was inspected rather than rerun. Final fresh tests and strict
lint cover the frozen source; failed intermediate compiles and the corrected
WHERE fixture are not counted as passing tests or production RED evidence.

- Rust: **38 passed, zero failed**: ten new focused tests and 28 exact existing
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

The specification and reviewed five-case protocol agree. A user-facing runtime
guide update and measured outcome documentation remain delivery work after the
campaign. They must state the exact activation conditions and must not recommend
relaxing business memory limits for performance. Unconfigured compact budgets,
ordinary finite-byte budgets and custom/bounded engine pools remain legacy.

## Current verdict

**SOURCE PASS** for the frozen local checkpoint above, and **tool PASS** for
`5372c5d2`. The scalarized dictionary, per-chunk canonical admission, complete
prepayment, cancellation and half-input selector satisfy the accepted design
constraints. No blocking source or style issue remains.

This permits publishing the reviewed candidate and preparing its fresh Release
build. Acquisition still requires the actual public candidate, complete matching
build/source/helper receipts and final frozen protocol. Performance acceptance,
outcome/activation documentation, final specialist delivery review and green
required checks remain pending. This is not merge approval.
