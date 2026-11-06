# Compiled integer sign CASE review

## Scope and checkpoint

**Candidate SOURCE and focused local verification PASS. Public candidate
build/final-protocol admission, acquisition/acceptance, final documentation and
required CI remain pending.** Review branch
`feature/expression-case-projection` against study-start main
`dc8393d2f33771706b32492e9c4fa5436c6baf12` using the
[study specification](../specs/sum-case-projection-20261106.md).
No candidate PR, check runs or prior reviews exist for this new study.

The reviewer performs read-only audits of source and author-produced evidence;
no native builds, tests, fixtures, workloads, profiles, statistical calculations
or CI polling were rerun. Only this review artifact is reviewer-owned; the
original dirty stock checkout and earlier sealed studies remain outside this
work item's edits.

## Source findings

The reviewer independently read the private compiled expression path,
template eligibility and the pinned DataFusion 55 physical CASE, binary,
negative, column and projector implementations.

`CaseBody::project` already computes a dependency union and remaps physical
columns. In the stock mixed-mask CASE this excludes account and stock, retaining
side and quantity. Its expression/expression evaluator filters that union
before calling each arm's `evaluate`; an arm `evaluate_selection` wrapper cannot
remove the gathers. A generic root projection patch would be redundant.

The proposed root-only rewrite uses existing DataFusion evaluators. A scalar/
scalar factor CASE creates one row-length Int64 sign array through Arrow zip;
the wrapping multiplication creates the result array. The two constants are
one-element scalar arrays, not two additional row-length constant arrays.
The source-derived value-buffer cost is approximately 16N bytes, before
validity, predicate and fixed allocation costs. This is not a measured peak.

Original uniform-true evaluation returns the column Arc without a new value
buffer. Uniform-false evaluation allocates its negation. The rewrite therefore
has real allocation regressions for both uniform shapes and must not be
described as a universal memory or speed improvement.

## Required implementation boundaries

Keep every one of these conditions explicit:

- Rewrite only a root physical `CaseExpr` in an already eligible cached
  projection: no base expression, exactly one WHEN, concrete Int64 `Column`
  THEN, and `NegativeExpr` of the identical column name and index as ELSE.
  Do not recurse into predicates or other expression shapes.
- Require the predicate's declared physical type to be Boolean against the
  validated ProjectionExec child schema, in its current column coordinates.
  Reject malformed, unknown or failed type checks as optional-rewrite misses,
  preserving the original projector.
- Keep the original predicate Arc on the factor CASE and place that CASE on
  the left of `BinaryExpr`. The pinned binary evaluator evaluates the left
  first; multiplication has no Boolean short-circuit path. On each real
  runtime batch the predicate remains once and before concrete column access.
- Use Int64 literals 1/-1 and explicit `with_fail_on_overflow(false)`.
  A physical Column produces an array, so the original `NegativeExpr` uses
  `neg_wrapping`; the replacement uses `mul_wrapping`. Do not generalize to
  scalar branches: scalar negation follows a different implementation.
- Check original/replacement data type and nullable equality. Preserve aliases,
  field metadata and schema metadata, and require full projector output-schema
  equality with the original plan. Never force an old schema onto expressions
  with different type or nullability.
- Keep the original projector when optional construction, validation or schema
  matching fails. Do not expand template eligibility or turn an optional
  optimization failure into a new user error.
- Retain existing selected-UDF exclusion, immutable-function eligibility,
  exact cache schema matching, evaluation order across outputs, cancellation
  checks, per-call filter state and error envelopes. Plans and diagnostic text
  outside the private compiled step remain unchanged.

For valid plans, NULL predicates still select ELSE, NULL quantities remain
NULL, and Int64 MIN uses wrapping negation/multiplication. Scalar predicates,
zero rows and nonzero-offset slices still require execution evidence. The
DataFusion expression/expression and scalar/scalar CASE methods use different
error wrappers for malformed non-Boolean predicates; the validated-Boolean
guard prevents broadening this compatibility risk into the admitted path.

No custom CASE evaluator, arithmetic UDF, dependency patch, new workers,
public API, SUM state, budget or checkpoint-format change is justified here.

## Test-first gate

Before production edits, observe a focused failure on the baseline's real
mixed-mask allocation/work behavior. A rewrite-presence counter or expression
shape assertion alone is insufficient. Check independent arithmetic values,
the existing DataFusion route and exact output schema before the expected
improvement assertion. Record the original output, source identity, command,
exit status and expected failure reason. Report uniform true and false
allocations as controls, including their increases after implementation.

Focused controls must exercise MIN/MAX/zero, nullable predicate and quantity,
scalar and array predicates, empty/sliced input, aliases and metadata, held
outputs and caller buffers, multiple records, cache/schema changes, cancellation
and recovery after a predicate error. Unsupported branches must retain lazy
errors/callback behavior: multiple WHEN, simple CASE, mismatched columns,
casts/scalars, non-Int64 types, arbitrary branch expressions and selected UDFs
are fallback controls. A fallible predicate must retain its error and output
evaluation order. Use the smallest applicable tests and existing controls;
full regression, platform and coverage gates remain CI responsibilities.

## Agreed measurement controls and remaining gate

Five workloads are a maximum, not a requirement to reuse the previous five
unchanged. An alternating uniform-true/uniform-false workload measures their
combined latency; it cannot establish the absence of a pure-true or pure-false
regression. Their allocation differences are asymmetric.

The updated specification resolves this concern with exactly five recipes:
dense one/four CPU targets, sparse one CPU, uniform buy one CPU and uniform
sell one CPU. The uniform controls derive the maintained one-cycle all-trade
fixture and replace side immutably, preserving exact schema/field metadata,
types, quantities, keys and other columns. Their independent full-snapshot
oracles and input seals must match those new semantics. Sparse remains an
alternating uniform-mask output/control case. The mixed all-trade/no-outer-
rejection workload is omitted and remains unmeasured. This is a reasonable
bounded selection pending review of the actual fixture implementation.

Exact cases, immutable public source, fresh build receipts, compatible compiler/
dependency/environment proof, helper/protocol seals, fixed paired samples,
both required profiles and one 600-second deadline require later independent
admission. No speed, equivalence or confirmed-regression verdict is available
at this checkpoint. Test-first production clearance, final source review,
measured acceptance, final documentation and green final-head CI still precede
merge.

The updated specification now includes validated Boolean admission, original/
replacement type and nullability comparison, complete metadata/schema guards
and original-projector fallback. No unresolved structural design blocker
remains under those conditions. Focused baseline RED authoring may proceed;
production editing still requires the parent's explicit test-first clearance.

## Measurement proposal and protocol-draft audit

**Planning PASS; neither draft nor proposal permits execution.** The reviewer
read the actual local proposal and protocol draft and verified their hashes:

- `proposal.json`: 9,980 bytes, SHA256
  `e32f97fac1989998dd402dc21692fc320d6ec4bbb9eb2fbb9434c3c2ff16bb92`.
- `protocol-draft.json`: 5,908 bytes, SHA256
  `8946736368211afbd4ea14f4283cdde485e3c60741e18d7dc9ab500c19114f91`.

The five exact case names are `stock-stream-1cpu`, `stock-stream-4cpu`,
`sparse-stream-1cpu`, `uniform-buy-stream-1cpu` and
`uniform-sell-stream-1cpu`. Their CPU/partition settings and workload reasons
match the specification. The fixed plan is two rounds of ten alternating pairs
per case: 200 formal observations, twenty warmups and twelve profile jobs,
with both dense single-CPU profiles requiring five admitted windows after
excluding each first job. Expected totals are 232 benchmark jobs, 4,872 full
snapshot checks and 223 managed native invocation receipts, not observed
results at this checkpoint.

The reviewer compared the proposed controls to the maintained `control_record`
and `DemoSource::trade_row` source. One complete key cycle is selected by
business index modulo ten, and the selected quantities alternate 200/100.
Twenty cycles therefore contribute 3,000 per key before the side override;
the zero seed contributes nothing. Forcing buy/sell gives the stated static
expected final positions +3,000/-3,000. Actual helper code must still preserve
all other values, exact field/schema metadata and source inputs, derive all
twenty-one snapshots independently, and bind transformed IPC identities.
No fixture or workload was generated for this audit.

The proposal retains one 600-second budget, prior debit zero and no retry or
residual campaign. It includes fixture/preflight/quiet work, warmups/startup,
oracles/resources, both captures and decoding, statistics, local publication,
post-seals and owned cleanup. Builds remain separately recorded costs. The
acceptance rule still needs one exact maintained dense `improved` verdict,
complete evidence and no confirmed regression on any of the five cases;
pooled medians and inconclusive results do not replace that rule.

Both products require fresh core/example compiler records and ordinary full
build receipts at current public source, common657, compatible environments/
dependencies, public-blob inventories, copied binaries/depfiles and pre/post
seals. Old 1fc/ranked-P products cannot be relabeled as current main. The
draft correctly leaves candidate, receipt and helper identities null and
portable proof pending. Planned thirteen helper paths are an inventory,
not a completed helper-source or acquisition gate.

One final metadata clarification is required before freezing: the proposal
currently describes parent GO as after the live quiet guard. The maintained
controller starts its one budgeted clock after parent GO and runs the live
guard inside that clock. Final protocol wording must say that parent GO
authorizes controller start and the in-budget quiet guard must pass before
fixture/profile/timing work. Preserve the original proposal/draft hashes;
this clarifies ordering without changing the run flow or counts.

No actual source/helper implementation, focused RED, candidate build packet
or final protocol has been approved by this planning audit. No native checks
were repeated. The parent's recorded CI-registration red/green remains
separate, limited test evidence and will be assessed with the final tool diff.

## Meaningful baseline RED audit

**PASS: the parent may grant production editing within the reviewed design.**
The reviewer read the actual attempt02 command, receipt, both logs, full frozen
test and source inventory. Frozen test SHA256 is
`b263877ad225fc999503b6811adcba785ac1a2bf63dc6929c56e8b2e855cf2e0`.
The native command ran one test, which failed with exit 101 after 0.02 seconds
of test execution; the separate build/command cost was 336.25410198402824
seconds. It was not a compilation/import failure.

All four probes complete independent arithmetic, exact field/schema metadata,
Batch metadata, legacy DataFusion output, shared-column Arc and deep caller
input checks before the final improvement assertion. All eight compiled/
reference lifecycle probes settle counted live bytes and allocations to zero.
The unchanged-column positive control passes. Only the final mixed-mask
allocated-byte comparison fails.

At N=1,024, compiled/reference total allocated bytes are:

- Mixed CASE: 40,878/40,166, with allocation counts 81/70, held bytes
  8,712/8,608 and peak bytes 39,792/39,584.
- Uniform buy: 1,390/678, with both results sharing the input quantity Arc.
- Uniform sell: 9,750/9,038.
- Plain columns: 776/64, with both results sharing the quantity Arc.

The baseline compiled envelope adds a constant 712 allocated bytes in all
four probes. This is acknowledged overhead, not a claimed optimization. The
failed assertion requires the mixed compiled path to beat the unchanged
DataFusion projector after accounting for that envelope; it does not merely
test for a new helper or rewritten expression shape.

The reviewer independently verified all 1,005 baseline public Git blob
identities and recorded current native hashes. The only native changes at the
RED checkpoint were the exact cfg(test) module declaration and the frozen test
file. The archived production `compiled.rs` bytes equal public dc8393d2,
SHA256 `88bf3a9a5aaee70a745675fed8863b76d724dd6a4b05a16b7d7c63b8d7c01719`.
No production rewrite existed during this actual RED.

Attempt01 is explicitly cancelled before any test executed: exit -15 after
77.51758815901121 seconds, with empty stdout. Its independent plain-column
oracle needed to retain quantity field metadata, while CASE delta metadata
is empty. The original files/receipt remain archived. It is not counted as
the meaningful RED. No benchmark acquisition time was spent by either attempt.

## Measurement-tool source and verification gate

**PASS** for the three reviewed Python source hashes:

- `scripts/benchmark_suite/sum_columnar_keys.py`:
  `3e6c61c19a93d9e244ab18753f12415e62b33acaf5f522fe4e64c19b47e9b4fb`.
- `scripts/benchmark_suite/sum_case_projection.py`:
  `a7ad9beb8bb043a5cb4753e6749c34d5425ee341a19ce4015ea3715f0ae6988d`.
- `scripts/test_sum_case_projection.py`:
  `7bb3d5e0874797f3ebfdd1da1f3959e3613976179da8f17a16478857e9ac0598`.

The reviewer read each changed helper, adapter and test. The extension is an
optional keyword-only side override on the maintained control recipe, a small
fixed contract adapter and fifteen focused tests. It adds no native harness,
global contract mutation, collector copy or measurement framework. Literal
buy/sell replacement preserves the original side type/Field, schema metadata,
other values and live buffer identities; the source IPC remains read-only.
Unknown sides, missing fields, failed/deadline fixture creation and owned-output
cleanup have focused controls.

The actual tool RED executes one test with two failing buy/sell subtests. Valid
legacy fixtures are written, but independent twenty-one-prefix wrapping
digests disagree with the requested pure masks. The raw failure log hash
`a686cfafdfa279c3677854299a2461135c4233b53e672602a2622d33654a3c4c`
matches the receipt; the pre-edit shared source equals public dc8393d2.
This is not a missing-API/import assertion. The original full RED test payload
was not preserved before later test expansion; only its recorded SHA256
`84afa0aa28afb4510666f7e154bc096d784c4aca00b2820a47fe0ffc8168bcb3`,
raw log and the reviewed current closed-form oracle remain. Do not reconstruct
or claim to archive the unavailable original payload.

Recorded verification is 52 passing tests: 26 existing columnar, eleven
existing candidate/rank and fifteen new CASE tests. After strengthening only
the new tests' buffer-address/size assertions, the final fifteen new tests pass
again; unchanged old tests were not repeated. Current source hashes match the
final receipt. Ruff/formatter evidence passes, and the sealed scoped Lizard
1.23.0 report covers 28 changed functions with maximum CCN six and no
exceptions. The reviewer did not rerun tests or lint.

Both pre/post legacy None IPC payloads are byte-identical for copies one/two,
SHA256 `e840d92a7b4e5c01ea73fb2000eda3ef84aa456e57520b1c493ed62e41d6d6ea`
and `b31f05d70c4e2a2d7634589c6a6141651d58f901614fc6c2aac7ccf35cd0f6eb`.
The receipts report identical legacy manifests; source inspection confirms
unchanged None names/mapping and control construction. Contract tests preserve
the two old studies, strict helper inventories, fresh-product/profile admission,
both uniform regression gates, original publication deadline and failure cleanup.

Final protocol metadata must be derived from actual CONTRACT cases/controls:
their descriptive reason/selector strings differ from the draft wording,
although IDs, recipes, counts and acceptance are unchanged. Merely filling the
draft's null fields would fail the strict protocol validator. Keep original
proposal/draft hashes and label these descriptive changes before freezing.

## Fresh baseline build admission

**PASS** for `build/baseline/build.json`, SHA256
`e87d3acbfd44504dea099397da03b3f5d6461e0dfcd2b1eb15fd8301cfa35e98`.
The copied binary SHA256 is
`dc35a092b79ae0c6d8b0835ac9a551447e734a3bccdaaee0cdf2d2c4f1f22f9e`.
This is an ordinary successful build with actual retained return codes and
durations, not a recovered receipt or relabeled old product.

The reviewer independently checked all 1,010 public source identities, before/
after/receipt/current source equality and exact public Git blob contents.
The sole overlay is common657 from public 84186554 in the permanently detached
dc8393d2 baseline tree. Manifest, lock, vendor, support and native sources
otherwise match the public baseline exactly.

The 327 original compiler JSON records contain exactly one rebuilt core and
one rebuilt example, both `fresh=false`, followed by build-finished success.
The two stored fresh artifacts equal those actual records. All 292 stored
registry semantic identities equal the normalized registry compiler records.
The reviewer verified compiler, original maintained receipt, registry, public
audit, copied depfile, nm, driver-result and copied-binary hashes.
Depfile contents reference the compiled CASE, current ASOF and SQL core paths
and exact frozen common harness; both fresh target source roots match that
baseline tree.

The actual Cargo-after-exec observation and both core/example rustc observations
agree on source cwd, Rust 1.94, jobs two, incremental zero, isolated target,
default Cargo home and the complete captured compiler environment. Wrappers,
RUSTFLAGS, encoded flags and release-profile overrides are unset; the original
Cargo command/actual argv agree. The initial rustup-shim observation is retained
and explicitly labeled; it is not misrepresented as actual Cargo environment.

Cargo exits zero in 904.5965224800166 seconds. The outer helper exits zero in
905.4985433990369 seconds, with its result saved before postproof. Cache copying
takes 4.966180767980404 seconds; postproof takes 0.8863347870064899 seconds and
Cargo observation finalization 0.012542576005216688 seconds, separately recorded.
The new target is an ordinary copy without hardlinks; old cache metadata and
sealed evidence remain intact. Build overlap with separately isolated debug
test work is disclosed, so these are costs, not controlled build benchmarks.
Final ownership records list no remaining build processes.

This baseline admission does not approve acquisition. Candidate production
source/GREEN, public candidate build, exact helper/final-protocol packet,
in-budget live quiet guard and explicit parent GO are still required. No
native process or benchmark invocation was started during this review.

## Final candidate source and focused verification gate

**SOURCE PASS** for these exact three source SHA256 values:

- `crates/calc-flow/src/datafusion_template/compiled.rs`:
  `13f13fe803f1eeeee6b063b18ea7af88a534dbff79a43212536cf13dab72b7a9`.
- `crates/calc-flow/src/datafusion_template/compiled/sign_case.rs`:
  `ef0b1937df4435f873b803383a636b9726a76ff26775591210c51ec27c42557d`.
- `crates/calc-flow/src/datafusion_template/compiled/sign_case_tests.rs`:
  `133a913ed9bf57b77b92c5664673443e074ec4356d453d8c6a5d3e71f5cb4a3b`.

The reviewer read the complete production helper, compiled integration and
test module, and reviewed the later targeted changes. The implementation
matches the root-only shape, concrete Boolean/Int64, identical column name/
index, explicit wrapping arithmetic and factor-on-left guards. It uses the
current projection child schema. Original/replacement type and nullability
must match, complete output schema/metadata equality is required, and an
optional miss keeps the original projector. There is no new CASE evaluator,
dependency, public API, worker policy, SUM state, budget or checkpoint change.

Only private borrowed-type naming and documentation formatting changed in
production after the first static review. No guard or arithmetic was relaxed.
The existing selected-UDF exclusion, immutable-template admission, exact cache
schema match, aliases, diagnostic plan and per-call filter state remain intact.

The final five new tests pass on all final source bytes, not merely on the
earlier partially passing test attempt. Actual exit status is zero, Cargo/
command elapsed time 175.5348984960001 seconds and test runtime 0.07 seconds.
Before/after source hashes equal current files and the final strict lint and
complexity packets. The maintained existing entry
`operator::expression::tests::stream_computed_and_filtered_projections_keep_sql_semantics`
also passes once from the exact resulting binary: 0.49333303701132536 seconds
total, 0.18 seconds test runtime, one test. Its recorded before/after and current
binary SHA256 agree at
`085efd75ac18b934326f1305282427c1e53db343d0924c46e0fb6015374a2786`.
The existing entry includes its schema/cache/UDF/error/cancellation controls.

The focused tests cover MIN/MAX/zero, nullable predicate and quantity,
scalar/array conditions, empty/nonzero-offset slices, fallible predicate
errors and retry, lazy rejected branches, exact schemas/metadata, input bytes
and Arcs, unsupported shapes/types/columns, held outputs and cache invalidation.
The MAX datum is confined to a new semantic fixture. Original RED quantity/
side functions and N=1,024 remain byte-identical to the frozen b263 source.

The actual stock expression test observes both acceptance and execution of a
rewritten cached projector, including child-column reorder/schema changes.
Cancellation and abandonment use a minimal first record of 10,240 rows,
yielding 8,192 trades, before a second 64-row record. This flushes the first
rewritten projector before the existing between-record yield. The test confirms
one rewritten evaluation before cancellation/drop, no published output, exact
input-buffer release and a successful retry, while held earlier outputs remain
unchanged. These are functional test traces, not an isolated release-profile
route-hit measurement.

The first candidate run's unsupported-shape test failed because its deliberately
inconsistent private Column name was rejected by the original DataFusion
projector constructor. The corrected test checks that upstream rejection
separately and retains the valid different-column/index fallback. The production
identity guard was not weakened. Initial lint errors were corrected through
documentation/type naming and test style, without lint exceptions. Older logs
remain retained; the final all-five execution supersedes them as source proof.

Strict scoped `cargo clippy --locked -p calc-flow --lib --tests --features
benchmark-internals -- -D warnings` passes on the same three source hashes,
exit zero in 86.82668208202813 seconds. The sealed Lizard 1.23.0 report has
68 functions, maximum CCN eight and no exceptions. The owner records scoped
Rust 2024 formatting. The reviewer independently checked current three-file
SHA256/Git blob/length identity, newline and whitespace, including untracked
new files, plus no schema drift and tracked whitespace errors. Ownership
records report no remaining native process groups. No native execution, lint
or statistical measurement was repeated by the reviewer.

### Schema-inference probe qualification

The predicate-once assertion covers each real runtime batch after projector
construction and counter reset. Pinned DataFusion `CaseExpr::nullable` can
const-probe WHEN using a dummy one-row, zero-column batch; optional original/
replacement schema checks can repeat that planning-only probe. The reviewer
checked that the public runtime exposes no session-catalog injection, selected
UDFs bypass compiled execution and template capture rejects mutable functions.
These extra probes therefore operate on trusted immutable expressions with no
real caller rows. They add planning work but present no new admitted-path
correctness blocker. Do not claim one evaluation across planning and execution;
whole-job timing includes the extra schema-inference cost.

### Final component evidence and limits

At the unchanged N=1,024 probe, compiled/legacy mixed total allocated bytes are
18,718/40,166, allocation counts 37/70, held bytes 8,528/8,608 and peaks
16,992/39,584. All compiled/reference lifecycle bytes and counts settle to zero.
Uniform buy and sell each allocate 18,534 compiled bytes versus legacy 678 and
9,038. Uniform-buy quantity sharing is lost. The plain-column positive control
remains 776/64 total bytes and shares the input quantity in both routes.

These observed component changes exclude prebuilt plans, inputs and oracle
allocations. They are not wall-time speedups, state-budget improvements or
whole-process RSS measurements. No selected end-to-end workload has run yet.
The fixed uniform controls must still decide whether their costs are acceptable
under the predeclared maintained rule.

No unresolved source/style/focused-test blocker remains. Full workspace,
cross-platform, connector, packaging and combined 90% coverage gates were not
run locally and remain required final-head CI. Source clearance permits public
candidate preparation; it does not bypass candidate build/helper/protocol
admission, parent acquisition GO, measured acceptance or final review/CI.
