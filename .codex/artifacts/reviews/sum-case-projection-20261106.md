# Compiled integer sign CASE review

## Scope and checkpoint

**Source-only delivery and partial-evidence audits PASS. The runtime prototype
is withdrawn and the delivered core equals latest main f78a7660. The only
acquisition stops before any fixture or workload; all five cases are unmeasured.
No performance acceptance is established. Committed-artifact integrity and
green required final-head CI remain merge gates.**
Review branch
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

## Final acquisition-packet admission

**PASS**, with acquisition still at zero seconds and requiring the parent's
explicit GO. The public measured candidate is
`b0a9633f44356022fd1aa169b4efa6b077bc2f7b`, tree
`07faa398d04efca2013913ae52d91f2378cef27c`. The development worktree was clean
at that commit before this review-only update. This audit starts no native
build, test, fixture, workload, profile or statistical calculation.

The candidate receipt `build/candidate/build.json` has SHA256
`26dece05230e874c36a657342ffb9e93c1d5d880457cbb67b2a0bc1d2f690778`;
its linked and copied binary both have SHA256
`de6ab7db510c0aabd70dd046d8998873ea883a5074f7096e545baca8d5b88a45`.
All 1,012 candidate native inputs agree before/after/receipt/current and match
public Git blobs, with common657 the sole harness overlay. Original compiler
JSON shows rebuilt core and example artifacts, both `fresh=false`, and terminal
build success. All 292 registry semantic artifact records match the baseline.
Copied depfiles, compiler log, nm output, original maintained receipt and
driver-result hashes match their receipts. The core depfile names the actual
`compiled/sign_case.rs` relative to the verified rustc cwd; the example points
at the exact frozen common harness.

The actual Cargo and both rustc observations share the complete fourteen-field
compiler environment. Baseline and candidate differ only in their declared
per-arm `CARGO_TARGET_DIR`; each target matches its own receipt. Compiler,
lock, features, vendor and shared-support compatibility pass. The baseline
receipt e87d3acb and copied baseline binary remain unchanged. The candidate
cache copy is recorded separately at 12.329258648969699 seconds, using an
ordinary copy without hardlinks and retaining the old cache metadata.

Cargo exits zero in 887.2990134509746 seconds and the maintained helper exits
zero in 888.2408762819832 seconds. Those actual outcomes were atomically saved
before postproof. The original outer wrapper exits one because its static
depfile expectation omitted `compiled/`; its original traceback, failure,
driver and maintained receipt remain retained. A separate static correction
checks the real public module path and all proofs, completing in
8.153216684004292 seconds including finalization. It performs no rebuild or
source/binary change. This is a successful native build with a disclosed failed
postproof wrapper and successful corrected static proof, not an all-zero
wrapper history. Final ownership records contain no remaining build processes.

The frozen `protocol-final.json` is 4,621,039 bytes, SHA256
`1fc506250e379baee916f38dc862a64ade9776825f87c19de8d461966045632d`.
The reviewer runs only the pure/static protocol, helper, current-build,
compatibility and fresh-product validators; all pass. Actual CONTRACT case/
control descriptions, thirteen helper seals and public P Git identities match
their embedded UTF-8 payloads byte-for-byte. Both profile executables match
their retained version-record hashes; active tool identities are checked again
inside the acquisition budget. No profile has yet run.

The protocol retains exactly five cases, two rounds of ten alternating pairs,
200 formal observations, twenty warmups and twelve profile jobs. Both dense
one-CPU profiles require five admitted windows after excluding the first of
six jobs. One 600-second deadline starts with the controller; parent GO is
before the live in-budget quiet guard. Fixtures, captures/decoding, correctness,
resources, statistics, seals, publication and cleanup remain inside that same
deadline. Prior debit is zero. No case, sample, threshold, retry or residual
campaign is added. The unchanged maintained decision rule requires a dense
`improved` verdict, complete evidence and no confirmed regression on any case;
inconclusive intervals establish neither equivalence nor a confirmed regression.

Portable baseline/candidate proofs have SHA256
`73d53bd1fca6ba4082962a7f9855cfa3eb78bc31f24756a78007d0a72a3816ab`
and `a04f6e01c44cef2d228bea25f04c5f29ea746cd8051a9523187cd9b766ec9250`.
All 31/32 embedded records, gzip round-trips, external binary/nm seals and
public helper identities match retained originals. Both proofs equal their
final-protocol embedded objects, including the failed wrapper and correction.
The separate source-only component archive is 202,469 bytes, SHA256
`bfad68fff1f38849cb365c5f46e4f1f3c949ed8741c0bee40a475b1e7fdc9773`.
It preserves all original 56 entries and ten public source identities, adding
only exact initial/final Python complexity reports for 58 entries. The original
ca53 archive is retained. Its missing original Python RED payload limitation
remains explicit. Neither portable builds nor component records count as
formal observations. Original proposal/draft hashes remain unchanged, and no
acquisition or publication directory exists at this gate.

### Main advancement and delivery qualification

Read-only inspection finds that main advanced from dc8393d2 to
`f78a7660fd905e7cf4143358497420e411ce06e0`. Its exact two-file diff changes
`datafusion_rolling/argmax.rs` and the corresponding Python tests, extending a
guarded Float64 ordered-window rewrite to minimum as well as maximum. It does
not change physical CASE kernels, compiled-template admission, sessions,
dependencies or these five fixtures. The stock/SUM graph has no matching
Window-to-Projection shape. There is no direct overlap blocking this frozen
controlled campaign.

Keep dc/P sources, receipts and protocol unchanged for this single comparison.
After acquisition, integrate latest main and preserve that rolling fix; review
the integration and require green final-head CI. All observed timings must
remain attributed to dc/P. The integrated binary is unmeasured, with no baseline
relabeling or extra residual-budget experiment. Pre-acquisition admission is
not measured acceptance, final source-integration approval or merge approval.

## Stopped acquisition and partial-evidence audit

**Partial evidence PASS; end-to-end performance UNMEASURED.** The parent grants
one GO for the frozen protocol. The single controller exits one when the live
global quiet guard detects unrelated Maturin 54831, Cargo 54863 and rustc 54875
builds. The retained traceback reaches `prepare_fixture` and then
`guard(context.builds)` before the prepare invocation. No unrelated process is
terminated. No retry, additional case or residual-budget campaign occurs.

The actual counts are zero formal observations, warmups, profile jobs,
accepted profiles, benchmark jobs, snapshot checks and physical workload
invocations. All five planned case IDs are explicitly unmeasured. Fixture,
prepare, timing and profile directories are absent; the sample CSV has only its
header. Raw cases, fixtures, profiles, invocation receipts, targets and profile
clock evidence are empty. `performance_acceptance=false` records missing
evidence, not a confirmed slowdown, equivalence or proof that the hypothesis
cannot improve performance.

The immutable final manifest SHA256 is
`f25ecf6aa2d24f7169b32bcb953bdebb5a7139e3f81e81cf262a62352b808920`.
Its final collector debit is 4.2318943699938245 seconds out of 600, with prior
debit zero. Raw/summary 4.079536221979652 seconds is an earlier publication
snapshot, not the final stopped clock. The outer controller takes
4.748112839995883 seconds including process startup and is reaped; its PID
55706 is absent at this audit. Owned build/measurement remainders are empty.

Protocol copying, read-only integrity checks and settlement reach
507.54136232601013 seconds from the same original maintained start, or
507.997293981025 seconds from the earlier outer launch start. Both remain
within the original 600-second deadline. These separately disclosed archival
tail intervals do not replace or rewrite the immutable final collector debit,
authorize a retry or create a second budget.

Complete raw evidence SHA256 is
`06e04536ca09a58050934834b20da8336fab355c9553aa345fc224616ac9ccbb`.
The reviewer checks every manifest file's exact bytes/hash, the archived
protocol's unchanged 1fc50625 identity, and raw embedded protocol/build objects.
Both permanently detached source trees retain all 1,010/1,012 native input
hashes and copied binary identities. Thirteen scientific helpers match the
machine pre-seals, raw post-seals and embedded public payloads. All 63 portable
build records still match their retained originals, and component bfad remains
unchanged. No native execution or statistical recomputation is repeated.

The handoff SHA256 is
`b0a9b32c0c339cb28014379fd4380065e2d0641a7c3f18ecab6dc8590672c72b`.
Its five exact UTF-8 orchestration records retain `acquire-once.py`, the frozen
argv/cwd and actual launch/outcome/stdout/stderr. They match local original
bytes, sizes and hashes. The launcher owns one standard Popen and an earlier
outer deadline; it does not change the thirteen scientific helpers, recipes or
statistics. Retain this handoff separately in final portable publication and
bind it through the separate source/publication manifest, without changing
collector-generated files.

The predeclared unproven-result policy therefore withdraws the runtime
prototype while retaining public P, source/unit evidence, focused tooling and
the exact stopped packet. Production rollback, latest-main preservation,
source-only documentation and final-head CI remain the next delivery gates.

## Source-only delivery review

**PASS for the reviewed final working-tree content**, pending committed-artifact
integrity, actual final PR-head review and required CI. Withdrawal commit
`d13a6e07d86bc549d4a2707d0aec07c62604c636` restores the existing compiled
module and removes both prototype modules. Latest main f78a7660 is integrated
by merge `96458d29f2aa8754d8137c0c6d50e56944eb9610`. The reviewer independently
checks `git diff --exit-code origin/main -- crates/calc-flow`: the complete
delivered core equals main, including its rolling-extrema change. This final
delivery adds no runtime optimization, dependency or public surface.

The retained Python delta is the reviewed optional immutable side override,
fixed CASE study adapter and focused tests. The only workflow/release-helper
changes register the new test module in the maintained Linux SUM helper step
and its configuration assertion. Existing reviewed RED/GREEN and lint receipts
cover those unchanged bytes. Experimental Rust verification remains evidence
about public P, not a claim that its withdrawn rewrite is enabled in delivery.
No native checks or CI snapshots are repeated at this pre-PR documentation
handoff. Full final-head regression, platform and coverage gates remain CI work.

The reviewer reads the complete manual study and SUM index, checks the changed
benchmark-index and changelog context, and cross-checks their claims against
the source and actual receipts. The manual report separates zero observations
from the frozen plan, explains all four clock scopes, preserves the missing
Python RED payload qualification and failed static-wrapper history, and reports
both uniform allocation increases and the 712-byte envelope difference from
the DataFusion reference. It qualifies planning-only predicate probes and
keeps all runtime claims unmeasured. One wording issue is fixed: the correction
is a static verifier, not a read-only verifier, because it writes evidence
while preserving native source and binary bytes.

The separate `source-verification-manifest.json`, SHA256
`fd75c9564952f8b8d35e85155b06f5e1a0fac8cb85c197281861a6bdc64ecee9`,
binds exact unchanged collector f25ecf6a, frozen protocol 1fc50625, component
bfad68ff and orchestration handoff b0a9b32c. Published component/handoff copies
equal their original bytes. All 58 component records, ten public source
identities and five orchestration payloads remain separate from the immutable
collector packet; no collector file or its acquisition manifest is rewritten.
The source-manifest bindings and every collector file seal pass again after
packaging. All 49 relative links in the four manual documentation files exist;
newlines, whitespace, both new table alignments and generated-schema drift
checks pass. No unresolved content, style, source-integration or partial-evidence
blocker remains at this checkpoint.

This approves retention of tools, source/allocation evidence and the stopped
attempt. It does not approve a runtime performance change, establish a result
for any of the five cases, authorize another acquisition or declare the branch
merge-ready before its committed evidence and required checks are verified.
