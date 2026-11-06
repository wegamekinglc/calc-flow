# Compiled integer sign CASE study

Continue the authorized bounded performance research from latest main
`dc8393d2f33771706b32492e9c4fa5436c6baf12`. Preserve the original dirty stock
checkout and all prior sealed studies. The user has authorized fixes, final
required-CI waiting and merge after independent approval and green checks.

## Source findings and selected hypothesis

DataFusion 55 already projects CASE dependency unions. Stock CASE evaluation
uses only side and quantity; account/stock are not passed into its branch
gathers. A root dependency projection would be redundant. Branch wrappers
cannot avoid gathers because the existing CaseBody filters before calling
branch evaluate. Arrow filter/coalescer fusion also misses the maintained
stock recipes: their selectivity and plain Utf8 columns force materialization.
These source findings do not establish any measured gain.

Investigate one narrow rewrite inside the private cached compiled projection:

```text
CASE WHEN p THEN c ELSE -c END
  -> (CASE WHEN p THEN Int64(1) ELSE Int64(-1) END) * c
```

Admit only a root searched physical CaseExpr with no base expression, exactly
one WHEN, a concrete Int64 physical Column c as THEN, and NegativeExpr of the
identical name/index Column c as ELSE. Keep the factor on the left. Use existing
DataFusion CaseExpr and BinaryExpr with explicit wrapping multiplication;
do not introduce a CASE evaluator, arithmetic UDF, dependency patch or SQL
text rewrite. Preserve existing template eligibility. Do not generalize to
arbitrary branches, casts, scalars, UInt64, floats, decimals or callbacks.

Admit only existing validated, exact-schema cached plans with a Boolean
predicate. Check original/replacement data type and nullability, and exact
complete projector output schema including field/schema metadata. A failed
admission, construction or schema comparison keeps the original projector;
it must not introduce a new runtime error or weaken existing fallback paths.
Use ProjectionExec's current child schema and column coordinates; an upstream
filter projection may already have dropped or reordered scan columns.

Audit actual physical APIs before production edits. Predicate evaluation on
each real runtime batch must remain once and before column access; preserve
NULL predicate/quantity,
Int64 MIN wrapping, empty inputs, exact output schema/nullability/aliases,
Batch metadata, field metadata, input immutability, error envelope/order,
cancellation and cache/schema invalidation. Unsupported shapes retain the
existing projector. Plans and diagnostics outside the private compiled step,
threads, public API, SUM state, budgets and checkpoint identity stay unchanged.

DataFusion's nullable/schema inference may additionally probe a predicate on
a dummy one-row, zero-column batch. Optional rewrite checks can repeat these
planning-only probes. Admit only existing immutable-function templates with
selected UDFs bypassed; no real caller rows reach those probes. Do not claim
one evaluation across planning and execution. Whole-job timing includes the
additional schema-inference work.

The stock mixed-mask path currently gathers the side/quantity union into each
arm. The rewrite avoids those gathers but allocates an N-element Int64 factor
and result, approximately 16N value bytes. Uniform true currently shares the
column; uniform false allocates only negation. Both uniform cases therefore
have a source-predicted allocation regression. There is no universal memory or
speed claim; the end-to-end controls must decide whether to retain a prototype.

## Test-first and source gate

Independent design review precedes production GO. Start with a meaningful
baseline RED demonstrating changed mixed-mask allocation/operation work, not
an implementation-presence assertion. Verify values and exact schema against
an independent arithmetic oracle and the existing DataFusion route before
the failing improvement assertion. Include a positive control and report
uniform true/false allocations rather than hiding them.

Focused controls cover nullable predicates and quantities, Int64 extremes,
nonzero slices, empty inputs, scalar/array conditions, fallible predicates,
unselected fallible/callback branches, multiple WHEN, simple CASE, non-Int64
and mismatched-column fallbacks, output aliases/metadata, caller buffers,
multiple records, held results, cache schema changes, cancellation and retry
after errors. Only selected applicable tests require native execution.
Preserve unsafe_code forbid. Run scoped format, strict lint and complexity
checks; full workspace/cross-platform/connector/90% coverage gates belong to CI.

## Proposed measurement gate

Freeze the actual source, cases and protocol before any acquisition. Use five
common657 stream recipes: dense one/four CPUs, sparse, uniform buy and uniform
sell. The last two derive the maintained one-cycle all-trade control, then
replace side immutably with the respective constant. Preserve original schema/
field metadata, types, quantities, keys and every other column. Verify each
through the independent complete-snapshot oracle and input-identity seals.
Alternating uniform batches alone cannot establish each mask's latency; do
not hide one branch's regression in their aggregate. Sparse retains its
alternating buy/sell business batches as a separate output/control case. The
mixed all-trade/no-outer-rejection control is deliberately omitted and remains
unmeasured. Record exact names/reasons before GO.

Use two rounds of ten alternating pairs per case: 200 formal observations and
twenty warmups. Both dense single-CPU products require six profile jobs,
excluding the first, with five admitted processing windows each. Keep complete
timestamps/loss/unwind evidence and full stacks; absent/inlined symbols do not
prove zero cost or an isolated release-path hit. Preserve the maintained paired
5% rule, not a new threshold tuned after seeing results.

One 600-second acquisition, prior debit zero, covers fixture creation,
preflight/quiet guard, startup, warmups, oracle/resource collection,
profiles/decoding, statistics, local publication, source/binary/helper post-seals
and owned cleanup. Record builds separately. Do not add cases, samples,
diagnostic reruns, favorable retries or a second residual-time campaign.

Current-main baseline and reviewed public candidate need fresh core/example
compiler records, complete ordinary receipts, matching dependencies/compiler/
flags and actual environments, exact public source/blob inventories, identical
common instrumentation, binary copies, depfiles and pre/post seals. Old 1fc
and ranked-P binaries are incompatible baselines. Compatible cache may be copied
into new isolated targets without modifying old sealed evidence or products.

Actual source/build/helper/protocol admission by an independent reviewer and
explicit parent GO precede acquisition. Stop and settle owned processes on
unsuitable environment, failed evidence or original deadline; retain partial
results without a retry. Accept only complete evidence, at least one dense
exact maintained improved verdict and no confirmed regression on any selected
case. Inconclusive is neither equivalence nor improvement. Otherwise withdraw
runtime changes, preserving reproducible experimental source/tools/evidence.

Integrate latest main before final delivery; later integrated binaries remain
unmeasured unless separately authorized and measured. Preserve immutable
original evidence and disclose source/layout differences. Final specialist
review and green final-head required checks remain merge prerequisites.
