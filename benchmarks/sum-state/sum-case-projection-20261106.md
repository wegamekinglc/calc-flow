# Compiled sign CASE study: runtime comparison unmeasured

The runtime prototype is withdrawn. A focused allocation test verifies lower
allocated bytes for mixed CASE masks and higher allocations for uniform masks.
The planned end-to-end comparison stops at its live global quiet guard before
creating fixtures or invoking either product. All five cases remain
**unmeasured**: there is no throughput, latency, CPU-profile, equivalence or
confirmed-regression result. Delivery preserves tools, source-verification
evidence, the experimental source and the immutable stopped-run record.

## Stopped acquisition

The guard detects unrelated active Maturin/Cargo builds in
`calc-flow-op-analysis/extrema` and a Rust compiler targeting
`calc-flow-batch-sql-admission/pr3-plan/baseline-release`. It stops the one
authorized attempt. No competing process is killed and no retry or second
campaign uses the remaining budget.

Actual counts are all zero: formal observations, warmups, profile jobs,
benchmark jobs, snapshot checks and native workload invocations. Fixture,
preparation, timing and profile directories are absent. The owned controller
is reaped and the final handoff records no remaining owned processes. The
[collector report](sum-case-projection-20261106/README.md) and
[raw evidence](sum-case-projection-20261106/complete-evidence.json.gz) preserve
the failure. Counts described in the protocol/report's fixed plan are planned
work, not observations from this attempt.

The raw/summary snapshot reports **4.079536221979652 seconds**. The immutable
final collector manifest reports **4.2318943699938245 seconds** of the single
600-second budget. The earlier outer startup clock covers **4.748112839995883
seconds** through controller exit. Protocol archiving, read-only integrity
checking and settlement verification finish **507.54136232601013 seconds**
from the original maintained start, or **507.997293981025 seconds** from the
earlier outer start. Both remain below 600. The original collector files and
debit are unchanged; the separate handoff records this longer archival tail.
There are no additional workload observations or statistical calculations.

## Source audit and selected prototype

DataFusion 55 already projects CASE input to the union of its physical column
dependencies. In the stock expression, that union is side and quantity;
account/stock are outside the branch gathers. Another root projection would
be redundant. A selection wrapper cannot remove gathers either: CaseBody
filters before invoking each branch's evaluate method, rather than invoking
branch evaluate_selection.

The alternative Arrow filter/coalescer fusion also does not help the maintained
recipes at that boundary. Dense input is one 100,000-row record selecting
80,000 rows, before the coalescer; it is not pre-sliced to 8,192. Dense/sparse
selectivity and required plain Utf8 fields force ordinary filtering, while
all-trade controls pass every row. No fusion prototype or timing claim follows
from this source finding.

The selected experiment changes only private cached compiled projection:

```text
CASE WHEN p THEN c ELSE -c END
  -> (CASE WHEN p THEN Int64(1) ELSE Int64(-1) END) * c
```

It admits only a root searched CASE with one WHEN, a concrete Int64 column c,
and NegativeExpr over the identical name/index column. Existing DataFusion
CASE and BinaryExpr perform the work, with explicit wrapping multiplication
and the factor on the left. No new CASE evaluator, UDF, dependency, worker or
configuration is introduced. Scalar branches, other types, different columns,
casts and other CASE shapes retain the original projector.

The rewrite uses the ProjectionExec child's current schema and coordinates,
including after filter projection. Original/replacement type, nullability
and the complete output schema including metadata must match. Optional
construction or guard failure retains the original projector. Public graph,
diagnostics, metadata, SUM state, budgets and checkpoint identity are unchanged.
The compiled entry is stream-only; finite execution retains its planned route.

Predicate evaluation remains once per real runtime batch and before column
access. DataFusion's nullable/schema inference may additionally probe WHEN on
a dummy one-row, zero-column batch. Rewrite checks can repeat planning-only
probes. The admitted public path requires immutable-function templates and
bypasses selected UDFs; no real caller rows reach those probes. Do not read
the runtime assertion as one call across planning and execution. Whole-job
timing would include the additional schema-inference work.

## Component evidence and correctness

The fixed 1,024-row allocation fixture uses ordinary buy/sell strings, nullable
conditions/quantities and Int64 MIN. Account/stock pass through the projection.
Input arrays, the plan and projector construction are outside the counter.
The compiled path returns an owned Batch envelope; the direct DataFusion
reference returns a RecordBatch. Baseline overhead is 712 bytes in these cases.

| Component case       | DF reference allocated bytes | Baseline compiled bytes | Candidate compiled bytes |
|----------------------|------------------------------|-------------------------|--------------------------|
| Mixed mask           | 40,166                       | 40,878                  | 18,718                   |
| Uniform buy          | 678                          | 1,390                   | 18,534                   |
| Uniform sell         | 9,038                        | 9,750                   | 18,534                   |
| Plain-column control | 64                           | 776                     | 776                      |

The actual baseline RED passes independent values, exact schema/field/schema
metadata, Batch metadata and caller-buffer/deep-value checks before failing
only the final mixed allocated-bytes assertion. The final source passes it.
Mixed peak live bytes are 16,992 in the candidate versus 39,584 in the direct
reference. All eight whole-call/output-drop lifecycle checks settle net count
and bytes to zero. Full six-field allocation records are retained.

The scalar-factor CASE allocates one N-element Int64 factor and multiplication
another N-element result, approximately 16N value bytes. Its literal inputs
are only one element each. Uniform buy loses the original shared quantity
buffer; uniform sell previously needed only negation. Those measured
allocation regressions are real costs, not evidence of a universal memory
gain. Component counters do not establish whole-job speed, RSS or budget gains.

Five new focused tests and one maintained compiled-expression control pass on
the exact final test binary. They cover MAX/MIN wrapping, NULLs, scalar/array
conditions, slices/empty inputs, aliases/metadata, fallback shapes, actual
stock SQL/filter/cached second-batch acceptance and execution, reordered child
schemas, held outputs, cancellation/abandonment, predicate errors and retry.
The cancellation fixture flushes 8,192 selected rows before cancellation,
proving the rewritten projector actually ran. Instrumentation is test-only.

The stopped initial setup attempt ran no tests. A later first GREEN run passed
four tests and exposed an invalid private column-name fixture; the corrected
test checks upstream constructor rejection and keeps a valid different-column
fallback. A separate semantic MAX datum leaves the frozen allocation data
unchanged. Final five-test GREEN, strict scoped Clippy, formatting, whitespace
and generated-schema checks pass. All 68 inspected functions have CCN at most
eight, with no lint exceptions. Original failed and passing records remain
available in the [component archive](sum-case-projection-20261106/component-evidence.json.gz).

## Frozen but unexecuted workloads

| Planned case          | Recipe and purpose                                                           | Actual observations |
|-----------------------|------------------------------------------------------------------------------|---------------------|
| Dense, one CPU        | 10k seeds + 20x100k source rows, 80k selected per business batch; mixed CASE | 0                   |
| Dense, four CPUs      | Same input, CPUs 0/2/4/6; existing concurrency                               | 0                   |
| Sparse, one CPU       | 10k seeds + 20x100 rows/80 selected; retained-output control                 | 0                   |
| Uniform buy, one CPU  | One-cycle all-trade control, side fixed to buy                               | 0                   |
| Uniform sell, one CPU | One-cycle all-trade control, side fixed to sell                              | 0                   |

Uniforms immutably replace side in seed and every business record, preserving
the exact field/schema metadata, type, quantities, keys and other columns.
Separate buy/sell controls prevent opposite regressions from cancelling in
an alternating-mask average. Sparse retains its alternating uniform masks.
The mixed all-trade/no-outer-rejection control is omitted and remains unmeasured.

The frozen plan calls for two rounds of ten alternating pairs per case:
200 formal observations and twenty warmups. Both dense one-CPU products require
six profile jobs, excluding the first, with five accepted windows each. Every
job would verify twenty-one complete snapshot prefixes. None ran here. The
predeclared acceptance requires complete evidence, at least one exact dense
improved verdict and no confirmed regression under the maintained 5% rule.
This attempt provides no timing data with which to apply that rule.

The minimal fixture extension passes 52 focused helper tests, followed by the
strengthened fifteen new tests; the unchanged thirty-seven checks are not
repeated after that test-only assertion change. Legacy side=None IPC bytes
and manifests remain identical. Original Python tool RED test source was not
retained; its recorded hash and actual prefix-digest failure logs are preserved,
and final oracle tests are public. The archive explicitly discloses that gap
and does not reconstruct the missing source. The original Rust RED test
payload is retained in full.

## Builds and reproducible provenance

The frozen baseline is public main at study start,
`dc8393d2f33771706b32492e9c4fa5436c6baf12`. Experimental candidate
[b0a9633f](https://github.com/wegamekinglc/calc-flow/commit/b0a9633f44356022fd1aa169b4efa6b077bc2f7b)
has tree `07faa398d04efca2013913ae52d91f2378cef27c`. Both builds use common657
from public `84186554b078e25e71003a74fedc91c63b2b7f70` as the sole native overlay.
All 1,010/1,012 source inputs match their public product/harness sources before
and after compilation. Core/example compiler records are fresh=false with
terminal success. The 292 registry semantic artifacts, Rust 1.94, locks,
features and actual environment fields match except declared per-arm target
directories. Old binaries were not relabeled; ordinary cache copies preserve
the old products and use distinct inodes.

Baseline Cargo/helper exit codes are both zero, with 904.5965224800166 /
905.4985433990369 seconds. Candidate Cargo/helper also exit zero, with
887.2990134509746 / 888.2408762819832 seconds. The outer candidate wrapper
exits one only because a static expected depfile path omitted compiled/.
Original outcomes and traceback are retained. A separate static verifier
checks the actual compiled/sign_case.rs dependency in 8.153 seconds without
rebuilding or changing source/binary. These costs are separate from acquisition
and are not controlled build benchmarks; baseline compilation overlapped unit
test compilation.

Build receipt SHA256s are `e87d3acb...` and `26dece05...`; binary SHA256s are
`dc35a092b79ae0c6d8b0835ac9a551447e734a3bccdaaee0cdf2d2c4f1f22f9e`
and `de6ab7db510c0aabd70dd046d8998873ea883a5074f7096e545baca8d5b88a45`.
The [protocol](sum-case-projection-20261106.protocol.json) SHA256 is
`1fc506250e379baee916f38dc862a64ade9776825f87c19de8d461966045632d`.
Raw evidence embeds 63 exact portable build-proof records and thirteen
scientific helper payloads. Binaries and large full nm inventories remain local
with hashes. The separate handoff preserves five exact orchestration payloads,
including the executed outer startup/deadline driver, argv and outcomes.
They do not alter the scientific helpers or original collector outputs.

## Delivery and remaining uncertainty

All three prototype Rust paths are restored to baseline contents or removed.
Delivery integrates latest main `f78a7660fd905e7cf4143358497420e411ce06e0`,
preserving its separately guarded Float64 rolling-window extrema improvement.
The entire delivered core equals that main revision. The sign CASE rewrite is
not enabled. Experimental source, copied binaries and original proof remain
preserved; the integrated product is also unmeasured.

The [source-verification manifest](sum-case-projection-20261106/source-verification-manifest.json)
binds the unchanged collector manifest/protocol and separately copied component
archive and orchestration handoff. Final required CI and independent delivery
review validate the tools and documentation separately from performance.

A future runtime comparison needs a suitable quiet environment and its own
explicitly fixed, admitted measurement plan. It must retain both uniform
controls and cannot treat the allocation result as a latency gain. Nullable,
other-type, finite, checkpoint and tight-memory latency remain unmeasured.
This stopped attempt neither accepts nor disproves the optimization's runtime
hypothesis, and its unused budget does not authorize another automatic run.
