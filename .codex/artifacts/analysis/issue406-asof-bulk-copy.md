# Issue 406: complete-payload copying evidence

The complete eight-target issue remains active. This increment starts from
reviewed parent `467a3a9b`, whose ordinary product source is `8d3f6933`.
It changes complete ASOF payload compaction, preserving existing direct
admission, funding, cancellation and checkpoint boundaries.

## Implementation and tradeoff

Exact-buffer sharing is checked first. Non-shareable complete records with
at least 64 rows and supported flat columns without physical validity buffers
copy each selected contiguous range through Arrow `MutableArrayData`. Variable
arrays preallocate from their selected first/last offsets. Short, nullable
(including all-valid physical validity buffers), view and unsupported arrays
retain the original identity-index/take path. Schema and metadata are retained;
the source arrays remain read-only.

This removes the per-row identity-index allocation on the eligible branch.
It does not reduce the number of all allocations: bounded Arrow builder setup
increases component allocation count. Existing reservation formulas and
lifetimes are unchanged. Final three-column sliced controls use the exact
former-production function and show:

| Rows  | Previous allocations | Candidate allocations | Previous peak bytes | Candidate peak bytes | Existing reservation bytes |
|-------|----------------------|-----------------------|---------------------|----------------------|----------------------------|
| 64    | 20                   | 39                    | 2,756               | 2,256                | 12,858                     |
| 1,024 | 20                   | 39                    | 33,476              | 25,296               | 166,458                    |

Final retained component observations increase by **44 bytes**: 2,188 to 2,232
and 25,228 to 25,272. The initial reference instead showed old peaks
2,740/33,460 and old retained bytes 2,172/25,212, a 60-byte apparent difference.
Final review identifies that the initial reference used infallible
`Vec<ArrayRef>` collection while former production used `Result<Vec<ArrayRef>>`.
Their iterator size hints and Vec capacities can differ. The initial previous
arm's counts/bytes must not be called an exact former-production comparison.
Its logs remain preserved. The corrected reference restores the complete
original fallible function, including error mapping and Result collection.
`reference-source-match.json` proves equality with ordinary parent `8d3f6933`
after only function-name, module-path and formatting normalization; both
normalized hashes are
`a6c6bd287bf541d4734661799834988aeb9216dc80a54e02c85e483dd2b793f0`.
`compact-exact-reference.log` contains the final observations in the table.

Candidate peaks and representative reservation checks remain valid. IPC parity
and caller immutability are unaffected by the successful-reference collection
shape. Logical charges and encoded checkpoint bytes retain their definitions.
These component controls do not establish whole-phase allocation or RSS acceptance.

## Recorded verification

Local evidence is under repository `target/issue406-asof-bulk-copy-20261014/`.
The existing registration is reused to preserve libtest table size.

- RED: `compact-red.log` observes 2,046 identity indices for 2,046 selected
  rows, failing the new no-identity-index control before implementation.
- The first GREEN attempt is a compilation failure: the test-owned operator
  lacks `mut`. `compact-green.log` is preserved. The corrected focused test
  passes in 0.03 seconds in `compact-green-fixed.log`.
- Explicit exact-buffer sharing and previous/candidate allocation controls
  then pass in 0.11 seconds (`compact-sharing-allocation.log`). Array controls
  cover 0/1/63/64/65/127/1,024-row cuts, 24 flat types, variable slice offsets,
  Boolean bit offsets, IEEE payload bits, hidden null spans and all-valid
  validity buffers. They compare original-reference IPC bytes, schema metadata,
  caller bytes before/after and independent owned buffers where supported.
- Eight directly affected direct-admission tests pass serially in 12.12 seconds
  (`direct-lifecycle.log`). They include exact legacy checkpoint/restore and
  continuation, nullable projection/caller immutability, workspace/state error
  priority, dictionary fallback, canonical descriptors and credit limits.
  The reused registration also retains cancellation/abandonment drain checks.
- Initial scoped Clippy fails on two test-only issues: a redundant closure and
  a potentially truncating `u64` to `usize` cast. Both are fixed without lint
  allowances (`clippy.log`). The next combined command stops at formatting:
  one changed iterator expression needs a single line. It runs no test or lint
  observation. Formatting is corrected before the final focused rerun.
- The final focused test passes in 0.03 seconds after a 2m51s compile
  (`compact-lint-fixed.log`). Scoped all-feature core-lib/tests Clippy passes
  in 2m17s with `-D warnings` (`clippy-fixed.log`).
- The subsequent reviewer correction restores the exact former-production
  reference before publishing allocation comparisons. Only test reference and
  its unwrap sites change. The existing focused registration and scoped lint
  are rerun because that reference changed; product and the eight already
  passing direct/lifecycle controls remain unchanged. The focused registration
  passes in 0.03 seconds after a 3m40s compile; final scoped Clippy passes in
  49.62 seconds (`clippy-exact-reference.log`).
- Lizard 1.24.1 recognizes all 19 changed/new functions, with maximum CCN eight
  after the exact-reference correction (`complexity-exact-reference.log`). Generated schema drift
  and whitespace checks pass. Artifact structure checks cover headings, final
  newlines, whitespace and Markdown table widths.

No full workspace regression, required CI or combined 90% coverage result is
certified by this focused local verification. The final specialist review is
recorded separately under the shared work-item slug.

## Measurement selection and limits

The earlier ordinary `05c13fd9` diagnostic attributes 11.3956% inclusive
samples to `compact_complete_record`; that function is byte-identical in
ordinary parent `8d3f6933`. This remains the earlier product's diagnostic,
not a new-candidate profile or speedup claim.

Prebuild selection under `target/issue406-bulk-copy-paired-20261014/` fixes
ordinary parent `8d3f6933` and native SHA-256
`3664ebab40c619a0c4e7b8a9cfe93a684ab2da24e5c3ca52f8cd88c831041d08`.
The exact committed candidate will be frozen after focused checks and source
review. The three cases each use one million rows per input and true OFF:

1. Row 6, 64,000-row batches, 32 CPUs: changed bulk-copy path.
2. Row 6, 64,000-row batches, one CPU: changed bulk-copy path.
3. Row 6 runtime with 1,024-row batches, one CPU: necessary ordinary
   setup/allocation diagnostic control, outside row-7 acceptance.

Counts are fixed at two rounds of ten adjacent AB/BA pairs, one warmup per arm
per round: 120 measured observations and 12 warmups. The 600-second inclusive
measurement budget starts when the candidate build completes; audit, freeze,
fixtures, startup, warmup, oracles, CPU reads, lifecycle, cleanup and sealing
are included. Candidate ordinary release build is estimated separately at
6–14 minutes; the audited exact parent is reused without rebuilding.

All three arms use maintained row-6 `processing_seconds`, with matching
original processing CPU bounds. The initial selection named row 7 for the
small-batch control. Before any build/timing, source inspection established that
maintained row 7 requires separate feature-enabled dispatch-trace wheels and
`row7_releases`; ordinary releases cannot execute that arm. The original
selection is preserved as `prebuild-selection-initial.json`. The pre-execution
amendment selects the ordinary row-6 runtime at 1,024 rows per batch instead,
without overriding any timer, expanding cases or retrying an observation.
This control cannot certify row-7 settled processing or the matched-projection
overhead ratio. Initial controller validation has four passing tests. After
amendment, three unchanged ON proof guards pass, while the dependency-feature
test fails because its fixture still names row 7 (`harness-tests-amended.log`).
The corrected fixture then passes that exact test alone
(`harness-feature-fixture-fixed.log`); no native/timing attempt occurred.

No timing acquisition or candidate release build has started at this phase.
Shared-environment window test/benchmark builds have been observed and must
settle before timing acquisition. Their processes and root-worktree edits are
owned elsewhere and are preserved. No stopped/inconclusive result, ON proof,
unmeasured target or allocation/RSS limitation is replaced by this increment.
