# Branch review: weekly implementation fixes

Review baseline: `1b91bcc00a0623d5ece37ac654bf6e2a4f045939`.
Reviewed product commit: `7dbd229d23fb4060c1209f518ef08bee8c82c508`;
tree `93ac3f2ea6b3090b9e1627452fe6255b78912d83`.
The initial `240611ff` candidate was withdrawn after peer review found a
further benchmark-only partition-template defect. No performance observations
were acquired from that withdrawn candidate.
Development branch: `fix/weekly-review-20261107`.
The [weekly scope](../specs/weekly-review-20261107.md) covers 36 mainline
commits since 2026-10-31 00:00 Asia/Shanghai. This independent review evaluates
the resulting remediation diff and its directly affected callers. It is not a
claim that every unchanged function in the repository has been audited.

The source review is complete. Focused native checks and exact candidate-wheel
validation pass. Performance acquisition was stopped before any observation
because another task was compiling native release tests on the same host.
No PR/CI result is available at this artifact checkpoint. This is not a
merge-ready verdict.

## Summary

The current repairs address concrete failures rather than speculative
refactoring:

- Native SQL admission now rejects nested CTEs and table alias column lists
  that its direct Arrow execution does not interpret. The fallback retains
  DataFusion's logical bindings. Differential result tests exercise both
  native join shapes, CTE shadowing and base/derived column rebinding.
- Native join planning uses an empty per-query rolling audit instead of the
  preceding DataFusion query's shared audit. The ordinary planner still owns
  its audit. The duplicate equality matcher is replaced by an existing helper
  with equivalent operand matching.
- Compact ASOF output charges the source stored in its compact position
  prefix. A recent registration is not proof that the prefix uses that source.
  The change preserves batched accounting and avoids a new per-row loop.
- Python sequence normalization retains zero-column row counts and supplied
  empty RecordBatch boundaries, using the existing native Arrow source. The
  normal nonempty-column concatenation path is retained. Sequence items are
  captured when the result owner is constructed; caller lists and Arrow
  buffers are not mutated.
- Validation errors reconstruct title and structured records for copy/pickle
  while preserving exception dictionary state, including notes and custom
  diagnostic attributes. This repair does not affect calculation paths.
- Benchmark cleanup attempts every worker after a successful round even if
  an earlier close fails. The first failure remains primary and later failures
  are notes; an existing measurement failure is not replaced.
- Experimental multi-partition finite execution bypasses both template
  lookup and capture. This handles both a changed requested partition count
  and a changed input batch layout under the same P4 setting. Default P1
  template reuse remains intact, including after switching back from P4.

The second source revision addresses the peer-reviewed changing-input case.
Only the finite-plan runtime sets experimental input partitions, so the
separate streaming expression cache is outside that setting. No additional
source blocker is established in the current repairs.

## Build and test results

Reviewed the actual failing logs rather than relying only on test names:

- Five SQL regressions fail against unchanged production source: two CTE
  shadowing results, two alias-column results and one stale rolling audit.
- Two ASOF regressions fail for differing source validity-buffer charges;
  the pre-existing focused control passes. The tests also check reservation
  release and compare compact accounting with the existing range accounting.
- The rebuilt native join module passes all 28 tests and the ASOF output-plan
  module passes all 15 tests. Both run the binary sealed as
  `5e95a8a1be19cc42c57a3dc3d445d94421f13efc45fb8d1579e0aa7a1f0393e4`.
- The warmed input-partition test fails against the unchanged setter:
  the P1 plan still reports one output partition after a P4 request. The
  initial repair passed six partition tests, but a subsequent actual RED
  showed that fixed P4 still reports P1 after moving from one input batch to
  eight. The second source revision bypasses template lookup and capture;
  its complete seven-test partition GREEN passes in 0.06 seconds. The final
  binary hash is
  `5a8e98e6d7e1ab0b7a064fb3eab20e5260b0a9699607c711f5ecbe45b62ecf2e`.
- Four Python stream regressions fail before the fixes and pass afterward;
  33 existing materialized-input controls also pass. These early checks use
  the explicitly documented adjacent #532 native module, not the exact new
  candidate wheel.
- Both complete changed Python test modules subsequently pass against the
  exact candidate release wheel: **93 passed in 1.32 seconds**. The log
  verifies package/native resolution under the isolated wheel site, with
  native SHA-256
  `97955f68501bdcdcd31f9d96e7f46ed6240e8e1d8bda8377f07b9017a54d75fe`.
  This supersedes the early adjacent-native qualification for those modules.
- Three exception round-trip tests fail before reconstruction support. A
  subsequent genuine RED detects omitted exception dictionary state; the
  final three tests pass with that state retained.
- The benchmark cleanup regression fails before the fix, and the focused
  eight-test module passes afterward.
- `cargo clippy --locked -p calc-flow --features benchmark-internals --lib
  -- -D warnings` passes. The final native receipt's eight source-file hashes
  match the reviewed files. SQL/ASOF passing filters were retained from the
  earlier rebuilt binary; their unchanged passing paths were not rerun solely
  to attach the later partition-fix binary hash.
- `cargo fmt --all --check`, focused Ruff checks, generated-schema comparison
  and `git diff --check` pass. The exact baseline and candidate release builds
  finish in about 1,127 and 1,029 seconds respectively, separately from the
  acquisition budget.

Performance remains unmeasured and required CI remains unresolved. No full
local regression or coverage matrix was run by this reviewer. Adjacent-native results do not
substitute for candidate verification; skipped coverage is not proof of the
workspace 90% floor.

The [evidence archive](../analysis/weekly-review-20261107-evidence.tar.gz)
contains 64 files, independently verified against its
[per-file index](../analysis/weekly-review-20261107-evidence-index.json).
Archive SHA-256:
`74aaccd7c952e0f89a4db0ce82e3944b7768b070e616602f524a0e4e12b86e6f`.

## Test coverage

The native differential tests cover CTE shadowing, base/derived column alias
rebinding and rolling-audit isolation. ASOF controls compare compact versus
legacy charges across nullable source changes, promotion, unused registration,
pool release and allocation/cancellation paths. Partition tests cover both
changing the setting and changing input batches without changing P4, while
retaining the warmed P1 control. Python regressions cover zero-column and
empty records, both stream-owner entry points, and exception round trips.

The complete Rust/Python/connector/platform and combined 90% line-coverage
gates remain CI responsibilities; the local focused results do not certify
these unrun suites.

## Performance status and limits

**Unmeasured; the no-regression requirement is not satisfied.** The readiness
receipt records a competing unowned `calc-flow-mt-impl` release-test build
finishing one compilation and starting another. No attempt was made to
terminate that task. A brief CPU quiet reading would not establish freedom
from cache, memory and I/O contention, so the repository's unsuitable-host
stop rule was applied before acquisition.

There are **zero observations, zero warmups and zero timed samples**. No
acquisition directory was created and no scientific retry was performed.
The frozen protocol SHA-256 is
`bb38cff0ec64c3c1b7e4b600eec5a8c01b6bb156869d46fa488126e04196cb52`;
this reviewer verified all 102 declared helper/harness source seals. Those
preparation checks do not establish runtime performance.

The four controls prepared for a later independent comparison are canonical
finite ASOF and interval SQL at
1,000 rows, fixed-width projected stream ASOF at 10,000 rows with 1,024-row
batches, and the public materialized-list Program lifecycle at 1,000 rows.
The stream ASOF projection is pushed into the operator, so it covers the
fixed-width compact accounting path despite the retained input's string key.
Nullable-source differences remain correctness-test coverage.

The frozen maintained release-pair protocol calls for two rounds of ten alternating
pairs. Every observation owns a fresh process, one warmup and three timed
samples reduced to their median: 160 planned observations, 160 planned warmups
and 480 planned timed samples over four cases. Fixture preparation,
validation, processes
and cleanup belong to one 600-second acquisition budget. Build cost is
separate. Frozen helper, product, machine, dependency and workload identities
must match their declared contracts before classifying timing.

The two-round paired 5% rule applies unchanged. An inconclusive result is
neither a confirmed regression nor proof of equivalence. Canonical controls
cannot establish an all-query no-slowdown guarantee: CTE and alias queries
that previously returned incorrect data now use a correct fallback, so their
old timing is not an equal-semantics baseline. No speedup, equivalence or
no-regression claim is issued from this unmeasured campaign. Experimental
input partitions above
one now replan on each execution; these four controls do not measure that
benchmark-only mode. Future P1/P4 comparisons must separate planning cost
from execution rather than treating an old cached P1 plan as valid P4 work.

## Remaining optimization priorities

The scoped [SQL](weekly-review-20261107-sql.md),
[SUM/stream](weekly-review-20261107-sum-stream.md) and
[Python](weekly-review-20261107-python.md) reviews separate source hypotheses
from measured results. The following ordering is reasonable for follow-up:

1. Cache immutable SUM key-output charges without changing conservative
   admission, rollback or checkpoint semantics. Profile that subphase before
   estimating benefit: historical transaction finishing accounts for about
   20% of selected samples, but it contains work beyond key charging. Check
   sparse large state, dense input and exact/one-byte-short budget controls.
2. Build native SQL interval outputs directly with typed Arrow builders.
   `BandState::next_batch` currently stages three vectors of optional values
   before Arrow 59.2.0 traverses them into value and validity buffers. Removing
   that output-sized staging is a stronger source hypothesis than setup-only
   cursor savings, but it has no measured speedup yet. Check null propagation,
   order, duplicates and chunk boundaries; compare dense/high-output,
   non-dense, small and sparse/short-final-batch controls. A builder reserved
   at full batch size can retain excess capacity in its returned Arrow
   buffers, whereas the current conversion knows the actual output length.
   Measure retained buffers and RSS as well as time and allocation count.
3. Represent interval output views with borrowed ranges and explicit bounds,
   eliminating repeated scans/copies across output chunks. Include nested
   slicing semantics before adding new callers: the recorded private
   double-slice defect currently has no identified production caller. Guard
   output order, multiplicity, cancellation, allocation and fragmentation.

Avoiding unused dense-band cursor allocations remains a narrower setup-only
candidate. Preserve irregular fallback indexing and compare mixed/dense,
non-dense and small inputs before adoption. Contiguous ASOF slice reuse,
early projection and native sequence preparation also remain secondary
candidates because buffer retention, budget accounting and
boundary ownership can offset their local allocation savings. The prior
ranked SUM, column-dictionary and CASE studies do not establish an accepted
end-to-end gain and should not be reinstated from component evidence alone.

Every future proposal needs its own narrow hypothesis, exact baseline,
fixed controls and acceptance/withdrawal rule. No unimplemented proposal is
certified regression-free by this review.

## Documentation consistency

The stream guides distinguish supplied RecordBatch boundaries from the
records actually exported by an Arrow Table, and describe call-time sequence
capture. Historical experiment evidence remains unchanged. The changelog
describes the actual corrections without asserting a measured speedup.
The fixed-control table is aligned. Experimental multi-partition replanning
and its effect on future benchmark interpretation are documented in the
weekly specification.

## Blocking issues

No unresolved source blocker was found in the final repairs. The user's
performance acceptance requirement remains unmet because no comparison ran.
Required final CI also remains unresolved. Both gates must be addressed before
merge; a later green CI result alone cannot waive the performance gate.

## Style issues

No unresolved convention violation was found in the reviewed diff. Scoped
Clippy, format/lint and whitespace checks pass.

## Verdict

**Comment Only.** The final source repairs have no unresolved code finding,
and focused Rust/Python checks pass. Performance is unmeasured, so this review
does not approve merging or certify benchmark non-regression. Keep delivery
as a draft until a suitable comparison and required green final-head CI are
available. The [main report](../analysis/weekly-review-20261107.md) records the
implementation, fixed comparison protocol and follow-up proposals together.
