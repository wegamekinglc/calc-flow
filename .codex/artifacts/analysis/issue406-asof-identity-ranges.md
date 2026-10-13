# Issue 406: typed identity workspace ranges

## Change and scope

The complete issue 406 plan remains active, including its original eight rows,
later planning targets and phase requirements. This increment replaces scalar
Arrow type dispatch during identity workspace estimation with one typed offsets
slice per accepted Utf8/LargeUtf8 range. Adjacent offsets still supply each
physical value's length, including a null entry's underlying stored span.
Every value retains its original fixed bytes, 64-byte rounding, 33/32 string
framing and retained-sequence copy charge. Fixed-width estimates, reservations,
late-row selection, cancellation boundaries and durable formats are preserved.

The branch parent is reviewed `88922e13`, with product source equivalent to
ordinary release `05c13fd9`. PR #421 is separately merged as `1606e516`; its
report correction keeps the original during-compaction target unmeasured.
This branch's comparator is its exact ordinary parent, rather than current
merged main. Earlier main/parent comparisons and both stopped ON attempts
retain their original identities and verdicts.

## Diagnostic evidence

`target/issue406-asof-admission-profile-20261014/` contains the single fixed
diagnostic attempt. Product `05c13fd983dd1ead625eef2278eb7127dac0c61c`, native
SHA-256 `cbc1a85acd9d3b1d6baf0ec2e5fe0c888a5ec53353ee2176d1ae56281a6c600b`,
uses one ASOF OFF/32 case with 1M rows per input and 64k messages. The frozen
plan SHA-256 is
`4b89acf1b4d7d5062880896e842dd743c570eb0381df53ece3485afbfc618766`.

One warmup plus five samples finish within **10.381440611 / 180 seconds**,
including preflight, fixtures, oracles, symbolization, analysis, owned cleanup
and seals. All six full-output oracles, lifecycles, cleanups and OFF zero-file
checks pass. All groups settle without forced kills or survivors. Owner and
reviewer independently verify **1,654 input hashes and 23 result seals**.
Five original processing windows contain **549/574/553/536/561 samples**,
totaling **2,773**, with **2,522 native stack samples**.

Identity range estimation has 6.6354% inclusive samples; its self share is
2.7047%, with 3.1374% self in scalar `variable_length`. Shares overlap. These
are diagnostic observations, with no paired timing or causal speedup claim.
Host isolation on WSL2 remains unverified. `launch-failure.json` preserves a
missing-PYTHONPATH import failure before plan freeze or acquisition; fixing
the launch environment starts the sole fixed acquisition. No sample retries,
removals, count changes or expanded workloads occur.

## Focused verification

- RED: `identity-red.log` records the existing range-parity test failing after
  **4,097 scalar length dispatches for 4,096 rows and two columns**.
- GREEN: `workspace-green.log` records all **ten workspace tests** passing in
  0.04 seconds. The regression now bounds scalar dispatches by column count.
  Its new helper compares every subrange with the unchanged historical scalar
  charge oracle, including Utf8/LargeUtf8, sliced and empty arrays, nonempty
  underlying null spans, lengths around rounding/block boundaries and string
  sequence copies. Synthetic overflow preserves the failing node and exact
  column-workspace error text.
- Admission: `admission-tests.log` records **59 passes and four infrastructure
  failures**. All four fail with `bounded process infrastructure exhausted`;
  the existing process service caps infrastructure at 32 MiB. The four exact
  tests then run in separate serial test processes, passing in
  0.03/0.04/0.01/0.01 seconds (`admission-isolated.log`). The unchanged 59 passes
  are not repeated. These checks cover state/workspace thresholds, caller
  immutability, reservation refunds, abandoned workers and cold recovery.
- Complexity: `complexity-fixed.log` recognizes all twelve changed/new
  functions with Lizard 1.24.1; the maximum observed CCN is five, below eight.
  The initial command incorrectly assumes the target-installed tool is a
  virtualenv; its failed launch is retained in `complexity.log`. Supplying its
  package directory on PYTHONPATH runs the tool successfully. Unchanged
  functions are outside this complexity certification.
- Formatting, schema-drift and whitespace checks pass. Scoped all-feature
  core/library/test Clippy passes in 2m 49s (`clippy.log`). Final specialist
  source/evidence review is **Approve**, with no findings. The later paired
  attempt below is complete; full-phase allocation/RSS acceptance remains
  incomplete.

The existing admission allocation controls pass and retain their raw counts,
including 95 caller allocations and 26 actual chunk-worker allocations for
64k rows. They compare existing fast/legacy paths inside this candidate, rather
than this increment against its parent, and do not certify complete phase-wide
allocation or peak-RSS acceptance.

## Fixed paired comparison

`target/issue406-identity-ranges-paired-20261014/prebuild-selection.json` fixes
two directly affected ASOF OFF cases: 1M rows per input, 64k messages, at 32
and one CPU. Both execute the changed workspace path. Exact ordinary parent
`05c13fd9` and its audited binary are reused; the exact committed candidate is
built separately, estimated at 6–14 minutes. Two rounds of ten adjacent AB/BA
pairs and one warmup per arm per round share a 600-second inclusive budget
starting at candidate build completion. Preflight, ordinary artifact audit,
fixtures, CPU reads, oracles, lifecycle, cleanup and seals consume that budget.

No retries, favorable filtering, larger sample counts or added cases are
permitted. Full CPU and observed resource evidence remain alongside paired
wall statistics. This selection cannot establish ON qualification, complete
phase allocation/RSS evidence, or complete issue acceptance.

## Completed paired evidence

The sole acquisition completes for exact parent product
`05c13fd983dd1ead625eef2278eb7127dac0c61c` and candidate
`8d3f6933c53f0dab1a562c79ad355cd9e08c665f`. Candidate native SHA-256 is
`3664ebab40c619a0c4e7b8a9cfe93a684ab2da24e5c3ca52f8cd88c831041d08`.
Both ordinary releases have matching complete inventories of **294 compiled
artifacts**, features/profiles, normalized dependency locks, interpreter,
toolchain and environment. Candidate build takes **380.894939 seconds**,
separate from acquisition. The frozen execution plan SHA-256 is
`cdee70dddc4b70fe92d456886a18e6742945a0641d70bf97c9a0e365b66e7859`.

All **80 measured observations and eight warmups** pass their full million-row
oracle, lifecycle, owned cleanup and OFF zero-file checks. The owner verifies
all **3,118 frozen input hashes and 198 result hashes**, recomputes both
statistics objects exactly, and checks actual adjacent AB/BA order. Additional
provenance checks verify all 88 raw/normalized processing bounds, CPU evidence,
product revisions and affinities, plus all eight worker native identities.
Inclusive acquisition and sealing take **40.763058131 / 600 seconds**. All
owned groups settle without forced kills, survivors, pending creations or
cleanup errors. No observations are retried, removed or added.

| CPUs | Parent p50, ms | Candidate p50, ms | Point change | Maintained verdict      |
|------|----------------|-------------------|--------------|-------------------------|
| 32   | 159.888209     | 148.191731        | -7.3154%     | no-confirmed-regression |
| 1    | 130.997559     | 120.456423        | -8.0468%     | no-confirmed-regression |

For 32 CPUs, round paired medians are -5.760743% and -7.830057%, with intervals
[-8.798251%, -4.210052%] and [-10.781846%, -5.764932%]. For one CPU, round
medians are -7.653816% and -5.916994%, with intervals
[-9.991739%, -3.347668%] and [-8.486696%, -4.021210%]. Each ten-pair
order-statistic interval has **97.8515625% coverage**. The generated report now
labels its comparator `Exact parent p50` and uses the actual interval coverage.
The maintained rule does not classify either case as improved. Favorable point
estimates do not certify a speedup, equivalence or complete absence of regression.

Processing CPU medians at 32 CPUs are **187.59475 → 175.1312 ms**, with busiest
observed thread **59.18415 → 56.7925 ms**. At one CPU they are
**130.7536 → 120.22965 ms**, with busiest observed thread
**119.0725 → 109.23445 ms**. CPU hooks retain the maintained processing clock
boundaries. Counter-read boundary uncertainty and retired-thread visibility
remain; these diagnostics cannot reconstruct exact thread accounting or
replace paired wall statistics. Whole-job CPU remains a separate raw scope.

All 32 resource snapshots record no competing controlled work. Maximum observed
aggregate RSS is **839,221,248 bytes**, including the controller and both
workers; the largest observed single worker is **377,720,832 bytes**. These
sampled acquisition maxima are not exact per-arm or per-phase peaks. Complete
phase allocation and peak-RSS acceptance remains unverified, and host isolation
on WSL2 remains unverified.

Candidate p50 values still exceed the original OFF one/32-CPU targets of
100 / 30 ms. This increment measures neither the other seven rows nor ON
checkpoint qualification. Earlier comparisons and stopped acquisitions retain
their original conclusions. Final specialist review of this result addendum
is **Approve**, with no findings; required CI and workspace coverage remain
uncertified. The reviewer independently reproduces the input/result hashes,
ordinary release audits, raw identities, pair order, statistics, CPU medians and
resource maxima above.
