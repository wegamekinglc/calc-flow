# PR #372 supplementary performance evidence review

**Author:** phase0_performance | **Branch:** feature/stream-asof-output-ranges-main
| **Scope:** fixed-50 fragmented-output report and saved evidence

## Summary

**Approve for evidence accuracy only.** The report faithfully records the fixed
400 observations, functional and resource attestations, statistical uncertainty
and external launch load. The environment verdict remains **INCONCLUSIVE**;
this approval provides no idle performance acceptance, overall gain claim or
full-fragment regression-risk clearance.

## Exact reviewed identities

- [Supplementary report](../analysis/stream-asof-output-ranges-fragment-followup-performance.md):
  SHA256 `c98dc372a82853ef0b918e0037336df7936d49950f2dd56bc6f7b79493f2dd11`.
- Raw `fragment-followup-attempt1/results.json`:
  `ca0215a25698d9e29f27569a9e200b8b36e42b0daabd9285c424fb01a6f8fa3c`.
- Archive `evidence-a3-fragment-followup-attempt1-v1.tar.gz`, 606,167 bytes:
  `f49611766bb741669394395e30c0bc7f66fcf9f737f815fa647480b6736f3f5f`.
- File manifest: `068a879e01f67aedc275d5fd0397708adb5dc08d157ebd72449e41d132d4cc97`.
- Tooling v2 seal: `01dd72c09662026a6675c4b7fd4aec9529dfa464a51378f2accd80d6cf2ba066`.
- Sampling fingerprint: `dd2876a21f1f3958042062c86dfb4ec6614cff0d8cf23e272d5e220aa80dae7d`.
- Independent proof `target/issue363-asof-fragment-accuracy-review/proof.json`:
  `7211004ff5b59b2aabeb126fdaa4afe516074a54a081ecec908153a56db738a5`.

The previously approved production source remains `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`;
candidate release metadata selects review-document commit
`764843e634ae1a1da7a5b010095c3017349a25f4`. Baseline remains
`eccb26973811bc476f0944b977ddedf8564b0237`. Native identities remain `06b152ba…`
and `e32eae86…`, with their full hashes matched in release metadata, all eight
worker hellos and preserved author post-run installed-file checks. No release,
source or native code was rebuilt or executed for this review.

## Build and test results

- Rust, Python product suites and Studio: **Not run**; this is a saved-evidence
  review, with the source approval unchanged.
- Independent stdlib verification: **Passed**, final command exit 0:
  `PYTHONDONTWRITEBYTECODE=1 python target/issue363-asof-fragment-accuracy-review/review.py`.
  The script and proof preserve the verification logic and outcomes. Two initial
  reviewer assertions were corrected for the metadata shape and the aggregate
  two-revision warmup scope; neither was a report or raw-data failure.
- New report/evidence failures: **None**. No new CI status snapshot was requested
  for this evidence-only handoff; required remote checks are not claimed green.

All 400 primary values were independently reconstructed from IPC and compared
with the raw arrays and embedded report arrays. Counts are two original shapes,
two rounds of 50 adjacent alternating AB/BA pairs, eight warmups and eight fresh
workers. Actual request and guard timestamps preserve order and nonoverlap;
rounds finish before the next round's requests. All workers have strict integer
exit code zero, distinct PIDs outside the original 44, and no current `/proc`
entry. Wrapper error, live-owner and cleanup-error inventories are empty.

All 408 measured/warmup full-row, full-column, canonical-order oracle
attestations, source cardinalities and terminal v3/EOF manifests were checked.
All 424 resource checks were independently recomputed from raw kB observations
using 1024-byte conversion, zero observed worker swap and the 1.25/70% gate.
Every IPC and both guard journals contribute per-PID HWM maxima and the global
minimum available RAM. All four conservative round gates pass; the maximum pair
sum is 4,966,617,088 bytes, or 6,208,271,360 bytes after the safety factor.

The complete 536-member archive contains 535 files plus its manifest. Every
member's size and SHA256 matches the manifest and corresponding saved file.
All verifier-consumed worker evidence is present, including eight IPC journals,
eight resource journals and 408 terminal manifests. The archive hash is correct.
All 36 sealed tooling inputs and the unchanged original report were verified;
large original wheels/native/archive contents were not unnecessarily rehashed.

## Statistical and environment findings

Independent binomial arithmetic gives N=50 ranks 18/33 and nominal iid coverage
0.9671608624357315. Four median paired-change intervals, descriptive p50/p95/p99,
rounded report tables and maintained `5 + 1e-12` classifications match exactly.
There is no combination with the original 440 observations or observed optional
stopping.

- Full fragmented output: p50 588.601 to 612.569 ms, +4.0719348095%; intervals
  [1.9682385076%, 4.5236473922%] and [3.3696266455%, 5.1503171490%].
  Statistical verdict: **inconclusive**.
- Projected fragmented output: p50 525.074 to 537.047 ms, +2.2801966564%;
  intervals [1.4129474442%, 2.5802619789%] and [2.0503550214%, 3.7969329457%].
  Statistical verdict: **no-confirmed-regression**, without an environment pass
  or gain claim.

The 21:07:04.153209 UTC launch snapshot records external PGID 1375610 and
C++/cc1plus PIDs 1375702/1375703, with cc1plus at 97.6% CPU. First hello was
21:07:06.208817; both first-round warmup preparations span
21:07:06.900827–21:07:11.057140; first measured request began
21:07:11.058919. The external compiler's exit timestamp is absent. Primary
timer overlap is **UNKNOWN**, and continuous idle is not attested. The report
correctly discloses this, rather than asserting contamination of a specific
sample or clearing regression risk.

Matrix wall 312.906851 seconds, wrapper wall 314.906997 seconds and caller wall
5:14.95 with exit zero match their separate journals. The pre-start deferral is
excluded; the report describes the wall scopes and forecast accurately.

## Blocking issues

None for publication of this accurate report. Environment uncertainty and
full-fragment risk remain limitations on performance acceptance; this evidence
review does not resolve them or authorize merging with unresolved required CI.

## Style issues

None identified in the report: five pipe tables align, whitespace is clean and
the file ends with a newline. Reviewer changes are confined to this document and
the independently owned target proof directory.

## Test coverage and documentation consistency

Evidence checks validate the frozen fixture's full payload/order attestations
and causal completion, not a new native run. Terminal manifests and EOF do not
prove durable restart; descriptive tails do not supply tail confidence bounds
or independence. No J1 during-active-preparation latency claim follows.

The supplementary PR body accurately preserves these limits and separates the
original and follow-up evidence. It retains draft status, environment
uncertainty, full-fragment risk and unresolved remote CI/coverage. The original
report and prior source/protocol/tooling reviews remain unchanged.

## Verdict

**Approve — accuracy evidence only**, bound to the report, raw, archive and
tooling identities above. **Environment: INCONCLUSIVE.** There is no idle
performance acceptance, confirmed gain or full-fragment regression clearance.
No new measurement or remote operation was performed.

Reviewer-owned native, build and background processes: **0**.
