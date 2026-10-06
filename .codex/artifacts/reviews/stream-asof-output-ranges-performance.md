## PR #372 Review: ASOF output-range performance evidence

**Author:** Cheng Li | **Branch:** `feature/stream-asof-output-ranges-main` → `main`

**Scope:** Final performance report, measurement protocol and preserved evidence.

**Approved production:** `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`

**Production tree:** `78065d9fe80635e3f6d64831215acacd9ad3e572`

**Measured checkout:** `764843e634ae1a1da7a5b010095c3017349a25f4`

**Measured tree:** `5b81c4caf3ca0503eac6541d805c23ad97543719`

**Baseline:** `eccb26973811bc476f0944b977ddedf8564b0237`

The [source review](stream-asof-output-ranges.md) remains Approved. The
measured checkout adds only that review document to the approved source;
its production inputs are unchanged.

### Summary

The [final report](../analysis/stream-asof-output-ranges-performance.md)
accurately records all original eleven cases: six have
no-confirmed-regression and five are inconclusive. Independent recomputation
confirms that no case establishes a greater-than-5% improvement or regression.
The report appropriately preserves the unresolved positive direction of
fragmented output and makes no overall equivalence or material pipeline-gain
claim.

Contiguous 1M full/projected p50 changed −2.58%/−2.03%; fragmented
full/projected changed +3.90%/+2.49%. These describe elapsed-time medians,
not throughput changes. The fragmented intervals crossing +5% cannot clear
regression.

### Build and Test Results

- Rust: **Not rerun by reviewer**; unchanged source checks belong to the
  approved source review.
- Python: **Passed independent stdlib evidence checks**. No engine import,
  native execution, benchmark rerun or shared-cache access occurred.
- Studio backend: **Not touched**.
- Studio frontend: **Not touched**.
- New failures: None in the final evidence. The original classification
  defect is resolved by the sealed v3 postprocessor.
- Regressions: None confirmed above the maintained threshold; five cases
  remain inconclusive. This is not an overall regression-clearance result.

The reviewer read the complete 1,457-line report, protocols, original
measurement scripts, v3 postprocessor, independent checker and saved proofs.
Three independent `python -` stdlib audits returned exit 0: complete timing
and lifecycle recomputation; archive/wheel sampling plus original preflight
cost recomputation; and identity/evidence-table verification. These audits
did not execute the author's checker implementation. The mathematical CI
oracle directly used the second and ninth ordered changes for each ten-pair
round, with coverage `1 - 22/1024 = 0.978515625`.

Reviewer `git diff --stat 9983cfd0 764843e6` confirmed the sole source-seal
increment is the prior review document. Small-file `sha256sum` checks and
stdlib archive/member hashing matched the bindings below. The maintained
harness's composite identity over 45 paths independently hashes to the
reported value.

No new CI snapshot or remote action was taken for this evidence-only
handoff. Required CI, cross-platform and coverage gates are separate from
these results; this review makes no green required-check claim.

### Blocking Issues

None in report accuracy, sampling scope or evidence binding.

The original `check.py::summarize` would incorrectly clear a mixed
`[6, 9]` / `[-3, 2]` result. The separate `checker_verdict_v3.py` preserves
the original v2 label and intervals while applying the maintained order:
all lower bounds above `5 + 1e-12` → regression; otherwise any upper bound
above it → inconclusive; otherwise all upper bounds below its negative →
improved; otherwise no-confirmed-regression. Eleven saved stdlib synthetic
cases agree with the separately extracted maintained function, including
mixed rounds and epsilon endpoints. The original v2 script, fixture,
timers and raw remain unchanged.

The final v2 summary is cryptographically bound as v3's input; every CI
was independently recomputed from the original samples and matches both
summaries. The v3 correction does not replace full oracle/resource checks
or turn an inconclusive result into a passing regression gate.

### Style Issues

None found. All five report tables have aligned pipe positions; timing,
tail and memory cells match independent rounded recomputation. Fences,
trailing whitespace and final newline are valid. Only this owned review
artifact was changed by the reviewer.

### Test Coverage

The primary timer retains its ready-empty-source-to-concatenated-output
scope: enqueue/backpressure, matching/planning/gather, sink delivery and
Arrow concatenation. Construction, compilation, startup, full equality,
EOF and cleanup remain outside that timer and inside complete worker wall
observations. The prior static fixture review establishes that its oracle
checks every output column and exact order; this final review validates
the saved execution attestations without rerunning native output.

Independent checks covered:

- The unchanged original eleven-case inventory and all 440 embedded timing
  values, excluding warmups, preflight and the resource bridge. Both rounds
  have ten aligned pairs; actual UTC requests follow alternating AB/BA and
  each response precedes the next measured request.
- All 44 distinct workers, hello/native/environment identities, exit 0,
  finish responses and absent recorded PIDs; all 44 warmups; 396 ASOF
  full-payload/order attestations and 88 full-vector control attestations.
- All 484 final v3 manifests with ended sources. ASOF observations have
  natural completion, task/error/pending/state zero, exact emitted rows,
  expected 1,024/64,000 ingress sizes, fragment halves and layout/accounting
  version 10. Workspace metadata remains 256 bytes/row plus 16 KiB.
- Every IPC resource record's original kB-to-1024-byte conversion, zero
  swap, cumulative worker high-water RSS and each round's 1.25 sum below
  70% of minimum observed available RAM. Maximum pair sum is
  3,584,421,888 bytes; with the multiplier it is 4,480,527,360 bytes.
- The original 26 preflight workers and 52 oracle attestations. Summing
  same-shape full-worker/sample costs for the original eleven cases gives
  278.2178146544611 seconds; actual matrix wall is 237.859102 seconds.

The original 91,144,704,000-byte 10k-to-1M estimate refusal remains
preserved. The independent 100k bridge predicts 12,557,107,200 bytes below
its unchanged threshold, followed by actual preflight of all original 1M
shapes. Its two extra cases do not enter timing classification or the
eleven-case wall estimate. No import-floor subtraction or budget relaxation
was used.

Twenty samples per revision support descriptive p95/p99, not a reliable
tail bound. Pair-order alternation does not prove independence on WSL2.
RSS observations do not prove paid-credit retirement. Final manifests
establish EOF/terminal state; this fixture has no replay or durable restart.
J1 raw-V1 restoration and FR10's during-active-preparation ≤2× gate are
separate: these A3 measurements establish neither.

### Documentation Consistency

The final report's seven identity rows and eleven evidence rows match
release metadata and independent hashes. The actual archived wheels contain
native `06b152ba…` for ECCB and `e32eae86…` for the candidate, matching every
worker hello. Historical `22fcb`/`188d` metadata is retained as preparation
history and is excluded from the measured comparison. Matching locks,
toolchain/settings and worker package/thread identities are preserved;
raw Cargo encounter order is not used to infer a dependency mismatch.

The reviewer independently checked the archive digest and size, all 1,492
regular-file headers (1,491 evidence files plus their manifest), the full
embedded manifest, twenty selected member hashes, and both exact wheels
and zipped native members. Samples include raw/summary/proof, fragmented
IPC and final manifest, a control manifest, preflight costs, both source
archives, build seals and statistical/fixture code. The author's separate
saved integrity proof additionally verifies every archive member hash;
the reviewer did not duplicate that entire 1,491-member content audit.
Mutable Cargo outputs and the later report are correctly excluded.

Final bindings are:

- Report, 54,539 bytes: `713f574867a50f60f459610a8f56a49f22021ee0adc564b8b4cbf862e862fad5`.
- Raw `matrix-attempt1/results.json`: `8ef9aab5ab7544655d91b63619a449d0274dbf39f610e23e025df8b711e728fe`.
- Verified v2 summary: `eda777ca9766462e6a24c7e186d34cc38e00acb5142361e9c0a6869231d94fde`.
- Verified v3 summary: `3b8123fe5060fc94472f41b7fce58e7246e4887cdb3a53cb84368c8b67654807`.
- Independent proof: `c7deb31b33f29047be92a365b47a3b72e6779937846f45e20e48fdf60674b1d0`.
- Archive, 109,885,045 bytes: `f02bba3b7bd3c759b08110e3db3505482da0b9f032db39ad9674184af40d5010`.
- Archive file manifest: `d8db6f7af958eeb3761f035212e8b58f37a68505cc386ed7efb9140b218bdbc9`.
- v3 checker: `00b6b2f91b5b879c127bce25ff0a4af73035ed8da652a64015adf0d58529dfec`.

The report embeds every measured second for independent recomputation and
accurately identifies the archive's current local publication boundary.
No further normative documentation change is required by this evidence.

### Verdict

**Approve — evidence report.** The report is accurate, traceable and
appropriately limited. This approval preserves five inconclusive cases;
it does not approve a greater-than-5% gain, clear fragmented-output
regression, establish durable recovery or declare the PR merge-ready.
