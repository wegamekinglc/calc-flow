## PR #372 Review: Fixed-50 fragmented-output follow-up protocol

**Author:** Cheng Li | **Branch:** `feature/stream-asof-output-ranges-main` → `main`

**Scope:** Static protocol/plan/seal and exact-rank proof; no execution approval.

**Approved production:** `9983cfd0c3d096da7a60c3a96ae03b785f5ae955`

**Sealed measured checkout:** `764843e634ae1a1da7a5b010095c3017349a25f4`

**Baseline:** `eccb26973811bc476f0944b977ddedf8564b0237`

### Summary

The fixed-50 proposal is a justified follow-up to the two unresolved original
fragmented-output shapes. It preserves the common fixture, primary timer,
full Arrow oracle and lifecycle while increasing each round to 50 pairs.
The original [440-sample evidence review](stream-asof-output-ranges-performance.md)
remains Approved and separate; its inconclusive results are not retroactively
cleared or combined with the planned new observations.

### Build and Test Results

- Rust: **Not run**; no build or source change proposed.
- Python: **Passed independent stdlib static checks** of exact-binomial
  ranks, counts, unchanged shape metadata and complete-IPC forecast.
- Studio backend: **Not touched**.
- Studio frontend: **Not touched**.
- New failures: None.
- Regressions: No new measurements or performance verdict; the original
  two fragment cases remain inconclusive.

The reviewer read the complete protocol, plan, proposal seal, static-check
source and saved proof. `sha256sum` of those five small files matched their
bindings. An independent `python -` exact-integer audit returned exit 0,
without executing the author's checker, importing an engine or starting
native work. No original full hashes, source checks or benchmark cases were
repeated. No CI snapshot or remote action was taken.

### Blocking Issues

None in the static protocol. The sampling wrapper and full fixed-50 raw
verifier are explicitly unimplemented/unreviewed at this seal; their
independent approval and a fresh root quiet grant remain prerequisites to
any native import or execution. This protocol approval supplies neither.

The implementation handoff must preserve the original serialized case/round
execution: eight fresh workers is the total across two cases and two rounds,
with one resident worker per revision in the active round. It does not
authorize parallel cases/rounds or eight concurrently sampling workers.
Runtime sampling-count changes need their own sealed fingerprint alongside
the unchanged original harness/driver/worker identities.

### Style Issues

None found. The proposal clearly separates observations, forecasts,
statistical assumptions and pending execution gates. This reviewer changed
only the owned protocol review artifact.

### Test Coverage

Independent exact-integer binomial enumeration gives the tightest symmetric
interval with at least 95% nominal iid coverage for 50 pairs: one-based
ranks **18/33**, coverage **0.9671608624357315**. Each tail contains
18,486,790,962,201 of 1,125,899,906,842,624 outcomes. The next tighter
19/32 interval has coverage **0.9350913529277278**, below 95%.
Alternation, larger N and repeated measurements in a resident worker do
not establish independence or identical distributions; the proposal states
this limitation accurately.

The count contract is exact: two cases × two rounds × 50 pairs × two
revisions = **400 measured timings**, with **eight fresh workers and eight
warmups**. Only `full-fragment` and `projected-fragment` at 1M rows/64k
batches are included. Their IDs, dimensions, timing scope, payloads and
odd/even split match the original raw case metadata. No passing case,
original PID, warmup or old timing enters the follow-up sample set.

The dedicated verifier must enforce exactly 50 complete pairs per round;
the original ten-pair v2 verifier cannot certify this experiment. It must
check original loaded-native/environment/workload identities, actual
sequential UTC AB/BA order, independently saved IPC arrays, every full
oracle/status/EOF proof, fresh PID/exit and resource observations. The
maintained verdict retains threshold `5 + 1e-12`, regression before
positive uncertainty before improvement, with no implied clearance for
inconclusive results. Lightweight rank/verdict/count synthetic checks and
the sealed wrapper/verifier receive separate review before execution.

N=50 is frozen before new observations. There is no optional stopping,
extension based on intermediate results, pair/round selection, old/new CI
combination or automatic rerun. All timings are reported on a valid
completed attempt; functional/resource/process failure instead stops and
preserves the partial attempt without a performance classification.

The schedule forecast was independently reconstructed from original
complete IPC costs. Full-fragment uses original whole-case wall
40.53913050799747 seconds plus 80 additional pairs at mean complete-pair
cost 1.5711084403475981 seconds, giving 166.2278057358053 seconds.
Projected-fragment similarly gives 152.77879294700688 seconds; total is
**319.0065986828122 seconds**. Startup/warmups remain included through
the original whole-case term. This is a same-shape estimate, not an
observed new wall time or a user execution cap.

The unchanged RAM admission uses actual MemAvailable, a 70% limit,
1.25 times the resident pair's observed whole-worker high-water RSS and
zero swap. A longer lifetime is not guaranteed by the old peak; fresh raw
VmHWM/VmRSS/VmSwap and available-RAM observations must survive in every
request, with actual refusal preserved. No import/allocation floor is
subtracted. There is no need to rehash the J1 inventory for this static
handoff; fresh execution authorization still controls concurrent native
work and cache/resource ownership.

### Documentation Consistency

Reviewed bindings are:

- Protocol: `a805bd8fede39385c7ec8a816d1da3683fce84e27b1d6986c675fc9c1b6cddee`.
- Plan: `3d7d2442e45faa89ca7a0f0066f205449c0e06e8311a385ed1a7c941a2653b96`.
- Proposal seal: `57033e54197332a460cea86a83b45989bd6b80f37bdbd0bfaeb12df5235affdd`.
- Static checker: `efe26c57f12a8bb39d86e2d9884cb9564801009a65ed71748b513893af994afb`.
- Static proof: `7a1d17c1c8fca6f048baf748c2c9290832739341847e2b47532e5f2285667159`.

The originals retain report `713f5748…`, raw `8ef9aab5…` and archive
`f02bba3b…` bindings from the completed evidence review. Baseline native
`06b152ba…` and candidate native `e32eae86…`, worker `51a5f50a…`, original
driver `f597597d…` and composite harness `6593983e…` match the original
selected comparison. Historical preparation versions are not substituted.
The fresh destination and new sampling fingerprint separate the proposed
evidence from the original archive.

A no-confirmed-regression result would not erase the original slowdown,
prove equivalence or establish a material pipeline gain. Tail quantiles
remain descriptive; this follow-up adds no replay/durable recovery or
J1 FR10 active-preparation evidence. The protocol preserves those scope
limits and introduces no public or normative API change.

### Verdict

**Approve — static protocol only.** Proceed with separately sealed wrapper,
fixed-50 verifier and synthetic preparation for review. No native execution
or performance conclusion is approved by this artifact.
