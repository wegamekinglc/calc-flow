# Issue 406: current-main ASOF installation profile and development

The active objective remains the complete eight-target development plan. PR
#412 merged as `8f890f28` after 38 green checks and the complete 809-result
benchmark run. That is the completed repair milestone, not acceptance of the
eight performance targets. The previous goal turn made concrete progress by
completing and merging that repair.

Work now uses `feature/issue406-asof-admission-bulk`, based on exact merged main
`8f890f287d44fa0c5ce8c79d3d23e595f8563dcd`. The original main worktree and its
unrelated changes are preserved. This increment implements the
[bucket-borrowing specification](../specs/issue406-asof-admission-bulk.md).

## Current profile

The ordinary main release was exported from the exact commit, built offline
with the locked dependencies and ordinary release optimization, retaining its
symbol table. Build time is 336.760 seconds; native SHA-256 is
`95b516587dc9633ac0af845f4046450e5894ce2861e70f3df40060f47de9dcf7`.
Its source archive, file hashes, actual compiler invocations, wheel and native
identities are under `target/issue406-asof-admission-bulk/`.

The fixed two-case acquisition uses 1M rows per ASOF input, 64k batches, true
checkpoint OFF and an unbounded default runtime pool. Both one/32-CPU workers
pass full oracles and complete their lifecycles. All ten processing windows
contain native samples. The first launch failed to save its profile because
the controller passed a relative output path into a different child working
directory. It is retained, fully settled and charged at 8.703 seconds. After
fixing that path, total charged acquisition and sealing is 20.134 / 180 seconds.
There are no surviving groups or forced kills.

| Profile region, inclusive unless stated | 1 CPU  | 32 CPU |
|-----------------------------------------|--------|--------|
| Admission                               | 52.08% | 41.13% |
| Admission preparation                   | 37.02% | 30.38% |
| Admission installation                  | 14.10% | 10.16% |
| Admission install, self                 | 7.78%  | 6.03%  |
| Unsigned bucket append, self            | 5.85%  | 3.68%  |
| Output-prefix matching                  | 29.34% | 33.35% |

Inclusive regions overlap and are not additive. Sample totals are 2,788 and
3,535, respectively. The one-CPU profiler reports 243 lost events over its
whole lifetime, whose processing-window membership is unavailable; the 32-CPU
log reports none. WSL2 host isolation remains unverified. These profiles choose
source work; neither their durations nor sample shares establish a speedup.

## Full target inventory

The following historical numbers retain their original artifacts and scopes.
They are not measurements of this new main/candidate pair. Fresh completed ASOF comparisons are below; the other seven rows are not
remeasured by this increment. Complete-suite evidence supplies LOW/ON timings
without replacing the true-OFF acceptance inventory.

| Workload                        | Historical main evidence         | Single-thread target         | Parallel target | This increment's evidence |
|---------------------------------|----------------------------------|------------------------------|-----------------|---------------------------|
| Sealed Join 1M                  | 103.33 / 93.10 ms, 1 / 32 CPU    | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not remeasured            |
| Sealed Join 10M                 | 1094.44 / 962.35 ms, 1 / 32 CPU  | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not remeasured            |
| Retained Join 4M / 1M           | 4.467×                           | ≤4.4×, OFF                   | —               | Not remeasured            |
| Join capture / restore per row  | 0.186 / 0.437 µs                 | ≤0.1 / ≤0.2 µs, enabled      | —               | Not remeasured            |
| Join handler during compaction  | 1.049× steady                    | ≤2× steady, enabled          | —               | Not remeasured            |
| ASOF 1M per input               | 243.45 / 277.23 ms, prior paired | ≤100 ms, OFF                 | ≤30 ms, OFF     | OFF 131.56 / 167.16 ms    |
| ASOF harness, 1,024-row batches | 488.66 ms over matched operator  | ≤2× matched projection floor | —               | Not remeasured            |
| Join full 1,024-row callback    | OFF 91.89; enabled 89.83 µs      | ≤50 µs                       | —               | Not remeasured            |

LOW/ON throughput goals remain ≥90%/≥75% of OFF. For ASOF's absolute target,
these imply LOW ≤111.11 / ≤33.33 ms and ON ≤133.33 / ≤40 ms at one/32 CPUs.
These are goals. Periodic impact requires actual advancing processing epochs;
configured intervals alone do not establish it. Capture/restore and compaction
retain their enabled-component timing boundaries.

## Implementation and validation

The selected change removes repeated bucket exclusivity checks from existing
ordered right admission, with a bounded safe stack selection. Larger domains
keep their existing fallback. The focused work-count regression failed before implementation: 4,097 input
rows over 64 keys caused 4,097 exclusive bucket checks instead of 64. After the
change, all 341 serial ASOF tests pass, including differential checkpoint/COW,
sparse handles, 65-key fallback and moved-slot/zero-allocation controls. The
existing test registrations remain unchanged. Source review found no
correctness, resource or atomic-commit blocker. Clippy, formatting, frozen spec
references, generated-contract and whitespace checks also pass. Final specialist approval and required CI remain merge gates.

## Main versus candidate paired results

The measured product revisions are main `8f890f287d44fa0c5ce8c79d3d23e595f8563dcd`
and candidate `098942239abfc694525791e57be622985790f59f`; later documentation
commits do not relabel either measured binary. Both are ordinary release wheels
from exact source exports, with identical compiler, build environment and
normalized lock graph. Actual compiler invocations, all 294 compiled artifact
identities per side, original-source inventories and installed wheel members
pass the maintained ordinary-release validation. Cargo only reordered the
`flatbuffers`/`fs2` dependency entries. Candidate build time is 470.664 seconds;
its native SHA-256 is
`c6e4b1ffdf92c06a1924df4dd33feba2cd38e033e6a34e77e57dca8b15ec47b1`.

The fixed selection is four ASOF cases, 1M rows per input and 64,000-row batches,
with two rounds of ten adjacent AB/BA pairs and one warmup per arm per round.
The 600-second budget includes readiness checks and waiting for an unrelated
C++ build, validation, fixtures, all oracles, terminal completion, worker
settlement and evidence sealing. That build ended before acquisition, then
another CMake/Make build began during ON. The resource guard stopped collection
on the first observed conflict. Total charged wall time is 427.487 seconds;
owned workers are settled without forced kills or surviving process groups.
There is no retry or increase to sample counts. The raw partial ON observations
remain diagnostic and receive no timing verdict.

| ASOF case  | Main p50   | Candidate p50 | Change | Round median / CI95                                | Verdict                 |
|------------|------------|---------------|--------|----------------------------------------------------|-------------------------|
| off/cpu-1  | 140.817 ms | 131.563 ms    | -6.57% | -6.83% [-8.77%, -3.56%]; -6.64% [-9.18%, +0.85%]   | no-confirmed-regression |
| off/cpu-32 | 176.486 ms | 167.164 ms    | -5.28% | -3.86% [-5.10%, -1.93%]; -6.12% [-7.65%, -3.32%]   | no-confirmed-regression |
| low/cpu-1  | 140.710 ms | 130.704 ms    | -7.11% | -9.65% [-10.89%, +6.40%]; -5.92% [-10.77%, -4.09%] | inconclusive            |
| on/cpu-1   | Incomplete | Incomplete    | —      | 29 of 40 observations; interrupted                 | No verdict              |

The three complete cases have all 120 measured and twelve warmup observations
validated with full 1M-output oracles and completed lifecycles. Both OFF cases
have no checkpoint files or advancing processing epochs. LOW is enabled at a
24-hour interval: its 40 samples exercise terminal checkpoint lifecycle, with
no advancing processing epochs, and do not measure periodic checkpoint impact.
The ON 100ms case has only 29 of the required 40 observations, each reporting
advancing processing epochs, and remains incomplete. Its final active workers
were terminated by the guard rather than treated as completed jobs. Peak
sampled controller/owned-worker RSS is 790,437,888 bytes; this is an observation,
not a runtime memory limit or continuous peak measurement.

The complete OFF medians are lower, but neither comparison meets the maintained
`improved` classification; LOW is explicitly inconclusive. These results do not
establish equivalence, a complete ON comparison, or eight-target acceptance.
OFF 131.56 / 167.16 ms still misses the 100 / 30 ms targets. WSL2 host isolation
is unverified and resource polling cannot prove the absence of all interference.
The remaining current-main profile identifies output-prefix matching at
29–33% inclusive; parallel scheduling and serial admission/output work remain
open development paths.

## Complete benchmark repair and follow-up

PR #412 is merged as `8f890f28`, with all 38 checks green. Its authorized full
repair run completed 22 shards and 809 unique result rows without missing rows,
oracle failures or confirmed regressions. The independently scheduled
[18:00 run](https://github.com/wegamekinglc/calc-flow/actions/runs/38235513830)
also completed all 26 jobs, including every one of the 22 shards. Its report has
809 unique rows, zero errors and zero confirmed regressions, retaining 149
inconclusive, nine informational slowdowns and sixteen declared untimed catalog
rows. Its run head is `e5380992`, and it must not be relabeled as a measurement
of the present candidate or as true-OFF acceptance.

The existing [complete benchmark workflow for this candidate](https://github.com/wegamekinglc/calc-flow/actions/runs/38254036436)
was dispatched with explicit main baseline `8f890f28` and measured candidate
`09894223`, retaining the full catalog and unchanged regression rule. It is
pending; no complete result is claimed for this new pair. Source review,
complete benchmark evidence and required CI must all be assessed before merge.

Raw paired evidence, resources, frozen plans, source/build provenance and hashes
remain under `target/issue406-asof-admission-bulk/`. The final execution-plan
SHA-256 is `bae947cc642cba95e061454492c52ee7602851cbc3586db1a63972370619e8f8`.
