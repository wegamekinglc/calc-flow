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
They are not measurements of this new main/candidate pair. Fresh comparisons
are pending, including ASOF after the earlier right-columnar change. The most
recent complete suite supplies LOW/ON measurements, without a true-OFF control.

| Workload                        | Historical main evidence         | Single-thread target         | Parallel target | This increment's evidence |
|---------------------------------|----------------------------------|------------------------------|-----------------|---------------------------|
| Sealed Join 1M                  | 103.33 / 93.10 ms, 1 / 32 CPU    | ≤60 ms, OFF                  | ≤15 ms, OFF     | Not remeasured            |
| Sealed Join 10M                 | 1094.44 / 962.35 ms, 1 / 32 CPU  | ≤600 ms, OFF                 | ≤150 ms, OFF    | Not remeasured            |
| Retained Join 4M / 1M           | 4.467×                           | ≤4.4×, OFF                   | —               | Not remeasured            |
| Join capture / restore per row  | 0.186 / 0.437 µs                 | ≤0.1 / ≤0.2 µs, enabled      | —               | Not remeasured            |
| Join handler during compaction  | 1.049× steady                    | ≤2× steady, enabled          | —               | Not remeasured            |
| ASOF 1M per input               | 243.45 / 277.23 ms, prior paired | ≤100 ms, OFF                 | ≤30 ms, OFF     | Current-main profile only |
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
correctness, resource or atomic-commit blocker; final lint, paired timings and
specialist approval remain pending. No timing gain is claimed yet.
