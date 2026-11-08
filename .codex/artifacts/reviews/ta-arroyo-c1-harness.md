## C1 Minimal Harness Source Review

**SourceBuildClear — source/harness may be frozen and built.** No source
must-fix was found. TARGET controller, actual build freshness, formal
performance, final-head CI and final specialist approval remain pending.

Reviewed HEAD `5e08d7144af63f529a54d82d27550342af297944` plus the five
uncommitted files sealed by `target/ta-arroyo-c1-performance/harness-handoff.json`.
All actual size/SHA entries match. Core SQL/cache bytes remain those previously
reviewed at 8d377; they were not re-reviewed. Only this new review artifact was
written; no implementation, protocol, root user edit or other review was changed.
No native execution, build, probe, measurement or remote operation occurred.

### Scope and correctness

The opt-in bench mode is separate from the unchanged default seven-case/v3
report and scheduled minimum20 validator. Without benchmark-internals it fails
closed. Existing default output serialization is merely extracted into
`write_report`; no scheduled case/sample semantics change.

One fresh process warms10,000 groups, performs twenty business appends and
contributes **one** primary observation: their summed process_data intervals,
including the original collector receipt and native commit. The held original
warm output is revalidated after every append. This is a native operator/
collector measurement, not Python/channel/full-job timing. The unmodified
release_pairs contract is two rounds×ten AB/BA pairs per case, no extra warmup:
two cases therefore require80 fresh processes, not nested twenty repetitions.

Sparse/dense slots, null key, exact schema/nullability/default metadata,
ordered typed row hashes and independent Python model agree statically.
Native validates every full10,000-row output in [8192,1808] chunks. Warm input
is hashed before consumption; retained primary input clones are hashed after
processing. Recovery input hashes cover empty and slot0/value1. The protocol
claims held warm OUTPUT immutability, not post-consumption warm/recovery INPUT
alias witnesses. No extra alias or case is needed for that stated scope.

The checkpoint uses existing capture/digest code; native checks layout and
accounting3, group-state3 columns/10,000 rows and30,000 logical input rows.
Cold restore emits an unchanged snapshot, then the slot0 continuation.
Counts are30,001 input/230,000 emitted output rows/23 snapshots across all
phases, plus200,000 held-warm revalidation rows. Timed throughput uses only
20,000 appended input rows. Recovery/validation are outside primary and inside
the outer phase. The checkpoint digest stays in workload identity and must
remain comparable across arms/invocations.

Resource reporting is honest: default known-Infinite ENGINE construction only,
physical engine/cache reservations unmeasured. PhaseClock ends before JSON;
external process startup/report writes/parse/seals/cleanup still require the
same outer600s budget. Current-thread Tokio, observed affinity/thread environment
and ordinary collector are not represented as a whole-job performance result.
No static Rust import/type/fixture mismatch was found; compilation has not run.

### Verification evidence

Saved focused evidence was read: initial unsupported-contract RED, then five
pure tests with eighteen mutation subcases passing; saved Ruff check passed.
No passing pure/native test was repeated. Root dirty release_pairs files were
not imported or modified during this review.

Necessary additional checks were purely static: system Lizard reports modified
main CC6 and all new support functions at most3; no changed Rust function
exceeds8. The maintained complexity ratchet actually passes using pinned
Ruff0.16 and a separate C1 TARGET cache. An earlier unavailable-lizard import
in the Ruff-only venv is not represented as a passing check. New protocol
pipe tables/newlines/whitespace and all five frozen seals match; no complexity
waiver, schema drift or old-validator relaxation was introduced.

### Required controller and build boundaries

- Import and seal release_pairs, its statistics and complete helper union from
  the frozen C1 source. Root's release_pairs/test files have unrelated user
  edits; CWD/sys.path or cached modules must not silently select those bytes.
- Bind requested case to report.selection and expected descriptor. The narrow
  extractor validly accepts either mode; the controller must reject a sparse/
  dense swap, retain timed input/output counts and checkpoint identity, and add
  real machine/dependency identity without dropping source/binary provenance.
- Enforce80 actual fresh PID/OS-exit-zero/reaped observations, strict normal-return
  deadline checks, sampled RSS and complete owned-tree cleanup. No extra native
  preflight, retry, partial success, forced-cleanup acceptance or budget reset.
  Setup/imports/oracles, all reports/statistics and final writes/seals belong
  to one600s phase with530s acquire/70s cleanup reserve.
- Freeze actual baseline a0b89ce4 plus identical driver and actual integrated
  candidate, then verify real unique fresh=false core+exact bench artifacts,
  flags/environment, source/lock/executable/common-helper seals. None is already
  proved by source review or a module-missing controller-test RED.

**Disposition:** freeze/publish and begin the separately authorized real builds.
TARGET controller behavior/mocks/review and actual quiet/resource admission
must clear before the sole fixed acquisition. Measured improvement, required
CI and final review remain unfulfilled merge gates; this is no gain or merge
approval. No additional framework, workload or test rerun is requested.
