# C1 acquisition controller source review

**Scope:** TARGET wrapper and pure-mock evidence; no native execution or merge approval.

**Frozen controller:** `7e6360585dde8edcf2abe339b0625f8c3689fd09556abf279af2c85ad713a8b9`.
**Frozen tests:** `48d4ba7839291c940d03f773dcfd6c0458a871fcaef4cb0c8c26ea82d6161f81`.
**Frozen config:** `2c04760068b7fb75f599b7aa85e709b08566374f7bcb11afcfb82944f129cc42`.

## Summary

The controller preserves the reviewed two-case protocol, one statistical sample
per native invocation, two rounds of ten AB/BA pairs, and eighty distinct physical
workers. Native PID and requested selection are checked against the strict report;
machine/dependency augmentation preserves workload and checkpoint provenance.
Three outer ownership/error-boundary issues must be fixed before acquisition.

## Build and test results

- Read the full controller, tests, frozen config, pinned ownership/quiet helpers,
  and the directly used identity, process, statistics and provenance functions.
- Read saved evidence: thirteen pure tests passed, the updated valid-JSON orphan
  test passed, and current Ruff passed. These are mock results, not native runs.
- Independently verified both actual terminal build receipts and original Cargo
  messages: one core and one benchmark artifact per arm, both `fresh=false`,
  correct source package paths, release optimization and benchmark feature.
- Rehashed all 1,249 baseline and 1,253 candidate source seals, confirmed clean
  checkout revisions/trees and identical before/after snapshots, and verified
  original artifact, copied binary, compiler log and build log hashes.
- Toolchain, lock, flags except owned target directory, five common harness
  files, and compiled dependency identity agree. Dependency fingerprint is
  `1d4585499faf5e561c22f05b6a3d4fcdf7adc49c423cb74495b48f9c2374bc79`.
- Baseline receipt `d6903f8d72294563af3382f84196eef1e6a0a75a5a2655aad5dd82cef687d193`;
  candidate receipt `ba85d3a0d5a70b7a0c66ae6718d0a32cc74921ba0c8aedf2cfeb74da36b368ee`.
  Build audit hash is
  `7c57c3d210e22479fc695abf8ae4308afbc52ad853668027eb8d90e803b6ac4c`.
- No tests, native workloads, builds, timing probes or remote operations were run
  during this review. Final head CI was not examined in this wrapper review.

## Blocking issues

1. **`main` — outer signal ownership:** `requested` is never connected to signal
   handlers. SIGTERM terminates the outer owner directly, leaving the separately
   sessioned collector/native tree without its owner. Route SIGINT/SIGTERM into
   the existing monitor's cancellation request while retaining the original
   absolute deadline; cancellation must reject acceptance and settle owned work.
2. **`main` / `invoke` — preserve the primary error:** outer
   `settle_after_error` or failure-journal errors replace the triggering error.
   The unconditional invocation receipt write can also replace its primary
   error. Guard secondary cleanup/journal failures and retain their evidence
   without replacing the original exception. Invocation cleanup itself already
   follows this rule.
3. **`main` — pre-spawn deadline:** after configuration/import/bootstrap work,
   the outer owner starts a collector without checking the original acquisition
   deadline. An exhausted setup can therefore create a fresh owned process even
   after the cleanup budget has expired. Apply the same original-clock cutoff
   immediately before this Popen, without resetting or extending the phase.

Focused pure mocks should cover these boundaries; unchanged passing checks need
not be repeated.

## Coverage and documentation

No further blocker was found in frozen-helper import isolation, common-helper
union seals, build comparison, quiet-process roots, native orphan rejection,
global worker uniqueness, original paired statistics or final success writes.
The parent and child share the pre-import clock, and successful final writes are
checked against its 600-second limit. The native polling reserve is stricter than
the 530-second acquisition cutoff; it does not extend the budget.

The review packet's candidate-build-pending wording predates the now verified
terminal receipts and should be updated when the repaired wrapper is frozen.
Actual quiet/resource admission, wrapper delta review, explicit single-phase
authorization, formal evidence, exact-head CI and final specialist review remain
separate gates. Mock topology is not evidence of eighty completed native runs.

## Verdict

**Request Changes** for the three TARGET wrapper boundaries above. Product source,
reviewed harness and release-build identity are clear within this review's scope;
formal acquisition and merge are not approved.

## Focused repair review

Controller `4a4f6ec232908de80b5e10a4211263f45d732afaace9971f9362d1070b28c065`
and tests `92190d56f23103760f682f71dc1fdf68ccafad0d5b21e470861ac74ab44aaa13`
match the repair packet; the frozen config is unchanged. Read the five actual
outer-boundary RED failures, the additional restoration RED, nineteen-test GREEN
and final Ruff log. No tests or native workloads were rerun.

The original-clock outer pre-spawn deadline and primary-error preservation are
closed. Cleanup, receipt/failure journaling and handler-restoration errors retain
the original exception; signals now write the original cancellation request and
handlers are restored.

One part of the signal finding remains: the pinned monitor checks normal exit
with no remaining owned processes before reading `requested.reason`. A signal
recorded as the collector finishes can therefore coexist with an accepted
receipt. `run_owned` currently accepts that receipt; its signal mock explicitly
returns rejection and does not exercise this ordering. Check the local request
before Popen and when accepting the monitor result, rejecting an already recorded
cancellation. A pure mock recording a request while returning an otherwise
accepted receipt is sufficient; the borrowed monitor needs no change.

**Delta verdict: Request Changes**, limited to that cancellation acceptance
boundary. Build/source identity remains clear. Actual quiet/resource admission,
the single 600-second formal acquisition, performance evidence, final CI and final
review remain unperformed gates.

## Final cancellation delta

**ControllerSourceClear** at controller
`bb122f99a4c0921a6f87373d60d1ebb2262789f6f2f621cf29e7bbfc2d17219c`
and tests `036c67ca84c1c017b92cd945897091c36e34e11007898e37c834dad4265c2455`;
config hash remains unchanged. The two actual request-race REDs are followed by
twenty-one pure tests passing, the formatting-only affected test passing and
final Ruff passing. Evidence was read, not rerun.

`run_owned` now rejects a recorded request before spawn, marks an otherwise
accepted receipt as cancelled after monitoring, and checks the request again at
the final success boundary. This closes the remaining finding without changing
the pinned monitor, statistics or original deadline. All review findings are
closed; the earlier build identity and compatibility clearance still applies.

**Comment Only:** the controller/build review gate is clear. Root must still
perform fresh OS quiet/resource admission and explicitly start the one fixed
600-second, eighty-process phase. No performance result, final CI or merge
approval follows from this source/mock review. No native process was launched.

## e988 epoch artifact delta

**EpochBuildClear / ControllerConfigClear** for one original preregistered phase,
subject to Root's fresh quiet/resource and latest-main admission. No native case,
probe, build or passing test was rerun for this review.

- Candidate clean head `cffee37ccda0ca9d4f6fb7ae28d0e985cc8a498d`, tree
  `17256620be57b711f79a287be007aa632d299f40`; baseline clean head
  `2de4772b4e6734d1dc33ccf8c35ae95b780c7e12`, tree
  `b2a40d0c1ba90d6c5d6678582b094e9fc6f29301`, based on main
  `e988e1e25b1ae52574a91985efcd19c6d88bd3ef`.
- Rehashed all 1,254/1,258 current source seals, source before/after snapshots,
  clean checkout identity, actual Cargo completion, unique core/bench artifacts,
  original and copied executable, compiler log and build log. Both arms genuinely
  recompiled core and bench (`fresh=false`) with the reviewed release profile,
  feature, package paths, toolchain, lock and flags.
- Baseline receipt
  `4337a0361a7567eabd6e823ea01ed71d07bf622e98dfde3aede84dc4e3e8006e`,
  binary `14885c538507e35f97afc9f5a0a421807dca5614c19cb03219ceace40d58e7e0`;
  candidate receipt
  `154c1938d83f77ea5e7aa846b9a0a78973d1b4f2745d4f99fc5462ad155c8007`,
  binary `fd236d8b3d9ce3baed6b00071100767c1cd10fb0b7eecd38c59fd747396599c2`.
  Build durations were 881.968862 and 676.571410 seconds, separate from sampling.
- Build audit
  `0bb27b4a978b04a8fcb6a01086be4bbd675d7afc469ff36950c7f5d233360041`
  and actual pair snapshot
  `66b7d1dc114071f4672fe926e267d6a306745c9261b93c1ffa2f226317a3c9ef`
  match current files; the latter's 113 common-helper and thirteen file seals
  were independently rehashed. Compiled dependency fingerprint remains
  `1d4585499faf5e561c22f05b6a3d4fcdf7adc49c423cb74495b48f9c2374bc79`.
- Config `controller-config-e988-review.json` matches
  `0163b02252932cd65809681deee275b5ce3c872e0374b81b3d04f351051c270c`.
  Only epoch source/revision/build/output/audit/controller-file paths and status
  prose differ. The cleared controller `bb122f99` and tests `036c67ca`, pinned
  ownership/guard helpers, cases and statistical/budget rules are unchanged.
  Its stale receipts-pending status is historical prose, superseded by the
  verified receipts above; functional paths identify those actual products.
- Both arms share all five harness files. The original four code-file hashes
  are unchanged; protocol prose only records the new main/unused old epoch.
  Candidate SQL source is byte-identical to cleared `6a384efc`. The new baseline
  differs from main only by the five benchmark/protocol files. Old a0 receipts
  and audit remain unchanged and are not relabelled or pooled.
- The fixture's checkpoint three-field assertion is correct statically. Both
  Cargo logs identify DataFusion aggregate 55.0.0; ordinary `Sum::state_fields`
  returns one nullable field and `SumAccumulator::state` returns one value.
  Calc Flow's `native_descriptor` constructs wire fields from the two keys plus
  that aggregate state, yielding three. The sliding-window accumulator's count
  is not an additional field on this grouped aggregate route.

The planned acquisition/output directories are absent, recorded native case
count is zero, and the original 600-second phase is unspent. The permitted
acquisition remains exactly two cases, two rounds of ten AB/BA pairs per case,
eighty fresh processes, under the same 530/70 split. Root still owns fresh OS
quiet/resource admission and the explicit single-phase operation. This clears
the epoch artifact/config gate, not performance, final-head CI or merge.

## Subsequent main affected-path admission

Compared local commit objects from frozen main `e988e1e2` to
`43027e28cd42b1889e3c4210ff550dfbe54523a2`, including fourteen intervening DAG
commits. No source edit, fetch, build, test or native execution was performed.

The net runtime changes are native array elementwise/Boolean storage and APIs,
native array mean/EWMA modules, and arithmetic extraction in
`operator/rolling.rs` / `operator/rolling/kernel.rs`. Python additions affect
`calc-flow-python` array bindings and Python array APIs. These are real product
changes, not documentation-only commits.

The following relevant paths have no delta: `batch.rs` (including TableBatch),
SQL operator/incremental/Int64 SUM/snapshot/cache/checkpoint implementation,
DataFusion execution and templates, state memory pools, streaming runtime and
state lifecycle, Rust benches and selected/common statistical helpers, Cargo
manifests/lock, toolchain and Cargo configuration. The fixed query has no rolling
or window stage; its integer grouped-SUM route does not call the changed native
floating rolling arithmetic. NativeArray storage byte accounting is separate
from the unchanged Arrow-table envelope/resource route.

**AffectedPathClear:** no directly affected C1 route or build/dependency change
requires a third build epoch before the original phase. The already sealed
baseline/candidate may run once under the existing e988 admission, explicitly as
**e988 evidence**. Keep their actual identities and the unchanged fixed cases,
sample count, budget and statistics; do not relabel or pool them with current
main. This is not binary timing equivalence or a performance claim for 43027e28.

Root must disclose the advancing main in the resulting report. Final merge
remains gated on separate integration/current-main delta assessment, applicable
checks, measured evidence and final specialist review. This decision leaves the
prior source/build/controller Clear unchanged and still requires fresh quiet/
resource admission before the one formal phase.

## Final stopped-phase disposition

**Withdraw / Do not merge.** The single authorized formal attempt failed its
environment contract. Closing draft PR571 without merging is consistent with
the preregistered stop rule and the requirement for demonstrated real gain.
This supersedes readiness as an execution gate; no replacement attempt is
approved by this review.

Verified `epoch-e988/stopped-phase.json` hash
`51e3513215a96e0f8ce4c7eebb43a0762ec8b1bfa8104ed6f245f7f7dd23e48c`,
all eleven sealed raw files and the four historical controller/config/review
copies. There is exactly one native spawn record: sparse baseline PID 62419,
exit zero, reaped, no cleanup events and no remaining owned processes. Its
report records `NUMEXPR_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`RAYON_NUM_THREADS` as null rather than the required `"1"`. The unchanged
strict normalizer rejects it with `native thread environment changed`.

The sparse pair contains an empty observation map. No accepted observation,
complete pair, candidate invocation, dense invocation or results file exists.
The outer collector PID 62396 exited one, was reaped, and reports no remaining
owned processes. The monitor receipt elapsed time is 1.228403 seconds; the
subsequent failure-cleanup receipt is at 1.230228 seconds. The earlier value
must not be described as an exact all-inclusive total through later failure
journal writes. Neither raw timing values nor this wrapper failure establishes
a runtime speedup, regression or equivalence.

The earlier source review missed the three-variable environment-construction
mismatch. The repaired controller
`5750cb9b92489c40086d7ec11c80979e2ccd03b85c3b4b5f93c1e0a49315ab68`
changes only environment construction to merge the frozen selected helper's
`THREAD_ENVIRONMENT` map into `child_environment`. Tests
`e33669813a0a4c2b95e9c73c20241bc2fee7f1dd37510560bf985865a5bd7860`
add an actual initialization assertion. Read the genuine one-test RED,
twenty-two-test GREEN and Ruff success; no check or native process was rerun
for this review. The repair addresses the cause without relaxing validation.

The historical phase controller remains `bb122f99`, preserved with its original
config and review. Both products/receipts and the frozen normalizer/common
helper union remain unchanged. There is no second native spawn or accepted
sample after the repair. Unused wall-clock allowance does not convert the
stopped attempt into an authorized retry or justify adding favorable samples.

CI is not green: Root reports the coverage-uploader checksum failure and Rust
coverage still pending; this review neither re-polled nor waived those gates.
No performance or final merge approval exists. Preserve the failed evidence and
wrapper correction in the withdrawal archive; any later comparison requires
its own authorized protocol rather than continuation of this attempt.
