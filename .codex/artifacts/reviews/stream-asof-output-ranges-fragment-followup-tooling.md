## PR #372 Review: Fixed-50 follow-up tooling

**Author:** Cheng Li | **Branch:** `feature/stream-asof-output-ranges-main` → `main`

**Scope:** Static/stdlib implementation review before native execution.

**Submitted tooling seal:** `36e732537a35c7465f2f3df44548bbab44d86fb680f8cf7a7bc6eef4ea8e1495`

**Wrapper:** `b62d52182647f71469bac3f48ebc7fac54cb95319b5f4ea8722431a114d3142b`

**Verifier:** `321fb9b91647d982f5917b258b8f682be32dd2ef64bf5553ee4159c66d91cc4e`

**Sampling fingerprint:** `8a3b349007a8d43e3ac6f9b284aa379442f15f9df387b6c00755644c5a9768a7`

The [fixed-50 protocol review](stream-asof-output-ranges-fragment-followup-protocol.md)
and [original performance evidence review](stream-asof-output-ranges-performance.md)
remain Approved. This tooling verdict does not revise the original 440 timings,
source approval or sealed ECCB/A3 wheel comparison.

### Summary

The wrapper preserves the original worker/timer/fixture and changes sampling
to two rounds of fifty pairs for the two fragmented-output cases. Its full
verifier covers fixed counts, original identities, actual IPC order, complete
oracles/manifests and per-request resource guards. Three independently
reproduced lifecycle/resource gaps require correction before execution.

### Build and Test Results

- Rust: **Not run**; no source change or build reviewed.
- Python: **16 author stdlib checks passed**; **three reviewer contract
  counterexamples reproduced**. No engine was imported or executed.
- Studio backend: **Not touched**.
- Studio frontend: **Not touched**.
- New failures: Unknown exit status accepted as zero; post-spawn journal
  failure leaves no tracked child owner; cross-worker guard maxima omitted
  from the conservative whole-round RAM gate.
- Regressions: No native/performance observation; this is tooling review.

The reviewer read both implementations in full, the focused test module,
tooling protocol/seal/proofs, and the unchanged driver/process/measurement
ownership paths they call. The author records 16/16 synthetic checks in
2.446 seconds, the original ten-pair checker's fixed-50 rejection as RED,
and validation of the 25 sealed input files. Unchanged passing checks and
historical large hashes were not repeated.

Two reviewer `PYTHONDONTWRITEBYTECODE=1` benchmark-interpreter `python -`
audits returned exit 0 after asserting the three counterexamples below.
They used the author's synthetic fixture in automatically removed temporary
directories; source/evidence files were not edited. The startup audit
extracted the actual original `RecordedWorker.start` AST and substituted
a fake subprocess, so it exercised the real journal ordering without
spawning a child. Imported modules contained no `calc_flow` or `pyarrow`.
Reviewer-owned native/build/background processes are **0**.

### Blocking Issues

- **`verify_fragment_followup_v1.py` — `worker_evidence`: unknown exit status
  accepted.** The truthiness check on `exit_["exit_code"]` accepts JSON
  `null`, then the verifier reports `all_worker_exit_codes: 0`. Starting
  from the complete valid synthetic fixture, changing one baseline
  `exit.json` code to `None` still validates all 400 timings. Require
  `type(code) is int` and `code == 0`, and reject null, booleans and other
  malformed statuses with focused tests.
- **`fragment_followup_v1.py` — `ResourceBoundWorker.start`: missing owner
  before the startup journal.** `live` is populated only after awaiting
  the original `RecordedWorker.start`. That original starts the subprocess
  and then writes `command.json` before returning. If the write fails,
  neither the original round's worker map nor the wrapper's `live` map
  owns the already-created child. The AST/fake-process reproduction raises
  `OSError` at that write with a child whose return code remains `None`
  and an empty cleanup map. Own the process/log from creation inside the
  new wrapper and close/reap on journal, constructor, registration or
  cancellation failure. Keep the frozen original driver unchanged; cover
  first/second-worker startup failure and cleanup with fake owners.
- **`verify_fragment_followup_v1.py` — `worker_evidence`/`verify`: incomplete
  whole-round HWM aggregation.** A worker's reported peak includes only
  its own IPC and its own guard records. It omits observations of that
  PID in the other worker's guards. Aggregate every IPC/guard observation
  by PID across the entire round, then use those maxima and the global
  minimum available RAM for both the conservative gate and output peaks.

The third reproduction preserves all 424 individually passing guards.
In a valid synthetic round, baseline's first hello guard records 1 GiB
available RAM for both PID observations. Candidate's final finish guard
records baseline HWM at 1 GiB; raw kB lines and byte fields agree, and both
guard results are recomputed. The verifier still accepts all 400 timings
and reports 40,960,000 bytes for each worker. Across all recorded guards,
the true maxima are 1,073,741,824 and 40,960,000 bytes: their 1.25 sum is
1,393,377,280 bytes, above 70% of minimum free RAM, 751,619,276 bytes.
The declared conservative whole-round gate must reject this fixture.

These are observed tooling contract failures, not findings inferred from
another feature's checker or hypothetical production changes. Preserve
the submitted v1 files/seal, add focused RED/GREEN evidence, and deliver
a new tooling/sampling seal for independent delta review.

### Style Issues

No additional style blocker found. Sampling overrides are confined to the
isolated coordinator process and explicitly fingerprinted; the wrapper
adds result metadata through a new mapping. The resource helper leaves
caller observations unchanged. No original source/worker/timer file needs
to change for the corrections.

### Test Coverage

The existing suite demonstrates complete synthetic verification of 400
timings, eight warmups, eight fresh PIDs, 408 full-column/EOF proofs and
424 guards, plus corruption checks for counts, arrays, native identity,
oracle columns, AB/BA order, ingress shape, swap/refusal, prediction,
terminal EOF, old PID reuse and sampling fingerprint. Its fake resource
failure cleanup starts both workers successfully, which misses the
post-spawn/pre-return owner gap. Exit-null and cross-worker maximum cases
also need focused coverage.

The reviewed implementation otherwise retains exact 50-pair/two-round
counts, ranks 18/33 and the maintained `5 + 1e-12` verdict order. It checks
actual UTC request order, round serialization, independent arrays, original
environment/workload/native/source bindings, the old 44-PID exclusion,
shape/workspace metadata, complete status/manifest inventory and actual
per-request 70%/1.25/zero-swap arithmetic. The original sealed same-shape
resource proof supplies initial admission, with fresh available-RAM data;
no confirmed missing-preflight/identity finding was identified there.

Partial failures are retained and cannot receive a performance verdict.
Fixed N, fresh destinations and serialized two-resident-worker rounds
preserve the protocol's no-pooling/no-optional-stopping scope. A corrected
attempt still requires an explicit root quiet grant; none has been issued
by this review. Tail quantiles remain descriptive, iid assumptions remain
explicit, and this tooling adds no durable restart or FR10 active-prep proof.

### Documentation Consistency

The tooling protocol's claim that all owned startup failures clean up and
that whole-round HWM is conservative is currently stronger than these
implementations. Fix the code and scoped tests to meet those statements;
do not weaken the original protocol or frozen evidence. Tooling protocol
SHA is `be7240fa7804e68ec2a24b14c993342b65b8eb4ddc80a641b9d57215984360d6`;
the author 16-test proof SHA is
`f70c617deff722ee24aeab8405df41af753e6c3e10c038d8d64b65ce941923cf`.
The submitted seal accurately states that implementation review and native
authorization are pending.

### Verdict

**Request Changes.** Resolve all three blockers and provide the corrected
sealed tools plus their focused stdlib evidence. Native execution remains
unapproved. Reviewer-owned native/build/background process count is **0**.
