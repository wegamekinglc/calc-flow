# A3 fixed-50 tooling v2 independent review

**Author:** phase0_performance | **Scope:** target-only tooling correction
| **Changed executable tools:** wrapper, verifier, focused tests

This is the focused follow-up to the [v1 Request Changes review](stream-asof-output-ranges-fragment-followup-tooling.md).
It reviews the three reproduced failures and their corrections, without reopening
the approved source change or original 440-observation performance evidence.
There are no new product changes, PR check runs or remote review actions in this
tooling handoff.

## Summary

Version 2 rejects missing or non-integer successful exit codes, owns the spawn
task and eventual process handle throughout startup and cancellation, and merges
every observed PID peak across both workers for the whole-round RAM gate. The
three prior blocking findings are resolved. The fixed sampling protocol and
original worker, primary timer, fixture shapes and release identities are
preserved.

## Exact reviewed identities

- Implementation seal: `target/issue363-asof-output-range-perf/fragment-followup-tooling-seal-v2.json`
  SHA256 `01dd72c09662026a6675c4b7fd4aec9529dfa464a51378f2accd80d6cf2ba066`.
- Sampling fingerprint:
  `dd2876a21f1f3958042062c86dfb4ec6614cff0d8cf23e272d5e220aa80dae7d`.
- Wrapper `fragment_followup_v1.py`:
  `040261b84281dba06910a93f200aab63a01fe92a4d175c35ecee0c26dcf0cc06`.
- Verifier `verify_fragment_followup_v1.py`:
  `c3f98d466de872a08ef0e693d53fd8be927698dd730419a4642b73e4103d1d3d`.
- Focused tests `test_fragment_followup_v1.py`:
  `906e6dbd4df5dcebde8c0eaec1128f0e48a67cbbfa1c2cb44c83262dbd95f195`.
- Independent [focused proof](stream-asof-output-ranges-fragment-followup-tooling-v2-proof.json):
  SHA256 `62f1f9163683c698cf8d8104d170f2cb7bb95e87ab392c43717e0568621060a8`.

The source approval remains bound to production commit
`9983cfd0c3d096da7a60c3a96ae03b785f5ae955`, tree
`78065d9fe80635e3f6d64831215acacd9ad3e572`; the selected candidate includes the
source review document at `764843e634ae1a1da7a5b010095c3017349a25f4`.
The baseline remains ECCB, with native SHA256
`06b152baf350be5d4ac21d9d8d4d5ca9481ef740e8d7d1ef62268098d44b3390`;
the A3 native identity remains
`e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889`.
This review checks their preserved metadata bindings; it does not rehash the
large native modules, wheels or original evidence archive.

## Build and test results

- Rust, Python product suites, Studio backend and frontend: **Not run**;
  product source is unchanged by this correction.
- Author focused RED: **Observed**, three assertion failures and zero errors
  against v1, preserved in `fragment-followup-blockers-red-v2.log`.
- Author focused GREEN: **Observed**, 27 stdlib checks passed in 2.658 seconds,
  with no native or Arrow modules loaded and no operating-system worker spawns.
  The final log and proof identities match the submitted seal. These checks
  were not rerun as a suite.
- Reviewer independent verification: **Passed**, 12 focused stdlib groups in
  0.597 seconds, using synthetic files and fake handles only. The proof records
  the individual outcomes and cleanup inventories.
- Reviewer static identity checks: **Passed**, 36 sealed entries and all 25
  archived v1 files, with 22 original inputs unchanged. Only the wrapper,
  verifier and tests differ from the archived inputs. The largest file read for
  these checks is the existing 4,496,420-byte matrix JSON.
- Ruff check: author evidence **Passed**; unchanged passing lint evidence was
  not repeated. CI was not consulted for this target-only tooling review.
- New failures: **None**. Unresolved regressions in this tooling scope: **None**.

The independent commands used `PYTHONDONTWRITEBYTECODE=1` and stdlib scripts via
the existing benchmark interpreter. They loaded the wrapper, verifier and test
fixture definitions, extracted the frozen startup AST, constructed temporary
synthetic evidence under `target/`, and replaced `create_subprocess_exec` with
fake handles. They did not invoke the native worker, build tools, bootstrap
statistics or measurements. Temporary synthetic files were removed.

## Blocking issues

None in this submitted v2 correction. Each prior finding was checked against
its original counterexample:

1. **Verifier — `worker_evidence`: resolved.** It requires
   `type(exit_code) is int` and `exit_code == 0` before asserting successful
   cleanup. The independent fixture rejects null, either boolean, floating
   zero, string zero and nonzero integers.
2. **Wrapper — `ResourceBoundWorker.start`: resolved.** The private startup
   opens the log and retains a shielded spawn task before awaiting its handle.
   Constructor, registry insertion and command-journal failures retain cleanup
   ownership. The independent checks use the actual frozen startup AST as the
   base class, inject journal and constructor errors, registry failures before
   and after insertion, second-worker journal failure and spawn failure, and
   cancel three times while the second handle arrives late. Every created fake
   handle is terminal and awaited, every log is closed, and the live registry
   and remaining task inventory are empty. Repeated cancellation during close
   also waits for owned cleanup; a subsequent close remains safe.
3. **Verifier — `worker_evidence` / `verify`: resolved.** Both complete guard
   journals and IPC observations contribute maxima by PID and the global
   minimum available RAM. The old counterexample still passes every individual
   guard, but the reconstructed whole-round estimate of 1,393,377,280 bytes
   exceeds the 751,619,276-byte limit and is rejected. A separate passing
   variant reports the peer-observed baseline maximum of 1,073,741,824 bytes.
   The reporting and admission gate use the same maxima.

## Style issues

None identified in this focused correction. The new mutation stays within the
owned process/log/registry lifecycle. Pure resource arithmetic and sampling
contract functions retain identical ASTs; caller observations are read-only.
No dependency, public API or native implementation changes are introduced.

## Test coverage

The preserved RED evidence matches all three independently reproduced v1
defects. The author's 27 checks cover the corrections and existing full raw
verification. Independent checks concentrate on the rejected exit values,
cross-peer RAM extrema, reported peaks and process ownership through startup
faults and repeated cancellation. They assert outcomes rather than reproducing
the implementation's internal sequence.

The synthetic valid fixture verifies 400 timings, eight warmups, eight workers,
408 full-column/terminal EOF proofs and 424 resource guards. These are tooling
acceptance checks, not measured performance or native runtime results.

## Documentation consistency

`fragment-followup-tooling-protocol-v2.md` accurately describes the corrections,
failure evidence and cleanup boundaries. Both command-line defaults select the
v2 seal. The actual sampling contract equals the sealed contract, and its
computed fingerprint matches the exact identity above.

The unchanged [approved protocol review](stream-asof-output-ranges-fragment-followup-protocol.md)
still governs two original 1M/64k fragmented cases, two rounds of 50 alternating
AB/BA pairs, eight fresh workers and 400 timings. Only the active round's two
workers are resident. Original request, timer, fixture and oracle implementations
remain frozen. Maintained ranks 18/33, nominal iid coverage
0.9671608624357315 and the `5 + 1e-12` decision rules are unchanged. There is no
old/new sample pooling, optional stopping or automatic rerun.

The 319-second forecast remains a schedule estimate. This review establishes
neither performance gains nor independence of WSL2 samples. Terminal EOF
manifests do not demonstrate durable restart, and this A3 follow-up supplies no
J1 during-active-preparation latency evidence.

## Verdict

**Approve** for the exact v2 tooling seal and sampling fingerprint above. All
three prior blocking findings are addressed, with no new confirmed blocker.
This is a tooling approval; benchmark execution still requires the root's fresh
quiet grant, and measured results require their own evidence review.

Reviewer-owned native, build and background processes: **0**. Actual
operating-system worker spawns: **0**. No remote actions were taken.
