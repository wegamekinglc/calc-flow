# A4 performance tooling v2 review

## PR #373 Review: corrected static admission tooling

**Verdict: Approve the exact static preparation below.** Both evidenced blockers
in the [v1 tooling review](stream-asof-columnar-admission-performance-tooling.md)
are addressed. No additional blocking or style finding is raised. This approves
the preparation for matching A3/A4 build work; it does not approve unknown
compiled artifacts, native functional preflight, measurements, or merging.

**Static-ready SHA-256:**
`f7746fdeddba93297cdd77ea5f7504aa532b7ca5202791823f2e262021219e13`.
The preparation's recorded file-inventory identity is
`958989b0a413e63666846c38c72435b0f6a15a9c47c3ca82ad6a986380dff696`.
Every small file's actual length/hash matches the sealed `files` inventory.

The reviewed directory remains
`target/issue363-asof-columnar-admission-perf/preparation-v1`, the root symlink
to the isolated A4 worktree's owned preparation. Changed files were read in
full: `coordinator.py`, `test_guards.py`, README, verification and static seal.
Author archived RED/GREEN evidence was read. Unchanged Rust/oracle/old tests
retain the full reads from v1 and matching small-file hashes.

## Summary

V2 preserves the complete public callback timer, full state/output archives and
separate E2E scope. Its prequiet validation now captures the actual complete
owned closure and binds that receipt to the real build seal and manifest.
Every child launch checks that closure and the mandatory exact accepted
10k → 100k → 1M predecessor chain, including current available RAM.

The production exports, [source approval](stream-asof-columnar-admission.md),
and [measurement protocol](stream-asof-columnar-admission-performance-protocol.md)
are unchanged. Baseline export is
`764843e634ae1a1da7a5b010095c3017349a25f4`, production
`9983cfd0c3d096da7a60c3a96ae03b785f5ae955`; candidate export is
`fd73c1a91bb9d1b017c9de471a99789f867c3830`, production
`e3e3e11f7982ae7144d2b760b4b77bdddc20c060`.

## Exact reviewed tool identities

- `coordinator.py`:
  `7a2c46d9a7adb1a1aa09fa1781b06590a92bd269f0a9056c744627326d4d8346`.
- `test_guards.py`:
  `528c4963e3dfdd1dff66ffea046c36be1bf46af607a1a7d454461ab2aa0661cf`.
- Unchanged `admission_probe.rs`:
  `1e3ac216b8d91921638545d86d8637c5fb3e1b6f9c81831e9795097460baa220`.
- Unchanged `oracle.py`:
  `735985e3c994d523ee1f7c1c96e93f1d78fa9cc8a35be990aa7d9ee9a1f8dd38`.
- Unchanged protocol:
  `3f321c719b6e76061bf5245f9df608623cc9b8b2237e70af21ca65d09ae4b93f`.

## Build and test results

- Rust: **Not run**. No Cargo, native import, core/native/sysroot hash scan,
  build, native probe or performance execution occurred in this review.
- Tooling Python: **Passed**, seven additional reviewer stdlib checks with
  temporary synthetic owned files and mocked launch sentinels. No subprocess
  was spawned by these checks. Fixtures were removed.
- Author evidence: eighteen new guard tests pass in the retained final raw log
  `corrections-v2/green-final-ownership.log`, exit 0, 0.237 seconds. Earlier
  assertion-based REDs and intermediate failures remain separate. The initial
  fixture errors and same-timestamp mutation failures are not presented as
  authoritative RED or GREEN. The twelve unchanged tests were not repeated.
- Author Ruff, formatting, Lizard, contracts and whitespace results are read
  from the v2 verification receipt, not claimed as reviewer runs.
- Application Python and both Studio surfaces: **Not touched / not run**.
- CI, workspace coverage, compiled probe behavior and native performance:
  **Not established** by this static preparation review.

Reviewer command, from the owned `reviewer-v2-proof/` directory:

```bash
python reviewer_guards.py > reviewer-v2-proof.log 2>&1
```

Actual exit 0, tool chunk `64a43b`; raw output read in chunk `3f31b8`:
**7 tests passed in 0.070 seconds**. The
[reviewer proof](../../../target/issue363-asof-columnar-admission-perf/reviewer-v2-proof/reviewer_guards.py)
has SHA-256
`72baeaa0c39cdb5287059dd86fd51ec58f1b1cd4602be487c42f0944385e4900`;
the [raw log](../../../target/issue363-asof-columnar-admission-perf/reviewer-v2-proof/reviewer-v2-proof.log)
has SHA-256
`855b2cf48f3827eb7c44cfb442ea685e5d99de0667886ea7f31626c45f11fbd3`.
Synthetic fixture helpers construct evidence only; they are not compiler
receipts or full functional engine observations.

## Blocking issues

**None remaining in the reviewed static tooling.** The original issues are
closed as follows.

### Actual complete closure binding

`_check_closure` now records the actual validated manifest, artifacts and
directories. The manifest covers every unique owned closure file. Actual
artifacts are hashed before quiet, must be read-only, and must have one link;
mutable-cache hardlinks are rejected. `_closure_identities` binds captured
entries to the actual manifest and copied provenance to the actual build seal.
The linked core must still match its frozen hash and exact `--extern` path.

Quiet checking includes resolved path, device, inode, bytes, mtime, ctime,
link count and mode for the complete inventory. It refuses altered/omitted
closure receipts and missing comparators. Directory identities also expose
observable inventory changes. The original changed-core attack now raises
before launch. An independent added-directory check also rejects.

`observe_pair` invokes `_launch_admission` before each child. The reviewer
changed the candidate core after the baseline sentinel returned: only the first
sentinel ran, the candidate guard recorded rejection, and no candidate attempt
directory was created. No real process ran in that check.

### Mandatory same-shape predecessor chain

`_resource_admission` precedes `_preflight` and requires exact predecessors at
100k and 1M. `_accepted_predecessor` checks accepted preflight mode, exact row
count, unchanged shape and both sealed comparators. Both predecessor workers
must show successful exit, matching executable/case, full-worker peak and
zero swap. `_preflight_chain` recursively validates the 100k report's accepted
10k ancestor before it can authorize 1M. Other undeclared scales above 64k
are refused.

The original missing-predecessor attack at both scales now records rejection
without reaching the launch sentinel. A direct 10k → 1M substitution is also
rejected. An exact synthetic accepted chain remains eligible without launching
anything. Author tests additionally cover wrong/failed modes, row counts,
types, values, batch policy, variant, lifecycle, binary and ancestor changes.

The original prediction is recomputed from predecessor peaks and the captured
raw RAM conversion, including the 1.25 factor and 70% screen. Before every
larger child, the full chain is checked again and the prediction is screened
against current available RAM. The reviewer lowered available RAM between
the two sentinels: the second launch was rejected with its own current-RAM
receipt. Sequential workers use the maximum prior worker peak, preserving
the approved topology; full actual peak/zero-swap checks remain after execution.

## Style and documentation consistency

README examples now require `--previous-shape` for both large preflight and
measurement. Rejected case/child guard receipts preserve the request and
reason. V1 receipts and reports remain archived rather than silently rewritten.
The description of closure ownership and mandatory chain matches the code.

The documented stat limitation is accurate: quiet checking relies on owned
files remaining immutable and observable filesystem identities. It is not
continuous cryptographic tamper detection. A same-size mutation with restored
mtime may share a filesystem ctime epoch; distinct-epoch checks demonstrate
the intended guard without claiming a stronger guarantee. No modification to
sealed artifacts is authorized by that limitation.

## Remaining acceptance boundaries

The matching fresh A3/A4 core builds, actual compiler receipts and frozen link
closures do not exist yet. Build owners must bind actual copied artifacts to
those receipts; synthetic seals do not prove provenance. ECCB/J1 core and the
A3 Python wheel remain invalid replacements for the new A3 Rust core.
Rust type correctness and all complete native preflights remain unverified.

The probe still measures sixteen complete cold-job left callbacks for 1M,
including dispatch, installation and retirement. Full checkpoint, real prefix,
live/restored continuation and all-column oracles stay outside that timer.
The frozen eleven-case A3 E2E timer and fixtures remain separate and unchanged.

Public logical zero/process exit do not prove exact pool refunds; private
worker phases, direct projection and managed delivery remain outside direct
probe observability. No source allocation saving or future callback result
establishes FR16's complete 1M output target of 100 ms. Final paired-evidence
review remains necessary after actual correct observations.

## Verdict and ownership

**Approve**, bound to the exact v2 static-ready hash above, for the next matching
build stage. Only this new review and owned reviewer proof were written;
author tools, earlier reviews, source, protocol and other worktrees were not
edited. Reviewer-owned persistent processes: **0**. Reviewer child/native/
build/measurement executions: **0**. No remote operation was performed.
