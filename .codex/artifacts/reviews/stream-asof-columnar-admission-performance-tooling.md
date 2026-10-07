# A4 static performance tooling review

## PR #373 Review: public admission probe preparation

**Scope:** independent static tooling review before builds. The reviewed source
export is `fd73c1a91bb9d1b017c9de471a99789f867c3830`; production remains
`e3e3e11f7982ae7144d2b760b4b77bdddc20c060`. The baseline is the exact A3 export
`764843e634ae1a1da7a5b010095c3017349a25f4`, production
`9983cfd0c3d096da7a60c3a96ae03b785f5ae955`.

**Verdict: Request Changes.** Two confirmed tooling guards must be repaired
before native builds or measurements are released. This does not reopen the
[approved source review](stream-asof-columnar-admission.md). The
[approved measurement protocol](stream-asof-columnar-admission-performance-protocol.md)
remains unchanged, SHA-256
`3f321c719b6e76061bf5245f9df608623cc9b8b2237e70af21ca65d09ae4b93f`.

## Summary

The standalone public Rust probe times complete left
`StreamOperator::process_data` callbacks with empty right input. Its post-timer
work captures admitted state, real prefix cuts, live and restored continuation,
all twelve output columns in canonical order, and terminal logical inventory.
The Python coordinator owns fresh sequential processes and complete raw
attempts, compares full archives in bounded blocks, and uses the maintained
paired confidence helpers.

The callback boundary and functional archive design match the bounded protocol.
The identity guard omits the verified closure files, and the RAM ladder remains
optional. Consequently an altered closure or an unbridged 1M preflight can
pass the launch guards.

## Reviewed bytes

The prepared directory is
`target/issue363-asof-columnar-admission-perf/preparation-v1`, a symlink to the
isolated A4 worktree's owned `target/` directory. All five source/test files,
README, `static-ready.json`, and `verification.json` were read in full.
Only small text files were hashed during this review:

- `admission_probe.rs`:
  `1e3ac216b8d91921638545d86d8637c5fb3e1b6f9c81831e9795097460baa220`.
- `coordinator.py`:
  `6a6ef5b41c571e7c09448dea2fca580866182610ab5d435860cc9f34db794d47`.
- `oracle.py`:
  `735985e3c994d523ee1f7c1c96e93f1d78fa9cc8a35be990aa7d9ee9a1f8dd38`.
- `test_coordinator.py`:
  `83d329690e71fe2efd5b0427f143ada23646b59f6b875175b973770967330417`.
- `test_oracle.py`:
  `6d36225d9b29ce9fc84e04d691491726e3d9b58718c192b499b74e95ea2d38bd`.

## Build and test results

- Rust: **Not run**. The probe has not been compiled or executed. No Cargo,
  native import, release build, or core/native/binary hash scan ran here.
- Tooling Python: full static review plus two additional stdlib-only adversarial
  checks, both confirming the guard defects below. No subprocess or engine ran
  in those checks. Temporary synthetic files were removed.
- Author evidence: `verification.json` records twelve unique passing stdlib
  tests, including launch-failure preservation and actual linked-core mismatch
  RED to GREEN. Those unchanged checks were read, not repeated. Author fmt,
  Ruff, Lizard and whitespace results are recorded evidence, not reviewer runs.
- Python application, Studio backend and frontend: **Not touched / not run**.
- Product CI, coverage and performance: **Not inspected or established** by this
  static target-only handoff. There are no measured native observations.
- New failures: two confirmed tooling guard defects. No production regression
  was observed or inferred from these synthetic checks.

Reviewer reproduction was one `python - <<'PY'` command from the prepared
directory, tool output chunk `74215b`, exit 0. It reported:

```text
CONFIRMED: quiet verified_inputs accepts a changed prevalidated closure core
CONFIRMED: 1M preflight reaches launch with no previous-shape RAM proof
Synthetic files removed; no subprocess/native probe/build executed
```

The setup called `validate_seals` on tiny synthetic read-only cores and binaries
with complete matching manifest hashes, expected source identities, non-fresh
release receipts and linked `--extern` paths. Statistics, verdict and protocol
fixtures were also tiny text files. The checks below ran after that complete
synthetic setup; no real compiler receipt or native artifact was substituted.

## Blocking issues

### `coordinator.py` — `_verify_side`, `_verified_identities`, `verified_inputs`

Prequiet `_check_closure` validates the actual closure manifest and complete
artifact hashes. The returned verified receipt then preserves identities only
for the seal and executable. `_verified_identities` never includes the closure
manifest, linked core, dependencies, native objects or sysroot entries. Quiet
validation therefore accepts changes to files that were declared immutable.

The actual reproduction changed the previously verified baseline core, then
called the real quiet validator again:

```python
coordinator.verified_inputs(verified_path)
cores["baseline"].chmod(0o644)
cores["baseline"].write_bytes(b"synthetic changed core")
cores["baseline"].chmod(0o444)
coordinator.verified_inputs(verified_path)  # Unexpectedly succeeds.
```

Required correction: capture actual resolved identities for the validated
closure manifest and every owned artifact during prequiet verification; check
all of them before launch without rehashing large files in quiet. Bind the
receipt to the actual manifest entries and owned paths, including the linked
core, rather than accepting a list asserted by seal fields. Preserve the
existing fresh matching source/core receipt, frozen-core hash and exact
`--extern` checks. No ECCB core substitution or mutable-cache hardlink is
permitted.

Add focused failing tests before the fix for changed manifest, core and another
closure artifact, plus unchanged complete-closure acceptance. A changed closure
must abort without launching a child or silently renewing verification.

### `coordinator.py` — `_run_case`, `estimate_next`

`--previous-shape` is optional even for the declared 100k and 1M preflight
cases. `_run_case` can enter `_preflight` without an accepted same-shape RAM
bridge. When supplied, `estimate_next` also permits arbitrary increasing prior
row counts instead of proving the prescribed 10k → 100k → 1M sequence.
The full-shape peak check occurs after launch and cannot replace the required
screen before launching that shape.

The actual reproduction replaced only `_preflight` with a no-launch sentinel:

```python
report = {
    "case": coordinator.shape(1_000_000, 64_000),
    "verified": {},
    "attempts": [],
}
args = SimpleNamespace(previous_shape=None, mode="preflight")
launched = []

async def sentinel(*args):
    launched.append(True)

with patch.object(coordinator, "_preflight", sentinel):
    await coordinator._run_case(args, report, Path("."))
assert launched == [True]  # Missing bridge still reaches the launch path.
```

Required correction: before any 100k/1M native launch, require the exact accepted
same-shape predecessor preflight for both sealed comparators. Preserve the
fixed batch policy, key/payload/types/variant/lifecycle, raw available-RAM
conversion, 1.25 prediction factor, 70% gate and actual full-worker zero-swap
checks. Prove the accepted predecessor chain; a direct 10k → 1M bridge, failed
or mismatched prior proof, or a proof from different sealed binaries must not
authorize launch. Root launch instructions alone do not close this guard.

Add focused failing tests before the fix for missing predecessors at both
scales, skipped/wrong/mismatched/failed predecessors, and accepted exact chain
launch eligibility. Preserve rejected requests and failure reasons as raw
attempt evidence. Do not relax the resource thresholds.

## Nonblocking observations and acceptance limits

- The contiguous callback timer includes native dispatch, constructor,
  installation and retirement. Input construction and the complete checkpoint,
  restore, output and oracle work remain outside that timer. The primary 1M
  inventory is fifteen 64k callbacks plus one 40k callback, validated as sixteen.
- Prefix cuts include 512 and 32,001 rows for 1M. Both live and restored
  trajectories preserve starts, real progress cuts, bounded continuation and
  terminal snapshots. The all-column oracle checks exact schema, independent
  global ordinals and typed NULL right fields without sorting output.
- Complete segment bytes use read-only hardlinks within one observation only.
  Cross-source state metadata, counters, hashes and complete files are compared
  in 1 MiB blocks. Live and restored journal topology are compared separately.
- Fresh workers are awaited before the next launch. Raw failed callback,
  process exit, launch failure and cancellation evidence remain visible, and a
  failed pair cannot become an accepted performance result. Whole worker/pair
  costs include post-timer oracle and archive work.
- The two ten-pair rounds alternate AB/BA and call the unchanged maintained
  order-statistic and verdict helpers. No aggregate-median ratio replaces
  paired inference. No performance observation has been collected here.
- Exact pool refunds, private worker phases, direct projection and managed
  delivery remain explicitly unobservable through this probe. Public logical
  state zero and process exit do not prove reservation balance. Existing private
  source tests and the separate unchanged eleven-case A3 E2E fixture retain
  their distinct roles.
- Matching A3/A4 core closures, actual fresh Cargo receipts, probe compilation,
  immutable artifact validation, full same-shape functional preflights and
  final independent measurement review remain future work. Static readability
  is not proof of Rust type correctness or runtime success.
- Left-only callback results cannot establish FR16's complete 1M ASOF output
  target of 100 ms. Source allocation savings remain separate from throughput.

## Style and documentation consistency

No additional source/style finding is raised. The README and protocol correctly
state the timer and public observability limits. After guard repair, refresh
tool fingerprints and verification evidence, and keep the README's sealed
closure and mandatory ladder claims synchronized with the actual guards.
No public pool API, unrelated documentation or production edit is required.

## Handoff

**Request Changes** for the two tooling guards. Add focused RED to GREEN tests,
then provide a new exact preparation seal for static re-review before builds.
Only this review artifact was written. Reviewer-owned subprocesses: **0**;
reviewer-owned Cargo/native/build/measurement processes: **0**. No remote
operation or unrelated worktree edit was performed.
