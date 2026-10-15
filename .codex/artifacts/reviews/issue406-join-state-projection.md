## Branch Review: issue 406 Join retained-state projection

**Branch:** `feature/issue406-join-state-projection`

**Reviewed commit:** `e884adc362ac8f8d03c1364d0dfdb4e205407696`

**Base:** `3cfe6c49f8f962d9fa7ae7476b9f2f3e2f2c2a56`

**Scope:** phase 2.2 local implementation; 58 changed files, including 50 Rust files.

### Summary

The implementation retains independently owned Join payload dependencies plus
identity/time columns while preserving complete logical input schemas, authentic
execution's V1 charges, output order and logical configuration/fingerprints.
Actually narrowed state uses strict inner layout 3/codec 2, validated legacy
migration, and identity-bound incremental checkpoints and rebase. This review
reuses the bounded protocol, runtime, ownership and recovery reviews and checks
their final committed integration against the accepted specification and critique.

### Build and Test Results

Evidence below is the owner's recorded execution, inspected by the reviewer;
no builds, tests, measurements or CI queries were repeated for this handoff.
Evidence paths are relative to the repository-root
`target/issue406-join-state-projection-20261014/` directory.

- Rust focused verification: **Passed**. The latest
  `selected-copy-worker-corrected.log` reports one existing registration passing
  in 0.18s after 1m21s compilation. It executes the selected-state suite,
  including six actual generic worker lifecycle/admission controls; it is not
  six additional test registrations. Default libtest registration count remains
  2,033.
- Writer/rebase verification: `history-identity-writer-focused.log` reports
  28 passing prepared/rebase/pending registrations in 1.15s. The later worker
  identity refusal and changed stale-cut/pending helpers also have passing
  focused logs. These earlier results retain their recorded source scope.
- Additional focused results: prior projection execution/recovery, malformed
  layout-3 restore, V1/V2 migration, dictionary component controls, emitted
  counter preflight, and actual native sparse replacement all have passing
  results. In particular, `prior-projection-recovery-contract-green.log`
  reports one pass in 0.23s; `emitted-counter-preflight-corrected-green.log`
  one in 0.01s; `retained-sparse-projection-controls.log` one in 0.20s; and
  `dictionary-copy-component-corrected.log` one in 0.14s. These registrations
  overlap other recorded runs and are not added into a fabricated unique total.
- Scoped core Clippy: **Passed**. The final
  `clippy-selected-copy-worker-corrected.log` completes in 56.81s with
  `--lib --tests --all-features -- -D warnings`.
- Python catalog: **Passed, limited scope**. Two inventory tests pass in 0.53s,
  with scoped Ruff checks recorded. The target-only overlay imported compatible
  old native artifact
  `552b6313ccabc59d5080ea30c97965075685fbb6164b8f0b6923fa75c460e924`,
  as recorded in `python-overlay-provenance.json`. This proves the changed
  Python inventory, not new-native Python execution, recovery or release.
- Final formatting, generated-schema no-drift and whitespace/newline checks:
  **Passed**, as recorded in the analysis. The documentation owner also checked
  84 local links/anchors and table structure.
- New unresolved failures: **None found**. Expected behavioral REDs, fixture
  compilation/type errors and Clippy failures remain recorded with their
  diagnosed corrections and later passing evidence.
- Regressions: none observed in the recorded focused checks. Full regression,
  cross-platform CI and the combined Rust 90% line gate remain unverified.

Recorded core command scope, from the isolated worktree with
`CARGO_TARGET_DIR=/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/issue406-join-hotpaths-build`,
`CARGO_BUILD_JOBS=2` and serial focused execution:

```bash
cargo test -p calc-flow --lib \
  test_charge_fast_path_falls_back_when_the_bound_exceeds_the_budget \
  -- --test-threads=1 --nocapture
cargo clippy -p calc-flow --lib --tests --all-features -- -D warnings
cargo fmt --all --check
git diff --exit-code -- schemas/project-v3.schema.json
git diff --check
```

The reviewer independently compared every changed Rust file on disk with its
committed bytes and the frozen manifest. All 50 files match, and the manifest
file set exactly equals the changed Rust file set. Independently recomputing
the sorted `file + ASCII space + sha256 + LF` aggregate yields:

```text
706257eb1a3e504deffe4b6dff91515a544aadae83330235bc254f21d8f00302
```

`complexity-phase22-review.json` records 293 changed/new functions, maximum CCN
8 and no changed/new parser omissions. Its five listed omissions are unchanged
inventory helpers; they are not represented as measured changed functions.

### Blocking Issues

**None remaining in the reviewed phase 2.2 implementation.** Findings from the
bounded reviews were addressed in the committed source:

- `state_projection::Layout`, admission and output/state-key mappings separate
  logical input indices from retained physical indices. Full-input charging
  and classification occur before selected ownership replaces admitted and
  retained records. Both sides, shifted identity columns and zero-column
  output retain their required identity/time fields.
- `projected_copy`, `views` and `views::dictionaries` now independently copy
  selected recursive buffers. The multiple-run nested-dictionary alias is
  removed by first-seen key remapping and recursive value copying. Short
  validity bitmaps and empty recursive offset buffers have explicit prepaid
  geometry. The generic eight-key copier does not broaden the historical V1
  runtime charge implementation's Int32 dictionary contract.
- `CopyWork`, observed cleanup and payload `Funding` preserve ownership until
  native settlement and the last escaping buffer. Actual input/result gates,
  admission pressure and cancellation priority controls distinguish this path
  from output gather and pure copier tests. Test hooks and the admission-token
  extension remain private and test-only.
- `metadata::projected`, canonical schema digests and `row::ChargeProof` enforce
  exact layout/descriptor/charge-mode binding. Legacy full rows retain exact
  charge validation; layout 3 explicitly validates authoritative carried
  charges against retained-cell/omitted-column lower bounds, side limits and
  live aggregates. It does not claim to reconstruct omitted values or sign
  historical admission through checksums.
- `candidate::legacy`, migration and atomic installation validate full legacy
  history before independently copying live selected rows. Shared retirement
  controls and their drop order remain funded. Migration installs no full
  payload history and requires a fresh narrow base before continuation;
  matching layout-3 restore preserves validated narrow incremental history.
- Writer history, cuts, pending/rebase inputs and workers bind the compiled
  layout identity. OFF mutation advances the checked dirty tracker, rejects
  untrusted capture and forces a live base before reuse. Empty/reset,
  stale-result and failed-restore controls preserve these gates.
- Emitted-row arithmetic is checked after range/sequence validation and before
  delivery. Successful emission only assigns the checked total. The adjacent
  retained-charge arithmetic has an equivalent earlier state-admission proof;
  narrowing preserves those charges. Collector/gather execution failures can
  still leave delivered chunks and abort the job, as documented.
- Native sparse replacement uses selected columns/schema funding, preserves
  identity and logical charge, and rebinds pending upserts. The earlier suspected
  ingress-publication omission was disproven: the unchanged repair helper
  publishes progress after both successful sides. No product fix was made for
  that hypothesis.

### Style Issues

No unresolved convention violation found. The latest scoped Clippy/format
evidence and final function manifest cover the final Rust source. No dependency,
unsafe-code exception, agent-definition change or executable configuration is
introduced. The review adds only this report.

### Test Coverage

The focused controls cover full/projected output and logical charge parity,
shifted keys/time, nullable/nested/view/dictionary data, sliced and multiple
parents, constants/arithmetic/filter/multiple consumers, caller immutability,
independent buffers, sparse replacement and pending rebind. Recovery controls
cover direct/managed layout 3, full-plan rejection, authentic V1/V2 migration,
dead history, mandatory fresh bases, dirty deltas/rebase, OFF/reset, stale
identities, malformed descriptors/charges and cancellation/refusal cleanup.

Actual generic worker tests prove input-stage cancellation/abandonment,
result-stage cancellation, Attempt/Generation refusal and cancellation at
refusal, including responsive actor return, unchanged installed state/writer,
no output, omitted-parent release, retained credit, pending drain and final
pool-zero settlement. Native sparse coverage intentionally exercises `Shared`;
it does not claim generic `RestoredV2` dictionary sparse repair.

Whole terminal-manifest descriptor validation has authoritative call-path
review evidence; the new empty/terminal negative fixtures directly exercise
native and managed Join restore. No new end-to-end malformed terminal-manifest
runner test was executed. No specific additional local coverage blocker remains.

### Documentation Consistency

The specification and critique decisions are implemented. Runtime-envelope,
streaming-guide, CHANGELOG and the Python inventory consistently distinguish
semantic capability 1, inner layout 3/codec 2, project format 3 and manifest v3.
They explain carried-charge proof limits, independent physical funding, strict
legacy migration, incremental continuation and partial-emission failures.
Terminal recovery prose now matches the audited source path.

The analysis header identifies current evidence and provenance; its historical
pending notes are explicitly superseded by later results. Previous stopped
performance attempts and the original eight-target acceptance scope are not
overwritten or represented as completed by this increment.

### Verdict

**Approve — final local phase 2.2 implementation review only**, for commit
`e884adc362ac8f8d03c1364d0dfdb4e205407696` and the digest above.

This is an unpushed branch handoff, with no PR checks or remote reviews queried.
Required CI, combined 90% coverage, current-native Python/release verification,
phase-wide allocation/RSS and performance acceptance remain unproven. It is
not a merge-ready verdict, approval of the whole issue, or certification of any
of the original eight performance targets. No remote mutation was performed.
