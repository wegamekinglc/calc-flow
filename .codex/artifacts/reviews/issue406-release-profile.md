## Branch Review: issue 406 release compilation profile

**Branch:** `feature/issue406-release-profile`

**Reviewed head:** `b7c0abeb8e00848ee7f9dbb0b19be56ffef8b589`

**Base:** `bb142320062995aeb318ad406c901288537fd766`

**Scope:** five changed files; independent phase 0.5/D8 increment. The previously
reviewed delivery branch and its increment approvals are outside this change.

### Summary

The workspace release profile selects Thin LTO and one codegen unit. The ordinary
wheel compiler audit derives its expected flags from the original, acquisition-
sealed Cargo.toml, accepting either historical defaults or this exact profile.
It handles Cargo's valueless library LTO option and rejects ambiguous or
unsupported compiler options. This is a build policy change, not a speedup result.

The initial review found an alternative compiler-option spelling that could
bypass the profile check. The owner recorded a focused RED and corrected the
exact-shape guard; the reviewed final head resolves that finding.

### Build and Test Results

Owner evidence is under `target/issue406-release-profile-20261014/`. Passing
checks were not repeated by the reviewer.

- Cargo manifest validation: **Passed**, locked/offline/no-deps metadata in
  0.266s; no calc-flow compilation occurred.
- Existing release configuration/ABI pin/build-command/publication-gate controls:
  owner records **six Passed** in 0.031s. Their scope includes the release
  workflow's installed-wheel tests and artifact-verification dependencies;
  checking those declarations does not execute the artifact gates.
- Ordinary audit controls: **Passed**. `ordinary-profile-green.log` records
  eight then-current tests in 0.051s. `ordinary-profile-boundaries.log` adds
  three controls in 0.003s. The two complete default/Thin-LTO fixtures passed
  again after strict profile-table validation was extracted.
- The final corrected codegen-alias control and both complete ordinary audit
  fixtures: **three Passed** in 0.019s, in `codegen-alias-green.log`. These
  reuse existing registrations rather than adding a broad matrix.
- Real Rust 1.88 compiler flag witness: **Passed**, for two tiny dependency-free
  lib/cdylib fixture crates only. `cargo-profile-fixture.log` finishes in 3.71s.
  The reviewer independently re-parsed both recorded rustc invocations, checked
  the fixture manifest hash, and matched the resulting maps to
  `cargo-codegen-witness.json` and its expected crate maps.
- Targeted Ruff check/format: owner records **Passed** after the final source
  correction. Changed/new Python AST branch complexity is recorded at most 8.
- Whitespace/newlines, lockfile and project-schema no-drift: **Passed**. The
  reviewer independently matched all five current files to committed bytes,
  checked their whitespace/newlines, and confirmed Cargo.lock and the project
  schema remain byte-identical to the base.
- Calc-flow native release build, wheels, ABI/cross-platform compatibility,
  release artifact tests, full regression CI and combined 90% Rust line coverage:
  **Not run/unverified** for this head.
- New unresolved failures: **None** within the scoped evidence.

`ordinary-profile-red.log` preserves the expected original failure on Cargo's
real valueless `-C linker-plugin-lto` shape: 0.009s execution, 0.165s runner
elapsed. `codegen-alias-red.log` preserves the later expected false-acceptance
failure: one registration with four failing subcases in 0.004s. Both are followed
by the relevant passing evidence; neither is omitted or reclassified as green.

The reviewer ran only a bounded new-concern parser probe and pinned
`rustc --help`, not a build or test suite. The probe confirmed the then-current
Thin-LTO audit accepted overriding long-form codegen options and `-g`; the owner
supplied the corrective regression and GREEN. No runtime/performance test was
performed by the reviewer.

### Blocking Issues

None remain at the reviewed head.

**Resolved — `_reject_attached_options` / `_ordinary_codegen`:** the parser read
only separated `-C` options, allowing a legal trailing `--codegen opt-level=2`
or `--codegen=lto=off` to escape its expected-map comparison. `-g` similarly
changes debuginfo without a parsed `-C`. Pinned Rust help confirms those aliases.
The final guard rejects both long codegen forms and `-g`/`-O` before parsing,
consistent with the intentionally exact Cargo command shape. The regression
covers all four spellings, with both accepted ordinary profile fixtures still
passing. This closes the false-acceptance finding without broadening historical
build compatibility.

### Style Issues

None identified. The Python changes retain future annotations, use the standard
library without adding dependencies, and leave caller-owned build/seal mappings
read-only. Returned core/binding codegen dictionaries have independent identities.
The workspace safety lints, toolchain/version pins, package features and release
publication workflow are unchanged. Documentation additions introduce no tables
or formatting violations.

### Test Coverage

The focused controls distinguish default and Thin-LTO complete audit paths;
require matching original and acquisition manifest seals; reject changed
manifest bytes; reject invalid table types and seven unsupported profile
configurations; distinguish integer one from boolean true; and reject duplicate,
dangling, unknown or alternatively spelled codegen options. Existing source,
native/wheel identity, fresh compiler inventory, Python 3.13 cfg and Maturin
invocation gates remain in place.

Source review confirms the manifest must be in the original source inventory,
its acquisition seal must equal that recorded hash, and current bytes must
match before policy selection. Only an empty release policy or the exact
`lto = "thin"`, integer `codegen-units = 1` table qualifies. Nested package or
build overrides do not silently qualify. The library expects valueless
linker-plugin LTO; the final binding expects `lto=thin`; both expect one codegen
unit, optimization level three and the existing strip setting. Duplicate options
are rejected before metadata/extra-filename normalization.

The tiny compiler witness establishes actual Cargo flag shape on Rust 1.88. It
cannot establish calc-flow dependency linking, Python ABI behavior, platform
compatibility, whole-product peak resources or build duration. Those remain
production release/CI gates; no additional local native build is claimed.

### Documentation Consistency

The getting-started paragraph accurately scopes the workspace release profile
to release builds/wheels and Cargo's inherited bench profile, while ordinary
development/test profiles retain their defaults. The owner records 24 checked
local links/anchors. The analysis distinguishes manifest/configuration checks,
the tiny compiler witness and missing whole-product evidence. It preserves the
alias RED/GREEN and does not relabel historical defaults or stopped acquisitions.

No performance acceptance follows from these flags or the earlier investigation's
6–11% claim. A future comparison must build/seal baseline and candidate separately
under their actual source/profile identities. All eight original issue targets
and remaining phases remain active.

### Verdict

**Approve — final local source/configuration review of
`b7c0abeb8e00848ee7f9dbb0b19be56ffef8b589`.**

This does not certify production release artifact compatibility, current-native
Python integration, cold build/link resource use or duration, required CI and
cross-platform checks, combined 90% coverage, a performance improvement, merge
readiness or completion of issue 406. No remote status was queried and no remote
review, commit or source/test edit was performed by the reviewer. Only this
review artifact was written.
