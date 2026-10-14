# Issue 406: release compilation profile

Base: `bb142320062995aeb318ad406c901288537fd766`. The independent
`feature/issue406-release-profile` branch addresses phase 0.5/D8. It remains
local; the previously reviewed delivery branch stays frozen. All eight issue
acceptance rows and remaining phases remain active.

## Change and provenance

The workspace release profile enables Thin LTO and one codegen unit. Release
wheels and ordinary release builds use these defaults; Cargo's bench profile
inherits them. Development/test profiles, pinned Rust/Maturin/actions, package
versions/features, release publication gates, lockfile and project schema remain
unchanged. This is a build configuration choice, not a measured speedup.

The maintained ordinary-wheel audit previously hardcoded default codegen flags
and could not parse Cargo's valueless `-C linker-plugin-lto` option. It now
selects exactly one of two configurations from the sealed source Cargo.toml:
historical defaults, or the exact Thin-LTO/single-unit table. Both the original
source inventory and acquisition seal must match the manifest bytes. Each
configuration still requires exact core/binding compiler parameters, Python ABI
cfgs, complete fresh compiler inventory, native/wheel identities and Maturin
invocation. Other profile policies, duplicate/incomplete codegen options,
unsealed manifests and changed source are refused.

This does not relabel or resume historical/stopped acquisitions. Their original
source/profile and scope remain intact. Future acceptance must build and seal
current baseline and candidate separately and include this manifest in provenance.

## Focused evidence

All retained logs are under `target/issue406-release-profile-20261014/`.

- Cargo accepts the workspace manifest with locked/offline/no-deps metadata;
  command finishes in 0.266s. No calc-flow compilation occurs.
- Six existing release command/version/pin/workflow gates pass (0.031s test
  time), including installed-wheel unit-test and artifact-verification dependencies.
- `ordinary-profile-red.log`: the new Thin-LTO audit control fails as expected
  on Cargo's valueless linker-plugin option; runner 0.165s, test 0.009s. The
  recorded compiler fixture establishes that this is a real Cargo flag shape.
- `ordinary-profile-green.log`: all eight then-current reuse controls pass
  (runner 0.165s, test 0.051s), including default and Thin-LTO complete audit
  fixtures. After extracting strict profile-table validation, both affected
  integration controls pass again (0.008s).
- `ordinary-profile-boundaries.log`: three new controls pass (runner 0.115s,
  test 0.003s), covering source seals/tampering, seven unsupported profile
  configurations, and duplicate/dangling/unknown codegen options.
- Specialist review found that long codegen aliases and debug/optimization
  shortcuts could bypass the short-form parser. `codegen-alias-red.log` records
  four expected false-acceptance failures (one test, four subcases, 0.004s).
  The exact Cargo-shape audit now refuses `--codegen`, `--codegen=...`, `-g`
  and `-O`; `codegen-alias-green.log` records the alias control and both
  default/Thin-LTO complete audit fixtures passing after this correction.
- `cargo-profile-fixture.log` and `cargo-codegen-witness.json`: real Rust1.88
  compilation of two tiny dependency-free fixture crates succeeds. The library
  receives linker-plugin LTO and one codegen unit; the cdylib receives Thin LTO
  and one codegen unit. Parsed actual flags exactly match the validator's
  expected maps. This proves flag shape only, not the calc-flow wheel or ABI.
- Targeted Ruff check passes; Ruff formats the two changed Python files.
  Changed/new Python functions have measured AST branch complexity at most 8.
  Whitespace, lockfile and project-schema no-drift checks pass.

No calc-flow native wheel, cross-platform release artifact, allocation/RSS,
paired timing or full regression run was executed. Cold build duration and
whole-product LTO link/resource compatibility remain for CI and separately
accounted release builds. The existing historical investigation's6–11% claim
is not evidence for this branch. The documentation specialist updated only
`docs/getting-started.md` under build-from-source, covering the configured release
profile and its wheel/bench applicability. All 24 inspected local links/anchors,
whitespace and final newline checks pass; no changelog entry qualifies. Final
[specialist review](../reviews/issue406-release-profile.md) is Approve at
`b7c0abeb8e00848ee7f9dbb0b19be56ffef8b589`. The narrow compiler-alias correction
resolves its sole concrete finding. This is a local source/configuration approval;
the wheel, cross-platform, CI, coverage and performance gates above remain pending.
