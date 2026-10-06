# A4 actual matched release build - Provenance Review

## Verdict and scope

**Approve the exact build-only artifacts below for a separately granted
functional preflight.** No build provenance blocker was found. This review
does not grant native execution or quiet measurements, establish functional
correctness, prove performance/FR16 target attainment, or approve merging.

Entry `target/issue363-asof-columnar-admission-perf/release-v1/build-only-ready.json`
SHA-256:
`73a86e218e705ad8a6545e6667a225063c97d7be16341011006bd43fe45d6730`.
Baseline side seal:
`3f1d79b167b99986db2ffdf47f0fe58f5a937b163632b99f582ad821f9917f5f`.
Candidate side seal:
`c365f63a9831d3331ed151b289821f7ce2fbbfb9797a185ce72686e9b653524a`.
Verified receipt:
`1d712a512e57fdd4d01a88ef9fffe75cbf2ed5eae1d7e7c76337bed97891b980`.

The [source approval](stream-asof-columnar-admission.md),
[protocol](stream-asof-columnar-admission-performance-protocol.md) and
[corrected tooling approval](stream-asof-columnar-admission-performance-tooling-v2.md)
remain separate gates. Their files and the approved preparation were unchanged.
This review wrote only this document and repository
`target/issue363-a4-build-review/` proof. Source, author tooling, immutable
seals, closures, other worktrees and other agents' edits were preserved.
No remote operation or new CI snapshot was needed for this local artifact gate.

## Summary

Both sides now have a matching freshly compiled default-feature Rust core
and the unchanged public admission probe linked against its owned core,
dependencies and Rust 1.88 sysroot. The E2E comparator uses the previously
sealed A3 wheel and a newly built A4 wheel. These are distinct comparator
surfaces; the A3 wheel was not substituted for the fresh A3 Rust core.

## Build and verification results

- Author's baseline/core, baseline/probe, candidate/core, candidate/probe and
  candidate/wheel builds have strict integer exit code zero. Actual core
  logs show compilation from the respective exported source directories;
  the candidate wheel log shows core, Python binding and connector compilation
  from candidate source. The five recorded build owner PIDs are gone.
- Reviewer independently hashed **3,530 actual files** with stdlib only,
  including both complete closures, probes, wheels/native modules, source
  archives/files, sysroot originals and tooling identities. No binary was
  executed, loaded or imported. The complete proof records observed hashes
  and filesystem identities rather than trusting asserted fingerprints.
- Each side has **537** unique complete closure files and **49** captured
  directory identities. Actual inventory equals the manifest with no extra
  file except the manifest itself. Files are regular, not symlinks, read-only,
  and have exactly one link. Byte totals are **1,394,027,557** baseline and
  **1,394,214,153** candidate. All hashes and recorded directory/file stat
  identities match the prequiet verified receipt.
- Actual Cargo JSONL contains **279** events per side: 251 compiler artifacts,
  27 build-script results and successful build completion. All **278** owned
  receipt rows match those actual events' package/target/kind/features and
  native-link metadata. The **250** non-core compiler profile records also
  match across sides. The comparator core differs as intended.
- **Functional/native/E2E tests and measurements:** not run by reviewer;
  the build-only readiness receipt records no such execution and zero samples.
  Python application/Studio product suites and coverage gates are not established
  by compilation or this provenance review.

Reviewer's actual command:

```bash
python target/issue363-a4-build-review/verify_build.py \
  > target/issue363-a4-build-review/verification.log 2>&1
```

Session 78970 completed with exit **0**, tool chunk `0f5dc9`; the complete
redirected summary was read in chunk `6d2774`. Additional stdlib receipt/profile
matching exited **0**, chunk `f9506f`, and wrote `receipt-match.json`.

The initial reviewer script exited one after successfully comparing the exact
277 normalized rows: its digest used spaced JSON rather than the published
compact JSON encoding. The retained `verification-initial-json-encoding.log`
records that helper assertion. A focused stdlib check identified the compact
encoding, after which the corrected verifier passed. No source or author
artifact was modified, and this is not reported as a native build failure.

## Provenance and comparability findings

### Exact source and actual fresh compilation

Baseline export is `764843e634ae1a1da7a5b010095c3017349a25f4`, production
`9983cfd0c3d096da7a60c3a96ae03b785f5ae955`, source-content digest
`fe7fdb4506b9e4e348afc29401e5b6555cb09107ecbce461e3e3d2feb42922bf`.
Candidate export is `fd73c1a91bb9d1b017c9de471a99789f867c3830`, production
`e3e3e11f7982ae7144d2b760b4b77bdddc20c060`, source-content digest
`8642e9b717d58b31b10cef518d0f82f7765e29bb65f8ad67146d3260d4f57b42`.

The reviewer checked actual exported Git HEAD/clean state, production-path
equality with the respective reviewed production commit, independently
recomputed the source digest, and compared **1,139** baseline / **1,143**
candidate archive files with current source. Git archive commit headers and
pre/post-build source identities agree. Neither source directory contains a
generated `python/calc_flow/_native*.so`.

The actual core Cargo artifact names, manifest and crate source paths point
to each export. `fresh: false` in Cargo means the artifact was compiled in
this build rather than reused from cache. Both events have features `[]`,
Rust 2024 library target, opt-level 3, no debug assertions, debuginfo zero,
overflow checks false and test profile false. Commands use
`cargo +1.88.0 build --release --locked ... -p calc-flow --lib`, jobs 2 and
empty Rust flags. Default core features are empty; neither side adds a
feature override. The shared mutable cache path is an input build cache;
subsequent probes link the independent copied closures, not that cache.

### Actual probe and dependency closure

The approved probe source remains
`1e3ac216b8d91921638545d86d8637c5fb3e1b6f9c81831e9795097460baa220`.
Actual commands use `rustc +1.88.0`, edition 2024, opt-level 3 and warnings
deny. Their explicit `--extern calc_flow=.../closure/libcalc_flow.rlib`, other
three externs, dependency/native link directories and `--sysroot` all point
inside that side's owned closure. The recorded compile commands, actual
linked-core session receipts and current core hashes agree.

The dependency canonical multiset has **277** non-core records per side and
independent recomputed digest
`48939bc229d11cfe8cc754f58886b58761254ff78c096666fbd19b20617bb41e`.
Comparison preserves package, target/kind, features, native libraries and
artifact name/content/size, including multiplicity. Encounter order,
freshness and side-specific paths are normalized; content/features are not
discarded to force equality. Raw encounter receipts remain distinct.

Both use the same recorded Rust 1.88.0 toolchain, Cargo lock, Python 3.13.9,
Maturin 1.15.0 and PyArrow 24.0.0 environment. The actual rustc executable
hash matches its receipt; all **37** copied sysroot entries match the recorded
Rust 1.88 original files and each side's closure. The full comparison
fingerprint is equal across sides. Quiet stat guards still rely on immutable
owned files and are not continuous cryptographic tamper detection.

### Actual wheel and frozen E2E source

Baseline wheel is the owned byte-identical A3 artifact with original release
manifest, export/source digest and native hash intact. Candidate wheel was
actually built from the A4 export using release/locked Maturin and
`pyo3/abi3-py313`; the normal connector-file default remains the wheel surface,
separate from the empty-feature direct core. Each wheel's **45** package
members match the extracted owned package. The sole native member is
`calc_flow/_native.abi3.so`, and its zipped bytes match the owned native file.

Native origin was resolved with stdlib `PathFinder` only, without executing
the loader or importing the package. Both origins are their owned extracted
wheel directories, with no source-tree native module. The local Linux wheel
tag warning remains accurately recorded; these artifacts are for local
evidence, not a publication approval.

The original A3 worker and paired driver hashes match their frozen preparation:
worker `51a5f50a23895b87c8c01045044762390bbc0c1ff7654f9a8a09879aa38ce47f`,
driver `f597597daf678b51c003ac1c09595a354db0ae874ca8e9fbe93b436f3d5f2e5e`.
The original eleven-case E2E work/timer is unchanged. Actual loaded dependency
and thread observations still belong to future functional preflight.

## Exact artifact identities

- Baseline core: `9ad8d447aee7bf00cf3d991b37e3b0093ed2217c7a8371f7f5a057cb39cef30f`.
- Candidate core: `8da7d6734367ee7947566a4ebc04e91afc07ce3b0c2726e2445b6f240ec181e1`.
- Baseline probe: `ca1744a8f9d8d8fd43667c0633d949a8fe50dac5307de7bd90088d497b5aab8b`.
- Candidate probe: `0987c4b744d2b94c133f2266744d0703e6cee761c9730a85c8f93ebf7fb1bbb7`.
- Baseline wheel: `ab38ee224ee306e1ce8e6524419a6f45cd34170612a80d9e7dc149f327b07e69`.
- Candidate wheel: `43872358a8c41b202dc96d70d6a222fc33567fbdad897ccb8d4d0b3ef251f8b2`.
- Baseline native: `e32eae86ab6a6b51445ba4a84ec6893e7ba4631d26c39bb5f57e20e6b4f0e889`.
- Candidate native: `2440a32261f4c536928538af2d4c5567821f6271159e1bfc8a0324aac4ea2bf6`.

## Blocking issues and remaining gates

**None in this exact build provenance.** Full post-timer state/output,
checkpoint bytes, live/restored continuation, accepted-prefix/cancellation,
cleanup and threshold/shape oracles remain unexecuted. Public status/RSS cannot
prove private pool refund, worker phases or managed delivery. The left-only
1M diagnostic remains sixteen callbacks (15 × 64,000 + 40,000); it does not
establish the complete FR16 output target of 100 ms.

A separate functional grant must precede native preflight. Complete same-shape
10k → 100k → 1M predecessors, resource screens and raw reject/failed attempts
remain mandatory. Measurements then require their own quiet grant and final
independent paired evidence review. Build success is not a substitute for
any of those gates. Reviewer-owned running processes at handoff: **0**.
