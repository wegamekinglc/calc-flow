This is the complete, checksum-verified crates.io `arrow-ipc` 59.2.0 release.
`provenance.json` records the archive checksum and SHA-256 of every unchanged
Rust source file. Apache license and notice files are retained.

The sole adaptation is the normal `zstd` dependency minimum, raised from
0.13 to 0.14.3. This backports the dependency-only Arrow upgrade in
https://github.com/apache/arrow-rs/commit/2e3a24304ce22ac20947b8e956b55af1049dc2bd
and requires the published security fixes in zstd 0.14.3, zstd-safe 8.2.0 and
zstd-sys 2.1.2. No codec feature, Arrow type, format version, or executable
source is removed or changed. DataFusion 55 and the existing Arrow 59
checkpoint identities remain in place.

This copy is selected with the workspace `[patch.crates-io]` and included
in Python source distributions. It introduces no Git/alternate registry
source and changes no advisory, license, or source policy. Remove the patch
when an upstream release compatible with the current engine accepts the
safe dependency chain. Keep codec compatibility and distribution checks
when doing so. The workspace patch applies to builds from this repository
and its Python source distribution; Cargo patches are not inherited by
independent consumers of a published Rust crate.
