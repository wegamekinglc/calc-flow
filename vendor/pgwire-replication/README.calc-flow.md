This is a library-only copy of the crates.io `pgwire-replication` 0.4.0 release.
`provenance.json` records its verified archive checksum, upstream revision and
the original and shipped SHA-256 of every Rust source file. Executable Rust
sources are unchanged; trailing whitespace in one rustdoc comment is removed.

The sole dependency adaptation is `webpki-roots = "0.26"` to `"=1.0.10"`.
The old 0.26.11 crate only reexports v1's trust anchors, so this preserves the
same Rust types, trust store, TLS negotiation, authentication, replication and
feedback behavior while removing the legacy advisory-matched package. Upstream
example/benchmark targets and development dependencies are omitted from the
library manifest; the whitespace-normalized upstream manifest is retained as
`Cargo.toml.orig`. Manifest/readme whitespace also follows repository checks.

This local backport is selected with `[patch.crates-io]`; it introduces no Git
or alternate-registry source and changes no audit/deny rule. Remove it when an
upstream release directly supports roots v1, with the connector TLS regressions
and source-distribution checks retained. Do not edit the vendored Rust sources
as part of routine connector work.
