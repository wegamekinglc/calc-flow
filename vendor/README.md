# Transport dependency patches

The workspace patches two published transport crates to require
`webpki-roots = "1.0.10"`, addressing
[RUSTSEC-2026-0381](https://rustsec.org/advisories/RUSTSEC-2026-0381.html).
Their published manifests constrain this dependency to 0.26, which the advisory
still rejects after its re-exported 1.x bundle is upgraded. Version 0.26.11 is a
compatibility re-export of the same `TLS_SERVER_ROOTS` API from 1.x, so these
patches change only the dependency requirement and preserve the transport code.
Trailing whitespace is normalized to satisfy the repository's diff check.

Each directory contains the published crates.io archive, including its licenses
and upstream Git revision in `.cargo_vcs_info.json`. The upstream `Cargo.lock`
is omitted; builds use the workspace lockfile. `Cargo.toml.orig` remains the
upstream reference. The only edit to each normalized `Cargo.toml` is the
`dependencies.webpki-roots.version` requirement.

| Crate              | Version | Published archive SHA-256                                        |
|--------------------|---------|------------------------------------------------------------------|
| pgwire-replication | 0.4.0   | 62dd737cbc9ef6eb7d0702a80b5a53ca974088a4edf04da40bc2b946747ca810 |
| tokio-tungstenite  | 0.30.0  | 17a073bfed563fa236697a068031408a93cd9522e08abf9933ead3e73411bd71 |

Remove a patch and its directory once a compatible published upstream release
requires a patched 1.x root bundle. Keep the crates outside the workspace member
list, as with the original registry dependencies. The audit, license and source
policies continue to check their resolved dependencies without new exceptions.
Maturin explicitly includes `vendor/**/*` in source distributions so their patch
paths also resolve when building outside a repository checkout. Upstream test
fixtures remain repository-only and are excluded from source distributions.
