# Rust crate release

Publish only `calc-flow` to crates.io. The Python binding remains unpublished as
a Rust crate; PyPI releases retain their existing workflow and tag namespace.
The package keeps the workspace's calendar version, currently `2026.11.2`, and
MSRV `1.94.0`. Do not independently bump the Rust version: workspace crates,
Python metadata, and exact binding dependencies must stay aligned.

## Rehearse

From a clean checkout at the intended release commit:

```bash
python scripts/verify_rust_release.py --version-only
python -m unittest scripts.test_verify_rust_release
cargo package -p calc-flow --locked
cargo publish -p calc-flow --locked --dry-run
python scripts/verify_rust_release.py --consumer
version=$(python scripts/verify_rust_release.py --version-only)
cargo check --manifest-path "target/package/calc-flow-${version}/Cargo.toml" --target-dir target --locked --all-features
cargo audit --file "target/package/calc-flow-${version}/Cargo.lock" \
  --ignore RUSTSEC-2026-0176 --ignore RUSTSEC-2026-0177 --ignore RUSTSEC-2026-0235
```

Cargo creates `target/package/calc-flow-<version>.crate` and verifies its extracted
library. The inspector checks the normalized crates.io manifest, license,
README, sources, examples, and benches, while excluding repository integration
tests. The consumer runs the packaged README quickstart in its own
workspace under `target/`, using the extracted distribution and registry
dependencies. Its `[workspace]` boundary prevents it from inheriting repository
patches. No Python process or local connector crate is needed at runtime.

`cargo publish --dry-run` verifies packaging without uploading or proving
ownership of a crates.io name. PRs and manual runs of `release-rust.yml` perform
the rehearsal and audit. The standard CI retains full regression, coverage,
cross-platform, and workspace supply-chain checks.

## Registry dependencies and availability

Cargo removes workspace patches from published crates; native consumers resolve
registry Arrow IPC and Parquet rather than the repository's vendored copies. See
[Cargo's packaging rules](https://doc.rust-lang.org/cargo/commands/cargo-package.html)
for the removal of workspace overrides. Keep the existing Python source
distribution patches intact and audit the actual packaged registry lockfile.

On 2026-11-08, the packaged lockfile resolved `zstd 0.13.4`, `zstd-safe 7.4.0`,
and `zstd-sys 2.1.2`. Audit passed with the refreshed RustSec database at
`309392b18f6b17022f82c701f300433126901068` (2026-11-06). These releases contain
the compatible zstd security backports, as recorded by
[RUSTSEC-2026-0391](https://rustsec.org/advisories/RUSTSEC-2026-0391.html),
[RUSTSEC-2026-0398](https://rustsec.org/advisories/RUSTSEC-2026-0398.html), and
[RUSTSEC-2026-0403](https://rustsec.org/advisories/RUSTSEC-2026-0403.html).
Refresh the advisory database when rehearsing a release; an older offline
database may lack these patched-version ranges. No new advisory exemptions or
codec changes are needed. Consumers should keep their resolved dependencies
updated and audit their own lockfiles as well.

The crates.io API returned 404 for `calc-flow` during preparation; that is no
guarantee of reservation or publish permission. Recheck the name and exact
version immediately before release. Registry ownership and credentials have not
been tested by a dry-run.

## Publish after review

The release coordinator verifies required CI checks and the registry audit are
green, and the reviewed commit is the current `main` head. Configure
the GitHub `crates-io` environment with required reviewers and a crates.io API
token in its `CARGO_REGISTRY_TOKEN` secret. The token must have publish permission
for `calc-flow`; first publication also requires permission to create the name.
Never put tokens in source, logs, command arguments, or issue comments.

An authorized coordinator creates and pushes an annotated core release tag:

```bash
version=$(python scripts/verify_rust_release.py --version-only)
git tag -a "calc-flow-v${version}" -m "Release calc-flow ${version}"
git push origin "calc-flow-v${version}"
```

The workflow validates the namespace, shared version, annotated tag, and current
`main` commit; packages the crate; runs the external consumer; and audits the
registry lockfile. Only a tag push can reach the environment-gated upload job.
Manual dispatch and PR runs never publish. After upload, confirm the crates.io
version and docs.rs build, then run the quickstart in a fresh Cargo application
with a registry `calc-flow` dependency. Crates.io versions are immutable: use a
new aligned version for a replacement release.
