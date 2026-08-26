# Python packaging and PyPI release

Calc Flow publishes two Python projects from the repository release workflow.
The native `calc-flow` project and the pure-Python `calc-flow-studio` project
share one version, but use separate PyPI Trusted Publishers so each OIDC token
has one package boundary.

## Release contract

The workflow produces the following Python artifact set for every manual or
tagged release run:

| Package            | Published artifacts                              | Python compatibility |
| ------------------ | ------------------------------------------------ | -------------------- |
| `calc-flow`        | Five abi3 wheels and one source distribution     | CPython 3.13+        |
| `calc-flow-studio` | One `py3-none-any` wheel with built React assets | Python 3.13+         |

The five core wheels cover this matrix:

| Operating system | Architecture | Required platform family    |
| ---------------- | ------------ | --------------------------- |
| Linux            | x86-64       | `manylinux_2_28_x86_64`     |
| Linux            | AArch64      | `manylinux_2_28_aarch64`    |
| macOS            | x86-64       | `macosx_*_x86_64`           |
| macOS            | ARM64        | `macosx_*_arm64`            |
| Windows          | AMD64        | `win_amd64`                 |

Every core filename must use `cp313-abi3`. The release verifier also checks
the wheel's internal tags, native extension, package metadata, Apache-2.0
license, exact version, platform family, source-distribution contents, Studio
assets, and Studio's dependency on the matching v4 core package.

## Local packaging rehearsal

Install the repository development dependencies, then run the cross-platform
helper from the repository root:

```bash
uv sync --extra dev
python scripts/build_python_release.py --clean
```

The helper builds the current platform's core wheel, the core source
distribution, the frontend and Studio wheel, then runs the existing artifact
content inspectors. It writes only beneath `target/python-release/`. Without
`--clean`, it refuses a nonempty output directory so stale artifacts cannot be
mistaken for the current build.

The local helper is for diagnostics and release rehearsal. Its single-platform
output is not the complete PyPI release set and must not be uploaded as an
official release.

To validate a combined seven-artifact directory produced by CI or assembled
for diagnosis, run:

```bash
python scripts/verify_python_release.py --dist-dir <release-directory>
```

The command prints a stable SHA-256 manifest with paths relative to the release
directory. Add `--tag v<version>` to enforce the tag contract. The CI tag path
also uses `--check-pypi`, which fails if either project version already exists.

## CI publication flow

The `Release artifacts` workflow in `.github/workflows/release.yml` implements
the publication boundary:

1. Validate aligned Rust, Python, Studio, and binding versions. A tag run must
   use an annotated `v<version>` tag pointing at the current `main` head, and
   both PyPI versions must still be unused.
2. Pass the exact-head performance, security, soak, packaging, crate, and audit
   gates.
3. Build and smoke-test all five core wheels, the source distribution, and the
   Studio wheel in their platform jobs.
4. Download the seven artifacts into one directory, validate the complete
   matrix and metadata, and record their exact bytes in
   `release-manifest.txt`.
5. Re-download the verified bundle and check every SHA-256 entry immediately
   before upload.
6. Publish `calc-flow` first through the `pypi` environment. Only after that
   succeeds, publish `calc-flow-studio` through `pypi-studio`.

Manual workflow dispatches are build-only rehearsals. Only `v4.*` tag runs can
reach the publication jobs. No API token or `skip-existing` behavior is used.

## One-time Trusted Publisher setup

Create these GitHub environments and configure one Trusted Publisher on each
existing PyPI project:

| PyPI project       | GitHub owner   | Repository  | Workflow filename | GitHub environment |
| ------------------ | -------------- | ----------- | ----------------- | ------------------ |
| `calc-flow`        | `wegamekinglc` | `calc-flow` | `release.yml`     | `pypi`             |
| `calc-flow-studio` | `wegamekinglc` | `calc-flow` | `release.yml`     | `pypi-studio`      |

Require manual deployment approval on both GitHub environments when the
repository plan supports it. The workflow requests short-lived OIDC credentials
with `id-token: write`; do not add long-lived PyPI API tokens.

## Release procedure

1. Choose a new final `X.Y.Z` version that does not exist for either PyPI
   project. Update all version surfaces and the changelog together, including
   the workspace, PyO3 binding constraints, core Python package, Studio,
   frontend package and lockfile, OpenAPI, and generated types.
2. Run the full verification groups in `AGENTS.md`, including the release
   helper unit tests and artifact inspectors.
3. Run `Release artifacts` manually from `main`. This does not publish. Confirm
   that the five core wheels, core source distribution, Studio wheel, and
   verified SHA-256 bundle are present.
4. Tag that reviewed `main` commit and push only the tag:

   ```bash
   git tag -a v<version> -m "Release calc-flow <version>"
   git push origin v<version>
   ```

5. Approve the `pypi` and then `pypi-studio` deployments after reviewing the
   workflow gates and manifest.
6. Confirm PyPI lists all six `calc-flow` files and the one
   `calc-flow-studio` wheel. Install both exact versions in clean environments
   on representative platforms and run the core and Studio smoke checks.

PyPI versions and files are immutable. If any package upload is incomplete,
fix the release problem, increment the shared version, and rerun the complete
process; never try to repair the old version with a partial or
skip-existing upload.
