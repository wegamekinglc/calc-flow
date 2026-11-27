# Python packaging and PyPI release

[Documentation](README.md) / 5.5 Releases

The [Python package release workflow](../.github/workflows/release.yml) builds
two mutually exclusive distributions at the same calendar version.
`calc-flow-python` is the portable default. `calc-flow-python-v3` is the opt-in
for x86-64-v3 processors when published. Both provide the same `calc_flow` import
package.

Install exactly one distribution in a clean environment with
`pip install calc-flow-python` or `pip install calc-flow-python-v3`. Do not
install both or install one over the other. Importing `calc_flow` when both
distributions are installed raises an `ImportError` instead of silently using
overwritten files. The v3 distribution requires x86-64-v3 CPU support, including
AVX2; there is no import-time native-module selection. See
[installation](getting-started.md#install-published-packages) for the package
commands and supported platforms.

## Version and artifacts

Use the planned release date as the calendar version `YYYY.M.D`, without zero
padding. Keep the workspace crate, both Python distributions, native dependency
constraints, and `calc_flow.__version__` aligned. The tag must be annotated,
named `calc-flow-python-v<version>`, and point to the current `main` head. The
version must not already exist on PyPI. Release tags must keep this exact
unpadded numeric shape: the scheduled benchmark suite anchors its baseline on
the newest reachable `calc-flow-python-v*` tag, and a tag deviating from that
shape fails the scheduled gate closed.

The workflow builds the following artifacts for every manual or tagged release
run:

| Package               | Artifacts                                      | PyPI upload            |
|-----------------------|------------------------------------------------|------------------------|
| `calc-flow-python`    | Thirty abi3 wheels and one source distribution | Tagged runs            |
| `calc-flow-python-v3` | Eighteen abi3 wheels; no source distribution   | Tagged runs by default |

The portable distribution covers five platform targets: Linux x86-64 and
AArch64, macOS x86-64 and ARM64, and Windows AMD64. The v3 distribution covers
only the three x86-64 targets: Linux x86-64, macOS x86-64, and Windows AMD64.
There are no v3 AArch64 wheels and no v3 source distribution.

Every target in both matrices gets wheels with explicit `cp39`, `cp310`,
`cp311`, `cp312`, `cp313`, and `cp314` Python tags. The workflow compiles two
native abi3 tiers per target: Python 3.9–3.12 share the 3.9-compatible binary,
and Python 3.13–3.14 share the 3.13-compatible binary. The `wheel tags` command
creates the additional version-specific wheel files and updates their WHEEL
metadata and RECORD. The portable source distribution builds against the
installing interpreter.

Only v3 wheel jobs set `RUSTFLAGS=-C target-cpu=x86-64-v3`. Portable x86-64 and
AArch64 build settings remain unchanged. Every portable or v3 wheel contains the
`calc_flow` package, exactly one `calc_flow._native` abi3 module, metadata/SBOM,
and license, with no second native module or runtime selector.

The core artifact verifier requires all thirty versioned wheels and one
versioned source distribution from the same workflow run. The v3 verifier
separately requires all eighteen v3 wheels and rejects a v3 source distribution
or any other extra artifact. Both verifiers check wheel platforms, ABI, Python
and distribution metadata, and contents, then record separate artifact-hash
manifests. The core verifier also checks the source distribution's contents.
V3 verification disassembles every Linux v3 wheel and requires vector
instructions in at least one designated P2/P4 numeric loop.

## Python dependency metadata

`[project.dependencies]` in `pyproject.toml` supplies each wheel's runtime
`Requires-Dist` metadata. The default Python dependencies are `pyarrow` and
`typing-extensions`; NumPy and JAX are optional extras, and Maturin is a build
requirement rather than a wheel runtime dependency. Project-document validation
uses the native Rust runtime and does not require Pydantic.

After changing these declarations, regenerate `benchmarks/requirements.lock`
with the workflow's compile command and review the separately maintained
`benchmarks/risingwave/requirements.lock` support environment. The local root
`uv.lock` is ignored by Git and is not a release artifact.

## Workflow

The workflow:

1. Checks the package version and, on a tag run, confirms the annotated tag is
   at the current `main` head and the core PyPI version is unused.
2. Builds ten portable and six v3 native wheel binaries, creates thirty portable
   and eighteen v3 explicitly tagged wheels, and builds one portable source
   distribution.
3. Installs each version's matching portable Linux wheel on Python 3.9–3.14 and
   runs the selected package unit tests. Python 3.9 also runs the optional JAX
   array unit test. A separate clean-install smoke test imports the v3 Linux
   CPython 3.13 wheel.
4. Verifies the thirty-one core artifacts and eighteen v3 artifacts through
   separate jobs, saves their hash manifests, and inspects the Linux v3 wheels
   for the required numeric vector instructions.
5. On a pushed release tag only, publishes the distributions through independent
   jobs. Core publication requires core artifact verification and the portable
   installed-wheel tests; v3 publication requires v3 artifact/vector verification
   and its clean-install smoke test. Each job downloads only its distribution's
   artifacts, checks them against that distribution's manifest, and publishes
   through the `pypi` environment using Trusted Publishing. The annotated tag
   can skip the v3 upload with the `Publish-Python-v3: false` trailer.

Manual workflow dispatch builds and verifies all artifacts and runs the
post-build unit tests and v3 smoke test without publishing. Benchmark, soak,
audit, documentation, and crate work are outside this Python publishing
workflow.

## Trusted Publisher

Configure a PyPI Trusted Publisher for each of `calc-flow-python` and
`calc-flow-python-v3` with this repository, the `release.yml` workflow filename,
and the `pypi` GitHub environment. Each publication job requests short-lived
OIDC credentials with `id-token: write`. No long-lived PyPI token or
`skip-existing` option is used.

## Release procedure

1. Update the aligned version surfaces and changelog for the planned date.
2. Run `Python package release` manually from the reviewed `main` commit.
   Confirm the forty-eight wheels, portable source distribution, both artifact
   verification jobs, v3 vector inspection, portable installed-wheel unit tests,
   and v3 clean-install smoke test finish successfully.
3. Tag that same `main` commit and push the annotated tag:

   ```bash
   git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>"
   git push origin calc-flow-python-v<version>
   ```

   To publish only the portable distribution, add a second message paragraph
   when creating the tag.

   ```bash
   git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>" \
     -m "Publish-Python-v3: false"
   ```

4. Approve the `pypi` deployments when prompted. Confirm PyPI lists thirty
   portable wheels and one source distribution. If the release includes the
   v3 distribution, confirm it lists eighteen wheels.

The publication jobs are independent; a failure in one does not roll back files
already uploaded for the other. PyPI versions and files are immutable. If either
upload is incomplete, resolve the problem and use a new calendar version for the
next attempt.
