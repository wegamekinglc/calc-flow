# Python packaging and PyPI release

[Documentation](README.md) / 5.5 Releases

The [Python package release workflow](../.github/workflows/release.yml) publishes
only `calc-flow-python`. Install it with `pip install calc-flow-python` and
import `calc_flow`. Studio is not uploaded to PyPI by this workflow.

## Version and artifacts

Use the planned release date as the calendar version `YYYY.M.D`, without zero
padding. Keep the workspace crate, Python core, Studio, frontend, native
dependency constraints, and `calc_flow.__version__` aligned. The tag must be
annotated, named `calc-flow-python-v<version>`, and point to the current `main`
head. The version must not already exist on PyPI.

The workflow builds the following artifacts for every manual or tagged release
run:

| Package            | Artifacts                                    | PyPI upload |
|--------------------|----------------------------------------------|-------------|
| `calc-flow-python` | Ten abi3 wheels and one source distribution  | Tagged runs |

Each of the five platform targets (Linux x86-64 and AArch64, macOS x86-64 and
ARM64, and Windows AMD64) gets a `cp39-abi3` wheel for Python 3.9–3.12 and a
`cp313-abi3` wheel for Python 3.13 and newer. The source distribution builds
against the installing interpreter. The artifact verifier accepts only the ten
versioned wheels and one versioned source distribution from the same workflow
run. Linux and Windows wheels are required together, alongside macOS wheels.
It checks each wheel's platform, ABI, metadata, and contents, checks the source
distribution's contents, and records artifact hashes.

## Workflow

The workflow:

1. Checks the package version and, on a tag run, confirms the annotated tag is
   at the current `main` head and the PyPI version is unused.
2. Compares the candidate commit with the previous release in three paired
   Python and Rust performance suites, checks security, and runs the continuous
   runtime soaks. These gates also run for manual rehearsals.
3. Builds ten core wheels and one source distribution. It inspects every wheel
   and installs and runs the native smoke check on both macOS targets, Windows,
   and Linux x86-64 for both ABI tiers. Linux AArch64 is cross-built and
   inspected without a native smoke run.
4. Installs the matching Linux wheel on Python 3.9–3.14 and runs the selected
   post-build unit tests. Python 3.9 also runs the optional JAX array unit test.
5. Verifies the eleven built artifacts and saves their hash manifest.
6. On a pushed release tag only, checks the downloaded artifacts against that
   manifest and publishes the Linux, Windows, and macOS wheels together with the
   source distribution through the `pypi` environment using Trusted Publishing.

Manual workflow dispatch exercises every gate without publishing. Crate and
Studio packaging remain in the repository CI rather than this Python-only
publication workflow.

The nearest reachable previous release tag supplies the paired performance
baseline. Before the first release tag exists, pass an ancestor commit's full
SHA as `initial-baseline` for the rehearsal and record the same SHA as
`Benchmark-Baseline: <sha>` in the annotated release tag. The optional
`allow-dependency-drift` input records an acknowledged dependency change; it
does not permit incomparable benchmark evidence.

## Trusted Publisher

Configure a PyPI Trusted Publisher for `calc-flow-python` with this repository,
the `release.yml` workflow filename, and the `pypi` GitHub environment. The
publication job requests short-lived OIDC credentials with `id-token: write`.
No long-lived PyPI token or `skip-existing` option is used.

## Release procedure

1. Update the aligned version surfaces and changelog for the planned date.
2. Run `Python package release` manually from the reviewed `main` commit.
   Confirm the performance, security, soak, artifact verification, cross-platform
   smoke, and installed-wheel unit tests all finish successfully.
3. Tag that same `main` commit and push the annotated tag:

   ```bash
   git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>"
   git push origin calc-flow-python-v<version>
   ```

4. Approve the `pypi` deployment when prompted and confirm PyPI lists the ten
   wheels and source distribution for that version.

PyPI versions and files are immutable. If an upload is incomplete, resolve the
problem and use a new calendar version for the next attempt.
