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
2. Builds ten core wheels and one source distribution.
3. Installs the matching Linux wheel on Python 3.9–3.14 and runs the selected
   package unit tests. Python 3.9 also runs the optional JAX array unit test.
4. Verifies the eleven built artifacts and saves their hash manifest.
5. On a pushed release tag only, checks the downloaded artifacts against that
   manifest and publishes the Linux, Windows, and macOS wheels together with the
   source distribution through the `pypi` environment using Trusted Publishing.

Manual workflow dispatch builds artifacts and runs the post-build unit tests
without publishing. Benchmark, soak, audit, documentation, crate, and Studio
work are outside this Python publishing workflow.

## Trusted Publisher

Configure a PyPI Trusted Publisher for `calc-flow-python` with this repository,
the `release.yml` workflow filename, and the `pypi` GitHub environment. The
publication job requests short-lived OIDC credentials with `id-token: write`.
No long-lived PyPI token or `skip-existing` option is used.

## Release procedure

1. Update the aligned version surfaces and changelog for the planned date.
2. Run `Python package release` manually from the reviewed `main` commit.
   Confirm that the wheels, source distribution, artifact verification, and
   installed-wheel unit tests finish successfully.
3. Tag that same `main` commit and push the annotated tag:

   ```bash
   git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>"
   git push origin calc-flow-python-v<version>
   ```

4. Approve the `pypi` deployment when prompted and confirm PyPI lists the ten
   wheels and source distribution for that version.

PyPI versions and files are immutable. If an upload is incomplete, resolve the
problem and use a new calendar version for the next attempt.
