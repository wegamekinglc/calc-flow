# Python packaging and PyPI release

[Documentation](README.md) / 5.5 Releases

The repository workflow publishes only `calc-flow-python`. Install it with
`pip install calc-flow-python` and import `calc_flow`. The Rust crate and
GitHub repository retain the name `calc-flow`.
Studio is not uploaded to PyPI. Its separate wheel is still built and tested
as release evidence; all repository version surfaces remain aligned.

## Version scheme

All version surfaces use calendar versioning `YYYY.M.D` — the release date
without zero padding, for example `2026.9.22`. Workspace crate, Python core,
Studio, and frontend carry the same date version, native internal crate
dependencies pin it exactly, and Studio accepts the release's calendar year
(`calc-flow-python>=<version>,<<next-year>`). Choose the planned release date
as the version; PyPI must not already list it. PyPI versions and files are
immutable, so a failed or incomplete publish is retried under the next
calendar date rather than repaired in place.

## Release contract

The workflow produces the following Python artifact set for every manual or
tagged release run:

| Package            | Built artifacts                                  | PyPI upload |
|--------------------|--------------------------------------------------|-------------|
| `calc-flow-python` | Five abi3 wheels and one source distribution     | Core only   |
| `calc-flow-studio` | One `py3-none-any` wheel with built React assets | Never       |

The five core wheels cover this matrix:

| Operating system | Architecture | Required platform family |
|------------------|--------------|--------------------------|
| Linux            | x86-64       | `manylinux_2_28_x86_64`  |
| Linux            | AArch64      | `manylinux_2_28_aarch64` |
| macOS            | x86-64       | `macosx_*_x86_64`        |
| macOS            | ARM64        | `macosx_*_arm64`         |
| Windows          | AMD64        | `win_amd64`              |

Every core filename starts with `calc_flow_python-` and uses `cp313-abi3`
(CPython 3.13+). The verifier checks internal tags, `calc_flow._native`,
metadata, license, exact version, platform family, source-distribution contents,
Studio assets, and Studio's dependency on `calc-flow-python` within the
release's calendar year.

## Local packaging rehearsal

Install development dependencies and run the cross-platform helper:

```bash
uv sync --extra dev
python scripts/build_python_release.py --clean
```

The helper builds the current platform's core wheel, core source distribution,
frontend, and Studio wheel, then runs artifact content inspectors. Outputs stay
beneath `target/python-release/`. Without `--clean`, a nonempty directory is
rejected. The single-platform output is not the complete PyPI release set and
must not be uploaded as an official release.

To validate a combined seven-artifact CI directory, run:

```bash
python scripts/verify_python_release.py --dist-dir <release-directory>
```

The command prints a stable SHA-256 manifest with relative paths. Add
`--tag calc-flow-python-v<version>` to enforce the tag contract. The CI tag
path also uses `--check-pypi`, which rejects an existing `calc-flow-python`
version. Studio's PyPI availability does not gate this core-only publication.

## CI publication flow

The `Release artifacts` workflow in `.github/workflows/release.yml`:

1. Validates aligned versions. A tag run must use an annotated
   `calc-flow-python-v<version>` tag at the current `main` head and an unused
   core PyPI version.
2. Passes exact-head performance, security, soak, packaging, crate, and audit
   gates. The crate is packaged and dry-run checked, not uploaded to crates.io.
3. Builds all five core wheels, the source distribution, and the Studio wheel.
   Native smoke checks run on Linux x86-64, both macOS targets, and Windows.
   Linux AArch64 receives artifact validation but no runtime smoke test.
4. Downloads all seven artifacts, validates their matrix and metadata, and
   records their exact bytes in `release-manifest.txt`.
5. Re-downloads the verified bundle and checks every SHA-256 before upload.
6. Publishes only the six files in `release-dist/core` through `pypi`.
   There is no Studio publication job or Studio OIDC permission.

Manual dispatches are build-only rehearsals, including dispatches on tags.
Only a pushed `calc-flow-python-v*` tag can reach the publication job. No API
token or `skip-existing` behavior is used.

Each of the three `performance-collection` jobs has a 360-minute limit for its
own cold exact-ref builds and paired cases. The two mandatory 20-minute soaks
run alongside those jobs in `soak-gates`. The acceptance job merges the suite
evidence and runs the security gate after collection. Benchmark thresholds and
soak requirements still apply without `continue-on-error`.

## First-release performance baseline

After the first release, the nearest reachable earlier release tag supplies
the performance baseline. Before any release tag exists, choose and review an
explicit full ancestor commit SHA. This is a pre-release comparison point,
not a previously published release, and all performance gates still run.

For the manual rehearsal, supply that SHA as the `initial-baseline` input.
Record the same SHA in the first release's annotated tag:

```bash
git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>" \
  -m "Benchmark-Baseline: <full-ancestor-commit-sha>"
git push origin calc-flow-python-v<version>
```

`scripts/release_baseline.py` rejects missing, abbreviated, malformed,
duplicate, non-ancestor, or candidate-equal bootstrap baselines. An input cannot
disagree with the annotation or override an existing earlier release.
Later release tags omit the bootstrap annotation and input.

## Performance acceptance and failure evidence

From a clean candidate checkout, prepare the selected baseline as
`target/release-baseline` and activate an environment installed from
`benchmarks/requirements.lock` with hash verification. The release workflow
invokes:

```bash
python -m scripts.release_performance \
  --baseline-source target/release-baseline \
  --output target/release-evidence --suite python
```

Use a fresh output directory for a new measurement. The command builds both
sealed releases and the allocation benchmark binary, collects ordinary Python
cases, and runs the lifecycle, rolling-kernel, and allocation gates. Run the
same command with `--suite core` and `--suite stream_join_perf` in separate fresh
output directories to collect the Rust cases. Each timed case uses two rounds
of ten adjacent AB/BA invocation pairs and the existing +5% paired-median
decision. Python release fixtures use `overhead`; the unified suite's lifecycle
shard uses `standard`
and must not be compared across those scales. See
[release measurement boundaries](benchmark-suite.md#release-acceptance-measurements).

Missing, corrupt, or incompatible identity/sample evidence blocks acceptance.
A timing-only `inconclusive` verdict is not a confirmed regression and does
not itself fail the timing gate, but does not prove equivalent performance.
The workflow input `allow-dependency-drift` and command option
`--allow-dependency-drift` only record acknowledgement; dependency mismatch
still rejects the comparison, including the lifecycle gate.

CI runs the three paired suites and the soak job in parallel. The acceptance
job checks the merged suite evidence before its security gate. Its `if: always()`
summary writes `acceptance.json` and `acceptance.md` with each step's `outcome`
and `conclusion`. Skipped steps name prior failed steps, or report setup/ref
selection failure or cancellation when no earlier gate outcome explains them.
The helper command `python -m scripts.release_performance --output
target/release-evidence --acceptance-summary` consumes `ACCEPTANCE_STEPS` and
`GITHUB_STEP_SUMMARY` from the workflow environment; it summarizes outcomes
without running measurements.

Each suite artifact and the merged `release-performance-evidence` artifact
upload with `if: always()` and 30-day retention. They include available raw
samples and pairs, failure records, results/summary files, command logs and
exit records, sealed-build and Rust provenance, and dependency listings. The
merged artifact also includes acceptance outcomes. Cargo
build caches, installed import directories, and copied Rust benchmark
executables are excluded. Inspect this evidence after a failed run; a skipped
security or soak step supplies no acceptance evidence for that gate.

## One-time Trusted Publisher setup

Configure a PyPI pending Trusted Publisher for the first upload, or a publisher
on the existing project for later uploads. Create the matching GitHub environment:

| PyPI project       | GitHub owner   | Repository  | Workflow filename | GitHub environment |
|--------------------|----------------|-------------|-------------------|--------------------|
| `calc-flow-python` | `wegamekinglc` | `calc-flow` | `release.yml`     | `pypi`             |

Require manual deployment approval when the repository plan supports it.
The workflow requests short-lived OIDC credentials with `id-token: write`;
do not add long-lived PyPI API tokens. No Studio publisher is required.

## Release procedure

1. Choose the planned release date as the `YYYY.M.D` version and confirm
   PyPI does not list it for `calc-flow-python`. Update all version surfaces
   and the changelog together, including the workspace, binding constraints,
   core Python package, Studio, frontend package and lockfile, OpenAPI, and
   generated types.
2. Run the full verification groups in `AGENTS.md`, including release helper
   tests, artifact inspectors, and clean core/Studio wheel smoke checks.
3. Run `Release artifacts` manually from the reviewed `main` commit. For the
   first release, provide `initial-baseline`. Confirm all seven build artifacts
   and the verified SHA-256 bundle are present.
4. Tag that same current `main` commit and push only the tag. Use the annotated
   bootstrap example above for the first release. For later releases:

   ```bash
   git tag -a calc-flow-python-v<version> -m "Release calc-flow-python <version>"
   git push origin calc-flow-python-v<version>
   ```

5. Approve the `pypi` deployment after reviewing gates and the manifest.
6. Confirm PyPI lists exactly six `calc-flow-python` files with matching SHA-256
   hashes and publisher provenance. Install the exact version from public PyPI
   in clean environments and run the native core smoke check. Confirm that no
   Studio package was uploaded.

PyPI versions and files are immutable. If an upload is incomplete, fix the
release problem, move to the next calendar date as the shared version, and
rerun the complete process; never repair the old version with a partial or
skip-existing upload.
