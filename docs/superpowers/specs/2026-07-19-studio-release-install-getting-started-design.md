# Studio Release Install and Getting Started Design

**Status:** Implemented and merged in PR #16 (historical design)

## Goal

Make the repository-managed Calc Flow Studio reuse a deliberately prepared
Python environment so spawned preview workers can always import the same
native `calc_flow._native` extension as the API process. Add one newcomer
guide that explains how package users and source developers obtain, build,
install, and run Calc Flow on Linux and native Windows.

## Current Failure

The Studio process manager currently starts the API with:

```text
uv run --package calc-flow-studio calc-flow-web
```

Because that command permits an environment sync, `uv` may replace a verified
non-editable release-wheel installation with the workspace's editable
installation. An editable PyO3 installation resolves `calc_flow` from
`python/calc_flow/` and depends on a generated `_native` library beside that
source. If the generated library is removed to keep the source tree clean, the
already-running API continues to work because it has loaded the library, but a
new `multiprocessing` spawn worker cannot import it. The worker exits before it
can write to the result queue, and the parent reports `worker exited without a
result`.

The durable boundary is therefore the prepared Python environment, not the
source directory. The launcher must reuse that environment without changing
its installation mode during startup.

## Selected Approach

Change the managed API child command to:

```text
uv run --no-sync --package calc-flow-studio calc-flow-web
```

The existing Windows PowerShell wrappers already use `uv run --no-sync` to
invoke the shared process manager. The API child will adopt the same rule on
all platforms. Startup remains responsible for process lifecycle, frontend
dependency preparation, readiness checks, logs, and cleanup; Python package
installation remains an explicit prerequisite.

This approach is preferred over rebuilding a wheel during every launch because
it keeps startup fast and avoids an unexpected environment mutation. It is
preferred over an import-only preflight because it prevents the environment
replacement that caused the failure instead of merely improving the error.

## Launcher Behavior

`web-ui/scripts/web_ui_process.py` remains the single cross-platform lifecycle
manager. Only the API command changes. The Bash and PowerShell wrapper
interfaces, process state format, log locations, URLs, frontend command,
readiness polling, idempotent start/stop behavior, and Windows/POSIX process
termination remain unchanged.

If the environment has not been prepared, `uv run --no-sync` will fail rather
than silently build or replace packages. The manager will stop any partial
start and direct the user to `.calc-flow-web/api.log`. The getting-started
guide will make the preparation step explicit for both supported operating
systems.

## Documentation Structure

The existing `docs/` directory and architectural `docs/introduction.md` remain
unchanged in purpose. Add `docs/getting-started.md` as the task-oriented entry
point and link it from the root `README.md` reference section.

The guide will contain these sections:

1. **Choose an installation path** distinguishes package users from source
   developers.
2. **Prerequisites** records Python 3.13 or newer, the workspace Rust/MSRV
   toolchain, `uv`, Git, and Node.js/npm for the repository Studio.
3. **Install published packages** gives Linux Bash and native Windows
   PowerShell commands for the Python package, optional providers, the Rust
   crate, and the packaged Studio server.
4. **Build and install from source** gives parallel Linux and Windows commands
   to clone the repository, create the environment, compile the Rust workspace
   in release mode, build a release wheel under `target/`, install that wheel
   non-editably, and prepare the independent Studio package and frontend.
5. **Start and stop Studio** distinguishes the packaged server from the
   repository-managed API plus Vite workflow. It documents the existing Bash
   and PowerShell launchers, URLs, logs, and stop commands.
6. **Verify the installation** imports `calc_flow._native`, checks the package
   version, probes the Studio API, and identifies the expected native-module
   location for a wheel installation.
7. **Troubleshooting** covers an unprepared `uv --no-sync` environment,
   missing native modules, the `worker exited without a result` symptom, port
   conflicts, and log locations.

Commands will keep Cargo, Maturin, uv, and generated release artifacts under
the repository `target/` tree where the tools support it. Linux examples use
Bash syntax; Windows examples use native PowerShell syntax and `.venv\Scripts`
paths. WSL follows the Linux instructions.

## Package and Source Flows

Published-package users do not need the repository or Vite. They install the
published `calc-flow` and, when desired, `calc-flow-studio` distributions. The
Studio wheel serves its bundled frontend from the loopback FastAPI service.

Source developers clone the repository and prepare one explicit environment.
They build the Rust workspace in release mode, build the PyO3 wheel into
`target/`, and install that wheel non-editably. They then install the Studio
package and frontend dependencies without replacing the core wheel. The
managed launchers reuse this prepared environment with `--no-sync`; spawned
workers therefore import the native module from `site-packages`, just like the
API process.

The guide will not recommend leaving a generated
`python/calc_flow/_native*.so` in the source tree. It will not add compatibility
instructions for the removed Python v1 implementation.

## Testing

Every behavioral change starts with a focused failing test.

- Extend the process-manager launcher test to require `--no-sync` in the API
  command and observe it fail before changing the manager.
- Run the focused Studio script tests after the command change.
- Add a release-configuration documentation test that requires the new guide,
  its Linux and Windows sections, its prepared-environment command, and the
  root README link. Observe the test fail before adding the guide.
- Run the Studio backend test suite with coverage, Ruff checks, release helper
  tests, and Markdown/diff checks after implementation.
- Build the release wheel, install it non-editably, start the managed Studio,
  and submit one real preview run. The run must complete with a result from a
  spawned worker while no generated native library remains under
  `python/calc_flow/`.

No API schema or frontend type changes are expected, so the checked schema and
OpenAPI artifacts must remain unchanged.

## Commit Structure

The approved design is committed separately as required by the design
workflow. Implementation then lands in one narrowly scoped commit containing
the launcher regression fix, its tests, the new getting-started guide, and the
README link. Runtime files, wheels, caches, logs, quarantined projects, and
other ignored artifacts are excluded from the commit.

## Acceptance Criteria

- The managed API child uses `uv run --no-sync` on Linux, WSL, macOS, and
  native Windows.
- A non-editable release-wheel installation remains non-editable after Studio
  startup.
- A spawned preview worker completes a real run without a source-tree native
  library.
- `docs/getting-started.md` covers package and source installation on Linux and
  native Windows, including Studio start and stop commands.
- The root README links to the new guide.
- Focused and relevant full verification passes, generated contracts remain
  unchanged, and only intended files are committed.
