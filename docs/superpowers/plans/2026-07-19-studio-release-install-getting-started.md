# Studio Release Install and Getting Started Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve a prepared non-editable Calc Flow release installation when the managed Studio starts, and document package and source installation plus Studio startup on Linux and native Windows.

**Architecture:** The shared Studio process manager will add `--no-sync` to the API child command so startup cannot replace the prepared Python environment. A new `docs/getting-started.md` will define two supported flows: published packages for end users and release wheels built from source for developers. Focused tests will lock both the launcher command and the cross-platform documentation contract before a real spawned-worker smoke test proves the boundary.

**Tech Stack:** Python 3.13, Rust 1.88, PyO3, Maturin 1.14.1, uv, FastAPI, multiprocessing spawn, Node.js 20.20.0, npm, Bash, PowerShell, pytest, unittest.

## Global Constraints

- Preserve the existing Bash and PowerShell wrapper interfaces, process-state format, URLs, logs, readiness checks, and stop behavior.
- Keep Cargo, Maturin, uv, wheel, and release outputs under the repository `target/` tree where supported.
- Never leave `python/calc_flow/_native*.so` or `_native*.pyd` in the source tree.
- Keep the Python core and Studio as independently installed distributions.
- Use Linux Bash syntax for Linux and WSL; use native PowerShell syntax and `.venv\Scripts` paths for Windows.
- Do not add Python v1 compatibility instructions.
- Do not modify generated project schema, OpenAPI, or TypeScript API artifacts.
- Preserve unrelated user files and stage only the design, plan, launcher, tests, guide, and README link.

---

## File Structure

- Modify `web-ui/backend/tests/test_web_ui_scripts.py`: lock the API child command to the prepared environment.
- Modify `web-ui/scripts/web_ui_process.py`: add `--no-sync` to the managed API command.
- Modify `scripts/test_release_config.py`: define the required getting-started documentation contract.
- Create `docs/getting-started.md`: package-user and source-developer instructions for Linux and Windows.
- Modify `README.md`: link the new guide from the reference section.
- Preserve `docs/introduction.md`: it remains the architecture guide rather than becoming an installation guide.

### Task 1: Preserve the prepared Studio Python environment

**Files:**
- Modify: `web-ui/backend/tests/test_web_ui_scripts.py`
- Modify: `web-ui/scripts/web_ui_process.py`

**Interfaces:**
- Consumes: the existing `uv` executable resolved by `_require_command("uv")`.
- Produces: API child command `uv run --no-sync --package calc-flow-studio calc-flow-web`.
- Preserves: `ServiceRecord`, `_spawn_service()`, readiness URLs, and lifecycle semantics.

- [ ] **Step 1: Write the failing launcher regression assertion**

Extend `test_web_ui_process_manager_launches_workspace_backend()` with the
exact prepared-environment requirement:

```python
def test_web_ui_process_manager_launches_workspace_backend() -> None:
    source = PROCESS_MANAGER.read_text(encoding="utf-8")

    assert 'WEB_UI = ROOT / "web-ui"' not in source
    assert '"--package", "calc-flow-studio"' in source
    assert 'command=[uv, "run", "--no-sync", "--package"' in source
    assert '"--extra", "web"' not in source
    assert "/api/v2/catalog" in source
```

- [ ] **Step 2: Run the focused test and record RED**

Run:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run --no-sync --project web-ui/backend \
  pytest web-ui/backend/tests/test_web_ui_scripts.py::test_web_ui_process_manager_launches_workspace_backend -q
```

Expected: FAIL at the new assertion because the API command currently omits
`"--no-sync"`.

- [ ] **Step 3: Implement the minimal launcher change**

Change only the API command in `start()`:

```python
api_process, api_service = _spawn_service(
    name="api",
    command=[
        uv,
        "run",
        "--no-sync",
        "--package",
        "calc-flow-studio",
        "calc-flow-web",
    ],
    cwd=ROOT,
    environment=environment,
    url="http://127.0.0.1:8765/api/v2/catalog",
    log_path=api_log,
)
```

Do not add an automatic build or package installation step.

- [ ] **Step 4: Run the focused script tests and record GREEN**

Run:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run --no-sync --project web-ui/backend \
  pytest web-ui/backend/tests/test_web_ui_scripts.py -q
```

Expected: PASS on Linux, with Windows-only process tests skipped.

### Task 2: Define and implement the cross-platform getting-started contract

**Files:**
- Modify: `scripts/test_release_config.py`
- Create: `docs/getting-started.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: Rust MSRV `1.88.0`, Python floor `3.13`, Node.js CI version `20.20.0`, Maturin `1.14.1`, and the existing Bash/PowerShell launchers.
- Produces: `docs/getting-started.md` and a root README reference link.
- Preserves: `docs/introduction.md` as the architecture guide.

- [ ] **Step 1: Write the failing documentation contract test**

Add this method to `ReleaseConfigTests` in `scripts/test_release_config.py`:

```python
def test_getting_started_covers_packages_source_and_studio_platforms(self) -> None:
    guide_path = ROOT / "docs/getting-started.md"
    self.assertTrue(guide_path.is_file())
    guide = guide_path.read_text()

    for heading in (
        "## Choose an installation path",
        "## Prerequisites",
        "## Install published packages",
        "## Build and install from source",
        "## Start and stop Studio",
        "## Verify the installation",
        "## Troubleshooting",
        "### Linux and WSL",
        "### Windows PowerShell",
    ):
        self.assertIn(heading, guide)

    for command in (
        "uv tool install calc-flow-studio",
        "cargo build --workspace --all-features --release",
        "maturin==1.14.1",
        "uv run --no-sync --package calc-flow-studio calc-flow-web",
        "./web-ui/scripts/start_web_ui.sh",
        r".\web-ui\scripts\start_web_ui.ps1",
    ):
        self.assertIn(command, guide)

    readme = (ROOT / "README.md").read_text()
    self.assertIn("[getting started](docs/getting-started.md)", readme)
```

- [ ] **Step 2: Run the documentation test and record RED**

Run:

```bash
python -m unittest \
  scripts.test_release_config.ReleaseConfigTests.test_getting_started_covers_packages_source_and_studio_platforms
```

Expected: FAIL because `docs/getting-started.md` does not exist.

- [ ] **Step 3: Create the getting-started guide**

Create `docs/getting-started.md` with this structure and command contract:

```markdown
# Getting started with Calc Flow

This guide covers published packages and release builds from source. Linux
commands use Bash, Windows commands use native PowerShell, and WSL follows the
Linux instructions.

## Choose an installation path

- Package users install `calc-flow` or `calc-flow-studio` from the package
  index and do not need the repository toolchain.
- Source developers clone the repository, build release wheels beneath
  `target/`, and use the managed Studio launchers.

## Prerequisites

Package users need Python 3.13 or newer and `uv`. Rust users need Rust 1.88 or
newer. Source developers additionally need Git, Rust 1.88, Python 3.13, `uv`,
Node.js 20.20 or newer, and npm.

## Install published packages

### Linux and WSL

```bash
uv init calc-flow-example
cd calc-flow-example
uv add calc-flow
uv add "calc-flow[numpy]"
uv tool install calc-flow-studio
```

### Windows PowerShell

```powershell
uv init calc-flow-example
Set-Location calc-flow-example
uv add calc-flow
uv add "calc-flow[numpy]"
uv tool install calc-flow-studio
```

Rust applications add the crate with `cargo add calc-flow@2.0.0`.

## Build and install from source

### Linux and WSL

```bash
git clone https://github.com/wegamekinglc/calc-flow.git
cd calc-flow
export UV_CACHE_DIR="$PWD/target/uv-cache"
export UV_TOOL_DIR="$PWD/target/uv-tools"
cargo build --workspace --all-features --release
cd web-ui
npm ci
npm run build
cd ..
uvx --from maturin==1.14.1 maturin build --release --out target/wheels
uv build --project web-ui/backend --wheel --out-dir target/wheels
uv venv --python 3.13
uv pip install --python .venv/bin/python \
  target/wheels/calc_flow-*.whl \
  target/wheels/calc_flow_studio-*.whl
```

### Windows PowerShell

```powershell
git clone https://github.com/wegamekinglc/calc-flow.git
Set-Location calc-flow
$env:UV_CACHE_DIR = "$PWD\target\uv-cache"
$env:UV_TOOL_DIR = "$PWD\target\uv-tools"
cargo build --workspace --all-features --release
Push-Location web-ui
npm ci
npm run build
Pop-Location
uvx --from maturin==1.14.1 maturin build --release --out target\wheels
uv build --project web-ui/backend --wheel --out-dir target\wheels
uv venv --python 3.13
$coreWheel = Get-ChildItem target\wheels\calc_flow-*.whl |
  Sort-Object LastWriteTime | Select-Object -Last 1
$studioWheel = Get-ChildItem target\wheels\calc_flow_studio-*.whl |
  Sort-Object LastWriteTime | Select-Object -Last 1
uv pip install --python .venv\Scripts\python.exe \
  $coreWheel.FullName $studioWheel.FullName
```

Rebuild and reinstall both wheels after changing Python, Rust, backend, or
frontend sources. Do not leave a generated native module under
`python/calc_flow/`.

## Start and stop Studio

The packaged Studio uses `calc-flow-web` and serves its bundled frontend at
`http://127.0.0.1:8765`. The repository launchers start the API at port 8765
and Vite at `http://127.0.0.1:5173`.

Start the packaged server on either platform with:

```text
calc-flow-web
```

### Linux and WSL

```bash
uv run --no-sync --package calc-flow-studio calc-flow-web
./web-ui/scripts/start_web_ui.sh
./web-ui/scripts/stop_web_ui.sh
```

### Windows PowerShell

```powershell
uv run --no-sync --package calc-flow-studio calc-flow-web
.\web-ui\scripts\start_web_ui.ps1
.\web-ui\scripts\stop_web_ui.ps1
```

Use either the packaged/static server command or the managed API-plus-Vite
launcher, not both at the same time. Managed logs live under
`.calc-flow-web/`.

## Verify the installation

### Linux and WSL

```bash
.venv/bin/python -c \
  'import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)'
curl --fail http://127.0.0.1:8765/api/v2/catalog
```

### Windows PowerShell

```powershell
.venv\Scripts\python.exe -c \
  "import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)"
Invoke-WebRequest http://127.0.0.1:8765/api/v2/catalog
```

The native path must resolve beneath `.venv` or the `uv tool` environment,
not beneath `python/calc_flow/`.

## Troubleshooting

- An `uv --no-sync` failure means the environment must be installed first.
- `worker exited without a result` with a native import traceback means the
  core wheel is missing or was replaced by an incomplete editable install.
- Inspect `.calc-flow-web/api.log` and `.calc-flow-web/studio.log` for managed
  startup failures.
- Stop the process using ports 5173 or 8765 before starting another Studio.
```

Keep explanatory prose concise and link to `docs/introduction.md`,
`docs/python-api.md`, and `docs/rust-api.md` rather than duplicating API
details.

- [ ] **Step 4: Link the guide from the root README**

Add this item at the start of the `## Reference` list in `README.md`:

```markdown
- [getting started](docs/getting-started.md)
```

- [ ] **Step 5: Run the documentation and release helper tests**

Run:

```bash
python -m unittest scripts.test_release_config scripts.test_inspect_wheel
```

Expected: PASS.

### Task 3: Verify the complete fix and commit the intended scope

**Files:**
- Verify: `web-ui/scripts/web_ui_process.py`
- Verify: `web-ui/backend/tests/test_web_ui_scripts.py`
- Verify: `scripts/test_release_config.py`
- Verify: `docs/getting-started.md`
- Verify: `README.md`
- Consume: the separately committed design and implementation plan.

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: verified launcher behavior, runnable cross-platform guide, and narrowly scoped commits.

- [ ] **Step 1: Run Python formatting and focused Studio verification**

Run:

```bash
UV_CACHE_DIR="$PWD/target/uv-cache" uv run --no-sync ruff check \
  web-ui/scripts/web_ui_process.py \
  web-ui/backend/tests/test_web_ui_scripts.py \
  scripts/test_release_config.py
UV_CACHE_DIR="$PWD/target/uv-cache" uv run --no-sync ruff format --check \
  web-ui/scripts/web_ui_process.py \
  web-ui/backend/tests/test_web_ui_scripts.py \
  scripts/test_release_config.py
UV_CACHE_DIR="$PWD/target/uv-cache" \
  uv run --no-sync --project web-ui/backend \
  pytest web-ui/backend/tests/test_web_ui_scripts.py -q
```

Expected: all commands exit zero.

- [ ] **Step 2: Run the full Studio backend suite with coverage**

Run:

```bash
cd web-ui/backend
UV_CACHE_DIR="$PWD/../../target/uv-cache" \
  uv run --no-sync pytest --cov=calc_flow_studio
```

Expected: PASS at or above the independent 85% coverage floor.

- [ ] **Step 3: Prove checked contracts did not drift**

Run:

```bash
git diff --exit-code -- \
  schemas/project-v2.schema.json \
  web-ui/openapi.json \
  web-ui/src/api/schema.d.ts
git diff --check
```

Expected: no generated-contract diff and no whitespace errors.

- [ ] **Step 4: Build and install non-editable release wheels**

Build both wheels beneath `target/wheels`, then install them into an isolated
environment without changing the repository `.venv`:

```bash
uv venv target/getting-started-smoke --python 3.13
uv pip install --python target/getting-started-smoke/bin/python \
  target/wheels/calc_flow-*.whl \
  target/wheels/calc_flow_studio-*.whl
```

Confirm:

```bash
target/getting-started-smoke/bin/python -c \
  'import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)'
```

Expected: version `2.0.0` and a native path beneath the isolated environment.
No `_native*.so` exists beneath `python/calc_flow/`.

- [ ] **Step 5: Start Studio and submit a real spawned-worker run**

Start with the prepared environment and managed launcher. Create or reuse a
valid v2 project containing inline records and one expression node, submit
`POST /api/v2/projects/{project_id}/runs`, and poll
`GET /api/v2/runs/{run_id}`. Expected terminal response:

```json
{
  "status": "completed",
  "error": null
}
```

Also verify HTTP 200 from `http://127.0.0.1:5173/` and
`http://127.0.0.1:8765/api/v2/catalog`, and verify the fresh API log contains
neither `ImportError` nor `worker exited without a result`.

- [ ] **Step 6: Replace the current demo with the verified launcher**

Stop the current managed Studio session, start it again through the updated
launcher using the prepared non-editable environment, and leave that verified
session running for the user. Confirm both endpoints return HTTP 200 after the
replacement and that the owning terminal remains alive.

- [ ] **Step 7: Review and commit the intended implementation scope**

Because the workspace `.git` is read-only, use the existing temporary commit
repository at `/tmp/calc-flow-docs-commit/.git` with the workspace as its work
tree. Inspect scope before committing:

```bash
GIT_DIR=/tmp/calc-flow-docs-commit/.git GIT_WORK_TREE="$PWD" git status --short
GIT_DIR=/tmp/calc-flow-docs-commit/.git GIT_WORK_TREE="$PWD" git diff --check
GIT_DIR=/tmp/calc-flow-docs-commit/.git GIT_WORK_TREE="$PWD" git diff --name-status
```

Stage only:

```text
README.md
docs/getting-started.md
scripts/test_release_config.py
web-ui/backend/tests/test_web_ui_scripts.py
web-ui/scripts/web_ui_process.py
```

Commit the implementation with:

```bash
git commit -m "fix: preserve release Studio environment"
```

Expected: the temporary branch contains the prior design commit, the plan
commit, and the verified implementation commit; runtime files and ignored
artifacts are not staged.
