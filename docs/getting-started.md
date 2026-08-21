# Getting started with Calc Flow

Calc Flow 3.0 is a Rust-native calculation engine with a Python binding and an
optional local Studio. This guide covers two installation paths:

- install published packages when you want to use Calc Flow in an application;
- build release artifacts from source when you want to develop Calc Flow or
  run the repository version of Studio.

Linux commands use Bash, Windows commands use native PowerShell, and WSL
follows the Linux instructions. See the [architecture guide](introduction.md)
for the engine boundaries and execution model.

## Choose an installation path

Use published packages if you only need the Python API, Rust crate, or packaged
Studio. This path does not require a repository checkout, a Rust compiler for
Python wheels, or Node.js.

Build from source if you are changing Calc Flow, need unreleased code, or want
the managed API-plus-Vite Studio. The source flow builds non-editable core and
Studio wheels so the API process and its spawned job workers import the
same native extension from the prepared environment.

## Prerequisites

Package users need:

- Python 3.13 or newer and [uv](https://docs.astral.sh/uv/) for Python;
- Rust 1.88 or newer and Cargo for Rust applications.

Source developers additionally need:

- Git;
- Rust 1.88 or newer;
- Python 3.13 or newer;
- uv;
- Node.js 20.20 or newer and npm.

Confirm the source toolchain before building.

### Linux and WSL

```bash
git --version
rustc --version
python3 --version
uv --version
node --version
npm --version
```

### Windows PowerShell

```powershell
git --version
rustc --version
py -3.13 --version
uv --version
node --version
npm --version
```

## Install published packages

Create a Python project and add the core package. Add only the optional array
providers your application uses.

### Linux and WSL

```bash
uv init calc-flow-example
cd calc-flow-example
uv add calc-flow

# Optional providers
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

Install the packaged Studio as an isolated command-line tool:

```bash
uv tool install calc-flow-studio
calc-flow-web
```

Open `http://127.0.0.1:8765`. The packaged server serves the built frontend and
the `/api/v3` API from the same loopback service. Stop it with `Ctrl+C`.

### Windows PowerShell

```powershell
uv init calc-flow-example
Set-Location calc-flow-example
uv add calc-flow

# Optional providers
uv add "calc-flow[numpy]"
uv add "calc-flow[jax]"
```

Install and start the packaged Studio:

```powershell
uv tool install calc-flow-studio
calc-flow-web
```

Open `http://127.0.0.1:8765`. Stop the server with `Ctrl+C`.

Rust applications add the published crate from their Cargo project:

```bash
cargo add calc-flow@3.0.0
```

Continue with the [Python API guide](python-api.md) or
[Rust API guide](rust-api.md) after installation.

## Build and install from source

The source build writes release artifacts and tool caches beneath `target/`.
It builds the frontend before the Studio wheel so that the wheel contains the
current static assets. The final environment contains non-editable core and
Studio wheel installations.

### Linux and WSL

```bash
git clone https://github.com/wegamekinglc/calc-flow.git
cd calc-flow

export UV_CACHE_DIR="$PWD/target/uv-cache"
export UV_TOOL_DIR="$PWD/target/uv-tools"

# Remove artifacts from a previous editable development build.
find python/calc_flow -maxdepth 1 -type f \
  \( -name '_native*.so' -o -name '_native*.pyd' \) -delete

# Build the complete Rust workspace in release mode.
cargo build --workspace --all-features --release

# Build the Studio frontend assets.
cd web-ui
npm ci
npm run build
cd ..

# Build the Python core and Studio wheels.
uvx --from maturin==1.14.1 maturin build --release --out target/wheels
uv build --project web-ui/backend --wheel --out-dir target/wheels

# Install both wheels into the repository environment.
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

# Remove artifacts from a previous editable development build.
Get-ChildItem python\calc_flow -Filter "_native*.so" | Remove-Item
Get-ChildItem python\calc_flow -Filter "_native*.pyd" | Remove-Item

# Build the complete Rust workspace in release mode.
cargo build --workspace --all-features --release

# Build the Studio frontend assets.
Push-Location web-ui
npm ci
npm run build
Pop-Location

# Build the Python core and Studio wheels.
uvx --from maturin==1.14.1 maturin build --release --out target\wheels
uv build --project web-ui/backend --wheel --out-dir target\wheels

# Install both wheels into the repository environment.
uv venv --python 3.13
$coreWheel = (
    Get-ChildItem target\wheels\calc_flow-*.whl |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
).FullName
$studioWheel = (
    Get-ChildItem target\wheels\calc_flow_studio-*.whl |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
).FullName
uv pip install --python .venv\Scripts\python.exe $coreWheel $studioWheel
```

Rebuild and reinstall both wheels after changing Rust, Python, Studio backend,
or frontend sources. Do not use `maturin develop` for this release-style
installation, and do not leave `_native*.so` or `_native*.pyd` beneath
`python/calc_flow/`.

## Start and stop Studio

There are two source-checkout server modes. Use one at a time because both use
port 8765.

The static mode runs only the installed Studio server. It serves the built
frontend and API together at `http://127.0.0.1:8765`:

```bash
uv run --no-sync --package calc-flow-studio calc-flow-web
```

The managed development mode starts the API at `http://127.0.0.1:8765` and
Vite at `http://127.0.0.1:5173`. It stores logs and process state under
`.calc-flow-web/`.

### Linux and WSL

Start both managed processes from the repository root:

```bash
./web-ui/scripts/start_web_ui.sh
```

Open `http://127.0.0.1:5173`, then stop both processes with:

```bash
./web-ui/scripts/stop_web_ui.sh
```

### Windows PowerShell

Start both managed processes from the repository root:

```powershell
.\web-ui\scripts\start_web_ui.ps1
```

Open `http://127.0.0.1:5173`, then stop both processes with:

```powershell
.\web-ui\scripts\stop_web_ui.ps1
```

Both managed launchers reuse the prepared Python environment without syncing
or replacing its wheel installations. If the environment is missing, prepare
it with the source installation steps before starting Studio.

## Verify the installation

Confirm that Calc Flow reports version `3.0.0` and that the native extension
loads from the environment rather than `python/calc_flow/`.

### Linux and WSL

```bash
.venv/bin/python -c \
  'import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)'

curl --fail http://127.0.0.1:8765/api/v3/catalog
```

### Windows PowerShell

```powershell
.venv\Scripts\python.exe -c `
  "import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)"

Invoke-RestMethod http://127.0.0.1:8765/api/v3/catalog
```

For a release-wheel installation, the printed native-module path is beneath
`.venv/lib/.../site-packages/calc_flow/` on Linux or
`.venv\Lib\site-packages\calc_flow\` on Windows.

## Smoke-test the engine

Run one tiny DataFusion pipeline end-to-end to confirm the package is installed
and the native extension loads correctly.

```python
import pyarrow as pa

from calc_flow import Batch, PipelineBuilder

batch = Batch.from_pyarrow(pa.table({"a": [1, 3], "b": [2, 4]}))
plan = (
    PipelineBuilder("totals")
    .expression("calculate", "total = a + b")
    .compile_batch()
)
result = plan.execute({"input": batch})
print(result.outputs["output"].to_pyarrow()["total"].to_pylist())
```

Save the script as `verify.py` and run it with the prepared environment's
interpreter (not `uv run`, which would sync and replace the wheel installation
you are verifying):

### Linux and WSL

```bash
.venv/bin/python verify.py
```

### Windows PowerShell

```powershell
.venv\Scripts\python.exe verify.py
```

You should see the computed totals: `[3, 7]`.

### Rust

Rust users get an equivalent example in the [Rust API guide](rust-api.md): the
`expression_pipeline` example builds the same expression pipeline over Arrow
`RecordBatch` values and awaits `BatchExecutionPlan::execute`. With a source
checkout you can run it directly:

```bash
cargo run -p calc-flow --example expression_pipeline
```

The command prints Arrow's debug representation:

```text
calculated totals: PrimitiveArray<Int64>
[
  3,
  7,
]
```

The [`expression_pipeline.rs`](../crates/calc-flow/examples/expression_pipeline.rs)
source and [Rust API guide](rust-api.md) walk through the example.

## Troubleshooting

### The launcher reports an unsynchronized environment

The managed launcher intentionally uses `uv --no-sync`. Build and install the
core and Studio wheels before starting it. This prevents startup from silently
replacing the release installation.

### `worker exited without a result`

Inspect `.calc-flow-web/api.log`. If it contains an import error for
`calc_flow._native`, the core release wheel is missing or an incomplete
editable installation replaced it. Reinstall both wheels from
`target/wheels/`, confirm the native path with the verification command, and
restart Studio.

### Port 5173 or 8765 is already in use

Stop the existing managed Studio with the platform stop script. If Studio was
started manually, stop that terminal with `Ctrl+C` before starting another
instance.

### Studio does not become ready

Inspect `.calc-flow-web/api.log` for backend failures and
`.calc-flow-web/studio.log` for npm or Vite failures. The managed launcher
cleans up a partial start and leaves both logs in place.
