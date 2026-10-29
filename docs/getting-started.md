# Getting started with Calc Flow

[Documentation](README.md) / 1. Installation

Calc Flow is a Python calculation library backed by an internal Rust runtime.
The optional local Studio application moved to the separate
[calc-flow-studio](https://github.com/wegamekinglc/calc-flow-studio) repository.
This guide covers two installation paths:

- install published packages when you want to use Calc Flow in an application;
- build release artifacts from source when you want to develop Calc Flow.

Linux commands use Bash, Windows commands use native PowerShell, and WSL
follows the Linux instructions. See the [introduction](introduction.md)
for capabilities and the execution model.

On this page:

- [Choose an installation path](#choose-an-installation-path)
- [Prerequisites](#prerequisites)
- [Install published packages](#install-published-packages)
- [Build and install from source](#build-and-install-from-source)
- [Verify the installation](#verify-the-installation)
- [Smoke-test the engine](#smoke-test-the-engine)
- [Troubleshooting](#troubleshooting)
- [Continue learning](#continue-learning)

## Choose an installation path

Use published packages for Python application development. This
path does not require a repository checkout or a Rust compiler.
[Build and install from source](#build-and-install-from-source) for unreleased
code or Calc Flow development. The source flow builds a non-editable core
wheel and installs it into the prepared environment.

## Prerequisites

Package users need:

- CPython 3.9 or newer and [uv](https://docs.astral.sh/uv/).

Source developers additionally need:

- Git;
- Rust 1.94 or newer;
- Python 3.13 or newer;
- uv.

Confirm the source toolchain before building.

### Linux and WSL

```bash
git --version
rustc --version
python3 --version
uv --version
```

### Windows PowerShell

```powershell
git --version
rustc --version
py -3.13 --version
uv --version
```

## Install published packages

Create a Python project and add the core package. For the table examples in this
guide, also install `calc-flow-python[pyarrow]`. A standalone application can use
Arrow protocol producers and explicit `output_format="batch"` without PyArrow.
Add only the optional array providers your application uses.

### Linux and WSL

```bash
uv init calc-flow-example
cd calc-flow-example
uv add calc-flow-python
uv add "calc-flow-python[pyarrow]"

# Optional providers
uv add "calc-flow-python[numpy]"
uv add "calc-flow-python[jax]"
```

### Windows PowerShell

```powershell
uv init calc-flow-example
Set-Location calc-flow-example
uv add calc-flow-python
uv add "calc-flow-python[pyarrow]"

# Optional providers
uv add "calc-flow-python[numpy]"
uv add "calc-flow-python[jax]"
```

Continue with the [batch guide](batch-guide.md) after installation. Native
runtime and extension authors can find crate installation and examples in the
[Rust runtime reference](rust-api.md).

## Build and install from source

The source build writes release artifacts and tool caches beneath `target/`.
The final environment contains a non-editable core wheel installation.

Workspace release builds, including Python wheels, use Thin LTO and one codegen
unit, configured as `lto = "thin"` and `codegen-units = 1` in
[Cargo.toml](../Cargo.toml). Cargo's `bench` profile inherits these settings;
ordinary development and test builds use their default profiles.

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

# Build the Python core wheel.
uvx --from maturin==1.14.1 maturin build --release --out target/wheels

# Install the core wheel into the repository environment.
uv venv --python 3.13
uv pip install --python .venv/bin/python \
  target/wheels/calc_flow_python-*.whl
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

# Build the Python core wheel.
uvx --from maturin==1.14.1 maturin build --release --out target\wheels

# Install the core wheel into the repository environment.
uv venv --python 3.13
$coreWheel = (
    Get-ChildItem target\wheels\calc_flow_python-*.whl |
        Sort-Object LastWriteTime |
        Select-Object -Last 1
).FullName
uv pip install --python .venv\Scripts\python.exe $coreWheel
```

Rebuild and reinstall the wheel after changing Rust or Python sources. Do not
use `maturin develop` for this release-style installation, and do not leave
`_native*.so` or `_native*.pyd` beneath `python/calc_flow/`.

## Verify the installation

For a build from this checkout, confirm that Calc Flow reports version `2026.10.27`.
A published installation reports the release selected by the package manager,
which can differ from the current source version. Confirm that the native
extension loads from the environment rather than `python/calc_flow/`.

### Linux and WSL

```bash
.venv/bin/python -c \
  'import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)'
```

### Windows PowerShell

```powershell
.venv\Scripts\python.exe -c `
  "import calc_flow, calc_flow._native as native; print(calc_flow.__version__); print(native.__file__)"
```

For a release-wheel installation, the printed native-module path is beneath
`.venv/lib/.../site-packages/calc_flow/` on Linux or
`.venv\Lib\site-packages\calc_flow\` on Windows.

## Smoke-test the engine

Run one tiny DataFusion pipeline end-to-end to confirm the package is installed
and the native extension loads correctly.

```python
import pyarrow as pa
import calc_flow as cf

data = pa.table({"a": [1, 3], "b": [2, 4]})
result = cf.compute(data, lambda t: t.select(total=t["a"] + t["b"]))
assert result.to_pydict() == {"total": [3, 7]}
print(result["total"].to_pylist())
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

The [Rust runtime reference](rust-api.md) provides the equivalent native
calculation for contributors developing the runtime or extensions.

## Troubleshooting

### Windows checkpoint examples cannot clean up their temporary directories

Managed checkpoint paths include hashed lineage, operator, and segment names.
They can exceed 260 characters beneath a checkout or user temporary directory.
If Windows long-path support is disabled, Python cleanup can report a
`WinError 145` even after a job completes successfully. Follow the
[Windows temporary-path setup](../examples/README.md#prepare-and-run) to use
an extended-length absolute `TEMP` and `TMP` for the example process.

## Continue learning

- Run the complete [executable example inventory](examples.md).
- Compose [SQL and Python pipelines](batch-guide.md#compose-sql-and-python-pipelines).
- Iterate a [stateful stream](streaming-guide.md#first-python-continuous-job),
  then add explicit recovery when needed.
- Configure production transports with the [connector guide](connectors/README.md).
- Understand component ownership in the [design and architecture guide](design.md).
- Look up exact names in the [Python API](python-api.md),
  [Rust runtime reference](rust-api.md), or [API reference](api-reference.md).
