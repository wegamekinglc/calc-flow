# Calc Flow 2.0 release guide

Calc Flow 2.0 replaces the frozen Python v1 execution engine with the native
`calc-flow` Rust crate and a PyO3 Python package. DataFusion remains the only
table engine, while NumPy/JAX are explicit optional Python providers. The
separately packaged Studio consumes the same v2 project schema and native
runtime.

## Compatibility

Calc Flow 2.0 does not load Calc Flow 1.x project documents or checkpoints.
Recreate projects with the v2 schema and restart stateful processing from a
chosen source boundary. No automated converter is provided.

The `v1-python-final` tag preserves the final Python v1 implementation.
`docs/v1-final-api.md` and `docs/migration-v0.2.md` remain historical references,
and `tests/fixtures/v1/` remains immutable semantic evidence. None of those
files is a v2 runtime path.

## Package versions

| Artifact               | Version | Notes                                         |
| ---------------------- | ------- | --------------------------------------------- |
| Rust `calc-flow` crate | 2.0.0   | Native engine and public Rust API             |
| Python `calc-flow`     | 2.0.0   | PyO3 binding and Python integrations          |
| `calc-flow-studio`     | 2.0.0   | Local FastAPI service and built Studio assets |
| `calc-flow-web-ui`     | 2.0.0   | Private frontend workspace package            |

Studio declares `calc-flow>=2.0.0,<3`. The PyO3 binding depends exactly on the
workspace crate version `=2.0.0`.

## Upgrade checklist

1. Stop all v1 runners and record the source boundary from which v2 will
   restart.
2. Preserve v1 project and checkpoint documents for audit only; do not place
   them in v2 stores.
3. Recreate each graph as a strict `format_version: 2` project. Use
   `schemas/project-v2.schema.json` or `project_json_schema()`.
4. Replace v1 Python operators/engines with `PipelineBuilder` expression, SQL,
   external nodes, and explicit runtime registrations.
5. Wrap PyArrow tables with `Batch.from_pyarrow`; wrap arrays with
   `Batch.from_array(..., backend=...)` after provider registration.
6. Register trusted Python scalar UDFs on `Runtime` and reference their exact
   `(provider, name, version)` tuple from nodes.
7. Create a fresh v2 checkpoint directory and restart from the chosen source
   boundary. Expect at-least-once sink delivery around the cutover.
8. Validate projects and run representative outputs against the preserved v1
   semantic corpus before switching production consumers.

## Release artifacts

The release workflow builds:

- manylinux 2.28 x86_64 and aarch64 abi3 wheels;
- macOS x86_64 and arm64 abi3 wheels;
- a Windows x64 abi3 wheel;
- a Python source distribution;
- the `calc-flow` crate package;
- a Studio wheel containing generated static assets.

Artifact inspectors require the Apache-2.0 license, native module, essential
build sources, and expected package assets. They reject frozen Python source,
tests/fixtures, executable project documents, guidance files, and other
repository-only content. Crate packaging excludes integration tests that depend
on repository fixtures/schema.

## Security gates

Release verification runs `cargo audit`, `cargo deny --locked check`, and
`npm audit --omit=dev`. The two narrowly documented PyO3 advisories in
`deny.toml` remain temporary until the DataFusion/PyO3 Arrow major versions can
be upgraded together; affected APIs are not used by Calc Flow.

Third-party workflow actions are pinned to reviewed commit SHAs. Maturin is
pinned to v1.14.1 and Rust to 1.88.0 for release wheel builds.

## Local release verification

Run the commands in `AGENTS.md`, then build and inspect final artifacts:

```bash
cargo package -p calc-flow --locked
cargo publish -p calc-flow --dry-run --locked
uv build --out-dir target/v2-dist
python scripts/inspect_wheel.py core-wheel target/v2-dist/calc_flow-*.whl
python scripts/inspect_wheel.py sdist target/v2-dist/calc_flow-*.tar.gz
python scripts/inspect_wheel.py crate target/package/calc-flow-*.crate

cd web-ui
npm run build
uv build --project backend --wheel --out-dir ../target/v2-studio-dist
cd ..
python scripts/inspect_wheel.py \
  studio-wheel target/v2-studio-dist/calc_flow_studio-*.whl
```

Install the built wheels in empty virtual environments and run
`scripts/smoke_wheel.py` plus the Studio import/route smoke checks before
publishing.
