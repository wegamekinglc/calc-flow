# Calc Flow documentation

This directory holds Calc Flow's published documentation and point-in-time
engineering records. The current-state guides linked under **Start here**
always reflect the latest state of the project on `main`; overwrite them in
place when the code changes rather than maintaining per-release copies.
Fundamental changes are recorded in the existing repo-root `CHANGELOG.md`, and
the v1-to-v2 migration boundary is preserved in the historical records listed
below.

## Start here

- **[Getting started](getting-started.md)** — published-package and
  from-source installation, Studio startup, and an install smoke test
- **[Executable examples](examples.md)** — complete Python/Rust inventory,
  one-command runner, expected behavior, and choosing an example to copy
- **[Introduction](introduction.md)** — architecture, data contract, graph
  compilation, table execution, optional array providers, and recovery
- **[Design and architecture](design.md)** — component ownership, batch and
  streaming data paths, checkpoint transactions, extension and security boundaries
- **[Continuous streaming guide](streaming-guide.md)** — sources, cursors,
  watermarks, windows, sinks, delivery, checkpoints, recovery, and operations
- **[Stream message envelope](runtime-envelope.md)** — the v3 stream message
  contract: typed messages, event-time progress, windows, state manifests and
  backends, the operator emission boundary, and current delivery guarantees
- **[Connectors and stream projects](connectors.md)** — exact connector
  identities, delivery limits, project fragments, windows, and recovery
- **[Python API guide](python-api.md)** — `PipelineBuilder`, batches, UDFs,
  async execution, NumPy/JAX, symbolic declarations and static analysis,
  projects, and runners
- **[Symbolic workflows](symbolic-workflows.md)** — composed financial
  features, batch/continuous execution, checkpoint recovery, static NumPy/JAX
  matrices, capability failures, Studio inspection, and performance output
- **[Rust API guide](rust-api.md)** — native batches, operators, graph
  compiler, UDF/provider registries, and recovery, with paired examples
- **[API reference](api-reference.md)** — the supported surfaces at a glance:
  Rust exports, Python members, and the Studio HTTP API
- **[Python release guide](python-release.md)** — local packaging rehearsal,
  artifact matrix verification, Trusted Publishers, and the PyPI procedure

## Project orientation

- **[Repository README](../README.md)** — workspace entry point, quick starts,
  architecture map, and examples
- **[AGENTS.md](../AGENTS.md)** — the authoritative agent guide: commands,
  coding style, architecture summary, test layout, and release invariants
- **[CLAUDE.md](../CLAUDE.md)** — Claude Code operational guidance, kept in
  step with AGENTS.md
- **[Codex agent team](../.codex/agents/README.md)** — Codex-native roster,
  workflow, artifact layout, and invocation examples
- **[Claude agent team](../.claude/agents/README.md)** — preserved Claude
  compatibility roster and workflow
- **[Examples](../examples/README.md)** — executable v3 Python examples
- **[Rust examples](../crates/calc-flow/examples/README.md)** — executable
  `calc-flow` crate examples
- **[Benchmarks](../benchmarks/README.md)** — informational benchmark harness
- **[Streaming engine research](research/2026-08-02-arroyo-risingwave-streaming-research.md)**
  — point-in-time Arroyo/RisingWave architecture research and Calc-Flow
  continuous-runtime recommendations
- **[Symbolic computation engine design](superpowers/specs/2026-08-22-symbolic-computation-engine-design.md)**
  — point-in-time Python symbolic IR, batch/stream lowering, and native
  operator design, paired with its
  [phased implementation plan](superpowers/plans/2026-08-22-symbolic-computation-engine.md).
  The immutable declaration and static-analysis layer is available today; see
  the [Python API guide](python-api.md)
- **[TA-Lib-inspired rolling engine upgrade plan](superpowers/plans/2026-09-04-ta-lib-streaming-rolling-engine-upgrade.md)**
  — point-in-time performance evidence for single and composed rolling
  indicators, analysis of TA-Lib's unreleased streaming API, and the phased
  Arrow-native rolling/DataFusion 54 upgrade path
- **[Project schema](../schemas/project-v3.schema.json)** — the canonical
  generated v3 project contract
- **[Studio README](../web-ui/README.md)** — the local calc-flow-studio
  application

## Historical records

These files are release history, not normative docs. They describe older
surfaces and are preserved for audit; leave them untouched. The current
(normative) docs above always override them.

- **[v2 release guide](v2-release.md)** — the v1-to-v2 migration boundary,
  package versions, upgrade checklist, and release artifacts. This is the
  pointer for anyone moving from frozen Python v1 to Rust-native v2.
- **[v1 final API](v1-final-api.md)** — the final Python v1 API reference.
  The frozen v1 implementation is preserved in
  [commit `c87324e`](https://github.com/wegamekinglc/calc-flow/tree/c87324ecaee30d8b883d3c30ae03704dee45f593).
- **[v0.2 migration](migration-v0.2.md)** — the v0.1-to-v0.2 prototype
  migration, predating v1.
- **[Engineering records](superpowers/)** — dated design notes,
  implementation plans, specifications, and hand-offs. These preserve the
  decisions and execution state at the time they were written; they are not
  current API or operational guidance.

Immutable v1 semantic fixtures live under
[`tests/fixtures/v1/`](../tests/fixtures/v1/) as historical parity evidence;
they are not a v2 runtime or package path.

## Conventions

All documentation uses GitHub-flavored Markdown: inline code with backticks,
file paths relative to the document, and cross-references as relative links.
Align table columns with pipes and pad separator rows so their dashes span the
full column width. Docs describe what exists, not the design history that led
there; cite the type, function, or file name rather than source line numbers.
