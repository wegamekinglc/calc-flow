# Calc Flow documentation

This directory holds Calc Flow's published documentation and point-in-time
engineering records. The current-state guides linked under **Start here**
always reflect the latest state of the project on `main`; overwrite them in
place when the code changes rather than maintaining per-release copies.
Fundamental changes are recorded in the existing repo-root `CHANGELOG.md`, and
the v1-to-v2 migration boundary is preserved in the historical records listed
below.

## Start here

- **[Introduction](introduction.md)** — architecture, data contract, graph
  compilation, table execution, optional array providers, and recovery
- **[Internal runtime envelope](runtime-envelope.md)** — crate-private
  data/control carrier, ordering, forwarding, fail-closed, rollback, and
  downstream semantic boundaries
- **[Getting started](getting-started.md)** — published-package and
  from-source installation, Studio startup, and an install smoke test
- **[Python API guide](python-api.md)** — `PipelineBuilder`, batches, UDFs,
  async execution, NumPy/JAX, projects, and runners
- **[Rust API guide](rust-api.md)** — native batches, operators, graph
  compiler, UDF/provider registries, and recovery, with paired examples
- **[API reference](api-reference.md)** — the supported surfaces at a glance:
  Rust exports, Python members, and the Studio HTTP API

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
- **[Examples](../examples/README.md)** — executable v2 Python examples
- **[Rust examples](../crates/calc-flow/examples/README.md)** — executable
  `calc-flow` crate examples
- **[Benchmarks](../benchmarks/README.md)** — informational benchmark harness
- **[Project schema](../schemas/project-v2.schema.json)** — the canonical
  generated v2 project contract
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
  The frozen v1 implementation is available at the `v1-python-final` tag.
- **[v0.2 migration](migration-v0.2.md)** — the v0.1-to-v0.2 prototype
  migration, predating v1.
- **[Engineering records](superpowers/)** — dated design notes,
  implementation plans, specifications, and hand-offs. These preserve the
  decisions and execution state at the time they were written; they are not
  current API or operational guidance.

Immutable v1 semantic fixtures live under
[`tests/fixtures/v1/`](../tests/fixtures/v1/) as v2 parity evidence; they are
not a v2 runtime or package path.

## Conventions

All documentation uses GitHub-flavored Markdown: inline code with backticks,
file paths relative to the document, and cross-references as relative links.
Align table columns with pipes and pad separator rows so their dashes span the
full column width. Docs describe what exists, not the design history that led
there; cite the type, function, or file name rather than source line numbers.
