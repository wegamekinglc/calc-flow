# Calc Flow v2 → v3 Migration Guide

Calc Flow 3.0 replaces the v2 project format and Studio REST surface
with the connector-oriented v3 contract. This guide records every
breaking change, the replacement surface, and the manual migration
steps. There is no automatic migration.

## Project format

### `format_version`

| v2                             | v3                          |
| ------------------------------ | --------------------------- |
| `format_version: 2`            | `format_version: 3`         |
| No runtime block               | `runtime.mode: batch\|stream` |
| Inline `data_sources` with raw data | Connector-bound `sources` with `SecretRef` |
| No sink binding                | Connector-bound `sinks` with delivery request |
| No state config                | `state.checkpoint_directory`, `state.checkpoint_interval_seconds` |

### v2 inline data sources → v3 connector bindings

v2 embedded raw JSON data inside the project document:

```json
{
  "data_sources": [
    {"id": "sample", "input": "input", "format": "inline_json", "data": [{"value": 1}]}
  ]
}
```

v3 binds sources to registered connectors by data-only identity:

```json
{
  "sources": [
    {
      "id": "sample",
      "input": "input",
      "connector": {"provider": "calc-flow-connectors", "name": "file", "version": "2.0.0"},
      "options": {"path": "data/input.csv", "format": "csv"},
      "secrets": {}
    }
  ]
}
```

### Secrets

v2 had no secret vocabulary; credentials could only appear as raw
values in options. v3 structurally rejects secret values:

- Every credential arrives as `SecretRef { resolver, key }`.
- No config field can hold a raw credential.
- Resolution happens only at connector open time.
- Secret values never enter errors, logs, metrics, or manifests.

### Schema artifact

`schemas/project-v2.schema.json` is removed;
`schemas/project-v3.schema.json` is the canonical contract.

## Studio REST API

`/api/v2` is removed; `/api/v3` is the only REST surface.

| v2 route                          | v3 route                               |
| --------------------------------- | -------------------------------------- |
| `GET /api/v2/catalog`             | `GET /api/v3/catalog`                  |
| `GET /api/v2/capabilities`        | `GET /api/v3/capabilities`             |
| `GET /api/v2/schema/project`      | `GET /api/v3/schema/project`           |
| `GET/POST /api/v2/projects`       | `GET/POST /api/v3/projects`            |
| `GET/PUT/DELETE /api/v2/projects/{id}` | `GET/PUT/DELETE /api/v3/projects/{id}` |
| `POST /api/v2/projects/{id}/runs` | `POST /api/v3/projects/{id}/runs`      |
| `GET /api/v2/runs/{id}`           | `GET /api/v3/runs/{id}`                |
| `GET /api/v2/runs/{id}/events`    | `GET /api/v3/runs/{id}/events`         |
| `DELETE /api/v2/runs/{id}`        | `DELETE /api/v3/runs/{id}`             |
| (new)                             | `GET /api/v3/jobs`                     |
| (new)                             | `GET /api/v3/jobs/{id}`                |
| (new)                             | `POST /api/v3/jobs/{id}/checkpoint`    |
| (new)                             | `POST /api/v3/jobs/{id}/shutdown`      |
| (new)                             | `POST /api/v3/jobs/{id}/cancel`        |
| (new)                             | `GET /api/v3/jobs/{id}/events` (SSE)   |
| (new)                             | `GET /api/v3/resource-limits`          |

### Resource limits

The v2 worker timeout no longer bounds continuous jobs. The v3
`ResourceLimits` endpoint publishes the equivalent bounds: maximum
concurrent jobs, per-job and global resident-memory ceilings, maximum
checkpoint/state disk usage, and the explicit user-stop lifecycle.

## Connector delivery guarantees

| Connector   | Delivery     | Replay         | Transaction         |
| ----------- | ------------ | -------------- | ------------------- |
| file        | at-least-once | replayable-exact | pre-commit-commit   |
| kafka       | at-least-once | replayable-exact | pre-commit-commit   |
| postgresql  | at-least-once | replayable-exact | pre-commit-commit   |
| clickhouse  | at-least-once | replayable-exact | ledger-idempotent   |
| http        | at-least-once | conditional      | none               |
| websocket   | best-effort  | unreplayable    | none               |

The HTTP replay capability depends on `conditional` mode (ETag /
Last-Modified cursors). WebSocket is always unreplayable;
`DropOldest` backpressure is explicit, observable, and incompatible
with exactly-once.

## Python surface

- `registered_connectors()` returns the compiled-in connector
  descriptors as data-only dictionaries.
- `ConnectorCapability` and `ConnectorCapabilities` dataclasses expose
  the eight-axis capability vocabulary.
- `connector_capabilities()` builds the sorted enumeration from the
  native listing.
- `_native.pyi` gains the `registered_connectors` stub.

## Rust surface

- The `calc-flow-connectors` crate ships behind per-transport feature
  gates (`file` default; `kafka`, `postgresql`, `clickhouse`,
  `http-websocket` opt-in).
- The core crate owns `ConnectorRegistry`, the capability vocabulary,
  `SecretRef`, and `FormatDecoder`/`FormatEncoder` contracts.
- `CalcFlowError::Connector(ConnectorError)` is the safe error
  projection for connector failures.

## Manual migration checklist

1. Bump `format_version` to 3.
2. Replace inline data sources with connector bindings.
3. Move credentials to environment variables and reference them as
   `SecretRef` values.
4. Add `runtime.mode` and mode-specific options.
5. Add `state` config for stream projects.
6. Update Studio API clients from `/api/v2` to `/api/v3`.
7. Regenerate the JSON Schema contract if consuming it downstream.
