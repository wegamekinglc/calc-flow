# Inner Join compatibility before ASOF

These immutable vectors were captured on 2026-09-09 from
`eda1583751abbd1ca4d246fcb8ee6b70f57d9b09`
(`feature/python-expression-api-refactor`, PR #259), before ASOF production
changes. Rust used the locked DataFusion 54 and Arrow 58.3 dependencies;
Python used the native wheel built from that same checkout.

Do not regenerate these files to accommodate an ASOF change. The tests have
no fixture-update mode. A deliberate future legacy-contract migration needs
separate versioned evidence and review.

- `raw-explicit.json` and `raw-omitted-prefixes.json` import to the exact
  `canonical-project.json` export. `operator-configuration.json` freezes
  the direct operator configuration bytes.
- `fingerprints.json` records the direct native graph and connector-backed
  project fingerprints separately. The connector project uses fixed relative
  file paths and is compiled without opening sources or sinks.
- `checkpoint.json` describes the layout-1 state and segment filenames.
  `checkpoint-metadata.json` freezes its compact metadata bytes; the binary
  files freeze the actual Arrow IPC-bearing state segments.
- The checkpoint trace admits left rows `("a",100,10)`, `("b",200,20)`,
  captures epoch 1, then admits right `("a",105,30)` and captures epoch 2.
  One match has already been emitted. Bounds are 10 microseconds before and
  15 after. Recovery admits right `("a",110,40)`, then left `("a",103,50)`;
  the latter still emits both matching right rows.
- `symbolic-identities.json` freezes v1/v2 node bytes, node digests and
  Program fingerprints. The lowering files freeze each version's complete
  lowered graph, including physical node IDs, schemas and edges. The v1
  digest also agrees with the older explicit vector in
  `python/tests/test_symbolic_relational_dag.py`.

Existing focused coverage remains authoritative for raw validation errors,
default-prefix values, versioned logical state charging, eviction tombstones
and compaction (`stream_join_validation.rs`, `stream_join_state.rs`), and
the complete old capability entry (`python/tests/test_capabilities.py`).
These vectors add compatibility evidence without changing those goldens.

The direct operator snapshot is not a managed v3 manifest. Managed source
cursor, barrier, sink and recovery acceptance remains in the existing
`python/tests/test_symbolic_stream_late_and_recovery.py` suite.
