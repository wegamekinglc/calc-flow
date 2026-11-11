# Public relation consumer

Run independently from the repository root:

```bash
cargo run --manifest-path crates/calc-flow/tests/relational-consumer/Cargo.toml
```

This separate Cargo workspace uses only the public `calc-flow` calculation API,
plus Tokio and async-trait for the source/sink lifecycle. It has no DataFusion,
PyO3, Python, repository support helper, manual ports, node IDs or Project JSON
construction. The patches preserve the repository's selected Arrow dependency
implementations.

The executable checks pure DataFrame, pure SQL, DF→SQL→DF→SQL and SQL→DF→SQL,
named fan-out, native schema/nullability and strict analysis, independently owned
stream jobs, source open/close counts, NULL/empty/duplicate rows, mode identities,
and data-only export followed by compilation and execution using the reloaded
plan's public physical bindings. Unordered results compare multisets.

It also checks one warm batch plan with SUM inputs producing 1 and 5, a stream job
producing 1 and 6, and a new job starting at 5. These are independent runs, not
checkpoint recovery. Managed recovery and wider relation operations keep their
existing contracts.
