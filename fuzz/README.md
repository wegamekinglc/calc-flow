# Parser fuzz targets

These targets exercise the data-only project parser and the SELECT/CTE SQL
validator without starting a runtime or connector. Input sizes are bounded to
keep parser work practical.

Run each target with `cargo-fuzz` installed:

```bash
cargo fuzz run project_parser
cargo fuzz run sql_validator
```

The fuzz crate is separate from the normal workspace and CI test matrix.
