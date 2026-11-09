# Captured native capability metadata

These fixtures contain Rust-produced data, not Python capability producers.

- `rust_sequence_bundle_ir1.json`: captured from the installed A6 wheel's actual
  `_native._native_array_capabilities()` query. Native module SHA-256:
  `dbfb2f0a0fb07f8decb988b8758c7f26609b0ee22a063785b4b5b20dd5a175ee`. The outer JSON was pretty-printed; each Rust canonical
  `options_json_schema` string was preserved verbatim.
- `rust_sequence_ir2.json`: exact output of the actual-source
  `NativeArraySequenceFactory::capability_descriptor()` at Rust checkpoint
  `6a7e34bb3b6951dcdfa8d97891371dd30e560606`. SHA-256:
  `efa17b876ce33f00903bb51577d85ae87c5f5b1be302313fd45fb57dd642a5b4`.

Tests inspect the two known schema-5 profiles, deliberately corrupt metadata
for rejection cases, and preserve frozen IR1 evidence. They do not install a
fake Rust factory or establish numeric/managed execution of the new IR2 wheel.
