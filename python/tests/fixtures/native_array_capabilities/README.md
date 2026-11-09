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

- `rust_sequence_ir3.json`: actual candidate-library output of
  `NativeArraySequenceFactory::capability_descriptor()` from source
  `846ff7a3b6b9c15aac8a5e4c0816946c31c80094`. Library SHA-256:
  `9eab145664f06e5c26e1944eb3864dd3b6998febe24eca3f9a7734a6463cb54b`.
  Fixture SHA-256:
  `bae2bb62e47c081d2b6cfabd447cb0634fa364dc5c7b1c1bf08f0148ab6f2160`.
  The canonical Rust schema string is retained verbatim. This is current Rust
  metadata evidence, not installed-wheel or final-parent equality evidence.

Tests inspect the three known schema-5 profiles, deliberately corrupt metadata
for rejection cases, and preserve frozen IR1 evidence. They do not install a
fake Rust factory or establish numeric/managed execution of a new combined wheel.
