# Historical benchmark helper fixture

`benchmark_internals.rs.txt` preserves the helper source from product commit
`5bd794d9cd4de2d40d90eb2fc7ff8c30c67c87cc`. Its SHA-256 is
`924fe08cef76b74164c8e7aedce011b1762ad10077c47b98e2317c955f2315da`.

The rustdoc compatibility tests use these exact bytes to exercise the narrow
historical artifact bridge. This is data for source-attestation tests, not a
compiled helper or a copy of the current implementation. Keep the fixture
unchanged when refactoring current benchmark code; tests retain the original
fingerprint and reject any edit outside the reviewed two-line rustdoc change.
