These local fixtures were generated with PyArrow 24.0.0, independently
of the Rust zstd upgrade. Both files contain 128 rows: non-nullable int64
`value = 0..127`, and nullable UTF-8 `tag = "snapshot"` except null when
`value % 7 == 0`. IPC uses two 64-row batches; Parquet uses two 64-row groups
without dictionary encoding. Both use Zstandard compression.

The compatibility tests read these existing files with the patched dependency
chain and round-trip new Zstandard files through the same Arrow 59.2 APIs.
