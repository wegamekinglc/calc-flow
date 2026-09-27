# Team artifacts

This directory keeps artifacts referenced by current source, tests, and
documentation. Completed historical artifacts through 2026-09-27 are in
[`../archive/artifacts-through-2026-09-27.tar.gz`](../archive/artifacts-through-2026-09-27.tar.gz).
The archive contains all 69 original files under their original paths, including
the 15 retained here for direct and transitive references. Its SHA-256 is
`f79b90a51f33b6b58c8010506a5c149d6b971c20608ca5b5489f96b512ec15c6`.

To inspect an archived file without changing this checkout:

```bash
mkdir -p target/archived-artifacts
tar -xzf .codex/archive/artifacts-through-2026-09-27.tar.gz \
  -C target/archived-artifacts
```
