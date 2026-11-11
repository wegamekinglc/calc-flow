# DAL-512: intermediate SQL schema diagnostics

Baseline: `745160b79d4438014c0a186f11b78d505e4e6371`.
Product fix: `9789320107155be744be42765fc80f9a5ef006a5`, on the existing
`feature/dal-512-relational-frontends` branch and PR #591. This fixes only the
remaining F3 branch. Closed F1/F2/F4 and both untracked review reports are preserved.

## Change

The existing final-output field check remains. The schema boundary additionally
uses QueryAnalysis's existing `_declaration_paths` to inspect reachable SQL and
snapshot-SQL schema facts from the same analyzer. Shared field validation projects
unsupported types to the existing CompileError before placeholder lowering can
create a generated input name. The logical role comes from the existing DAG
path mapping, including escaping of output names.

For the reviewed SQL→DataFrame cast example, the error begins with
`outputs["a.b"].project.value.with_columns.value.schema[0].data_type: unsupported_type:`.
Decimal remains unsupported even when the final projection casts back to int64.
Supported DOUBLE→DataFrame int64 casts still plan successfully. No execution
lowering, type domain, native interface, dependency or project format changed.

## RED / GREEN

The six new regressions cover batch/stream and simple/dotted/quoted-newline
logical output names. Each final projection casts the SQL Decimal field to a
supported int64 field. RED used the reviewer-verified baseline package/native:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=target/relational/revision-wheel-python \
  python -m pytest python/tests/test_relational_frontend.py -q -k intermediate_sql_schema
```

Result: **6 failed**, each with bare ValueError and
`inputs.cf_sql_input_...schema[0].data_type`. After the local fix, the same six
tests pass using the source-matched test package.

Final directly affected check:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=target/relational/f3-python \
  python -m pytest python/tests/test_relational_frontend.py -q -k schema \
  --junitxml=target/relational/f3-schema-final.xml
```

Result: **25 passed, 25 deselected**, zero failure/error/skip. This includes the
new six intermediate failures, two supported-cast controls, the six retained
direct SQL output diagnostics, known empty/unresolved schemas and the existing
schema/nullability/error checks. Original assertions are retained.

Ruff lint/format for the two changed files passes. Python 3.9 grammar passes
(not an executed version matrix). Lizard 1.26.0 CCNs are 4/3 for the changed
check functions and 2/1 for the new tests, all <=8. The existing complexity
ratchet, spec references and full-base whitespace check pass without gate changes.

## Source and native provenance

Only Python production code and its regression tests changed. The native was
not rebuilt. Its verified SHA256 remains
`2a436ee1eee9d78deeaa32f8e462d6f02f37d60eefca3e9f5311f8dc4c0c2ba8`,
built from `1fda9d3b7087c59f740ba67914c4201a16adfe65`.

The test package contains all **55** tracked Python package files copied from
the new source, checked byte for byte, alongside that reused native. The JSON
records every Python file hash, the preparation script, commands/results/log
hashes and both preserved review hashes. This assembled test package is not
represented as a newly rebuilt wheel.

Unchanged native source trees:

- core: `b85b012571586fa2e5fee070a3879ee76068ab25`
- PyO3: `0598f7aaefb029408ada38d02dd6d51bf6c78289`

New Python source tree: `40b9a33ef2baefa88d408b16f96bc0988784f578`.
Root manifests/lockfiles and the native schema planner are unchanged. Prior
Rust/native/lifecycle evidence therefore remains valid for those unchanged
paths; prior wheel bytes are not claimed to match the new Python code. No Rust,
native rebuild, full Python, services, coverage or performance matrix was run.

The reviewer baseline CI snapshot was 9 SUCCESS / 12 IN_PROGRESS with current
documentation and Codacy SUCCESS. Older-head failures are not current failures.
A single new nonblocking snapshot is taken after push; pending checks do not
establish merge readiness. DAL-512 remains `in_progress`, with only F3's missing
branch to be independently rechecked. No merge or delegation occurs here.
