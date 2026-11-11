# DAL-515 finite union implementation evidence

Base: `a5a5f4d8972e5eba47147589d695bf0ecbb20994` (DAL-514 / PR #597).
Dedicated checkout reused as requested; branch renamed in place to
`feature/dal-515-finite-union`. The only inherited untracked file was the frozen
`../api-notes/finite-union.md`. No team definitions or gates are changed.

## Design

Follow the frozen note: add stateless finite execution to the existing
UnionOperator, preserving stream forwarding and checkpoint identity. Add ordered
binary TableExpr.union_all declarations in Rust and Python. Recognize the approved
multi-alias SQL UNION ALL tree in the Rust AST/planner, compare independently
planned arm schemas before coercion, and expose one data-only descriptor to both
frontends. Lower each arm through existing SQL, then the same native union.
Preserve declaration digests and legacy SQL outside that subset.

Analysis revision 4 proves append-history reduction at all-sources EOF, rejects
snapshot-to-union edges for latest/strict final comparison, and clears ordering.
Plan reuse and failure recovery must contain only the current execution's rows.
Keep project-v3 and existing stream union state unchanged.

The requested .codex/references files and a finite-union spec/critique/methodology
do not exist in this repository. Read AGENTS.md, cf-implementer, the repository
code-style skill, docs/introduction.md, relevant existing symbolic contracts and
approved critique, predecessor notes, finite-union note, source and nearby tests.
The frozen note and explicit delegation are the design checkpoint.

## Environment and verification

Commands and observed results are recorded below as they complete. Full suites,
coverage, connector services, benchmarks and cross-platform checks are not local
acceptance substitutes and are left to CI. No merge is authorized before independent
review and required checks.

- Rust 1.94.0; rustfmt 1.8.0 is available in this runtime.
- CPython 3.13.9 and PyArrow 24.0.0 interpreter:
  `/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/.venv/bin/python`.
- Cargo dependency cache:
  `/home/wegamekinglc/dev/github/my-claude/workspace/calc-flow/target/pr-repair-20261112/cargo-debug`;
  jobs=2. Cargo runs from this checkout and rebuilds its sources.
- Logs and packaged verification source go under `target/finite-union`.
- An initial Python test attempt could not import native; it is an environment
  setup failure, not RED evidence or a passing check.

## Test-first observations

All Cargo commands below use `--locked -p calc-flow`, the target/cache and
jobs recorded above, and remove PYO3_PYTHON for standalone core checks.

- RED: `cargo test --test finite_union finite_union_reuses_one_plan_without_old_rows -- --exact`
  failed with node merge still stream-only (0 passed / 1 failed, exit 101).
  The first test draft used a private Port constructor; that setup error was fixed
  to use public Port::new before recording this RED.
- GREEN: the same test passed after adding the existing UnionOperator's finite
  trait/compiled-plan dispatch. One plan executes five rows, then [9], then a
  typed empty output without reset. Missing input, wrong schema and cancellation
  are interleaved with successful calls; metadata uses merge/0/empty attributes.
- RED: `cargo test --test finite_union_frontend dataframe_union_exports_and_compiles_both_lifecycles -- --exact`
  failed because TableExpr.union_all was absent. GREEN: it exported a real union
  and compiled both lifecycles after the ordered relation/lowering was added.
- RED: `cargo test --test finite_union_sql sql_union_uses_native_union_with_textual_arm_order -- --exact`
  failed because the SQL export contained no union. GREEN: the arm for z is
  input_0 even though a sorts before z, and both lifecycle compilers accept it.
- RED: `cargo test --test finite_union_frontend append_union_proves_unordered_eof_history -- --exact`
  observed revision 3. GREEN: revision 4 proves unordered accepted-history/EOF;
  latest-history and declared-order obligations remain rejected/unproved.
- RED: `cargo test --test finite_union finite_union_rejects_non_exact_or_optional_ports_in_analysis -- --exact`
  incorrectly reported batch supported. GREEN: array/schema-less/optional shapes
  report unsupported_mode with the required table/schema/required constraints;
  their original stream forwarding still compiles.
- RED: `cargo test --test finite_union_frontend union_normalizes_declaration_metadata_without_mutating_it -- --exact`
  retained caller metadata in the union schema. GREEN: the typed union boundary
  normalizes metadata while sharing Arrow buffers; input declarations and caller
  schemas retain their exact original schemas.
- RED: `cargo test --test finite_union_frontend union_does_not_invent_a_cross_ingress_batch_boundary -- --exact`
  incorrectly proved per_batch. GREEN: cross-ingress batch grouping is unknown.
  This is a conservative boundary within the frozen final-relation contract.
- RED: `cargo test --test finite_union finite_union_enforces_existing_table_result_limit -- --exact`
  accepted 100,000,002 NULL rows. The fixture uses zero-allocation NullArray
  payloads. GREEN: union uses the existing 100M-row/1-GiB visible-slice collector
  limits, preserves endpoint context, and a same-plan empty retry succeeds.
- RED: `cargo test --test finite_union_sql sql_union_arm_schema_errors_keep_the_arm_path -- --exact`
  reported a generic SQL schema path. GREEN: the shared descriptor retains
  arm index/error, and Rust/Python report outputs.out.sql.arms[1].
- Python declaration RED raised AttributeError for missing union_all; its GREEN
  construction test passed after adding the immutable primitive. This early
  construction-only test loaded an existing native library solely to import
  adapters; it is not native/execution provenance for the final change.
- The first wheel built from this checkout matched all 56 packaged Python/stub
  files. Its 14-case union run had 9 passes and 5 failures. Fixed schema-error
  precedence and union admission in exact update-graph schema pinning. Corrected
  the zero-Data fixture to use an empty async iterable, since empty materialized
  sequences are intentionally invalid. The next matching adapter/native stage
  passed 25 union cases and two additional legacy union SUM oracles (1/7 in
  stream, finite 6). That stage predates final Clippy-only core edits and is
  not substituted for the final matching build below.
- The first external-consumer run exposed a real regression: normalizing every
  table_input schema broke the existing exact Source capability preflight.
  Reverted that global change. A private frontend Union constructor now keeps
  original ingress schemas and normalizes only the union output; the public
  UnionOperator::new and global Source/Port validation remain exact. The focused
  metadata test now also executes original-metadata and canonical inputs in the
  same finite union. The consumer then passed the original metadata oracle and
  the new DF/SQL metadata union stream oracles.
- That consumer's new multi-source fixture initially assigned every cursor to
  events, causing the expected ownership error for a/b. Corrected the fixture to
  use the existing public Cursor::unbound; runtime cursor ownership is unchanged.
  The full external consumer then passed (consumer-cursor-green.log, exit 0).
- `pytest python/tests/test_symbolic_late_output.py -q -k late_successors_reject_even_after_allowed_transform`
  exposed two failures in the old SQL merge constructor rejection. Restored that
  early gate and added DF union to the same boundary matrix. Final matching-wheel
  run: 8 passed / 50 deselected. Existing assertions were retained.

## Scope and compatibility

Core changes are in operator/union.rs, pipeline/batch.rs, pipeline/analysis.rs,
config.rs, dataframe.rs, program/lower and sql/union.rs. Query proof/update
changes are shared by native Rust and Python. Python changes are declaration,
analysis, lowering and cache adapters plus the private native descriptor/stub;
there is no Python row evaluator or SQL parser.

The existing SQL declaration byte encoding is untouched. Only exact, admitted
multi-alias SQL gets the private finite_union_v1 preparation policy and new union
topology. Legacy SQL/coercion and saved SQL projects keep their paths. Low-level
stream union forwarding/configuration/stateless capability are preserved. The
typed union output may rewrap shared columns to remove declaration metadata;
its envelope metadata is preserved. Stream runtime/state sources are unchanged.
Project-v3 has no new fields or operator variant.

Examples are separate `examples/finite_union.py` and
`crates/calc-flow/examples/finite_union.rs`; example 32 and the SQL performance
probe retain their purpose/location. The existing external Rust consumer now
also executes both public union frontends with independent EOF/duplicate/NULL/
empty oracles, logical bindings and shared output.

Clippy and complexity findings are fixed in code: no new allow/forgive marker,
baseline update, threshold change or gate exclusion. New/changed functions stay
at CCN <= 8; the predecessor's revision-only assertion function remains CCN 9
(unchanged). No public design deviation or extra feature scope is intended.

## Final source-matched verification

All entries below exited 0. Full argv, exit codes, log excerpts/hashes and source
provenance are in finite-union.json. Raw logs and verification scripts accompany
the issue handoff as an evidence archive. Cargo uses the environment above;
standalone core/consumer runs explicitly remove PYO3_PYTHON. Native builds and
native Clippy set it to the recorded interpreter. Python commands below use
PYTHONPATH=target/finite-union/delivery-site (the newly built wheel extracted into
an isolated directory) and JAX_PLATFORMS=cpu.

| Command / scope                                                                                                                                                     | Observed result                      |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------|
| `cargo clippy --locked -p calc-flow --lib --example finite_union --test finite_union --test finite_union_frontend --test finite_union_sql -- -D warnings`           | pass                                 |
| `python -m maturin build --locked --profile dev --interpreter <recorded-python> --out target/finite-union/delivery-wheels`                                          | pass; matching CPython 3.13 wheel    |
| `python -m pytest python/tests/{test_finite_union,test_update_semantics,test_query_history_portability,test_aggregate_replacement,test_relational_frontend}.py -q`  | 181 passed                           |
| `python -m pytest python/tests/test_symbolic_late_output.py -q -k late_successors_reject_even_after_allowed_transform`                                              | 8 passed, 50 deselected              |
| `python examples/finite_union.py`                                                                                                                                   | pass; DF/SQL batch and stream oracle |
| `cargo clippy --locked -p calc-flow-python --lib -- -D warnings`                                                                                                    | pass                                 |
| `cargo build --locked -p calc-flow --lib`                                                                                                                           | pass; standalone core, no PYO3       |
| `cargo run --locked -p calc-flow --example finite_union`                                                                                                            | pass                                 |
| `cargo run --locked --manifest-path crates/calc-flow/tests/relational-consumer/Cargo.toml`                                                                          | pass; full external public consumer  |
| `cargo test --locked -p calc-flow --test finite_union --test finite_union_frontend --test finite_union_sql --test latest_relation --test query_history_portability` | 30 passed                            |
| `cargo test --locked -p calc-flow --test stream_operator union`                                                                                                     | 2 passed, 19 filtered out            |
| Ruff check / format --check on the 15 changed Python/stub files                                                                                                     | pass                                 |
| `cargo fmt --all --check`; `rustfmt --edition 2024 --check crates/calc-flow/tests/relational-consumer/src/main.rs`                                                  | pass                                 |
| `python scripts/check_spec_references.py`; `python scripts/verify_complexity_gates.py`                                                                              | pass                                 |
| Lizard 1.26.0 on changed functions; `git diff --check`                                                                                                              | pass; no complexity increase >8      |

The final source manifest was identical before and after the build/check sequence:
1,667 Rust/Python/vendor/build-input files, SHA-256
e5a61bb9844c709696a91f3362cb407c69267ea442527687ee67d677b26fff60.
All 56 packaged .py/.pyi files match this checkout byte-for-byte. The imported
package and extension paths both resolve under delivery-site, not the source or
an older installed package. Wheel SHA-256:
70e4e50745f66b6f45f0b41710a04d6bc77e225d2fa1d73b07df274194c7e418;
native extension SHA-256:
c04dd4911d60da5eaf4cfc5f0d65b063a7f73e439b6d3f642d98b69912a61169.
The final imported native project_json_schema equals schemas/project-v3.schema.json;
the schema file has no diff.

The external consumer's final pass used this same production/consumer source.
The only subsequent Rust edit was renaming a test local (plain to canonical) to
resolve Clippy similar_names; that integration test and core Clippy then passed
in the final sequence. No consumer rerun was needed for that name-only test edit.

Direct evidence covers one compiled finite plan executing five rows, then [9],
then typed empty; interleaved missing/schema/cancel/row-limit failures do not leak
rows. Public DF/SQL paths in Rust/Python use native union, preserve duplicate/NULL
multiplicity and per-source FIFO, support shared DAG inputs, and finish only after
all sources EOF. Latest snapshot rejection checks untouched Source acquisition,
binding resolution, state root and job creation, including warmed preparation.
Legacy SQL coercion/topology/digests and snapshot append 7 versus finite 6 retain
their contract. The standalone consumer also retains predecessor aggregate/state
and export/reload oracles.

Not run locally: full Rust/Python suites, all-features workspace Clippy/rustdoc,
90% coverage, connector services, benchmarks, release wheel matrix, or other
platforms. These are not reported as passing; regular CI and independent review
remain required before merge. No merge or parent-issue close was performed.

## Commits and handoff

Product/docs/frozen-note commit:
`3b24bcea192a3a05e923c9e71582d5cef4687bdb`
(`feat: add finite union across relation frontends`, 45 files).
All 1,667 recorded source/build files are tracked, match their verification
hashes, and have no diff against that commit. Component tree hashes are recorded
in finite-union.json. The verification record is committed separately without
changing those component trees; the issue handoff identifies the final HEAD.

The inherited checkout lacked Git author identity, so the successful commit used
`git -c user.name='Cheng Li' -c user.email='scrappedprince.li@gmail.com' commit ...`
with this turn's on-behalf-of identity. No global or shared Git identity changed.
The frozen note's contract text was preserved; only its extra trailing blank
line was trimmed after the staged whitespace check identified it. The subsequent
`git diff --cached --check` passed. The first unconfigured-identity commit attempt
failed before creating any commit.

Target is main, one PR titled with DAL-515 and body containing Summary/Test plan
and Closes DAL-515. Only one nonblocking post-push CI snapshot is permitted; its
actual result belongs to the final issue comment, after this evidence commit.
DAL-515 remains in_progress by the explicit orchestrator instruction. Independent
review and required checks are pending handoff responsibilities; no merge is run.
