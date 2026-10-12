## Branch Review: public Runtime registration API

**Branch:** `feature/public-runtime-registration-api` → `main`
**Base:** `8f890f287d44fa0c5ce8c79d3d23e595f8563dcd`
**Reviewed:** 2026-10-13, the uncommitted working tree (12 modified files) plus
the untracked API note and critique. There is no PR, so no check runs or prior
GitHub reviews exist to consult.
**Files:** 14

### Summary

Three underscore-private `calc_flow.Runtime` methods become public
(`register_mapping_provider`, `registration_snapshot`,
`capability_registration_snapshot`) with no aliases, so Studio can depend on
them through a published `calc-flow-python`. Every in-repository caller moves
to the public names, new core tests pin the record, copy, replay, pair, and
failure contracts, and the published docs gain a `Trusted registration
records` section.

The source change is a pure rename plus docstrings, the migration is complete,
the tests fail on the baseline and catch the regressions that matter, and the
published claims match the source. One mechanical Markdown fix is required in
the API note; everything else is optional.

### Build and Test Results

- Rust: **Not touched**
- Python: **Passed**. `test_runtime.py`, `test_execution_options.py`,
  `test_capabilities.py`, `test_array.py`: 347 passed, 1 skipped (the Kafka
  record test, `requires connector-kafka`). Matches the implementer's result.
- Studio backend: **Passed**. `tests/test_run_manager.py`: 124 passed. Matches
  the implementer's result.
- Studio frontend: **Not touched**
- Lint: `ruff check` and `ruff format --check` clean on the nine changed
  Python files; `git diff --check` clean.
- New failures: None
- Regressions: None
- Not run: full Python and Studio suites, the Rust 90% and Studio 85% coverage
  gates, Rust, frontend, benchmarks, and CPython 3.9 to 3.12 and 3.14. These
  belong to CI; nothing here was pushed, so there is no CI state to report.

### Blocking Issues

None in source, tests, or published documentation.

### Style Issues

- **`.codex/artifacts/api-notes/public-runtime-registration-api.md` — the
  tables under `Registration Record Contract` and `Error Cases`**: both
  separator rows use spaced dashes (`| --- |`). The Markdown rule requires
  dashes that span the full column width including the padding spaces, which
  is how the published copy of the same table in `docs/python-api.md` is
  written. Rewrite the two separator rows before the note is committed.
  Owner: `cf-api-designer`. This is the only item that gates approval; it
  needs no re-review.

### Judgments On The Review Questions

1. **Behavior preservation: confirmed.** After mapping the three names and
   dropping the docstrings of the three methods, the syntax trees of
   `pipeline.py`, `array.py`, `symbolic_support.py`, and `run_manager.py` are
   identical to the baseline. Lock scope, copying, and error paths are
   untouched. `Runtime.capabilities` now calls the public pair method, which
   is the same function.
2. **Migration completeness: confirmed.** The boundary-aware search matches
   only the native seam (`config.rs`, `_native.pyi`, the one native call
   inside `Runtime.register_mapping_provider`), the two native-seam tests in
   `test_execution_options.py`, the API note, and the `CHANGELOG.md` migration
   bullet. Studio source imports only public `calc_flow` names, and every
   underscore attribute access in it has `self` as its receiver with none
   chained through the held runtime. Studio tests reach only Studio's own
   private members. `counting_matmul_runtime` in the benchmark support module
   was executed once and registers through the public name.
3. **Test quality: sound.** No new test relates a revision to a record count:
   one compares a revision with its own earlier value, the other asserts it
   is higher than an earlier one. No new test mutates caller-owned input,
   sleeps, starts a thread, or adds a shared `conftest.py`; names follow
   `test_<behavior>`. Against baseline `pipeline.py` and `array.py` the module
   is red (25 failed, 4 passed, 1 skipped), and ten out-of-source mutants are
   all caught (table below).
   - (a) **Adequate.** The double in
     `test_capabilities_reject_a_malformed_runtime_snapshot` defines only the
     public pair method, and a private call would raise `AttributeError`,
     which `pytest.raises(CapabilitySnapshotError)` does not accept. Reverting
     either read in `RunManager.capabilities` fails the same four tests that
     build a real `Runtime`, so the success path is covered by name. The
     acceptance item was worded for a rejection test; the implementer kept the
     test's intent.
   - (b) **Accepted as stated residual risk.** The Kafka record test has never
     run: the local native module compiles only the `file` connector, and no
     workflow runs pytest on a `connector-kafka` build. Its body passes when
     the native call is stubbed, so the assertions and the Python-layer record
     are right; the native half remains verified by reading only.
   - (c) **Appropriate.** Calc Flow's own messages are matched in full; the
     unpack error and the PyO3 argument error are matched by the fragment that
     is stable across CPython and PyO3 versions.
   - (d) **Correct.** The method is no longer private, the assertions are
     unchanged, and nothing else references the old test name.
4. **Documentation accuracy: confirmed, with two wording notes below.** Every
   record key, type, and presence rule matches the four registration methods.
   The copy, pair, ordering, monotonic-revision, and lifecycle-limit claims
   were reproduced. The section states no relation between a revision and a
   record count, no starting revision, no prefix or append-only relation, and
   does not call the table exact or closed. The published example runs. The
   table is aligned, the new anchor and both links resolve, and there is no
   trailing whitespace or missing final newline.
   - **"Today, equal `(scope.session_id, scope.revision)` means the same
     registrations in the same order with the same callables."** True for one
     `Runtime`: the revision is the length of a list that is only appended to,
     read under the registration lock, and no path replaces a record or its
     callable. It is safe to publish beside the lifecycle limit, because both
     open fix routes keep it true. It is not true across `os.fork()`: the
     child inherits the session ID, and parent and child were observed at the
     same key holding different providers. Scope the sentence to one runtime.
5. **CHANGELOG: correct.** One bullet at the top of `## 2026-10` names the
   public methods, the removed spellings, and the migration. Repository
   practice is the authoring day, not the landing day: of 117 bullet and
   commit pairs in history, 84 share a day and 32 bullets are dated before
   the commit that landed them (squash merges keep the branch's dates); the
   one bullet dated after its commit was corrected back. Keep `2026-10-12`.
6. **Scope discipline: confirmed.** Nothing changed under `crates/`,
   `_native.pyi`, `__init__.py`, `web-ui/openapi.json`,
   `web-ui/src/api/schema.d.ts`, or `schemas/project-v3.schema.json`. Studio
   differs from the baseline only at the four call sites. No generated native
   module is in source.
7. **Release consequence: nothing to change now.**
   `project_configuration` in `scripts/verify_python_release.py` and
   `test_release_versions_are_calver_and_aligned` both require Studio's
   requirement to equal `calc-flow-python>=<project version>,<next year>`.
   Raising the floor here without moving the version would fail both, and
   moving the version is the release's job; the next release then raises the
   floor to a version that has the public names. Two consequences follow from
   the no-alias decision and should be known at release time: Studio at this
   commit does not work with the wheel built from tag
   `calc-flow-python-v2026.10.12`, and any Studio build at or before that tag
   stops working with the next core release although its `<2027` range admits
   it. The split-out repository must start its floor at the first release
   that carries the public names.

### Test Coverage

| Copy mutated     | Mutation                                                | Tests failing |
|------------------|---------------------------------------------------------|---------------|
| `pipeline.py`    | `registration_snapshot` returns the runtime's own dicts | 3             |
| `pipeline.py`    | record appended before the native registration          | 4             |
| `pipeline.py`    | port sequences stored without copying                   | 3             |
| `pipeline.py`    | mapping record always carries `accepts_context`         | 1             |
| `pipeline.py`    | pair returns fewer records than it describes            | 1             |
| `pipeline.py`    | underscore alias of `registration_snapshot` kept        | 1             |
| `run_manager.py` | `_restore_registrations` uses the private name          | 3             |
| `run_manager.py` | `RunManager.capabilities` first read uses it            | 5             |
| `run_manager.py` | `RunManager.capabilities` revalidation uses it          | 4             |
| `run_manager.py` | `RunManager.submit` uses the private name               | 4             |

- Covered: signature, docstrings, alias removal, record shapes for single,
  context, mapping, and scalar UDF registrations, fresh containers with shared
  values, all eight failure rows leaving records and revision unchanged,
  replay onto a fresh runtime, pair consistency, and batch-only replay of the
  `register_numpy` records.
- Not covered by any executed test: the `kafka_decoder` record rows.
- Not testable by design, as the API note records: a locking regression in
  the pair method, because both values come from one local copy.

### Non-Blocking Notes

- `docs/python-api.md`, the "Today, equal" sentence: begin it with "On one
  `Runtime`," so it cannot be read as holding across a forked process.
  Owner: `cf-doc-writer`.
- `CHANGELOG.md`, the new bullet: it shares its date with the `2026.10.12`
  release bullet directly below it, but that release has only the private
  spellings. Add that the public names first ship in the release after
  `2026.10.12`. Owner: `cf-doc-writer`.
- `docs/python-api.md`, "Port and input-type tuples are the runtime's own
  immutable copies": an `input_types` argument that is already a `tuple` is
  stored as that same object. Say "immutable tuples built from the caller's
  sequences". The difference is visible only to `is`. Owner: `cf-doc-writer`.
- `docs/python-api.md`, "It is not for the `expression@1` and
  `symbolic_matrix@1` providers": reads as "not intended for". "That is not
  the whole registration for" is clearer. Owner: `cf-doc-writer`.
- `test_registration_records_match_the_published_contract`: whole-record
  equality is the regression pin of acceptance item 2 and is stricter than
  the published tolerant-reader rule. A one-line comment saying so keeps a
  later additive key from being read as a contract break. Owner:
  `cf-implementer`.
- `test_failed_mapping_registration_leaves_no_record_and_keeps_the_revision`,
  the positional-ports case: it matches CPython's full "takes 5 positional
  arguments but 7 were given"; "positional argument" pins the same contract
  with less of CPython's wording. Owner: `cf-implementer`.
- `test_kafka_decoder_record_matches_the_published_contract`: track a pytest
  job on a `connector-kafka` build so the test and the `kafka_decoder` rows
  stop being read-verified only. Owner: `cf-tester`, as a follow-up.
- API note, acceptance item 18: its list of allowed matches omits
  `CHANGELOG.md`, which item 17 necessarily adds. Owner: `cf-api-designer`.

### Documentation Consistency

- `docs/python-api.md` and `docs/api-reference.md` are consistent with each
  other and with the source; no other document names the removed spellings or
  describes Studio's use of them.
- `AGENTS.md` needs no change: no command or architecture statement moved.
- Between releases the guide's opening still says it describes version
  `2026.10.12` in the current checkout. That is the repository's standing
  wording and is corrected by the next release, not by this change.

### Commands Run

All commands ran from the worktree with the main checkout's interpreter. The
prebuilt native module was made importable through a symlink overlay outside
the source tree, since removed.

- `git grep -n --untracked -P
  '(?<![A-Za-z0-9_])_(register_mapping_provider|registration_snapshot|capability_registration_snapshot)(?![A-Za-z0-9_])'`:
  17 lines, all in the allowed locations.
- `python -m pytest python/tests/test_runtime.py
  python/tests/test_execution_options.py python/tests/test_capabilities.py
  python/tests/test_array.py -q -p no:cacheprovider -rs`: 347 passed,
  1 skipped.
- From `web-ui/backend`: `python -m pytest tests/test_run_manager.py -q
  -p no:cacheprovider -rs`: 124 passed.
- `ruff check --no-cache` and `ruff format --no-cache --check` on the nine
  changed Python files: clean.
- `git diff --check`: clean.
- `python -m unittest scripts.test_release_config -k doc -k table -k calver
  -k version -k changelog -k markdown`: 9 passed.
- `python -m pytest python/tests/test_runtime.py` against baseline copies of
  `pipeline.py` and `array.py`: 25 failed, 4 passed, 1 skipped.
- The same module against six mutated copies of `pipeline.py`, and
  `tests/test_run_manager.py` against four mutated copies of
  `run_manager.py`: every mutant failed (table above).
- Scratch scripts, outside the source tree: a normalized syntax-tree
  comparison of the four source files; execution of the published example;
  probes of the copy, pair, lifecycle-limit, and fork behavior; the Kafka
  record test body with the native call stubbed; Markdown table, whitespace,
  final-newline, and link checks; and a tally of changelog bullet dates
  against commit dates.

### Worktree State

`git status` lists the same 12 modified files and 2 untracked artifacts as
before the review, plus this report. The ignored-file set is unchanged, the
source tree holds no generated native module, cache directory, or symlink,
and the overlays are removed. This report is the only file the review wrote.

### Verdict

**Approve with non-blocking notes**, on the condition that the API note's two
table separator rows are rewritten before the commit. Source, tests, and
published documentation are approved as they stand.
