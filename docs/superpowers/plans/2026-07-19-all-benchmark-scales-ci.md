# All-Scale Benchmark CI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run every supported benchmark scale in pull-request and main-branch CI with an isolated JSON report for each scale.

**Architecture:** Convert the existing `benchmark-smoke` job into a fail-fast-disabled GitHub Actions matrix whose ordered scale list matches `benchmarks.support.SCALES`. Add a focused release-configuration regression test that derives the canonical scale names from `benchmarks/support.py` without importing the benchmark package, then checks the CI matrix, environment selection, result path, and artifact name.

**Tech Stack:** GitHub Actions YAML, Python 3.13 standard-library `ast` and `unittest`, pytest-benchmark, uv

## Global Constraints

- Work only on the isolated `ci/all-benchmark-scales` branch based on current `main`.
- Preserve all pinned action SHAs and unrelated workflow behavior.
- Keep benchmark timing changes informational; this task only broadens the executed scale set and retained reports.
- Treat `benchmarks/support.py::SCALES` as the canonical ordered list: `overhead`, `small`, `standard`, `nightly`.
- Follow test-driven development: add and observe the focused test failing before editing the workflow.
- Stage and commit only the plan, focused regression test, and benchmark workflow change.

---

### Task 1: Add a failing all-scale workflow contract test

**Files:**

- Modify: `scripts/test_release_config.py`
- Read: `benchmarks/support.py:62`
- Test: `scripts/test_release_config.py`

- [ ] **Step 1: Add the focused regression test**

Add `ast` to the standard-library imports and add this method to `ReleaseConfigTests`:

```python
def test_benchmark_smoke_runs_every_supported_scale(self) -> None:
    support_tree = ast.parse((ROOT / "benchmarks/support.py").read_text())
    scales_assignment = next(
        node
        for node in support_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "SCALES"
            for target in node.targets
        )
    )
    self.assertIsInstance(scales_assignment.value, ast.Dict)
    scales = [ast.literal_eval(key) for key in scales_assignment.value.keys]

    workflow = (ROOT / ".github/workflows/ci.yml").read_text()
    benchmark_job = workflow.split("  benchmark-smoke:\n", 1)[1].split(
        "  rust-core:\n", 1
    )[0]

    self.assertEqual(scales, ["overhead", "small", "standard", "nightly"])
    self.assertIn("fail-fast: false", benchmark_job)
    self.assertIn(f"scale: [{', '.join(scales)}]", benchmark_job)
    self.assertIn(
        "CALC_FLOW_BENCHMARK_SCALE: ${{ matrix.scale }}", benchmark_job
    )
    self.assertIn("JAX_PLATFORMS: cpu", benchmark_job)
    self.assertIn(
        '--benchmark-json="benchmark-results/${CALC_FLOW_BENCHMARK_SCALE}.json"',
        benchmark_job,
    )
    self.assertIn("name: benchmark-smoke-${{ matrix.scale }}", benchmark_job)
    self.assertIn("path: benchmark-results/${{ matrix.scale }}.json", benchmark_job)
```

This parses the Python source instead of importing `benchmarks.support`, so the release configuration test does not require NumPy, PyArrow, or the native `calc_flow` module.

- [ ] **Step 2: Run the focused test and confirm the RED state**

Run:

```bash
python -m unittest \
  scripts.test_release_config.ReleaseConfigTests.test_benchmark_smoke_runs_every_supported_scale
```

Expected: `FAIL` because the current CI job has no `fail-fast: false` matrix and fixes `CALC_FLOW_BENCHMARK_SCALE` to `overhead`.

- [ ] **Step 3: Inspect the failure**

Confirm the failure is an assertion about the missing matrix contract, not a syntax, import, or fixture error. Do not edit the expected scale list to make the old workflow pass.

---

### Task 2: Convert benchmark smoke to an all-scale matrix

**Files:**

- Modify: `.github/workflows/ci.yml:244`
- Test: `scripts/test_release_config.py`

- [ ] **Step 1: Add the matrix and per-scale job identity**

Change the job header to:

```yaml
  benchmark-smoke:
    name: Informational ${{ matrix.scale }} benchmark smoke
    needs: lint-and-test
    runs-on: ubuntu-latest
    timeout-minutes: 60
    strategy:
      fail-fast: false
      matrix:
        scale: [overhead, small, standard, nightly]
    env:
      CALC_FLOW_BENCHMARK_SCALE: ${{ matrix.scale }}
      JAX_PLATFORMS: cpu
```

The four runners remain independent, and `fail-fast: false` lets the other scale reports finish if one runner fails.

- [ ] **Step 2: Make benchmark output and artifact names scale-specific**

Replace the fixed overhead step with:

```yaml
      - name: Run benchmark scale
        run: |
          mkdir -p benchmark-results
          uv run pytest benchmarks -q --benchmark-only \
            --benchmark-json="benchmark-results/${CALC_FLOW_BENCHMARK_SCALE}.json"
      - name: Upload benchmark report
        uses: actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7
        with:
          name: benchmark-smoke-${{ matrix.scale }}
          path: benchmark-results/${{ matrix.scale }}.json
          if-no-files-found: error
          retention-days: 30
```

Keep the checkout, Python, uv, dependency installation, upload action pin, and retention period unchanged.

- [ ] **Step 3: Run the focused test and confirm the GREEN state**

Run:

```bash
python -m unittest \
  scripts.test_release_config.ReleaseConfigTests.test_benchmark_smoke_runs_every_supported_scale
```

Expected: `OK` with one test passing.

- [ ] **Step 4: Run the complete release configuration tests**

Run:

```bash
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
```

Expected: all release helper tests pass.

- [ ] **Step 5: Commit the tested workflow change**

Run:

```bash
git add .github/workflows/ci.yml scripts/test_release_config.py
git diff --cached --check
git commit -m "ci: run all benchmark scales"
```

Expected: one narrow commit containing only the workflow and its regression test.

---

### Task 3: Verify the branch and publish the separate PR

**Files:**

- Verify: `.github/workflows/ci.yml`
- Verify: `scripts/test_release_config.py`
- Verify: `docs/superpowers/specs/2026-07-19-all-benchmark-scales-ci-design.md`
- Verify: `docs/superpowers/plans/2026-07-19-all-benchmark-scales-ci.md`

- [ ] **Step 1: Run scoped formatting and configuration checks**

Run:

```bash
UV_CACHE_DIR=/tmp/calc-flow-all-benchmark-scales-uv-cache uv run ruff check scripts/test_release_config.py
UV_CACHE_DIR=/tmp/calc-flow-all-benchmark-scales-uv-cache uv run ruff format --check scripts/test_release_config.py
python -m unittest scripts.test_inspect_wheel scripts.test_release_config
git diff --check
```

Expected: Ruff and every unittest pass, with no whitespace errors.

- [ ] **Step 2: Review exact branch scope**

Run:

```bash
git status --short --branch
git log --oneline origin/main..HEAD
git diff --stat origin/main...HEAD
git diff --check origin/main...HEAD
```

Expected: only the design document, implementation plan, CI workflow, and release configuration test differ from `origin/main`.

- [ ] **Step 3: Push and create a draft pull request**

Push `ci/all-benchmark-scales` and open a draft PR targeting `main` with title `ci: run all benchmark scales`. The PR body must contain:

```markdown
## Summary

- run the benchmark smoke job as an overhead, small, standard, and nightly matrix
- keep scale reports isolated with scale-specific JSON files and artifacts
- guard the workflow against drift from the canonical benchmark scale set

## Test plan

- `python -m unittest scripts.test_inspect_wheel scripts.test_release_config`
- `uv run ruff check scripts/test_release_config.py`
- `uv run ruff format --check scripts/test_release_config.py`
- `git diff --check origin/main...HEAD`
```

- [ ] **Step 4: Verify the remote head and check rollup**

Confirm the PR head SHA equals the pushed local `HEAD`. Inspect the GitHub Actions check rollup and report each benchmark matrix runner separately. Do not claim remote success while checks are queued or running.
