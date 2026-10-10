"""Closed, benchmark-only source overlay for the eight-row acceptance task."""

from __future__ import annotations

import hashlib
import json
import shlex
import tomllib
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.benchmark_suite.process import ROOT
from scripts.benchmark_suite.rust_harness import _clean_revision
from scripts.toolkit import command_output, fingerprint_json, sha256_file, write_json

SCHEMA = "calc-flow.issue406-native-harness.v1"
TARGETS = ("stream_join_perf", "stream_asof_e2e")
WHEEL_TARGET = "calc_flow_python"
COPIED_FILES = (
    "crates/calc-flow/benches/stream_join_perf.rs",
    "crates/calc-flow/benches/stream_asof_e2e.rs",
    "crates/calc-flow/benches/support/native_join_components.rs",
    "crates/calc-flow/src/benchmark_internals.rs",
    "crates/calc-flow/src/benchmark_internals/dispatch_trace.rs",
    "crates/calc-flow/src/continuous/benchmark_trace.rs",
    "crates/calc-flow/src/runtime/streaming/operator_task/benchmark_replay.rs",
    "crates/calc-flow/src/runtime/streaming/operator_task/benchmark_replay/dispatch.rs",
)
INSERTIONS = {
    "crates/calc-flow/Cargo.toml": (
        "\n[features]\nbenchmark-internals = []\n"
        'benchmark-checkpoint-off = ["benchmark-internals"]\n'
    ),
    "crates/calc-flow/src/lib.rs": (
        '\n#[cfg(feature = "benchmark-internals")]\n#[doc(hidden)]\n'
        "pub mod benchmark_internals;\n"
    ),
    "crates/calc-flow/src/runtime/streaming/operator_task.rs": (
        '\n#[cfg(feature = "benchmark-internals")]\n'
        '#[path = "operator_task/benchmark_replay.rs"]\n'
        "pub(crate) mod benchmark_replay;\n"
    ),
    "crates/calc-flow/src/runtime/streaming/mod.rs": (
        '\n#[cfg(feature = "benchmark-internals")]\n'
        "pub(crate) use operator_task::benchmark_replay;\n"
    ),
    "crates/calc-flow/src/continuous.rs": (
        '\n#[cfg(feature = "benchmark-internals")]\nmod benchmark_trace;\n'
    ),
}

LITERAL_PATCHES = {
    "crates/calc-flow/src/runtime/streaming/context.rs": (
        (
            "    gather: super::gather_work::JobGatherOwner,\n",
            "    gather: super::gather_work::JobGatherOwner,\n"
            '    #[cfg(feature = "benchmark-internals")]\n'
            "    dispatch_trace: crate::benchmark_internals::DispatchTraceOwner,\n",
        ),
        (
            "            gather: super::gather_work::JobGatherOwner::new("
            "job_id.to_string().into()),\n",
            "            gather: super::gather_work::JobGatherOwner::new("
            "job_id.to_string().into()),\n"
            '            #[cfg(feature = "benchmark-internals")]\n'
            "            dispatch_trace: "
            "crate::benchmark_internals::DispatchTraceOwner::default(),\n",
        ),
        (
            "    pub(crate) fn with_gather_owner(mut self, "
            "gather: super::gather_work::JobGatherOwner) -> Self {\n",
            '    #[cfg(feature = "benchmark-internals")]\n'
            "    pub(crate) fn dispatch_trace(&self) -> "
            "&crate::benchmark_internals::DispatchTraceOwner {\n"
            "        &self.dispatch_trace\n    }\n\n"
            "    pub(crate) fn with_gather_owner(mut self, "
            "gather: super::gather_work::JobGatherOwner) -> Self {\n",
        ),
    ),
    "crates/calc-flow/src/continuous.rs": (
        (
            "        let spec = ContinuousJobSpec {\n",
            '        #[cfg(feature = "benchmark-internals")]\n'
            "        let benchmark_context = context.clone();\n"
            "        let spec = ContinuousJobSpec {\n",
        ),
        (
            "            .map(|inner| StreamingJob {\n                inner,\n",
            "            .map(|inner| StreamingJob {\n                inner,\n"
            '                #[cfg(feature = "benchmark-internals")]\n'
            "                benchmark_context,\n",
        ),
        (
            "    inner: OwningContinuousJob,\n",
            "    inner: OwningContinuousJob,\n"
            '    #[cfg(feature = "benchmark-internals")]\n'
            "    benchmark_context: StreamJobContext,\n",
        ),
    ),
    "crates/calc-flow/src/runtime/streaming/operator_task.rs": (
        (
            "    let result = dispatch_message_inner(\n",
            '    #[cfg(feature = "benchmark-internals")]\n'
            "    let benchmark_watermark = message.as_watermark();\n"
            "    let result = dispatch_message_inner(\n",
        ),
        (
            "    .await;\n    observe_join_status(inputs, input_progress)?;\n",
            """    .await;
    #[cfg(feature = "benchmark-internals")]
    if result.is_ok() {
        if let Some(watermark) = benchmark_watermark {
            inputs
                .context
                .job()
                .dispatch_trace()
                .complete_watermark(&inputs.node_id, watermark.as_micros());
        }
    }
    observe_join_status(inputs, input_progress)?;
""",
        ),
        (
            "    match message.kind() {\n        StreamMessageKind::Data => {\n"
            "            let watermark = input_progress.input_watermark();\n",
            """    #[cfg(feature = "benchmark-internals")]
    if matches!(inputs.operator, CompiledStreamOperator::StreamAsofJoin(_)) {
        inputs
            .context
            .job()
            .dispatch_trace()
            .record(&inputs.node_id, ingress_name, &message)?;
    }
    match message.kind() {
        StreamMessageKind::Data => {
            let watermark = input_progress.input_watermark();
""",
        ),
        (
            "    result?;\n    inputs\n        .metrics\n"
            "        .observe_operator_window_metrics("
            "&inputs.node_id, &inputs.progress.snapshot())?;\n",
            """    result?;
    #[cfg(feature = "benchmark-internals")]
    if matches!(inputs.operator, CompiledStreamOperator::StreamAsofJoin(_)) {
        inputs
            .context
            .job()
            .dispatch_trace()
            .callback(&inputs.node_id, "data")?;
    }
    inputs
        .metrics
"""
            + "        .observe_operator_window_metrics("
            "&inputs.node_id, &inputs.progress.snapshot())?;\n",
        ),
        (
            "    result?;\n    inputs\n        .metrics\n"
            "        .record_operator_watermark(&inputs.node_id, &processing_timer)\n",
            """    result?;
    #[cfg(feature = "benchmark-internals")]
    if matches!(inputs.operator, CompiledStreamOperator::StreamAsofJoin(_)) {
        inputs
            .context
            .job()
            .dispatch_trace()
            .callback(&inputs.node_id, "watermark")?;
    }
    inputs
        .metrics
        .record_operator_watermark(&inputs.node_id, &processing_timer)
""",
        ),
        (
            "    progress_result?;\n    let emit_idle = apply_progress_emissions(\n",
            """    progress_result?;
    #[cfg(feature = "benchmark-internals")]
    if matches!(inputs.operator, CompiledStreamOperator::StreamAsofJoin(_)) {
        inputs
            .context
            .job()
            .dispatch_trace()
            .callback(&inputs.node_id, "ingress_progress")?;
    }
    let emit_idle = apply_progress_emissions(
""",
        ),
    ),
    "crates/calc-flow-python/Cargo.toml": (
        (
            "[features]\n",
            '[features]\nbenchmark-internals = ["calc-flow/benchmark-internals"]\n',
        ),
    ),
    "crates/calc-flow-python/src/continuous.rs": (
        (
            "    fn _take_callback_profile(&self) -> PyResult<String> {\n",
            """    #[cfg(feature = "benchmark-internals")]
    #[pyo3(signature = (allow_barriers=false))]
    fn _benchmark_start_dispatch_trace(&self, allow_barriers: bool) -> PyResult<()> {
        self.job()?
            .benchmark_start_dispatch_trace(allow_barriers)
            .map_err(streaming_py_err)
    }

    #[cfg(feature = "benchmark-internals")]
    fn _benchmark_take_dispatch_trace(&self) -> PyResult<String> {
        let trace = self
            .job()?
            .benchmark_take_dispatch_trace()
            .map_err(streaming_py_err)?;
        let encoded = serde_json::to_string(&trace);
        encoded.map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    #[cfg(feature = "benchmark-internals")]
    fn _benchmark_dispatch_frontier(&self, node_id: &str) -> PyResult<Option<i64>> {
        Ok(self.job()?.benchmark_dispatch_frontier(node_id))
    }

    fn _take_callback_profile(&self) -> PyResult<String> {
""",
        ),
    ),
}


def _files(source: Path) -> dict[str, str]:
    paths = command_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=source
    ).splitlines()
    return {path: sha256_file(source / path) for path in sorted(set(paths))}


def _effective(source: Path) -> dict[str, bytes]:
    effective = {path: (ROOT / path).read_bytes() for path in COPIED_FILES}
    for path, insertion in INSERTIONS.items():
        original = (source / path).read_text()
        if insertion in original:
            if original.count(insertion) != 1:
                raise ValueError("duplicate benchmark feature insertion")
            effective[path] = original.encode()
            continue
        if path.endswith(".toml") and "[features]" in original:
            raise ValueError("unreviewed product feature table")
        if "benchmark-internals" in original or "benchmark_replay" in original:
            raise ValueError("unreviewed benchmark feature declaration")
        effective[path] = (original + insertion).encode()
    for path, patches in LITERAL_PATCHES.items():
        contents = effective.get(path, (source / path).read_bytes()).decode()
        for before, after in patches:
            if after in contents:
                if contents.count(after) != 1:
                    raise ValueError("duplicate benchmark literal patch")
            elif contents.count(before) == 1:
                contents = contents.replace(before, after, 1)
            else:
                raise ValueError(f"benchmark literal anchor is not unique: {path}")
        effective[path] = contents.encode()
    return effective


def _lock_identity(source: Path) -> str:
    lock = tomllib.loads((source / "Cargo.lock").read_text())
    packages = [
        {
            **package,
            **(
                {"dependencies": sorted(package["dependencies"])}
                if "dependencies" in package
                else {}
            ),
        }
        for package in lock.get("package", [])
    ]
    return fingerprint_json({**lock, "package": sorted(packages, key=fingerprint_json)})


def _verify_mirror(mirror: Path, original: dict, effective: dict, lock: str) -> None:
    actual = _files(mirror)
    expected = {**original, **effective}
    # Cargo may reorder lock keys without changing its dependency graph.
    if _lock_identity(mirror) != lock:
        raise ValueError("native harness changed the product dependency lock")
    actual.pop("Cargo.lock", None)
    expected.pop("Cargo.lock", None)
    if actual != expected:
        raise ValueError("native harness build changed unapproved product files")


@contextmanager
def common_acceptance_harness(source: Path, output: Path, target: str):
    """Build an exact product with only the reviewed fixed benchmark overlay."""
    if target not in (*TARGETS, WHEEL_TARGET):
        raise ValueError("unsupported issue406 native target")
    source, output = source.resolve(), output.resolve()
    product_sha, harness_sha = _clean_revision(source), _clean_revision(ROOT)
    tracked = command_output(["git", "ls-files"], cwd=ROOT).splitlines()
    if not set(COPIED_FILES).issubset(tracked):
        raise ValueError("native helper files must belong to the harness revision")
    original = _files(source)
    overlay = _effective(source)
    lock = _lock_identity(source)
    output.mkdir(parents=True, exist_ok=True)
    evidence = output / f"build-{target}.acceptance-harness.json"
    if evidence.exists():
        raise ValueError("native harness requires fresh build evidence")
    with TemporaryDirectory(prefix=f"{target}-", dir=output) as raw:
        mirror = Path(raw) / "source"
        command_output(
            [
                "git",
                "clone",
                "--shared",
                "--no-checkout",
                "--quiet",
                str(source),
                str(mirror),
            ]
        )
        command_output(
            [
                "git",
                "-c",
                "filter.lfs.process=",
                "-c",
                "filter.lfs.smudge=",
                "-c",
                "filter.lfs.required=false",
                "checkout",
                "--detach",
                "--quiet",
                product_sha,
            ],
            cwd=mirror,
        )
        if _files(mirror) != original:
            raise ValueError("owned mirror differs from exact product files")
        for path, contents in overlay.items():
            destination = mirror / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(contents)
        effective = {path: sha256_file(mirror / path) for path in overlay}
        record = {
            "schema": SCHEMA,
            "target": target,
            "targets": list(TARGETS),
            "product_git_sha": product_sha,
            "harness_git_sha": harness_sha,
            "product_source": str(source),
            "build_source": str(mirror),
            "build_output": str(output),
            "original_files_fingerprint": fingerprint_json(original),
            "original_overlay": {path: original.get(path) for path in overlay},
            "effective_files": effective,
            "lock_fingerprint": lock,
            "product_files_verified": False,
        }
        write_json(evidence, record)
        _verify_mirror(mirror, original, effective, lock)
        yield mirror
        _verify_mirror(mirror, original, effective, lock)
        write_json(evidence, {**record, "product_files_verified": True})


def validate_acceptance_harness(output: Path, target: str) -> dict:
    """Revalidate the persisted product, helper, and closed insertion recipe."""
    try:
        record = json.loads(
            (output / f"build-{target}.acceptance-harness.json").read_text()
        )
        source = Path(record["product_source"])
        mirror = Path(record["build_source"])
        original = _files(source)
        overlay = _effective(source)
        if record["harness_git_sha"] != _clean_revision(ROOT):
            native_harness_reuse(record["harness_git_sha"])
        expected = {
            "schema": SCHEMA,
            "target": target,
            "targets": list(TARGETS),
            "product_git_sha": _clean_revision(source),
            "harness_git_sha": record["harness_git_sha"],
            "build_output": str(output.resolve()),
            "original_files_fingerprint": fingerprint_json(original),
            "original_overlay": {path: original.get(path) for path in overlay},
            "effective_files": {
                path: hashlib.sha256(data).hexdigest() for path, data in overlay.items()
            },
            "lock_fingerprint": _lock_identity(source),
            "product_files_verified": True,
        }
        if (
            target not in (*TARGETS, WHEEL_TARGET)
            or any(record.get(key) != value for key, value in expected.items())
            or mirror.name != "source"
            or not mirror.parent.name.startswith(f"{target}-")
            or mirror.parent.parent != output.resolve()
        ):
            raise ValueError("mismatched evidence")
        return record
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid issue406 native harness attestation") from error


def native_harness_reuse(build_revision: str) -> dict:
    """Prove a Python-only correction leaves the historical native overlay intact."""
    current = _clean_revision(ROOT)
    history = command_output(["git", "rev-list", current], cwd=ROOT).splitlines()
    if build_revision not in history:
        raise ValueError("native build harness is missing from execution history")
    recipe_path = "scripts/benchmark_suite/issue406_harness.py"
    boundary = "\ndef validate_acceptance_harness("
    historical = command_output(
        ["git", "show", f"{build_revision}:{recipe_path}"], cwd=ROOT
    ).split(boundary)[0]
    active = (ROOT / recipe_path).read_text().split(boundary)[0]
    if historical != active:
        raise ValueError("historical native overlay recipe changed")
    blobs = {}
    for revision in (build_revision, current):
        values = command_output(
            ["git", "rev-parse", *(f"{revision}:{path}" for path in COPIED_FILES)],
            cwd=ROOT,
        ).splitlines()
        blobs[revision] = dict(zip(COPIED_FILES, values, strict=True))
    if blobs[build_revision] != blobs[current]:
        raise ValueError("historical copied native sources changed")
    return {
        "original_build_harness_git_sha": build_revision,
        "execution_harness_git_sha": current,
        "recipe_prefix_sha256": hashlib.sha256(active.encode()).hexdigest(),
        "copied_file_git_blobs": blobs[current],
    }


def validate_native_profile(artifacts: list[dict], *, checkpoint_off: bool) -> None:
    """Reject test-configured or non-release core code in component measurements."""
    expected_features = ["benchmark-internals"]
    if checkpoint_off:
        expected_features.append("benchmark-checkpoint-off")
    core = [
        row
        for row in artifacts
        if row["package"].get("workspace_package") == "crates/calc-flow"
        and row["target"]["kind"] == ["lib"]
    ]
    if len(core) != 1 or any(
        row.get("profile", {}).get("test") is not False
        or row["profile"].get("opt_level") != "3"
        or sorted(row.get("features", [])) != sorted(expected_features)
        for row in core
    ):
        raise ValueError("native benchmark requires the declared release core features")


def validate_traced_wheel_invocations(record: dict) -> None:
    """Require actual release compilation of the feature-only core and binding."""
    for key, crate, features in (
        ("actual_core_invocation", "calc_flow", {"benchmark-internals"}),
        (
            "actual_binding_invocation",
            "calc_flow_python",
            {"benchmark-internals", "connector-file", "default"},
        ),
    ):
        try:
            line = record[key]
            prefix, suffix = (
                line.index("Running `") + len("Running `"),
                line.rindex("`"),
            )
            tokens = shlex.split(line[prefix:suffix])
            cfg = [
                tokens[index + 1]
                for index, token in enumerate(tokens)
                if token == "--cfg"
            ]
            actual_features = {
                value.removeprefix('feature="').removesuffix('"')
                for value in cfg
                if value.startswith('feature="')
            }
            if (
                tokens[tokens.index("--crate-name") + 1] != crate
                or not any(
                    tokens[index : index + 2] == ["-C", "opt-level=3"]
                    for index in range(len(tokens))
                )
                or "--test" in tokens
                or "test" in cfg
                or actual_features != features
            ):
                raise ValueError("unexpected release configuration")
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise ValueError("invalid traced wheel compiler invocation") from error
