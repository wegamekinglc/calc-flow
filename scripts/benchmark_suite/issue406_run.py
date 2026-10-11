"""Fixed two-round collector for the complete Issue 406 acceptance inventory."""

from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import shutil
import signal
import sys
import time
from contextlib import suppress
from dataclasses import asdict, replace
from pathlib import Path

from scripts.benchmark_suite import issue406_fresh_v3 as v3
from scripts.benchmark_suite.issue406_acceptance import (
    acceptance_inventory,
    validate_complete_results,
    validate_observation,
)
from scripts.benchmark_suite.issue406_fresh import SCHEMA as FRESH_SCHEMA
from scripts.benchmark_suite.issue406_harness import (
    WHEEL_TARGET,
    native_harness_reuse,
    validate_acceptance_harness,
    validate_fresh_acceptance_harness,
    validate_native_profile,
    validate_traced_wheel_invocations,
)
from scripts.benchmark_suite.process import (
    ROOT,
    Worker,
    _complete_owned,
    child_environment,
)
from scripts.benchmark_suite.release_pairs import pair_order
from scripts.benchmark_suite.report import ROUNDS, SAMPLES
from scripts.benchmark_suite.rust_harness import _clean_revision
from scripts.toolkit import fingerprint_json, sha256_file, write_json

RUNTIME_WORKLOADS = ("join_sealed", "join_retained", "asof", "projection")
WORKER_RESPONSE_LIMIT = 32 << 20
FRESH_SCHEMAS = (FRESH_SCHEMA, v3.SCHEMA)


def collection_groups() -> tuple[tuple, ...]:
    """Share native fixture invocations while retaining every target metric."""
    groups = {}
    for case in acceptance_inventory():
        key = f"shared-native-row-{case.row}" if case.row in (4, 5) else case.id
        groups.setdefault(key, []).append(case)
    return tuple(tuple(group) for group in groups.values())


async def collect(runner, output: Path) -> dict:
    """Persist each observation and settle a round before advancing its workload."""
    records = {
        (case.id, side): {
            "case": case.id,
            "side": side,
            "rounds": [[], []],
            "warmups": [],
        }
        for case in acceptance_inventory()
        for side in case.sides
    }
    report = {"measurements": list(records.values()), "state": "running"}
    output.mkdir(parents=True, exist_ok=True)

    try:
        for group in collection_groups():
            for round_index in range(ROUNDS):
                await _collect_round(runner, output, group, round_index, records)
    except BaseException as error:
        report.update(state="stopped", error=f"{type(error).__name__}: {error}")
        raise
    else:
        report["state"] = "collected"
        return report
    finally:
        write_json(output / "partial.json", report)


async def _collect_round(runner, output, group, round_index, records):
    try:
        for side in group[0].sides:
            samples = await _invoke_collection(
                runner, output, group, side, (round_index, None)
            )
            _record_collection(
                records, output, group, (side, round_index, None), samples
            )
        for index in range(SAMPLES):
            for side in pair_order(index):
                if side in group[0].sides:
                    samples = await _invoke_collection(
                        runner, output, group, side, (round_index, index)
                    )
                    _record_collection(
                        records, output, group, (side, round_index, index), samples
                    )
    finally:
        cleanup = asyncio.create_task(runner.finish_round())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await _complete_owned(cleanup)
            raise


def _child_environment(site: Path | None, cpus: list[int]) -> dict[str, str]:
    return {
        **child_environment(site, source=ROOT, polars_threads=1),
        "TOKIO_WORKER_THREADS": str(len(cpus)),
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
    }


def _taskset_executable() -> str:
    executable = shutil.which("taskset")
    if executable is None:
        raise ValueError("acceptance requires taskset before any library import")
    return executable


def _taskset(cpus: list[int], arguments: list[str]) -> list[str]:
    return [_taskset_executable(), "--cpu-list", ",".join(map(str, cpus)), *arguments]


def competing_work(owned: set[int]) -> list[dict]:
    import psutil

    conflicts = []
    for process in psutil.process_iter(("pid", "name", "cmdline")):
        if process.pid in owned | {os.getpid()}:
            continue
        command = process.info["cmdline"] or []
        if process.info["name"] in ("cargo", "rustc", "rust-lld", "maturin") or any(
            argument in ("scripts.benchmark_suite.issue406_run", "--benchmark-only")
            for argument in command
        ):
            conflicts.append({"pid": process.pid, "name": process.info["name"]})
    return conflicts


def validate_plan(plan: dict) -> None:
    """Require the complete frozen inventory and hash every declared input."""
    files = _validate_plan_inventory(plan)
    fresh = plan.get("schema") in FRESH_SCHEMAS
    _validate_release_policy(plan, files)
    if competing_work(set()):
        raise ValueError("another controlled build or measurement is active")
    enabled_builds = {}
    traced_identities = []
    for side in ("baseline", "candidate"):
        build_harness, wheel_harness = _plan_build_harnesses(plan, side)
        wheel_build = _validate_plan_wheel(plan, side, files, wheel_harness)
        environment = _plan_wheel_environment(wheel_build, fresh)
        traced_identities.append((environment, wheel_build.get("validated_artifacts")))
        _validate_plan_binaries(
            plan,
            side,
            files,
            build_harness,
            enabled_builds,
        )
    if traced_identities[0] != traced_identities[1]:
        raise ValueError(
            "row-7 build environment or dependency/profile inventory differs"
        )


def _validate_release_policy(plan, files):
    if plan.get("schema") == v3.SCHEMA:
        v3.validate_releases(plan, files)
    elif plan.get("schema") == FRESH_SCHEMA:
        _validate_fresh_releases(plan, files)
    else:
        reuse = plan["native_harness_reuse"]
        if reuse != native_harness_reuse(reuse["original_build_harness_git_sha"]):
            raise ValueError("native build reuse witness changed")


def _plan_build_harnesses(plan, side):
    if plan.get("schema") == v3.SCHEMA:
        harness = plan.get("build_harness_git_sha", plan["harness_git_sha"])
        return harness, harness
    if plan.get("schema") == FRESH_SCHEMA:
        return tuple(
            plan[key][side]["original_build_harness_git_sha"]
            for key in ("native_harness_reuse", "row7_harness_reuse")
        )
    harness = plan["native_harness_reuse"]["original_build_harness_git_sha"]
    return harness, harness


def _validate_fresh_releases(plan: dict, files: dict) -> Path:
    from scripts.benchmark_suite.issue406_failures import validate_main_failure_identity
    from scripts.benchmark_suite.issue406_reuse import (
        FROZEN,
        HISTORICAL,
        WITNESS_SHA256,
        validate_ordinary_release,
    )

    witness = plan["historical_harness_witness"]
    compatibility = Path(witness["path"])
    if (
        witness["sha256"] != WITNESS_SHA256
        or files.get(str(compatibility)) != WITNESS_SHA256
    ):
        raise ValueError("historical compatibility witness is not sealed")
    _validate_plan_reuse_roles(plan, compatibility, HISTORICAL, FROZEN)
    releases = [*plan["releases"].values(), plan["retained_control"]]
    builds = [validate_ordinary_release(release, files) for release in releases]
    validate_main_failure_identity(plan, files)
    if (
        any(build != builds[0] for build in builds[1:])
        or builds[0]["rustc"] != plan["expected_native_rustc"]
    ):
        raise ValueError(
            "ordinary release dependencies, compiler or build settings differ"
        )
    return compatibility


class AcceptanceRunner:
    """Own fixed-affinity children, their evidence, and the all-inclusive budget."""

    def __init__(self, plan: dict, output: Path, started: float):
        self.plan, self.output, self.started = plan, output, started
        self.workers = {}
        self.processes = {}
        self.groups = {}
        self.creations = set()
        self.cleanup_tasks = set()
        self.prepared_cases = {}
        self.retired = {}
        self.environments = {}
        self.resources = []
        self.stop = asyncio.Event()

    async def register_process(self, creation):
        self.creations.add(creation)
        try:
            try:
                process = await asyncio.shield(creation)
            except asyncio.CancelledError:
                process = await _complete_owned(creation)
                self._own_process(process)
                raise
            self._own_process(process)
            return process
        finally:
            self.creations.discard(creation)

    def _own_process(self, process):
        self.processes[process.pid] = process
        if self.fresh:
            self.groups[process.pid] = process

    @property
    def fresh(self) -> bool:
        return self.plan.get("schema") in FRESH_SCHEMAS

    def cpus(self, case) -> list[int]:
        return self.plan["cpus"][: case.threads]

    def release(self, case, side) -> dict:
        if side == "previous":
            from scripts.benchmark_suite.issue406_fresh import retained_cases

            if not self.fresh or case not in retained_cases():
                raise ValueError(
                    "previous release is restricted to retained OFF controls"
                )
            return self.plan["retained_control"]
        key = "row7_releases" if case.row == 7 else "releases"
        return self.plan[key][side]

    def destination(self, group, side, round_index, index) -> Path:
        name = "warmup" if index is None else f"sample-{index}"
        return self.output / "raw" / group[0].id / f"round-{round_index}" / side / name

    def fixture_root(self, side, mode, round_index, *, join=False) -> Path:
        return (
            self.output
            / "fixtures"
            / ("join" if join else "asof")
            / side
            / mode
            / f"round-{round_index}"
        )

    async def _start_worker(self, case, side, round_index) -> Worker:
        root = self.output / "workers" / case.id / f"round-{round_index}" / side
        root.mkdir(parents=True)
        log = (root / "stderr.log").open("wb")
        release = self.release(case, side)
        creation = asyncio.create_task(
            asyncio.create_subprocess_exec(
                _taskset_executable(),
                "--cpu-list",
                ",".join(map(str, self.cpus(case))),
                sys.executable,
                "-m",
                "scripts.benchmark_suite",
                "worker",
                "--root",
                str(root),
                cwd=ROOT,
                env=_child_environment(Path(release["site"]), self.cpus(case)),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                limit=WORKER_RESPONSE_LIMIT,
                start_new_session=self.fresh,
            )
        )
        try:
            process = await self.register_process(creation)
        except BaseException:
            log.close()
            raise
        worker = Worker(process, log)
        self._record_worker_invocation(root, process, case, release)
        self.workers[side] = worker
        identity = await self._worker_request(worker, operation="hello", scope="core")
        if (
            identity["native_sha256"] != release["native_sha256"]
            or identity["cpu_affinity"] != self.cpus(case)
            or identity["tokio_worker_threads"] != str(case.threads)
        ):
            raise ValueError("worker loaded the wrong release or CPU configuration")
        comparable = {
            key: value for key, value in identity.items() if key != "native_sha256"
        }
        previous = self.environments.setdefault(case.id, comparable)
        if previous != comparable:
            raise ValueError("paired environment changed")
        write_json(root / "environment.json", identity)
        return worker

    def _record_worker_invocation(self, root, process, case, release):
        if self.plan.get("schema") == v3.SCHEMA:
            write_json(
                root / "invocation.json",
                {
                    "pid": process.pid,
                    "argv": [
                        sys.executable,
                        "-m",
                        "scripts.benchmark_suite",
                        "worker",
                        "--root",
                        str(root),
                    ],
                    "cpu_affinity": self.cpus(case),
                    "product_commit": release["product_commit"],
                    "native_sha256": release["native_sha256"],
                    "harness_git_sha": self.plan["harness_git_sha"],
                },
            )

    async def _worker_request(self, worker, **message):
        if self.plan.get("schema") != v3.SCHEMA:
            return await worker.request(**message)
        root = Path(worker.log.name).parent
        with (root / "requests.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(message, allow_nan=False) + "\n")
        worker.process.stdin.write((json.dumps(message) + "\n").encode())
        await worker.process.stdin.drain()
        line = await asyncio.wait_for(worker.process.stdout.readline(), timeout=900)
        with (root / "stdout.log").open("ab") as stream:
            stream.write(line)
        if not line:
            raise RuntimeError(f"benchmark worker exited; see {worker.log.name}")
        response = json.loads(line)
        if not isinstance(response, dict) or "error" in response:
            raise RuntimeError(f"invalid worker result: {response}")
        return response

    def _normalize(self, group, side, raw, observation_id, raw_digest):
        case = group[0]
        product = (
            self.release(case, side)["product_commit"]
            if self.fresh
            else self.plan["releases"][side]["product_commit"]
        )
        self._validate_runtime_product(case, raw, product)
        if case.row in (4, 5):
            mapped = self._component_observations(group, raw)
        elif case.workload in ("asof_operator", "join_small_batch"):
            mapped = self._matched_native_observation(case, raw)
        else:
            mapped = {case.id: raw}
        normalized = {
            key: {
                **value,
                "product_commit": product,
                "observation_id": observation_id,
                "raw_observation_sha256": raw_digest,
                "group_member_ids": [member.id for member in group],
            }
            for key, value in mapped.items()
        }
        _require_v3_samples(self.plan, normalized)
        for member in group:
            validate_observation(
                member,
                "candidate" if side == "previous" else side,
                normalized[member.id],
            )
        return normalized

    def _validate_runtime_product(self, case, raw, product):
        if (
            self.fresh
            and case.workload in RUNTIME_WORKLOADS
            and (
                raw.get("product_commit") != product
                or raw.get("cpu_affinity") != self.cpus(case)
            )
        ):
            raise ValueError("raw runtime product or affinity differs from its release")

    def _component_observations(self, group, raw):
        from scripts.benchmark_suite.native_components import validate_native_component

        raw = validate_native_component(raw)
        if group[0].row == 4:
            mapped = {member.id: raw for member in group}
        else:
            mapped = {
                member.id: {
                    **raw,
                    **raw[
                        "armed" if member.workload.endswith("compacting") else "steady"
                    ],
                }
                for member in group
            }
        mapped = {
            key: {
                **value,
                "oracle_passed": True,
                "cleanup_complete": True,
                "lifecycle": "completed",
            }
            for key, value in mapped.items()
        }
        return mapped

    def _matched_native_observation(self, case, raw):
        from scripts.benchmark_suite.asof_e2e import validate_matched_native
        from scripts.benchmark_suite.issue406_acceptance import matched_native_identity
        from scripts.benchmark_suite.native_components import validate_small_batch

        validated = (
            validate_matched_native if case.row == 7 else validate_small_batch
        )(raw)
        if validated["checkpoint_enabled"] is not (case.mode != "off"):
            raise ValueError("native checkpoint semantics differ from the case")
        mapped = {
            case.id: {
                **validated,
                "cpu_affinity": validated["observed_cpu_affinity"],
                "mode": case.mode,
                "native_nonterminal_epoch_count": raw["nonterminal_epochs"],
                "nonterminal_epochs": [],
                "processing_checkpoint_free": True,
                "processing_settled": True,
                **(
                    {
                        "match_identity": matched_native_identity(
                            validated, case.mode, self.cpus(case)
                        )
                    }
                    if case.row == 7
                    else {}
                ),
            }
        }
        return mapped

    async def prepare(self, group, side, round_index):
        case = group[0]
        if case.workload not in RUNTIME_WORKLOADS:
            if case.row == 8:
                worker = await self._start_worker(case, side, round_index)
                specification = {
                    **asdict(case),
                    "fixture_root": str(
                        self.fixture_root(side, case.mode, round_index, join=True)
                    ),
                }
                result = await self._worker_request(
                    worker, operation="acceptance-fixture", case=specification
                )
                if (
                    sha256_file(Path(result["fixture_path"]))
                    != result["fixture_sha256"]
                ):
                    raise ValueError("exported Join fixture changed")
                if self.fresh:
                    await self.close_worker(side)
                else:
                    await worker.close()
                    self.processes.pop(worker.process.pid)
                self.workers.pop(side)
            return await self.sample(group, side, round_index, None)
        worker = await self._start_worker(case, side, round_index)
        specification = {
            **asdict(case),
            "family": "join-asof-acceptance",
            "side": side,
            "product_commit": self.release(case, side)["product_commit"],
            "fixture_root": str(self.fixture_root(side, case.mode, round_index)),
        }
        self.prepared_cases[(case.id, side, round_index)] = specification
        response = await self._worker_request(
            worker, operation="prepare", case=specification
        )
        if response["case"] != specification:
            raise ValueError("worker prepared a different acceptance workload")
        return self._persist(group, side, round_index, None, response["warmup"])

    def _persist(self, group, side, round_index, index, raw):
        destination = self.destination(group, side, round_index, index)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "observation.json", raw)
        if self.plan.get("schema") == v3.SCHEMA and raw.get("lifecycle") == "failed":
            self._validate_runtime_product(
                group[0], raw, self.release(group[0], side)["product_commit"]
            )
            raise v3.FailedLifecycle(
                "v3 returned failed lifecycle; diagnostic observation preserved"
            )
        normalized = self._normalize(
            group,
            side,
            raw,
            str(destination.relative_to(self.output)),
            sha256_file(destination / "observation.json"),
        )
        case = group[0]
        if case.row == 7 and case.workload == "asof":
            observation = normalized[case.id]
            envelope = destination / "trace-envelope.json"
            write_json(
                envelope,
                {
                    "schema": "calc-flow.issue406-dispatch-replay.v1",
                    "source_observation_id": observation["observation_id"],
                    "source_raw_sha256": observation["raw_observation_sha256"],
                    "match_identity": raw["match_identity"],
                    "dispatch_trace": raw["dispatch_trace"],
                },
            )
            normalized[case.id] = {
                **observation,
                "dispatch_trace_envelope_path": str(envelope),
                "dispatch_trace_envelope_sha256": sha256_file(envelope),
            }
        write_json(destination / "metrics.json", normalized)
        return normalized

    def dispatch_trace_path(self, case, side, round_index, index) -> Path:
        """Bind native replay to this exact mode, round and source observation."""
        source_case = replace(case, workload="asof")
        source = self.destination((source_case,), side, round_index, index)
        observation = json.loads((source / "observation.json").read_text())
        metrics = json.loads((source / "metrics.json").read_text())[source_case.id]
        path = source / "trace-envelope.json"
        envelope = json.loads(path.read_text())
        if (
            str(path) != metrics["dispatch_trace_envelope_path"]
            or sha256_file(path) != metrics["dispatch_trace_envelope_sha256"]
            or envelope
            != {
                "schema": "calc-flow.issue406-dispatch-replay.v1",
                "source_observation_id": str(source.relative_to(self.output)),
                "source_raw_sha256": sha256_file(source / "observation.json"),
                "match_identity": observation["match_identity"],
                "dispatch_trace": observation["dispatch_trace"],
            }
        ):
            raise ValueError(
                "native dispatch trace differs from its source observation"
            )
        return path

    async def sample(self, group, side, round_index, index):
        case = group[0]
        if case.workload in RUNTIME_WORKLOADS:
            raw = await self._worker_request(self.workers[side], operation="sample")
        else:
            raw = await self._native(group, side, round_index, index)
        return self._persist(group, side, round_index, index, raw)

    async def _native(self, group, side, round_index, index):
        case, binary, arguments, trace_path = self._native_invocation(
            group, side, round_index, index
        )
        destination = self.destination(group, side, round_index, index)
        destination.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with (
            (destination / "stderr.log").open("wb") as log,
            (destination / "stdout.log").open("wb") as stdout_file,
        ):
            creation = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    _taskset_executable(),
                    "--cpu-list",
                    ",".join(map(str, self.cpus(case))),
                    binary["path"],
                    *arguments,
                    cwd=ROOT,
                    env=_child_environment(None, self.cpus(case)),
                    stdout=stdout_file,
                    stderr=log,
                    start_new_session=self.fresh,
                )
            )
            process = await self.register_process(creation)
            await process.wait()
        stdout = (destination / "stdout.log").read_bytes()
        write_json(
            destination / "process.json",
            {
                "pid": process.pid,
                "returncode": process.returncode,
                "argv": [binary["path"], *arguments],
                "binary_sha256": binary["sha256"],
                "elapsed_seconds": time.monotonic() - started,
                "cpu_affinity": self.cpus(case),
            },
        )
        if self.fresh:
            await self.retire_process(process.pid)
        else:
            self.processes.pop(process.pid)
        if process.returncode:
            raise RuntimeError(f"native component failed; see {destination}")
        observed = json.loads(stdout)
        self._validate_native_trace(observed, trace_path)
        self._validate_native_environment(case, observed)
        return {**observed, "native_executable_sha256": binary["sha256"]}

    def _native_invocation(self, group, side, round_index, index):
        case = group[0]
        target = (
            "stream_asof_e2e"
            if case.workload == "asof_operator"
            else "stream_join_perf"
        )
        variant = "off" if case.mode == "off" else "enabled"
        binary = self.plan["native_binaries"][side][f"{target}/{variant}"]
        if case.row in (4, 5):
            arguments = [
                "--native-component",
                "join_capture_restore_1m" if case.row == 4 else "join_compaction_1m",
            ]
        else:
            arguments = [
                "--matched-fixture",
                str(
                    self.fixture_root(side, case.mode, round_index, join=case.row == 8)
                    / "fixture.json"
                ),
            ]
        trace_path = None
        if case.workload == "asof_operator":
            trace_path = self.dispatch_trace_path(case, side, round_index, index)
            arguments.extend(("--dispatch-trace", str(trace_path)))
        return case, binary, arguments, trace_path

    def _validate_native_trace(self, observed, trace_path):
        if trace_path is not None:
            source = json.loads(trace_path.read_text())
            if (
                observed.get("dispatch_trace_verified") is not True
                or observed.get("dispatch_replay_state_verified") is not True
                or observed.get("replay_source_observation_id")
                != source["source_observation_id"]
                or observed.get("replay_source_raw_sha256")
                != source["source_raw_sha256"]
                or observed.get("replay_trace_sha256") != sha256_file(trace_path)
            ):
                raise ValueError("native replay did not verify its exact source trace")

    def _validate_native_environment(self, case, observed):
        thread_environment = observed.get("observed_thread_environment", {})
        expected_environment = _child_environment(None, self.cpus(case))
        if observed.get("observed_cpu_affinity") != self.cpus(case) or any(
            thread_environment.get(name) != expected_environment[name]
            for name in (
                "TOKIO_WORKER_THREADS",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "RAYON_NUM_THREADS",
                "POLARS_MAX_THREADS",
            )
        ):
            raise ValueError("native observed CPU/thread configuration differs")

    async def finish_round(self, selected_side=None):
        for side, worker in tuple(self.workers.items()):
            if selected_side is not None and side != selected_side:
                continue
            if self.fresh:
                response = await self._worker_request(worker, operation="finish")
                if (
                    response.get("state") != "completed"
                    or response.get("samples") != SAMPLES + 1
                ):
                    raise ValueError("worker did not finish exactly eleven invocations")
                await self.close_worker(side)
                self.workers.pop(side)
                continue
            await self._finish_legacy_worker(side, worker)

    async def _finish_legacy_worker(self, side, worker):
        try:
            if worker.process.returncode is None:
                response = await self._worker_request(worker, operation="finish")
                if response.get("state") != "completed":
                    raise ValueError("worker failed to finish its round")
        finally:
            await worker.close()
            self.processes.pop(worker.process.pid, None)
            self.workers.pop(side, None)

    def group_members(self) -> dict:
        import psutil

        members = {pid: [] for pid in self.groups}
        for process in psutil.process_iter(("pid", "status")):
            with suppress(ProcessLookupError, psutil.NoSuchProcess):
                group = os.getpgid(process.pid)
                if group in members and process.info["status"] != psutil.STATUS_ZOMBIE:
                    members[group].append(process.pid)
        return members

    async def retire_process(self, pid: int) -> None:
        process = self.processes[pid]
        if process.returncode is None or self.group_members().get(pid):
            raise RuntimeError("exited process has live owned descendants")
        self.retired[pid] = {"returncode": process.returncode, "group_empty": True}
        self.groups.pop(pid, None)
        self.processes.pop(pid)

    async def close_worker(self, side: str, *, allow_failure=False) -> None:
        worker = self.workers[side]
        worker.process.stdin.close()
        await worker.process.wait()
        if (
            worker.process.returncode is None
            or worker.process.returncode < 0
            or (worker.process.returncode and not allow_failure)
        ):
            raise RuntimeError("worker did not exit naturally")
        await self.retire_process(worker.process.pid)
        worker.log.close()

    async def known_main_failure(self, group, side, round_index, index, error):
        from scripts.benchmark_suite.issue406_failures import recognize_owned_failure

        return await recognize_owned_failure(
            self, group, (side, round_index, index), error
        )

    async def known_fresh_failure(self, group, side, round_index, index, error):
        return await v3.recognize_owned_failure(
            self, group, (side, round_index, index), error
        )

    async def resources_guard(self):
        import psutil

        while not self.stop.is_set():
            controller_rss = psutil.Process().memory_info().rss
            rss = controller_rss
            owned = self._owned_resource_pids()
            rss, children = self._process_resources(owned, rss)
            memory = psutil.virtual_memory().available
            disk = shutil.disk_usage(self.output).free
            used = self._output_bytes()
            conflicts = competing_work(owned)
            self.resources.append(
                {
                    "elapsed_seconds": time.monotonic() - self.started,
                    "rss": rss,
                    "controller_rss": controller_rss,
                    "available_memory": memory,
                    "free_disk": disk,
                    "output_bytes": used,
                    "processes": children,
                    "competing_work": conflicts,
                }
            )
            if rss > 24 << 30 or memory < 4 << 30 or used > 20 << 30 or disk < 10 << 30:
                raise RuntimeError("acceptance resource limit reached")
            if conflicts:
                raise RuntimeError("another controlled build or measurement started")
            with suppress(TimeoutError):
                await asyncio.wait_for(self.stop.wait(), 1)

    def _owned_resource_pids(self):
        return (
            {pid for members in self.group_members().values() for pid in members}
            if self.fresh
            else {
                pid
                for pid, process in self.processes.items()
                if process.returncode is None
            }
        )

    def _process_resources(self, owned, rss):
        import psutil

        children = []
        for pid in owned:
            try:
                observed = psutil.Process(pid)
                for item in (
                    (observed,)
                    if self.fresh
                    else (observed, *observed.children(recursive=True))
                ):
                    memory = item.memory_info().rss
                    rss += memory
                    children.append(
                        {
                            "pid": item.pid,
                            "rss": memory,
                            "cpu_seconds": sum(item.cpu_times()[:2]),
                            "threads": [list(thread) for thread in item.threads()],
                            "cpu_affinity": item.cpu_affinity(),
                        }
                    )
            except psutil.NoSuchProcess:
                pass
        return rss, children

    def _output_bytes(self):
        used = 0
        for path in self.output.rglob("*"):
            with suppress(FileNotFoundError):
                if path.is_file():
                    used += path.stat().st_size
        return used

    async def settle(self):
        if self.fresh:
            return await self.settle_fresh()
        for process in self.processes.values():
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.send_signal(signal.SIGTERM)
        forced = []
        for pid, process in tuple(self.processes.items()):
            try:
                await asyncio.wait_for(process.wait(), 60)
            except TimeoutError:
                forced.append(pid)
                process.kill()
                await process.wait()
        for worker in self.workers.values():
            worker.log.close()
        self.workers.clear()
        self.processes.clear()
        return {"settled": True, "forced_kill_pids": forced}

    async def settle_fresh(self):
        deadline = min(
            self.started + self.plan["settlement_deadline_seconds"],
            time.monotonic() + self.plan["settlement_seconds"],
        )
        self.settlement_deadline = deadline
        reserve = min(5, max(0, deadline - time.monotonic()) / 2)
        process_deadline = deadline - reserve
        cleanup = tuple(self.cleanup_tasks)
        for task in cleanup:
            task.cancel()
        signaled, forced, processes = await self._settle_groups(process_deadline)
        creations = await self._cancel_pending_creations(cleanup, deadline)
        final_signaled, final_forced, final_processes = await self._settle_groups(
            deadline, force=True
        )
        signaled = _merged_settlement_pids(signaled, final_signaled)
        forced = _merged_settlement_pids(forced, final_forced)
        processes.update(final_processes)
        errors = await self._cleanup_task_errors(cleanup, deadline)
        io_settled = await self._settle_io(processes, deadline)
        for worker in self.workers.values():
            worker.log.close()
        self.workers.clear()
        members = self.group_members()
        return {
            "settled": self._fresh_settlement_complete(deadline, io_settled, cleanup),
            "forced_kill_pids": forced,
            "terminated_groups": signaled,
            "surviving_group_pids": [
                pid for group in members.values() for pid in group
            ],
            "unsettled_groups": sorted(self.groups),
            "pending_creations": len(self.creations),
            "cancelled_creations": sum(task.cancelled() for task in creations),
            "cleanup_errors": errors,
            "deadline_satisfied": time.monotonic() <= deadline,
        }

    async def _cancel_pending_creations(self, cleanup, deadline):
        creations = tuple(self.creations)
        for task in (*creations, *cleanup):
            task.cancel()
        if creations:
            await asyncio.wait(
                creations, timeout=max(0, deadline - time.monotonic()) / 2
            )
        return creations

    async def _cleanup_task_errors(self, cleanup, deadline):
        if cleanup:
            await asyncio.wait(cleanup, timeout=max(0, deadline - time.monotonic()))
        errors = []
        for task in cleanup:
            if task.done() and not task.cancelled() and task.exception() is not None:
                errors.append(str(task.exception()))
        return errors

    def _fresh_settlement_complete(self, deadline, io_settled, cleanup):
        return (
            time.monotonic() <= deadline
            and not self.groups
            and not self.creations
            and io_settled
            and all(task.done() for task in cleanup)
        )

    async def _settle_groups(self, deadline, *, force=False):
        grace = time.monotonic() if force else min(deadline, time.monotonic() + 10)
        signaled, forced, processes = [], [], {}

        async def settle_group(pid, process, members):
            if not members[pid] and process.returncode is not None:
                await self.retire_process(pid)
                return
            if pid not in signaled:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGTERM)
                signaled.append(pid)
            if time.monotonic() >= grace and pid not in forced:
                with suppress(ProcessLookupError):
                    os.killpg(pid, signal.SIGKILL)
                forced.append(pid)

        while (self.groups or self.creations) and time.monotonic() < deadline:
            processes.update(self.processes)
            members = self.group_members()
            for pid, process in tuple(self.groups.items()):
                await settle_group(pid, process, members)
            if self.groups or self.creations:
                await asyncio.sleep(min(0.02, max(0, deadline - time.monotonic())))
        return signaled, forced, processes

    async def _settle_io(self, processes, deadline):
        tasks = [
            asyncio.create_task(process.communicate()) for process in processes.values()
        ]
        if not tasks:
            return True
        done, pending = await asyncio.wait(
            tasks, timeout=max(0, deadline - time.monotonic())
        )
        for task in done:
            with suppress(BaseException):
                task.result()
        for task in pending:
            task.cancel()
        if pending:
            for process in processes.values():
                process._transport.close()
            await asyncio.wait(pending, timeout=max(0, deadline - time.monotonic()))
        return not pending


def _merged_settlement_pids(initial: list[int], final: list[int]) -> list[int]:
    merged = list(initial)
    merged.extend(pid for pid in final if pid not in merged)
    return merged


def _prepare_collection_output(output: Path) -> None:
    if output.exists() and any(output.iterdir()):
        raise ValueError("acceptance execution requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)


async def _await_collection_result(report, collection, guard, remaining):
    async with asyncio.timeout(remaining):
        done, _ = await asyncio.wait(
            (collection, guard), return_when=asyncio.FIRST_COMPLETED
        )
        if guard in done:
            guard.result()
            raise RuntimeError("resource guard exited unexpectedly")
        return {**report, **collection.result()}


def _initial_report(plan):
    return {
        "state": "preflight",
        "plan_fingerprint": fingerprint_json(plan),
        "assessment": {
            "compatibility_verified": False,
            "environment_suitable": False,
            "machine_type": plan.get("preflight", {})
            .get("host", {})
            .get("machine_type", "unverified"),
        },
        "scheduling_proofs": plan.get("scheduling_proofs", {}),
        **(v3.report_identity(plan) if plan.get("schema") == v3.SCHEMA else {}),
    }


def _require_v3_samples(plan, samples):
    if plan.get("schema") == v3.SCHEMA:
        v3.require_completed(samples)


def _validate_collection(plan, report):
    if plan.get("schema") == v3.SCHEMA:
        v3.validate_complete_results(report)
    else:
        validate_complete_results(report)


def _failed_run_state(report):
    return "incomplete" if report.get("state") == "incomplete" else "stopped"


async def run(plan: dict, output: Path) -> dict:
    """Execute once, charging setup, preflight, collection, cleanup and sealing."""
    started = (
        time.monotonic()
        - plan.get("precharged_seconds", 0)
        - plan.get("startup_allowance_seconds", 0)
    )
    _prepare_collection_output(output)
    runner = AcceptanceRunner(plan, output, started)
    report = _initial_report(plan)
    collection = guard = None
    parent = asyncio.current_task()
    loop = asyncio.get_running_loop()
    previous_signals = {
        sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)
    }
    for sig in previous_signals:
        loop.add_signal_handler(sig, parent.cancel)
    try:
        validate_plan(plan)
        report["assessment"]["compatibility_verified"] = True
        write_json(output / "execution-plan.json", plan)
        write_json(
            output / "host.json",
            {
                "platform": platform.platform(),
                "uname": list(platform.uname()),
                "load_average": list(os.getloadavg()),
                "controller_affinity": sorted(os.sched_getaffinity(0)),
                "child_affinity": plan["cpus"],
                "resource_interval_seconds": 1,
                "worker_response_limit_bytes": WORKER_RESPONSE_LIMIT,
            },
        )
        remaining = plan.get(
            "collection_cutoff_seconds",
            plan["budget_seconds"] - plan["settlement_seconds"],
        ) - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("collection budget exhausted during preflight")
        guard = asyncio.create_task(runner.resources_guard())
        from scripts.benchmark_suite.issue406_fresh import collect_fresh

        collection = asyncio.create_task(
            (collect_fresh if runner.fresh else collect)(runner, output)
        )
        report = await _await_collection_result(report, collection, guard, remaining)
        report["assessment"]["environment_suitable"] = True
        _validate_collection(plan, report)
        report["state"] = "complete"
        report["assessment"]["environment_suitable"] = True
    except BaseException as error:
        state = _failed_run_state(report)
        report.update(state=state, error=f"{type(error).__name__}: {error}")
    finally:
        await _settle_acceptance_collection(runner, collection, guard, report, output)
        _seal_acceptance_report(plan, output, started, report, runner.resources)
        for sig, handler in previous_signals.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)
    return report


async def _run_cli(plan: dict, output: Path) -> dict:
    """Prevent the dedicated controller's final asyncio sweep from hanging."""
    started = (
        time.monotonic()
        - plan.get("precharged_seconds", 0)
        - plan.get("startup_allowance_seconds", 0)
    )
    report = await run(plan, output)
    pending = {
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    }
    if plan.get("schema") in FRESH_SCHEMAS and pending:
        _exit_unsettled_controller(plan, output, started, pending)
    return report


def _exit_unsettled_controller(
    plan: dict, output: Path, started: float, pending
) -> None:
    """Stop only this CLI's descendants before an explicit, nonzero failure exit."""
    import psutil

    deadline = min(
        time.monotonic() + 1,
        started + plan["budget_seconds"] - plan["final_seal_allowance_seconds"],
    )
    processes = psutil.Process().children(recursive=True)
    groups = set()
    for process in processes:
        with suppress(ProcessLookupError):
            group = os.getpgid(process.pid)
            if group != os.getpgrp():
                groups.add(group)
    for group in groups:
        with suppress(ProcessLookupError):
            os.killpg(group, signal.SIGKILL)
    for process in processes:
        with suppress(psutil.NoSuchProcess):
            process.kill()
    _, alive = psutil.wait_procs(processes, timeout=max(0, deadline - time.monotonic()))
    record = {
        "schema": "calc-flow.issue406-ownership-failure.v1",
        "settled": False,
        "reason": "owned tasks survived the absolute settlement deadline",
        "pending_tasks": len(pending),
        "terminated_pids": [process.pid for process in processes],
        "terminated_groups": sorted(groups),
        "surviving_pids": [process.pid for process in alive],
        "deadline_satisfied": time.monotonic() <= deadline,
        "elapsed_seconds": time.monotonic() - started,
        "exit_code": 1,
    }
    path = output / "ownership-failure.json"
    write_json(path, record)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())
    os._exit(1)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("execution plan differs from the reviewed seal")
    report = asyncio.run(_run_cli(json.loads(args.plan.read_text()), args.output))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "measurements"}
        )
    )
    return 0 if report["state"] == "complete" and report["budget_satisfied"] else 1


def _validate_plan_inventory(plan: dict) -> dict:
    _validate_plan_envelope(plan)
    _validate_plan_precharge(plan)
    _validate_plan_cpus(plan)
    files = plan.get("sealed_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("missing sealed source and artifact inventory")
    if plan.get("harness_git_sha") != _clean_revision(ROOT):
        raise ValueError("execution harness revision changed")
    required = _plan_python_sources()
    if not required.issubset(files):
        raise ValueError("incomplete sealed Python harness inventory")
    for path, digest in files.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"sealed execution input changed: {path}")
    return files


def _validate_plan_wheel(
    plan: dict, side: str, files: dict, build_harness: str
) -> dict:
    release = plan["releases"][side]
    if files.get(release["native_path"]) != release["native_sha256"]:
        raise ValueError("release native is missing from the sealed inventory")
    _validate_plan_installed_release(release, files)
    traced = plan["row7_releases"][side]
    _validate_plan_traced_identity(traced, release, files)
    _validate_plan_traced_installation(traced, files)
    traced_output = Path(traced["build_output"])
    reconstruction = traced.get("helper_reconstruction")
    _validate_plan_reconstruction(plan, side, reconstruction)
    traced_attestation = _validate_source_attestation(
        plan, traced_output, WHEEL_TARGET, reconstruction=reconstruction
    )
    if (
        str(traced_output / f"build-{WHEEL_TARGET}.acceptance-harness.json")
        not in files
    ):
        raise ValueError("row-7 source attestation is not sealed")
    wheel_build = json.loads(Path(traced["build_metadata"]).read_text())
    _validate_plan_compiled_wheel(
        wheel_build,
        build_harness,
        traced_attestation,
        plan,
        traced,
    )
    validate_traced_wheel_invocations(wheel_build)
    if plan.get("schema") in FRESH_SCHEMAS:
        from scripts.benchmark_suite.issue406_reuse import (
            validate_traced_audit,
            validate_wheel_members,
        )

        validate_wheel_members(traced, files)
        return {
            **wheel_build,
            "validated_artifacts": validate_traced_audit(
                traced, traced_attestation, files
            ),
        }
    return wheel_build


def _validate_plan_binaries(
    plan: dict,
    side: str,
    files: dict,
    build_harness: str,
    enabled_builds: dict,
) -> None:
    expected_binaries = _plan_binary_inventory(side)
    if set(plan["native_binaries"][side]) != expected_binaries:
        raise ValueError("native binary inventory omits or adds an acceptance arm")
    for name in plan["native_binaries"][side]:
        target, mode, build = _validate_plan_binary(
            plan, side, name, files, build_harness
        )
        if mode == "enabled":
            previous = enabled_builds.setdefault(target, build["artifacts"])
            if previous != build["artifacts"]:
                raise ValueError("paired compiled dependency/profile inventory differs")


async def _invoke_collection(
    runner, output: Path, group, side: str, position: tuple
) -> dict:
    round_index, index = position

    def journal(value):
        with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    started = time.monotonic()
    attempt = {
        "cases": [case.id for case in group],
        "side": side,
        "round": round_index,
        "index": index,
        "phase": "warmup" if index is None else "sample",
    }
    journal({**attempt, "state": "started"})
    try:
        if index is None:
            result = await runner.prepare(group, side, round_index)
        else:
            result = await runner.sample(group, side, round_index, index)
    except BaseException as error:
        journal(
            {
                **attempt,
                "state": "failed",
                "elapsed_seconds": time.monotonic() - started,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        raise
    journal(
        {
            **attempt,
            "state": "completed",
            "elapsed_seconds": time.monotonic() - started,
        }
    )
    return result


def _seal_acceptance_report(
    plan: dict, output: Path, started: float, report: dict, resources: list
) -> None:
    if plan.get("schema") in FRESH_SCHEMAS:
        _seal_fresh_report(plan, output, started, report, resources)
        return
    from scripts.benchmark_suite.issue406_report import (
        render_acceptance_markdown,
        summarize_acceptance,
    )

    report["elapsed_seconds"] = time.monotonic() - started
    report["budget_seconds"] = plan["budget_seconds"]
    report["precharged_seconds"] = plan.get("precharged_seconds", 0)
    report["budget_satisfied"] = report["elapsed_seconds"] <= plan["budget_seconds"]
    summary = summarize_acceptance(report)
    write_json(output / "summary.json", summary)
    (output / "performance.md").write_text(
        render_acceptance_markdown(summary), encoding="utf-8"
    )
    write_json(output / "resources.json", resources)
    report["elapsed_seconds"] = time.monotonic() - started
    report["budget_satisfied"] = report["elapsed_seconds"] <= plan["budget_seconds"]
    write_json(output / "result.json", report)
    write_json(
        output / "evidence-sha256.json",
        {
            str(path.relative_to(output)): sha256_file(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    )
    elapsed = time.monotonic() - started
    report["elapsed_seconds_including_sealing"] = elapsed
    report["budget_satisfied"] = elapsed <= plan["budget_seconds"]
    write_json(
        output / "budget.json",
        {
            "elapsed_seconds_including_sealing": elapsed,
            "budget_seconds": plan["budget_seconds"],
            "budget_satisfied": report["budget_satisfied"],
            "evidence_seal_sha256": sha256_file(output / "evidence-sha256.json"),
        },
    )


def _hash_fresh_evidence(output, deadline, complete):
    hashes = {}
    for path in sorted(output.rglob("*")):
        if time.monotonic() >= deadline:
            complete = False
            break
        if path.is_file():
            hashes[str(path.relative_to(output))] = sha256_file(path)
    return hashes, complete


def _seal_fresh_report(plan, output, started, report, resources):
    allowance = plan["final_seal_allowance_seconds"]
    deadline = started + plan["budget_seconds"] - allowance
    report.update(
        elapsed_seconds=time.monotonic() - started,
        budget_seconds=plan["budget_seconds"],
        precharged_seconds=plan.get("precharged_seconds", 0),
        startup_allowance_seconds=plan["startup_allowance_seconds"],
        final_seal_allowance_seconds=allowance,
        budget_satisfied=time.monotonic() < deadline,
    )
    write_json(output / "resources.json", resources)
    complete = _write_fresh_summary(output, report, deadline)
    write_json(output / "result.json", report)
    hashes, complete = _hash_fresh_evidence(output, deadline, complete)
    if not complete or time.monotonic() >= deadline:
        complete = False
        _expire_fresh_report(output, report)
        write_json(output / "result.json", report)
        for name in ("result.json", "summary.json", "performance.md"):
            hashes[name] = sha256_file(output / name)
    write_json(output / "evidence-sha256.json", hashes)
    elapsed = time.monotonic() - started
    if complete and elapsed + allowance > plan["budget_seconds"]:
        complete = False
        _expire_fresh_report(output, report)
        write_json(output / "result.json", report)
        for name in ("result.json", "summary.json", "performance.md"):
            hashes[name] = sha256_file(output / name)
        write_json(output / "evidence-sha256.json", hashes)
        elapsed = time.monotonic() - started
    report["elapsed_seconds_including_sealing"] = elapsed
    report["budget_satisfied"] = (
        complete and elapsed + allowance <= plan["budget_seconds"]
    )
    write_json(
        output / "budget.json",
        {
            "elapsed_seconds_including_sealing": elapsed,
            "final_seal_allowance_seconds": allowance,
            "accounted_seconds": elapsed + allowance,
            "budget_seconds": plan["budget_seconds"],
            "budget_satisfied": report["budget_satisfied"],
            "evidence_complete": complete,
            "evidence_seal_sha256": sha256_file(output / "evidence-sha256.json"),
        },
    )


def _write_fresh_summary(output, report, deadline):
    from scripts.benchmark_suite.issue406_report import (
        render_acceptance_markdown,
        summarize_acceptance,
    )

    if report.get("execution_schema") == v3.SCHEMA:
        from scripts.benchmark_suite.issue406_fresh_report import (
            render_v3,
            summarize_v3,
        )

        summarize_acceptance, render_acceptance_markdown = summarize_v3, render_v3
    if time.monotonic() >= deadline:
        _expire_fresh_report(output, report)
        return False
    summary = summarize_acceptance(report)
    if time.monotonic() >= deadline:
        _expire_fresh_report(output, report)
        return False
    write_json(output / "summary.json", summary)
    (output / "performance.md").write_text(
        render_acceptance_markdown(summary), encoding="utf-8"
    )
    return True


def _expire_fresh_report(output, report):
    report.update(
        state="stopped",
        error="analysis or sealing budget exhausted",
        budget_satisfied=False,
    )
    report["assessment"]["environment_suitable"] = False
    if report.get("execution_schema") == v3.SCHEMA:
        _write_aborted_v3(output, report)
        return
    write_json(
        output / "summary.json",
        {
            "schema": "calc-flow.issue406-summary-aborted.v1",
            "evidence_complete": False,
            "verdict": "inconclusive",
            "reason": report["error"],
        },
    )
    (output / "performance.md").write_text(
        "Analysis or sealing budget exhausted; no acceptance verdict. "
        "Raw observations are retained.\n",
        encoding="utf-8",
    )


def _write_aborted_v3(output, report):
    from scripts.benchmark_suite.issue406_fresh_report import aborted_v3_summary
    from scripts.benchmark_suite.report import table

    summary = aborted_v3_summary(report)
    write_json(output / "summary.json", summary)
    rows = [[item["row"], item["title"], item["status"]] for item in summary["targets"]]
    (output / "performance.md").write_text(
        "Acquisition incomplete; acceptance inconclusive. Analysis or sealing budget "
        "exhausted; raw observations are retained.\n\n"
        + table(["Row", "Workload", "Evidence"], rows)
        + "\n",
        encoding="utf-8",
    )


def _record_collection(
    records: dict, output: Path, group, position: tuple, samples: dict
) -> None:
    side, round_index, index = position
    if set(samples) != {case.id for case in group}:
        raise ValueError("shared invocation omitted or added a target metric")
    for case in group:
        target = records[case.id, side]
        if index is None:
            target["warmups"].append(samples[case.id])
        else:
            target["rounds"][round_index].append(samples[case.id])
    write_json(
        output / "progress.json",
        {
            "case": group[0].id,
            "side": side,
            "round": round_index,
            "index": index,
            "state": "running",
        },
    )


def _failed_controller_task(guard) -> bool:
    return (
        guard is not None
        and guard.done()
        and not guard.cancelled()
        and guard.exception() is not None
    )


def _partial_collection_fields(output: Path) -> dict:
    partial = output / "partial.json"
    if not partial.exists():
        return {}
    return {
        key: value
        for key, value in json.loads(partial.read_text()).items()
        if key not in ("state", "error")
    }


def _settlement_incomplete(settlement: dict) -> bool:
    return bool(
        settlement["forced_kill_pids"]
        or not settlement["settled"]
        or settlement.get("terminated_groups")
        or settlement.get("cleanup_errors")
    )


async def _settle_acceptance_collection(
    runner, collection, guard, report: dict, output: Path
) -> None:
    if not runner.fresh:
        runner.stop.set()
    if collection is not None and not collection.done():
        collection.cancel()
    # Signal cancellation before awaiting the collector's owned round cleanup.
    report["settlement"] = await _complete_owned(asyncio.create_task(runner.settle()))
    runner.stop.set()
    pending = await _settle_controller_tasks(runner, (collection, guard))
    if runner.fresh:
        report["settlement"]["pending_controller_tasks"] = pending
        report["settlement"]["settled"] &= not pending
    if _failed_controller_task(guard):
        report["state"] = "stopped"
        report["assessment"]["environment_suitable"] = False
        report["error"] = f"resource guard: {guard.exception()}"
    for key, value in _partial_collection_fields(output).items():
        report.setdefault(key, value)
    if _settlement_incomplete(report["settlement"]):
        report["assessment"]["environment_suitable"] = False
        report["state"] = "stopped"


async def _settle_controller_tasks(runner, owned) -> int:
    tasks = [task for task in owned if task is not None]
    if runner.fresh:
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            _, pending = await asyncio.wait(
                tasks, timeout=max(0, runner.settlement_deadline - time.monotonic())
            )
            return len(pending)
    else:
        for task in tasks:
            with suppress(BaseException):
                await _complete_owned(task)
    return 0


def _validate_plan_native_build(
    plan: dict, descriptor: dict, files: dict, mode: str
) -> dict:
    build_path = Path(descriptor["compiled_artifacts"])
    if str(build_path) not in files:
        raise ValueError("compiled native settings are not sealed")
    build = json.loads(build_path.read_text())
    if build["flags"] != plan["expected_native_flags"]:
        raise ValueError("native build flags differ from the approved flags")
    if build["rustc"] != plan["expected_native_rustc"]:
        raise ValueError("native compiler differs from the approved compiler")
    if "cfg" in " ".join(str(value) for value in build["flags"].values()):
        raise ValueError("external Rust cfg flags are not approved")
    validate_native_profile(build["artifacts"], checkpoint_off=mode == "off")
    return build


def _plan_wheel_environment(wheel_build, fresh):
    environment = {
        key: value
        for key, value in wheel_build["environment"].items()
        if not fresh or key != "CARGO_TARGET_DIR"
    }
    return environment


def _validate_plan_reuse_roles(
    plan, compatibility, historical_revision, frozen_revision
):
    for key, baseline_revision in (
        ("native_harness_reuse", historical_revision),
        ("row7_harness_reuse", frozen_revision),
    ):
        if set(plan[key]) != {"baseline", "candidate"}:
            raise ValueError("fresh build-harness roles differ")
        for side, reuse in plan[key].items():
            historical = key == "native_harness_reuse" and side == "baseline"
            revision = (
                baseline_revision if side == "baseline" else plan["harness_git_sha"]
            )
            expected = native_harness_reuse(
                revision, compatibility=compatibility if historical else None
            )
            if reuse != expected:
                raise ValueError("fresh native build reuse witness changed")


def _validate_plan_precharge(plan):
    precharged = plan.get("precharged_seconds", 0)
    if plan.get("schema") not in FRESH_SCHEMAS and (
        type(precharged) not in (int, float)
        or not math.isfinite(precharged)
        or not 0 <= precharged < 4200
    ):
        raise ValueError("invalid precharged final correctness budget")


def _validate_plan_cpus(plan):
    cpus = plan.get("cpus")
    if (
        not isinstance(cpus, list)
        or len(cpus) != 32
        or len(set(cpus)) != 32
        or any(type(cpu) is not int for cpu in cpus)
        or not set(cpus).issubset(os.sched_getaffinity(0))
    ):
        raise ValueError("execution requires one available fixed 32-CPU set")


def _validate_plan_envelope(plan):
    if plan.get("schema") == v3.SCHEMA:
        v3.validate_envelope(plan)
    elif plan.get("schema") == FRESH_SCHEMA:
        from scripts.benchmark_suite.issue406_fresh import validate_envelope

        validate_envelope(plan)
    elif (
        plan.get("schema") != "calc-flow.issue406-execution.v1"
        or plan.get("budget_seconds") != 4500
        or plan.get("settlement_seconds") != 300
        or plan.get("cases") != [asdict(case) for case in acceptance_inventory()]
    ):
        raise ValueError("execution plan differs from the frozen acceptance contract")


def _validate_plan_installed_release(release, files):
    installed = {
        str(path.resolve())
        for path in (Path(release["site"]) / "calc_flow").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not installed or not installed.issubset(files):
        raise ValueError("installed Python package files are not fully sealed")


def _validate_plan_traced_identity(traced, release, files):
    if (
        traced["product_commit"] != release["product_commit"]
        or traced["instrumentation"] != "asof-dispatch-trace"
        or traced["harness_target"] != WHEEL_TARGET
        or files.get(traced["native_path"]) != traced["native_sha256"]
        or files.get(traced["wheel_path"]) != traced["wheel_sha256"]
        or traced["build_metadata"] not in files
    ):
        raise ValueError("row-7 instrumented release identity differs")


def _validate_plan_traced_installation(traced, files):
    traced_installed = {
        str(path.resolve())
        for path in (Path(traced["site"]) / "calc_flow").rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    if not traced_installed or not traced_installed.issubset(files):
        raise ValueError("row-7 installed package is not fully sealed")


def _validate_plan_reconstruction(plan, side, reconstruction):
    if plan.get("schema") == v3.SCHEMA and reconstruction is not None:
        raise ValueError("v3 forbids historical helper reconstruction")
    if plan.get("schema") == FRESH_SCHEMA:
        from scripts.benchmark_suite.issue406_reuse import helper_reconstruction

        expected = (
            helper_reconstruction(Path(plan["historical_harness_witness"]["path"]))
            if side == "baseline"
            else None
        )
        if reconstruction != expected:
            raise ValueError("fresh traced helper reconstruction differs")


def _validate_plan_compiled_wheel(
    wheel_build, build_harness, traced_attestation, plan, traced
):
    reconstruction = traced.get("helper_reconstruction")
    if (
        wheel_build["failure"] is not None
        or wheel_build["harness_git_sha"] != build_harness
        or traced_attestation["harness_git_sha"] != build_harness
        or wheel_build["revision"] != traced["product_commit"]
        or traced_attestation["product_git_sha"] != traced["product_commit"]
        or not _plan_compiled_wheel_settings(
            wheel_build, plan, traced_attestation, traced, reconstruction
        )
    ):
        raise ValueError("row-7 compiled wheel settings differ")


def _plan_compiled_wheel_settings(
    wheel_build, plan, traced_attestation, traced, reconstruction
):
    return (
        wheel_build["flags"] == plan["expected_native_flags"]
        and wheel_build["rustc"] == plan["expected_native_rustc"]
        and wheel_build["lock_fingerprint"] == traced_attestation["lock_fingerprint"]
        and wheel_build["native_sha256"] == traced["native_sha256"]
        and wheel_build["wheel_sha256"] == traced["wheel_sha256"]
        and wheel_build.get("helper_reconstruction") == reconstruction
    )


def _validate_plan_binary(plan, side, name, files, build_harness):
    descriptor = plan["native_binaries"][side][name]
    release = plan["releases"][side]
    if files.get(descriptor["path"]) != descriptor["sha256"]:
        raise ValueError("native benchmark is missing from the sealed inventory")
    target, mode = name.split("/")
    if side == "baseline" and mode == "off":
        raise ValueError("main does not support checkpoint-off")
    record = _validate_source_attestation(
        plan,
        Path(descriptor["build_output"]),
        descriptor["harness_target"],
        compatibility=_plan_native_compatibility(plan, side),
    )
    attestation = str(
        Path(descriptor["build_output"])
        / f"build-{descriptor['harness_target']}.acceptance-harness.json"
    )
    if attestation not in files:
        raise ValueError("native source attestation is not sealed")
    if (
        target not in record["targets"]
        or record["harness_git_sha"] != build_harness
        or record["product_git_sha"] != release["product_commit"]
    ):
        raise ValueError("native and wheel product revisions differ")
    build = _validate_plan_native_build(plan, descriptor, files, mode)
    return target, mode, build


def _plan_native_compatibility(plan, side):
    return (
        Path(plan["historical_harness_witness"]["path"])
        if plan.get("schema") == FRESH_SCHEMA and side == "baseline"
        else None
    )


def _validate_source_attestation(
    plan, output, target, *, compatibility=None, reconstruction=None
):
    if plan.get("schema") == v3.SCHEMA:
        if compatibility is not None or reconstruction is not None:
            raise ValueError("v3 forbids historical source exceptions")
        source = plan.get("build_harness_source")
        return validate_fresh_acceptance_harness(
            output,
            target,
            harness_git_sha=plan.get("build_harness_git_sha", plan["harness_git_sha"]),
            **({"harness_source": Path(source)} if source is not None else {}),
        )
    return validate_acceptance_harness(
        output, target, compatibility=compatibility, reconstruction=reconstruction
    )


def _plan_python_sources():
    return {
        str(path.resolve())
        for directory in (ROOT / "benchmarks", ROOT / "scripts/benchmark_suite")
        for path in directory.rglob("*.py")
    } | {str(ROOT / "scripts/toolkit.py")}


def _plan_binary_inventory(side):
    return {
        f"{target}/{mode}"
        for target in ("stream_join_perf", "stream_asof_e2e")
        for mode in (("enabled",) if side == "baseline" else ("enabled", "off"))
    }


if __name__ == "__main__":
    raise SystemExit(main())
