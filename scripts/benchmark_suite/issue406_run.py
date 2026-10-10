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

from scripts.benchmark_suite.issue406_acceptance import (
    acceptance_inventory,
    validate_complete_results,
    validate_observation,
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

    def journal(value):
        with (output / "attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value, allow_nan=False) + "\n")

    async def invoke(group, side, round_index, index):
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

    def record(group, side, round_index, index, samples):
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

    try:
        for group in collection_groups():
            for round_index in range(ROUNDS):
                try:
                    for side in group[0].sides:
                        samples = await invoke(group, side, round_index, None)
                        record(group, side, round_index, None, samples)
                    for index in range(SAMPLES):
                        for side in pair_order(index):
                            if side in group[0].sides:
                                samples = await invoke(group, side, round_index, index)
                                record(group, side, round_index, index, samples)
                finally:
                    cleanup = asyncio.create_task(runner.finish_round())
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        await _complete_owned(cleanup)
                        raise
    except BaseException as error:
        report.update(state="stopped", error=f"{type(error).__name__}: {error}")
        raise
    else:
        report["state"] = "collected"
        return report
    finally:
        write_json(output / "partial.json", report)


def _child_environment(site: Path | None, cpus: list[int]) -> dict[str, str]:
    return {
        **child_environment(site, source=ROOT, polars_threads=1),
        "TOKIO_WORKER_THREADS": str(len(cpus)),
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
    }


def _taskset(cpus: list[int], arguments: list[str]) -> list[str]:
    executable = shutil.which("taskset")
    if executable is None:
        raise ValueError("acceptance requires taskset before any library import")
    return [executable, "--cpu-list", ",".join(map(str, cpus)), *arguments]


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
    if (
        plan.get("schema") != "calc-flow.issue406-execution.v1"
        or plan.get("budget_seconds") != 4500
        or plan.get("settlement_seconds") != 300
        or plan.get("cases") != [asdict(case) for case in acceptance_inventory()]
    ):
        raise ValueError("execution plan differs from the frozen acceptance contract")
    precharged = plan.get("precharged_seconds", 0)
    if (
        type(precharged) not in (int, float)
        or not math.isfinite(precharged)
        or not 0 <= precharged < 4200
    ):
        raise ValueError("invalid precharged final correctness budget")
    cpus = plan.get("cpus")
    if (
        not isinstance(cpus, list)
        or len(cpus) != 32
        or len(set(cpus)) != 32
        or any(type(cpu) is not int for cpu in cpus)
        or not set(cpus).issubset(os.sched_getaffinity(0))
    ):
        raise ValueError("execution requires one available fixed 32-CPU set")
    files = plan.get("sealed_files")
    if not isinstance(files, dict) or not files:
        raise ValueError("missing sealed source and artifact inventory")
    if plan.get("harness_git_sha") != _clean_revision(ROOT):
        raise ValueError("execution harness revision changed")
    required = {
        str(path.resolve())
        for directory in (ROOT / "benchmarks", ROOT / "scripts/benchmark_suite")
        for path in directory.rglob("*.py")
    } | {str(ROOT / "scripts/toolkit.py")}
    if not required.issubset(files):
        raise ValueError("incomplete sealed Python harness inventory")
    for path, digest in files.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"sealed execution input changed: {path}")
    from scripts.benchmark_suite.issue406_harness import (
        WHEEL_TARGET,
        validate_acceptance_harness,
        validate_native_profile,
        validate_traced_wheel_invocations,
    )

    if competing_work(set()):
        raise ValueError("another controlled build or measurement is active")
    enabled_builds = {}
    traced_environment = None
    for side in ("baseline", "candidate"):
        release = plan["releases"][side]
        if files.get(release["native_path"]) != release["native_sha256"]:
            raise ValueError("release native is missing from the sealed inventory")
        installed = {
            str(path.resolve())
            for path in (Path(release["site"]) / "calc_flow").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if not installed or not installed.issubset(files):
            raise ValueError("installed Python package files are not fully sealed")
        traced = plan["row7_releases"][side]
        if (
            traced["product_commit"] != release["product_commit"]
            or traced["instrumentation"] != "asof-dispatch-trace"
            or traced["harness_target"] != WHEEL_TARGET
            or files.get(traced["native_path"]) != traced["native_sha256"]
            or files.get(traced["wheel_path"]) != traced["wheel_sha256"]
            or traced["build_metadata"] not in files
        ):
            raise ValueError("row-7 instrumented release identity differs")
        traced_installed = {
            str(path.resolve())
            for path in (Path(traced["site"]) / "calc_flow").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        }
        if not traced_installed or not traced_installed.issubset(files):
            raise ValueError("row-7 installed package is not fully sealed")
        traced_output = Path(traced["build_output"])
        traced_attestation = validate_acceptance_harness(traced_output, WHEEL_TARGET)
        if (
            str(traced_output / f"build-{WHEEL_TARGET}.acceptance-harness.json")
            not in files
        ):
            raise ValueError("row-7 source attestation is not sealed")
        wheel_build = json.loads(Path(traced["build_metadata"]).read_text())
        if (
            wheel_build["failure"] is not None
            or wheel_build["revision"] != release["product_commit"]
            or traced_attestation["product_git_sha"] != release["product_commit"]
            or wheel_build["flags"] != plan["expected_native_flags"]
            or wheel_build["rustc"] != plan["expected_native_rustc"]
            or wheel_build["lock_fingerprint"] != traced_attestation["lock_fingerprint"]
            or wheel_build["native_sha256"] != traced["native_sha256"]
            or wheel_build["wheel_sha256"] != traced["wheel_sha256"]
        ):
            raise ValueError("row-7 compiled wheel settings differ")
        validate_traced_wheel_invocations(wheel_build)
        if traced_environment is None:
            traced_environment = wheel_build["environment"]
        elif traced_environment != wheel_build["environment"]:
            raise ValueError("row-7 paired wheel build environments differ")
        expected_binaries = {
            f"{target}/{mode}"
            for target in ("stream_join_perf", "stream_asof_e2e")
            for mode in (("enabled",) if side == "baseline" else ("enabled", "off"))
        }
        if set(plan["native_binaries"][side]) != expected_binaries:
            raise ValueError("native binary inventory omits or adds an acceptance arm")
        for name, descriptor in plan["native_binaries"][side].items():
            if files.get(descriptor["path"]) != descriptor["sha256"]:
                raise ValueError(
                    "native benchmark is missing from the sealed inventory"
                )
            target, mode = name.split("/")
            if side == "baseline" and mode == "off":
                raise ValueError("main does not support checkpoint-off")
            record = validate_acceptance_harness(
                Path(descriptor["build_output"]), descriptor["harness_target"]
            )
            attestation = str(
                Path(descriptor["build_output"])
                / f"build-{descriptor['harness_target']}.acceptance-harness.json"
            )
            if attestation not in files:
                raise ValueError("native source attestation is not sealed")
            if (
                target not in record["targets"]
                or record["product_git_sha"] != release["product_commit"]
            ):
                raise ValueError("native and wheel product revisions differ")
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
            if mode == "enabled":
                previous = enabled_builds.setdefault(target, build["artifacts"])
                if previous != build["artifacts"]:
                    raise ValueError(
                        "paired compiled dependency/profile inventory differs"
                    )


class AcceptanceRunner:
    """Own fixed-affinity children, their evidence, and the all-inclusive budget."""

    def __init__(self, plan: dict, output: Path, started: float):
        self.plan, self.output, self.started = plan, output, started
        self.workers = {}
        self.processes = {}
        self.environments = {}
        self.resources = []
        self.stop = asyncio.Event()

    async def register_process(self, creation):
        try:
            process = await asyncio.shield(creation)
        except asyncio.CancelledError:
            process = await _complete_owned(creation)
            self.processes[process.pid] = process
            raise
        self.processes[process.pid] = process
        return process

    def cpus(self, case) -> list[int]:
        return self.plan["cpus"][: case.threads]

    def release(self, case, side) -> dict:
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
                *_taskset(
                    self.cpus(case),
                    [
                        sys.executable,
                        "-m",
                        "scripts.benchmark_suite",
                        "worker",
                        "--root",
                        str(root),
                    ],
                ),
                cwd=ROOT,
                env=_child_environment(Path(release["site"]), self.cpus(case)),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
                limit=WORKER_RESPONSE_LIMIT,
            )
        )
        try:
            process = await self.register_process(creation)
        except BaseException:
            log.close()
            raise
        worker = Worker(process, log)
        self.workers[side] = worker
        identity = await worker.request(operation="hello", scope="core")
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

    def _normalize(self, group, side, raw, observation_id, raw_digest):
        from scripts.benchmark_suite.asof_e2e import validate_matched_native
        from scripts.benchmark_suite.issue406_acceptance import matched_native_identity
        from scripts.benchmark_suite.native_components import (
            validate_native_component,
            validate_small_batch,
        )

        case = group[0]
        if case.row in (4, 5):
            raw = validate_native_component(raw)
            if case.row == 4:
                mapped = {member.id: raw for member in group}
            else:
                mapped = {
                    member.id: {
                        **raw,
                        **raw[
                            "armed"
                            if member.workload.endswith("compacting")
                            else "steady"
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
        elif case.workload in ("asof_operator", "join_small_batch"):
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
        else:
            mapped = {case.id: raw}
        normalized = {
            key: {
                **value,
                "product_commit": self.plan["releases"][side]["product_commit"],
                "observation_id": observation_id,
                "raw_observation_sha256": raw_digest,
                "group_member_ids": [member.id for member in group],
            }
            for key, value in mapped.items()
        }
        for member in group:
            validate_observation(member, side, normalized[member.id])
        return normalized

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
                result = await worker.request(
                    operation="acceptance-fixture", case=specification
                )
                if (
                    sha256_file(Path(result["fixture_path"]))
                    != result["fixture_sha256"]
                ):
                    raise ValueError("exported Join fixture changed")
                await worker.close()
                self.processes.pop(worker.process.pid)
                self.workers.pop(side)
            return await self.sample(group, side, round_index, None)
        worker = await self._start_worker(case, side, round_index)
        specification = {
            **asdict(case),
            "family": "join-asof-acceptance",
            "side": side,
            "product_commit": self.plan["releases"][side]["product_commit"],
            "fixture_root": str(self.fixture_root(side, case.mode, round_index)),
        }
        response = await worker.request(operation="prepare", case=specification)
        if response["case"] != specification:
            raise ValueError("worker prepared a different acceptance workload")
        return self._persist(group, side, round_index, None, response["warmup"])

    def _persist(self, group, side, round_index, index, raw):
        destination = self.destination(group, side, round_index, index)
        destination.mkdir(parents=True, exist_ok=True)
        write_json(destination / "observation.json", raw)
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
            raw = await self.workers[side].request(operation="sample")
        else:
            raw = await self._native(group, side, round_index, index)
        return self._persist(group, side, round_index, index, raw)

    async def _native(self, group, side, round_index, index):
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
        destination = self.destination(group, side, round_index, index)
        destination.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        with (
            (destination / "stderr.log").open("wb") as log,
            (destination / "stdout.log").open("wb") as stdout_file,
        ):
            creation = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *_taskset(self.cpus(case), [binary["path"], *arguments]),
                    cwd=ROOT,
                    env=_child_environment(None, self.cpus(case)),
                    stdout=stdout_file,
                    stderr=log,
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
        self.processes.pop(process.pid)
        if process.returncode:
            raise RuntimeError(f"native component failed; see {destination}")
        observed = json.loads(stdout)
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
        return {**observed, "native_executable_sha256": binary["sha256"]}

    async def finish_round(self):
        for side, worker in tuple(self.workers.items()):
            try:
                if worker.process.returncode is None:
                    response = await worker.request(operation="finish")
                    if response.get("state") != "completed":
                        raise ValueError("worker failed to finish its round")
            finally:
                await worker.close()
                self.processes.pop(worker.process.pid, None)
                self.workers.pop(side, None)

    async def resources_guard(self):
        import psutil

        while not self.stop.is_set():
            controller_rss = psutil.Process().memory_info().rss
            rss, children = controller_rss, []
            for pid, process in tuple(self.processes.items()):
                if process.returncode is not None:
                    continue
                try:
                    observed = psutil.Process(pid)
                    for item in (observed, *observed.children(recursive=True)):
                        memory = item.memory_info().rss
                        rss += memory
                        children.append(
                            {
                                "pid": item.pid,
                                "rss": memory,
                                "cpu_seconds": sum(item.cpu_times()[:2]),
                                "threads": [list(thread) for thread in item.threads()],
                            }
                        )
                except psutil.NoSuchProcess:
                    pass
            memory = psutil.virtual_memory().available
            disk = shutil.disk_usage(self.output).free
            used = 0
            for path in self.output.rglob("*"):
                with suppress(FileNotFoundError):
                    if path.is_file():
                        used += path.stat().st_size
            conflicts = competing_work(set(self.processes))
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

    async def settle(self):
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


async def run(plan: dict, output: Path) -> dict:
    """Execute once, charging setup, preflight, collection, cleanup and sealing."""
    started = time.monotonic() - plan.get("precharged_seconds", 0)
    if output.exists() and any(output.iterdir()):
        raise ValueError("acceptance execution requires a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    runner = AcceptanceRunner(plan, output, started)
    report = {
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
    }
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
        guard = asyncio.create_task(runner.resources_guard())
        collection = asyncio.create_task(collect(runner, output))
        remaining = (
            plan["budget_seconds"]
            - plan["settlement_seconds"]
            - (time.monotonic() - started)
        )
        async with asyncio.timeout(remaining):
            done, _ = await asyncio.wait(
                (collection, guard), return_when=asyncio.FIRST_COMPLETED
            )
            if guard in done:
                guard.result()
                raise RuntimeError("resource guard exited unexpectedly")
            report = {**report, **collection.result()}
        validate_complete_results(report)
        report["state"] = "complete"
        report["assessment"]["environment_suitable"] = True
    except BaseException as error:
        report.update(state="stopped", error=f"{type(error).__name__}: {error}")
    finally:
        runner.stop.set()
        if collection is not None and not collection.done():
            collection.cancel()
        # Signal cancellation before awaiting the collector's owned round cleanup.
        report["settlement"] = await _complete_owned(
            asyncio.create_task(runner.settle())
        )
        for task in (collection, guard):
            if task is not None:
                with suppress(BaseException):
                    await _complete_owned(task)
        if (
            guard is not None
            and not guard.cancelled()
            and guard.exception() is not None
        ):
            report["state"] = "stopped"
            report["assessment"]["environment_suitable"] = False
            report["error"] = f"resource guard: {guard.exception()}"
        partial = output / "partial.json"
        if partial.exists() and "measurements" not in report:
            report["measurements"] = json.loads(partial.read_text())["measurements"]
        if report["settlement"]["forced_kill_pids"]:
            report["assessment"]["environment_suitable"] = False
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
        write_json(output / "resources.json", runner.resources)
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
        for sig, handler in previous_signals.items():
            loop.remove_signal_handler(sig)
            signal.signal(sig, handler)
    return report


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if sha256_file(args.plan) != args.plan_sha256:
        raise ValueError("execution plan differs from the reviewed seal")
    report = asyncio.run(run(json.loads(args.plan.read_text()), args.output))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "measurements"}
        )
    )
    return 0 if report["state"] == "complete" and report["budget_satisfied"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
