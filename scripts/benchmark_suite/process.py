"""Bounded subprocess ownership shared by measurement and legacy adapters."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from scripts.benchmark_suite.catalog import THREADS

ROOT = Path(__file__).resolve().parents[2]


def child_environment(site: Path | None = None) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(str(p) for p in (site, ROOT) if p is not None),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TOKIO_WORKER_THREADS": str(THREADS),
        "POLARS_MAX_THREADS": str(THREADS),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "JAX_PLATFORMS": "cpu",
        "npm_config_cache": str(ROOT / "target/npm-cache"),
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(ROOT / "target/uv-cache")),
    }


async def command(
    argv: list[str],
    *,
    cwd: Path,
    log: Path,
    env: dict | None = None,
    timeout: float = 7200,
) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("wb") as output:
        process = await asyncio.create_subprocess_exec(
            *argv, cwd=cwd, env=env, stdout=output, stderr=asyncio.subprocess.STDOUT
        )
        try:
            code = await asyncio.wait_for(process.wait(), timeout=timeout)
        except BaseException:
            await stop(process)
            raise
    if code:
        raise RuntimeError(f"command exited {code}; see {log}")


async def stop(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except TimeoutError:
        process.kill()
        await process.wait()


async def install(release: dict, root: Path) -> Path:
    site = root / "site"
    await command(
        [
            "uv",
            "pip",
            "install",
            "--python",
            sys.executable,
            "--no-deps",
            "--target",
            str(site),
            release["wheel_path"],
        ],
        cwd=ROOT,
        log=root / "install.log",
        env=child_environment(),
    )
    return site


class Worker:
    def __init__(self, process: asyncio.subprocess.Process, log) -> None:
        self.process, self.log = process, log

    @classmethod
    async def start(cls, site: Path, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        log = (root / "stderr.log").open("wb")
        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "scripts.benchmark_suite",
                "worker",
                "--root",
                str(root),
                cwd=ROOT,
                env=child_environment(site),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=log,
            )
        except BaseException:
            log.close()
            raise
        return cls(process, log)

    async def request(self, **message) -> dict:
        self.process.stdin.write((json.dumps(message) + "\n").encode())
        await self.process.stdin.drain()
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout=900)
        if not line:
            raise RuntimeError(f"benchmark worker exited; see {self.log.name}")
        response = json.loads(line)
        if not isinstance(response, dict) or "error" in response:
            raise RuntimeError(f"invalid worker result: {response}")
        return response

    async def close(self) -> None:
        try:
            if self.process.returncode is None:
                self.process.stdin.close()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=10)
                except TimeoutError:
                    await stop(self.process)
        finally:
            self.log.close()
