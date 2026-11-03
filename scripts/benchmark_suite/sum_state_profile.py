"""Decode focused SUM CPU profiles using exact ELF ranges and verified clocks."""

from __future__ import annotations

import bisect
import collections
import gzip
import json
import os
import re
import subprocess
import time
from pathlib import Path

from scripts.toolkit import write_json


def command(argv, deadline, env):
    remaining = deadline - time.monotonic()
    if remaining <= 5:
        raise TimeoutError("profile analysis deadline")
    return subprocess.check_output(argv, env=env, text=True, timeout=remaining - 2)


def symbol_tables(profile, deadline, env):
    tables = {}
    for library in profile["libs"]:
        path = Path(library["path"])
        if not path.is_file():
            continue
        result = command(
            ["nm", "-n", "-S", "-C", "--defined-only", str(path)], deadline, env
        )
        if not result.strip():
            result = command(
                ["nm", "-D", "-n", "-S", "-C", "--defined-only", str(path)],
                deadline,
                env,
            )
        rows = []
        for line in result.splitlines():
            fields = line.split(maxsplit=3)
            if len(fields) == 4 and fields[2] in "tTwW":
                start, size = int(fields[0], 16), int(fields[1], 16)
                if size:
                    rows.append((start, start + size, fields[3]))
        rows.sort()
        tables[library["debugName"]] = ([r[0] for r in rows], rows)
    return tables


def frame_names(profile, thread, tables):
    names = []
    frames, funcs = thread["frameTable"], thread["funcTable"]
    resources, strings = thread["resourceTable"], thread["stringArray"]
    for index in range(frames["length"]):
        func = frames["func"][index]
        resource = funcs["resource"][func]
        address = frames["address"][index]
        name = strings[funcs["name"][func]]
        if (
            resource is not None
            and resource >= 0
            and address is not None
            and address >= 0
        ):
            library = profile["libs"][resources["lib"][resource]]["debugName"]
            table = tables.get(library)
            resolved = None
            if table:
                position = bisect.bisect_right(table[0], address) - 1
                if position >= 0 and address < table[1][position][1]:
                    resolved = table[1][position][2]
            name = f"{resolved or hex(address)} [{library}]"
        names.append(re.sub(r"::h[0-9a-f]{16}(?= |$)", "", name))
    return names


def clock_alignment(raw, profile, pid):
    original = collections.defaultdict(list)
    for line in raw.splitlines():
        match = re.match(r"\s*(\d+)/(\d+)\s+(\d+)\.(\d+):", line)
        if match and int(match[1]) == pid:
            original[match[2]].append(
                int(match[3]) * 1000 + int(match[4].ljust(9, "0")) / 1e6
            )
    imported = collections.defaultdict(list)
    for thread in profile["threads"]:
        if str(thread["pid"]).split(".")[0] == str(pid):
            imported[str(thread["tid"]).split(".")[0]].extend(thread["samples"]["time"])
    imported = {key: sorted(value) for key, value in imported.items() if value}
    if set(original) != set(imported) or not original:
        raise ValueError("profile clock thread inventory mismatch")
    offset = min(min(v) for v in original.values()) - min(
        min(v) for v in imported.values()
    )
    errors, count = [], 0
    for tid, values in original.items():
        if len(values) != len(imported[tid]):
            raise ValueError("profile clock sample count mismatch")
        errors.extend(
            abs(a - b - offset) for a, b in zip(values, imported[tid], strict=True)
        )
        count += len(values)
    if max(errors) >= 0.00001:
        raise ValueError("profile clock alignment exceeds10ns")
    return {
        "offset_ms": offset,
        "max_error_ms": max(errors),
        "matched_sample_count": count,
    }


def decode(args, acquisition, builds, deadline):
    env = dict(os.environ)
    if args.perf_library:
        env["LD_LIBRARY_PATH"] = str(args.perf_library)
    reports = []
    for item in acquisition["profiles"]:
        root = args.output / item["path"]
        output = root / "profile.json.gz"
        command(
            [
                str(args.samply),
                "import",
                "--save-only",
                "--output",
                str(output),
                str(root / "perf.data"),
            ],
            deadline,
            env,
        )
        profile = json.loads(gzip.decompress(output.read_bytes()))
        tables = symbol_tables(profile, deadline, env)
        pid = item["rows"][0]["pid"]
        raw = command(
            [
                str(args.perf),
                "script",
                "-i",
                str(root / "perf.data"),
                "--ns",
                "-F",
                "pid,tid,time",
                "-G",
            ],
            deadline,
            env,
        )
        (root / "sample-timestamps.txt").write_text(raw)
        alignment = clock_alignment(raw, profile, pid)
        write_json(root / "clock-alignment.json", alignment)
        windows = [
            (
                r["start_ns"] / 1e6 - alignment["offset_ms"],
                r["end_ns"] / 1e6 - alignment["offset_ms"],
            )
            for r in item["windows"]
        ]
        own, inclusive, threads, phases, stacks = (
            collections.Counter() for _ in range(5)
        )
        window_counts = [0] * len(windows)
        unresolved = total = 0
        for thread in profile["threads"]:
            if str(thread["pid"]).split(".")[0] != str(pid):
                continue
            names = frame_names(profile, thread, tables)
            table = thread["stackTable"]
            samples = thread["samples"]
            weights = samples.get("weight") or [1] * samples["length"]
            cache = {}
            for stack, weight, when in zip(
                samples["stack"], weights, samples["time"], strict=True
            ):
                selected = next(
                    (
                        i
                        for i, (start, end) in enumerate(windows)
                        if start <= when <= end
                    ),
                    None,
                )
                if stack is None or selected is None:
                    continue
                if stack not in cache:
                    chain, cursor = [], stack
                    while cursor is not None:
                        chain.append(names[table["frame"][cursor]])
                        cursor = table["prefix"][cursor]
                    cache[stack] = tuple(chain)
                chain = cache[stack]
                count = weight or 1
                total += count
                unresolved += count if chain[0].startswith("0x") else 0
                window_counts[selected] += count
                own[chain[0]] += count
                inclusive.update({name: count for name in set(chain)})
                stacks[chain] += count
                threads[f"{thread['name']} tid={thread['tid']}"] += count
                phase = next(
                    (
                        phase
                        for phase in (
                            "update_chunk",
                            "finish_transaction",
                            "commit",
                            "ExpressionOperator",
                        )
                        if any(("::" + phase) in frame for frame in chain)
                    ),
                    "other",
                )
                phases[phase] += count
        if not total or not all(window_counts):
            raise ValueError("missing sampled processing window")

        def ranked(values, sample_total=total):
            return [
                {"name": name, "samples": count, "percent": 100 * count / sample_total}
                for name, count in values.most_common(40)
            ]

        frames = sorted({frame for stack in stacks for frame in stack})
        frame_ids = {name: index for index, name in enumerate(frames)}
        stack_data = {
            "order": "leaf first",
            "frames": frames,
            "stacks": [
                {"frames": [frame_ids[frame] for frame in chain], "samples": count}
                for chain, count in stacks.most_common()
            ],
        }
        (root / "stacks.json.gz").write_bytes(
            gzip.compress(
                json.dumps(stack_data, separators=(",", ":")).encode(), mtime=0
            )
        )
        report = {
            "side": item["side"],
            "binary_sha256": builds[item["side"]]["binary_sha256"],
            "perf_data_sha256": item["perf_data_sha256"],
            "windows": item["windows"],
            "clock_alignment": alignment,
            "samples": total,
            "samples_per_window": window_counts,
            "unresolved_leaf_samples": unresolved,
            "self": ranked(own),
            "inclusive": ranked(inclusive),
            "threads": ranked(threads),
            "disjoint_phases": ranked(phases),
            "scope": "user-mode CPU only; first job excluded; inclusive shares overlap",
            "symbolization": (
                "GNU nm -n -S -C exact nonzero-sized code ranges; "
                "dynamic symbol fallback; no source-line/inline attribution"
            ),
        }
        write_json(root / "summary.json", report)
        reports.append(report)
    write_json(args.output / "profile-summary.json", reports)
    return reports
