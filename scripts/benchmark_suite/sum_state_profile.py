"""Decode focused SUM CPU profiles using exact ELF ranges and verified clocks."""

from __future__ import annotations

import bisect
import collections
import gzip
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from scripts.toolkit import command_output, require_executable, write_json


def command(argv, deadline, env):
    remaining = deadline - time.monotonic()
    if remaining <= 5:
        raise TimeoutError("profile analysis deadline")
    executable = require_executable(argv[0])
    return command_output([executable, *argv[1:]], env=env, timeout=remaining - 2)


def _symbols(output):
    rows = []
    for line in output.splitlines():
        fields = line.split(maxsplit=3)
        if len(fields) != 4 or fields[2] not in "tTwW":
            continue
        start, size = int(fields[0], 16), int(fields[1], 16)
        if size:
            rows.append((start, start + size, fields[3]))
    return sorted(rows)


def _library_symbols(path, deadline, env):
    argv = ["nm", "-n", "-S", "-C", "--defined-only", str(path)]
    result = command(argv, deadline, env)
    if not result.strip():
        result = command([argv[0], "-D", *argv[1:]], deadline, env)
    rows = _symbols(result)
    return [row[0] for row in rows], rows


def symbol_tables(profile, deadline, env):
    tables = {}
    for library in profile["libs"]:
        path = Path(library["path"]).resolve()
        if path.is_file():
            tables[library["debugName"]] = _library_symbols(path, deadline, env)
    return tables


def _symbol_at(table, address):
    if table is None:
        return None
    position = bisect.bisect_right(table[0], address) - 1
    if position >= 0 and address < table[1][position][1]:
        return table[1][position][2]
    return None


def _frame_name(profile, thread, tables, index):
    frames, funcs = thread["frameTable"], thread["funcTable"]
    func = frames["func"][index]
    resource = funcs["resource"][func]
    address = frames["address"][index]
    name = thread["stringArray"][funcs["name"][func]]
    if resource is None or address is None:
        return name
    if resource < 0 or address < 0:
        return name
    library_index = thread["resourceTable"]["lib"][resource]
    library = profile["libs"][library_index]["debugName"]
    resolved = _symbol_at(tables.get(library), address)
    return f"{resolved or hex(address)} [{library}]"


def frame_names(profile, thread, tables):
    return [
        re.sub(
            r"::h[0-9a-f]{16}(?= |$)", "", _frame_name(profile, thread, tables, index)
        )
        for index in range(thread["frameTable"]["length"])
    ]


def _original_samples(raw, pid):
    original = collections.defaultdict(list)
    for line in raw.splitlines():
        match = re.match(r"\s*(\d+)/(\d+)\s+(\d+)\.(\d+):", line)
        if match and int(match[1]) == pid:
            original[match[2]].append(
                int(match[3]) * 1000 + int(match[4].ljust(9, "0")) / 1e6
            )
    return original


def _same_pid(thread, pid):
    return str(thread["pid"]).split(".")[0] == str(pid)


def _imported_samples(profile, pid):
    imported = collections.defaultdict(list)
    for thread in profile["threads"]:
        if _same_pid(thread, pid):
            imported[str(thread["tid"]).split(".")[0]].extend(thread["samples"]["time"])
    return {key: sorted(value) for key, value in imported.items() if value}


def _thread_errors(original, imported, offset):
    if len(original) != len(imported):
        raise ValueError("profile clock sample count mismatch")
    return [abs(a - b - offset) for a, b in zip(original, imported, strict=True)]


def clock_alignment(raw, profile, pid):
    original = _original_samples(raw, pid)
    imported = _imported_samples(profile, pid)
    if set(original) != set(imported) or not original:
        raise ValueError("profile clock thread inventory mismatch")
    offset = min(map(min, original.values())) - min(map(min, imported.values()))
    errors = [
        _thread_errors(values, imported[tid], offset)
        for tid, values in original.items()
    ]
    maximum = max(map(max, errors))
    if maximum >= 0.00001:
        raise ValueError("profile clock alignment exceeds10ns")
    return {
        "offset_ms": offset,
        "max_error_ms": maximum,
        "matched_sample_count": sum(map(len, errors)),
    }


def _phase(chain, rules=()):
    for label, needle in rules:
        if any(frame.rsplit(" [", 1)[0] == needle for frame in chain):
            return label
    for phase in ("update_chunk", "finish_transaction", "commit", "ExpressionOperator"):
        if any(("::" + phase) in frame for frame in chain):
            return phase
    return "other"


def _chain(stack, names, table):
    result = []
    cursor = stack
    while cursor is not None:
        result.append(names[table["frame"][cursor]])
        cursor = table["prefix"][cursor]
    return tuple(result)


def _window_index(when, windows):
    return next(
        (i for i, (start, end) in enumerate(windows) if start <= when <= end), None
    )


@dataclass(slots=True)
class SampleCounts:
    """Own the mutable counters of one decoded profile."""

    windows: list[int]
    phase_frames: tuple = ()
    total: int = 0
    unresolved: int = 0
    own: collections.Counter = field(default_factory=collections.Counter)
    inclusive: collections.Counter = field(default_factory=collections.Counter)
    threads: collections.Counter = field(default_factory=collections.Counter)
    phases: collections.Counter = field(default_factory=collections.Counter)
    stacks: collections.Counter = field(default_factory=collections.Counter)
    worker_threads: collections.Counter = field(default_factory=collections.Counter)

    def observe(self, chain, thread, count, selected):
        self.total += count
        self.unresolved += count if chain[0].startswith("0x") else 0
        self.windows[selected] += count
        self.own[chain[0]] += count
        self.inclusive.update({name: count for name in set(chain)})
        self.stacks[chain] += count
        self.threads[f"{thread['name']} tid={thread['tid']}"] += count
        phase = _phase(chain, self.phase_frames)
        self.phases[phase] += count
        if phase == "sql_local_update":
            tid = str(thread["tid"]).split(".")[0]
            self.worker_threads[(selected + 1, tid, thread["name"])] += count

    def worker_distribution(self):
        return [
            {"window": window, "tid": tid, "name": name, "samples": samples}
            for (window, tid, name), samples in sorted(self.worker_threads.items())
        ]

    def ranked(self, values):
        return [
            {"name": name, "samples": count, "percent": 100 * count / self.total}
            for name, count in values.most_common(40)
        ]

    def stack_data(self):
        frames = sorted({frame for stack in self.stacks for frame in stack})
        ids = {name: index for index, name in enumerate(frames)}
        return {
            "order": "leaf first",
            "frames": frames,
            "stacks": [
                {"frames": [ids[frame] for frame in chain], "samples": count}
                for chain, count in self.stacks.most_common()
            ],
        }


def _count_thread(profile, thread, tables, windows, counts):
    names = frame_names(profile, thread, tables)
    samples = thread["samples"]
    weights = samples.get("weight") or [1] * samples["length"]
    cache = {}
    for stack, weight, when in zip(
        samples["stack"], weights, samples["time"], strict=True
    ):
        selected = _window_index(when, windows)
        if stack is None or selected is None:
            continue
        if stack not in cache:
            cache[stack] = _chain(stack, names, thread["stackTable"])
        counts.observe(cache[stack], thread, weight or 1, selected)


def _count_samples(profile, tables, pid, windows, phase_frames=()):
    counts = SampleCounts([0] * len(windows), phase_frames=phase_frames)
    for thread in profile["threads"]:
        if _same_pid(thread, pid):
            _count_thread(profile, thread, tables, windows, counts)
    if not counts.total or not all(counts.windows):
        raise ValueError("missing sampled processing window")
    return counts


def _import_profile(args, root, deadline, env):
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
    return json.loads(gzip.decompress(output.read_bytes()))


def _aligned_windows(args, item, root, profile, deadline):
    env = _environment(args)
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
    alignment = clock_alignment(raw, profile, item["rows"][0]["pid"])
    write_json(root / "clock-alignment.json", alignment)
    windows = [
        (
            r["start_ns"] / 1e6 - alignment["offset_ms"],
            r["end_ns"] / 1e6 - alignment["offset_ms"],
        )
        for r in item["windows"]
    ]
    return alignment, windows


def _environment(args):
    env = dict(os.environ)
    if args.perf_library:
        env["LD_LIBRARY_PATH"] = str(args.perf_library)
    return env


def _report(item, build, alignment, counts):
    return {
        "side": item["side"],
        "binary_sha256": build["binary_sha256"],
        "perf_data_sha256": item["perf_data_sha256"],
        "windows": item["windows"],
        "clock_alignment": alignment,
        "samples": counts.total,
        "samples_per_window": counts.windows,
        "unresolved_leaf_samples": counts.unresolved,
        "self": counts.ranked(counts.own),
        "inclusive": counts.ranked(counts.inclusive),
        "threads": counts.ranked(counts.threads),
        "disjoint_phases": counts.ranked(counts.phases),
        "phase_frame_rules": counts.phase_frames,
        "sql_worker_samples_by_window": counts.worker_distribution(),
        "scope": "user-mode CPU only; first job excluded; inclusive shares overlap",
        "symbolization": (
            "GNU nm -n -S -C exact nonzero-sized code ranges; "
            "dynamic symbol fallback; no source-line/inline attribution"
        ),
    }


def _decode_item(args, item, build, deadline):
    root = args.output / item["path"]
    env = _environment(args)
    profile = _import_profile(args, root, deadline, env)
    tables = symbol_tables(profile, deadline, env)
    alignment, windows = _aligned_windows(args, item, root, profile, deadline)
    counts = _count_samples(
        profile,
        tables,
        item["rows"][0]["pid"],
        windows,
        getattr(args, "sql_phase_frames", ()),
    )
    (root / "stacks.json.gz").write_bytes(
        gzip.compress(
            json.dumps(counts.stack_data(), separators=(",", ":")).encode(), mtime=0
        )
    )
    report = _report(item, build, alignment, counts)
    write_json(root / "summary.json", report)
    return report


def decode(args, acquisition, builds, deadline):
    reports = [
        _decode_item(args, item, builds[item["side"]], deadline)
        for item in acquisition["profiles"]
    ]
    write_json(args.output / "profile-summary.json", reports)
    return reports
