"""Bounded exact-output evidence for the disjoint-block interval Join diagnostic.

The oracle is deliberately part of long-run sink work. Its recorded cost must not
be subtracted from overlapping native execution to invent kernel-only throughput.
Finite engine comparisons retain their full Arrow oracle outside both timers.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc


@dataclass(frozen=True, slots=True)
class JoinInventory:
    """Fixed input inventory; small instances support exhaustive oracle tests."""

    block_rows: int = 200_000
    blocks: int = 80
    warmup_blocks: int = 4
    batch_rows: int = 8192

    def __post_init__(self):
        if any(
            type(value) is not int or value <= 0
            for value in (
                self.block_rows,
                self.blocks,
                self.warmup_blocks,
                self.batch_rows,
            )
        ):
            raise ValueError("inventory values must be positive integers")
        if self.block_rows % 64 or self.batch_rows % 64:
            raise ValueError("input sizes must preserve complete 64-key ticks")
        if self.warmup_blocks >= self.blocks:
            raise ValueError("warmup must leave a nonempty measured suffix")

    @property
    def expected_rows_per_block(self) -> int:
        return sum(
            max(0, self.block_rows - abs(offset) * 64) for offset in range(-5, 6)
        )

    @property
    def expected_rows(self) -> int:
        return self.blocks * self.expected_rows_per_block

    @property
    def bitmap_bytes(self) -> int:
        return self.blocks * self.block_rows * 11

    @property
    def block_stride_seconds(self) -> int:
        return self.block_rows // 64 + 6


def build_input_blocks(
    base: pa.Table, inventory: JoinInventory
) -> tuple[pa.Table, ...]:
    """Advance IDs and time while sharing immutable maintained prices and keys."""
    if base.num_rows != inventory.block_rows:
        raise ValueError("base fixture row count differs from the inventory")
    sequence = base["sequence"].to_numpy()
    if not np.array_equal(sequence, np.arange(inventory.block_rows, dtype=np.uint64)):
        raise ValueError("base fixture sequence must enumerate its block")
    sequence_index = base.column_names.index("sequence")
    time_index = base.column_names.index("event_time")
    result = []
    for block in range(inventory.blocks):
        shifted = base.set_column(
            sequence_index,
            base.schema.field(sequence_index),
            pa.array(sequence + block * inventory.block_rows, type=pa.uint64()),
        )
        elapsed = pa.scalar(
            block * inventory.block_stride_seconds * 1_000_000, type=pa.duration("us")
        )
        shifted = shifted.set_column(
            time_index,
            base.schema.field(time_index),
            pc.add(base["event_time"], elapsed),
        )
        result.append(shifted)
    return tuple(result)


def _validate_output_shape(table: pa.Table, maximum_rows: int) -> None:
    if table.num_rows > maximum_rows:
        raise ValueError("output batch exceeds the declared Join output bound")
    if table.column_names != ["sequence", "right_sequence", "value"]:
        raise ValueError("output columns differ from the full oracle")
    if [column.type for column in table.columns] != [
        pa.uint64(),
        pa.uint64(),
        pa.float64(),
    ] or any(column.null_count for column in table.columns):
        raise ValueError("output types/nulls differ from the full oracle")


def _validated_pair_geometry(table: pa.Table, inventory: JoinInventory):
    left = table["sequence"].to_numpy()
    right = table["right_sequence"].to_numpy()
    limit = inventory.blocks * inventory.block_rows
    if np.any(left >= limit) or np.any(right >= limit):
        raise ValueError("output identity outside input range")
    left = left.astype(np.int64, copy=False)
    right = right.astype(np.int64, copy=False)
    block = left // inventory.block_rows
    if np.any(block != right // inventory.block_rows):
        raise ValueError("output contains an impossible cross-block match")
    delta = right - left
    if np.any(delta % 64):
        raise ValueError("output match keys differ")
    if np.any(np.abs(delta) > 5 * 64):
        raise ValueError("output match exceeds inclusive time bounds")
    return left, right, block, delta


class BoundedJoinOracle:
    """Own exact pair occupancy without retaining delivered output tables."""

    def __init__(self, inventory: JoinInventory):
        self.inventory = inventory
        self.seen = np.zeros(inventory.bitmap_bytes, dtype=np.bool_)
        self.block_counts = np.zeros(inventory.blocks, dtype=np.int64)
        self.rows = 0

    @property
    def bitmap_bytes(self) -> int:
        return self.seen.nbytes

    def accept(self, table: pa.Table) -> None:
        """Validate every identity/value before changing owned occupancy."""
        _validate_output_shape(table, self.inventory.batch_rows * 11)
        if table.num_rows == 0:
            return
        left, right, block, delta = _validated_pair_geometry(table, self.inventory)
        local_left = left % self.inventory.block_rows
        local_right = right % self.inventory.block_rows
        left_price = 100 + local_left % 257 / 8 + local_left % 64 / 8
        right_price = 100 + local_right % 257 / 8 + local_right % 64 / 8
        if not np.all(table["value"].to_numpy() == left_price * right_price):
            raise ValueError("output value differs from the full oracle")
        indices = np.sort(left * 11 + delta // 64 + 5)
        if np.any(indices[1:] == indices[:-1]) or np.any(self.seen[indices]):
            raise ValueError("duplicate output pair")
        self.seen[indices] = True
        self.block_counts += np.bincount(block, minlength=self.inventory.blocks)
        self.rows += table.num_rows

    def finish(self) -> dict:
        """Prove complete multiplicity after admitting only distinct valid pairs."""
        if (
            self.rows != self.inventory.expected_rows
            or np.any(self.block_counts != self.inventory.expected_rows_per_block)
            or np.count_nonzero(self.seen) != self.rows
        ):
            raise ValueError("incomplete output pair inventory")
        return {
            "rows": self.rows,
            "expected_rows": self.inventory.expected_rows,
            "block_counts": self.block_counts.tolist(),
            "bitmap_bytes": self.bitmap_bytes,
            "maximum_absolute_error": 0.0,
        }


class SustainedJoinSink:
    """Own the bounded oracle and actual observed steady-interval boundaries."""

    def __init__(
        self,
        inventory: JoinInventory,
        *,
        snapshot: Callable[[], dict],
        clock: Callable[[], int] = time.perf_counter_ns,
        cpu_clock: Callable[[], int] = time.thread_time_ns,
    ):
        self.oracle = BoundedJoinOracle(inventory)
        self.expected_rows = inventory.expected_rows
        self.snapshot = snapshot
        self.clock = clock
        self.cpu_clock = cpu_clock
        self.opened = asyncio.Event()
        self.complete = asyncio.Event()
        self.steady_start = None
        self.steady_end = None
        self.validation_wall_ns = 0
        self.validation_cpu_ns = 0
        self.arrow_wall_ns = 0
        self.maximum_batch_rows = 0

    @property
    def rows(self) -> int:
        return self.oracle.rows

    def boundary(self) -> dict:
        boundary = {
            "clock_ns": self.clock(),
            "rows": self.rows,
            "validation_wall_ns": self.validation_wall_ns,
            "validation_cpu_ns": self.validation_cpu_ns,
            "arrow_wall_ns": self.arrow_wall_ns,
        }
        boundary["status_before_ns"] = self.clock()
        boundary["status"] = self.snapshot()
        boundary["status_after_ns"] = self.clock()
        return boundary

    async def open(self) -> None:
        self.opened.set()

    async def write(self, batch) -> None:
        arrow_started = self.clock()
        table = batch.to_pyarrow()
        started = self.clock()
        self.arrow_wall_ns += started - arrow_started
        cpu_started = self.cpu_clock()
        self.oracle.accept(table)
        self.validation_cpu_ns += self.cpu_clock() - cpu_started
        self.validation_wall_ns += self.clock() - started
        self.maximum_batch_rows = max(self.maximum_batch_rows, table.num_rows)
        inventory = self.oracle.inventory
        if self.steady_start is None and np.all(
            self.oracle.block_counts[: inventory.warmup_blocks]
            == inventory.expected_rows_per_block
        ):
            self.steady_start = self.boundary()
        if self.rows == self.expected_rows:
            self.steady_end = self.boundary()
            self.complete.set()

    async def close(self) -> None:
        return None


def _epoch_retired(epoch, current, completed) -> bool:
    return current != epoch and (
        (completed is not None and completed >= epoch)
        or (current is not None and current > epoch)
    )


def _observation_clock_invalid(
    before: int, after: int, previous_before: int | None
) -> bool:
    return before > after or (previous_before is not None and before < previous_before)


def checkpoint_windows(
    observations: list[dict], *, start_ns: int, end_ns: int
) -> list[dict]:
    """Bound only observed epoch transitions; retain gaps and terminal promotion."""
    epochs = {}
    previous_before = None

    def state(epoch):
        return epochs.setdefault(
            epoch,
            {
                "epoch": epoch,
                "start_lower_ns": None,
                "start_upper_ns": None,
                "completion_lower_ns": None,
                "completion_upper_ns": None,
                "terminal_observed": False,
                "observed_phases": [],
            },
        )

    def record_current(checkpoint, current, before, after):
        if current is not None:
            item = state(current)
            item["terminal_observed"] |= checkpoint["terminal"]
            phase = checkpoint["phase"]
            if phase not in item["observed_phases"]:
                item["observed_phases"].append(phase)
            elapsed = checkpoint["elapsed_micros"]
            if elapsed is not None:
                lower, upper = before - elapsed * 1000 - 999, after - elapsed * 1000
                prior_lower, prior_upper = (
                    item["start_lower_ns"],
                    item["start_upper_ns"],
                )
                item["start_lower_ns"] = (
                    lower if prior_lower is None else max(prior_lower, lower)
                )
                item["start_upper_ns"] = (
                    upper if prior_upper is None else min(prior_upper, upper)
                )

    for observation in observations:
        before, after = observation["before_ns"], observation["after_ns"]
        if _observation_clock_invalid(before, after, previous_before):
            raise ValueError("checkpoint observation clocks are not monotonic")
        checkpoint = observation["checkpoint"]
        current = checkpoint["current_epoch"]
        record_current(checkpoint, current, before, after)
        completed = checkpoint["last_completed_epoch"]
        if completed is not None:
            state(completed)
        for epoch, item in epochs.items():
            retired = _epoch_retired(epoch, current, completed)
            if retired and item["completion_upper_ns"] is None:
                item["completion_lower_ns"] = previous_before
                item["completion_upper_ns"] = after
        previous_before = before

    return _finalize_checkpoint_windows(epochs, start_ns, end_ns)


def _inside_steady(
    item: dict, lower, upper, completed, start_ns: int, end_ns: int
) -> bool:
    return (
        lower is not None
        and upper is not None
        and lower <= upper
        and lower >= start_ns
        and completed is not None
        and completed >= upper
        and completed <= end_ns
        and not item["terminal_observed"]
    )


def _finalize_checkpoint_windows(epochs, start_ns, end_ns):
    result = []
    for epoch in sorted(epochs):
        item = {**epochs[epoch]}
        lower, upper = item["start_lower_ns"], item["start_upper_ns"]
        completed = item["completion_upper_ns"]
        item["inside_steady"] = _inside_steady(
            item, lower, upper, completed, start_ns, end_ns
        )
        if upper is not None and item["completion_lower_ns"] is not None:
            item["duration_lower_ns"] = max(0, item["completion_lower_ns"] - upper)
            item["duration_upper_ns"] = max(0, completed - lower)
        result.append(item)
    return result
