"""Explicit row-count, duration, and event-time grouping declarations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from calc_flow.symbolic.domains import namespace_error
from calc_flow.symbolic.expr import ColumnExpr
from calc_flow.symbolic.types import require_positive_int


@dataclass(frozen=True, slots=True)
class RowFrame:
    """A positive row-count rolling frame ``rows [i - size + 1, i]``."""

    size: int

    def __post_init__(self) -> None:
        require_positive_int(self.size, "RowFrame.size")


@dataclass(frozen=True, slots=True)
class DurationFrame:
    """A positive exact-microseconds duration frame ``(t - micros, t]``."""

    micros: int

    def __post_init__(self) -> None:
        require_positive_int(self.micros, "DurationFrame.micros")


@dataclass(frozen=True, slots=True, eq=False)
class EventTimeBucket:
    """A fixed UTC event-time bucket of exact positive width."""

    event_time: ColumnExpr
    width_micros: int
    partition_by: tuple[ColumnExpr, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.event_time, ColumnExpr):
            raise namespace_error(
                "event_time_bucket", "event_time", "ColumnExpr", self.event_time
            )
        require_positive_int(
            self.width_micros, "calc_flow.symbolic.event_time_bucket.width_micros"
        )
        object.__setattr__(self, "partition_by", tuple(self.partition_by))
        for index, column in enumerate(self.partition_by):
            if not isinstance(column, ColumnExpr):
                raise namespace_error(
                    "event_time_bucket",
                    f"partition_by[{index}]",
                    "ColumnExpr",
                    column,
                )


@dataclass(frozen=True, slots=True, eq=False)
class CrossSectionGroup:
    """One complete cross-section grouping declaration."""

    event_time: ColumnExpr
    bucket: EventTimeBucket | None
    partition_by: tuple[ColumnExpr, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.event_time, ColumnExpr):
            raise namespace_error(
                "cross_section_group",
                "event_time",
                "ColumnExpr",
                self.event_time,
            )
        if self.bucket is not None and not isinstance(self.bucket, EventTimeBucket):
            raise namespace_error(
                "cross_section_group",
                "bucket",
                "EventTimeBucket | None",
                self.bucket,
            )
        object.__setattr__(self, "partition_by", tuple(self.partition_by))
        for index, column in enumerate(self.partition_by):
            if not isinstance(column, ColumnExpr):
                raise namespace_error(
                    "cross_section_group",
                    f"partition_by[{index}]",
                    "ColumnExpr",
                    column,
                )


def rows(size: int, /) -> RowFrame:
    """Declare a row-count rolling frame."""

    require_positive_int(size, "calc_flow.symbolic.rows.size")
    return RowFrame(size)


def duration(micros: int, /) -> DurationFrame:
    """Declare a duration rolling frame in exact microseconds."""

    require_positive_int(micros, "calc_flow.symbolic.duration.micros")
    return DurationFrame(micros)


def exact_time(
    event_time: ColumnExpr,
    /,
    *,
    partition_by: Sequence[ColumnExpr] = (),
) -> CrossSectionGroup:
    """Group by one exact event-time value plus the ordered partition key."""

    if not isinstance(event_time, ColumnExpr):
        raise namespace_error("exact_time", "event_time", "ColumnExpr", event_time)
    columns = tuple(partition_by)
    for index, column in enumerate(columns):
        if not isinstance(column, ColumnExpr):
            raise namespace_error(
                "exact_time", f"partition_by[{index}]", "ColumnExpr", column
            )
    return CrossSectionGroup(event_time, None, columns)


def event_time_bucket(
    event_time: ColumnExpr,
    /,
    *,
    width_micros: int,
    partition_by: Sequence[ColumnExpr] = (),
) -> CrossSectionGroup:
    """Group by fixed UTC buckets of exact positive width."""

    if not isinstance(event_time, ColumnExpr):
        raise namespace_error(
            "event_time_bucket", "event_time", "ColumnExpr", event_time
        )
    require_positive_int(
        width_micros, "calc_flow.symbolic.event_time_bucket.width_micros"
    )
    columns = tuple(partition_by)
    for index, column in enumerate(columns):
        if not isinstance(column, ColumnExpr):
            raise namespace_error(
                "event_time_bucket",
                f"partition_by[{index}]",
                "ColumnExpr",
                column,
            )
    bucket = EventTimeBucket(event_time, width_micros, columns)
    return CrossSectionGroup(event_time, bucket, columns)
