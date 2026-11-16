"""Independent exact-arithmetic acceptance tests for the rolling numeric contract.

Finite binary64 values are exact ``Fraction`` inputs. The oracle replays window
membership and rebuild cadence independently of the production accumulator.
Error bounds use exact rational arithmetic, with scaled ``isqrt`` enclosures for
irrational values.
"""

from __future__ import annotations

import math
import random
import struct
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from fractions import Fraction

import pyarrow as pa

from calc_flow import Batch, Runtime, SourceProvidedWatermarks
from calc_flow.symbolic import (
    FeatureSet,
    Field,
    Program,
    duration,
    rows,
    table_input,
    ts,
)

BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)
U = Fraction(1, 1 << 53)
ETA = Fraction(1, 1 << 1074)
DOMAIN = Fraction(2) ** 1000


def _gamma(count: int) -> Fraction:
    return count * U / (1 - count * U)


def _sqrt(value: Fraction, *, upper: bool) -> Fraction:
    if value < 0:
        raise ValueError("square root of a negative exact value")
    if value == 0:
        return Fraction(0)
    scale = 1 << 128
    root = math.isqrt((value.numerator * scale * scale) // value.denominator)
    return Fraction(root + int(upper), scale)


def _interval_error(value: float, lower: Fraction, upper: Fraction) -> Fraction:
    exact = Fraction(value)
    return max(abs(exact - lower), abs(exact - upper))


def _bits(value: float | int | None) -> int | None:
    if value is None or isinstance(value, int):
        return value
    return struct.unpack(">Q", struct.pack(">d", value))[0]


def _valid(value: float | None) -> bool:
    return value is not None and not math.isnan(value)


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


def _sum_abs(values: list[float]) -> Fraction:
    return sum((abs(Fraction(value)) for value in values), Fraction(0))


def _cadence(rows: int | None, n_prev: int) -> int:
    return max(rows, 64) if rows is not None else max(n_prev, 64)


def _finite_pair(value: tuple[float | None, float | None] | None) -> bool:
    return value is not None and _finite(value[0]) and _finite(value[1])


@dataclass
class SeriesOracle:
    rows: int | None = None
    duration_us: int | None = None
    history: list[float | None] = field(default_factory=list)
    times: list[int] = field(default_factory=list)
    since: int = 0
    n_prev: int = 64
    shift: float | None = None
    a_sum: Fraction = Fraction(0)
    a_d: Fraction = Fraction(0)
    a_q: Fraction = Fraction(0)
    operands: int = 0
    integer: bool = False

    def __init__(
        self,
        window: int | None = None,
        duration_us: int | None = None,
        integer: bool = False,
    ) -> None:
        self.rows = window
        self.duration_us = duration_us
        self.integer = integer
        self.history = []
        self.times = []

    def _reset(self) -> None:
        self.shift = None
        self.since = 0
        self.n_prev = 64
        self.a_sum = self.a_d = self.a_q = Fraction(0)
        self.operands = 0

    def _accumulate(self, values: list[float]) -> None:
        offsets = [] if self.shift is None else [value - self.shift for value in values]
        self.a_sum += _sum_abs(values)
        self.a_d += _sum_abs(offsets)
        self.a_q += _sum_abs([offset * offset for offset in offsets])
        self.operands += len(values)

    def _evictions(self, timestamp: int) -> list[float]:
        leaving = []
        if self.rows is not None and len(self.history) == self.rows:
            leaving.append(self.history.pop(0))
            self.times.pop(0)
        while (
            self.duration_us is not None
            and self.times
            and timestamp - self.times[0] >= self.duration_us
        ):
            leaving.append(self.history.pop(0))
            self.times.pop(0)
        return [value for value in leaving if _finite(value)]

    def _shift_for(self, value: float) -> float:
        return float(value) if self.integer else value

    def _rebuild(self) -> None:
        finite = [item for item in self.history if _finite(item)]
        self.shift = self._shift_for(finite[-1]) if finite else None
        self.since = 0
        self.a_sum = self.a_d = self.a_q = Fraction(0)
        self.operands = 0
        self._accumulate(finite)
        self.n_prev = len(self.history)

    def advance(self, value: float | None, timestamp: int) -> None:
        leaving = self._evictions(timestamp)
        self._accumulate(leaving)
        if not any(_finite(item) for item in self.history):
            self._reset()
        self.history.append(value)
        self.times.append(timestamp)
        if _finite(value):
            if self.shift is None:
                self.shift = self._shift_for(value)
            self._accumulate([value])
        self.since += 1
        if self.since >= _cadence(self.rows, self.n_prev):
            self._rebuild()

    def _variance_values(self, valid: list[float]) -> list[Fraction]:
        return [
            Fraction(float(item)) if self.integer else Fraction(item) for item in valid
        ]

    def _qk(self, values: list[Fraction]) -> Fraction:
        if self.shift is None:
            return Fraction(0)
        return sum(
            ((value - Fraction(self.shift)) ** 2 for value in values), Fraction(0)
        )

    @staticmethod
    def _exact_variance(
        values: list[Fraction], mean: Fraction | None, n: int, ddof: int
    ) -> Fraction | None:
        return (
            sum(((value - mean) ** 2 for value in values), Fraction(0)) / (n - ddof)
            if n > ddof
            else None
        )

    def exact(self, ddof: int = 0) -> dict[str, Fraction | int | bool | None]:
        valid = [item for item in self.history if _valid(item)]
        finite = [Fraction(item) for item in valid if math.isfinite(item)]
        n = len(valid)
        nonfinite = len(finite) != n
        if nonfinite:
            return {"count": n, "nonfinite": True}
        total = sum(finite, Fraction(0))
        mean = total / n if n else None
        variance_values = self._variance_values(valid)
        variance_mean = sum(variance_values, Fraction(0)) / n if n else None
        qk = self._qk(variance_values)
        variance = self._exact_variance(variance_values, variance_mean, n, ddof)
        return {
            "count": n,
            "sum": total,
            "mean": mean,
            "variance": variance,
            "qk": qk,
            "nonfinite": False,
        }

    def _variance_bracket(self, qk: Fraction, count: int) -> Fraction:
        gamma = _gamma(self.operands) ** 2
        return (
            12 * U * qk
            + 3 * gamma * self.a_q
            + 5 * gamma * self.a_d * _sqrt(qk / count, upper=True)
            + 4 * self.operands * ETA
        )

    def bounds(self, ddof: int = 0) -> dict[str, Fraction]:
        exact = self.exact(ddof)
        if exact["nonfinite"]:
            return {}
        n = exact["count"]
        assert isinstance(n, int) and n > 0
        total = exact["sum"]
        mean = exact["mean"]
        qk = exact["qk"]
        assert (
            isinstance(total, Fraction)
            and isinstance(mean, Fraction)
            and isinstance(qk, Fraction)
        )
        gamma = _gamma(self.operands) ** 2
        result = {
            "sum": U * abs(total) + gamma * self.a_sum,
            "mean": (
                3 * U * abs(mean)
                if self.integer
                else 3 * U * abs(mean) + 3 * gamma * self.a_sum / n + ETA
            ),
        }
        variance = exact["variance"]
        if variance is not None:
            assert isinstance(variance, Fraction)
            delta = self._variance_bracket(qk, n) / (n - ddof) + U * variance
            root_delta = _sqrt(delta, upper=True)
            if variance == 0:
                std_bound = root_delta * (1 + U)
            else:
                std_bound = min(
                    root_delta, delta / _sqrt(variance, upper=False)
                ) + U * (_sqrt(variance, upper=True) + root_delta)
            result["variance"] = delta
            result["stddev"] = std_bound
        return result

    def _in_domain(self, exact: dict) -> bool:
        qk = exact["qk"]
        return (
            isinstance(qk, Fraction)
            and max(self.a_sum, self.a_q, exact["count"] * qk) <= DOMAIN
        )

    @staticmethod
    def _check_nonfinite_moments(
        actual: dict[str, float | int | None], mixed: bool, expected: float
    ) -> None:
        for name in ("sum", "mean"):
            value = actual[name]
            if mixed:
                assert value is not None and math.isnan(value)
            else:
                assert value == expected

    @staticmethod
    def _check_nonfinite_spread(
        actual: dict[str, float | int | None], n: int, ddof: int
    ) -> None:
        for name in ("variance", "stddev"):
            value = actual[name]
            if n <= ddof:
                assert value is None
            else:
                assert value is not None and math.isnan(value)

    def _check_nonfinite(
        self, actual: dict[str, float | int | None], n: int, ddof: int
    ) -> None:
        positive = any(item == math.inf for item in self.history)
        negative = any(item == -math.inf for item in self.history)
        mixed = positive and negative
        expected = math.nan if mixed else math.inf if positive else -math.inf
        self._check_nonfinite_moments(actual, mixed, expected)
        self._check_nonfinite_spread(actual, n, ddof)

    @staticmethod
    def _check_finite_moments(
        actual: dict[str, float | int | None], exact: dict, bounds: dict[str, Fraction]
    ) -> None:
        for name in ("sum", "mean"):
            value = actual[name]
            assert value is not None and math.isfinite(value)
            reference = exact[name]
            assert isinstance(reference, Fraction)
            assert abs(Fraction(value) - reference) <= bounds[name], (
                name,
                value,
                reference,
            )

    @staticmethod
    def _spread_error(name: str, value: float, reference: Fraction) -> Fraction:
        if name == "variance":
            return abs(Fraction(value) - reference)
        return _interval_error(
            value, _sqrt(reference, upper=False), _sqrt(reference, upper=True)
        )

    @staticmethod
    def _check_spread_value(
        name: str,
        value: float | int | None,
        exact: dict,
        bounds: dict[str, Fraction],
        n: int,
        ddof: int,
    ) -> None:
        if n <= ddof:
            assert value is None
            return
        assert value is not None and math.isfinite(value)
        reference = exact["variance" if name == "stddev" else name]
        assert isinstance(reference, Fraction)
        error = SeriesOracle._spread_error(name, value, reference)
        assert error <= bounds[name], (name, value, reference, bounds[name])

    def _check_spread(
        self,
        actual: dict[str, float | int | None],
        exact: dict,
        bounds: dict[str, Fraction],
        n: int,
        ddof: int,
    ) -> None:
        for name in ("variance", "stddev"):
            self._check_spread_value(name, actual[name], exact, bounds, n, ddof)

    def _check_finite(
        self, actual: dict[str, float | int | None], exact: dict, n: int, ddof: int
    ) -> None:
        bounds = self.bounds(ddof)
        self._check_finite_moments(actual, exact, bounds)
        self._check_spread(actual, exact, bounds, n, ddof)

    def check(self, actual: dict[str, float | int | None], ddof: int = 0) -> None:
        exact = self.exact(ddof)
        n = exact["count"]
        assert isinstance(n, int)
        if exact["nonfinite"]:
            self._check_nonfinite(actual, n, ddof)
            return
        if n == 0:
            assert all(value is None for value in actual.values())
            return
        if not self._in_domain(exact):
            return
        self._check_finite(actual, exact, n, ddof)


@dataclass
class PairOracle:
    rows: int | None = None
    duration_us: int | None = None
    history: list[tuple[float, float] | None] = field(default_factory=list)
    times: list[int] = field(default_factory=list)
    since: int = 0
    n_prev: int = 64
    shifts: tuple[float, float] | None = None
    changed_left: bool = False
    changed_right: bool = False
    a_p: Fraction = Fraction(0)
    a_dx: Fraction = Fraction(0)
    a_dy: Fraction = Fraction(0)
    a_qx: Fraction = Fraction(0)
    a_qy: Fraction = Fraction(0)
    operands: int = 0
    integer: bool = False

    def __init__(
        self,
        window: int | None = None,
        duration_us: int | None = None,
        integer: bool = False,
    ) -> None:
        self.rows = window
        self.duration_us = duration_us
        self.integer = integer
        self.history = []
        self.times = []

    def _reset_error_terms(self) -> None:
        self.changed_left = self.changed_right = False
        self.since = 0
        self.a_p = self.a_dx = self.a_dy = Fraction(0)
        self.a_qx = self.a_qy = Fraction(0)
        self.operands = 0

    def _reset(self) -> None:
        self.shifts = None
        self.n_prev = 64
        self._reset_error_terms()

    def _accumulate(self, values: list[tuple[float, float]]) -> None:
        if self.shifts is None:
            return
        kx, ky = self.shifts
        for x, y in values:
            dx, dy = x - kx, y - ky
            self.a_dx += abs(Fraction(dx))
            self.a_dy += abs(Fraction(dy))
            self.a_p += abs(Fraction(dx * dy))
            self.a_qx += abs(Fraction(dx * dx))
            self.a_qy += abs(Fraction(dy * dy))
            self.operands += 1
            self.changed_left |= dx != 0
            self.changed_right |= dy != 0

    def _evictions(self, timestamp: int) -> list[tuple[float, float]]:
        leaving = []
        if self.rows is not None and len(self.history) == self.rows:
            leaving.append(self.history.pop(0))
            self.times.pop(0)
        while (
            self.duration_us is not None
            and self.times
            and timestamp - self.times[0] >= self.duration_us
        ):
            leaving.append(self.history.pop(0))
            self.times.pop(0)
        return [value for value in leaving if _finite_pair(value)]

    def _shift_for(self, value: tuple[float, float]) -> tuple[float, float]:
        return (float(value[0]), float(value[1])) if self.integer else value

    def _rebuild(self) -> None:
        finite = [item for item in self.history if _finite_pair(item)]
        self.shifts = self._shift_for(finite[-1]) if finite else None
        self._reset_error_terms()
        self._accumulate(finite)
        self.n_prev = len(self.history)

    def advance(
        self, value: tuple[float | None, float | None] | None, timestamp: int
    ) -> None:
        leaving = self._evictions(timestamp)
        self._accumulate(leaving)
        if not any(_finite_pair(item) for item in self.history):
            self._reset()
        self.history.append(value)
        self.times.append(timestamp)
        if _finite_pair(value):
            if self.shifts is None:
                self.shifts = self._shift_for(value)
                self.changed_left = self.changed_right = False
            self._accumulate([value])
        self.since += 1
        if self.since >= _cadence(self.rows, self.n_prev):
            self._rebuild()

    def _exact_values(
        self, valid: list[tuple[float, float]]
    ) -> list[tuple[Fraction, Fraction]]:
        return [
            (Fraction(float(x)), Fraction(float(y)))
            if self.integer
            else (Fraction(x), Fraction(y))
            for x, y in valid
        ]

    @staticmethod
    def _correlation(
        covariance: Fraction, vx: Fraction, vy: Fraction
    ) -> Fraction | None:
        if vx == 0 or vy == 0:
            return None
        return covariance / _sqrt(vx * vy, upper=False)

    @staticmethod
    def _moments(
        values: list[tuple[Fraction, Fraction]], ddof: int
    ) -> tuple[Fraction, Fraction, Fraction]:
        n = len(values)
        mx = sum((x for x, _ in values), Fraction(0)) / n
        my = sum((y for _, y in values), Fraction(0)) / n
        vx = sum(((x - mx) ** 2 for x, _ in values), Fraction(0)) / n
        vy = sum(((y - my) ** 2 for _, y in values), Fraction(0)) / n
        covariance = sum(((x - mx) * (y - my) for x, y in values), Fraction(0)) / (
            n - ddof
        )
        return vx, vy, covariance

    def exact(self, ddof: int = 0) -> dict[str, Fraction | int | bool | None]:
        valid = [item for item in self.history if item is not None]
        n = len(valid)
        if any(not math.isfinite(x) or not math.isfinite(y) for x, y in valid):
            return {"count": n, "nonfinite": True}
        values = self._exact_values(valid)
        if n <= ddof:
            return {
                "count": n,
                "nonfinite": False,
                "covariance": None,
                "correlation": None,
            }
        vx, vy, covariance = self._moments(values, ddof)
        correlation = self._correlation(covariance, vx, vy)
        return {
            "count": n,
            "nonfinite": False,
            "covariance": covariance,
            "correlation": correlation,
            "vx": vx,
            "vy": vy,
        }

    def _bounds(
        self, exact: dict, ddof: int
    ) -> tuple[Fraction, Fraction, Fraction, Fraction]:
        n = exact["count"]
        assert isinstance(n, int)
        vx, vy = exact["vx"], exact["vy"]
        assert isinstance(vx, Fraction) and isinstance(vy, Fraction)
        if self.shifts is None:
            qx = qy = Fraction(0)
        else:
            finite = [item for item in self.history if item is not None]
            kx, ky = Fraction(self.shifts[0]), Fraction(self.shifts[1])
            qx = sum(((Fraction(x) - kx) ** 2 for x, _ in finite), Fraction(0))
            qy = sum(((Fraction(y) - ky) ** 2 for _, y in finite), Fraction(0))
        gamma = _gamma(self.operands) ** 2
        variance_x = (
            12 * U * qx
            + 3 * gamma * self.a_qx
            + 5 * gamma * self.a_dx * _sqrt(qx / n, upper=True)
            + 4 * self.operands * ETA
        )
        variance_y = (
            12 * U * qy
            + 3 * gamma * self.a_qy
            + 5 * gamma * self.a_dy * _sqrt(qy / n, upper=True)
            + 4 * self.operands * ETA
        )
        bracket = (
            12 * U * _sqrt(qx * qy, upper=True)
            + 3 * gamma * self.a_p
            + 5
            * gamma
            * (
                self.a_dx * _sqrt(qy / n, upper=True)
                + self.a_dy * _sqrt(qx / n, upper=True)
            )
            + 4 * self.operands * ETA
        )
        return bracket, bracket / (n - ddof), variance_x / n, variance_y / n

    @staticmethod
    def _check_nonfinite(
        covariance: float | None,
        correlation: float | None,
        n: int,
        ddof: int,
    ) -> None:
        if n <= ddof:
            assert covariance is None and correlation is None
        else:
            assert covariance is not None and math.isnan(covariance)
            assert correlation is not None and math.isnan(correlation)

    @staticmethod
    def _assert_correlation_present(
        correlation: float | None,
        vx: Fraction,
        vy: Fraction,
        delta_x: Fraction,
        delta_y: Fraction,
    ) -> None:
        if delta_x < vx and delta_y < vy and correlation is None:
            raise AssertionError(
                "correlation must be finite when both error bounds are smaller"
            )

    @staticmethod
    def _skip_correlation(
        correlation: float | None,
        vx: Fraction,
        vy: Fraction,
        delta_x: Fraction,
        delta_y: Fraction,
    ) -> bool:
        return correlation is None or delta_x > vx / 2 or delta_y > vy / 2

    @staticmethod
    def _assert_correlation(
        correlation: float,
        covariance: Fraction,
        vx: Fraction,
        vy: Fraction,
        bracket: Fraction,
        delta_x: Fraction,
        delta_y: Fraction,
    ) -> None:
        product = vx * vy
        lower = (
            covariance / _sqrt(product, upper=True)
            if covariance >= 0
            else covariance / _sqrt(product, upper=False)
        )
        upper = (
            covariance / _sqrt(product, upper=False)
            if covariance >= 0
            else covariance / _sqrt(product, upper=True)
        )
        if lower > upper:
            lower, upper = upper, lower
        r_up = max(abs(lower), abs(upper))
        bound = (
            2 * (1 + _gamma(4)) * bracket / _sqrt(product, upper=False)
            + 2 * r_up * (delta_x / vx + delta_y / vy)
            + 6 * U * r_up
        )
        assert _interval_error(correlation, lower, upper) <= bound, (
            correlation,
            covariance,
            float(bound),
            float(_interval_error(correlation, lower, upper)),
            bound,
        )

    def _check_correlation(
        self,
        correlation: float | None,
        exact: dict,
        bracket: Fraction,
        delta_x: Fraction,
        delta_y: Fraction,
    ) -> None:
        vx, vy = exact["vx"], exact["vy"]
        assert isinstance(vx, Fraction) and isinstance(vy, Fraction)
        if vx == 0 or vy == 0:
            assert correlation is None
            return
        self._assert_correlation_present(correlation, vx, vy, delta_x, delta_y)
        if self._skip_correlation(correlation, vx, vy, delta_x, delta_y):
            return
        covariance = exact["covariance"]
        assert isinstance(covariance, Fraction)
        self._assert_correlation(
            correlation, covariance, vx, vy, bracket, delta_x, delta_y
        )

    def _check_finite(
        self,
        covariance: float | None,
        correlation: float | None,
        exact: dict,
        ddof: int,
    ) -> None:
        assert covariance is not None and math.isfinite(covariance)
        reference = exact["covariance"]
        assert isinstance(reference, Fraction)
        bracket, delta, delta_x, delta_y = self._bounds(exact, ddof)
        assert abs(Fraction(covariance) - reference) <= delta + U * abs(reference)
        if not self.changed_left or not self.changed_right:
            assert correlation is None
            return
        self._check_correlation(correlation, exact, bracket, delta_x, delta_y)

    def check(
        self, covariance: float | None, correlation: float | None, ddof: int = 0
    ) -> None:
        exact = self.exact(ddof)
        n = exact["count"]
        assert isinstance(n, int)
        if exact["nonfinite"]:
            self._check_nonfinite(covariance, correlation, n, ddof)
            return
        if n <= ddof:
            assert covariance is None and correlation is None
            return
        self._check_finite(covariance, correlation, exact, ddof)


def _program(window, *, integer: bool = False, unsigned: bool = False) -> Program:
    value_type = "uint64" if unsigned else "int64" if integer else "float64"
    quotes = table_input(
        "quotes",
        schema=(
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", value_type),
            Field("y", value_type),
        ),
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("seq",),
    )
    features = (
        ("sum", ts.sum(quotes["x"], window=window)),
        ("mean", ts.mean(quotes["x"], window=window)),
        ("variance", ts.variance(quotes["x"], window=window, ddof=0)),
        ("stddev", ts.stddev(quotes["x"], window=window, ddof=0)),
        ("covariance", ts.covariance(quotes["x"], quotes["y"], window=window, ddof=0)),
        (
            "correlation",
            ts.correlation(quotes["x"], quotes["y"], window=window, ddof=0),
        ),
    )
    output = quotes.with_columns(FeatureSet(features)).select(
        *(name for name, _ in features)
    )
    return Program(
        "numeric-oracle",
        engine="streaming",
        inputs=(quotes,),
        outputs=(("output", output),),
    )


def _table(
    values: list[tuple[float | int | None, float | int | None]],
    timestamps: list[int],
    integer: bool,
    unsigned: bool = False,
) -> pa.Table:
    value_type = pa.uint64() if unsigned else pa.int64() if integer else pa.float64()
    return pa.table(
        {
            "ts": [BASE + timedelta(microseconds=value) for value in timestamps],
            "symbol": ["a"] * len(values),
            "seq": list(range(len(values))),
            "x": [value[0] for value in values],
            "y": [value[1] for value in values],
        },
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("x", value_type),
                pa.field("y", value_type),
            ]
        ),
    )


def _execute(
    program: Program,
    values: list[tuple[float | int | None, float | int | None]],
    timestamps: list[int],
    integer: bool = False,
    unsigned: bool = False,
) -> dict[str, list[float | int | None]]:
    table = _table(values, timestamps, integer, unsigned)
    batch_program = Program(
        program.name, engine="sql", inputs=program.inputs, outputs=program.outputs
    )
    batch = (
        batch_program.compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(table)})
        .outputs["output"]
        .to_pyarrow()
    )
    with program.session(watermarks=SourceProvidedWatermarks()) as session:
        streamed = []
        for start in range(0, len(values), 7):
            end = min(start + 7, len(values))
            chunk = table.slice(start, end - start)
            streamed.append(
                session.push(
                    {"quotes": chunk},
                    watermark=BASE + timedelta(microseconds=timestamps[end - 1]),
                )["output"]
            )
        streamed.append(session.finish()["output"])
    streamed_table = pa.concat_tables(streamed)
    names = batch.column_names
    for name in names:
        assert [_bits(value) for value in batch[name].to_pylist()] == [
            _bits(value) for value in streamed_table[name].to_pylist()
        ], name
    return {name: batch[name].to_pylist() for name in names}


def _pair_value(
    x: float | int | None, y: float | int | None
) -> tuple[float | int | None, float | int | None] | None:
    return (x, y) if _valid(x) and _valid(y) else None


def _check_values(
    values: list[tuple[float | int | None, float | int | None]],
    window: int | None = None,
    duration_us: int | None = None,
    integer: bool = False,
    timestamps: list[int] | None = None,
    unsigned: bool = False,
) -> None:
    timestamps = timestamps or list(range(len(values)))
    actual = _execute(
        _program(
            rows(window) if window is not None else duration(duration_us),
            integer=integer,
            unsigned=unsigned,
        ),
        values,
        timestamps,
        integer,
        unsigned,
    )
    exact_integer = integer or unsigned
    series = SeriesOracle(window, duration_us, integer=exact_integer)
    pair = PairOracle(window, duration_us, integer=exact_integer)
    for index, (x, y) in enumerate(values):
        series.advance(x, timestamps[index])
        pair.advance(_pair_value(x, y), timestamps[index])
        series.check(
            {
                name: actual[name][index]
                for name in ("sum", "mean", "variance", "stddev")
            }
        )
        pair.check(actual["covariance"][index], actual["correlation"][index])


def test_exact_oracle_rebuilds_cancellation_and_batch_split_identity() -> None:
    rng = random.Random(20261116)
    close = [
        (1e9 + rng.gauss(0, 1e-3), 1e9 + rng.gauss(0, 1e-3)) for _ in range(10_000)
    ]
    _check_values(close, 20)
    alternating = [
        (((-1) ** index) * 1e16, 1.0 if index % 3 else float(index))
        for index in range(1_000)
    ]
    _check_values(alternating, 20)
    rebuilds = [(float(index % 17), float(index % 29)) for index in range(10_000)]
    _check_values(rebuilds, 5)


def test_exact_oracle_extremes_nulls_infinities_and_constants() -> None:
    extremes = [2.0**53 - 1, 2.0**53 + 1, float(2**63 - 1), -float(2**63), 1.0]
    _check_values([(value, value / 2) for value in extremes], 2)
    special = [
        (None, 1.0),
        (math.nan, 2.0),
        (math.inf, 3.0),
        (4.0, -math.inf),
        (-5.0, 6.0),
        (7.0, 8.0),
    ] * 20
    _check_values(special, 5)
    _check_values([(3.0, 7.0)] * (20 - 1 + 64 + 3), 20)
    _check_values(
        [
            (2**53 - 1, 2),
            (2**53 + 1, 3),
            (-(2**53 - 1), 4),
            (2**63 - 1, 5),
            (-(2**63 - 1), 6),
            (1, 7),
            (-(2**63) + 1, 8),
        ],
        2,
        integer=True,
    )
    _check_values(
        [(2**64 - 1, 2), (0, 3), (1, 4), (0, 5), (2**64 - 1, 6), (0, 7)],
        2,
        unsigned=True,
    )


def test_duration_frame_matches_the_same_exact_oracle() -> None:
    values = [(float(index % 9), float(index % 5)) for index in range(500)]
    timestamps = [index * 7_000 for index in range(500)]
    _check_values(values, duration_us=10_000, timestamps=timestamps)


def test_datafusion_avg_only_rewrite_keeps_sticky_nan_policy() -> None:
    quotes = table_input(
        "quotes",
        schema=(
            Field("ts", "timestamp[us, UTC]", nullable=False),
            Field("symbol", "string", nullable=False),
            Field("seq", "uint64", nullable=False),
            Field("x", "float64"),
        ),
        entity_by=("symbol",),
        event_time="ts",
        sequence_by=("seq",),
    )
    output = quotes.sql(
        "SELECT AVG(x) OVER w AS mean FROM input "
        "WINDOW w AS (PARTITION BY symbol ORDER BY ts, seq "
        "ROWS BETWEEN 1 PRECEDING AND CURRENT ROW)"
    )
    program = Program(
        "sticky-nan", engine="sql", inputs=(quotes,), outputs=(("output", output),)
    )
    table = pa.table(
        {
            "ts": [BASE + timedelta(microseconds=index) for index in range(4)],
            "symbol": ["a"] * 4,
            "seq": [0, 1, 2, 3],
            "x": [1.0, math.nan, 3.0, 5.0],
        },
        schema=pa.schema(
            [
                pa.field("ts", pa.timestamp("us", tz="UTC"), nullable=False),
                pa.field("symbol", pa.string(), nullable=False),
                pa.field("seq", pa.uint64(), nullable=False),
                pa.field("x", pa.float64()),
            ]
        ),
    )
    result = (
        program.compile_batch(Runtime())
        .execute({"input": Batch.from_pyarrow(table)})
        .outputs["output"]
        .to_pyarrow()
    )
    values = result["mean"].to_pylist()
    assert values[0] == 1.0
    assert all(math.isnan(value) for value in values[1:])
