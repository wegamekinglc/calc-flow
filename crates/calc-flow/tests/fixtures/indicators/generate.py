"""Regenerate the fixed RSI/ATR IEEE and qualified Fraction reference fixtures."""

from __future__ import annotations

import hashlib
import json
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

PERIODS = (1, 2, 3, 7, 32)
PRICE = tuple(
    ((index * index * 17 + 13 * index + 7) % 127 - 63) / 16 for index in range(64)
)
HIGH = tuple(price + 0.5 for price in PRICE)
LOW = tuple(price - 0.75 for price in PRICE)
CLOSE = tuple(price + 0.125 for price in PRICE)


def ieee_mean(previous: float, sample: float, count: int) -> float:
    if count == 1:
        return 0.0 if sample == 0.0 else sample
    difference = sample - previous
    increment = difference / float(count)
    value = previous + increment
    return 0.0 if value == 0.0 else value


def ieee_strength(gain: float, loss: float) -> float:
    if gain == 0.0 and loss == 0.0:
        return 50.0
    if loss == 0.0:
        return 100.0
    if gain == 0.0:
        return 0.0
    if gain >= loss:
        ratio = loss / gain
        return 100.0 / (1.0 + ratio)
    ratio = gain / loss
    return 100.0 * (ratio / (1.0 + ratio))


def ieee_outputs(period: int) -> tuple[list[float | None], list[float | None]]:
    rsi: list[float | None] = [None]
    atr: list[float | None] = [None]
    gain = loss = true_range = 0.0
    for index in range(1, len(PRICE)):
        count = min(index, period)
        delta = PRICE[index] - PRICE[index - 1]
        gain = ieee_mean(gain, max(delta, 0.0), count)
        loss = ieee_mean(loss, max(-delta, 0.0), count)
        span = HIGH[index] - LOW[index]
        high_gap = HIGH[index] - CLOSE[index - 1]
        low_gap = LOW[index] - CLOSE[index - 1]
        true_range = ieee_mean(
            true_range, max(span, abs(high_gap), abs(low_gap)), count
        )
        ready = index >= period
        rsi.append(ieee_strength(gain, loss) if ready else None)
        atr.append(true_range if ready else None)
    return rsi, atr


def assert_mean_domain(value: Fraction) -> None:
    assert isinstance(value, Fraction)
    assert value == 0 or Fraction(1, 2**20) <= value <= 2**13


def fraction_outputs(period: int) -> tuple[list[float | None], list[float | None]]:
    rsi: list[float | None] = [None]
    atr: list[float | None] = [None]
    gains: list[Fraction] = []
    losses: list[Fraction] = []
    ranges: list[Fraction] = []
    gain = loss = true_range = Fraction(0)
    for index in range(1, len(PRICE)):
        delta = Fraction(PRICE[index]) - Fraction(PRICE[index - 1])
        gains.append(max(delta, Fraction(0)))
        losses.append(max(-delta, Fraction(0)))
        ranges.append(
            max(
                Fraction(HIGH[index]) - Fraction(LOW[index]),
                abs(Fraction(HIGH[index]) - Fraction(CLOSE[index - 1])),
                abs(Fraction(LOW[index]) - Fraction(CLOSE[index - 1])),
            )
        )
        if index <= period:
            gain = sum(gains, Fraction(0)) / index
            loss = sum(losses, Fraction(0)) / index
            true_range = sum(ranges, Fraction(0)) / index
        else:
            gain = (gain * (period - 1) + gains[-1]) / period
            loss = (loss * (period - 1) + losses[-1]) / period
            true_range = (true_range * (period - 1) + ranges[-1]) / period
        for mean in (gain, loss, true_range):
            assert_mean_domain(mean)
        ready = index >= period
        strength = Fraction(50) if gain == loss == 0 else 100 * gain / (gain + loss)
        rsi.append(float(strength) if ready else None)
        atr.append(float(true_range) if ready else None)
    return rsi, atr


def validate_fixed_inputs() -> None:
    assert all(1 <= period <= 32 for period in PERIODS)
    assert len(PRICE) - 1 <= 64
    for values in (PRICE, HIGH, LOW, CLOSE):
        for value in values:
            assert -4096 <= value <= 4096
            assert (Fraction(value) * 16).denominator == 1
    assert all(high >= low for high, low in zip(HIGH, LOW))


def float_literal(value: float) -> str:
    mantissa, separator, exponent = repr(value).partition("e")
    sign = "-" if mantissa.startswith("-") else ""
    integer, decimal, fraction = mantissa.removeprefix("-").partition(".")
    tail = "_".join(fraction[index : index + 3] for index in range(0, len(fraction), 3))
    return f"{sign}{int(integer):_d}{decimal}{tail}" + (
        f"e{exponent}" if separator else ""
    )


def float_options(values: list[float | None]) -> str:
    return ", ".join(
        "None" if value is None else f"Some({float_literal(value)})" for value in values
    )


def bit_options(values: list[float | None]) -> str:
    def option(value: float | None) -> str:
        if value is None:
            return "None"
        bits = struct.unpack("<Q", struct.pack("<d", value))[0]
        return f"Some({bits:_d})"

    return ", ".join(option(value) for value in values)


def rust_fixture(period: int) -> str:
    rsi, atr = ieee_outputs(period)
    exact_rsi, exact_atr = fraction_outputs(period)
    return (
        f"Fixture {{ period: {period}, rsi: &[{bit_options(rsi)}], "
        f"atr: &[{bit_options(atr)}], fraction_rsi: &[{float_options(exact_rsi)}], "
        f"fraction_atr: &[{float_options(exact_atr)}] }},"
    )


def rust_source() -> str:
    lines = [
        "// Independent Python float64 operation-order and Fraction fixtures.",
        "// Fixed periods/data; the generator validates "
        "the approved well-scaled predicate.",
    ]
    for name, values in (
        ("PRICE", PRICE),
        ("HIGH", HIGH),
        ("LOW", LOW),
        ("CLOSE", CLOSE),
    ):
        literals = ", ".join(float_literal(value) for value in values)
        lines.append(f"pub const {name}: &[f64] = &[{literals}];")
    lines.extend(
        [
            "pub struct Fixture { pub period: u64, pub rsi: &'static [Option<u64>], "
            "pub atr: &'static [Option<u64>], "
            "pub fraction_rsi: &'static [Option<f64>], "
            "pub fraction_atr: &'static [Option<f64>] }",
            "pub const FIXTURES: &[Fixture] = &[",
            *(rust_fixture(period) for period in PERIODS),
            "];",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    validate_fixed_inputs()
    directory = Path(__file__).resolve().parent
    golden = directory / "golden.rs"
    golden.write_text(rust_source(), encoding="utf-8")
    subprocess.run(["rustfmt", "--edition", "2024", str(golden)], check=True)
    provenance = {
        "reference": (
            "independent standard-library Python float64 and Fraction; "
            "no production imports"
        ),
        "periods": PERIODS,
        "rows": len(PRICE),
        "updates_per_segment": len(PRICE) - 1,
        "fixed_input_formula": "((17*i*i+13*i+7)%127-63)/16, i=0..63",
        "atr_offsets": {"high": "price+1/2", "low": "price-3/4", "close": "price+1/8"},
        "fraction_predicate_verified": True,
        "input_grid_and_bounds": "multiples of 1/16 in [-4096,4096]; high>=low",
        "exact_positive_mean_bounds": ["2^-20", "2^13"],
        "fraction_seed": "sum(samples)/count; ready at period differences/TRs",
        "fraction_recurrence": "(previous*(period-1)+sample)/period",
        "fraction_rsi": "100*gain/(gain+loss), with explicit zero/flat rules",
        "accuracy": {"rsi_absolute": "100*256*2^-52", "atr_relative": "256*2^-52"},
        "oracle_correction": {
            "old_issue": (
                "zero-only integer sums divided by count became float0.0, "
                "contaminating later supposedly exact Fraction recurrence"
            ),
            "affected_references": "period2 fraction_rsi rows54 and62, one ULP each",
            "unchanged": "all input/primary IEEE bits and accuracy tolerances",
        },
        "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
    }
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
