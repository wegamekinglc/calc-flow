"""Calculate a moving mean and conditional result using Arrow and native Rust."""

from __future__ import annotations

import pyarrow as pa

import calc_flow.array as cfa


def main() -> None:
    values = cfa.from_arrow(pa.array([2.0, None, 4.0, 8.0]))
    average = cfa.rolling_mean(values, window=2)
    threshold = cfa.from_arrow(pa.array([4.0]), shape=())
    zero = cfa.from_arrow(pa.array([0.0]), shape=())
    result = cfa.where_(cfa.greater_equal(average, threshold), average, zero)
    exported = result.to_arrow()
    total = cfa.sum(result).to_arrow()[0].as_py()
    assert exported.to_pylist() == [0.0, 0.0, 4.0, 6.0]
    assert total == 10.0
    print(f"Native values: {exported.to_pylist()}")
    print(f"Native sum: {total}")


if __name__ == "__main__":
    main()
