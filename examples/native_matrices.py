"""Packed matrix arithmetic and row reductions through the native CPU core."""

from __future__ import annotations

import pyarrow as pa

import calc_flow.array as cfa


def main() -> None:
    left = cfa.from_arrow(pa.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), shape=(2, 3))
    right = cfa.from_arrow(pa.array([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]), shape=(3, 2))
    product = cfa.matmul(left, right)
    means = cfa.mean(product, axis=1)
    transposed = cfa.transpose(product)
    flat = cfa.reshape(product, (4,))
    _validate_product(product)
    _validate_outputs(means, transposed, flat)
    print(f"Matrix product: {product.to_arrow().to_pylist()}")
    print(f"Row means: {means.to_arrow().to_pylist()}")


def _validate_product(product: cfa.NativeArray) -> None:
    if product.shape != (2, 2):
        raise AssertionError(f"unexpected product shape: {product.shape}")
    if product.element_count != 4:
        raise AssertionError(
            f"unexpected product element count: {product.element_count}"
        )
    if len(product) != 2:
        raise AssertionError(f"unexpected product row count: {len(product)}")


def _validate_outputs(
    means: cfa.NativeArray, transposed: cfa.NativeArray, flat: cfa.NativeArray
) -> None:
    if transposed.to_arrow().to_pylist() != [4.0, 10.0, 5.0, 11.0]:
        raise AssertionError("unexpected transposed values")
    if flat.shape != (4,):
        raise AssertionError(f"unexpected flattened shape: {flat.shape}")
    if flat.to_arrow().to_pylist() != [4.0, 5.0, 10.0, 11.0]:
        raise AssertionError("unexpected flattened values")
    if means.to_arrow().to_pylist() != [4.5, 10.5]:
        raise AssertionError("unexpected row means")


if __name__ == "__main__":
    main()
