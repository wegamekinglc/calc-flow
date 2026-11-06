"""Repeat the fixed sign CASE comparison against the current main revision."""

from __future__ import annotations

from dataclasses import replace

from scripts.benchmark_suite import sum_case_projection as original
from scripts.benchmark_suite import sum_columnar_keys as shared

SLUG = "sum-case-retest-20261106"
BASELINE = "5a66b6d5cb3f9fa2f6c5d6c8d7ce03db0cea565f"
CONTRACT = replace(
    original.CONTRACT,
    slug=SLUG,
    baseline=BASELINE,
    tools=(*original.CONTRACT.tools, "scripts/benchmark_suite/sum_case_retest.py"),
    report_lines=tuple(
        line.replace("publicdc8393d2", "public5a66b6d5")
        for line in original.CONTRACT.report_lines
    ),
)


def validate_protocol(protocol):
    return shared.validate_protocol(protocol, CONTRACT)


def main():
    shared.main(contract=CONTRACT)


if __name__ == "__main__":
    main()
