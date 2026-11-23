from __future__ import annotations

import unittest
from unittest.mock import Mock


class DurationPublicPhasesTests(unittest.TestCase):
    def test_scoped_hooks_restore_original_and_count_sync_calls(self):
        from scripts.duration_public_phases import Phases

        target = Mock()
        target.call = lambda value: value + 2
        original = target.call
        phases = Phases()
        with phases:
            phases.wrap(target, "call", "validation")
            self.assertEqual(target.call(4), 6)
        self.assertIs(target.call, original)
        self.assertEqual(phases.calls["validation"], 1)
        self.assertGreater(phases.nanoseconds["validation"], 0)

    def test_public_async_iterable_records_actual_python_sink_and_full_clock(self):
        import asyncio
        from datetime import UTC, datetime

        import pyarrow as pa

        from benchmarks.public_duration_sum import INPUT, expected, validate
        from scripts.duration_public_phases import profiled

        table = pa.table(
            {
                "event_time": [datetime(2026, 1, 1, tzinfo=UTC)],
                "sequence": [0],
                "symbol": ["a"],
                "price": [2.0],
            },
            schema=INPUT,
        )
        elapsed, outputs, details = asyncio.run(profiled([table]))
        self.assertEqual(validate(outputs, expected(table)), [1, 0])
        self.assertEqual(elapsed, sum(details["disjoint_ns"].values()))
        self.assertEqual(details["calls"]["source_import"], 1)
        self.assertEqual(details["calls"]["sink_callback_inclusive"], len(outputs))
