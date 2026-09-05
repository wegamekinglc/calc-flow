from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

from scripts.benchmark_suite.worker import dispatch


class BenchmarkWorkerTests(unittest.TestCase):
    def setUp(self):
        engine = ModuleType("benchmarks.engine_comparison")
        self.factory = Mock()
        engine.EngineCase = self.factory
        self.modules = patch.dict("sys.modules", {engine.__name__: engine})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_unprepared_sample_and_finish_are_protocol_errors(self):
        for operation in ("sample", "finish"):
            with (
                self.subTest(operation=operation),
                self.assertRaisesRegex(ValueError, "no active benchmark"),
            ):
                dispatch({"operation": operation}, None, self.root)

    def test_prepare_sample_finish_releases_one_case(self):
        case = {"family": "engines"}
        active = self.factory.return_value
        active.sample.return_value = {"seconds": 1.0}
        response, prepared = dispatch(
            {"operation": "prepare", "case": case}, None, self.root
        )
        self.assertIs(prepared, active)
        self.assertEqual(response["case"], case)
        self.assertEqual(
            dispatch({"operation": "sample"}, active, self.root),
            ({"seconds": 1.0}, active),
        )
        self.assertEqual(
            dispatch({"operation": "finish"}, active, self.root),
            ({"state": "completed"}, None),
        )
        active.close.assert_called_once_with()

    def test_failed_warmup_closes_the_unpublished_case(self):
        active = self.factory.return_value
        active.sample.side_effect = ValueError("invalid output")
        with self.assertRaisesRegex(ValueError, "invalid output"):
            dispatch(
                {"operation": "prepare", "case": {"family": "engines"}},
                None,
                self.root,
            )
        active.close.assert_called_once_with()

    def test_prepare_does_not_replace_a_live_case(self):
        with self.assertRaisesRegex(ValueError, "already active"):
            dispatch({"operation": "prepare"}, Mock(), self.root)
        self.factory.assert_not_called()

    def test_unknown_request_is_a_protocol_error(self):
        with self.assertRaisesRegex(ValueError, "unknown worker request"):
            dispatch({"operation": "inject"}, None, self.root)
