from __future__ import annotations

import unittest


class PublicSessionStudyTest(unittest.TestCase):
    def test_requires_exact_build_checkout_and_all_steady_pushes(self):
        from scripts.hosted_steady_session import verify_worker
        from scripts.sparse_changelog_wheels import product_tree

        manifest = {
            "requested_sha": "workflow",
            "artifact": 123,
            "checkout": "product",
            "tree": "tree",
        }
        self.assertEqual(
            product_tree({"candidate": manifest}, "workflow", 123, "product", "tree"),
            "tree",
        )
        with self.assertRaises(ValueError):
            product_tree({"candidate": manifest}, "workflow", 123, "workflow", "tree")
        obs = {
            "append_ns": [1] * 64,
            "validated_rows": 4160,
            "finish_rows": 0,
            "gc_enabled": True,
            "environment": {
                "cpu_affinity": [1],
                "native_sha256": "native",
                "tokio_worker_threads": "1",
            },
        }
        verify_worker(obs, 1, "native")
        for key, value in [
            ("append_ns", [1] * 63),
            ("validated_rows", 4159),
            ("gc_enabled", False),
            ("finish_rows", 1),
        ]:
            with self.assertRaises(ValueError):
                verify_worker({**obs, key: value}, 1, "native")

    def test_toolchain_seal_uses_the_installed_artifact(self):
        from scripts.hosted_steady_session import verify_compiler

        compiler = ["rustc version 1.94.0 (hash)", "Linker: LLD 21", "GCC: 15.2"]
        raw = "\n".join("[  42]  " + line for line in compiler)
        verify_compiler(raw, compiler)
        with self.assertRaises(ValueError):
            verify_compiler(raw.replace("1.94.0", "1.95.0"), compiler)

    def test_rejects_a_missing_or_noncontiguous_probed_push(self):
        from scripts.hosted_steady_session import verify_phases

        valid = {
            "mode": "probed",
            "append_ns": [10] * 64,
            "phases_ns": [[1, 7, 2]] * 64,
            "calibration_ns": 0.5,
        }
        verify_phases(valid, "probed")
        for wrong in ([], [[1, 7, 3]] * 64, [[1, 9]] * 64, [[0, 9, 1]] * 64):
            with self.assertRaises(ValueError):
                verify_phases({**valid, "phases_ns": wrong}, "probed")
        with self.assertRaises(ValueError):
            verify_phases(valid, "plain")
