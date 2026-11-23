from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path


class SourceProofTests(unittest.TestCase):
    def test_installed_session_adapter_must_match_the_registered_product_bytes(self):
        from scripts.hosted_session_record import verify_session_adapter

        with tempfile.TemporaryDirectory() as directory:
            site = Path(directory)
            package = site / "calc_flow"
            package.mkdir()
            content = b"input adapter"
            (package / "session.py").write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            self.assertEqual(verify_session_adapter(site, digest), digest)
            with self.assertRaisesRegex(ValueError, "session adapter"):
                verify_session_adapter(site, "0" * 64)
