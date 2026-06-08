import hashlib
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.protocol_v1_ephemeral_overlay_preflight import build_report


class TestProtocolV1EphemeralOverlayPreflight(unittest.TestCase):
    def _manifest(self, *, expected_sha: str):
        return {
            "snapshot_available": True,
            "source_git_ref": "HEAD",
            "source_git_commit": "abc",
            "tracked_baseline_file_count": 1,
            "files": [
                {
                    "path": "gl_gym/experiments/frozen_benchmark_protocol.py",
                    "sha256": expected_sha,
                    "object_available": True,
                }
            ],
        }

    def test_exports_manifest_file_and_validates_hash(self):
        data = b"VALUE = 1\n"
        expected_sha = hashlib.sha256(data).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(
                protocol_v1_snapshot=self._manifest(expected_sha=expected_sha),
                overlay_root=tmp,
                object_reader=lambda _ref, _path: data,
                validate_import=False,
            )

            exported = Path(tmp) / "gl_gym" / "experiments" / "frozen_benchmark_protocol.py"
            self.assertTrue(exported.exists())
            self.assertTrue(report["protocol_v1_overlay_validated"])
            self.assertTrue(report["protocol_v1_overlay_available"])
            self.assertEqual(report["hash_mismatch_count"], 0)
            self.assertTrue(report["compile_pass"])
            self.assertFalse(report["metadata_replay_execution_allowed"])

    def test_hash_mismatch_blocks_overlay_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_report(
                protocol_v1_snapshot=self._manifest(expected_sha="not-the-real-hash"),
                overlay_root=tmp,
                object_reader=lambda _ref, _path: b"VALUE = 1\n",
                validate_import=False,
            )

            self.assertFalse(report["protocol_v1_overlay_validated"])
            self.assertFalse(report["protocol_v1_overlay_available"])
            self.assertEqual(report["hash_mismatch_count"], 1)
            self.assertEqual(report["next_action"], "repair_protocol_v1_ephemeral_overlay_preflight")


if __name__ == "__main__":
    unittest.main()
