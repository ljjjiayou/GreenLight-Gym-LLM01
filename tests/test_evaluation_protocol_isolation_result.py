import unittest

from gl_gym.experiments.evaluation_protocol_isolation_result import build_report


class TestEvaluationProtocolIsolationResult(unittest.TestCase):
    def test_runtime_error_gate_semantics_require_protocol_authorization(self):
        diff_text = """
diff --git a/gl_gym/experiments/frozen_benchmark_protocol.py b/gl_gym/experiments/frozen_benchmark_protocol.py
--- a/gl_gym/experiments/frozen_benchmark_protocol.py
+++ b/gl_gym/experiments/frozen_benchmark_protocol.py
@@ -1,2 +1,4 @@
+        runtime_error_steps = int(aggregate.get("runtime_error_steps", 0) or 0)
+        if runtime_error_steps:
+            reasons.append("runtime_error_steps")
"""
        report = build_report(diff_text)

        self.assertFalse(report["metadata_replay_allowed"])
        self.assertFalse(report["protocol_isolation_pass"])
        self.assertTrue(report["gate_semantics_changed"])
        self.assertTrue(report["requires_protocol_baseline_authorization"])
        self.assertEqual(report["decision"], "protocol_v2_candidate_requires_authorization")
        self.assertIn("stricter_safety_gate_delta", report["delta_categories"])

    def test_metadata_only_delta_is_explained_but_not_replay_evidence(self):
        diff_text = """
diff --git a/gl_gym/experiments/frozen_benchmark_protocol.py b/gl_gym/experiments/frozen_benchmark_protocol.py
--- a/gl_gym/experiments/frozen_benchmark_protocol.py
+++ b/gl_gym/experiments/frozen_benchmark_protocol.py
@@ -1,2 +1,3 @@
+                "timesteps": [],
+        env["timestep_count"] = int(len(timesteps))
"""
        report = build_report(diff_text)

        self.assertTrue(report["protocol_delta_explained"])
        self.assertFalse(report["gate_semantics_changed"])
        self.assertFalse(report["requires_protocol_baseline_authorization"])
        self.assertIn("metadata_report_only_delta", report["delta_categories"])
        self.assertFalse(report["metadata_replay_allowed"])


if __name__ == "__main__":
    unittest.main()
