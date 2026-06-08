import unittest

from gl_gym.experiments.evaluation_protocol_delta_classification_v2 import build_report


class TestEvaluationProtocolDeltaClassificationV2(unittest.TestCase):
    def test_runtime_error_and_strict_cache_miss_are_stricter_gate(self):
        diff_text = """
diff --git a/gl_gym/experiments/frozen_benchmark_protocol.py b/gl_gym/experiments/frozen_benchmark_protocol.py
--- a/gl_gym/experiments/frozen_benchmark_protocol.py
+++ b/gl_gym/experiments/frozen_benchmark_protocol.py
@@ -10,2 +10,5 @@ def validate_strict_replay_result():
+    runtime_error_steps = int(metrics.get("runtime_error_steps", 0) or 0)
+    strict_cache_miss_runtime_error_steps = int(metrics.get("strict_cache_miss_runtime_error_steps", 0) or 0)
+    reasons.append("strict_cache_miss_runtime_error_steps")
"""
        report = build_report(diff_text)

        self.assertFalse(report["protocol_isolation_pass"])
        self.assertTrue(report["gate_semantics_changed"])
        self.assertEqual(report["unknown_hunk_count"], 0)
        self.assertIn("stricter_safety_gate", report["category_counts"])
        self.assertEqual(report["category_counts"]["stricter_safety_gate"], 1)

    def test_hot_dry_proposer_and_agent_overrides_are_runner_control_surface(self):
        diff_text = """
diff --git a/gl_gym/experiments/run_frozen_benchmark.py b/gl_gym/experiments/run_frozen_benchmark.py
--- a/gl_gym/experiments/run_frozen_benchmark.py
+++ b/gl_gym/experiments/run_frozen_benchmark.py
@@ -20,2 +20,5 @@
+    "llm_rspc_v2_hot_dry_proposer": "hot dry proposer",
+    agent_config_overrides = parse_agent_config_overrides(args.agent_config_overrides)
+    run_llm_trace(controller="llm_rspc_v2_hot_dry_proposer", agent_config_overrides=agent_config_overrides)
"""
        report = build_report(diff_text)

        self.assertTrue(report["runner_control_surface_changed"])
        self.assertEqual(report["unknown_hunk_count"], 0)
        self.assertEqual(report["category_counts"]["runner_control_surface"], 1)

    def test_frozen_benchmark_test_hunk_is_test_oracle(self):
        diff_text = """
diff --git a/tests/test_frozen_benchmark.py b/tests/test_frozen_benchmark.py
--- a/tests/test_frozen_benchmark.py
+++ b/tests/test_frozen_benchmark.py
@@ -5,2 +5,4 @@ class TestFrozenBenchmark(unittest.TestCase):
+    def test_runtime_error_gate_blocks(self):
+        self.assertEqual(result.failures, ["runtime_error_steps"])
"""
        report = build_report(diff_text)

        self.assertTrue(report["test_oracle_changed"])
        self.assertEqual(report["unknown_hunk_count"], 0)
        self.assertEqual(report["category_counts"]["test_oracle"], 1)

    def test_planning_extensions_diff_is_safety_boundary_test_oracle(self):
        diff_text = """
diff --git a/tests/test_planning_extensions.py b/tests/test_planning_extensions.py
--- a/tests/test_planning_extensions.py
+++ b/tests/test_planning_extensions.py
@@ -20,2 +20,5 @@ class TestPlanningExtensions(unittest.TestCase):
+    def test_tomato_safety_v2_hot_dry_guard(self):
+        self.assertIn("hot_dry_cooling_guard", info["reasons"])
+        self.assertGreaterEqual(float(control[3]), 0.599)
"""
        report = build_report(diff_text)

        self.assertTrue(report["test_oracle_changed"])
        self.assertTrue(report["safety_boundary_test_oracle_changed"])
        self.assertEqual(report["safety_boundary_test_oracle_hunk_count"], 1)
        self.assertEqual(report["unknown_hunk_count"], 0)
        self.assertTrue(report["rows"][0]["safety_boundary_test_oracle"])

    def test_unknown_hunk_keeps_protocol_isolation_false(self):
        diff_text = """
diff --git a/gl_gym/experiments/frozen_benchmark_protocol.py b/gl_gym/experiments/frozen_benchmark_protocol.py
--- a/gl_gym/experiments/frozen_benchmark_protocol.py
+++ b/gl_gym/experiments/frozen_benchmark_protocol.py
@@ -1,2 +1,3 @@
+    unexplained_new_protocol_branch = maybe_value
"""
        report = build_report(diff_text)

        self.assertFalse(report["protocol_isolation_pass"])
        self.assertFalse(report["protocol_delta_explained"])
        self.assertEqual(report["unknown_hunk_count"], 1)
        self.assertEqual(report["category_counts"]["unknown_requires_manual_review"], 1)


if __name__ == "__main__":
    unittest.main()
