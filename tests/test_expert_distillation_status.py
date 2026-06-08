import unittest
from unittest.mock import patch

from gl_gym.experiments import expert_distillation_status as status_mod


class _Result:
    def __init__(self, stdout=""):
        self.stdout = stdout
        self.returncode = 0


class TestExpertDistillationStatus(unittest.TestCase):
    def test_clean_status_is_not_hard_blocker(self):
        with patch.object(status_mod, "_run_git", side_effect=[_Result(""), _Result("")]):
            report = status_mod.build_report()

        self.assertEqual(report["decision"], "clean_not_current_blocker")
        self.assertFalse(report["hard_blocker"])
        self.assertFalse(report["metadata_replay_allowed"])

    def test_modified_status_is_hard_blocker(self):
        with patch.object(status_mod, "_run_git", side_effect=[_Result(" M gl_gym/agent/expert_distillation.py"), _Result("diff")]):
            report = status_mod.build_report()

        self.assertEqual(report["decision"], "hard_blocker_restore_or_authorization_required")
        self.assertTrue(report["hard_blocker"])


if __name__ == "__main__":
    unittest.main()
