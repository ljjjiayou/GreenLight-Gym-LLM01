import unittest

from gl_gym.experiments.run_frozen_benchmark import build_jobs, parse_controller_list


class TestFrozenBenchmark(unittest.TestCase):
    def test_builds_full_job_matrix(self):
        jobs = build_jobs(
            years=[2010, 2020],
            days=[59, 240],
            seeds=[42],
            controllers=["llm", "llm_hem", "ppo"],
            max_steps=240,
        )

        self.assertEqual(len(jobs), 12)
        self.assertEqual(jobs[0].scenario_id, "y2010_d59_s42_n240")
        self.assertEqual(jobs[-1].controller, "ppo")

    def test_rejects_unknown_controller(self):
        with self.assertRaises(ValueError):
            parse_controller_list("llm,bad")


if __name__ == "__main__":
    unittest.main()

