import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from gl_gym.experiments.controlled_canary_weather_shadow_cache_acquisition_execution_record import build_report


def _manifest():
    return {
        "weather_shadow_cache_acquisition_manifest_ready": True,
        "scenario_count": 9,
        "isolated_cache_path": "gl_gym/result/plan_cache/weather_expansion_shadow_cache_v36_20260529.json",
        "scenario_ids": [f"y2006_d197_s{seed}_n240" for seed in [42, 43, 44]]
        + [f"y2006_d183_s{seed}_n240" for seed in [42, 43, 44]]
        + [f"y2005_d169_s{seed}_n240" for seed in [42, 43, 44]],
        "command_groups": [
            {
                "group_id": "weather_v36_y2006_d197",
                "years": [2006],
                "days": [197],
                "seeds": [42, 43, 44],
                "max_steps": 240,
                "cartesian_product_safe": True,
            },
            {
                "group_id": "weather_v36_y2006_d183",
                "years": [2006],
                "days": [183],
                "seeds": [42, 43, 44],
                "max_steps": 240,
                "cartesian_product_safe": True,
            },
            {
                "group_id": "weather_v36_y2005_d169",
                "years": [2005],
                "days": [169],
                "seeds": [42, 43, 44],
                "max_steps": 240,
                "cartesian_product_safe": True,
            },
        ],
    }


def _overlay(runner: str):
    return {
        "controlled_canary_overlay_validated": True,
        "controlled_canary_runner_surface_validated": True,
        "protocol_hash_mismatch_count": 0,
        "overlay_runner_path": runner,
        "protocol_v1_overlay_root": "overlay",
    }


class TestControlledCanaryWeatherShadowCacheAcquisitionExecutionRecord(unittest.TestCase):
    def test_record_authorized_but_not_executable_without_api_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="user_delegated_v36_shadow_cache_acquisition_authorization_20260529",
                api_key_present=False,
            )

        self.assertTrue(report["weather_shadow_cache_acquisition_authorized"])
        self.assertFalse(report["weather_shadow_cache_acquisition_executable"])
        self.assertIn("online_llm_credentials_missing", report["failure_taxonomy"])
        self.assertTrue(report["cache_fill_authorized"])
        self.assertTrue(report["online_llm_allowed"])
        self.assertFalse(report["controlled_replay_allowed"])

    def test_record_executable_with_api_key_and_record_mode_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                api_key_present=True,
            )

        self.assertTrue(report["weather_shadow_cache_acquisition_executable"])
        self.assertEqual(report["controllers"], ["llm_rspc_v2"])
        self.assertNotEqual(report["isolated_cache_path"], report["selected_cache_path"])
        for command in report["planned_commands"]:
            self.assertIn("--plan-cache-mode", command)
            self.assertIn("record", command)
            self.assertIn("--plan-cache-key-policy", command)
            self.assertIn("scenario_timestep", command)
            self.assertNotIn("--plan-cache-strict", command)
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", command)

    def test_record_loads_dotenv_without_leaking_secret(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runner = root / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            dotenv = root / ".env"
            dotenv.write_text("BAILIAN_API_KEY=super-secret-test-key\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                report = build_report(
                    manifest=_manifest(),
                    overlay_preflight=_overlay(str(runner)),
                    authorization_source="source",
                    dotenv_path=dotenv,
                )

        self.assertTrue(report["weather_shadow_cache_acquisition_executable"])
        self.assertTrue(report["precheck"]["online_llm_credentials_present"])
        self.assertNotIn("super-secret-test-key", str(report))

    def test_record_prefers_environment_key_over_missing_dotenv(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            with patch.dict("os.environ", {"BAILIAN_API_KEY": "env-secret"}, clear=True):
                report = build_report(
                    manifest=_manifest(),
                    overlay_preflight=_overlay(str(runner)),
                    authorization_source="source",
                    dotenv_path=Path(tmp) / "missing.env",
                )

        self.assertTrue(report["weather_shadow_cache_acquisition_executable"])
        self.assertTrue(report["precheck"]["online_llm_credentials_present"])
        self.assertNotIn("env-secret", str(report))

    def test_record_blocks_when_env_and_dotenv_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            with patch.dict("os.environ", {}, clear=True):
                report = build_report(
                    manifest=_manifest(),
                    overlay_preflight=_overlay(str(runner)),
                    authorization_source="source",
                    dotenv_path=Path(tmp) / "missing.env",
                )

        self.assertFalse(report["weather_shadow_cache_acquisition_executable"])
        self.assertIn("online_llm_credentials_missing", report["failure_taxonomy"])

    def test_record_blocks_selected_cache_overwrite(self):
        manifest = _manifest()
        manifest["isolated_cache_path"] = "gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json"
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=manifest,
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                api_key_present=True,
            )

        self.assertFalse(report["weather_shadow_cache_acquisition_authorized"])
        self.assertIn("selected_cache_overwrite_blocked", report["failure_taxonomy"])

    def test_record_blocks_controlled_controller(self):
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp) / "run_frozen_benchmark.py"
            runner.write_text("# runner\n", encoding="utf-8")
            report = build_report(
                manifest=_manifest(),
                overlay_preflight=_overlay(str(runner)),
                authorization_source="source",
                controllers=["llm_rspc_v2", "llm_rspc_v2_hot_dry_proposer_strict"],
                api_key_present=True,
            )

        self.assertFalse(report["weather_shadow_cache_acquisition_authorized"])
        self.assertIn("controller_scope_mismatch", report["failure_taxonomy"])


if __name__ == "__main__":
    unittest.main()
