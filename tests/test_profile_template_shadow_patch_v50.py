import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.profile_template_shadow_patch_v50 import (
    build_readiness,
    build_shadow_patch_audit,
    fallback_post_selection_veto_shadow,
    profile_template_shadow_targets,
    shadow_patch_for_row,
)


FIELDS = [
    "step",
    "source",
    "temp_air",
    "rh_air",
    "vpd_air",
    "glob_rad",
    "dew_margin_air",
    "canopy_dew_margin",
    "temp_violation",
    "dry_risk",
    "dry_vent_risk",
    "target_temp",
    "target_rh",
    "target_co2",
    "u_heating",
    "u_co2",
    "u_screen",
    "u_ventilation",
    "u_lighting",
    "u_shading",
    "tomato_safety_v2_applied",
    "tomato_safety_v2_vent_before",
    "tomato_safety_v2_vent_after",
    "tomato_safety_v2_heat_before",
    "tomato_safety_v2_heat_after",
    "tomato_safety_v2_screen_before",
    "tomato_safety_v2_screen_after",
    "tomato_safety_v2_shade_before",
    "tomato_safety_v2_shade_after",
    "tomato_safety_v2_reasons",
    "rspc_action_candidates_json",
    "rspc_action_actual_candidate_count",
    "rspc_action_candidate_count",
    "profile_scorer_safety_gate_reason",
    "rspc_action_post_shape_safety_gate_reason",
]


def _candidate(name: str, selected: bool, *, vent: float = 0.9, heat: float = 0.2, co2: float = 900.0):
    return {
        "name": name,
        "score": 1.0 if selected else 2.0,
        "selected": selected,
        "action": {"heat": heat, "co2": co2, "screen": 0.0, "vent": vent, "lamp": 0.0, "shade": 0.0},
        "score_terms": {"total_score": 1.0 if selected else 2.0},
    }


def _row(
    *,
    step=0,
    source="anchor",
    hard=True,
    safety_required=True,
    dry=True,
    high_targets=True,
    candidates=True,
):
    cand = [_candidate("anchor", True), _candidate("hold", False, vent=0.2, heat=0.0, co2=0.0)]
    return {
        "step": step,
        "source": source,
        "temp_air": 29.5,
        "rh_air": 48.0 if dry else 92.0,
        "vpd_air": 1.8 if dry else 0.2,
        "glob_rad": 500.0,
        "dew_margin_air": 0.7 if safety_required else 3.0,
        "canopy_dew_margin": 0.8 if safety_required else 3.0,
        "temp_violation": 0.0,
        "dry_risk": "true" if dry else "false",
        "dry_vent_risk": "true" if dry else "false",
        "target_temp": 24.0,
        "target_rh": 82.0 if high_targets else 70.0,
        "target_co2": 900.0 if high_targets else 430.0,
        "u_heating": 0.2,
        "u_co2": 0.0,
        "u_screen": 0.0,
        "u_ventilation": 0.9,
        "u_lighting": 0.0,
        "u_shading": 0.0,
        "tomato_safety_v2_applied": "true" if hard else "false",
        "tomato_safety_v2_vent_before": 0.2,
        "tomato_safety_v2_vent_after": 0.9 if hard else 0.2,
        "tomato_safety_v2_heat_before": 0.0,
        "tomato_safety_v2_heat_after": 0.0,
        "tomato_safety_v2_screen_before": 0.0,
        "tomato_safety_v2_screen_after": 0.0,
        "tomato_safety_v2_shade_before": 0.0,
        "tomato_safety_v2_shade_after": 0.0,
        "tomato_safety_v2_reasons": "hard_canopy" if hard else "",
        "rspc_action_candidates_json": json.dumps(cand) if candidates else "",
        "rspc_action_actual_candidate_count": 2 if candidates else 0,
        "rspc_action_candidate_count": 2 if candidates else 0,
        "profile_scorer_safety_gate_reason": "canopy_gate" if safety_required else "none",
        "rspc_action_post_shape_safety_gate_reason": "canopy_gate" if safety_required else "none",
    }


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            base = {field: "" for field in FIELDS}
            base.update({key: str(value) for key, value in row.items()})
            writer.writerow(base)


class TestProfileTemplateShadowPatchV50(unittest.TestCase):
    def test_safety_required_vent_suppresses_dry_high_vent_penalty(self):
        result = profile_template_shadow_targets(_row(safety_required=True, dry=True))

        self.assertTrue(result["mode"]["vent_required_by_safety"])
        self.assertIn("dry_high_vent_penalty_suppressed_by_safety_vent", result["forbidden_combinations"])
        self.assertNotIn("dry_high_vent_penalty_active", result["forbidden_combinations"])

    def test_rh_dew_canopy_pressure_blocks_humidity_retention(self):
        result = profile_template_shadow_targets(_row(safety_required=True, dry=False))

        self.assertFalse(result["mode"]["humidity_retention_allowed"])
        self.assertIn("rh_dew_canopy_pressure_blocks_humidity_retention", result["forbidden_combinations"])

    def test_high_vent_high_rh_high_co2_template_is_repaired(self):
        result = profile_template_shadow_targets(_row(safety_required=True, high_targets=True))

        self.assertIn("safety_vent_blocks_high_rh_high_co2", result["forbidden_combinations"])
        self.assertLess(result["template_shadow_targets"]["target_rh"], result["original_targets"]["target_rh"])
        self.assertLess(result["template_shadow_targets"]["target_co2"], result["original_targets"]["target_co2"])

    def test_fallback_selected_hard_safety_rewrite_is_shadow_blocked(self):
        row = _row(source="anchor", hard=True, candidates=True)
        veto = fallback_post_selection_veto_shadow(row)
        patch = shadow_patch_for_row(row)

        self.assertTrue(veto["fallback_veto_shadow_would_block"])
        self.assertEqual(veto["veto_reason"], "fallback_post_selection_hard_safety_rewrite")
        self.assertTrue(patch["would_block_hard_safety_leak"])

    def test_build_audit_includes_all_v49_leak_steps_and_readiness_blocks_claims(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            v48 = root / "v48"
            original.mkdir()
            v48.mkdir()
            scenario = "y2010_d180_s43_n720"
            rows = [_row(step=i, hard=True, source="anchor") for i in range(16)]
            _write(original / f"{scenario}_llm_rspc_v2.csv", rows)
            _write(v48 / f"{scenario}_llm_rspc_v2.csv", rows)
            leak_entries = [
                {"scenario_id": scenario, "step": i, "classification": "fallback_path_bypassed_veto"}
                for i in range(16)
            ]
            v49 = root / "v49.json"
            v49.write_text(json.dumps({"hard_safety_veto_leak_audit": {"leak_entries": leak_entries}}), encoding="utf-8")

            audit = build_shadow_patch_audit(
                v49_runtime_audit_json=v49,
                original_trace_dir=original,
                v48_trace_dir=v48,
                failure_scenarios=[scenario],
            )
            readiness = build_readiness(audit)

        self.assertEqual(audit["expected_leak_step_count"], 16)
        self.assertEqual(audit["evaluated_leak_step_count"], 16)
        self.assertEqual(audit["would_block_hard_safety_leak_count"], 16)
        self.assertTrue(audit["all_leak_steps_shadow_blocked"])
        self.assertFalse(readiness["controlled_replay_allowed"])
        self.assertFalse(readiness["metadata_replay_execution_allowed"])
        self.assertFalse(readiness["performance_claim_allowed"])
        self.assertFalse(readiness["promotion_evidence"])


if __name__ == "__main__":
    unittest.main()
