import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.rspc_action_scoring_audit import audit_trace, audit_traces, build_report, classify_row


def _candidate(
    name,
    *,
    score,
    selected=False,
    heat=0.0,
    co2=0.0,
    screen=0.2,
    vent=0.75,
    lamp=0.0,
    shade=0.0,
    temp_next=29.0,
    rh_next=50.0,
    vpd_next=1.8,
    dry_penalty=25.0,
    vpd_penalty=1.0,
    temp_penalty=0.0,
    dew_penalty=0.0,
):
    return {
        "name": name,
        "score": float(score),
        "selected": bool(selected),
        "action": {
            "heat": float(heat),
            "co2": float(co2),
            "screen": float(screen),
            "vent": float(vent),
            "lamp": float(lamp),
            "shade": float(shade),
        },
        "score_terms": {
            "score": float(score),
            "temp_next": float(temp_next),
            "rh_next": float(rh_next),
            "vpd_next": float(vpd_next),
            "dry_penalty": float(dry_penalty),
            "vpd_penalty": float(vpd_penalty),
            "temp_penalty": float(temp_penalty),
            "dew_penalty": float(dew_penalty),
            "hot_dry_penalty": 0.0,
        },
    }


def _row(**extra):
    selected = _candidate("rule", score=1.0, selected=True)
    data = {
        "step": 0,
        "rspc_action_audit_enabled": True,
        "rspc_action_selected_name": "rule",
        "rspc_action_selected_score": 1.0,
        "rspc_action_hot_dry_active": True,
        "rspc_action_hot_dry_candidate_count": 0,
        "rspc_action_candidates_json": json.dumps([selected], separators=(",", ":")),
        "rh_air": 50.0,
        "vpd_air": 1.8,
        "temp_air": 29.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
        "u_boil": 0.0,
        "u_co2": 0.0,
        "u_th_scr": 0.2,
        "u_ventilation": 0.75,
        "u_lamp": 0.0,
        "u_bl_scr": 0.0,
        "tomato_safety_v2_applied": False,
        "rspc_action_dry_recovery_applied": False,
    }
    data.update(extra)
    return data


def _write_csv(path, rows):
    rows = list(rows)
    fieldnames = sorted({key for row in rows for key in row})
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


class TestRspcActionScoringAudit(unittest.TestCase):
    def test_classifies_candidate_gap_when_pool_lacks_safe_dry_action(self):
        result = classify_row(_row())

        self.assertEqual(result["classification"], "candidate_gap")
        self.assertEqual(result["dry_candidate_count"], 0)

    def test_classifies_scoring_gap_when_safe_dry_candidate_loses(self):
        selected = _candidate("rule", score=1.0, selected=True)
        dry = _candidate(
            "humidity_retention",
            score=1.4,
            selected=False,
            vent=0.25,
            shade=0.5,
            rh_next=52.0,
            vpd_next=1.65,
            dry_penalty=20.0,
            vpd_penalty=0.6,
        )
        result = classify_row(
            _row(
                rspc_action_candidates_json=json.dumps([selected, dry], separators=(",", ":")),
            )
        )

        self.assertEqual(result["classification"], "scoring_gap")
        self.assertEqual(result["best_dry_name"], "humidity_retention")
        self.assertGreater(result["best_dry_score_delta"], 0.0)

    def test_classifies_post_score_override_before_candidate_gap(self):
        result = classify_row(
            _row(
                rspc_action_dry_recovery_applied=True,
                u_ventilation=0.25,
                u_bl_scr=0.5,
            )
        )

        self.assertEqual(result["classification"], "post_score_override_gap")

    def test_classifies_post_shape_selection_gap_before_post_override(self):
        selected = _candidate("rule", score=1.0, selected=True)
        shaped = _candidate(
            "humidity_retention",
            score=1.2,
            selected=False,
            vent=0.25,
            shade=0.5,
            rh_next=52.0,
            vpd_next=1.65,
            dry_penalty=20.0,
            vpd_penalty=0.6,
        )
        result = classify_row(
            _row(
                rspc_action_candidates_json=json.dumps([selected, shaped], separators=(",", ":")),
                rspc_action_dry_recovery_applied=True,
                rspc_action_post_shape_enabled=True,
                rspc_action_post_shape_would_switch=True,
                rspc_action_post_shape_best_name="humidity_retention",
                rspc_action_post_shape_margin=0.25,
                rspc_action_post_shape_alignment="dry_benefit",
                rspc_action_post_shape_dry_benefit=True,
                u_ventilation=0.25,
                u_bl_scr=0.5,
            )
        )

        self.assertEqual(result["classification"], "post_shape_selection_gap")
        self.assertTrue(result["post_shape_would_switch"])
        self.assertEqual(result["post_shape_best_name"], "humidity_retention")

    def test_tracks_hot_dry_proposer_shadow_selection(self):
        selected = _candidate("rule", score=1.0, selected=True)
        proposer = _candidate(
            "shadow_hot_dry_humidity_retention",
            score=0.7,
            selected=False,
            screen=0.65,
            vent=0.25,
            shade=0.74,
            rh_next=53.0,
            vpd_next=1.55,
            dry_penalty=16.0,
            vpd_penalty=0.4,
        )
        proposer["shadow_proposer"] = True
        result = classify_row(
            _row(
                rspc_action_hot_dry_active=False,
                rspc_action_hot_dry_semantic_active=True,
                rspc_action_hot_dry_proposer_active=True,
                rspc_action_hot_dry_proposer_candidate_count=1,
                rspc_action_candidates_json=json.dumps([selected, proposer], separators=(",", ":")),
                rspc_action_post_shape_enabled=True,
                rspc_action_post_shape_would_switch=True,
                rspc_action_post_shape_best_name="shadow_hot_dry_humidity_retention",
                rspc_action_post_shape_best_is_proposer=True,
                rspc_action_post_shape_margin=0.30,
                rspc_action_post_shape_alignment="dry_benefit",
                rspc_action_post_shape_dry_benefit=True,
            )
        )

        self.assertEqual(result["classification"], "post_shape_selection_gap")
        self.assertTrue(result["hot_dry_proposer_best"])
        self.assertTrue(result["post_shape_proposer_dry_benefit"])
        self.assertEqual(result["hot_dry_proposer_candidate_count"], 1)

    def test_classifies_post_shape_unsafe_conflict_under_safety(self):
        selected = _candidate("rule", score=1.0, selected=True, temp_next=33.0, temp_penalty=1.0)
        risky = _candidate("screen_hold", score=1.2, screen=0.9, vent=0.2, temp_next=33.2, temp_penalty=1.2)
        result = classify_row(
            _row(
                temp_air=33.0,
                rspc_action_candidates_json=json.dumps([selected, risky], separators=(",", ":")),
                rspc_action_post_shape_enabled=True,
                rspc_action_post_shape_would_switch=True,
                rspc_action_post_shape_best_name="screen_hold",
                rspc_action_post_shape_margin=0.20,
                rspc_action_post_shape_alignment="unsafe_conflict",
                rspc_action_post_shape_unsafe_conflict=True,
            )
        )

        self.assertEqual(result["classification"], "post_shape_unsafe_conflict")
        self.assertTrue(result["post_shape_unsafe_conflict"])

    def test_proposer_unsafe_conflict_remains_blocking(self):
        selected = _candidate("rule", score=1.0, selected=True, temp_next=33.0, temp_penalty=1.0)
        risky = _candidate(
            "shadow_hot_dry_humidity_retention",
            score=0.7,
            screen=0.85,
            vent=0.18,
            shade=0.2,
            temp_next=33.5,
            temp_penalty=2.0,
        )
        risky["shadow_proposer"] = True
        result = classify_row(
            _row(
                temp_air=33.0,
                rspc_action_hot_dry_proposer_candidate_count=1,
                rspc_action_candidates_json=json.dumps([selected, risky], separators=(",", ":")),
                rspc_action_post_shape_enabled=True,
                rspc_action_post_shape_would_switch=True,
                rspc_action_post_shape_best_name="shadow_hot_dry_humidity_retention",
                rspc_action_post_shape_best_is_proposer=True,
                rspc_action_post_shape_margin=0.20,
                rspc_action_post_shape_alignment="unsafe_conflict",
                rspc_action_post_shape_unsafe_conflict=True,
            )
        )

        self.assertEqual(result["classification"], "post_shape_unsafe_conflict")
        self.assertTrue(result["hot_dry_proposer_best"])

    def test_safety_row_does_not_count_as_hot_dry_performance_gap(self):
        selected = _candidate("rule", score=1.0, selected=True, temp_next=33.0, temp_penalty=1.0)
        cooling = _candidate(
            "shade_cooling",
            score=1.2,
            shade=0.6,
            vent=0.65,
            temp_next=31.0,
            rh_next=52.0,
            vpd_next=1.55,
            dry_penalty=20.0,
            vpd_penalty=0.5,
        )
        result = classify_row(
            _row(
                temp_air=33.0,
                rspc_action_candidates_json=json.dumps([selected, cooling], separators=(",", ":")),
            )
        )

        self.assertEqual(result["classification"], "no_action_layer_signal")
        self.assertFalse(result["safe_hot_dry"])

    def test_audit_trace_and_report_are_stable(self):
        rows = [_row(step=0)]
        selected = _candidate("rule", score=1.0, selected=True)
        dry = _candidate("humidity_retention", score=1.4, vent=0.2, rh_next=52.0, vpd_next=1.65, dry_penalty=20.0)
        rows.append(_row(step=1, rspc_action_candidates_json=json.dumps([selected, dry], separators=(",", ":"))))
        rows.append(_row(step=2, rspc_action_dry_recovery_applied=True, u_ventilation=0.25))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, rows)

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["action_audit_steps"], 3)
        self.assertEqual(trace["classification_counts"]["candidate_gap"], 1)
        self.assertEqual(trace["classification_counts"]["scoring_gap"], 1)
        self.assertEqual(trace["classification_counts"]["post_score_override_gap"], 1)
        self.assertIn("RSPC/Fallback Action Scoring Audit", build_report(audit))

    def test_warns_when_metadata_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d180_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["action_audit_steps"], 0)
        self.assertIn("missing_rspc_action_scoring_metadata", trace["warnings"])
        self.assertEqual(trace["recommendation"]["decision"], "needs_trace_metadata")


if __name__ == "__main__":
    unittest.main()
