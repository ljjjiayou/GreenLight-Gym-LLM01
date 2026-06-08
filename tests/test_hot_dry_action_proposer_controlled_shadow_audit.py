import csv
import json
import tempfile
import unittest
from pathlib import Path

from gl_gym.experiments.hot_dry_action_proposer_controlled_shadow_audit import (
    audit_trace,
    audit_traces,
    build_report,
    classify_row,
)


def _candidate(
    name,
    *,
    score=1.0,
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
    dry_penalty=8.0,
    vpd_penalty=0.8,
    temp_penalty=0.0,
    dew_penalty=0.0,
    hot_dry_penalty=0.0,
    shadow_proposer=False,
):
    return {
        "name": name,
        "score": float(score),
        "selected": bool(selected),
        "shadow_proposer": bool(shadow_proposer),
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
            "rh_penalty": 0.0,
            "dew_penalty": float(dew_penalty),
            "energy_penalty": 0.0,
            "smooth_penalty": 0.0,
            "conflict_penalty": 0.0,
            "hot_dry_penalty": float(hot_dry_penalty),
            "mitigation_bonus": 0.0,
        },
    }


def _row(*candidates, **extra):
    if not candidates:
        candidates = (_candidate("rule", selected=True),)
    data = {
        "step": 0,
        "rspc_action_audit_enabled": True,
        "rspc_action_selected_name": "rule",
        "rspc_action_hot_dry_active": True,
        "rspc_action_hot_dry_proposer_active": True,
        "rspc_action_candidates_json": json.dumps(list(candidates), separators=(",", ":")),
        "rh_air": 50.0,
        "vpd_air": 1.8,
        "temp_air": 29.0,
        "dew_margin_air": 3.0,
        "canopy_dew_margin": 3.0,
        "rh_low_violation": 0.0,
        "vpd_high_excess": 0.0,
        "temp_violation": 0.0,
        "rh_high_violation": 0.0,
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


class TestHotDryActionProposerControlledShadowAudit(unittest.TestCase):
    def test_detects_safe_proposer_gain(self):
        selected = _candidate("rule", selected=True, dry_penalty=10.0, vpd_penalty=1.0)
        proposer = _candidate(
            "shadow_hot_dry_humidity_retention",
            shadow_proposer=True,
            screen=0.65,
            vent=0.30,
            shade=0.70,
            rh_next=52.0,
            vpd_next=1.65,
            dry_penalty=8.4,
            vpd_penalty=0.7,
        )

        result = classify_row(_row(selected, proposer))

        self.assertEqual(result["classification"], "controlled_shadow_safe_proposer_gain")
        self.assertEqual(result["best_gain_name"], "shadow_hot_dry_humidity_retention")
        self.assertGreater(result["safe_gain_variant_count"], 0)

    def test_detects_no_proposer(self):
        selected = _candidate("rule", selected=True)

        result = classify_row(_row(selected))

        self.assertEqual(result["classification"], "controlled_shadow_no_proposer")

    def test_detects_no_eligible_proposer(self):
        selected = _candidate("rule", selected=True, dry_penalty=10.0, vpd_penalty=1.0)
        proposer = _candidate(
            "shadow_hot_dry_humidity_retention",
            shadow_proposer=True,
            heat=0.20,
            rh_next=52.0,
            vpd_next=1.65,
            dry_penalty=10.0,
            vpd_penalty=1.0,
        )

        result = classify_row(_row(selected, proposer))

        self.assertEqual(result["classification"], "controlled_shadow_no_eligible_proposer")
        self.assertEqual(result["eligible_proposer_count"], 0)

    def test_detects_unsafe_preferred_proposer(self):
        selected = _candidate("rule", selected=True, dry_penalty=10.0, vpd_penalty=1.0)
        proposer = _candidate(
            "shadow_hot_dry_humidity_retention",
            shadow_proposer=True,
            screen=0.95,
            vent=0.05,
            rh_next=53.0,
            vpd_next=1.55,
            dry_penalty=7.8,
            vpd_penalty=0.5,
            temp_next=32.4,
            temp_penalty=0.9,
        )

        result = classify_row(_row(selected, proposer))

        self.assertEqual(result["classification"], "controlled_shadow_unsafe_proposer")
        self.assertGreater(result["unsafe_preferred_variant_count"], 0)

    def test_ignores_non_hot_dry_rows(self):
        selected = _candidate("rule", selected=True)
        proposer = _candidate("shadow_hot_dry_humidity_retention", shadow_proposer=True)

        result = classify_row(
            _row(
                selected,
                proposer,
                rh_air=70.0,
                vpd_air=0.8,
                rspc_action_hot_dry_active=False,
                rspc_action_hot_dry_proposer_active=False,
            )
        )

        self.assertEqual(result["classification"], "not_safe_hot_dry")

    def test_audit_trace_and_report_are_stable(self):
        selected = _candidate("rule", selected=True, dry_penalty=10.0, vpd_penalty=1.0)
        proposer = _candidate(
            "shadow_hot_dry_humidity_retention",
            shadow_proposer=True,
            vent=0.30,
            rh_next=52.0,
            vpd_next=1.65,
            dry_penalty=8.4,
            vpd_penalty=0.7,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm_rspc_v2.csv"
            _write_csv(path, [_row(selected, proposer), _row(selected)])

            trace = audit_trace(path)
            audit = audit_traces([tmp])

        self.assertEqual(trace["metadata_steps"], 2)
        self.assertEqual(trace["safe_gain_steps"], 1)
        self.assertEqual(audit["aggregate"]["recommendation"]["decision"], "controlled_shadow_signal_detected")
        self.assertIn("Hot-Dry Action Proposer Controlled Shadow Audit", build_report(audit))

    def test_missing_metadata_warns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "y2015_d120_s44_n240_llm.csv"
            _write_csv(path, [{"step": 0, "reward": 0.0}])

            trace = audit_trace(path)

        self.assertEqual(trace["metadata_steps"], 0)
        self.assertEqual(trace["recommendation"]["decision"], "needs_trace_metadata")
        self.assertIn("missing_rspc_action_scoring_metadata", trace["warnings"])


if __name__ == "__main__":
    unittest.main()
