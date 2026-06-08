import unittest
import tempfile
from pathlib import Path

from gl_gym.cstcc.config import load_config
from gl_gym.cstcc.contracts import (
    BOUNDARY_FALSE_FIELDS,
    CSTCCAuditRecord,
    PREDICTION_LEVEL,
    REGIMES,
    SCORE_VALIDITY,
    SemanticSuggestion,
    asdict_clean,
)


class TestCSTCCV80Contracts(unittest.TestCase):
    def test_config_preserves_v80_boundaries(self):
        config = load_config()

        self.assertEqual(config["version"], "v80")
        self.assertFalse(config["controller_changed"])
        self.assertFalse(config["final_action_changed"])
        self.assertFalse(config["online_llm_enabled"])
        self.assertEqual(config["prediction_level"], PREDICTION_LEVEL)
        self.assertEqual(config["score_validity"], SCORE_VALIDITY)

    def test_config_rejects_online_llm_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.yaml"
            path.write_text(
                "\n".join(
                    [
                        "version: v80",
                        "controller_changed: false",
                        "final_action_changed: false",
                        "online_llm_enabled: true",
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ValueError):
                load_config(path)

    def test_semantic_suggestion_clamps_confidence_and_weights(self):
        suggestion = SemanticSuggestion(
            regime="NORMAL_BALANCED",
            regime_confidence=2.0,
            llm_reported_confidence=-1.0,
            priority_weight_suggestions={"vpd_risk": 4.0, "energy": -0.5},
        )

        self.assertEqual(suggestion.regime_confidence, 1.0)
        self.assertEqual(suggestion.llm_reported_confidence, 0.0)
        self.assertEqual(suggestion.priority_weight_suggestions["vpd_risk"], 1.0)
        self.assertEqual(suggestion.priority_weight_suggestions["energy"], 0.0)

    def test_audit_record_rejects_boundary_true(self):
        with self.assertRaises(ValueError):
            CSTCCAuditRecord(final_action_changed=True)

    def test_audit_record_serializes_all_false_boundaries(self):
        record = CSTCCAuditRecord()
        payload = asdict_clean(record)

        for field in BOUNDARY_FALSE_FIELDS:
            self.assertFalse(payload[field])
        self.assertEqual(len(REGIMES), 7)


if __name__ == "__main__":
    unittest.main()
