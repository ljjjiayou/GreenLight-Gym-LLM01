import unittest

from gl_gym.cstcc.constraints import check_hard_constraints, soft_penalties
from gl_gym.cstcc.contracts import ACTION_FIELDS
from gl_gym.cstcc.sequence_templates import TEMPLATE_NAMES, generate_sequence, generate_sequence_with_metadata


LAST_ACTION = {
    "u_heating": 0.2,
    "u_co2": 0.4,
    "u_screen": 0.5,
    "u_ventilation": 0.3,
    "u_lighting": 0.1,
    "u_shading": 0.2,
}

PRIOR_ACTION = {
    "u_heating": 0.8,
    "u_co2": 0.7,
    "u_screen": 0.2,
    "u_ventilation": 0.9,
    "u_lighting": 0.7,
    "u_shading": 0.8,
}


class TestCSTCCV80SequenceAndConstraints(unittest.TestCase):
    def test_all_minimal_templates_generate_horizon_actions(self):
        for template in TEMPLATE_NAMES:
            with self.subTest(template=template):
                sequence = generate_sequence(template, last_action=LAST_ACTION, prior_action=PRIOR_ACTION, horizon=12)

                self.assertEqual(len(sequence), 12)
                for action in sequence:
                    self.assertEqual(tuple(action.keys()), ACTION_FIELDS)
                    self.assertTrue(all(0.0 <= value <= 1.0 for value in action.values()))

    def test_ventilation_template_is_rate_limited(self):
        sequence = generate_sequence(
            "ventilation_ramp_limited",
            last_action=LAST_ACTION,
            prior_action=PRIOR_ACTION,
            horizon=3,
        )

        self.assertLessEqual(sequence[0]["u_ventilation"] - LAST_ACTION["u_ventilation"], 0.080000001)

    def test_previous_shift_all_is_sequentially_projected_from_current_action(self):
        previous_sequence = [
            dict(LAST_ACTION),
            {**LAST_ACTION, "u_ventilation": 1.0, "u_screen": 0.9},
            {**LAST_ACTION, "u_ventilation": 0.8, "u_screen": 0.7, "u_heating": 0.5},
        ]
        max_delta = {
            "u_heating": 0.20,
            "u_co2": 0.25,
            "u_screen": 0.20,
            "u_ventilation": 0.20,
            "u_lighting": 0.30,
            "u_shading": 0.20,
        }

        sequence, metadata = generate_sequence_with_metadata(
            "previous_shift_all",
            last_action=LAST_ACTION,
            previous_sequence=previous_sequence,
            horizon=4,
            max_delta_by_field=max_delta,
        )

        self.assertEqual(len(sequence), 4)
        self.assertTrue(metadata["previous_shift_projection_applied"])
        self.assertEqual(metadata["previous_shift_projection_reference_action"], LAST_ACTION)
        self.assertLessEqual(abs(sequence[0]["u_ventilation"] - LAST_ACTION["u_ventilation"]), max_delta["u_ventilation"])
        self.assertLessEqual(abs(sequence[0]["u_screen"] - LAST_ACTION["u_screen"]), max_delta["u_screen"])
        previous = LAST_ACTION
        for action in sequence:
            for field in ACTION_FIELDS:
                self.assertLessEqual(abs(action[field] - previous[field]), max_delta[field] + 1e-9)
            previous = action
        self.assertGreater(sum(abs(sequence[-1][field] - LAST_ACTION[field]) for field in ACTION_FIELDS), 0.0)
        self.assertFalse(check_hard_constraints(sequence, previous_action=LAST_ACTION, max_delta=max_delta))

    def test_conflict_hard_constraints_are_reported(self):
        violations = check_hard_constraints(
            [{"u_heating": 0.9, "u_co2": 0.8, "u_screen": 0.2, "u_ventilation": 0.9, "u_lighting": 0.0, "u_shading": 0.0}]
        )
        reasons = {violation["reason"] for violation in violations}

        self.assertIn("co2_injection_with_high_ventilation", reasons)
        self.assertIn("heating_with_strong_ventilation", reasons)

    def test_soft_penalty_report_is_normalized(self):
        sequence = generate_sequence("ppo_follow_limited", last_action=LAST_ACTION, prior_action=PRIOR_ACTION, horizon=12)
        penalties = soft_penalties(sequence)

        self.assertTrue(all(0.0 <= value <= 1.0 for value in penalties.values()))


if __name__ == "__main__":
    unittest.main()
