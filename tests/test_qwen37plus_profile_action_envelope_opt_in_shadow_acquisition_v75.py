import json
import unittest

from gl_gym.experiments import qwen37plus_profile_action_envelope_opt_in_shadow_acquisition_v75 as v75


def _v74_readiness(**overrides):
    data = {
        "version": "v74",
        "next_action": "needs_profile_action_envelope_opt_in_shadow_trace_acquisition",
        "reason": "No v74 opt-in envelope shadow trace metadata was found.",
        "final_action_invariant": True,
        "online_llm_called": False,
        "promotion_evidence": False,
    }
    data.update(overrides)
    return data


class TestQwen37PlusProfileActionEnvelopeOptInShadowAcquisitionV75(unittest.TestCase):
    def test_authorization_delegates_online_acquisition_without_calling_llm(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=True,
        )

        self.assertTrue(authorization["user_authorized_online_llm_acquisition"])
        self.assertTrue(authorization["online_llm_allowed_for_acquisition"])
        self.assertFalse(authorization["online_llm_called"])
        self.assertFalse(authorization["default_llm_rspc_v2_changed"])
        self.assertFalse(authorization["final_action_changed"])
        self.assertTrue(authorization["shadow_only"])
        self.assertEqual(authorization["model"], "qwen3.7-plus")
        self.assertIn("profile_action_envelope_shadow_candidates_json", authorization["required_provenance_fields"])

    def test_manifest_commands_are_shadow_only_llm_rspc_v2_with_envelope_overrides(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=True,
        )
        manifest = v75.build_manifest(authorization)

        self.assertTrue(manifest["rollout_command_generated"])
        self.assertFalse(manifest["controlled_controller_included"])
        self.assertEqual(manifest["controller"], "llm_rspc_v2")
        self.assertEqual(len(manifest["commands"]), 2)
        for item in manifest["commands"]:
            argv = item["argv"]
            self.assertIn("--controllers", argv)
            self.assertEqual(argv[argv.index("--controllers") + 1], "llm_rspc_v2")
            self.assertIn("--plan-cache-mode", argv)
            self.assertEqual(argv[argv.index("--plan-cache-mode") + 1], "record")
            self.assertIn("--agent-config-overrides", argv)
            overrides = json.loads(argv[argv.index("--agent-config-overrides") + 1])
            self.assertTrue(overrides["profile_action_envelope_shadow_enabled"])
            self.assertTrue(overrides["profile_action_envelope_shadow_record_provenance"])
            self.assertEqual(overrides["profile_action_envelope_shadow_max_candidates"], 5)
            self.assertFalse(overrides["transition_gate_enabled"])
            self.assertFalse(overrides["fallback_post_selection_veto_enabled"])
            self.assertNotIn("llm_rspc_v2_hot_dry_proposer_strict", argv)

    def test_readiness_is_executable_only_with_v74_gate_authorization_and_credentials(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=True,
        )
        manifest = v75.build_manifest(authorization)
        readiness = v75.build_readiness(authorization, manifest)

        self.assertTrue(readiness["executable"])
        self.assertEqual(readiness["next_action"], "execute_v75_opt_in_shadow_trace_acquisition_commands")
        self.assertTrue(readiness["rollout_command_generated"])
        self.assertFalse(readiness["online_llm_called"])

    def test_missing_credentials_blocks_execution_but_keeps_authorization_packet(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=False,
        )
        readiness = v75.build_readiness(authorization, v75.build_manifest(authorization))

        self.assertTrue(authorization["user_authorized_online_llm_acquisition"])
        self.assertFalse(readiness["executable"])
        self.assertEqual(readiness["next_action"], "online_llm_credentials_precheck_blocker")
        self.assertEqual(readiness["blocked_reason"], "BAILIAN_API_KEY_not_detected")

    def test_boundary_flags_do_not_allow_replay_promotion_or_performance_claims(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=True,
        )
        manifest = v75.build_manifest(authorization)
        readiness = v75.build_readiness(authorization, manifest)

        for payload in (authorization, manifest, readiness):
            self.assertFalse(payload["controlled_replay_allowed"])
            self.assertFalse(payload["controlled_replay_execution_allowed"])
            self.assertFalse(payload["metadata_replay_execution_allowed"])
            self.assertFalse(payload["performance_claim_allowed"])
            self.assertFalse(payload["promotion_evidence"])
            self.assertFalse(payload["fallback_enhanced"])
            self.assertFalse(payload["final_action_changed"])

    def test_markdown_reports_include_authorization_and_command_boundary(self):
        authorization = v75.build_authorization(
            v74_readiness=_v74_readiness(),
            user_authorized=True,
            credentials_present=True,
        )
        manifest = v75.build_manifest(authorization)
        readiness = v75.build_readiness(authorization, manifest)

        self.assertIn("online_llm_allowed_for_acquisition=True", v75.build_authorization_markdown(authorization))
        self.assertIn("online_llm_called=False", v75.build_manifest_markdown(manifest))
        self.assertIn("next_action=execute_v75_opt_in_shadow_trace_acquisition_commands", v75.build_readiness_markdown(readiness))


if __name__ == "__main__":
    unittest.main()
