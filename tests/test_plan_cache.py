import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from langchain_core.messages import AIMessage

from gl_gym.agent.llm_agent import AgentConfig, RuleBasedLLMDirector, create_langchain_tools
from gl_gym.agent.plan_cache import (
    PlanCache,
    config_fingerprint,
    stable_hash,
    state_summary_from_state,
    text_hash,
    to_jsonable,
)
from gl_gym.agent.tools import ControlAction


@dataclass
class DummyConfig:
    model_name: str = "qwen3.7-max"
    base_url: str = "https://example.test/v1"
    temperature: float = 0.0
    max_tokens: int = 260
    max_iterations: int = 1
    control_interval: int = 12
    fallback_strategy: str = "adaptive"
    humidity_memory_enabled: bool = False


@dataclass
class DummyState:
    timestep: int = 24
    day_of_year: float = 240.0
    hour_of_day: float = 6.0
    temp_air: float = 18.123456
    rh_air: float = 82.0
    co2_air: float = 430.0
    glob_rad: float = 100.0
    temp_out: float = 12.0
    rh_out: float = 65.0
    wind_speed: float = 1.0
    dew_margin_air: float = 2.0
    canopy_dew_margin: float = 2.0


class FakeTools:
    def __init__(self):
        self.buffered_action = ControlAction()
        self.buffered_setpoints = {}
        self.status_calls_this_round = 0

    def reset_buffer(self):
        self.buffered_action = ControlAction()
        self.buffered_setpoints = {}
        self.status_calls_this_round = 0

    def get_status(self):
        return "status"


class RecordingGraph:
    def __init__(self, tools):
        self.tools = tools
        self.calls = 0

    def invoke(self, *_args, **_kwargs):
        self.calls += 1
        self.tools.buffered_action = ControlAction(
            u_boil=0.1,
            u_co2=0.0,
            u_th_scr=0.2,
            u_vent=0.3,
            u_lamp=0.0,
            u_bl_scr=0.0,
            action_set=True,
        )
        self.tools.buffered_setpoints = {"target_temp": 18.0, "target_co2": 430.0, "target_rh": 76.0}
        return {"messages": [AIMessage(content="raw response")]}


class RaisingGraph:
    def invoke(self, *_args, **_kwargs):
        raise AssertionError("replay should not call the LLM graph")


def make_director(tmp_path, mode, tools, graph, key_policy="prompt"):
    director = object.__new__(RuleBasedLLMDirector)
    director.config = AgentConfig(
        plan_cache_mode=mode,
        plan_cache_path=str(tmp_path),
        plan_cache_strict=True,
        plan_cache_key_policy=key_policy,
        max_iterations=1,
    )
    director.env_id = "TomatoEnv"
    director.active_control_interval = 12
    director.plan_cache = PlanCache(tmp_path, mode=mode, strict=True)
    director.agent_graph = graph
    director.last_fallback_selection = {}
    director.last_plan_cache_event = {}
    director.last_control = None
    director.last_llm_state = None
    director.last_plan_message = ""
    director.current_plan = None
    director._extract_status_brief = lambda status: str(status)
    director._build_compact_prompt = lambda state, analysis, reason, status_brief, horizon: "prompt"
    director._select_fallback_control = lambda state, analysis=None: np.zeros(6, dtype=np.float32)
    director._enforce_setpoint_contract = lambda state, control, tt, tc, tr: (
        18.0 if tt is None else float(tt),
        430.0 if tc is None else float(tc),
        76.0 if tr is None else float(tr),
        [],
        [],
    )
    director._build_target_profiles = lambda setpoints, tt, tc, tr, horizon: (
        {
            "target_temp": [float(tt)] * int(horizon),
            "target_co2": [float(tc)] * int(horizon),
            "target_rh": [float(tr)] * int(horizon),
        },
        {},
    )
    director._record_decision_reasoning = lambda plan, state, analysis: {}
    create_langchain_tools.instance = tools
    return director


class TestPlanCache(unittest.TestCase):
    def test_state_summary_rounds_and_hashes_stably(self):
        state = DummyState()
        summary = state_summary_from_state(state)
        config_hash = stable_hash(config_fingerprint(DummyConfig()))

        self.assertEqual(summary["temp_air"], 18.1235)
        self.assertEqual(config_hash, stable_hash(config_fingerprint(DummyConfig(humidity_memory_enabled=True))))
        self.assertEqual(text_hash("abc"), text_hash("abc"))

    def test_record_and_reload_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            cache = PlanCache(path, mode="record")
            key = cache.make_key(
                env_id="TomatoEnv",
                state_summary=state_summary_from_state(DummyState()),
                reason="init_plan",
                planning_horizon=12,
                config_hash=stable_hash(config_fingerprint(DummyConfig())),
                prompt_hash=text_hash("prompt"),
                attempt=1,
            )
            cache.put(
                key,
                {
                    "raw_response": "ok",
                    "buffered_action": {
                        "u_boil": np.float32(0.1),
                        "u_co2": 0.0,
                        "u_th_scr": 0.2,
                        "u_vent": 0.3,
                        "u_lamp": 0.0,
                        "u_bl_scr": 0.0,
                        "action_set": True,
                    },
                },
            )

            reloaded = PlanCache(path, mode="replay")
            self.assertEqual(reloaded.get(key)["raw_response"], "ok")
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn(key, data["entries"])

    def test_jsonable_handles_numpy(self):
        value = to_jsonable({"x": np.asarray([1.0, 2.0], dtype=np.float32)})

        self.assertEqual(value, {"x": [1.0, 2.0]})

    def test_director_record_then_replay_uses_same_cached_anchor(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            state = DummyState()
            tools = FakeTools()
            graph = RecordingGraph(tools)
            director = make_director(path, "record", tools, graph)

            result = RuleBasedLLMDirector._replan_with_llm(director, state, {"state": state}, "init_plan")

            self.assertTrue(result["success"])
            self.assertEqual(graph.calls, 1)
            self.assertTrue(path.exists())

            replay_tools = FakeTools()
            replay_director = make_director(path, "replay", replay_tools, RaisingGraph())
            replay = RuleBasedLLMDirector._replan_with_llm(replay_director, state, {"state": state}, "init_plan")

            self.assertTrue(replay["success"])
            self.assertTrue(replay["llm_action_found"])
            self.assertTrue(replay["plan"]["plan_cache_event"]["hit"])
            self.assertAlmostEqual(float(replay_tools.buffered_action.u_boil), 0.1, places=5)
            self.assertAlmostEqual(float(replay_tools.buffered_action.u_vent), 0.3, places=5)

    def test_scenario_timestep_policy_replays_after_state_divergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            tools = FakeTools()
            director = make_director(path, "record", tools, RecordingGraph(tools), key_policy="scenario_timestep")
            state = DummyState(timestep=36, rh_air=82.0)

            result = RuleBasedLLMDirector._replan_with_llm(director, state, {"state": state}, "plan_expired")
            self.assertTrue(result["success"])

            replay_tools = FakeTools()
            replay_director = make_director(
                path,
                "replay",
                replay_tools,
                RaisingGraph(),
                key_policy="scenario_timestep",
            )
            diverged_state = DummyState(timestep=36, rh_air=88.0, temp_air=16.0)
            replay = RuleBasedLLMDirector._replan_with_llm(
                replay_director,
                diverged_state,
                {"state": diverged_state},
                "emergency_replan",
            )

            self.assertTrue(replay["success"])
            self.assertTrue(replay["plan"]["plan_cache_event"]["hit"])
            self.assertEqual(replay["plan"]["plan_cache_event"]["key_policy"], "scenario_timestep")

    def test_strict_replay_cache_miss_raises_instead_of_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "plans.json"
            replay_tools = FakeTools()
            replay_director = make_director(path, "replay", replay_tools, RaisingGraph())

            with self.assertRaises(RuntimeError):
                RuleBasedLLMDirector._replan_with_llm(
                    replay_director,
                    DummyState(),
                    {"state": DummyState()},
                    "init_plan",
                )


if __name__ == "__main__":
    unittest.main()
