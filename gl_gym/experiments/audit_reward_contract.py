"""Audit the greenhouse reward contract before RL retraining.

This script does not change training. It records whether the configured reward
actually optimizes the safety penalties we intend to report in papers.
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gl_gym.environments.rewards import GreenhouseReward


def audit_greenhouse_reward() -> Dict[str, Any]:
    source = inspect.getsource(GreenhouseReward.compute_reward)
    return_line = ""
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("return "):
            return_line = stripped
    weighted_penalty_computed = "self.penalty = self.output_penalty_reward(violations)" in source
    weighted_penalty_returned = "self.penalty" in return_line
    scaled_unweighted_penalty_returned = "scaled_pen" in return_line and "violations" in source
    control_penalty_returned = "self.control_pen" in return_line
    return {
        "reward_class": "GreenhouseReward",
        "reward_version_recommendation": "freeze_current_as_reward_v0_before_rl_sweeps",
        "weighted_penalty_computed": bool(weighted_penalty_computed),
        "weighted_penalty_returned": bool(weighted_penalty_returned),
        "scaled_unweighted_state_penalty_returned": bool(scaled_unweighted_penalty_returned),
        "control_penalty_returned": bool(control_penalty_returned),
        "return_line": return_line,
        "finding": (
            "pen_weights are computed into self.penalty but are not used directly in the returned reward; "
            "state violations enter through an unweighted normalized sum. PPO/SAC sweeps should either "
            "freeze this as reward_v0 or introduce a documented reward_v1."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit greenhouse reward implementation.")
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    audit = audit_greenhouse_reward()
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

