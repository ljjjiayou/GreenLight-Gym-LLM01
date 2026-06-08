"""Build v55 qwen3.7 recovery-anchor shadow scoring artifacts.

This pass is offline-only. It designs recovery-anchor candidates for the v54
qwen3.7 no-compatible-anchor rows and scores them against the existing Tomato
Safety boundary. It does not run rollout, call online LLMs, mutate the default
controller, authorize controlled replay, or make performance claims.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    _candidate_action,
    _candidate_name,
    _num,
    _parse_candidates,
    _read_rows,
    _trace_path,
)
from gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 import (  # noqa: E402
    MODEL_NAME,
    V52_TRACE_DIR,
)
from gl_gym.experiments.qwen37_planning_anchor_contract_fix_v54 import (  # noqa: E402
    ANCHOR_AUDIT_JSON as V54_ANCHOR_AUDIT_JSON,
    CONTRACT_DESIGN_JSON as V54_CONTRACT_DESIGN_JSON,
    _load_json,
    _rel,
    _rows_by_step,
    _split_values,
    _tomato_safety_prediction,
)
from gl_gym.experiments.qwen37_profile_template_patch_failure_attribution_v53 import (  # noqa: E402
    LEAK_AUDIT_JSON as V53_LEAK_AUDIT_JSON,
)


RECOVERY_DESIGN_JSON = AUDIT_DIR / "qwen37_recovery_anchor_candidate_design_20260601_v55.json"
RECOVERY_DESIGN_MD = AUDIT_DIR / "qwen37_recovery_anchor_candidate_design_20260601_v55.md"
RECOVERY_AUDIT_JSON = AUDIT_DIR / "qwen37_recovery_anchor_shadow_scoring_audit_20260601_v55.json"
RECOVERY_AUDIT_MD = AUDIT_DIR / "qwen37_recovery_anchor_shadow_scoring_audit_20260601_v55.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v55.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v55.md"

RECOVERY_CANDIDATE_NAMES = (
    "tomato_safety_projected_anchor",
    "hot_dry_canopy_relief_anchor",
    "safe_rule_blend_anchor",
)
ACTION_FIELDS = ("heat", "co2", "screen", "vent", "lamp", "shade")


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _action_dict(action: np.ndarray) -> dict[str, float]:
    arr = np.asarray(action, dtype=np.float32)
    return {name: float(arr[idx]) if idx < arr.size else 0.0 for idx, name in enumerate(ACTION_FIELDS)}


def _parse_action_json(value: Any) -> np.ndarray | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        decoded = json.loads(raw)
    except Exception:
        return None
    if not isinstance(decoded, list) or len(decoded) < 6:
        return None
    return np.asarray([_num(item) for item in decoded[:6]], dtype=np.float32)


def _selected_action_from_row(row: Mapping[str, Any]) -> np.ndarray | None:
    before = _parse_action_json(row.get("tomato_safety_v2_before"))
    if before is not None:
        return before
    candidates = _parse_candidates(row)
    for candidate in candidates:
        if str(candidate.get("selected", "")).lower() in {"1", "true", "yes"}:
            action = _candidate_action(candidate)
            return np.asarray([action[name] for name in ("heating", "co2", "screen", "ventilation", "lighting", "shading")], dtype=np.float32)
    return None


def _projected_action_from_row(row: Mapping[str, Any], selected: np.ndarray) -> np.ndarray:
    after = _parse_action_json(row.get("tomato_safety_v2_after"))
    if after is not None:
        return np.clip(after, 0.0, 1.0).astype(np.float32)
    prediction = _tomato_safety_prediction(row, selected)
    return np.clip(np.asarray(prediction.get("after", selected), dtype=np.float32), 0.0, 1.0).astype(np.float32)


def _rule_action_from_row(row: Mapping[str, Any]) -> np.ndarray | None:
    for candidate in _parse_candidates(row):
        if _candidate_name(candidate).lower() == "rule":
            action = _candidate_action(candidate)
            return np.asarray([action[name] for name in ("heating", "co2", "screen", "ventilation", "lighting", "shading")], dtype=np.float32)
    values = [
        row.get("rule_heating"),
        row.get("rule_co2"),
        row.get("rule_screen"),
        row.get("rule_ventilation"),
        row.get("rule_lighting"),
        row.get("rule_shading"),
    ]
    if any(str(item or "").strip() for item in values):
        return np.asarray([_num(item) for item in values], dtype=np.float32)
    return None


def _safety_reasons(row: Mapping[str, Any], contract_row: Mapping[str, Any] | None = None) -> list[str]:
    reasons = _split_values(row.get("tomato_safety_v2_reasons"))
    if reasons:
        return reasons
    if contract_row:
        selected = contract_row.get("selected_prediction", {}) or {}
        if isinstance(selected, Mapping):
            return [str(item) for item in selected.get("safety_reasons", []) or []]
    return []


def _hot_dry_or_canopy_reason(reasons: Sequence[str]) -> bool:
    text = " ".join(str(item).lower() for item in reasons)
    return any(token in text for token in ("hot_dry", "cooling_guard", "canopy", "dew", "high_rad"))


def _action_conflicts(action: np.ndarray) -> list[str]:
    heat, co2, screen, vent, lamp, shade = [float(x) for x in np.asarray(action, dtype=np.float32)[:6]]
    conflicts: list[str] = []
    if heat >= 0.10 and vent >= 0.40:
        conflicts.append("heat_vent_conflict")
    if co2 >= 0.05 and vent >= 0.20:
        conflicts.append("co2_vent_leak")
    if screen >= 0.75 and vent >= 0.85:
        conflicts.append("screen_vent_extreme_latch")
    if shade >= 0.85 and screen >= 0.80:
        conflicts.append("shade_screen_extreme_latch")
    if lamp >= 0.10 and shade >= 0.80:
        conflicts.append("lamp_shade_conflict")
    return conflicts


def build_recovery_candidates(
    row: Mapping[str, Any],
    contract_row: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selected = _selected_action_from_row(row)
    if selected is None:
        return []
    projected = _projected_action_from_row(row, selected)
    reasons = _safety_reasons(row, contract_row)
    candidates: list[dict[str, Any]] = [
        {
            "name": "tomato_safety_projected_anchor",
            "enabled": True,
            "action": projected,
            "rationale": "reuse Tomato Safety projected action as an explicit recovery anchor",
        }
    ]
    if _hot_dry_or_canopy_reason(reasons):
        relief = projected.copy()
        relief[0] = 0.0
        relief[1] = 0.0
        relief[2] = max(float(relief[2]), 0.60)
        relief[3] = max(float(relief[3]), 0.60)
        relief[4] = 0.0
        relief[5] = max(float(relief[5]), 0.50)
        candidates.append(
            {
                "name": "hot_dry_canopy_relief_anchor",
                "enabled": True,
                "action": np.clip(relief, 0.0, 1.0).astype(np.float32),
                "rationale": "conservative hot-dry/canopy relief anchor",
            }
        )

    rule = _rule_action_from_row(row)
    if rule is not None:
        blend = np.clip(0.15 * rule + 0.85 * projected, 0.0, 1.0).astype(np.float32)
        if float(blend[3]) >= 0.40:
            blend[0] = 0.0
        if float(blend[3]) >= 0.20:
            blend[1] = 0.0
        if float(blend[5]) >= 0.80:
            blend[4] = 0.0
        candidates.append(
            {
                "name": "safe_rule_blend_anchor",
                "enabled": True,
                "action": blend,
                "rationale": "small rule/projection blend with actuator conflict suppression",
            }
        )
    return candidates


def _score_recovery_candidate(
    row: Mapping[str, Any],
    candidate: Mapping[str, Any],
    selected_action: np.ndarray,
) -> dict[str, Any]:
    action = np.clip(np.asarray(candidate.get("action"), dtype=np.float32), 0.0, 1.0)
    prediction = _tomato_safety_prediction(row, action)
    conflicts = _action_conflicts(action)
    delta = float(np.sum(np.abs(action - selected_action)))
    compatible = bool(not prediction.get("hard_safety_rewrite", False) and not conflicts)
    score = delta + 5.0 * int(bool(prediction.get("hard_safety_rewrite", False))) + 2.0 * len(conflicts)
    return {
        "name": str(candidate.get("name") or "recovery_anchor"),
        "rationale": str(candidate.get("rationale") or ""),
        "action": _action_dict(action),
        "delta_from_selected": delta,
        "conflicts": conflicts,
        "conflict_count": len(conflicts),
        "tomato_safety_prediction": {
            "hard_safety_rewrite": bool(prediction.get("hard_safety_rewrite", False)),
            "tomato_safety_applied": bool(prediction.get("tomato_safety_applied", False)),
            "safety_reasons": list(prediction.get("safety_reasons", []) or []),
            "rewrite_fields": list(prediction.get("rewrite_fields", []) or []),
            "delta_abs_sum": float(prediction.get("delta_abs_sum", 0.0) or 0.0),
        },
        "compatible_recovery_anchor": compatible,
        "recovery_score": float(score),
    }


def _rows_for_scenarios(trace_dir: str | Path, scenarios: Sequence[str]) -> dict[str, dict[int, dict[str, Any]]]:
    return {scenario: _rows_by_step(_read_rows(_trace_path(trace_dir, scenario))) for scenario in scenarios}


def build_recovery_anchor_shadow_scoring_audit(
    *,
    v54_contract_audit_json: str | Path = V54_ANCHOR_AUDIT_JSON,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    contract_audit = _load_json(v54_contract_audit_json)
    allowed = set(scenarios)
    contract_rows = [
        dict(row)
        for row in contract_audit.get("contract_rows", []) or []
        if str(row.get("scenario_id") or "") in allowed
    ]
    rows_by_scenario = _rows_for_scenarios(v52_trace_dir, scenarios)
    row_reports: list[dict[str, Any]] = []
    candidate_coverage: Counter[str] = Counter()
    missing_count = 0
    no_recovery_count = 0
    covered_count = 0
    for contract_row in contract_rows:
        scenario = str(contract_row.get("scenario_id") or "")
        step = int(_num(contract_row.get("step"), -1))
        row = rows_by_scenario.get(scenario, {}).get(step)
        if row is None:
            missing_count += 1
            row_reports.append(
                {
                    "scenario_id": scenario,
                    "step": step,
                    "row_found": False,
                    "status": "candidate_metadata_or_control_missing",
                    "no_recovery_anchor_available": True,
                }
            )
            continue
        selected_action = _selected_action_from_row(row)
        if selected_action is None:
            missing_count += 1
            row_reports.append(
                {
                    "scenario_id": scenario,
                    "step": step,
                    "row_found": True,
                    "status": "selected_action_missing",
                    "no_recovery_anchor_available": True,
                }
            )
            continue
        recovery_candidates = build_recovery_candidates(row, contract_row)
        scored = [_score_recovery_candidate(row, candidate, selected_action) for candidate in recovery_candidates]
        compatible = [item for item in scored if item["compatible_recovery_anchor"]]
        best = min(compatible, key=lambda item: item["recovery_score"]) if compatible else None
        if best:
            covered_count += 1
            candidate_coverage[best["name"]] += 1
        else:
            no_recovery_count += 1
        row_reports.append(
            {
                "scenario_id": scenario,
                "step": step,
                "row_found": True,
                "v54_classification": str(contract_row.get("classification") or ""),
                "selected_candidate": str(contract_row.get("selected_candidate") or row.get("rspc_action_selected_name") or ""),
                "selected_source": str(contract_row.get("source") or row.get("source") or ""),
                "selected_safety_reasons": _safety_reasons(row, contract_row),
                "selected_action": _action_dict(selected_action),
                "recovery_candidates": scored,
                "compatible_recovery_candidate_count": len(compatible),
                "best_recovery_candidate": best,
                "no_recovery_anchor_available": best is None,
            }
        )

    evaluated = len(row_reports)
    coverage_rate = float(covered_count / evaluated) if evaluated else 0.0
    return {
        "artifact": "qwen37_recovery_anchor_shadow_scoring_audit_20260601_v55",
        "scope": "offline_recovery_anchor_shadow_scoring_only",
        "source_artifacts": {
            "v54_contract_audit_json": _rel(v54_contract_audit_json),
            "v52_trace_dir": _rel(v52_trace_dir),
        },
        "model_name": MODEL_NAME,
        "scenarios": list(scenarios),
        "expected_v54_no_compatible_anchor_count": int(
            contract_audit.get("no_compatible_anchor_available_count", len(contract_rows)) or 0
        ),
        "evaluated_leak_count": evaluated,
        "recovery_anchor_covered_count": covered_count,
        "no_recovery_anchor_available_count": no_recovery_count,
        "candidate_metadata_or_control_missing_count": missing_count,
        "recovery_anchor_coverage_rate": coverage_rate,
        "candidate_coverage_counts": dict(sorted(candidate_coverage.items())),
        "candidate_names": list(RECOVERY_CANDIDATE_NAMES),
        "row_reports": row_reports,
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def build_recovery_anchor_candidate_design(
    audit: Mapping[str, Any],
    *,
    v54_contract_design_json: str | Path = V54_CONTRACT_DESIGN_JSON,
    v53_leak_audit_json: str | Path = V53_LEAK_AUDIT_JSON,
) -> dict[str, Any]:
    coverage = float(audit.get("recovery_anchor_coverage_rate", 0.0) or 0.0)
    return {
        "artifact": "qwen37_recovery_anchor_candidate_design_20260601_v55",
        "scope": "offline_recovery_anchor_design_only",
        "source_artifacts": {
            "v54_contract_design_json": _rel(v54_contract_design_json),
            "v53_leak_audit_json": _rel(v53_leak_audit_json),
        },
        "model_name": MODEL_NAME,
        "design_ready": bool(
            int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0) == 0
            and int(audit.get("recovery_anchor_covered_count", 0) or 0) > 0
        ),
        "candidate_definitions": [
            {
                "name": "tomato_safety_projected_anchor",
                "intent": "make the Tomato Safety projected action explicit before execution",
                "boundary": "shadow-only; does not weaken Tomato Safety",
            },
            {
                "name": "hot_dry_canopy_relief_anchor",
                "intent": "provide conservative hot-dry/canopy relief when those safety reasons are present",
                "boundary": "enabled only for hot-dry/canopy/dew safety reason rows",
            },
            {
                "name": "safe_rule_blend_anchor",
                "intent": "blend a small rule component into projected anchor while suppressing heat/vent and co2/vent conflicts",
                "boundary": "shadow-only; excluded if Tomato Safety prediction remains hard",
            },
        ],
        "audit_summary": {
            "evaluated_leak_count": int(audit.get("evaluated_leak_count", 0) or 0),
            "recovery_anchor_covered_count": int(audit.get("recovery_anchor_covered_count", 0) or 0),
            "no_recovery_anchor_available_count": int(audit.get("no_recovery_anchor_available_count", 0) or 0),
            "recovery_anchor_coverage_rate": coverage,
            "candidate_coverage_counts": audit.get("candidate_coverage_counts", {}),
        },
        "implementation_boundary": {
            "modify_default_controller_now": False,
            "run_rollout_now": False,
            "online_llm_called": False,
            "controlled_replay_allowed": False,
            "controlled_replay_execution_allowed": False,
            "metadata_replay_execution_allowed": False,
            "performance_claim_allowed": False,
            "promotion_evidence": False,
        },
    }


def _next_action(audit: Mapping[str, Any]) -> str:
    missing = int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0)
    if missing:
        return "recovery_anchor_instrumentation_patch_plan"
    evaluated = int(audit.get("evaluated_leak_count", 0) or 0)
    covered = int(audit.get("recovery_anchor_covered_count", 0) or 0)
    if evaluated and covered > evaluated / 2.0:
        return "minimal_qwen37_recovery_anchor_opt_in_shadow_rollout_plan"
    return "profile_or_tomato_safety_boundary_reconciliation_plan"


def build_readiness(audit: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    missing = int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0)
    no_recovery = int(audit.get("no_recovery_anchor_available_count", 0) or 0)
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v55",
        "current_stage": "v55_qwen37_recovery_anchor_candidate_design",
        "model_name": MODEL_NAME,
        "v55_recovery_anchor_design_done": True,
        "v55_recovery_anchor_design_complete": bool(missing == 0 and design.get("design_ready", False)),
        "evaluated_leak_count": int(audit.get("evaluated_leak_count", 0) or 0),
        "expected_v54_no_compatible_anchor_count": int(
            audit.get("expected_v54_no_compatible_anchor_count", 0) or 0
        ),
        "recovery_anchor_covered_count": int(audit.get("recovery_anchor_covered_count", 0) or 0),
        "recovery_anchor_coverage_rate": float(audit.get("recovery_anchor_coverage_rate", 0.0) or 0.0),
        "no_recovery_anchor_available_count": no_recovery,
        "candidate_metadata_or_control_missing_count": missing,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": _next_action(audit),
        "stop_taxonomy": [
            item
            for item in (
                "metadata_missing" if missing else "",
                "no_recovery_anchor_available" if no_recovery else "",
            )
            if item
        ],
    }


def _write_json_md(path_json: Path, path_md: Path, data: Mapping[str, Any], title: str) -> None:
    path_json.parent.mkdir(parents=True, exist_ok=True)
    path_json.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    lines = [f"# {title}", ""]
    for key in (
        "artifact",
        "scope",
        "current_stage",
        "model_name",
        "next_action",
    ):
        if key in data:
            lines.append(f"- {key}: `{data.get(key)}`")
    for key in (
        "evaluated_leak_count",
        "expected_v54_no_compatible_anchor_count",
        "recovery_anchor_covered_count",
        "recovery_anchor_coverage_rate",
        "no_recovery_anchor_available_count",
        "candidate_metadata_or_control_missing_count",
    ):
        if key in data:
            lines.append(f"- {key}: `{data.get(key)}`")
    for key in (
        "controlled_replay_allowed",
        "controlled_replay_execution_allowed",
        "metadata_replay_execution_allowed",
        "performance_claim_allowed",
        "promotion_evidence",
    ):
        if key in data:
            lines.append(f"- {key}: `{str(data.get(key)).lower()}`")
    for key in ("audit_summary", "candidate_definitions", "candidate_coverage_counts", "stop_taxonomy"):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    if "row_reports" in data:
        lines.extend(
            [
                "",
                "## row_reports_sample",
                "```json",
                json.dumps(list(data.get("row_reports", []))[:20], indent=2, sort_keys=True),
                "```",
            ]
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    v54_contract_audit_json: str | Path = V54_ANCHOR_AUDIT_JSON,
    v54_contract_design_json: str | Path = V54_CONTRACT_DESIGN_JSON,
    v53_leak_audit_json: str | Path = V53_LEAK_AUDIT_JSON,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = build_recovery_anchor_shadow_scoring_audit(
        v54_contract_audit_json=v54_contract_audit_json,
        v52_trace_dir=v52_trace_dir,
        scenarios=scenarios,
    )
    design = build_recovery_anchor_candidate_design(
        audit,
        v54_contract_design_json=v54_contract_design_json,
        v53_leak_audit_json=v53_leak_audit_json,
    )
    readiness = build_readiness(audit, design)
    _write_json_md(RECOVERY_DESIGN_JSON, RECOVERY_DESIGN_MD, design, "v55 Qwen3.7 Recovery Anchor Candidate Design")
    _write_json_md(RECOVERY_AUDIT_JSON, RECOVERY_AUDIT_MD, audit, "v55 Qwen3.7 Recovery Anchor Shadow Scoring Audit")
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v55 Metadata Replay Readiness")
    return design, audit, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v54-contract-audit-json", default=str(V54_ANCHOR_AUDIT_JSON))
    parser.add_argument("--v54-contract-design-json", default=str(V54_CONTRACT_DESIGN_JSON))
    parser.add_argument("--v53-leak-audit-json", default=str(V53_LEAK_AUDIT_JSON))
    parser.add_argument("--v52-trace-dir", default=str(V52_TRACE_DIR))
    parser.add_argument("--failure-scenario", action="append")
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    design, audit, readiness = write_all(
        v54_contract_audit_json=args.v54_contract_audit_json,
        v54_contract_design_json=args.v54_contract_design_json,
        v53_leak_audit_json=args.v53_leak_audit_json,
        v52_trace_dir=args.v52_trace_dir,
        scenarios=scenarios,
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "design_ready": design.get("design_ready"),
                    "evaluated_leak_count": audit.get("evaluated_leak_count"),
                    "recovery_anchor_covered_count": audit.get("recovery_anchor_covered_count"),
                    "no_recovery_anchor_available_count": audit.get("no_recovery_anchor_available_count"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
