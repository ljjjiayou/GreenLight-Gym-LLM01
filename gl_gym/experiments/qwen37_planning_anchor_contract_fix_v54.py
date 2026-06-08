"""Build v54 qwen3.7 planning-anchor contract audit artifacts.

This pass is offline-only. It reads v53's step-aligned qwen3.7 leak rows and
the v52 shadow traces, then checks whether the selected anchor/fallback action
had a hard-safety-compatible alternative under the same Tomato Safety boundary.
It does not run rollout, call online LLMs, mutate the default controller,
authorize controlled replay, or make performance claims.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.agent.llm_agent import AgentConfig, apply_tomato_safety_v2  # noqa: E402
from gl_gym.experiments.candidate_guardrail_shadow_scoring_v46 import (  # noqa: E402
    ACTION_NAMES,
    AUDIT_DIR,
    FAILURE_SCENARIOS,
    _candidate_action,
    _candidate_name,
    _candidate_score,
    _num,
    _parse_candidates,
    _read_rows,
    _trace_path,
    _truthy,
)
from gl_gym.experiments.profile_template_opt_in_shadow_rollout_v52 import (  # noqa: E402
    MODEL_NAME,
    V52_TRACE_DIR,
)
from gl_gym.experiments.qwen37_profile_template_patch_failure_attribution_v53 import (  # noqa: E402
    ATTRIBUTION_JSON as V53_ATTRIBUTION_JSON,
    LEAK_AUDIT_JSON as V53_LEAK_AUDIT_JSON,
)


ANCHOR_AUDIT_JSON = AUDIT_DIR / "qwen37_planning_anchor_contract_audit_20260601_v54.json"
ANCHOR_AUDIT_MD = AUDIT_DIR / "qwen37_planning_anchor_contract_audit_20260601_v54.md"
CONTRACT_DESIGN_JSON = AUDIT_DIR / "qwen37_post_selection_hard_safety_contract_design_20260601_v54.json"
CONTRACT_DESIGN_MD = AUDIT_DIR / "qwen37_post_selection_hard_safety_contract_design_20260601_v54.md"
READINESS_JSON = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v54.json"
READINESS_MD = AUDIT_DIR / "metadata_replay_readiness_checklist_20260601_v54.md"

CONTRACT_CLASSIFICATIONS = (
    "direct_strategy_bypassed_veto",
    "recent_or_hold_anchor_contract_unsafe",
    "fallback_scored_no_compatible_alternative",
    "planner_anchor_empty_contract_filled",
    "candidate_metadata_or_control_missing",
    "unknown_requires_instrumentation",
)

STATE_NUMERIC_FIELDS = (
    "temp_air",
    "rh_air",
    "co2_air",
    "glob_rad",
    "hour_of_day",
    "dew_margin_air",
    "canopy_dew_margin",
    "forecast_rad_mean_1h",
    "forecast_rad_peak_2h",
    "forecast_temp_out_delta_1h",
    "temp_air_delta_1h",
    "temp_air_rise_1h",
    "temp_air_trend_c_per_hour",
    "temp_air_slope_c_per_step",
    "temp_air_slope",
    "temp_trend_slope",
    "u_th_scr",
    "u_bl_scr",
    "u_vent",
    "vpd_air",
)


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p


def _rel(path: str | Path) -> str:
    p = _resolve(path)
    try:
        return str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p).replace("\\", "/")


def _load_json(path: str | Path) -> dict[str, Any]:
    p = _resolve(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _rows_by_step(rows: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        out[int(_num(row.get("step"), -1))] = dict(row)
    return out


def _split_values(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip() and part.strip() != "none"]


def _state_from_row(row: Mapping[str, Any]) -> SimpleNamespace:
    attrs = {field: _num(row.get(field)) for field in STATE_NUMERIC_FIELDS}
    return SimpleNamespace(**attrs)


def _target_rh_for_prediction(row: Mapping[str, Any]) -> float | None:
    repaired = row.get("profile_template_patch_repaired_target_rh")
    if str(repaired or "").strip():
        return _num(repaired)
    target = row.get("target_rh")
    if str(target or "").strip():
        return _num(target)
    return None


def _candidate_action_array(candidate: Mapping[str, Any]) -> np.ndarray:
    action = _candidate_action(candidate)
    return np.asarray([action[name] for name in ACTION_NAMES], dtype=np.float32)


def _tomato_safety_prediction(
    row: Mapping[str, Any],
    control: np.ndarray,
    *,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AgentConfig(tomato_safety_v2_enabled=True)
    shaped, info = apply_tomato_safety_v2(
        _state_from_row(row),
        np.asarray(control, dtype=np.float32),
        config=cfg,
        target_rh=_target_rh_for_prediction(row),
    )
    before = np.asarray(control, dtype=np.float32)
    after = np.asarray(shaped, dtype=np.float32)
    reason_text = " ".join(str(item) for item in info.get("reasons", []) or []).lower()
    hard_safety = bool(
        info.get("applied", False)
        and any(token in reason_text for token in ("hard", "dew", "canopy", "extreme", "hot_temperature"))
    )
    rewrite_fields = [
        name
        for idx, name in enumerate(("heat", "co2", "screen", "vent", "lamp", "shade"))
        if idx < after.size and idx < before.size and abs(float(after[idx]) - float(before[idx])) > 1e-6
    ]
    return {
        "hard_safety_rewrite": hard_safety,
        "tomato_safety_applied": bool(info.get("applied", False)),
        "safety_reasons": list(info.get("reasons", []) or []),
        "rewrite_fields": rewrite_fields,
        "delta_abs_sum": float(np.sum(np.abs(after - before))),
        "before": [float(x) for x in before.tolist()],
        "after": [float(x) for x in after.tolist()],
    }


def _selected_candidate_name(row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]) -> str:
    for candidate in candidates:
        if _truthy(candidate.get("selected")):
            return _candidate_name(candidate)
    for key in ("rspc_action_selected_name", "selected_fallback_candidate", "source"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    return "unknown"


def _selected_candidate(
    row: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    selected_name: str,
) -> Mapping[str, Any] | None:
    for candidate in candidates:
        if _truthy(candidate.get("selected")):
            return candidate
    lowered = selected_name.lower()
    for candidate in candidates:
        if _candidate_name(candidate).lower() == lowered:
            return candidate
    return None


def _candidate_prediction(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    prediction = _tomato_safety_prediction(row, _candidate_action_array(candidate))
    return {
        "name": _candidate_name(candidate),
        "score": float(_candidate_score(candidate)),
        "selected_original": bool(_truthy(candidate.get("selected"))),
        "hard_safety_rewrite": bool(prediction["hard_safety_rewrite"]),
        "tomato_safety_applied": bool(prediction["tomato_safety_applied"]),
        "safety_reasons": list(prediction["safety_reasons"]),
        "rewrite_fields": list(prediction["rewrite_fields"]),
        "delta_abs_sum": float(prediction["delta_abs_sum"]),
    }


def _selected_prediction_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    before_raw = str(row.get("tomato_safety_v2_before") or "").strip()
    if before_raw:
        try:
            decoded = json.loads(before_raw)
            if isinstance(decoded, list) and len(decoded) >= 6:
                return _tomato_safety_prediction(row, np.asarray(decoded[:6], dtype=np.float32))
        except Exception:
            pass
    control = np.asarray(
        [
            _num(row.get("u_heating")),
            _num(row.get("u_co2")),
            _num(row.get("u_screen")),
            _num(row.get("u_ventilation")),
            _num(row.get("u_lighting")),
            _num(row.get("u_shading")),
        ],
        dtype=np.float32,
    )
    return _tomato_safety_prediction(row, control)


def classify_anchor_contract(
    leak_entry: Mapping[str, Any],
    *,
    row_found: bool,
    candidate_count: int,
    selected_hard_safety: bool,
    compatible_alternative_count: int,
    selected_source: str = "",
    selected_fallback_candidate: str = "",
    fallback_veto_applied: bool = False,
) -> str:
    if not row_found or candidate_count <= 0:
        return "candidate_metadata_or_control_missing"
    if not selected_hard_safety:
        return "unknown_requires_instrumentation"
    if compatible_alternative_count <= 0:
        return "fallback_scored_no_compatible_alternative"

    source_text = " ".join(
        [
            selected_source,
            selected_fallback_candidate,
            str(leak_entry.get("anchor_source") or ""),
            str(leak_entry.get("candidate_selection_source") or ""),
        ]
    ).lower()
    if any(token in source_text for token in ("recent_anchor", "hold_current")):
        return "recent_or_hold_anchor_contract_unsafe"
    if not fallback_veto_applied and any(token in source_text for token in ("rule", "legacy", "anchor")):
        return "direct_strategy_bypassed_veto"
    if str(leak_entry.get("classification") or "") == "planner_anchor_empty_contract_filled":
        return "planner_anchor_empty_contract_filled"
    return "unknown_requires_instrumentation"


def _contract_matrix_entry(
    leak_entry: Mapping[str, Any],
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    scenario = str(leak_entry.get("scenario_id") or "")
    step = int(_num(leak_entry.get("step"), -1))
    if row is None:
        return {
            "scenario_id": scenario,
            "step": step,
            "row_found": False,
            "classification": "candidate_metadata_or_control_missing",
            "candidate_count": 0,
            "compatible_alternative_count": 0,
            "no_compatible_anchor_available": True,
        }

    candidates = _parse_candidates(row)
    selected_name = _selected_candidate_name(row, candidates)
    selected_candidate = _selected_candidate(row, candidates, selected_name)
    candidate_predictions = [_candidate_prediction(row, candidate) for candidate in candidates]
    compatible = [item for item in candidate_predictions if not item["hard_safety_rewrite"]]
    selected_prediction = (
        _candidate_prediction(row, selected_candidate)
        if selected_candidate is not None
        else _selected_prediction_from_row(row)
    )
    best_compatible = min(compatible, key=lambda item: item["score"]) if compatible else None
    selected_source = str(row.get("source") or leak_entry.get("source") or "")
    selected_fallback = str(row.get("selected_fallback_candidate") or leak_entry.get("selected_fallback_candidate") or "")
    fallback_veto_applied = _truthy(row.get("profile_template_patch_fallback_veto_applied"))
    classification = classify_anchor_contract(
        leak_entry,
        row_found=True,
        candidate_count=len(candidates),
        selected_hard_safety=bool(selected_prediction.get("hard_safety_rewrite", False)),
        compatible_alternative_count=len(compatible),
        selected_source=selected_source,
        selected_fallback_candidate=selected_fallback,
        fallback_veto_applied=fallback_veto_applied,
    )
    return {
        "scenario_id": scenario,
        "step": step,
        "row_found": True,
        "model_name": str(row.get("model_name") or ""),
        "v53_classification": str(leak_entry.get("classification") or ""),
        "classification": classification,
        "source": selected_source,
        "anchor_source": str(row.get("anchor_source") or leak_entry.get("anchor_source") or ""),
        "candidate_selection_source": str(
            row.get("candidate_selection_source") or leak_entry.get("candidate_selection_source") or ""
        ),
        "selected_candidate": selected_name,
        "selected_fallback_candidate": selected_fallback,
        "fallback_veto_applied": fallback_veto_applied,
        "fallback_veto_no_alternative": _truthy(row.get("profile_template_patch_fallback_veto_no_alternative")),
        "profile_template_patch": {
            "enabled": _truthy(row.get("profile_template_patch_enabled")),
            "applied": _truthy(row.get("profile_template_patch_applied")),
            "mode": str(row.get("profile_template_patch_mode") or ""),
            "corrections": _split_values(row.get("profile_template_patch_corrections")),
            "forbidden_combinations": _split_values(row.get("profile_template_patch_forbidden_combinations")),
            "vent_required_by_safety": _truthy(row.get("profile_template_patch_vent_required_by_safety")),
            "humidity_retention_allowed": _truthy(row.get("profile_template_patch_humidity_retention_allowed")),
            "co2_enrichment_allowed": _truthy(row.get("profile_template_patch_co2_enrichment_allowed")),
        },
        "targets": {
            "target_temp": _num(row.get("target_temp")),
            "target_rh": _num(row.get("target_rh")),
            "target_co2": _num(row.get("target_co2")),
            "repaired_target_temp": _num(row.get("profile_template_patch_repaired_target_temp")),
            "repaired_target_rh": _num(row.get("profile_template_patch_repaired_target_rh")),
            "repaired_target_co2": _num(row.get("profile_template_patch_repaired_target_co2")),
        },
        "selected_prediction": selected_prediction,
        "candidate_count": len(candidates),
        "candidate_predictions": candidate_predictions,
        "compatible_alternative_count": len(compatible),
        "best_compatible_alternative": best_compatible,
        "no_compatible_anchor_available": best_compatible is None,
    }


def build_anchor_contract_audit(
    *,
    leak_audit_json: str | Path = V53_LEAK_AUDIT_JSON,
    attribution_json: str | Path = V53_ATTRIBUTION_JSON,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> dict[str, Any]:
    leak_audit = _load_json(leak_audit_json)
    attribution = _load_json(attribution_json)
    allowed = set(scenarios)
    leak_entries = [
        dict(entry)
        for entry in leak_audit.get("leak_entries", []) or []
        if str(entry.get("scenario_id") or "") in allowed
    ]
    rows_by_scenario: dict[str, dict[int, dict[str, Any]]] = {}
    for scenario in scenarios:
        rows_by_scenario[scenario] = _rows_by_step(_read_rows(_trace_path(v52_trace_dir, scenario)))

    contract_rows: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    compatible_count = 0
    no_compatible_count = 0
    missing = 0
    unknown = 0
    for entry in leak_entries:
        scenario = str(entry.get("scenario_id") or "")
        step = int(_num(entry.get("step"), -1))
        row = rows_by_scenario.get(scenario, {}).get(step)
        matrix = _contract_matrix_entry(entry, row)
        contract_rows.append(matrix)
        classification_counts[str(matrix.get("classification") or "unknown_requires_instrumentation")] += 1
        compatible_count += int(bool(matrix.get("best_compatible_alternative")))
        no_compatible_count += int(bool(matrix.get("no_compatible_anchor_available")))
        missing += int(matrix.get("classification") == "candidate_metadata_or_control_missing")
        unknown += int(matrix.get("classification") == "unknown_requires_instrumentation")

    missing_categories = {name: 0 for name in CONTRACT_CLASSIFICATIONS if name not in classification_counts}
    classification_payload = {**missing_categories, **dict(sorted(classification_counts.items()))}
    return {
        "artifact": "qwen37_planning_anchor_contract_audit_20260601_v54",
        "scope": "offline_qwen37_planning_anchor_contract_audit_only",
        "source_artifacts": {
            "v53_leak_audit_json": _rel(leak_audit_json),
            "v53_attribution_json": _rel(attribution_json),
            "v52_trace_dir": _rel(v52_trace_dir),
        },
        "model_name": MODEL_NAME,
        "scenarios": list(scenarios),
        "v53_dominant_root_cause": attribution.get("dominant_root_cause"),
        "expected_v53_step_aligned_leak_count": int(
            leak_audit.get("step_aligned_new_hard_safety_leak_count", len(leak_entries)) or 0
        ),
        "evaluated_leak_count": len(contract_rows),
        "all_v53_leaks_evaluated": len(contract_rows)
        == int(leak_audit.get("step_aligned_new_hard_safety_leak_count", len(leak_entries)) or 0),
        "compatible_alternative_count": compatible_count,
        "no_compatible_anchor_available_count": no_compatible_count,
        "candidate_metadata_or_control_missing_count": missing,
        "unknown_requires_instrumentation_count": unknown,
        "classification_counts": classification_payload,
        "contract_rows": contract_rows,
        "online_llm_called": False,
        "new_rollout_run": False,
        "default_llm_rspc_v2_changed": False,
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
    }


def _next_action(audit: Mapping[str, Any]) -> str:
    missing = int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0)
    unknown = int(audit.get("unknown_requires_instrumentation_count", 0) or 0)
    if missing or unknown:
        return "anchor_contract_instrumentation_patch_plan"
    compatible = int(audit.get("compatible_alternative_count", 0) or 0)
    no_compatible = int(audit.get("no_compatible_anchor_available_count", 0) or 0)
    if compatible > no_compatible:
        return "minimal_qwen37_anchor_contract_opt_in_shadow_rollout_plan"
    return "recovery_anchor_candidate_design_plan"


def build_contract_design(audit: Mapping[str, Any]) -> dict[str, Any]:
    next_action = _next_action(audit)
    no_compatible = int(audit.get("no_compatible_anchor_available_count", 0) or 0)
    compatible = int(audit.get("compatible_alternative_count", 0) or 0)
    return {
        "artifact": "qwen37_post_selection_hard_safety_contract_design_20260601_v54",
        "scope": "offline_contract_design_only",
        "model_name": MODEL_NAME,
        "recommended_next_action": next_action,
        "design_ready": bool(next_action != "anchor_contract_instrumentation_patch_plan"),
        "dominant_contract_gap": (
            "no_compatible_anchor_available"
            if no_compatible >= compatible
            else "compatible_alternative_exists_but_selected_anchor_contract_unsafe"
        ),
        "contract_requirements": [
            "post-selection hard-safety prediction must cover every selected source",
            "fallback/rule/anchor/recent/hold_current selected actions must not bypass the contract",
            "if selected action predicts hard-safety rewrite, choose the best compatible candidate when available",
            "if no compatible candidate exists, emit no_compatible_anchor_available provenance",
            "do not mark fallback-only or no-compatible rows as clean qwen3.7 planning evidence",
        ],
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
        "audit_summary": {
            "evaluated_leak_count": int(audit.get("evaluated_leak_count", 0) or 0),
            "compatible_alternative_count": compatible,
            "no_compatible_anchor_available_count": no_compatible,
            "classification_counts": audit.get("classification_counts", {}),
        },
    }


def build_readiness(audit: Mapping[str, Any], design: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "artifact": "metadata_replay_readiness_checklist_20260601_v54",
        "current_stage": "v54_qwen37_planning_anchor_contract_fix",
        "model_name": MODEL_NAME,
        "v54_anchor_contract_audit_done": True,
        "v54_anchor_contract_audit_complete": bool(
            audit.get("all_v53_leaks_evaluated", False)
            and int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0) == 0
            and int(audit.get("unknown_requires_instrumentation_count", 0) or 0) == 0
        ),
        "v53_dominant_root_cause": audit.get("v53_dominant_root_cause"),
        "evaluated_leak_count": int(audit.get("evaluated_leak_count", 0) or 0),
        "expected_v53_step_aligned_leak_count": int(audit.get("expected_v53_step_aligned_leak_count", 0) or 0),
        "all_v53_leaks_evaluated": bool(audit.get("all_v53_leaks_evaluated", False)),
        "compatible_alternative_count": int(audit.get("compatible_alternative_count", 0) or 0),
        "no_compatible_anchor_available_count": int(audit.get("no_compatible_anchor_available_count", 0) or 0),
        "candidate_metadata_or_control_missing_count": int(
            audit.get("candidate_metadata_or_control_missing_count", 0) or 0
        ),
        "unknown_requires_instrumentation_count": int(audit.get("unknown_requires_instrumentation_count", 0) or 0),
        "contract_design_ready": bool(design.get("design_ready", False)),
        "controlled_replay_allowed": False,
        "controlled_replay_execution_allowed": False,
        "metadata_replay_execution_allowed": False,
        "performance_claim_allowed": False,
        "promotion_evidence": False,
        "next_action": design.get("recommended_next_action"),
        "stop_taxonomy": [
            item
            for item in (
                "metadata_missing"
                if int(audit.get("candidate_metadata_or_control_missing_count", 0) or 0)
                else "",
                "unknown_requires_instrumentation"
                if int(audit.get("unknown_requires_instrumentation_count", 0) or 0)
                else "",
                "no_compatible_anchor_available"
                if int(audit.get("no_compatible_anchor_available_count", 0) or 0)
                else "",
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
        "v53_dominant_root_cause",
        "dominant_contract_gap",
        "recommended_next_action",
        "next_action",
    ):
        if key in data:
            lines.append(f"- {key}: `{data.get(key)}`")
    for key in (
        "evaluated_leak_count",
        "expected_v53_step_aligned_leak_count",
        "compatible_alternative_count",
        "no_compatible_anchor_available_count",
        "candidate_metadata_or_control_missing_count",
        "unknown_requires_instrumentation_count",
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
    for key in ("classification_counts", "audit_summary", "contract_requirements", "stop_taxonomy"):
        if key in data:
            lines.extend(["", f"## {key}", "```json", json.dumps(data[key], indent=2, sort_keys=True), "```"])
    if "contract_rows" in data:
        lines.extend(
            [
                "",
                "## contract_rows_sample",
                "```json",
                json.dumps(list(data.get("contract_rows", []))[:20], indent=2, sort_keys=True),
                "```",
            ]
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_all(
    *,
    leak_audit_json: str | Path = V53_LEAK_AUDIT_JSON,
    attribution_json: str | Path = V53_ATTRIBUTION_JSON,
    v52_trace_dir: str | Path = V52_TRACE_DIR,
    scenarios: Sequence[str] = FAILURE_SCENARIOS,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    audit = build_anchor_contract_audit(
        leak_audit_json=leak_audit_json,
        attribution_json=attribution_json,
        v52_trace_dir=v52_trace_dir,
        scenarios=scenarios,
    )
    design = build_contract_design(audit)
    readiness = build_readiness(audit, design)
    _write_json_md(ANCHOR_AUDIT_JSON, ANCHOR_AUDIT_MD, audit, "v54 Qwen3.7 Planning Anchor Contract Audit")
    _write_json_md(
        CONTRACT_DESIGN_JSON,
        CONTRACT_DESIGN_MD,
        design,
        "v54 Qwen3.7 Post-Selection Hard-Safety Contract Design",
    )
    _write_json_md(READINESS_JSON, READINESS_MD, readiness, "v54 Metadata Replay Readiness")
    return audit, design, readiness


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leak-audit-json", default=str(V53_LEAK_AUDIT_JSON))
    parser.add_argument("--attribution-json", default=str(V53_ATTRIBUTION_JSON))
    parser.add_argument("--v52-trace-dir", default=str(V52_TRACE_DIR))
    parser.add_argument("--failure-scenario", action="append")
    parser.add_argument("--write-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    scenarios = args.failure_scenario or list(FAILURE_SCENARIOS)
    audit, design, readiness = write_all(
        leak_audit_json=args.leak_audit_json,
        attribution_json=args.attribution_json,
        v52_trace_dir=args.v52_trace_dir,
        scenarios=scenarios,
    )
    if not args.write_all:
        print(
            json.dumps(
                {
                    "evaluated_leak_count": audit.get("evaluated_leak_count"),
                    "compatible_alternative_count": audit.get("compatible_alternative_count"),
                    "no_compatible_anchor_available_count": audit.get("no_compatible_anchor_available_count"),
                    "next_action": readiness.get("next_action"),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
