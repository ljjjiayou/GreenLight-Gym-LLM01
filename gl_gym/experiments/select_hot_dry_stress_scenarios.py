"""Select focused hot-dry stress scenarios for scoring-preset validation.

The selector is read-only: it mines existing stress manifests and trace
directories, ranks scenarios by hot-dry pressure, and attaches plan-cache
coverage status.  It does not record new plans or call an online LLM.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gl_gym.experiments.check_plan_cache_coverage import audit_cache_coverage  # noqa: E402
TRACE_RE = re.compile(r"^(?P<scenario_id>y(?P<year>\d+)_d(?P<day>\d+)_s(?P<seed>\d+)_n(?P<max_steps>\d+))_(?P<controller>.+)$")


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def _read_json(path: str | Path) -> Any:
    raw_path = Path(path)
    if not raw_path.is_absolute():
        raw_path = PROJECT_ROOT / raw_path
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _trace_identity(path: Path) -> dict[str, Any] | None:
    match = TRACE_RE.match(path.stem)
    if not match:
        return None
    data = match.groupdict()
    return {
        "scenario_id": data["scenario_id"],
        "year": int(data["year"]),
        "day": int(data["day"]),
        "seed": int(data["seed"]),
        "max_steps": int(data["max_steps"]),
        "controller": data["controller"],
        "trace_path": str(path),
    }


def _rad(row: Mapping[str, Any]) -> float:
    return max(
        _num(row.get("rad_heat_load")),
        _num(row.get("glob_rad")),
        _num(row.get("forecast_rad_peak_2h")),
        _num(row.get("forecast_rad_mean_1h")),
    )


def _summarise_trace(path: str | Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    hot_dry_steps = 0
    radiation_steps = 0
    rh_low_area = 0.0
    vpd_high_area = 0.0
    temp_area = 0.0
    max_rad = 0.0
    max_vpd = 0.0
    max_temp = -999.0
    min_rh = 999.0
    min_dew = 999.0
    min_canopy = 999.0
    for row in rows:
        temp = _num(row.get("temp_air"), 20.0)
        rh = _num(row.get("rh_air"), 70.0)
        vpd = _num(row.get("vpd_air"), _num(row.get("vpd_kpa")))
        rad = _rad(row)
        max_rad = max(max_rad, rad)
        max_vpd = max(max_vpd, vpd)
        max_temp = max(max_temp, temp)
        min_rh = min(min_rh, rh)
        min_dew = min(min_dew, _num(row.get("dew_margin_air"), 99.0))
        min_canopy = min(min_canopy, _num(row.get("canopy_dew_margin"), 99.0))
        rh_low_area += _num(row.get("rh_low_violation"))
        vpd_high_area += _num(row.get("vpd_high_excess"))
        temp_area += _num(row.get("temp_violation"))
        if rad >= 650.0:
            radiation_steps += 1
        if temp >= 30.0 and rad >= 600.0 and (rh <= 55.0 or vpd >= 1.6):
            hot_dry_steps += 1
    return {
        "steps": len(rows),
        "regime_step_counts": {"hot_dry": hot_dry_steps, "radiation_spike": radiation_steps},
        "rh_low_area": rh_low_area,
        "vpd_high_area": vpd_high_area,
        "temp_violation_area": temp_area,
        "max_rad_heat_load": max_rad,
        "max_vpd_air": max_vpd,
        "max_temp_air": max_temp,
        "min_rh_air": min_rh,
        "min_dew_margin": min_dew,
        "min_canopy_dew_margin": min_canopy,
    }


def _pressure_score(summary: Mapping[str, Any]) -> float:
    counts = summary.get("regime_step_counts", {}) if isinstance(summary.get("regime_step_counts"), Mapping) else {}
    hot_dry_steps = _num(counts.get("hot_dry"))
    radiation_steps = _num(counts.get("radiation_spike"))
    return (
        hot_dry_steps * 5.0
        + _num(summary.get("vpd_high_area")) * 1.0
        + _num(summary.get("rh_low_area")) * 0.45
        + radiation_steps * 0.35
        + _num(summary.get("max_rad_heat_load")) * 0.01
        + _num(summary.get("max_vpd_air")) * 5.0
    )


def _regime_conflict_tags(summary: Mapping[str, Any]) -> list[str]:
    counts = summary.get("regime_step_counts", {}) if isinstance(summary.get("regime_step_counts"), Mapping) else {}
    hot_dry_steps = _num(counts.get("hot_dry"))
    radiation_steps = _num(counts.get("radiation_spike"))
    dry_pressure = (
        _num(summary.get("rh_low_area")) > 0.0
        or _num(summary.get("vpd_high_area")) > 0.0
        or _num(summary.get("min_rh_air"), 99.0) <= 60.0
        or _num(summary.get("max_vpd_air")) >= 1.55
    )
    dew_near = _num(summary.get("min_dew_margin"), 99.0) < 1.0 or _num(summary.get("min_canopy_dew_margin"), 99.0) < 1.0
    dew_hard = _num(summary.get("min_dew_margin"), 99.0) < 0.0 or _num(summary.get("min_canopy_dew_margin"), 99.0) < 0.0
    tags: list[str] = []
    if hot_dry_steps > 0 and not dew_hard:
        tags.append("pure_hot_dry")
    if dry_pressure:
        tags.append("dry_pressure")
    if radiation_steps > 0 and dry_pressure:
        tags.append("radiation_dry")
    if dry_pressure and dew_near:
        tags.append("hot_dry_dew_conflict")
    if dew_hard:
        tags.append("preexisting_dew_hard_risk")
    return tags


def _scenario_from_manifest(raw: Mapping[str, Any]) -> dict[str, Any]:
    risk = raw.get("risk_summary", {}) if isinstance(raw.get("risk_summary"), Mapping) else {}
    return {
        "scenario_id": str(raw.get("scenario_id", "")),
        "year": int(raw.get("year", 0)),
        "day": int(raw.get("day", 0)),
        "seed": int(raw.get("seed", 0)),
        "max_steps": int(raw.get("max_steps", 240)),
        "source": "manifest",
        "split": str(raw.get("split", "")),
        "regime_tags": list(raw.get("regime_tags", []) or []),
        "risk_summary": dict(risk),
    }


def _candidate_key(item: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (int(item["year"]), int(item["day"]), int(item["seed"]), int(item.get("max_steps", 240)))


def load_candidates(manifest_path: str | Path, trace_dirs: Sequence[str | Path]) -> list[dict[str, Any]]:
    candidates: dict[tuple[int, int, int, int], dict[str, Any]] = {}
    manifest = _read_json(manifest_path)
    for raw in manifest.get("scenarios", []) if isinstance(manifest, Mapping) else []:
        if not isinstance(raw, Mapping):
            continue
        item = _scenario_from_manifest(raw)
        candidates[_candidate_key(item)] = item
    for raw_dir in trace_dirs:
        root = Path(raw_dir)
        if not root.is_absolute():
            root = PROJECT_ROOT / root
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            ident = _trace_identity(path)
            if not ident:
                continue
            summary = _summarise_trace(path)
            key = _candidate_key(ident)
            existing = candidates.get(key)
            if existing:
                merged = dict(existing)
                merged["source"] = f"{existing.get('source', 'manifest')}+trace"
                merged["trace_path"] = str(path)
                merged["risk_summary"] = {**dict(existing.get("risk_summary", {})), **summary}
                tags = set(str(tag) for tag in existing.get("regime_tags", []) or [])
                if summary["regime_step_counts"]["hot_dry"] > 0:
                    tags.add("hot_dry")
                if summary["regime_step_counts"]["radiation_spike"] > 0:
                    tags.add("radiation_spike")
                merged["regime_tags"] = sorted(tags)
                candidates[key] = merged
            else:
                candidates[key] = {
                    "scenario_id": ident["scenario_id"],
                    "year": ident["year"],
                    "day": ident["day"],
                    "seed": ident["seed"],
                    "max_steps": ident["max_steps"],
                    "source": "trace",
                    "split": "",
                    "regime_tags": ["hot_dry"] if summary["regime_step_counts"]["hot_dry"] > 0 else [],
                    "trace_path": str(path),
                    "risk_summary": summary,
                }
    out = []
    for item in candidates.values():
        risk = item.get("risk_summary", {}) if isinstance(item.get("risk_summary"), Mapping) else {}
        item = dict(item)
        item["hot_dry_pressure_score"] = _pressure_score(risk)
        item["hard_risk_flags"] = {
            "dew_lt0": _num(risk.get("min_dew_margin"), 99.0) < 0.0,
            "canopy_lt0": _num(risk.get("min_canopy_dew_margin"), 99.0) < 0.0,
            "temp_ge32": _num(risk.get("max_temp_air")) >= 32.0,
        }
        item["regime_conflict_tags"] = _regime_conflict_tags(risk)
        out.append(item)
    return sorted(out, key=lambda x: (-_num(x.get("hot_dry_pressure_score")), int(x["year"]), int(x["day"]), int(x["seed"])))


def select_hot_dry_scenarios(
    *,
    manifest_path: str | Path,
    trace_dirs: Sequence[str | Path],
    plan_cache_path: str | Path,
    min_candidates: int = 12,
    max_candidates: int = 18,
) -> dict[str, Any]:
    all_candidates = load_candidates(manifest_path, trace_dirs)
    pressure_candidates = [
        item
        for item in all_candidates
        if _num(item.get("hot_dry_pressure_score")) > 0.0
        and (
            "hot_dry" in {str(tag) for tag in item.get("regime_tags", [])}
            or _num(item.get("risk_summary", {}).get("vpd_high_area")) > 0.0
            or _num(item.get("risk_summary", {}).get("rh_low_area")) > 0.0
        )
    ]
    selected = pressure_candidates[: max(0, int(max_candidates))]
    coverage = audit_cache_coverage(
        plan_cache_path=plan_cache_path,
        years=[],
        days=[],
        seeds=[],
        max_steps=240,
        control_interval=12,
        scenario_specs=selected,
    )
    env_coverage = coverage.get("envs", {}) if isinstance(coverage.get("envs"), Mapping) else {}
    executable: list[dict[str, Any]] = []
    covered_selected: list[dict[str, Any]] = []
    for item in selected:
        env_id = f"TomatoEnv_y{item['year']}_d{item['day']}_s{item['seed']}"
        item = dict(item)
        item["plan_cache_env_id"] = env_id
        item["plan_cache_ok"] = bool(isinstance(env_coverage.get(env_id), Mapping) and env_coverage[env_id].get("ok"))
        item["plan_cache_issues"] = list(env_coverage.get(env_id, {}).get("issues", [])) if isinstance(env_coverage.get(env_id), Mapping) else ["missing_env"]
        covered_selected.append(item)
        if item["plan_cache_ok"]:
            executable.append(item)
    can_execute = bool(coverage.get("can_strict_replay")) and len(executable) >= int(min_candidates)
    return {
        "schema_version": "selected_hot_dry_stress_suite_v1",
        "manifest_path": str(manifest_path),
        "trace_dirs": [str(path) for path in trace_dirs],
        "plan_cache_path": str(plan_cache_path),
        "min_candidates": int(min_candidates),
        "max_candidates": int(max_candidates),
        "all_candidate_count": len(all_candidates),
        "pressure_candidate_count": len(pressure_candidates),
        "selected_count": len(selected),
        "executable_count": len(executable),
        "can_execute_strict_replay": can_execute,
        "selected_scenarios": covered_selected,
        "executable_scenarios": executable,
        "coverage": coverage,
    }


def build_markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# Selected Hot-Dry Stress Suite",
        "",
        f"- Selected: {report.get('selected_count', 0)}",
        f"- Executable: {report.get('executable_count', 0)}",
        f"- Can execute strict replay: **{str(report.get('can_execute_strict_replay')).upper()}**",
        "",
        "| scenario | tags | split | source | score | hot_dry_steps | RHlow | VPDhi | max_rad | max_vpd | hard_risk | cache_ok |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in report.get("selected_scenarios", []) or []:
        risk = item.get("risk_summary", {}) if isinstance(item.get("risk_summary"), Mapping) else {}
        counts = risk.get("regime_step_counts", {}) if isinstance(risk.get("regime_step_counts"), Mapping) else {}
        flags = item.get("hard_risk_flags", {}) if isinstance(item.get("hard_risk_flags"), Mapping) else {}
        hard = ",".join(k for k, v in flags.items() if v) or "-"
        conflict_tags = ",".join(str(tag) for tag in item.get("regime_conflict_tags", []) or []) or "-"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("scenario_id", "")),
                    conflict_tags,
                    str(item.get("split", "")),
                    str(item.get("source", "")),
                    f"{_num(item.get('hot_dry_pressure_score')):.3f}",
                    f"{_num(counts.get('hot_dry')):.0f}",
                    f"{_num(risk.get('rh_low_area')):.3f}",
                    f"{_num(risk.get('vpd_high_area')):.3f}",
                    f"{_num(risk.get('max_rad_heat_load')):.3f}",
                    f"{_num(risk.get('max_vpd_air')):.3f}",
                    hard,
                    str(item.get("plan_cache_ok", "")),
                ]
            )
            + " |"
        )
    tag_counts: dict[str, int] = {}
    for item in report.get("selected_scenarios", []) or []:
        for tag in item.get("regime_conflict_tags", []) or []:
            tag_counts[str(tag)] = tag_counts.get(str(tag), 0) + 1
    lines.extend(["", "## Regime Conflict Tag Counts", ""])
    lines.append("| tag | count |")
    lines.append("| --- | ---: |")
    for tag, count in sorted(tag_counts.items()):
        lines.append(f"| {tag} | {count} |")
    return "\n".join(lines) + "\n"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="gl_gym/result/stress_suites/regime_stress_v1.json")
    parser.add_argument("--trace-dir", action="append", default=[])
    parser.add_argument("--plan-cache-path", default="gl_gym/result/plan_cache/holdout_seed42_43_hot_dry_qwen_20260517_merged.json")
    parser.add_argument("--min-candidates", type=int, default=12)
    parser.add_argument("--max-candidates", type=int, default=18)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = select_hot_dry_scenarios(
        manifest_path=args.manifest,
        trace_dirs=args.trace_dir,
        plan_cache_path=args.plan_cache_path,
        min_candidates=args.min_candidates,
        max_candidates=args.max_candidates,
    )
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    if not output_json.is_absolute():
        output_json = PROJECT_ROOT / output_json
    if not output_md.is_absolute():
        output_md = PROJECT_ROOT / output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")
    print(
        f"selected={report['selected_count']} executable={report['executable_count']} "
        f"can_execute={report['can_execute_strict_replay']}"
    )
    print(f"wrote {output_json}")
    print(f"wrote {output_md}")
    return 0 if report.get("can_execute_strict_replay") else 1


if __name__ == "__main__":
    raise SystemExit(main())
