"""Coverage metrics over stable state/action/transition graphs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

def graph_from_transitions(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    states, actions, transitions, routes = set(), set(), set(), set()
    for row in rows:
        states.update(value for value in (row.get("state_id_before"), row.get("state_id_after")) if value)
        if row.get("action_id"): actions.add(str(row["action_id"]))
        if row.get("transition_id"): transitions.add(str(row["transition_id"]))
        for key in ("url_before", "url_after"):
            if row.get(key): routes.add(str(row[key]))
    return {"states": sorted(states), "actions": sorted(actions), "transitions": sorted(transitions), "routes": sorted(routes)}

def compare_coverage(actual: Mapping[str, Any], reference: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if reference is None:
        return {
            "reference_available": False,
            "coverage_score": None,
            "visited_states": len(set(actual.get("states", []) or [])),
            "visited_actions": len(set(actual.get("actions", []) or [])),
            "visited_transitions": len(set(actual.get("transitions", []) or [])),
            "visited_routes": len(set(actual.get("routes", []) or [])),
            "novel_states": None,
        }
    metrics, weighted = {}, 0.0
    weights = {"states": 0.30, "actions": 0.25, "transitions": 0.20, "routes": 0.15}
    for key, weight in weights.items():
        actual_values, reference_values = set(actual.get(key, []) or []), set(reference.get(key, []) or [])
        ratio = len(actual_values & reference_values) / max(1, len(reference_values))
        metrics[f"{key[:-1]}_coverage"] = round(ratio, 4)
        metrics[f"visited_{key}"] = len(actual_values)
        metrics[f"reference_{key}"] = len(reference_values)
        weighted += weight * ratio
    metrics["reference_available"] = True
    metrics["coverage_score"] = round((weighted / sum(weights.values())) * 100, 2)
    metrics["novel_states"] = len(set(actual.get("states", [])) - set(reference.get("states", [])))
    return metrics

def load_graph(path: str | Path | None) -> dict[str, Any] | None:
    if not path: return None
    candidate = Path(path)
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.exists() else None
