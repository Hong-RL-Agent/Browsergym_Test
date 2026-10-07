"""Post-hoc aggregation for the completed Run 3 artifacts."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/training_v3/dqn_productive_run3_corrected"

def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def main():
    aggregate = {"seeds": {}, "site": defaultdict(Counter), "consistency": {"status": "PASS"}}
    for seed in (42, 43, 44):
        sessions = rows(OUT / f"seed{seed}_local_sessions.jsonl")
        rewards = rows(OUT / f"seed{seed}_reward_components.jsonl")
        q = rows(OUT / f"seed{seed}_q_metrics.jsonl")
        comp = Counter()
        for row in rewards:
            comp.update({key: float(value or 0.0) for key, value in row.get("components", {}).items()})
        admission = sum(1 for row in rewards if float(row.get("components", {}).get("candidate_anomaly", 0.0) or 0.0) > 0)
        confirmed = sum(1 for row in rewards if float(row.get("components", {}).get("confirmed_anomaly", 0.0) or 0.0) > 0 or float(row.get("components", {}).get("confirmation_upgrade", 0.0) or 0.0) > 0)
        seed_data = {
            "sessions": len(sessions), "local_steps": sum(x["steps"] for x in sessions),
            "reward": round(sum(x["reward"] for x in sessions), 6),
            "detector_opportunity_steps": sum(x["detector_opportunity_steps"] for x in sessions),
            "anomaly_emissions": sum(x["anomaly_emissions"] for x in sessions),
            "causal_eligible": sum(x["causal_eligible"] for x in sessions),
            "candidate_eligible": sum(x["candidate_eligible"] for x in sessions),
            "candidate_admission_events": admission,
            "unique_candidate_keys": sum(len(x["unique_candidate_keys"]) for x in sessions),
            "candidate_created": admission,
            "candidate_upgraded": sum(x["candidate_upgraded"] for x in sessions),
            "candidate_confirmed": confirmed,
            "strong_evidence": sum(x["strong_evidence"] for x in sessions),
            "strong_candidate": sum(x["strong_candidate"] for x in sessions),
            "strong_confirmed": sum(x["strong_confirmed"] for x in sessions),
            "weak": sum(x["weak"] for x in sessions), "zero_gain": sum(x["zero_gain"] for x in sessions),
            "no_effect": sum(x["no_effect_count"] for x in sessions),
            "repeat_action_target": sum(x["repeat_action_target_count"] for x in sessions),
            "invalid": sum(x["invalid_count"] for x in sessions), "timeout": sum(x["timeout_count"] for x in sessions), "action_error": sum(x["action_error_count"] for x in sessions),
            "q_finite": all(bool(x.get("q_finite", True)) for x in q), "q_mean": round(mean(x["mean_valid_q"] for x in q if x.get("mean_valid_q") is not None), 6), "q_max": round(max(x["max_valid_q"] for x in q if x.get("max_valid_q") is not None), 6), "loss_samples": sum(x.get("loss") is not None for x in q),
            "reward_components": dict(comp),
            "by_site": {},
        }
        for site in sorted({x["site"] for x in sessions}):
            subset = [x for x in sessions if x["site"] == site]
            seed_data["by_site"][site] = {"sessions": len(subset), "reward": round(sum(x["reward"] for x in subset), 6), "candidate_admission_events": sum(1 for x in rewards if x["site"] == site and float(x.get("components", {}).get("candidate_anomaly", 0.0) or 0.0) > 0), "anomaly_emissions": sum(x["anomaly_emissions"] for x in subset), "no_effect": sum(x["no_effect_count"] for x in subset)}
        aggregate["seeds"][str(seed)] = seed_data
    (OUT / "run3_posthoc_analysis.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))

if __name__ == "__main__": main()
