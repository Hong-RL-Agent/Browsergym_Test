
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv
from services.anomaly_detection_service import detect_anomalies


def main():
    parser = argparse.ArgumentParser(
        description="Debug BrowserGym action and anomaly detection."
    )
    parser.add_argument("--site-id", default="B01_SNS")
    parser.add_argument("--base-url", default="http://127.0.0.1:5316")
    parser.add_argument("--candidate-index", type=int, default=5)
    args = parser.parse_args()

    env = BrowserGymJAWSEnv(
        site_id=args.site_id,
        base_url=args.base_url,
        max_steps=10,
        max_candidates=100,
        headless=True,
        site_profile={},
    )

    try:
        before, _ = env.reset()

        candidates = before.get("candidate_elements", [])
        idx = args.candidate_index

        if not 0 <= idx < len(candidates):
            raise ValueError(
                f"Invalid candidate index {idx}. "
                f"Available candidates: {len(candidates)}"
            )

        action_id = env.action_space.encode("click_element", idx)
        after, reward, done, info = env.step(action_id)

        action_error = (
            bool(info.get("action_error"))
            if isinstance(info, dict)
            else False
        )

        target_identity = str(candidates[idx].get("bid"))

        action_context = {
            "action": {
                "action_type": "click_element",
                "candidate_index": idx,
                "action_id": action_id,
                "execution_success": not action_error,
                "target_identity": target_identity,
            },
            "action_type": "click_element",
            "trigger_action_type": "click_element",
            "trigger_target_identity": target_identity,
            "site_id": args.site_id,
            "last_action_error": action_error,
        }

        anomalies = detect_anomalies(
            before,
            after,
            action_context,
            site_profile={},
        )

        result = {
            "reward": reward,
            "info": info,
            "before_url": before.get("page_state", {}).get("url"),
            "after_url": after.get("page_state", {}).get("url"),
            "anomalies": anomalies,
        }

        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))

    finally:
        env.close()


if __name__ == "__main__":
    main()
