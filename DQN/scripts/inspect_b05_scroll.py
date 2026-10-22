
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))

from envs.browsergym_jaws_env import BrowserGymJAWSEnv, _active_page


def main():
    env = BrowserGymJAWSEnv(
        site_id="B05_상품검색비교",
        base_url="http://127.0.0.1:5320",
        max_steps=20,
        max_candidates=300,
        headless=True,
        site_profile={},
    )

    try:
        observation, _ = env.reset()

        observation, _, _, _ = env.step(
            env.action_space.encode("click_element", 19)
        )

        for _ in range(3):
            observation, _, _, _ = env.step(
                env.action_space.encode("scroll_down", 0)
            )

        page = _active_page(env.env)

        details = page.evaluate("""
            () => Array.from(
                document.querySelectorAll('input[type=checkbox]')
            )
            .filter(
                x => x.getAttribute('browsergym_visibility_ratio') === '1'
            )
            .map(x => ({
                bid: x.getAttribute('bid'),
                parent: x.parentElement?.outerHTML.slice(0, 500)
            }))
        """)

        spec_labels = page.evaluate("""
            () => Array.from(
                document.querySelectorAll('label')
            )
            .filter(x => /32GB|256GB|Core i5/.test(x.innerText))
            .map(x => ({
                text: x.innerText,
                bid: x.getAttribute('bid'),
                html: x.outerHTML.slice(0, 500)
            }))
        """)

        candidates = [
            (
                i,
                candidate.get("bid"),
                candidate.get("text"),
                candidate.get("role"),
                candidate.get("visible"),
            )
            for i, candidate in enumerate(
                observation.get("candidate_elements", [])
            )
        ]

        result = {
            "details": details,
            "spec_labels": spec_labels,
            "candidates": candidates,
        }

        print(json.dumps(result, ensure_ascii=False))

    finally:
        env.close()


if __name__ == "__main__":
    main()
