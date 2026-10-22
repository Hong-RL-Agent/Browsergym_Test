from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "DQN"))
from envs.browsergym_jaws_env import BrowserGymJAWSEnv, _active_page  # noqa: E402


def main() -> int:
    env = BrowserGymJAWSEnv(
        site_id="B05_상품검색비교",
        base_url="http://127.0.0.1:5320",
        max_steps=40,
        max_candidates=100,
        headless=True,
        site_profile={},
    )
    try:
        obs, _ = env.reset()
        steps = [("laptop", 19), ("checkbox1", 53), ("checkbox2", 54)]
        for label, index in steps:
            obs, info, done, _ = env.step(env.action_space.encode("click_element", index))
            page = _active_page(env.env)
            state = page.evaluate("() => window.__compareLabState || null")
            checks = page.evaluate("""() => Array.from(document.querySelectorAll('input[type=checkbox]')).map((x) => ({checked:x.checked, outer:x.outerHTML.slice(0,220)}))""")
            print(json.dumps({"step": label, "info": info, "state": state, "checks": checks, "done": done}, ensure_ascii=False))
        for j in range(3):
            obs, info, done, _ = env.step(env.action_space.encode("scroll_down", 0))
            page = _active_page(env.env)
            visible = page.evaluate("""() => Array.from(document.querySelectorAll('input[type=checkbox]')).filter((x) => x.getAttribute('browsergym_visibility_ratio') === '1').map((x) => x.getAttribute('bid'))""")
            print(json.dumps({"scroll": j, "visible_checkbox_bids": visible, "candidate_tail": [(i, c.get("text"), c.get("bid"), c.get("visible")) for i, c in list(enumerate(obs.get("candidate_elements", [])))[-40:]]}, ensure_ascii=False))
        page = _active_page(env.env)
        visible_inputs = page.evaluate("""() => Array.from(document.querySelectorAll('input[type=checkbox]')).filter((x) => x.getAttribute('browsergym_visibility_ratio') === '1').map((x) => Number(x.getAttribute('bid')) - 1)""")
        print(json.dumps({"visible_input_details": page.evaluate("""() => Array.from(document.querySelectorAll('input[type=checkbox]')).filter((x) => x.getAttribute('browsergym_visibility_ratio') === '1').map((x) => ({bid:x.getAttribute('bid'), parent:x.parentElement ? x.parentElement.outerHTML.slice(0,700) : ''}))""")}, ensure_ascii=False))
        for desired_bid in visible_inputs[:2]:
            candidates = obs.get("candidate_elements", [])
            matches = [i for i, c in enumerate(candidates) if str(c.get("bid")) == str(desired_bid) and c.get("visible", True)]
            print(json.dumps({"desired_bid": desired_bid, "matches": matches}, ensure_ascii=False))
            if matches:
                obs, info, done, _ = env.step(env.action_space.encode("click_element", matches[0]))
                page = _active_page(env.env)
                print(json.dumps({"clicked_bid": desired_bid, "state": page.evaluate("() => window.__compareLabState || null")}, ensure_ascii=False))
        page = _active_page(env.env)
        print(json.dumps(page.evaluate("() => ({url:location.href, text:document.body.innerText.slice(0,4000)})"), ensure_ascii=False))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
