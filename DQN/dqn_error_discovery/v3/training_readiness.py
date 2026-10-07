"""Run v3 wiring/readiness checks without loading or changing v2 checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.ppo_agent import PPOAgent
from dqn_error_discovery.agents.dqn_agent import FactorizedQNetwork
from dqn_error_discovery.hybrid.hybrid_controller import AgentMode, HybridController
from envs.browsergym_jaws_env import BrowserGymJAWSEnv

from .action_contract import is_fill_input_target, is_select_option_target
from .action_space import V3ActionSpace
from .dimensions import v3_dimensions
from .select_option_executor import execute_select_option


ROOT = Path(__file__).resolve().parents[2]
SITES = {
    "SITE001": "http://localhost:9220/",
    "DEV_REACT": "http://localhost:9225/",
    "DEV_NEXT": "http://localhost:9226/",
}


class _FakeOption:
    def __init__(self, value: str, text: str, disabled: bool = False) -> None:
        self.value, self.text, self.disabled = value, text, disabled

    def get_attribute(self, name: str) -> str | None:
        if name == "value":
            return self.value
        if name == "disabled" and self.disabled:
            return "disabled"
        return None

    def text_content(self) -> str:
        return self.text


class _FakeOptionList:
    def __init__(self, options: list[_FakeOption]) -> None:
        self.options = options

    def count(self) -> int:
        return len(self.options)

    def nth(self, index: int) -> _FakeOption:
        return self.options[index]


class _FakeSelect:
    def __init__(self) -> None:
        self.value = ""
        self.selected: str | None = None
        self._options = _FakeOptionList([_FakeOption("", ""), _FakeOption("a", "A"), _FakeOption("b", "B")])

    def input_value(self) -> str:
        return self.value

    def locator(self, selector: str) -> _FakeOptionList:
        assert selector == "option"
        return self._options

    def select_option(self, *, value: str) -> None:
        self.value = value
        self.selected = value


class _FakePage:
    def __init__(self) -> None:
        self.select = _FakeSelect()

    def locator(self, selector: str) -> _FakeSelect:
        assert selector == '[bid="select-1"]'
        return self.select


def _candidate(tag: str, role: str = "", **extra: Any) -> dict[str, Any]:
    return {"bid": f"{tag}-1", "tag": tag, "role": role, "bbox": [0, 0, 100, 24], "visible": True, "enabled": True, **extra}


def run_readiness(*, output: Path | None = None, live_sites: bool = True) -> dict[str, Any]:
    space = V3ActionSpace()
    dims = v3_dimensions(space.max_candidates)
    select = _candidate("select", "combobox")
    text_input = _candidate("input", "textbox", input_type="text", fillable=True)
    custom_combo = _candidate("div", "combobox", clickable=True)
    observation = {"candidate_elements": [select, text_input, custom_combo], "page_state": {"url": "http://localhost:9225/"}, "history": {"step_index": 6}}
    mask = space.build_action_mask(observation)
    select_id = space.encode("select_option", 0)
    fill_select_id = space.encode("fill_input", 0)
    fill_text_id = space.encode("fill_input", 1)
    fake_page = _FakePage()
    selection_result = execute_select_option(fake_page, {**select, "bid": "select-1"})

    ppo = PPOAgent(dims["ppo_obs_dim"], dims["action_dim"])
    ppo_action = ppo.select_greedy_action(np.zeros(dims["ppo_obs_dim"], dtype=np.float32), np.ones(dims["action_dim"], dtype=np.float32))
    dqn = FactorizedQNetwork(dims["dqn_obs_dim"], dims["action_type_count"], dims["candidate_count"])
    dqn_output = dqn(torch.zeros((1, dims["dqn_obs_dim"]), dtype=torch.float32))
    optimizer = torch.optim.Adam(dqn.parameters(), lr=1e-4)
    loss = sum(value.square().mean() for value in dqn_output.values())
    loss.backward()
    optimizer.step()

    controller = HybridController(max_local_steps=5, cooldown_ppo_steps=1, functional_region_novelty_gate_enabled=True)
    controller.record_agent_step(AgentMode.PPO_GLOBAL)
    handoff = controller.maybe_handoff_after_ppo_step(
        step=1, observation={"page_state": {"url": "http://localhost:9225/"}, "candidate_elements": [{"is_form_field": True, "role": "textbox", "tag": "input"}]},
        anomalies=[], reward_components={}, confirmation_metadata={}, history={},
    )
    controller.record_agent_step(AgentMode.DQN_LOCAL)
    returned = controller.maybe_return_after_dqn_step(
        step=2, observation={"page_state": {"url": "http://localhost:9225/"}, "candidate_elements": []},
        anomalies=[], reward_components={}, repeat_diagnostic={}, done=True,
    )

    report: dict[str, Any] = {
        "version": "RAWD_V3",
        "dimensions": dims,
        "action_types": list(space.action_types),
        "contract": {
            "native_select_fill_input_valid": bool(mask[fill_select_id]),
            "native_select_select_option_valid": bool(mask[select_id]),
            "text_input_fill_input_valid": bool(mask[fill_text_id]),
            "custom_combobox_fill_input_valid": bool(space.build_action_mask({"candidate_elements": [custom_combo]})[space.encode("fill_input", 0)]),
            "select_executor_selected": selection_result["selected_option"],
        },
        "ppo_one_step": {"obs_dim": dims["ppo_obs_dim"], "output_dim": len(ppo_action) and dims["action_dim"]},
        "dqn_one_step": {"obs_dim": dims["dqn_obs_dim"], "type_head": int(dqn_output["type_values"].shape[-1]), "candidate_head": list(dqn_output["candidate_advantages"].shape[1:])},
        "hybrid_wiring_smoke": {"handoff": handoff, "returned": returned, "ppo_mode_after_return": controller.active_agent.value == "PPO_GLOBAL"},
        "live_site_smoke": {},
        "gt_used": False,
    }
    if live_sites:
        for site_id, url in SITES.items():
            env = BrowserGymJAWSEnv(site_id=site_id.lower(), base_url=url, max_steps=1, headless=True)
            try:
                obs, _ = env.reset()
                live_mask = space.build_action_mask(obs)
                report["live_site_smoke"][site_id] = {"reset": True, "ppo_obs_dim": 635, "mask_dim": int(live_mask.size), "candidate_count": len(obs.get("candidate_elements", []))}
            except Exception as exc:
                report["live_site_smoke"][site_id] = {"reset": False, "error": type(exc).__name__ + ": " + str(exc)}
            finally:
                env.close()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


if __name__ == "__main__":
    path = ROOT / "artifacts/models/rawd_v3_training_readiness.json"
    print(json.dumps(run_readiness(output=path), ensure_ascii=False, indent=2, default=str))
