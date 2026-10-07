"""BrowserGym wrapper that exposes J.A.W.S raw observations and actions."""

from __future__ import annotations

from typing import Any, Dict, Optional

from adapters.browsergym_action_adapter import BrowserGymActionAdapter
from adapters.browsergym_observation_adapter import BrowserGymObservationAdapter
from models.action_space import ActionSpace
from models.observation_encoder import ObservationEncoder


class BrowserGymJAWSEnv:
    def __init__(
        self,
        site_id: Optional[str] = None,
        base_url: str = "",
        max_steps: int = 20,
        max_candidates: int = 32,
        headless: bool = True,
    ) -> None:
        self.site_id = site_id
        if not base_url:
            raise ValueError("BrowserGymJAWSEnv requires an explicit base_url.")
        self.base_url = base_url
        self.max_steps = max_steps
        self.max_candidates = max_candidates
        self.headless = headless

        self.observation_adapter = BrowserGymObservationAdapter(max_candidates=max_candidates)
        self.action_adapter = BrowserGymActionAdapter()
        self.action_space = ActionSpace(max_candidates=max_candidates)
        self.observation_encoder = ObservationEncoder(max_candidates=max_candidates)

        self.env = None
        self.step_index = 0
        self.no_change_steps = 0
        self.previous_action_type = "noop"
        self._browser_obs: Optional[Dict[str, Any]] = None
        self._jaws_obs: Optional[Dict[str, Any]] = None

    def reset(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        gym = _import_browsergym()
        if self.env is not None:
            self.close()

        self.env = gym.make(
            "browsergym/openended",
            task_kwargs={"start_url": self.base_url},
            wait_for_user_message=False,
            headless=self.headless,
        )
        browser_obs, info = self.env.reset()
        self.step_index = 0
        self.no_change_steps = 0
        self.previous_action_type = "noop"
        self._browser_obs = browser_obs
        self._jaws_obs = self.observation_adapter.adapt(
            browser_obs,
            info,
            previous_obs=None,
            history=self._history(),
        )
        return self._jaws_obs, self._info(info, self.previous_action_type)

    def step(self, action_id: int) -> tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.env is None or self._jaws_obs is None:
            raise RuntimeError("Call reset() before step().")

        decoded_action = self.action_space.decode(action_id)
        browser_action = self.action_adapter.adapt(decoded_action, self._jaws_obs)
        action_error = False
        done = False
        info: Dict[str, Any] = {}
        browser_obs = self._browser_obs

        if browser_action.action_type == "finish_episode":
            done = True
        elif browser_action.executable:
            try:
                browser_obs, _, terminated, truncated, info = _normalize_step_result(
                    self.env.step(browser_action.action)
                )
                done = bool(terminated or truncated)
            except Exception as exc:
                action_error = True
                info = {"action_error": str(exc)}
                if browser_action.action != browser_action.fallback_action:
                    try:
                        browser_obs, _, terminated, truncated, fallback_info = _normalize_step_result(
                            self.env.step(browser_action.fallback_action)
                        )
                        done = bool(terminated or truncated)
                        info["fallback_info"] = fallback_info
                    except Exception as fallback_exc:
                        info["fallback_error"] = str(fallback_exc)
        else:
            info = {"internal_action": browser_action.action_type}

        self.step_index += 1
        if self.step_index >= self.max_steps:
            done = True

        previous_jaws_obs = self._jaws_obs
        next_jaws_obs = self.observation_adapter.adapt(
            browser_obs,
            info,
            previous_obs=previous_jaws_obs,
            history=self._next_history(browser_action.action_type, previous_jaws_obs, browser_obs),
        )

        self._browser_obs = browser_obs
        self._jaws_obs = next_jaws_obs
        self.previous_action_type = browser_action.action_type

        step_info = self._info(info, browser_action.action_type)
        step_info.update(
            {
                "action": decoded_action,
                "browsergym_action": browser_action.action,
                "last_action_error": action_error or bool(next_jaws_obs["runtime_signals"]["last_action_error"]),
                "candidate_count": len(next_jaws_obs.get("candidate_elements", [])),
            }
        )
        return next_jaws_obs, 0.0, done, step_info

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None

    def _history(self) -> Dict[str, Any]:
        return {
            "step_index": self.step_index,
            "previous_action_type": self.previous_action_type,
            "no_change_steps": self.no_change_steps,
        }

    def _next_history(
        self,
        action_type: str,
        previous_jaws_obs: Dict[str, Any],
        browser_obs: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        previous_url = previous_jaws_obs.get("page_state", {}).get("url")
        next_url = ""
        if isinstance(browser_obs, dict):
            next_url = str(browser_obs.get("url") or "")
        if previous_url and next_url and previous_url == next_url:
            self.no_change_steps += 1
        else:
            self.no_change_steps = 0
        return {
            "step_index": self.step_index,
            "previous_action_type": action_type,
            "no_change_steps": self.no_change_steps,
        }

    def _info(self, info: Dict[str, Any], action_type: str) -> Dict[str, Any]:
        merged = dict(info or {})
        merged.setdefault("site_id", self.site_id)
        merged.setdefault("action_type", action_type)
        merged.setdefault("candidate_count", len(self._jaws_obs.get("candidate_elements", [])) if self._jaws_obs else 0)
        return merged


def _import_browsergym():
    try:
        import browsergym.core  # noqa: F401
        import gymnasium as gym
    except ImportError as exc:
        raise ImportError(
            "BrowserGymJAWSEnv requires gymnasium and browsergym. "
            "Install BrowserGym before running the environment wrapper."
        ) from exc
    return gym


def _normalize_step_result(result: Any) -> tuple[Any, float, bool, bool, Dict[str, Any]]:
    if not isinstance(result, tuple):
        raise TypeError("env.step() returned a non-tuple result")
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward or 0.0), bool(terminated), bool(truncated), dict(info or {})
    if len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward or 0.0), bool(done), False, dict(info or {})
    raise ValueError(f"Unsupported env.step() result length: {len(result)}")
