"""Confirmation reproduction semantics aligned with BrowserGym actions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
from urllib.parse import urlparse


ACTION_REPRODUCTION = "ACTION_REPRODUCTION"
OBSERVATION_REPRODUCTION = "OBSERVATION_REPRODUCTION"
ACTION_REPLAY = "ACTION_REPLAY"
OBSERVATION_RECHECK = "OBSERVATION_RECHECK"

TARGET_ACTION_TYPES = frozenset({"click_element", "fill_input", "press_enter"})
OBSERVATION_RULES = frozenset({"R04", "R05", "R06"})


@dataclass(frozen=True)
class ExpectedReproduction:
    action_type: str | None
    target_identity: str | None
    mode: str
    context_requirements: tuple[str, ...] = field(default_factory=tuple)
    rule_id: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {ACTION_REPRODUCTION, OBSERVATION_REPRODUCTION}:
            raise ValueError(f"unsupported confirmation mode: {self.mode}")
        if self.mode == OBSERVATION_REPRODUCTION:
            if self.action_type is not None or self.target_identity is not None:
                raise ValueError("observation reproduction cannot contain an action or target instruction")
        elif self.action_type not in TARGET_ACTION_TYPES or not self.target_identity:
            raise ValueError("action reproduction requires a target-consuming action and target")

    @property
    def requires_target(self) -> bool:
        return self.mode == ACTION_REPRODUCTION

    def as_dict(self) -> dict[str, Any]:
        return {
            "confirmation_mode": self.mode,
            "expected_action_type": self.action_type,
            "expected_target_identity": self.target_identity,
            "expected_action_requires_target": self.requires_target,
            "context_requirements": list(self.context_requirements),
            "rule_id": self.rule_id,
        }


@dataclass(frozen=True)
class ReproductionContext:
    evidence_target: str | None
    trigger_action_type: str | None
    trigger_target: str | None
    strategy: str
    reproduction_action_type: str | None
    reproduction_target: str | None
    requires_action_reproduction: bool
    reason: str
    rule_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_target_identity": self.evidence_target,
            "trigger_action_type": self.trigger_action_type,
            "trigger_target_identity": self.trigger_target,
            "reproduction_strategy": self.strategy,
            "reproduction_action_type": self.reproduction_action_type,
            "reproduction_target_identity": self.reproduction_target,
            "requires_action_reproduction": self.requires_action_reproduction,
            "reproduction_derivation_reason": self.reason,
            "rule_id": self.rule_id,
        }


def target_identity_from_anomaly(anomaly: Mapping[str, Any]) -> str | None:
    target_id = str(anomaly.get("target_id") or "").strip()
    if not target_id:
        evidence = anomaly.get("evidence", {})
        if isinstance(evidence, Mapping):
            target_id = str(evidence.get("target_bid") or evidence.get("target_id") or "").strip()
    if not target_id:
        return None
    if ":" in target_id:
        return target_id
    return f"bid:{target_id}"


def _identity(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    return text if ":" in text else f"bid:{text}"


def derive_reproduction_context(anomaly: Mapping[str, Any]) -> ReproductionContext:
    """Derive confirmation semantics from runtime provenance, not evidence target."""
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    rule_id = str(anomaly.get("rule_id") or evidence.get("rule_id") or "")
    evidence_target = _identity(anomaly.get("evidence_target_identity") or evidence.get("evidence_target_identity")) or target_identity_from_anomaly(anomaly)
    trigger_action = str(anomaly.get("trigger_action_type") or evidence.get("trigger_action_type") or anomaly.get("action_type") or evidence.get("action_type") or "").strip() or None
    trigger_target = _identity(anomaly.get("trigger_target_identity") or evidence.get("trigger_target_identity"))
    if trigger_target is None and not any(key in anomaly or key in evidence for key in ("evidence_target_identity", "trigger_target_identity", "reproduction_strategy")):
        trigger_target = target_identity_from_anomaly(anomaly)  # legacy candidate compatibility
    explicit_strategy = str(anomaly.get("reproduction_strategy") or evidence.get("reproduction_strategy") or "").upper()
    if explicit_strategy == OBSERVATION_RECHECK or rule_id in OBSERVATION_RULES:
        return ReproductionContext(evidence_target, trigger_action, trigger_target, OBSERVATION_RECHECK, None, None, False, "observation-rule", rule_id)
    if explicit_strategy == ACTION_REPLAY or (trigger_action in TARGET_ACTION_TYPES and trigger_target):
        if trigger_action in TARGET_ACTION_TYPES and trigger_target:
            return ReproductionContext(evidence_target, trigger_action, trigger_target, ACTION_REPLAY, trigger_action, trigger_target, True, "runtime-trigger-provenance", rule_id)
    return ReproductionContext(evidence_target, trigger_action, trigger_target, OBSERVATION_RECHECK, None, None, False, "no-executable-trigger-provenance", rule_id)


def expected_reproduction_for_anomaly(anomaly: Mapping[str, Any]) -> ExpectedReproduction | None:
    context = derive_reproduction_context(anomaly)
    rule_id = context.rule_id
    if context.strategy == OBSERVATION_RECHECK:
        if context.reason == "no-executable-trigger-provenance" and rule_id not in OBSERVATION_RULES:
            return None
        requirements = ("url_path", "viewport_mode", "logical_entity", "issue_subtype")
        if rule_id == "R06":
            requirements = ("url_path", "viewport_width_or_mode", "logical_entity", "issue_subtype")
        return ExpectedReproduction(
            action_type=None,
            target_identity=None,
            mode=OBSERVATION_REPRODUCTION,
            context_requirements=requirements,
            rule_id=rule_id,
        )
    action_type = str(context.reproduction_action_type or "")
    target = context.reproduction_target
    if action_type in TARGET_ACTION_TYPES and target:
        return ExpectedReproduction(action_type, target, ACTION_REPRODUCTION, ("url_path", "logical_entity"), rule_id)
    return None


def action_semantics_matrix(action_space: Any) -> list[dict[str, Any]]:
    rows = []
    for action_type in action_space.action_types:
        consumes = action_type in TARGET_ACTION_TYPES
        if action_type == "click_element":
            roles = ["button", "link", "checkbox", "radio", "menuitem", "tab"]
            tags = ["button", "a", "summary", "option", "label"]
        elif action_type in {"fill_input", "press_enter"}:
            roles = ["textbox", "searchbox", "combobox"]
            tags = ["input", "textarea"]
        else:
            roles, tags = [], []
        rows.append({
            "action_type": action_type,
            "requires_candidate_target": consumes,
            "consumes_candidate_slot": consumes,
            "candidate_index_semantic": "executable candidate index" if consumes else "canonical slot 0; ignored by executor",
            "executable_without_target": not consumes,
            "valid_target_roles": roles,
            "valid_target_tags": tags,
            "usable_for_confirmation_reproduction": consumes,
        })
    return rows


def _url_path(value: Any) -> str:
    parsed = urlparse(str(value or ""))
    return parsed.path or "/"


def reproduction_context_signature(anomaly: Mapping[str, Any]) -> tuple[Any, ...]:
    evidence = anomaly.get("evidence", {}) if isinstance(anomaly.get("evidence"), Mapping) else {}
    viewport = evidence.get("viewport") or evidence.get("viewport_bbox") or []
    width = None
    if isinstance(viewport, (list, tuple)) and len(viewport) >= 3:
        try:
            width = round(float(viewport[2]), 1)
        except (TypeError, ValueError):
            width = None
    return (
        _url_path(anomaly.get("url_path") or anomaly.get("url") or evidence.get("url_path") or evidence.get("url")),
        width,
        target_identity_from_anomaly(anomaly),
        str(anomaly.get("issue_type") or evidence.get("issue_type") or anomaly.get("type") or ""),
    )


def same_reproduction_context(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return reproduction_context_signature(left) == reproduction_context_signature(right)


def is_reproduction_instruction_executable(
    instruction: ExpectedReproduction,
    *,
    action_space: Any,
    observation: Mapping[str, Any],
    mask: Any,
) -> dict[str, Any]:
    if instruction.mode != ACTION_REPRODUCTION:
        return {"mask_valid": False, "semantically_executable": False, "reason": "observation_reproduction"}
    action_type_valid = instruction.action_type in action_space.action_types
    candidates = observation.get("candidate_elements", []) if isinstance(observation, Mapping) else []
    candidates = candidates if isinstance(candidates, list) else []
    target_slot = None
    for index, candidate in enumerate(candidates[: action_space.max_candidates]):
        if isinstance(candidate, Mapping) and target_identity_from_anomaly({"target_id": candidate.get("bid")}) == instruction.target_identity:
            target_slot = index
            break
    action_id = action_space.encode(instruction.action_type, target_slot) if action_type_valid and target_slot is not None else None
    mask_valid = bool(action_id is not None and float(mask[action_id]) > 0.0)
    return {
        "action_type_valid": action_type_valid,
        "target_slot": target_slot,
        "mask_valid": mask_valid,
        "semantically_executable": bool(action_type_valid and target_slot is not None and mask_valid),
        "action_id": action_id,
    }
