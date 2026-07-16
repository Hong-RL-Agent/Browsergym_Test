"""Small, inspectable behavior-cloning prior for goal-conditioned warm starts."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


def build_imitation_prior(paths: Iterable[str | Path]) -> Dict[str, Any]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                goal = str(row.get("exploration_goal") or "novel_state_exploration")
                action = row.get("action") or {}
                action_type = str(action.get("action_type") or row.get("action_type") or "")
                if action_type:
                    counts[goal][action_type] += 1
    return {
        goal: {action: count / sum(counter.values()) for action, count in counter.items()}
        for goal, counter in counts.items()
        if counter
    }


def save_imitation_prior(paths: Iterable[str | Path], output: str | Path) -> Dict[str, Any]:
    prior = build_imitation_prior(paths)
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
    return prior
