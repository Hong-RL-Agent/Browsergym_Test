"""Shared repeat/cycle semantics used by reward diagnostics and observations."""

from __future__ import annotations


def cycle_repeat_length(action_identities: list[str], *, max_cycle: int = 3) -> int:
    if not action_identities:
        return 0
    for length in range(1, max_cycle + 1):
        if len(action_identities) >= 2 * length:
            if action_identities[-2 * length:-length] == action_identities[-length:]:
                return length
    return 0
