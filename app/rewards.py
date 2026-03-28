from __future__ import annotations

from app.models import Reward
from app.utils import clamp


_FIX_REWARD_FULL = 0.5
_FIX_REWARD_PARTIAL = 0.45


def compute_reward(
    *,
    diagnosis_completed_now: bool,
    identified_subsystem_now: bool,
    used_inspect_dependency_before_fix: bool,
    resolved_now: bool,
    unnecessary: bool,
    repeated_consecutive: bool,
    premature_fix: bool,
    timeout_failure: bool,
) -> Reward:
    components = {
        "diagnosis_complete": 0.0,
        "subsystem": 0.0,
        "inspect_before_fix": 0.0,
        "fix": 0.0,
        "irrelevant": 0.0,
        "repetition": 0.0,
        "premature_fix": 0.0,
        "timeout": 0.0,
    }

    if diagnosis_completed_now:
        components["diagnosis_complete"] = 0.2

    if identified_subsystem_now:
        components["subsystem"] = 0.2

    if used_inspect_dependency_before_fix:
        components["inspect_before_fix"] = 0.2

    if resolved_now:
        components["fix"] = (
            _FIX_REWARD_FULL if used_inspect_dependency_before_fix else _FIX_REWARD_PARTIAL
        )

    if unnecessary:
        components["irrelevant"] = -0.1

    if repeated_consecutive:
        components["repetition"] = -0.2

    if premature_fix:
        components["premature_fix"] = -0.3

    if timeout_failure:
        components["timeout"] = -0.3

    total = clamp(sum(components.values()), -1.0, 1.0)
    reason_parts = [name for name, value in components.items() if value != 0.0]

    return Reward(
        value=total,
        components=components,
        reason=", ".join(reason_parts) if reason_parts else "neutral",
    )
