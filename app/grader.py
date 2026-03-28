from __future__ import annotations

from typing import Any, Dict, List

from app.utils import clamp


_IDEAL_STEP_RANGES = {
    "easy": (3, 4),
    "medium": (4, 5),
    "hard": (5, 6),
}

_DIFFICULTY_OFFSETS = {
    "easy": 0.10,
    "medium": 0.23,
    "hard": 0.30,
}


def _efficiency_score(task_id: str, steps: int) -> float:
    low, high = _IDEAL_STEP_RANGES.get(task_id, (3, 4))
    if low <= steps <= high:
        return 1.0

    distance = (low - steps) if steps < low else (steps - high)
    return clamp(1.0 - (0.18 * float(distance)), 0.0, 1.0)


def _is_correct_sequence(
    task_id: str,
    action_types: List[str],
    *,
    first_fix_index: int,
    diagnosis_before_fix: int,
    used_dependency_before_fix: bool,
    final_fix_correct: bool,
) -> bool:
    if first_fix_index < 0 or not final_fix_correct:
        return False

    if len(action_types) < 2:
        return False
    if action_types[0] != "read_logs" or action_types[1] != "check_metrics":
        return False

    if task_id == "easy":
        return first_fix_index in (2, 3)

    if task_id == "medium":
        return (
            first_fix_index >= 3
            and diagnosis_before_fix >= 3
            and used_dependency_before_fix
            and "inspect_dependency" in action_types[:first_fix_index]
        )

    # Hard requires deeper investigation before the fix.
    return (
        first_fix_index >= 4
        and diagnosis_before_fix >= 4
        and used_dependency_before_fix
        and "inspect_dependency" in action_types[:first_fix_index]
    )


def grade_episode(final_state: Dict[str, Any], action_history: List[Dict[str, Any]]) -> float:
    task_id = str(final_state.get("task_id", ""))
    resolved = bool(final_state.get("resolved", False))
    final_fix_correct = bool(final_state.get("final_fix_correct", False))
    unnecessary_actions = int(final_state.get("unnecessary_actions", 0))
    harmful_actions = int(final_state.get("harmful_actions", 0))
    diagnosed = bool(final_state.get("diagnosed", False))
    inspected_dependency = bool(final_state.get("inspected_dependency", False))
    identified_subsystem = bool(final_state.get("identified_subsystem", False))
    diagnosis_steps = int(final_state.get("diagnosis_steps", 0))
    timed_out = bool(final_state.get("timed_out", False))

    correctness = 1.0 if resolved and final_fix_correct else 0.0

    steps = len(action_history)
    efficiency = _efficiency_score(task_id, steps)

    discipline = clamp(1.0 - (0.12 * unnecessary_actions) - (0.35 * harmful_actions), 0.0, 1.0)

    action_types = [str(item.get("action_type", "")) for item in action_history]
    first_fix_index = next(
        (i for i, item in enumerate(action_history) if bool(item.get("is_fix", False))),
        -1,
    )
    used_dependency_before_fix = False
    diagnosis_before_fix = 0
    if first_fix_index >= 0:
        diagnosis_before_fix = sum(
            1 for item in action_history[:first_fix_index] if bool(item.get("is_diagnosis", False))
        )
        used_dependency_before_fix = any(
            bool(item.get("action_type") == "inspect_dependency")
            for item in action_history[:first_fix_index]
        )

    guessed_fix = first_fix_index >= 0 and diagnosis_before_fix == 0
    proper_diagnosis = diagnosed and diagnosis_steps >= 2
    if task_id in {"medium", "hard"}:
        proper_diagnosis = proper_diagnosis and inspected_dependency

    sequence_ok = _is_correct_sequence(
        task_id,
        action_types,
        first_fix_index=first_fix_index,
        diagnosis_before_fix=diagnosis_before_fix,
        used_dependency_before_fix=used_dependency_before_fix,
        final_fix_correct=final_fix_correct,
    )

    sequence_penalty = 0.0

    if first_fix_index >= 0:
        if not inspected_dependency and resolved:
            sequence_penalty += 0.3
        if first_fix_index < 2:
            sequence_penalty += 0.2
        if diagnosis_steps < 2:
            sequence_penalty += 0.2

    if guessed_fix:
        sequence_penalty += 0.15
    if first_fix_index >= 0 and not used_dependency_before_fix:
        sequence_penalty += 0.08
    if not sequence_ok:
        sequence_penalty += 0.06
    if timed_out:
        sequence_penalty += 0.12

    diagnosis_quality = 1.0 if proper_diagnosis and identified_subsystem else (0.7 if proper_diagnosis else 0.0)

    score = (0.55 * correctness) + (0.20 * efficiency) + (0.15 * diagnosis_quality) + (0.10 * discipline)
    score -= sequence_penalty
    score -= _DIFFICULTY_OFFSETS.get(task_id, 0.0)

    if task_id == "hard":
        if steps < 4:
            score -= 0.2
        if unnecessary_actions > 0:
            score -= 0.1
        if not sequence_ok:
            score -= 0.08

    if task_id == "medium" and not used_dependency_before_fix:
        score = min(score, 0.80)

    no_wasted_steps = unnecessary_actions == 0 and harmful_actions == 0
    in_ideal_range = _IDEAL_STEP_RANGES.get(task_id, (3, 4))[0] <= steps <= _IDEAL_STEP_RANGES.get(task_id, (3, 4))[1]
    can_be_perfect = correctness == 1.0 and sequence_ok and proper_diagnosis and no_wasted_steps and in_ideal_range
    if not can_be_perfect:
        score = min(score, 0.95)

    if correctness == 0.0:
        score = min(score, 0.35)

    return round(clamp(score, 0.0, 1.0), 4)
