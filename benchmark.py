from __future__ import annotations

import json
import random
from dataclasses import dataclass
from statistics import mean
from typing import Any, Callable, Dict, List

from app.env import IncidentResponseEnv
from app.grader import grade_episode

ALLOWED_ACTIONS = [
    "read_logs",
    "check_metrics",
    "inspect_dependency",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "update_config",
]


@dataclass
class EpisodeResult:
    task_id: str
    policy: str
    score: float
    steps: int
    resolved: bool
    unnecessary_actions: int
    harmful_actions: int
    diagnosis_steps: int


def _diagnosis_first_policy(task_id: str, state: Dict[str, Any], rng: random.Random) -> str:
    history = state.get("action_history", [])
    step = int(state.get("step_count", 0))
    logs = str(state.get("recent_logs", "")).lower()

    if step == 0:
        return "read_logs"
    if step == 1:
        return "check_metrics"

    if not state.get("inspected_dependency", False):
        return "inspect_dependency"

    # Apply the highest-confidence fix after diagnosis.
    if task_id == "easy":
        return "restart_service"
    if task_id == "medium":
        if "pool" in logs or "database" in logs:
            return "update_config"
        return "scale_service"
    return "rollback_deployment"


def _fix_first_policy(task_id: str, state: Dict[str, Any], rng: random.Random) -> str:
    # Deliberately poor policy to expose reasoning gap.
    if task_id == "easy":
        return "scale_service"
    if task_id == "medium":
        return "restart_service"
    return "scale_service"


def _random_policy(task_id: str, state: Dict[str, Any], rng: random.Random) -> str:
    return rng.choice(ALLOWED_ACTIONS)


POLICIES: Dict[str, Callable[[str, Dict[str, Any], random.Random], str]] = {
    "diagnosis_first": _diagnosis_first_policy,
    "fix_first": _fix_first_policy,
    "random": _random_policy,
}


def run_episode(task_id: str, policy_name: str, seed: int = 0, max_steps: int = 8) -> EpisodeResult:
    env = IncidentResponseEnv(task_id=task_id, max_steps=max_steps)
    _ = env.reset(task_id=task_id)
    rng = random.Random(seed)

    done = False
    while not done:
        state = env.state()
        action_type = POLICIES[policy_name](task_id, state, rng)
        _, _, done, _ = env.step({"action_type": action_type})

    final_state = env.state()
    score = grade_episode(final_state, final_state.get("action_history", []))

    return EpisodeResult(
        task_id=task_id,
        policy=policy_name,
        score=score,
        steps=len(final_state.get("action_history", [])),
        resolved=bool(final_state.get("resolved", False)),
        unnecessary_actions=int(final_state.get("unnecessary_actions", 0)),
        harmful_actions=int(final_state.get("harmful_actions", 0)),
        diagnosis_steps=int(final_state.get("diagnosis_steps", 0)),
    )


def run_benchmark(episodes_per_task: int = 3) -> Dict[str, Any]:
    tasks = ("easy", "medium", "hard")
    all_results: List[EpisodeResult] = []

    for policy in POLICIES:
        for task in tasks:
            for i in range(episodes_per_task):
                seed = (hash(policy) % 1000) + i
                all_results.append(run_episode(task, policy, seed=seed))

    leaderboard: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        items = [r for r in all_results if r.policy == policy]
        leaderboard[policy] = {
            "avg_score": round(mean(r.score for r in items), 4),
            "resolve_rate": round(mean(1.0 if r.resolved else 0.0 for r in items), 4),
            "avg_steps": round(mean(r.steps for r in items), 3),
            "avg_unnecessary": round(mean(r.unnecessary_actions for r in items), 3),
            "avg_harmful": round(mean(r.harmful_actions for r in items), 3),
        }

    by_task_policy: Dict[str, Dict[str, float]] = {}
    for policy in POLICIES:
        by_task_policy[policy] = {}
        for task in tasks:
            items = [r for r in all_results if r.policy == policy and r.task_id == task]
            by_task_policy[policy][task] = round(mean(r.score for r in items), 4)

    return {
        "episodes_per_task": episodes_per_task,
        "leaderboard": leaderboard,
        "task_scores": by_task_policy,
        "total_episodes": len(all_results),
    }


def main() -> None:
    report = run_benchmark(episodes_per_task=3)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
