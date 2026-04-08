from __future__ import annotations

import json
import os
from typing import Any, Dict

from openai import OpenAI

from app.env import IncidentResponseEnv
from app.grader import grade_episode
from app.models import Action, Observation


ALLOWED_ACTIONS = [
    "read_logs",
    "check_metrics",
    "inspect_dependency",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "update_config",
]


def _emit(line: str) -> None:
    print(line, flush=True)


def _heuristic_action(observation: Observation) -> Action:
    logs_lower = observation.logs.lower()
    latency = observation.metrics.get("latency", 0.0)
    cpu = observation.metrics.get("cpu", 0.0)
    memory = observation.metrics.get("memory", 0.0)

    # First gather broad context, then disambiguate root cause via dependency inspection.
    if observation.step_count == 0:
        return Action(action_type="read_logs")
    if observation.step_count == 1:
        return Action(action_type="check_metrics")

    needs_disambiguation = (
        (cpu >= 80.0 and latency >= 280.0)
        or (memory >= 85.0 and latency >= 300.0)
        or ("deploy" in logs_lower)
        or ("error_rate" in logs_lower)
    )

    if needs_disambiguation and "dependency report" not in logs_lower:
        return Action(action_type="inspect_dependency")

    if "checksum mismatch" in logs_lower or "config" in logs_lower:
        return Action(action_type="rollback_deployment")
    if "pool exhausted" in logs_lower or "lock wait" in logs_lower:
        return Action(action_type="update_config")
    if latency >= 300.0 and cpu < 90.0:
        return Action(action_type="scale_service")
    if "spin loop" in logs_lower or (cpu >= 90.0 and latency < 420.0):
        return Action(action_type="restart_service")
    return Action(action_type="inspect_dependency")


def _llm_action(client: OpenAI, model_name: str, observation: Observation) -> Action:
    prompt = {
        "role": "user",
        "content": (
            "You are operating a production incident response environment. "
            "Return only JSON with keys action_type, target, parameters. "
            f"Allowed action_type values: {ALLOWED_ACTIONS}. "
            "Choose one best next action.\n"
            f"Observation: {observation.model_dump_json()}"
        ),
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a precise SRE assistant."},
                prompt,
            ],
            temperature=0,
            max_tokens=120,
        )
    except Exception:
        return _heuristic_action(observation)

    content = response.choices[0].message.content or "{}"
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return _heuristic_action(observation)

    try:
        return Action.model_validate(payload)
    except Exception:
        return _heuristic_action(observation)


def run_task(task_id: str, use_openai: bool = True, max_steps: int = 8) -> Dict[str, Any]:
    env = IncidentResponseEnv(task_id=task_id, max_steps=max_steps)
    observation = env.reset(task_id=task_id)
    _emit(f"[START] task={task_id}")

    # Validator injects API_KEY/API_BASE_URL for proxy-routed calls.
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini")

    client = None
    if use_openai and api_key:
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
        except Exception:
            client = None

    done = False
    step_num = 0
    while not done:
        try:
            if client is not None:
                action = _llm_action(client, model_name, observation)
            else:
                action = _heuristic_action(observation)
        except Exception:
            action = _heuristic_action(observation)

        observation, reward, done, _info = env.step(action)
        step_num += 1
        _emit(
            "[STEP] "
            f"task={task_id} "
            f"step={step_num} "
            f"action={action.action_type} "
            f"reward={reward.value} "
            f"done={done}"
        )

    final_state = env.state()
    score = grade_episode(final_state, final_state["action_history"])
    _emit(
        "[END] "
        f"task={task_id} "
        f"score={score} "
        f"steps={len(final_state['action_history'])} "
        f"resolved={final_state['resolved']}"
    )

    return {
        "task_id": task_id,
        "score": score,
        "steps": len(final_state["action_history"]),
        "resolved": final_state["resolved"],
        "action_history": final_state["action_history"],
    }


def run_baseline_suite(use_openai: bool = True) -> Dict[str, Any]:
    results = {}
    for task_id in ("easy", "medium", "hard"):
        results[task_id] = run_task(task_id=task_id, use_openai=use_openai)

    average_score = round(
        sum(item["score"] for item in results.values()) / float(len(results)), 4
    )
    return {"results": results, "average_score": average_score}


if __name__ == "__main__":
    suite = run_baseline_suite(use_openai=True)
    print(json.dumps(suite, indent=2))
