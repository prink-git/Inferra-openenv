from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from app.grader import grade_episode
from app.models import Action, Observation, Reward
from app.rewards import compute_reward
from app.tasks import (
    TASK_CATALOG,
    evolve_metrics,
    get_task,
    inspect_dependency_snapshot,
    list_tasks,
    next_log_line,
)
from app.utils import normalize_metrics


_ALLOWED_ACTIONS = {
    "read_logs",
    "check_metrics",
    "restart_service",
    "scale_service",
    "rollback_deployment",
    "update_config",
    "inspect_dependency",
}


class IncidentResponseEnv:
    def __init__(self, task_id: str = "easy", max_steps: int = 8) -> None:
        self.max_steps = max_steps
        self._task_id = task_id
        self._task = get_task(task_id)

        self._step_count = 0
        self._resolved = False
        self._diagnosed = False
        self._final_fix_correct = False
        self._timed_out = False
        self._inspected_dependency = False
        self._identified_subsystem = False
        self._diagnosis_steps = 0
        self._diagnosis_actions_taken: set[str] = set()
        self._diagnosis_completed = False
        self._inspect_before_fix_rewarded = False

        self._metrics: Dict[str, float] = dict(self._task.initial_metrics)
        self._alerts = self._derive_ambiguous_alerts(normalize_metrics(self._metrics))
        self._logs = list(self._task.initial_logs)

        self._action_history: list[dict[str, Any]] = []
        self._unnecessary_actions = 0
        self._harmful_actions = 0
        self._last_reward = Reward(value=0.0, components={}, reason="init")

    def reset(self, task_id: Optional[str] = None) -> Observation:
        if task_id is not None:
            self._task_id = task_id
        self._task = get_task(self._task_id)

        self._step_count = 0
        self._resolved = False
        self._diagnosed = False
        self._final_fix_correct = False
        self._timed_out = False
        self._inspected_dependency = False
        self._identified_subsystem = False
        self._diagnosis_steps = 0
        self._diagnosis_actions_taken = set()
        self._diagnosis_completed = False
        self._inspect_before_fix_rewarded = False

        self._metrics = dict(self._task.initial_metrics)
        self._alerts = self._derive_ambiguous_alerts(normalize_metrics(self._metrics))
        self._logs = list(self._task.initial_logs)

        self._action_history = []
        self._unnecessary_actions = 0
        self._harmful_actions = 0
        self._last_reward = Reward(value=0.0, components={}, reason="reset")

        return self._build_observation()

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        if self._resolved or self._step_count >= self.max_steps:
            done_obs = self._build_observation()
            terminal_reward = Reward(value=0.0, components={}, reason="episode_complete")
            return done_obs, terminal_reward, True, self._build_info(done=True)

        parsed_action = self._parse_action(action)
        action_type = str(parsed_action["action_type"])

        self._step_count += 1

        diagnosed_before = self._diagnosis_completed
        identified_subsystem_now = False
        resolved_now = False
        unnecessary = False
        harmful = False
        premature_fix = False
        used_inspect_dependency_before_fix = False

        previous_action = self._action_history[-1]["action_type"] if self._action_history else None
        repeated_consecutive = previous_action == action_type
        is_diagnosis = action_type in {"read_logs", "check_metrics", "inspect_dependency"}
        is_fix = action_type in {
            "restart_service",
            "scale_service",
            "rollback_deployment",
            "update_config",
        }

        if action_type == "read_logs":
            seen_before = action_type in self._diagnosis_actions_taken
            self._logs.append(next_log_line(self._task, self._step_count, action_type))
            self._register_diagnosis_action(action_type)
            if seen_before and diagnosed_before:
                unnecessary = True

        elif action_type == "check_metrics":
            seen_before = action_type in self._diagnosis_actions_taken
            current = normalize_metrics(evolve_metrics(self._task, self._step_count, self._resolved))
            self._logs.append(
                f"metrics snapshot cpu={current['cpu']} memory={current['memory']} latency={current['latency']}"
            )
            self._logs.append(next_log_line(self._task, self._step_count, action_type))
            self._register_diagnosis_action(action_type)
            if seen_before and diagnosed_before:
                unnecessary = True

        elif action_type == "inspect_dependency":
            seen_before = action_type in self._diagnosis_actions_taken
            self._inspected_dependency = True
            if not self._identified_subsystem:
                self._identified_subsystem = True
                identified_subsystem_now = True
            report = inspect_dependency_snapshot(self._task, self._resolved)
            self._logs.append(
                "dependency report "
                f"database_status={report['database_status']}; "
                f"service_health={report['service_health']}; "
                f"queue_length={report['queue_length']}"
            )
            self._logs.append(next_log_line(self._task, self._step_count, action_type))
            self._register_diagnosis_action(action_type)
            if seen_before and diagnosed_before:
                unnecessary = True

        elif is_fix:
            has_logs_and_metrics = self._has_logs_and_metrics_diagnosis()
            if not has_logs_and_metrics:
                premature_fix = True
            if self._inspected_dependency and not self._inspect_before_fix_rewarded:
                used_inspect_dependency_before_fix = True
                self._inspect_before_fix_rewarded = True
            if action_type in self._task.correct_fixes:
                resolved_now = True
                self._resolved = True
                self._final_fix_correct = True
                self._logs.append(self._success_log(action_type))
            else:
                if action_type in self._task.harmful_actions:
                    harmful = True
                    self._logs.append(self._harmful_log(action_type))
                else:
                    unnecessary = True
                    self._logs.append(self._unnecessary_log(action_type))
        else:
            unnecessary = True
            self._logs.append(self._unnecessary_log(action_type))

        self._diagnosed = len(self._diagnosis_actions_taken) > 0
        self._diagnosis_completed = self._is_diagnosis_completed()

        if unnecessary:
            self._unnecessary_actions += 1
        if harmful:
            self._harmful_actions += 1

        self._metrics = normalize_metrics(evolve_metrics(self._task, self._step_count, self._resolved))
        if harmful:
            self._metrics["cpu"] = min(100.0, self._metrics["cpu"] + 7.0)
            self._metrics["memory"] = min(100.0, self._metrics["memory"] + 5.0)
            self._metrics["latency"] = min(1000.0, self._metrics["latency"] + 90.0)
            self._metrics = normalize_metrics(self._metrics)

        self._alerts = self._derive_ambiguous_alerts(self._metrics)

        timeout_failure = (
            (not self._resolved)
            and self._step_count >= self.max_steps
            and not resolved_now
        )
        self._timed_out = timeout_failure

        reward = compute_reward(
            diagnosis_completed_now=(not diagnosed_before and self._diagnosis_completed),
            identified_subsystem_now=identified_subsystem_now,
            used_inspect_dependency_before_fix=used_inspect_dependency_before_fix,
            resolved_now=resolved_now,
            unnecessary=unnecessary,
            repeated_consecutive=repeated_consecutive,
            premature_fix=premature_fix,
            timeout_failure=timeout_failure,
        )
        self._last_reward = reward

        self._action_history.append(
            {
                "step": self._step_count,
                "action_type": action_type,
                "target": parsed_action.get("target"),
                "parameters": parsed_action.get("parameters"),
                "reward": reward.value,
                "is_diagnosis": is_diagnosis,
                "is_fix": is_fix,
                "premature_fix": premature_fix,
                "repeated_consecutive": repeated_consecutive,
                "identified_subsystem_now": identified_subsystem_now,
            }
        )

        done = self._resolved or self._step_count >= self.max_steps
        observation = self._build_observation()
        info = self._build_info(done=done)
        return observation, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "title": self._task.title,
            "root_cause": self._task.root_cause,
            "step_count": self._step_count,
            "max_steps": self.max_steps,
            "resolved": self._resolved,
            "diagnosed": self._diagnosed,
            "diagnosis_steps": self._diagnosis_steps,
            "diagnosis_actions_taken": sorted(self._diagnosis_actions_taken),
            "diagnosis_completed": self._diagnosis_completed,
            "inspected_dependency": self._inspected_dependency,
            "identified_subsystem": self._identified_subsystem,
            "timed_out": self._timed_out,
            "final_fix_correct": self._final_fix_correct,
            "metrics": dict(self._metrics),
            "alerts": list(self._alerts),
            "unnecessary_actions": self._unnecessary_actions,
            "harmful_actions": self._harmful_actions,
            "action_history": list(self._action_history),
            "recent_logs": "\n".join(self._logs[-8:]),
            "last_reward": self._last_reward.model_dump(),
        }

    def _build_observation(self) -> Observation:
        return Observation(
            logs="\n".join(self._logs[-6:]),
            metrics=dict(self._metrics),
            alerts=list(self._alerts),
            step_count=self._step_count,
        )

    def _build_info(self, done: bool) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "task_id": self._task.task_id,
            "resolved": self._resolved,
            "diagnosed": self._diagnosed,
            "diagnosis_completed": self._diagnosis_completed,
            "inspected_dependency": self._inspected_dependency,
            "identified_subsystem": self._identified_subsystem,
            "timed_out": self._timed_out,
        }
        if done:
            info["score"] = grade_episode(self.state(), self._action_history)
        return info

    def _register_diagnosis_action(self, action_type: str) -> None:
        if action_type in self._task.diagnosis_actions:
            self._diagnosis_steps += 1
            self._diagnosis_actions_taken.add(action_type)

    def _has_logs_and_metrics_diagnosis(self) -> bool:
        return (
            "read_logs" in self._diagnosis_actions_taken
            and "check_metrics" in self._diagnosis_actions_taken
        )

    def _is_diagnosis_completed(self) -> bool:
        return self._has_logs_and_metrics_diagnosis() or self._inspected_dependency

    def _derive_ambiguous_alerts(self, metrics: Dict[str, float]) -> list[str]:
        alerts: list[str] = []
        if metrics["latency"] >= 250.0:
            alerts.append("Latency variance increasing")
        if metrics["memory"] >= 80.0 or metrics["cpu"] >= 80.0:
            alerts.append("System performance degraded")
        if metrics["latency"] >= 320.0 or metrics["memory"] >= 86.0:
            alerts.append("Service instability detected")
        if not alerts:
            alerts.append("Service health uncertain")
        return alerts

    def _parse_action(self, action: Action | Dict[str, Any]) -> Dict[str, Any]:
        if isinstance(action, Action):
            payload = action.model_dump()
        elif isinstance(action, dict):
            payload = action
        elif hasattr(action, "model_dump"):
            payload = action.model_dump()
        else:
            raise ValueError("Action must be a dict or Action-like model.")

        action_type = str(payload.get("action_type", "")).strip()
        if action_type not in _ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action_type: {action_type}")

        return {
            "action_type": action_type,
            "target": payload.get("target"),
            "parameters": payload.get("parameters"),
        }

    def _diagnostic_log_from_logs(self) -> str:
        if self._task.task_id == "easy":
            return "worker ERROR loop_guard exceeded; thread did not exit cleanly"
        if self._task.task_id == "medium":
            return "db WARN query queue depth rising; read replica saturated"
        return "deploy ERROR release contains invalid config payload from latest push"

    def _success_log(self, action_type: str) -> str:
        if action_type == "restart_service":
            return "ops INFO worker restarted; CPU dropped and queue draining"
        if action_type == "scale_service":
            return "ops INFO service scaled horizontally; latency recovering"
        if action_type == "update_config":
            return "ops INFO connection pool updated; database contention reduced"
        return "ops INFO rollback completed; service health restored"

    def _harmful_log(self, action_type: str) -> str:
        return f"ops WARN action={action_type} increased instability in current incident"

    def _unnecessary_log(self, action_type: str) -> str:
        return f"ops INFO action={action_type} executed with no material impact"


class ResetRequest(BaseModel):
    task_id: Optional[str] = None


class GraderRequest(BaseModel):
    final_state: Optional[Dict[str, Any]] = None
    action_history: Optional[list[Dict[str, Any]]] = None


class StepRequest(BaseModel):
    action_type: str
    target: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


app = FastAPI(title="AI Incident Response Environment")
ENV = IncidentResponseEnv()


@app.get("/tasks")
def tasks_endpoint() -> Dict[str, Any]:
    return {"tasks": list_tasks()}


@app.post("/reset")
def reset_endpoint(request: Optional[ResetRequest] = Body(default=None)) -> Dict[str, Any]:
    task_id = "easy"
    if request and request.task_id:
        task_id = request.task_id

    if task_id not in ["easy", "medium", "hard"]:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown task_id: {task_id}",
        )

    observation = ENV.reset(task_id=task_id)
    return observation.model_dump()


@app.post("/step")
def step_endpoint(action: StepRequest) -> Dict[str, Any]:
    try:
        observation, reward, done, info = ENV.step(action.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state")
def state_endpoint() -> Dict[str, Any]:
    return ENV.state()


@app.post("/grader")
def grader_endpoint(request: GraderRequest) -> Dict[str, float]:
    final_state = request.final_state or ENV.state()
    action_history = request.action_history or final_state.get("action_history", [])
    return {"score": grade_episode(final_state, action_history)}


@app.get("/baseline")
def baseline_endpoint() -> Dict[str, Any]:
    from inference import run_baseline_suite

    return run_baseline_suite(use_openai=True)
