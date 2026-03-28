from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class IncidentTask:
    task_id: str
    difficulty: str
    title: str
    root_cause: str
    primary_subsystem: str
    correct_fixes: Tuple[str, ...]
    diagnosis_actions: Tuple[str, ...]
    harmful_actions: Tuple[str, ...]
    initial_metrics: Dict[str, float]
    initial_logs: Tuple[str, ...]
    dependency_report: Dict[str, str | bool]


def _catalog() -> Dict[str, IncidentTask]:
    return {
        "easy": IncidentTask(
            task_id="easy",
            difficulty="easy",
            title="High CPU Issue",
            root_cause="Infinite loop bug in worker thread",
            primary_subsystem="service",
            correct_fixes=("restart_service",),
            diagnosis_actions=("read_logs", "check_metrics", "inspect_dependency"),
            harmful_actions=("rollback_deployment",),
            initial_metrics={"cpu": 86.0, "memory": 79.0, "latency": 286.0},
            initial_logs=(
                "2026-03-27T10:00:00Z [INFO] scheduler accepted 128 batch jobs",
                "2026-03-27T10:00:01Z [WARN] worker timeout threshold crossed intermittently for shard=2",
                "2026-03-27T10:00:02Z [DEBUG] heartbeat delayed by 142ms on worker-pool",
                "2026-03-27T10:00:03Z [ERROR] queue backlog increasing while ack rate stable",
                "2026-03-27T10:00:04Z [INFO] retries enabled for batch_processor stage=transform",
                "2026-03-27T10:00:05Z [WARN] service response variance increased during peak window",
            ),
            dependency_report={
                "database_status": "healthy",
                "service_health": "worker thread pool degraded; one thread spinning",
                "queue_length": "1287",
            },
        ),
        "medium": IncidentTask(
            task_id="medium",
            difficulty="medium",
            title="High Latency",
            root_cause="Database bottleneck under read-heavy traffic",
            primary_subsystem="database",
            correct_fixes=("scale_service", "update_config"),
            diagnosis_actions=("read_logs", "check_metrics", "inspect_dependency"),
            harmful_actions=("restart_service",),
            initial_metrics={"cpu": 83.0, "memory": 82.0, "latency": 336.0},
            initial_logs=(
                "2026-03-27T10:05:00Z [INFO] api throughput increased 3.2x over baseline",
                "2026-03-27T10:05:01Z [WARN] cache hit ratio dropped to 0.61",
                "2026-03-27T10:05:02Z [DEBUG] worker cpu bursts observed every 30s",
                "2026-03-27T10:05:03Z [WARN] connection acquisition jitter increased beyond normal",
                "2026-03-27T10:05:04Z [ERROR] request queue age exceeded SLO in gateway",
                "2026-03-27T10:05:05Z [INFO] retry middleware activated for read endpoints",
            ),
            dependency_report={
                "database_status": "pool saturation and lock wait spikes detected",
                "service_health": "api service healthy but waiting on downstream reads",
                "queue_length": "743",
            },
        ),
        "hard": IncidentTask(
            task_id="hard",
            difficulty="hard",
            title="Deployment Failure",
            root_cause="Deployment introduced configuration corruption in database connection pool",
            primary_subsystem="database",
            correct_fixes=("rollback_deployment",),
            diagnosis_actions=("read_logs", "check_metrics", "inspect_dependency"),
            harmful_actions=("scale_service", "restart_service"),
            initial_metrics={"cpu": 82.0, "memory": 84.0, "latency": 358.0},
            initial_logs=(
                "2026-03-27T10:10:00Z [WARN] gateway request timeout while waiting for upstream connection",
                "2026-03-27T10:10:01Z [INFO] retry middleware attempt=2 for /v1/search",
                "2026-03-27T10:10:02Z [WARN] query execution exceeded threshold duration=1480ms",
                "2026-03-27T10:10:03Z [ERROR] api error_rate reached 9.1% in last 5m window",
                "2026-03-27T10:10:04Z [DEBUG] cpu scheduler stable across service nodes",
                "2026-03-27T10:10:05Z [INFO] client retry budget nearing exhaustion",
            ),
            dependency_report={
                "database_pool": "misconfigured",
                "recent_deployment": True,
                "database_status": "pool acquisition failures with timeout spikes",
                "service_health": "api pods healthy but blocked on db pool",
                "queue_length": "611",
            },
        ),
    }


TASK_CATALOG = _catalog()


def get_task(task_id: str) -> IncidentTask:
    if task_id not in TASK_CATALOG:
        raise ValueError(f"Unknown task_id: {task_id}")
    return TASK_CATALOG[task_id]


def list_tasks() -> List[Dict[str, str]]:
    return [
        {
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "title": task.title,
        }
        for task in TASK_CATALOG.values()
    ]


def evolve_metrics(task: IncidentTask, step_count: int, resolved: bool) -> Dict[str, float]:
    if resolved:
        if task.task_id == "easy":
            return {"cpu": 44.0, "memory": 58.0, "latency": 105.0}
        if task.task_id == "medium":
            return {"cpu": 53.0, "memory": 62.0, "latency": 128.0}
        return {"cpu": 49.0, "memory": 60.0, "latency": 118.0}

    if task.task_id == "easy":
        return {
            "cpu": min(92.0, 86.0 + (0.9 * step_count)),
            "memory": min(90.0, 79.0 + (0.7 * step_count)),
            "latency": min(380.0, 286.0 + (8.0 * step_count)),
        }
    if task.task_id == "medium":
        return {
            "cpu": min(90.0, 83.0 + (0.5 * step_count)),
            "memory": min(92.0, 82.0 + (0.8 * step_count)),
            "latency": min(430.0, 336.0 + (10.0 * step_count)),
        }
    return {
        "cpu": min(88.0, 82.0 + (0.4 * step_count)),
        "memory": min(93.0, 84.0 + (0.9 * step_count)),
        "latency": min(460.0, 358.0 + (11.0 * step_count)),
    }


def inspect_dependency_snapshot(task: IncidentTask, resolved: bool) -> Dict[str, str]:
    if resolved:
        return {
            "database_status": "healthy",
            "service_health": "healthy",
            "queue_length": "112",
        }
    return dict(task.dependency_report)


def next_log_line(task: IncidentTask, step_count: int, action_type: str) -> str:
    if action_type == "inspect_dependency":
        if task.task_id == "easy":
            return (
                "2026-03-27T10:00:10Z [DEBUG] dependency check: db healthy; "
                "worker thread spin loop observed; queue drain stalled"
            )
        if task.task_id == "medium":
            return (
                "2026-03-27T10:05:10Z [DEBUG] dependency check: db pool exhausted; "
                "service threads idle waiting on I/O"
            )
        return (
            "2026-03-27T10:10:10Z [DEBUG] dependency check: database_pool=misconfigured; "
            "recent_deployment=true"
        )

    if task.task_id == "easy":
        return (
            f"2026-03-27T10:00:{10 + step_count:02d}Z [WARN] batch ack delay persists; "
            "cpu scheduler pressure unchanged"
        )
    if task.task_id == "medium":
        return (
            f"2026-03-27T10:05:{10 + step_count:02d}Z [WARN] read path saturation remains; "
            "tail latency oscillating"
        )
    return (
        f"2026-03-27T10:10:{10 + step_count:02d}Z [ERROR] connection retries rising; "
        "slow-query tail persists with stable CPU"
    )
