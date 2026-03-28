from __future__ import annotations

from typing import Dict


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "cpu": round(clamp(metrics.get("cpu", 0.0), 0.0, 100.0), 2),
        "memory": round(clamp(metrics.get("memory", 0.0), 0.0, 100.0), 2),
        "latency": round(clamp(metrics.get("latency", 0.0), 0.0, 1000.0), 2),
    }


def derive_alerts(metrics: Dict[str, float]) -> list[str]:
    alerts: list[str] = []
    if metrics["cpu"] >= 85.0:
        alerts.append("CPU usage critical")
    if metrics["memory"] >= 85.0:
        alerts.append("Memory pressure high")
    if metrics["latency"] >= 250.0:
        alerts.append("Latency SLO breach")
    if not alerts:
        alerts.append("System nominal")
    return alerts
