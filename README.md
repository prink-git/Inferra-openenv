---
title: inferra-openenv
sdk: docker
app_port: 7860
pinned: false
---

# Inferra - AI Incident Response Environment

A deterministic OpenEnv-compatible environment for training agents to debug production incidents.

This environment evaluates how effectively AI agents perform root-cause analysis under partial observability and ambiguous production signals, simulating real-world incident debugging workflows.

## Features

- Implements OpenEnv core interface:
  - `reset() -> Observation`
  - `step(action) -> (Observation, Reward, done, info)`
  - `state() -> dict`
- Uses Pydantic models for `Observation`, `Action`, and `Reward`
- Action space includes `inspect_dependency` for disambiguating root cause before applying fixes
- Includes three deterministic tasks: easy, medium, hard
- Deterministic grader with score range `[0.0, 1.0]`
- FastAPI server exposing HF Spaces-compatible endpoints
- Baseline inference script using OpenAI API client

## Why this is challenging

- Logs are intentionally ambiguous and do not reveal the root cause directly.
- Multiple misleading signals and decoy metrics can point toward incorrect fixes.
- Solving incidents requires multi-step reasoning rather than single-step prediction.
- Strong performance depends on executing the correct action sequence before applying a fix.
- Shortcut behavior is explicitly penalized to discourage guessing and reward disciplined diagnosis.

## Project Structure

```text
Inferra/
├── app/
│   ├── env.py
│   ├── models.py
│   ├── tasks.py
│   ├── grader.py
│   ├── rewards.py
│   └── utils.py
├── inference.py
├── openenv.yaml
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tasks

1. easy: High CPU issue
   - Cause: infinite loop bug
   - Correct fix: `restart_service`
2. medium: High latency
   - Cause: database bottleneck
   - Correct fix: `scale_service` or `update_config`
3. hard: Deployment failure
   - Cause: bad config push
   - Correct fix: `rollback_deployment`

## API Endpoints

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

## Action Space

- `read_logs`
- `check_metrics`
- `inspect_dependency`
- `restart_service`
- `scale_service`
- `rollback_deployment`
- `update_config`

The baseline policy follows a diagnosis-first strategy: read logs -> check metrics -> inspect dependency -> apply fix.

## Reward and Evaluation

The reward function is non-binary and captures incremental progress instead of only final success.

- Partial progress is rewarded during diagnosis.
- Penalties are applied for repeated actions.
- Penalties are applied for premature fixes.
- Penalties are applied for irrelevant actions.

The episode grader evaluates:

- final correctness
- reasoning sequence quality
- efficiency of the action trajectory

## Baseline Performance

- easy: 0.80
- medium: 0.77
- hard: 0.52

Interpretation:

- easy is solvable with a basic diagnosis-first strategy
- medium requires stronger reasoning and dependency-aware investigation
- hard remains challenging even with competent policy behavior

## Advanced Benchmarking (Top-Tier Evaluation)

This project now includes a comparative benchmark harness to evaluate reasoning quality across multiple policies:

- `diagnosis_first`: disciplined investigation-first strategy
- `fix_first`: shortcut-heavy strategy to expose failure modes
- `random`: seeded random-action policy for robustness calibration

The benchmark reports:

- average score and resolve rate per policy
- efficiency and action-discipline metrics (steps, harmful, unnecessary)
- per-task score breakdown across easy, medium, hard

Run:

```bash
python benchmark.py
```

### Example Requests

Reset to hard task:

```bash
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"hard"}'
```

Step with action:

```bash
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"read_logs"}'
```

Inspect dependencies:

```bash
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"inspect_dependency"}'
```

## Baseline Inference

Environment variables used by `inference.py`:

- `OPENAI_API_KEY`
- `API_BASE_URL` (default: `https://api.openai.com/v1`)
- `MODEL_NAME` (default: `gpt-4.1-mini`)

Run baseline over all tasks:

```bash
python inference.py
```

If `OPENAI_API_KEY` is not provided, the script falls back to a deterministic heuristic policy.

## Local Run

```bash
pip install -r requirements.txt
uvicorn app.env:app --host 0.0.0.0 --port 7860
```

## Docker

Build:

```bash
docker build -t inferra-openenv .
```

Run:

```bash
docker run --rm -p 7860:7860 inferra-openenv
```

## OpenEnv Validation Notes

- `openenv.yaml` is included and references environment entrypoints.
- Environment and grader are deterministic.
- Runtime and compute constraints are specified in `openenv.yaml`.

This environment is designed to expose failure modes in LLM-based agents operating under uncertainty and incomplete system visibility.
