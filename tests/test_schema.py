"""Unit tests for core Pydantic schemas and the JSON/YAML trajectory loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from agent_probe.adapters.json_loader import (
    TraceLoadError,
    format_validation_error,
    load_trajectory,
)
from agent_probe.core.schema import (
    AssertionResult,
    EvalReport,
    Observation,
    TokenUsage,
    ToolCall,
    Trajectory,
    TrajectoryDiff,
    TrajectoryStep,
)


def _sample_trace_payload() -> dict:
    return {
        "trace_id": "trace-001",
        "agent_version": "v1.0.0",
        "task_input": "Where is my order ORD-100?",
        "final_output": "Your order is delayed; a 20 CNY coupon was issued.",
        "status": "SUCCESS",
        "steps": [
            {
                "step_id": 1,
                "state": "TOOL_EXECUTION",
                "thought": "Query logistics status for order ORD-100.",
                "action": {
                    "tool_name": "query_logistics",
                    "arguments": {"order_id": "ORD-100"},
                },
                "observation": {
                    "output": {"delay_hours": 60, "status": "DELAYED"},
                    "is_error": False,
                    "is_mocked": True,
                },
                "latency_ms": 120.5,
                "tokens": {"prompt_tokens": 100, "completion_tokens": 30},
            },
            {
                "step_id": 2,
                "state": "RESPONDING",
                "thought": "Provide final response.",
                "action": None,
                "observation": None,
                "latency_ms": 80.25,
                "tokens": {
                    "prompt_tokens": 150,
                    "completion_tokens": 50,
                    "total_tokens": 200,
                },
            },
        ],
    }


def test_token_usage_auto_computes_and_validates() -> None:
    auto = TokenUsage(prompt_tokens=120, completion_tokens=30)
    assert auto.total_tokens == 150

    explicit_ok = TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert explicit_ok.total_tokens == 15

    total_only = TokenUsage(total_tokens=42)
    assert total_only.total_tokens == 42

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=99)

    with pytest.raises(ValidationError):
        TokenUsage(prompt_tokens=-1, completion_tokens=5)


def test_trajectory_computed_fields() -> None:
    traj = Trajectory.model_validate(_sample_trace_payload())

    assert traj.total_steps == 2
    assert traj.total_tokens == 330
    assert traj.total_latency_ms == pytest.approx(200.75)
    assert traj.tool_sequence == ["query_logistics"]

    dumped = traj.model_dump()
    assert dumped["total_steps"] == 2
    assert dumped["total_tokens"] == 330
    assert dumped["tool_sequence"] == ["query_logistics"]


def test_schema_rejects_invalid_fields() -> None:
    invalid_status = _sample_trace_payload()
    invalid_status["status"] = "UNKNOWN"
    with pytest.raises(ValidationError):
        Trajectory.model_validate(invalid_status)

    invalid_step = _sample_trace_payload()
    invalid_step["steps"][0]["step_id"] = -1
    with pytest.raises(ValidationError):
        Trajectory.model_validate(invalid_step)

    missing_tokens = _sample_trace_payload()
    del missing_tokens["steps"][0]["tokens"]
    with pytest.raises(ValidationError):
        Trajectory.model_validate(missing_tokens)

    with pytest.raises(ValidationError):
        AssertionResult(rule_name="r1", passed=True, score=1.5, reason="out of bounds")


def test_evaluation_and_diff_models_validate() -> None:
    assertion = AssertionResult(
        rule_name="max_steps",
        passed=False,
        score=0.0,
        reason="Exceeded max steps",
        failed_step_ids=[5, 6],
    )
    report = EvalReport(
        trace_id="t1",
        agent_version="v2",
        overall_passed=False,
        overall_score=0.0,
        assertions=[assertion],
    )
    diff = TrajectoryDiff(
        baseline_trace_id="t0",
        candidate_trace_id="t1",
        divergence_step_id=2,
        step_delta=2,
        token_delta=500,
        latency_delta_ms=123.4,
        step_comparisons=[{"step_id": 1, "state_match": True}],
    )
    assert report.overall_passed is False
    assert diff.divergence_step_id == 2


def test_load_trajectory_valid_json_and_yaml(tmp_path: Path) -> None:
    payload = _sample_trace_payload()
    json_file = tmp_path / "trace.json"
    json_file.write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_trajectory(json_file)
    assert loaded.trace_id == "trace-001"
    assert loaded.total_steps == 2
    assert loaded.steps[0].observation == Observation(
        output={"delay_hours": 60, "status": "DELAYED"},
        is_error=False,
        is_mocked=True,
    )
    assert loaded.steps[0].action == ToolCall(
        tool_name="query_logistics",
        arguments={"order_id": "ORD-100"},
    )
    assert isinstance(loaded.steps[0], TrajectoryStep)

    yaml_file = tmp_path / "trace.yaml"
    yaml_file.write_text(
        """
trace_id: trace-yaml
agent_version: v1.0.0
task_input: Ping check
status: SUCCESS
steps:
  - step_id: 1
    state: RESPONDING
    tokens:
      prompt_tokens: 10
      completion_tokens: 5
""".strip(),
        encoding="utf-8",
    )
    loaded_yaml = load_trajectory(yaml_file)
    assert loaded_yaml.trace_id == "trace-yaml"
    assert loaded_yaml.total_tokens == 15


def test_load_trajectory_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trajectory(tmp_path / "missing.json")

    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TraceLoadError):
        load_trajectory(bad_json)

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TraceLoadError):
        load_trajectory(non_dict)

    bad_schema = tmp_path / "invalid_schema.json"
    payload = _sample_trace_payload()
    payload["status"] = "CRASHED"
    bad_schema.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError) as exc_info:
        load_trajectory(bad_schema)

    formatted = format_validation_error(exc_info.value, source=bad_schema)
    assert "status" in formatted
    assert str(bad_schema) in formatted
