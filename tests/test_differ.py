"""Unit tests for trajectory comparison and divergence detection."""

from __future__ import annotations

import pytest

from agent_probe.core.differ import diff_trajectories
from agent_probe.core.schema import (
    Observation,
    TokenUsage,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)


def _step(
    step_id: int,
    state: str = "TOOL_EXECUTION",
    tool_name: str | None = None,
    arguments: dict | None = None,
    total_tokens: int = 100,
    latency_ms: float = 50.0,
) -> TrajectoryStep:
    action = (
        ToolCall(tool_name=tool_name, arguments=arguments or {})
        if tool_name is not None
        else None
    )
    obs = Observation(output={"ok": True}) if action is not None else None
    return TrajectoryStep(
        step_id=step_id,
        state=state,
        thought=f"thought-{step_id}",
        action=action,
        observation=obs,
        latency_ms=latency_ms,
        tokens=TokenUsage(total_tokens=total_tokens),
    )


def _traj(trace_id: str, steps: list[TrajectoryStep], status: str = "SUCCESS") -> Trajectory:
    return Trajectory(
        trace_id=trace_id,
        agent_version="v1",
        task_input="Check order",
        final_output="Done",
        status=status,  # type: ignore[arg-type]
        steps=steps,
    )


def test_diff_identical_trajectories() -> None:
    steps = [
        _step(1, state="TOOL_EXECUTION", tool_name="query_logistics", arguments={"id": "1"}),
        _step(2, state="RESPONDING", tool_name=None),
    ]
    baseline = _traj("base-1", steps)
    candidate = _traj("cand-1", steps)

    diff = diff_trajectories(baseline, candidate)
    assert diff.divergence_step_id is None
    assert diff.step_delta == 0
    assert diff.token_delta == 0
    assert diff.latency_delta_ms == 0.0
    assert [c["status"] for c in diff.step_comparisons] == ["MATCH", "MATCH"]


def test_diff_detects_state_and_tool_divergence() -> None:
    baseline = _traj(
        "base-2",
        [
            _step(
                1,
                tool_name="query_logistics",
                arguments={"id": "1"},
                total_tokens=100,
                latency_ms=40.0,
            ),
            _step(2, state="PLANNING", tool_name=None, total_tokens=120, latency_ms=30.0),
            _step(
                3,
                tool_name="issue_coupon",
                arguments={"amount": 20},
                total_tokens=150,
                latency_ms=60.0,
            ),
        ],
    )
    candidate = _traj(
        "cand-2",
        [
            _step(
                1,
                tool_name="query_logistics",
                arguments={"id": "1"},
                total_tokens=110,
                latency_ms=45.0,
            ),
            _step(
                2,
                tool_name="query_logistics",
                arguments={"id": "1"},
                total_tokens=200,
                latency_ms=90.0,
            ),
            _step(
                3,
                tool_name="query_logistics",
                arguments={"id": "1"},
                total_tokens=200,
                latency_ms=90.0,
            ),
            _step(4, state="ERROR", tool_name=None, total_tokens=100, latency_ms=20.0),
        ],
        status="LOOP_ABORTED",
    )

    diff = diff_trajectories(baseline, candidate)
    assert diff.divergence_step_id == 2
    assert diff.step_delta == 1
    assert diff.token_delta == 240  # 610 - 370
    assert diff.latency_delta_ms == pytest.approx(115.0)  # 245 - 130
    assert diff.step_comparisons[0]["status"] == "MATCH"
    assert diff.step_comparisons[1]["status"] == "DIVERGED"
    assert diff.step_comparisons[1]["is_first_divergence"] is True
    assert diff.step_comparisons[2]["is_first_divergence"] is False
    assert diff.step_comparisons[3]["status"] == "CANDIDATE_ONLY"


def test_diff_handles_argument_diff_without_structural_divergence() -> None:
    baseline = _traj(
        "base-3",
        [_step(1, tool_name="query_order", arguments={"order_id": "A1"})],
    )
    candidate = _traj(
        "cand-3",
        [_step(1, tool_name="query_order", arguments={"order_id": "A2"})],
    )

    diff = diff_trajectories(baseline, candidate)
    assert diff.divergence_step_id is None
    assert diff.step_comparisons[0]["status"] == "ARGS_DIFF"
    assert diff.step_comparisons[0]["state_match"] is True
    assert diff.step_comparisons[0]["tool_match"] is True
    assert diff.step_comparisons[0]["args_match"] is False


def test_diff_handles_shorter_candidate_and_empty_traces() -> None:
    baseline = _traj(
        "base-4",
        [
            _step(1, tool_name="query_order", total_tokens=200, latency_ms=100.0),
            _step(2, state="RESPONDING", total_tokens=100, latency_ms=50.0),
        ],
    )
    candidate = _traj(
        "cand-4",
        [_step(1, tool_name="query_order", total_tokens=150, latency_ms=80.0)],
    )

    diff = diff_trajectories(baseline, candidate)
    assert diff.divergence_step_id == 2
    assert diff.step_delta == -1
    assert diff.token_delta == -150
    assert diff.latency_delta_ms == pytest.approx(-70.0)
    assert diff.step_comparisons[1]["status"] == "BASELINE_ONLY"

    empty_diff = diff_trajectories(_traj("e1", []), _traj("e2", []))
    assert empty_diff.divergence_step_id is None
    assert empty_diff.step_comparisons == []
