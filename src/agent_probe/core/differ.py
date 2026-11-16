"""Trajectory comparison and first-divergence detection engine."""

from __future__ import annotations

from typing import Any

from agent_probe.core.schema import Trajectory, TrajectoryDiff, TrajectoryStep


def _step_summary(step: TrajectoryStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "step_id": step.step_id,
        "state": step.state,
        "thought": step.thought,
        "tool_name": step.action.tool_name if step.action is not None else None,
        "arguments": step.action.arguments if step.action is not None else None,
        "observation": step.observation.model_dump() if step.observation is not None else None,
        "latency_ms": step.latency_ms,
        "tokens": step.tokens.model_dump(),
    }


def _classify_step_pair(
    b_step: TrajectoryStep | None,
    c_step: TrajectoryStep | None,
) -> tuple[bool, bool, bool, str]:
    """Return (state_match, tool_match, args_match, status) for an aligned step pair."""
    if b_step is None:
        return False, False, False, "CANDIDATE_ONLY"
    if c_step is None:
        return False, False, False, "BASELINE_ONLY"

    b_tool = b_step.action.tool_name if b_step.action is not None else None
    c_tool = c_step.action.tool_name if c_step.action is not None else None
    b_args = b_step.action.arguments if b_step.action is not None else None
    c_args = c_step.action.arguments if c_step.action is not None else None

    state_match = b_step.state == c_step.state
    tool_match = b_tool == c_tool
    args_match = b_args == c_args

    if not state_match or not tool_match:
        return state_match, tool_match, args_match, "DIVERGED"
    if not args_match:
        return state_match, tool_match, args_match, "ARGS_DIFF"
    return state_match, tool_match, args_match, "MATCH"


def diff_trajectories(baseline: Trajectory, candidate: Trajectory) -> TrajectoryDiff:
    """Compare baseline and candidate trajectories step-by-step and compute deltas."""
    max_len = max(len(baseline.steps), len(candidate.steps))
    divergence_step_id: int | None = None
    step_comparisons: list[dict[str, Any]] = []

    for idx in range(max_len):
        b_step = baseline.steps[idx] if idx < len(baseline.steps) else None
        c_step = candidate.steps[idx] if idx < len(candidate.steps) else None

        state_match, tool_match, args_match, status = _classify_step_pair(b_step, c_step)
        is_diverged = status in {"DIVERGED", "BASELINE_ONLY", "CANDIDATE_ONLY"}
        step_id = c_step.step_id if c_step is not None else b_step.step_id  # type: ignore[union-attr]

        is_first_divergence = False
        if is_diverged and divergence_step_id is None:
            divergence_step_id = step_id
            is_first_divergence = True

        b_tokens = b_step.tokens.total_tokens if b_step is not None else 0
        c_tokens = c_step.tokens.total_tokens if c_step is not None else 0
        b_latency = b_step.latency_ms if b_step is not None else 0.0
        c_latency = c_step.latency_ms if c_step is not None else 0.0

        step_comparisons.append(
            {
                "index": idx,
                "step_id": step_id,
                "status": status,
                "state_match": state_match,
                "tool_match": tool_match,
                "args_match": args_match,
                "is_diverged": is_diverged,
                "is_first_divergence": is_first_divergence,
                "token_delta": c_tokens - b_tokens,
                "latency_delta_ms": round(c_latency - b_latency, 6),
                "baseline": _step_summary(b_step),
                "candidate": _step_summary(c_step),
            }
        )

    return TrajectoryDiff(
        baseline_trace_id=baseline.trace_id,
        candidate_trace_id=candidate.trace_id,
        divergence_step_id=divergence_step_id,
        step_delta=candidate.total_steps - baseline.total_steps,
        token_delta=candidate.total_tokens - baseline.total_tokens,
        latency_delta_ms=round(candidate.total_latency_ms - baseline.total_latency_ms, 6),
        step_comparisons=step_comparisons,
    )
