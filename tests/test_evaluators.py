"""Unit tests for deterministic evaluators and the EvaluationEngine."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_probe.core.engine import EvaluationEngine, SuiteConfigError
from agent_probe.core.schema import (
    Observation,
    TokenUsage,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)
from agent_probe.evaluators import (
    LoopDetectionGuard,
    MaxStepGuard,
    RequiredToolCallMatch,
    TokenBudgetGuard,
    TrajectoryPathMatch,
)


def _make_step(
    step_id: int,
    state: str = "TOOL_EXECUTION",
    tool_name: str | None = None,
    arguments: dict | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    is_error: bool = False,
) -> TrajectoryStep:
    action = (
        ToolCall(tool_name=tool_name, arguments=arguments or {})
        if tool_name is not None
        else None
    )
    obs = Observation(output={"ok": not is_error}, is_error=is_error) if action else None
    return TrajectoryStep(
        step_id=step_id,
        state=state,
        thought=f"Step {step_id}",
        action=action,
        observation=obs,
        latency_ms=50.0,
        tokens=TokenUsage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def _make_trajectory(steps: list[TrajectoryStep], status: str = "SUCCESS") -> Trajectory:
    return Trajectory(
        trace_id="trace-eval-test",
        agent_version="v1.0",
        task_input="Handle delayed order",
        final_output="Done",
        status=status,  # type: ignore[arg-type]
        steps=steps,
    )


def test_max_step_guard() -> None:
    traj = _make_trajectory([_make_step(1), _make_step(2), _make_step(3), _make_step(4)])
    assert MaxStepGuard(4).evaluate(traj).passed is True
    assert MaxStepGuard(5).evaluate(traj).passed is True

    failed = MaxStepGuard(2).evaluate(traj)
    assert failed.passed is False
    assert failed.score == 0.0
    assert failed.failed_step_ids == [3, 4]

    with pytest.raises(ValueError):
        MaxStepGuard(-1)


def test_token_budget_guard() -> None:
    traj = _make_trajectory(
        [
            _make_step(1, prompt_tokens=300, completion_tokens=100),  # 400
            _make_step(2, prompt_tokens=300, completion_tokens=100),  # 800
            _make_step(3, prompt_tokens=300, completion_tokens=100),  # 1200
        ]
    )
    assert TokenBudgetGuard(1200).evaluate(traj).passed is True

    failed = TokenBudgetGuard(900).evaluate(traj)
    assert failed.passed is False
    assert failed.score == 0.0
    assert failed.failed_step_ids == [3]

    failed_early = TokenBudgetGuard(350).evaluate(traj)
    assert failed_early.failed_step_ids == [1, 2, 3]

    with pytest.raises(ValueError):
        TokenBudgetGuard(-10)


def test_loop_detection_guard() -> None:
    # 2 consecutive identical calls are allowed when max_consecutive_repeats=2
    ok_traj = _make_trajectory(
        [
            _make_step(1, tool_name="query_logistics", arguments={"order_id": "A1"}),
            _make_step(2, tool_name="query_logistics", arguments={"order_id": "A1"}),
            _make_step(3, tool_name="query_logistics", arguments={"order_id": "A2"}),
            _make_step(4, tool_name="query_logistics", arguments={"order_id": "A2"}),
        ]
    )
    assert LoopDetectionGuard(max_consecutive_repeats=2).evaluate(ok_traj).passed is True

    # 3 consecutive identical calls even with an intervening PLANNING step triggers loop
    loop_traj = _make_trajectory(
        [
            _make_step(1, tool_name="query_logistics", arguments={"order_id": "A1"}),
            _make_step(2, state="PLANNING", tool_name=None),
            _make_step(3, tool_name="query_logistics", arguments={"order_id": "A1"}),
            _make_step(4, tool_name="query_logistics", arguments={"order_id": "A1"}),
            _make_step(5, tool_name="query_logistics", arguments={"order_id": "A1"}),
        ],
        status="LOOP_ABORTED",
    )
    res = LoopDetectionGuard(max_consecutive_repeats=2).evaluate(loop_traj)
    assert res.passed is False
    assert res.score == 0.0
    assert res.failed_step_ids == [4, 5]

    with pytest.raises(ValueError):
        LoopDetectionGuard(max_consecutive_repeats=0)


def test_required_tool_call_match() -> None:
    traj = _make_trajectory(
        [
            _make_step(
                1,
                tool_name="issue_coupon",
                arguments={
                    "order_id": "ORD-1",
                    "reason": "delay_48h",
                    "meta": {"amount": 20, "currency": "CNY"},
                },
            )
        ]
    )

    assert RequiredToolCallMatch("issue_coupon").evaluate(traj).passed is True
    assert (
        RequiredToolCallMatch(
            "issue_coupon",
            expected_args_subset={"reason": "delay_48h", "meta": {"amount": 20}},
        )
        .evaluate(traj)
        .passed
        is True
    )

    wrong_args = RequiredToolCallMatch(
        "issue_coupon", expected_args_subset={"reason": "damaged_item"}
    ).evaluate(traj)
    assert wrong_args.passed is False
    assert wrong_args.failed_step_ids == [1]

    # bool vs int strictness check (True != 1)
    bool_traj = _make_trajectory([_make_step(1, tool_name="flag", arguments={"count": 1})])
    assert (
        RequiredToolCallMatch("flag", expected_args_subset={"count": True})
        .evaluate(bool_traj)
        .passed
        is False
    )

    missing_tool = RequiredToolCallMatch("send_email").evaluate(traj)
    assert missing_tool.passed is False
    assert missing_tool.failed_step_ids == []


def test_trajectory_path_match() -> None:
    traj = _make_trajectory(
        [
            _make_step(1, state="PLANNING", tool_name=None),
            _make_step(2, state="TOOL_EXECUTION", tool_name="query_order"),
            _make_step(3, state="TOOL_EXECUTION", tool_name="query_logistics"),
            _make_step(4, state="TOOL_EXECUTION", tool_name="issue_refund"),
            _make_step(5, state="RESPONDING", tool_name=None),
        ]
    )

    assert (
        TrajectoryPathMatch(["query_order", "issue_refund"], mode="subsequence")
        .evaluate(traj)
        .passed
        is True
    )
    assert (
        TrajectoryPathMatch(["issue_refund", "query_order"], mode="subsequence")
        .evaluate(traj)
        .passed
        is False
    )
    assert (
        TrajectoryPathMatch(
            ["query_order", "query_logistics", "issue_refund"], mode="exact"
        )
        .evaluate(traj)
        .passed
        is True
    )

    exact_fail = TrajectoryPathMatch(
        ["query_order", "issue_refund"], mode="exact"
    ).evaluate(traj)
    assert exact_fail.passed is False
    assert exact_fail.failed_step_ids == [3, 4]

    state_match = TrajectoryPathMatch(
        ["PLANNING", "TOOL_EXECUTION", "RESPONDING"],
        mode="subsequence",
        target="state",
    ).evaluate(traj)
    assert state_match.passed is True


def test_evaluation_engine_from_yaml(tmp_path: Path) -> None:
    suite_file = tmp_path / "eval_suite.yaml"
    suite_file.write_text(
        """
suite_name: test-suite
rules:
  - type: max_steps
    name: max_4_steps
    params:
      max_steps: 4
  - type: TokenBudgetGuard
    max_tokens: 1000
  - type: loop_detection
    params:
      max_consecutive_repeats: 2
  - type: required_tool_call
    params:
      tool_name: issue_coupon
      expected_args_subset:
        reason: delay
  - type: path_match
    params:
      expected_sequence: ["query_logistics", "issue_coupon"]
      mode: subsequence
""".strip(),
        encoding="utf-8",
    )

    engine = EvaluationEngine.from_yaml(suite_file)
    assert engine.suite_name == "test-suite"
    assert len(engine.evaluators) == 5

    good_traj = _make_trajectory(
        [
            _make_step(1, tool_name="query_logistics", arguments={"order_id": "1"}),
            _make_step(2, tool_name="issue_coupon", arguments={"reason": "delay"}),
        ]
    )
    pass_report = engine.evaluate(good_traj)
    assert pass_report.overall_passed is True
    assert pass_report.overall_score == 1.0
    assert len(pass_report.assertions) == 5

    bad_traj = _make_trajectory(
        [
            _make_step(1, tool_name="query_logistics", arguments={"id": 1}),
            _make_step(2, tool_name="query_logistics", arguments={"id": 1}),
            _make_step(3, tool_name="query_logistics", arguments={"id": 1}),
            _make_step(4, tool_name="query_logistics", arguments={"id": 1}),
            _make_step(5, tool_name="query_logistics", arguments={"id": 1}),
        ]
    )
    fail_report = engine.evaluate(bad_traj)
    assert fail_report.overall_passed is False
    assert fail_report.overall_score == 0.2  # Only token_budget (5*150=750 <= 1000) passed


def test_evaluation_engine_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        EvaluationEngine.from_yaml(tmp_path / "missing.yaml")

    bad_rule = tmp_path / "bad_rule.yaml"
    bad_rule.write_text(
        """
rules:
  - type: nonexistent_guard
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(SuiteConfigError):
        EvaluationEngine.from_yaml(bad_rule)

    bad_params = tmp_path / "bad_params.yaml"
    bad_params.write_text(
        """
rules:
  - type: max_steps
    params:
      max_steps: -5
""".strip(),
        encoding="utf-8",
    )
    with pytest.raises(SuiteConfigError):
        EvaluationEngine.from_yaml(bad_params)
