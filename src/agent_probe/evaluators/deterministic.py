"""Deterministic trajectory evaluation guards and path matchers."""

from __future__ import annotations

from typing import Any, Literal

from agent_probe.core.schema import AssertionResult, Trajectory
from agent_probe.evaluators.base import BaseEvaluator


def _is_subset_match(actual: Any, expected: Any) -> bool:
    """Return True if expected is recursively a subset of actual."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        return all(
            key in actual and _is_subset_match(actual[key], val)
            for key, val in expected.items()
        )
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    return actual == expected


class MaxStepGuard(BaseEvaluator):
    """Verify total trajectory steps do not exceed an upper bound."""

    default_rule_name = "max_step_guard"

    def __init__(self, max_steps: int, rule_name: str | None = None) -> None:
        if max_steps < 0:
            raise ValueError("max_steps must be >= 0")
        super().__init__(rule_name=rule_name)
        self.max_steps = max_steps

    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        actual = trajectory.total_steps
        if actual <= self.max_steps:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=f"Total steps ({actual}) within limit ({self.max_steps}).",
                failed_step_ids=[],
            )
        overflow_ids = [step.step_id for step in trajectory.steps[self.max_steps :]]
        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=f"Total steps ({actual}) exceeded max_steps ({self.max_steps}).",
            failed_step_ids=overflow_ids,
        )


class TokenBudgetGuard(BaseEvaluator):
    """Verify total token consumption across all steps stays within budget."""

    default_rule_name = "token_budget_guard"

    def __init__(self, max_tokens: int, rule_name: str | None = None) -> None:
        if max_tokens < 0:
            raise ValueError("max_tokens must be >= 0")
        super().__init__(rule_name=rule_name)
        self.max_tokens = max_tokens

    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        actual = trajectory.total_tokens
        if actual <= self.max_tokens:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=f"Total tokens ({actual}) within budget ({self.max_tokens}).",
                failed_step_ids=[],
            )

        running_tokens = 0
        failed_ids: list[int] = []
        for step in trajectory.steps:
            running_tokens += step.tokens.total_tokens
            if running_tokens > self.max_tokens:
                failed_ids.append(step.step_id)

        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=f"Total tokens ({actual}) exceeded budget ({self.max_tokens}).",
            failed_step_ids=failed_ids,
        )


class LoopDetectionGuard(BaseEvaluator):
    """Detect consecutive identical tool calls (same tool_name and arguments)."""

    default_rule_name = "loop_detection_guard"

    def __init__(
        self, max_consecutive_repeats: int = 2, rule_name: str | None = None
    ) -> None:
        if max_consecutive_repeats < 1:
            raise ValueError("max_consecutive_repeats must be >= 1")
        super().__init__(rule_name=rule_name)
        self.max_consecutive_repeats = max_consecutive_repeats

    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        last_signature: tuple[str, dict[str, Any]] | None = None
        streak = 0
        failed_ids: list[int] = []
        looped_tools: list[str] = []

        for step in trajectory.steps:
            if step.action is None:
                continue

            signature = (step.action.tool_name, step.action.arguments)
            if last_signature is not None and signature == last_signature:
                streak += 1
            else:
                last_signature = signature
                streak = 1

            if streak > self.max_consecutive_repeats:
                failed_ids.append(step.step_id)
                if step.action.tool_name not in looped_tools:
                    looped_tools.append(step.action.tool_name)

        if not failed_ids:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=(
                    "No consecutive identical tool calls exceeded limit "
                    f"({self.max_consecutive_repeats})."
                ),
                failed_step_ids=[],
            )

        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=(
                "Detected repeated identical tool call loop "
                f"(limit={self.max_consecutive_repeats}) for tool(s): "
                f"{', '.join(looped_tools)}."
            ),
            failed_step_ids=failed_ids,
        )


class RequiredToolCallMatch(BaseEvaluator):
    """Assert that a specified tool was called at least once with matching argument subset."""

    default_rule_name = "required_tool_call_match"

    def __init__(
        self,
        tool_name: str,
        expected_args_subset: dict[str, Any] | None = None,
        rule_name: str | None = None,
    ) -> None:
        if not tool_name:
            raise ValueError("tool_name must not be empty")
        super().__init__(rule_name=rule_name)
        self.tool_name = tool_name
        self.expected_args_subset = expected_args_subset

    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        matching_steps = [
            step
            for step in trajectory.steps
            if step.action is not None and step.action.tool_name == self.tool_name
        ]
        if not matching_steps:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=False,
                score=0.0,
                reason=f"Required tool '{self.tool_name}' was never called.",
                failed_step_ids=[],
            )

        if not self.expected_args_subset:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=f"Required tool '{self.tool_name}' was called.",
                failed_step_ids=[],
            )

        for step in matching_steps:
            if step.action is not None and _is_subset_match(
                step.action.arguments, self.expected_args_subset
            ):
                return AssertionResult(
                    rule_name=self.rule_name,
                    passed=True,
                    score=1.0,
                    reason=(
                        f"Required tool '{self.tool_name}' called at step {step.step_id} "
                        "with expected arguments."
                    ),
                    failed_step_ids=[],
                )

        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=(
                f"Tool '{self.tool_name}' was called, but arguments did not match "
                f"expected subset {self.expected_args_subset}."
            ),
            failed_step_ids=[step.step_id for step in matching_steps],
        )


class TrajectoryPathMatch(BaseEvaluator):
    """Verify tool or state sequence adheres to an expected subsequence or exact order."""

    default_rule_name = "trajectory_path_match"

    def __init__(
        self,
        expected_sequence: list[str],
        mode: Literal["subsequence", "exact"] = "subsequence",
        target: Literal["tool", "state"] = "tool",
        rule_name: str | None = None,
    ) -> None:
        if mode not in ("subsequence", "exact"):
            raise ValueError(f"Unsupported mode: {mode}")
        if target not in ("tool", "state"):
            raise ValueError(f"Unsupported target: {target}")
        super().__init__(rule_name=rule_name)
        self.expected_sequence = list(expected_sequence)
        self.mode = mode
        self.target = target

    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        if self.target == "tool":
            actual_items = [
                (step.step_id, step.action.tool_name)
                for step in trajectory.steps
                if step.action is not None
            ]
        else:
            actual_items = [(step.step_id, step.state) for step in trajectory.steps]

        actual_sequence = [item[1] for item in actual_items]

        if self.mode == "subsequence":
            return self._evaluate_subsequence(actual_sequence)
        return self._evaluate_exact(actual_items, actual_sequence)

    def _evaluate_subsequence(self, actual_sequence: list[str]) -> AssertionResult:
        exp_idx = 0
        for token in actual_sequence:
            if exp_idx < len(self.expected_sequence) and token == self.expected_sequence[exp_idx]:
                exp_idx += 1

        if exp_idx == len(self.expected_sequence):
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=(
                    f"Actual {self.target} sequence matched expected subsequence "
                    f"{self.expected_sequence}."
                ),
                failed_step_ids=[],
            )

        missing = self.expected_sequence[exp_idx]
        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=(
                f"Actual {self.target} sequence {actual_sequence} missing expected "
                f"subsequence element '{missing}' from {self.expected_sequence}."
            ),
            failed_step_ids=[],
        )

    def _evaluate_exact(
        self, actual_items: list[tuple[int, str]], actual_sequence: list[str]
    ) -> AssertionResult:
        if actual_sequence == self.expected_sequence:
            return AssertionResult(
                rule_name=self.rule_name,
                passed=True,
                score=1.0,
                reason=f"Actual {self.target} sequence matched expected sequence exactly.",
                failed_step_ids=[],
            )

        failed_ids = [
            step_id
            for idx, (step_id, value) in enumerate(actual_items)
            if idx >= len(self.expected_sequence) or value != self.expected_sequence[idx]
        ]
        return AssertionResult(
            rule_name=self.rule_name,
            passed=False,
            score=0.0,
            reason=(
                f"Actual {self.target} sequence {actual_sequence} did not match "
                f"expected exact sequence {self.expected_sequence}."
            ),
            failed_step_ids=failed_ids,
        )
