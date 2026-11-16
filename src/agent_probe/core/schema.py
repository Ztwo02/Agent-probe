"""Pydantic v2 data contracts for Agent-Probe trajectories, evaluations, and diffs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, model_validator


class TokenUsage(BaseModel):
    """Token consumption metrics for a single trajectory step."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _compute_or_validate_total(self) -> TokenUsage:
        expected = self.prompt_tokens + self.completion_tokens
        if "total_tokens" not in self.model_fields_set:
            self.total_tokens = expected
            return self

        has_breakdown = (
            "prompt_tokens" in self.model_fields_set
            or "completion_tokens" in self.model_fields_set
            or expected > 0
        )
        if has_breakdown and self.total_tokens != expected:
            raise ValueError(
                f"total_tokens ({self.total_tokens}) must equal "
                f"prompt_tokens + completion_tokens ({expected})"
            )
        return self


class ToolCall(BaseModel):
    """Tool invocation details executed during an agent step."""

    tool_name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class Observation(BaseModel):
    """Environment or tool execution result returned to the agent."""

    output: Any
    is_error: bool = False
    is_mocked: bool = False


class TrajectoryStep(BaseModel):
    """Single step within an agent execution trajectory."""

    step_id: int = Field(..., ge=0)
    state: str = Field(..., min_length=1)
    thought: str | None = None
    action: ToolCall | None = None
    observation: Observation | None = None
    latency_ms: float = Field(default=0.0, ge=0.0)
    tokens: TokenUsage


class Trajectory(BaseModel):
    """Standardized agent execution trajectory contract."""

    trace_id: str = Field(..., min_length=1)
    agent_version: str = Field(..., min_length=1)
    task_input: str
    final_output: str | None = None
    status: Literal["SUCCESS", "FAILED", "TIMEOUT", "LOOP_ABORTED"]
    steps: list[TrajectoryStep]

    @computed_field
    @property
    def total_steps(self) -> int:
        """Total number of steps in the trajectory."""
        return len(self.steps)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Cumulative token count across all trajectory steps."""
        return sum(step.tokens.total_tokens for step in self.steps)

    @computed_field
    @property
    def total_latency_ms(self) -> float:
        """Cumulative latency in milliseconds across all trajectory steps."""
        return round(sum(step.latency_ms for step in self.steps), 6)

    @computed_field
    @property
    def tool_sequence(self) -> list[str]:
        """Ordered list of tool names invoked across the trajectory."""
        return [step.action.tool_name for step in self.steps if step.action is not None]


class AssertionResult(BaseModel):
    """Outcome of a single evaluator assertion against a trajectory."""

    rule_name: str = Field(..., min_length=1)
    passed: bool
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    failed_step_ids: list[int] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Aggregated evaluation report for a trajectory across an evaluator suite."""

    trace_id: str = Field(..., min_length=1)
    agent_version: str = Field(..., min_length=1)
    overall_passed: bool
    overall_score: float = Field(..., ge=0.0, le=1.0)
    assertions: list[AssertionResult]


class TrajectoryDiff(BaseModel):
    """Step-aligned comparison between a baseline and candidate trajectory."""

    baseline_trace_id: str = Field(..., min_length=1)
    candidate_trace_id: str = Field(..., min_length=1)
    divergence_step_id: int | None = None
    step_delta: int
    token_delta: int
    latency_delta_ms: float
    step_comparisons: list[dict[str, Any]]
