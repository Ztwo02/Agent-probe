"""Evaluation execution engine loading declarative suites and running assertions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from agent_probe.core.schema import AssertionResult, EvalReport, Trajectory
from agent_probe.evaluators.base import BaseEvaluator
from agent_probe.evaluators.deterministic import (
    LoopDetectionGuard,
    MaxStepGuard,
    RequiredToolCallMatch,
    TokenBudgetGuard,
    TrajectoryPathMatch,
)

_EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "max_steps": MaxStepGuard,
    "max_step_guard": MaxStepGuard,
    "maxstepguard": MaxStepGuard,
    "token_budget": TokenBudgetGuard,
    "token_budget_guard": TokenBudgetGuard,
    "tokenbudgetguard": TokenBudgetGuard,
    "loop_detection": LoopDetectionGuard,
    "loop_detection_guard": LoopDetectionGuard,
    "loopdetectionguard": LoopDetectionGuard,
    "required_tool_call": RequiredToolCallMatch,
    "required_tool_call_match": RequiredToolCallMatch,
    "requiredtoolcallmatch": RequiredToolCallMatch,
    "path_match": TrajectoryPathMatch,
    "trajectory_path_match": TrajectoryPathMatch,
    "trajectorypathmatch": TrajectoryPathMatch,
}


class SuiteConfigError(ValueError):
    """Raised when an evaluation suite YAML configuration is invalid."""


class EvalRuleConfig(BaseModel):
    """Schema for a single declarative evaluator rule inside eval_suite.yaml."""

    type: str = Field(..., min_length=1)
    name: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_inline_params(cls, raw: Any) -> Any:
        if not isinstance(raw, dict):
            return raw
        data = dict(raw)
        params = dict(data.get("params") or {})
        for key in list(data.keys()):
            if key not in {"type", "name", "params"}:
                params[key] = data.pop(key)
        data["params"] = params
        return data


class EvalSuiteConfig(BaseModel):
    """Top-level schema for an evaluation suite YAML file."""

    suite_name: str = "default"
    rules: list[EvalRuleConfig] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _support_evaluators_alias(cls, raw: Any) -> Any:
        if isinstance(raw, dict) and "rules" not in raw and "evaluators" in raw:
            data = dict(raw)
            data["rules"] = data.pop("evaluators")
            return data
        return raw


def _build_evaluator(rule: EvalRuleConfig) -> BaseEvaluator:
    normalized_type = rule.type.strip().lower().replace("-", "_")
    evaluator_cls = _EVALUATOR_REGISTRY.get(normalized_type)
    if evaluator_cls is None:
        supported = sorted({cls.__name__ for cls in _EVALUATOR_REGISTRY.values()})
        raise SuiteConfigError(
            f"Unknown evaluator type '{rule.type}'. Supported evaluators: {', '.join(supported)}"
        )

    kwargs = dict(rule.params)
    if rule.name is not None:
        kwargs["rule_name"] = rule.name

    try:
        return evaluator_cls(**kwargs)
    except (TypeError, ValueError) as exc:
        raise SuiteConfigError(
            f"Invalid parameters for evaluator '{rule.type}': {exc}"
        ) from exc


class EvaluationEngine:
    """Orchestrates a collection of BaseEvaluator operators against a Trajectory."""

    def __init__(
        self, evaluators: list[BaseEvaluator], suite_name: str = "default"
    ) -> None:
        self.evaluators = list(evaluators)
        self.suite_name = suite_name

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> EvaluationEngine:
        """Build an EvaluationEngine from a parsed suite configuration dictionary."""
        suite = EvalSuiteConfig.model_validate(config)
        evaluators = [_build_evaluator(rule) for rule in suite.rules]
        return cls(evaluators=evaluators, suite_name=suite.suite_name)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> EvaluationEngine:
        """Load an evaluation suite from a YAML file."""
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Evaluation suite file not found: {path}")

        raw_text = path.read_text(encoding="utf-8")
        try:
            payload = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise SuiteConfigError(f"Invalid YAML in suite file '{path}': {exc}") from exc

        if not isinstance(payload, dict):
            raise SuiteConfigError(
                f"Expected YAML mapping at root of '{path}', got {type(payload).__name__}."
            )
        return cls.from_config(payload)

    def evaluate(self, trajectory: Trajectory) -> EvalReport:
        """Execute all configured evaluators and aggregate results into an EvalReport."""
        assertions: list[AssertionResult] = [
            evaluator.evaluate(trajectory) for evaluator in self.evaluators
        ]
        overall_passed = all(result.passed for result in assertions)
        overall_score = (
            round(sum(result.score for result in assertions) / len(assertions), 4)
            if assertions
            else 1.0
        )
        return EvalReport(
            trace_id=trajectory.trace_id,
            agent_version=trajectory.agent_version,
            overall_passed=overall_passed,
            overall_score=overall_score,
            assertions=assertions,
        )
