"""Abstract base class for trajectory evaluation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

from agent_probe.core.schema import AssertionResult, Trajectory


class BaseEvaluator(ABC):
    """Base interface implemented by all trajectory evaluators."""

    default_rule_name: str = "base_evaluator"

    def __init__(self, rule_name: str | None = None) -> None:
        self.rule_name = rule_name or self.default_rule_name

    @abstractmethod
    def evaluate(self, trajectory: Trajectory) -> AssertionResult:
        """Evaluate a trajectory and return a deterministic or heuristic assertion result."""
