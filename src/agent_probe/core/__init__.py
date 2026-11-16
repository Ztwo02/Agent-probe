"""Core domain layer: schemas, evaluation engine, and trajectory differ."""

from agent_probe.core.differ import diff_trajectories
from agent_probe.core.engine import EvaluationEngine, SuiteConfigError
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

__all__ = [
    "AssertionResult",
    "EvalReport",
    "EvaluationEngine",
    "Observation",
    "SuiteConfigError",
    "TokenUsage",
    "ToolCall",
    "Trajectory",
    "TrajectoryDiff",
    "TrajectoryStep",
    "diff_trajectories",
]
