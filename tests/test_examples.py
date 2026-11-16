"""Integration tests verifying bundled example traces and eval suite."""

from __future__ import annotations

from pathlib import Path

from agent_probe.adapters.json_loader import load_trajectory
from agent_probe.core.differ import diff_trajectories
from agent_probe.core.engine import EvaluationEngine

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_example_dataset_baseline_passes_and_candidate_fails() -> None:
    baseline = load_trajectory(EXAMPLES_DIR / "baseline_trace.json")
    candidate = load_trajectory(EXAMPLES_DIR / "candidate_trace_loop.json")
    engine = EvaluationEngine.from_yaml(EXAMPLES_DIR / "eval_suite.yaml")

    base_report = engine.evaluate(baseline)
    cand_report = engine.evaluate(candidate)
    diff = diff_trajectories(baseline, candidate)

    assert base_report.overall_passed is True
    assert base_report.overall_score == 1.0
    assert len(base_report.assertions) == 5

    assert cand_report.overall_passed is False
    assert cand_report.overall_score == 0.0
    assert all(not assertion.passed for assertion in cand_report.assertions)

    loop_assertion = next(
        a for a in cand_report.assertions if a.rule_name == "no_consecutive_tool_loop"
    )
    assert loop_assertion.failed_step_ids == [4, 5]

    assert diff.divergence_step_id == 2
    assert diff.step_delta == 2
    assert diff.token_delta == 3750 - 1140
