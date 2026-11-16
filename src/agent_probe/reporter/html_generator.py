"""Single-file self-contained HTML report generator powered by Jinja2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from agent_probe.core.schema import EvalReport, Trajectory, TrajectoryDiff, TrajectoryStep

_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def _pretty_json(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def _format_step_payload(step: TrajectoryStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "step_id": step.step_id,
        "state": step.state,
        "thought": step.thought,
        "tool_name": step.action.tool_name if step.action is not None else None,
        "arguments_pretty": (
            _pretty_json(step.action.arguments) if step.action is not None else None
        ),
        "observation": step.observation.model_dump() if step.observation is not None else None,
        "observation_pretty": (
            _pretty_json(step.observation.output) if step.observation is not None else None
        ),
        "latency_ms": step.latency_ms,
        "tokens": step.tokens.model_dump(),
    }


def _extract_failure_sets(report: EvalReport | None) -> tuple[set[int], set[int]]:
    failed_steps: set[int] = set()
    loop_steps: set[int] = set()
    if report is None:
        return failed_steps, loop_steps

    for assertion in report.assertions:
        if not assertion.passed:
            failed_steps.update(assertion.failed_step_ids)
            if "loop" in assertion.rule_name.lower():
                loop_steps.update(assertion.failed_step_ids)
    return failed_steps, loop_steps


def render_html_report(
    *,
    baseline: Trajectory | None = None,
    candidate: Trajectory | None = None,
    diff: TrajectoryDiff | None = None,
    baseline_report: EvalReport | None = None,
    candidate_report: EvalReport | None = None,
    single_trace: Trajectory | None = None,
    single_report: EvalReport | None = None,
    title: str | None = None,
) -> str:
    """Render a self-contained single-file HTML report string."""
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template("report.html.j2")

    if diff is not None:
        if baseline is None or candidate is None:
            raise ValueError("Both baseline and candidate trajectories are required for diff HTML.")
        primary_report = candidate_report
        failed_steps, loop_steps = _extract_failure_sets(candidate_report)

        base_by_idx = {idx: step for idx, step in enumerate(baseline.steps)}
        cand_by_idx = {idx: step for idx, step in enumerate(candidate.steps)}
        timeline_rows: list[dict[str, Any]] = []

        for item in diff.step_comparisons:
            idx = item["index"]
            b_step = base_by_idx.get(idx)
            c_step = cand_by_idx.get(idx)
            c_id = c_step.step_id if c_step is not None else None
            is_cand_failed = c_id is not None and c_id in failed_steps
            is_loop = c_id is not None and c_id in loop_steps
            c_error = (
                c_step is not None
                and c_step.observation is not None
                and c_step.observation.is_error
            )
            has_issue = bool(item["is_diverged"] or is_cand_failed or c_error)
            is_fork_point = bool(item.get("is_fork_point", False))
            timeline_rows.append(
                {
                    "status": item["status"],
                    "is_diverged": item["is_diverged"],
                    "is_first_divergence": item["is_first_divergence"],
                    "is_fork_point": is_fork_point,
                    "is_highlight_point": bool(is_fork_point or item["is_first_divergence"]),
                    "step_id": item["step_id"],
                    "token_delta": item["token_delta"],
                    "is_candidate_failed": is_cand_failed,
                    "is_loop_step": is_loop,
                    "has_issue": has_issue,
                    "baseline": _format_step_payload(b_step),
                    "candidate": _format_step_payload(c_step),
                }
            )

        token_change_pct = (
            (diff.token_delta / baseline.total_tokens) * 100.0
            if baseline.total_tokens > 0
            else 0.0
        )
        passed_count = (
            sum(1 for a in primary_report.assertions if a.passed)
            if primary_report is not None
            else 0
        )
        default_title = f"Trajectory Diff: {baseline.trace_id} vs {candidate.trace_id}"
        context = {
            "report_title": title or default_title,
            "task_input": candidate.task_input or baseline.task_input,
            "baseline": baseline,
            "candidate": candidate,
            "diff": diff,
            "baseline_report": baseline_report,
            "primary_report": primary_report,
            "passed_count": passed_count,
            "token_change_pct": token_change_pct,
            "timeline_rows": timeline_rows,
            "fork_step_id": diff.fork_step_id,
            "replayed_count": sum(
                1
                for s in candidate.steps
                if s.observation is not None and s.observation.is_replayed
            ),
            "mocked_count": sum(
                1 for s in candidate.steps if s.observation is not None and s.observation.is_mocked
            ),
            "single_trace": None,
            "single_rows": [],
        }
    elif single_trace is not None:
        failed_steps, loop_steps = _extract_failure_sets(single_report)
        single_rows = []
        for step in single_trace.steps:
            is_failed = step.step_id in failed_steps
            is_loop = step.step_id in loop_steps
            is_err = step.observation is not None and step.observation.is_error
            single_rows.append(
                {
                    "step": _format_step_payload(step),
                    "is_failed": is_failed,
                    "is_loop_step": is_loop,
                    "has_issue": bool(is_failed or is_err),
                    "is_fork_point": (
                        single_trace.fork_step_id is not None
                        and step.step_id == single_trace.fork_step_id
                    ),
                }
            )
        passed_count = (
            sum(1 for a in single_report.assertions if a.passed)
            if single_report is not None
            else 0
        )
        context = {
            "report_title": title or f"Trajectory Evaluation: {single_trace.trace_id}",
            "task_input": single_trace.task_input,
            "baseline": None,
            "candidate": None,
            "diff": None,
            "baseline_report": None,
            "primary_report": single_report,
            "passed_count": passed_count,
            "token_change_pct": 0.0,
            "timeline_rows": [],
            "fork_step_id": single_trace.fork_step_id,
            "replayed_count": 0,
            "mocked_count": 0,
            "single_trace": single_trace,
            "single_rows": single_rows,
        }
    else:
        raise ValueError("Either diff or single_trace must be provided to render HTML report.")

    return template.render(**context)


def generate_html_report(output_path: str | Path, **kwargs: Any) -> Path:
    """Render and write a single-file self-contained HTML report to disk."""
    html_content = render_html_report(**kwargs)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_content, encoding="utf-8")
    return out
