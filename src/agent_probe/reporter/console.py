"""Rich terminal output formatters for evaluation reports and trajectory diffs."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent_probe.core.schema import EvalReport, Trajectory, TrajectoryDiff


def _format_status_badge(passed: bool) -> str:
    return "[bold green]PASS[/bold green]" if passed else "[bold red]FAIL[/bold red]"


def _collect_failed_steps(report: EvalReport | None) -> set[int]:
    if report is None:
        return set()
    failed: set[int] = set()
    for assertion in report.assertions:
        if not assertion.passed:
            failed.update(assertion.failed_step_ids)
    return failed


def _render_assertions_table(report: EvalReport, target_console: Console) -> None:
    table = Table(title=f"Evaluation Assertions ({report.trace_id})", expand=True)
    table.add_column("Rule", style="cyan", no_wrap=True)
    table.add_column("Result", justify="center", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Failed Steps", justify="center", no_wrap=True)
    table.add_column("Reason", style="white")

    for assertion in report.assertions:
        steps_str = (
            ", ".join(str(sid) for sid in assertion.failed_step_ids)
            if assertion.failed_step_ids
            else "-"
        )
        table.add_row(
            assertion.rule_name,
            _format_status_badge(assertion.passed),
            f"{assertion.score:.2f}",
            f"[red]{steps_str}[/red]" if assertion.failed_step_ids else "-",
            assertion.reason,
        )

    target_console.print(table)


def render_eval_report(
    report: EvalReport,
    trajectory: Trajectory | None = None,
    console: Console | None = None,
) -> None:
    """Render a single-trajectory evaluation report to the terminal using Rich."""
    target_console = console or Console()
    status_label = _format_status_badge(report.overall_passed)
    summary_lines = [
        f"[bold]Trace ID:[/bold] {report.trace_id}    "
        f"[bold]Agent Version:[/bold] {report.agent_version}",
        f"[bold]Overall Result:[/bold] {status_label}    "
        f"[bold]Overall Score:[/bold] {report.overall_score * 100:.1f}%",
    ]
    if trajectory is not None:
        summary_lines.append(
            f"[bold]Execution Status:[/bold] {trajectory.status}    "
            f"[bold]Steps:[/bold] {trajectory.total_steps}    "
            f"[bold]Tokens:[/bold] {trajectory.total_tokens}    "
            f"[bold]Latency:[/bold] {trajectory.total_latency_ms:.1f} ms"
        )

    border_style = "green" if report.overall_passed else "red"
    target_console.print(
        Panel(
            "\n".join(summary_lines),
            title="Agent-Probe Evaluation Report",
            border_style=border_style,
        )
    )
    _render_assertions_table(report, target_console)


def render_diff_report(
    diff: TrajectoryDiff,
    baseline: Trajectory,
    candidate: Trajectory,
    baseline_report: EvalReport | None = None,
    candidate_report: EvalReport | None = None,
    console: Console | None = None,
) -> None:
    """Render a Baseline vs Candidate trajectory diff and evaluation summary."""
    target_console = console or Console()

    token_pct = (
        (diff.token_delta / baseline.total_tokens) * 100.0
        if baseline.total_tokens > 0
        else 0.0
    )
    divergence_text = (
        f"[bold red]Step {diff.divergence_step_id}[/bold red]"
        if diff.divergence_step_id is not None
        else "[bold green]None (Aligned)[/bold green]"
    )

    lines = [
        f"[bold]Baseline:[/bold]  {baseline.trace_id} ({baseline.agent_version}) "
        f"| status={baseline.status} | steps={baseline.total_steps} "
        f"| tokens={baseline.total_tokens} | latency={baseline.total_latency_ms:.1f} ms",
        f"[bold]Candidate:[/bold] {candidate.trace_id} ({candidate.agent_version}) "
        f"| status={candidate.status} | steps={candidate.total_steps} "
        f"| tokens={candidate.total_tokens} | latency={candidate.total_latency_ms:.1f} ms",
        f"[bold]Deltas:[/bold]    steps={diff.step_delta:+d}    "
        f"tokens={diff.token_delta:+d} ({token_pct:+.1f}%)    "
        f"latency={diff.latency_delta_ms:+.1f} ms    "
        f"[bold]First Divergence:[/bold] {divergence_text}",
    ]

    if baseline_report is not None and candidate_report is not None:
        base_badge = _format_status_badge(baseline_report.overall_passed)
        cand_badge = _format_status_badge(candidate_report.overall_passed)
        lines.append(
            f"[bold]Suite Score:[/bold] Baseline {base_badge} "
            f"({baseline_report.overall_score * 100:.1f}%)  vs  "
            f"Candidate {cand_badge} "
            f"({candidate_report.overall_score * 100:.1f}%)"
        )

    border_style = "yellow" if diff.divergence_step_id is not None else "green"
    if candidate_report is not None and not candidate_report.overall_passed:
        border_style = "red"

    target_console.print(
        Panel(
            "\n".join(lines),
            title="Agent-Probe Trajectory Diff Summary",
            border_style=border_style,
        )
    )

    failed_steps = _collect_failed_steps(candidate_report)
    diff_table = Table(title="Step-by-Step Trajectory Comparison", expand=True)
    diff_table.add_column("Step", justify="center", no_wrap=True)
    diff_table.add_column("Baseline (State / Tool)", style="cyan")
    diff_table.add_column("Candidate (State / Tool)", style="magenta")
    diff_table.add_column("Token Delta", justify="right", no_wrap=True)
    diff_table.add_column("Diff Status", justify="center", no_wrap=True)

    for row in diff.step_comparisons:
        b = row["baseline"]
        c = row["candidate"]
        b_label = (
            f"{b['state']} / {b['tool_name'] or '-'}"
            if b is not None
            else "[dim]<none>[/dim]"
        )
        c_label = (
            f"{c['state']} / {c['tool_name'] or '-'}"
            if c is not None
            else "[dim]<none>[/dim]"
        )

        status = row["status"]
        cand_step_id = c["step_id"] if c is not None else None
        is_failed = cand_step_id is not None and cand_step_id in failed_steps

        if row["is_first_divergence"]:
            status_badge = "[bold red]FIRST DIVERGENCE[/bold red]"
        elif status in {"DIVERGED", "CANDIDATE_ONLY", "BASELINE_ONLY"}:
            status_badge = f"[red]{status}[/red]"
        elif status == "ARGS_DIFF":
            status_badge = "[yellow]ARGS_DIFF[/yellow]"
        else:
            status_badge = "[green]MATCH[/green]"

        if is_failed:
            status_badge = f"{status_badge} [bold red][GUARD FAIL][/bold red]"

        diff_table.add_row(
            str(row["step_id"]),
            b_label,
            c_label,
            f"{row['token_delta']:+d}",
            status_badge,
        )

    target_console.print(diff_table)

    if candidate_report is not None:
        _render_assertions_table(candidate_report, target_console)
