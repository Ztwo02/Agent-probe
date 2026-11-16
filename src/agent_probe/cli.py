"""Typer CLI entrypoint for Agent-Probe (run, diff, demo)."""

from __future__ import annotations

from pathlib import Path

import typer
from pydantic import ValidationError
from rich.console import Console

from agent_probe.adapters.json_loader import (
    TraceLoadError,
    format_validation_error,
    load_trajectory,
)
from agent_probe.core.differ import diff_trajectories
from agent_probe.core.engine import EvaluationEngine, SuiteConfigError
from agent_probe.core.schema import Trajectory
from agent_probe.reporter.console import render_diff_report, render_eval_report
from agent_probe.reporter.html_generator import generate_html_report

app = typer.Typer(
    name="agent-probe",
    help="Lightweight white-box trajectory evaluation & diff harness for AI Agents.",
    no_args_is_help=True,
)


def _resolve_examples_dir() -> Path:
    """Locate the examples/ directory from CWD or repository root."""
    required = ("baseline_trace.json", "candidate_trace_loop.json", "eval_suite.yaml")
    cwd_examples = Path.cwd() / "examples"
    if all((cwd_examples / name).is_file() for name in required):
        return cwd_examples

    repo_examples = Path(__file__).resolve().parents[2] / "examples"
    if all((repo_examples / name).is_file() for name in required):
        return repo_examples

    raise FileNotFoundError(
        "Could not locate built-in examples/ directory containing demo trace fixtures."
    )


def _load_trace_or_exit(path: Path, console: Console) -> Trajectory:
    try:
        return load_trajectory(path)
    except ValidationError as exc:
        console.print(f"[red]{format_validation_error(exc, source=path)}[/red]")
        raise typer.Exit(code=1) from exc
    except (FileNotFoundError, TraceLoadError) as exc:
        console.print(f"[red]Error loading trace: {exc}[/red]")
        raise typer.Exit(code=1) from exc


def _load_engine_or_exit(suite_path: Path, console: Console) -> EvaluationEngine:
    try:
        return EvaluationEngine.from_yaml(suite_path)
    except (FileNotFoundError, SuiteConfigError, ValidationError) as exc:
        console.print(f"[red]Error loading eval suite: {exc}[/red]")
        raise typer.Exit(code=1) from exc


@app.command()
def run(
    trace_file: Path = typer.Argument(..., help="Path to trajectory JSON/YAML file."),
    suite: Path = typer.Option(..., "--suite", "-s", help="Path to eval_suite.yaml file."),
    html: Path | None = typer.Option(
        None, "--html", "-o", help="Optional path to export a self-contained HTML report."
    ),
) -> None:
    """Evaluate a single trajectory against a declarative YAML assertion suite."""
    console = Console()
    trajectory = _load_trace_or_exit(trace_file, console)
    engine = _load_engine_or_exit(suite, console)
    report = engine.evaluate(trajectory)

    render_eval_report(report, trajectory=trajectory, console=console)

    if html is not None:
        out_path = generate_html_report(
            html,
            single_trace=trajectory,
            single_report=report,
        )
        console.print(f"[cyan]HTML report generated:[/cyan] {out_path}")

    if not report.overall_passed:
        raise typer.Exit(code=1)


@app.command()
def diff(
    baseline_file: Path = typer.Argument(..., help="Path to baseline trajectory file."),
    candidate_file: Path = typer.Argument(..., help="Path to candidate trajectory file."),
    suite: Path | None = typer.Option(
        None, "--suite", "-s", help="Optional eval_suite.yaml to score both trajectories."
    ),
    html: Path | None = typer.Option(
        None, "--html", "-o", help="Optional path to export a single-file HTML diff report."
    ),
) -> None:
    """Compare baseline and candidate trajectories and detect the first divergence."""
    console = Console()
    baseline = _load_trace_or_exit(baseline_file, console)
    candidate = _load_trace_or_exit(candidate_file, console)
    diff_result = diff_trajectories(baseline, candidate)

    baseline_report = None
    candidate_report = None
    if suite is not None:
        engine = _load_engine_or_exit(suite, console)
        baseline_report = engine.evaluate(baseline)
        candidate_report = engine.evaluate(candidate)

    render_diff_report(
        diff_result,
        baseline=baseline,
        candidate=candidate,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        console=console,
    )

    if html is not None:
        out_path = generate_html_report(
            html,
            baseline=baseline,
            candidate=candidate,
            diff=diff_result,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
        )
        console.print(f"[cyan]HTML diff report generated:[/cyan] {out_path}")


@app.command()
def demo(
    html: Path = typer.Option(
        Path("probe_demo_report.html"),
        "--html",
        "-o",
        help="Output path for the generated self-contained HTML demo report.",
    ),
) -> None:
    """Run zero-config demo comparing built-in baseline and loop-failure trajectories."""
    console = Console()
    try:
        examples_dir = _resolve_examples_dir()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    baseline = _load_trace_or_exit(examples_dir / "baseline_trace.json", console)
    candidate = _load_trace_or_exit(examples_dir / "candidate_trace_loop.json", console)
    engine = _load_engine_or_exit(examples_dir / "eval_suite.yaml", console)

    diff_result = diff_trajectories(baseline, candidate)
    baseline_report = engine.evaluate(baseline)
    candidate_report = engine.evaluate(candidate)

    render_diff_report(
        diff_result,
        baseline=baseline,
        candidate=candidate,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        console=console,
    )

    out_path = generate_html_report(
        html,
        baseline=baseline,
        candidate=candidate,
        diff=diff_result,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        title="Agent-Probe Demo: After-Sales Logistics Agent Regression Report",
    )
    console.print(f"[bold green]Demo HTML report saved to:[/bold green] {out_path}")
