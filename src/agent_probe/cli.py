"""Typer CLI entrypoint for Agent-Probe (run, diff, demo, harness-demo, replay)."""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import inspect
import json
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

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
from agent_probe.sandbox.mocker import SandboxConfigError, ToolSandbox
from agent_probe.sandbox.replay import ReplayEngine, ReplayResult

app = typer.Typer(
    name="agent-probe",
    help="Lightweight white-box trajectory evaluation, sandbox & replay harness for AI Agents.",
    no_args_is_help=True,
)


_DEMO_FIXTURES = ("baseline_trace.json", "candidate_trace_loop.json", "eval_suite.yaml")
_HARNESS_FIXTURES = ("demo_agent_harness.py", "mock_rules.yaml")


def _resolve_examples_dir(required: tuple[str, ...] = _DEMO_FIXTURES) -> Path:
    """Locate the examples/ directory from CWD or repository root."""
    cwd_examples = Path.cwd() / "examples"
    if all((cwd_examples / name).is_file() for name in required):
        return cwd_examples

    repo_examples = Path(__file__).resolve().parents[2] / "examples"
    if all((repo_examples / name).is_file() for name in required):
        return repo_examples

    raise FileNotFoundError(
        "Could not locate built-in examples/ directory containing demo fixtures: "
        + ", ".join(required)
    )


def _load_python_file(path: Path, module_name: str) -> ModuleType:
    """Import a standalone Python file (e.g. an examples/ script) as a module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_agent_callable(spec: str) -> Callable[..., Any]:
    """Resolve ``path/to/file.py:func`` or ``package.module:func`` to a callable."""
    target, sep, attr = spec.rpartition(":")
    if not sep or not target or not attr:
        raise ValueError(
            f"Agent spec must look like 'module:function' or 'file.py:function': {spec}"
        )
    if target.endswith(".py") or Path(target).is_file():
        path = Path(target).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Agent file not found: {path}")
        module = _load_python_file(path, f"agent_probe_user_agent_{path.stem}")
    else:
        module = importlib.import_module(target)
    agent = getattr(module, attr, None)
    if not callable(agent):
        raise ValueError(f"{attr!r} in {target!r} is not a callable agent function")
    return agent


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


def _render_replay_outcome(
    result: ReplayResult,
    console: Console,
    *,
    engine: EvaluationEngine | None,
    html: Path | None,
    title: str,
) -> None:
    baseline = result.baseline_trajectory
    candidate = result.candidate_trajectory
    baseline_report = engine.evaluate(baseline) if engine is not None else None
    candidate_report = engine.evaluate(candidate) if engine is not None else None

    render_diff_report(
        result.diff,
        baseline=baseline,
        candidate=candidate,
        baseline_report=baseline_report,
        candidate_report=candidate_report,
        console=console,
    )
    if result.early_divergence_step_id is not None:
        console.print(
            f"[yellow]Warning: agent deviated from baseline history at step "
            f"{result.early_divergence_step_id} (before the fork point); "
            "that call ran live.[/yellow]"
        )
    if html is not None:
        out_path = generate_html_report(
            html,
            baseline=baseline,
            candidate=candidate,
            diff=result.diff,
            baseline_report=baseline_report,
            candidate_report=candidate_report,
            title=title,
        )
        console.print(f"[bold green]Replay HTML report saved to:[/bold green] {out_path}")


@app.command("harness-demo")
def harness_demo(
    html: Path = typer.Option(
        Path("probe_harness_report.html"),
        "--html",
        "-o",
        help="Output path for the generated sandbox & fork-replay HTML report.",
    ),
    fork_at_step: int = typer.Option(
        2, "--fork-at-step", "-k", min=1, help="Step at which the fixed agent forks."
    ),
) -> None:
    """Run the zero-API-key sandbox fault-injection & breakpoint fork-replay demo."""
    console = Console()
    try:
        examples_dir = _resolve_examples_dir(_HARNESS_FIXTURES)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    demo_module = _load_python_file(
        examples_dir / "demo_agent_harness.py", "agent_probe_examples_demo_agent_harness"
    )
    outcome = demo_module.run_harness_demo(
        fork_at_step=fork_at_step, rules_path=examples_dir / "mock_rules.yaml"
    )

    console.print(
        "[bold]Harness:[/bold] v1 agent recorded under ToolSandbox fault injection "
        f"(TimeoutError on realtime carrier refresh); v2 agent fork-replayed from step "
        f"{fork_at_step} on top of v1's frozen history."
    )
    render_diff_report(
        outcome.replay.diff,
        baseline=outcome.baseline,
        candidate=outcome.candidate,
        baseline_report=outcome.baseline_report,
        candidate_report=outcome.candidate_report,
        console=console,
    )
    out_path = generate_html_report(
        html,
        baseline=outcome.baseline,
        candidate=outcome.candidate,
        diff=outcome.replay.diff,
        baseline_report=outcome.baseline_report,
        candidate_report=outcome.candidate_report,
        title="Agent-Probe Harness Demo: Fault Injection & Fork Replay",
    )
    console.print(f"[bold green]Harness HTML report saved to:[/bold green] {out_path}")


@app.command()
def replay(
    baseline_file: Path = typer.Argument(..., help="Path to the baseline trajectory file."),
    agent: str = typer.Option(
        ...,
        "--agent",
        "-a",
        help="Agent callable to re-run: 'package.module:function' or 'path/file.py:function'.",
    ),
    fork_at_step: int = typer.Option(
        ..., "--fork-at-step", "-k", min=1, help="First step executed live (1-based)."
    ),
    sandbox_rules: Path | None = typer.Option(
        None, "--sandbox", "-m", help="Optional mock_rules.yaml for post-fork tool calls."
    ),
    strict_sandbox: bool = typer.Option(
        False, "--strict-sandbox", help="Block tool calls that have no matching mock rule."
    ),
    suite: Path | None = typer.Option(
        None, "--suite", "-s", help="Optional eval_suite.yaml to score both trajectories."
    ),
    html: Path | None = typer.Option(
        None, "--html", "-o", help="Optional path to export a single-file HTML replay report."
    ),
    save_candidate: Path | None = typer.Option(
        None, "--save-candidate", help="Optional path to save the forked trajectory as JSON."
    ),
) -> None:
    """Fork-replay an agent against a recorded baseline from a breakpoint step."""
    console = Console()
    baseline = _load_trace_or_exit(baseline_file, console)
    engine = _load_engine_or_exit(suite, console) if suite is not None else None

    try:
        agent_fn = _load_agent_callable(agent)
        sandbox = (
            ToolSandbox.from_yaml(sandbox_rules, strict=strict_sandbox or None)
            if sandbox_rules is not None
            else None
        )
    except (ImportError, ValueError, FileNotFoundError, SandboxConfigError) as exc:
        console.print(f"[red]Error preparing replay: {exc}[/red]")
        raise typer.Exit(code=1) from exc

    replay_engine = ReplayEngine(baseline, fork_at_step=fork_at_step, sandbox=sandbox)
    try:
        if inspect.iscoroutinefunction(agent_fn):
            asyncio.run(replay_engine.arun(agent_fn))
        else:
            replay_engine.run(agent_fn)
    except Exception as exc:  # the forked agent crashed; still report its trajectory
        console.print(f"[red]Agent raised {type(exc).__name__}: {exc}[/red]")

    result = replay_engine.result
    if result is None:  # pragma: no cover - engine always builds a result on exit
        raise typer.Exit(code=1)

    _render_replay_outcome(
        result,
        console,
        engine=engine,
        html=html,
        title=f"Fork Replay: {baseline.trace_id} @ step {fork_at_step}",
    )
    if save_candidate is not None:
        save_candidate.parent.mkdir(parents=True, exist_ok=True)
        save_candidate.write_text(
            json.dumps(
                result.candidate_trajectory.model_dump(),
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        console.print(f"[cyan]Candidate trajectory saved to:[/cyan] {save_candidate}")
