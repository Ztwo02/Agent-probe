"""End-to-end tests for the v0.2.0 harness-demo / replay CLI commands and HTML badges."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_probe.cli import _load_python_file, app

runner = CliRunner()
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
HARNESS_SCRIPT = EXAMPLES_DIR / "demo_agent_harness.py"


def _load_demo_module():
    return _load_python_file(HARNESS_SCRIPT, "agent_probe_test_demo_agent_harness")


def test_demo_harness_module_outcome() -> None:
    outcome = _load_demo_module().run_harness_demo()
    baseline, candidate, diff = outcome.baseline, outcome.candidate, outcome.replay.diff

    assert baseline.status == "LOOP_ABORTED"
    assert baseline.tool_sequence == ["query_logistics"] * 5
    assert all(s.observation.is_mocked for s in baseline.steps if s.action is not None)

    assert candidate.status == "SUCCESS"
    assert candidate.tool_sequence == [
        "query_logistics",
        "query_logistics",
        "issue_apology_coupon",
    ]
    assert outcome.replay.replayed_step_ids == [1]
    assert candidate.steps[1].observation.is_mocked
    assert candidate.steps[1].observation.error_type == "TimeoutError"
    assert candidate.parent_trace_id == baseline.trace_id
    assert diff.fork_step_id == 2
    assert diff.divergence_step_id == 3

    assert outcome.baseline_report.overall_passed is False
    assert outcome.candidate_report.overall_passed is True


def test_cli_harness_demo_generates_badged_html(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["harness-demo"])
    assert result.exit_code == 0, result.output
    assert "Fork Replay" in result.output
    assert "[REPLAYED]" in result.output
    assert "[MOCKED]" in result.output
    assert "[FORK POINT]" in result.output

    report = tmp_path / "probe_harness_report.html"
    assert report.is_file()
    html = report.read_text(encoding="utf-8")
    assert "[MOCKED 🛡️]" in html
    assert "[REPLAYED ⏪]" in html
    assert "[FORK / DIVERGENCE POINT ⚡]" in html
    assert "Fork Replay Point" in html
    assert "TimeoutError" in html
    assert "Baseline Final Output" in html
    assert "Candidate Final Output" in html
    assert "btn-filter-mocked" in html
    assert "btn-filter-replayed" in html
    assert "btn-filter-fork" in html
    assert "harness-v1-baseline" in html
    assert "harness-v2-fork-replay" in html
    assert "issue_apology_coupon" in html
    assert "<script src=" not in html
    assert "#0b1020" not in html  # dark neon background removed in favor of clean light theme


def test_cli_harness_demo_custom_output(tmp_path: Path) -> None:
    out = tmp_path / "custom" / "harness.html"
    result = runner.invoke(app, ["harness-demo", "--html", str(out), "--fork-at-step", "3"])
    assert result.exit_code == 0, result.output
    assert out.is_file()


def test_cli_replay_command_with_file_agent(tmp_path: Path) -> None:
    module = _load_demo_module()
    baseline = module.record_baseline(module.load_sandbox())
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(baseline.model_dump(), ensure_ascii=False, default=str), encoding="utf-8"
    )
    html = tmp_path / "replay.html"
    saved = tmp_path / "candidate.json"

    result = runner.invoke(
        app,
        [
            "replay",
            str(baseline_path),
            "--agent",
            f"{HARNESS_SCRIPT}:aftersales_agent_v2",
            "--fork-at-step",
            "2",
            "--sandbox",
            str(EXAMPLES_DIR / "mock_rules.yaml"),
            "--strict-sandbox",
            "--html",
            str(html),
            "--save-candidate",
            str(saved),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "FIRST DIVERGENCE" in result.output
    assert html.is_file()
    assert "[REPLAYED ⏪]" in html.read_text(encoding="utf-8")

    candidate = json.loads(saved.read_text(encoding="utf-8"))
    assert candidate["parent_trace_id"] == "harness-v1-baseline"
    assert candidate["fork_step_id"] == 2
    assert candidate["status"] == "SUCCESS"

    # Single-trace HTML report via `agent-probe run` also surfaces Phase 2 markers.
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        "suite_name: s\nrules:\n  - type: max_steps\n    params:\n      max_steps: 5\n",
        encoding="utf-8",
    )
    single_html = tmp_path / "single.html"
    run_res = runner.invoke(
        app,
        ["run", str(saved), "--suite", str(suite_path), "--html", str(single_html)],
    )
    assert run_res.exit_code == 0, run_res.output
    single_content = single_html.read_text(encoding="utf-8")
    assert "Fork Replay Point" in single_content
    assert "1 replayed" in single_content
    assert "2 mocked steps" in single_content
    assert "TimeoutError" in single_content
    assert "Final Output" in single_content


def test_cli_replay_reports_bad_agent_spec() -> None:
    result = runner.invoke(
        app,
        [
            "replay",
            str(EXAMPLES_DIR / "baseline_trace.json"),
            "--agent",
            "no_colon_here",
            "--fork-at-step",
            "2",
        ],
    )
    assert result.exit_code == 1
    assert "Error preparing replay" in result.output


def test_cli_replay_survives_crashing_agent(tmp_path: Path) -> None:
    agent_file = tmp_path / "crashing_agent.py"
    agent_file.write_text(
        "def agent():\n    raise ValueError('boom')\n",
        encoding="utf-8",
    )
    result = runner.invoke(
        app,
        [
            "replay",
            str(EXAMPLES_DIR / "baseline_trace.json"),
            "--agent",
            f"{agent_file}:agent",
            "--fork-at-step",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Agent raised ValueError: boom" in result.output
