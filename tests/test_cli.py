"""End-to-end tests for CLI commands and single-file HTML report generation."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from agent_probe.cli import app

runner = CliRunner()
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


def test_cli_run_pass_and_fail_with_html(tmp_path: Path) -> None:
    pass_html = tmp_path / "pass_report.html"
    res_pass = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES_DIR / "baseline_trace.json"),
            "--suite",
            str(EXAMPLES_DIR / "eval_suite.yaml"),
            "--html",
            str(pass_html),
        ],
    )
    assert res_pass.exit_code == 0
    assert "PASS" in res_pass.output
    assert pass_html.is_file()
    pass_content = pass_html.read_text(encoding="utf-8")
    assert "trace-baseline-20261116-001" in pass_content
    assert "EVAL PASSED" in pass_content
    assert "cdn.tailwindcss.com" not in pass_content

    fail_html = tmp_path / "fail_report.html"
    res_fail = runner.invoke(
        app,
        [
            "run",
            str(EXAMPLES_DIR / "candidate_trace_loop.json"),
            "--suite",
            str(EXAMPLES_DIR / "eval_suite.yaml"),
            "--html",
            str(fail_html),
        ],
    )
    assert res_fail.exit_code == 1
    assert "FAIL" in res_fail.output
    assert fail_html.is_file()
    fail_content = fail_html.read_text(encoding="utf-8")
    assert "LOOP DETECTED" in fail_content
    assert "EVAL FAILED" in fail_content


def test_cli_diff_command(tmp_path: Path) -> None:
    diff_html = tmp_path / "diff.html"
    result = runner.invoke(
        app,
        [
            "diff",
            str(EXAMPLES_DIR / "baseline_trace.json"),
            str(EXAMPLES_DIR / "candidate_trace_loop.json"),
            "--suite",
            str(EXAMPLES_DIR / "eval_suite.yaml"),
            "--html",
            str(diff_html),
        ],
    )
    assert result.exit_code == 0
    assert "Step 2" in result.output
    assert diff_html.is_file()
    html = diff_html.read_text(encoding="utf-8")
    assert "FIRST DIVERGENCE" in html
    assert "trace-baseline-20261116-001" in html
    assert "trace-candidate-20261116-002" in html
    assert "<script src=" not in html


def test_cli_demo_generates_default_html(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0
    report_path = tmp_path / "probe_demo_report.html"
    assert report_path.is_file()
    html = report_path.read_text(encoding="utf-8")
    assert len(html) > 2000
    assert "trace-baseline-20261116-001" in html
    assert "trace-candidate-20261116-002" in html
    assert "query_logistics" in html
    assert "issue_coupon" in html
    assert "FIRST DIVERGENCE" in html


def test_cli_reports_invalid_trace_cleanly(tmp_path: Path) -> None:
    bad_trace = tmp_path / "bad.json"
    bad_trace.write_text(json.dumps({"trace_id": "x"}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "run",
            str(bad_trace),
            "--suite",
            str(EXAMPLES_DIR / "eval_suite.yaml"),
        ],
    )
    assert result.exit_code == 1
    assert "Invalid trajectory schema" in result.output
