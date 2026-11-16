"""Reporting layer: Rich console tables and self-contained HTML reports."""

from agent_probe.reporter.console import render_diff_report, render_eval_report
from agent_probe.reporter.html_generator import generate_html_report, render_html_report

__all__ = [
    "generate_html_report",
    "render_diff_report",
    "render_eval_report",
    "render_html_report",
]
