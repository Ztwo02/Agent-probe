# Agent-Probe

[![CI Status](https://github.com/Ztwo02/Agent-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/Ztwo02/Agent-probe/actions)
[![Live Demo Report](https://img.shields.io/badge/Live_Demo-HTML_Trace_Diff-blue)](https://ztwo02.github.io/Agent-probe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Lightweight, local-first white-box trajectory evaluation & replay harness for AI Agents (`pytest` + `Playwright Trace Viewer` for Agent execution paths).

## Features (v0.1.0 MVP)

- **Strict Trajectory Schema (`Pydantic v2`)**: Validates step-by-step states, thoughts, tool calls, observations, latency, and token budgets.
- **Deterministic Evaluators**:
  - `MaxStepGuard`: Enforces maximum execution step limits.
  - `TokenBudgetGuard`: Guards total token consumption and locates budget-breaching steps.
  - `LoopDetectionGuard`: Detects consecutive identical tool calls (same tool name and arguments) and identifies loop step IDs.
  - `RequiredToolCallMatch`: Verifies required tool invocation with recursive argument subset matching.
  - `TrajectoryPathMatch`: Validates tool or state execution order (`subsequence` or `exact`).
- **Trajectory Diff Engine**: Compares baseline vs candidate runs, pinpoints `divergence_step_id`, and computes step/token/latency deltas.
- **Zero-Config Reporting**: Prints Rich console tables and exports self-contained single-file HTML diff reports with zero external assets.

## Quick Start

```bash
pip install -e ".[dev]"
pytest -v
agent-probe demo
```

Running `agent-probe demo` compares the bundled `examples/baseline_trace.json` and `examples/candidate_trace_loop.json` against `examples/eval_suite.yaml` and generates `probe_demo_report.html`.

## CLI Commands

```bash
# Evaluate a single trajectory against a declarative YAML suite
agent-probe run examples/baseline_trace.json --suite examples/eval_suite.yaml --html report.html

# Compare baseline vs candidate trajectories with optional suite assertions
agent-probe diff examples/baseline_trace.json examples/candidate_trace_loop.json \
  --suite examples/eval_suite.yaml --html diff_report.html

# Run the built-in zero-config demo
agent-probe demo
```

## Architecture

```text
src/agent_probe/
├── cli.py                   # Typer CLI commands (run, diff, demo)
├── core/
│   ├── schema.py            # Pydantic v2 trajectory, report, and diff contracts
│   ├── engine.py            # Declarative YAML suite evaluation engine
│   └── differ.py            # Step-aligned trajectory diff & divergence detector
├── evaluators/
│   ├── base.py              # BaseEvaluator interface
│   └── deterministic.py     # Five deterministic guard & path evaluators
├── adapters/
│   └── json_loader.py       # Anti-corruption JSON/YAML trace loader
└── reporter/
    ├── console.py           # Rich terminal tables & panels
    ├── html_generator.py    # Self-contained HTML generator
    └── templates/
        └── report.html.j2   # Inline CSS/JS trace diff UI template
```
