# Agent-Probe

[![CI Status](https://github.com/Ztwo02/Agent-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/Ztwo02/Agent-probe/actions)
[![Live Demo Report](https://img.shields.io/badge/Live_Demo-HTML_Trace_Diff-blue)](https://ztwo02.github.io/Agent-probe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Lightweight, local-first white-box trajectory evaluation & replay harness for AI Agents (`pytest` + `Playwright Trace Viewer` for Agent execution paths).

## Features

### v0.2.0 — Dynamic Agent Harness (Sandbox & Fork Replay)

- **Non-intrusive Tracer (`TraceSession` + `@trace_tool`)**: `contextvars`-based recorder for sync and async tools. It captures arguments, latency, return values and exceptions (`error_type`) and builds a standard `Trajectory` automatically. Outside a session, decorated tools pass through untouched.
- **Deterministic Tool Sandbox (`ToolSandbox`)**: declarative YAML/`MockRule` stubs. The most specific argument-subset match wins, with a per-tool default fallback. Supports fault injection (`TimeoutError`, `ConnectionError`, custom names) and a `strict` mode that blocks unmocked external calls. Mocked steps are flagged `is_mocked`.
- **Time-Travel Fork Replay (`ReplayEngine`)**: freezes the first `K-1` steps of a baseline (0 ms, zero tool execution, recorded errors re-raised), then runs the modified agent live from step `K`. It produces a `ReplayResult` with lineage (`parent_trace_id`, `fork_step_id`) and a Phase 1 `TrajectoryDiff`.
- **Visual Badges**: HTML reports render `[MOCKED 🛡️]`, `[REPLAYED ⏪]` and `[FORK / DIVERGENCE POINT ⚡]` markers.

### v0.1.0 — Offline Trajectory Evaluation

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
agent-probe demo           # v0.1.0: offline trajectory diff & evaluation
agent-probe harness-demo   # v0.2.0: sandbox fault injection & fork replay
```

Running `agent-probe demo` compares the bundled `examples/baseline_trace.json` and `examples/candidate_trace_loop.json` against `examples/eval_suite.yaml` and generates `probe_demo_report.html`.

Running `agent-probe harness-demo` executes [`examples/demo_agent_harness.py`](examples/demo_agent_harness.py) with no API key required. The demo runs in three phases:

1. A defective v1 after-sales agent is recorded under `ToolSandbox` fault injection. When `query_logistics` times out, v1 retries the same call 3 times and ends up `LOOP_ABORTED`.
2. A fixed v2 agent is fork-replayed from step 2. Step 1 is replayed from history. Step 2 hits the same injected timeout, and v2 then degrades to `issue_apology_coupon`.
3. Both runs are diffed and scored, and the results are written to `probe_harness_report.html`.

## Phase 2 Harness Quick Start

### 1. Record trajectories with `@trace_tool` + `TraceSession`

```python
from agent_probe import TraceSession, trace_tool

@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict:
    return carrier_api.get(order_id, realtime=realtime_refresh)

@trace_tool
async def search_kb(query: str) -> list[str]:  # async tools work the same way
    ...

with TraceSession(agent_version="aftersales-v1", task_input="Where is my order?") as session:
    session.record_thought("Look up logistics first.", prompt_tokens=220, completion_tokens=45)
    query_logistics("ORD-20261116-8891")          # recorded as step 1 (thought attached)
    session.set_final_output("Your parcel arrives tomorrow.")

session.save_json("trace.json")                   # loadable by `agent-probe run/diff`
```

`record_thought()` is attached to the next traced tool call (ReAct-style step). A thought with no following tool call becomes a standalone step. If a tool raises, the tracer records an `is_error=True` step with `error_type` and re-raises the exception.

### 2. Stub tools & inject faults with `ToolSandbox`

```yaml
# examples/mock_rules.yaml (excerpt)
rules:
  - tool_name: query_logistics
    match_args: {order_id: ORD-20261116-8891}
    return_value: {status: CARRIER_SYNC_PENDING}
  - tool_name: query_logistics            # more specific -> wins when realtime_refresh=true
    match_args: {order_id: ORD-20261116-8891, realtime_refresh: true}
    raise_error: TimeoutError
    error_message: "504 Gateway Timeout from carrier SF-Express"
  - tool_name: query_logistics            # default "*" fallback
    return_value: {error: ORDER_NOT_FOUND}
```

```python
from agent_probe import ToolSandbox, TraceSession

sandbox = ToolSandbox.from_yaml("examples/mock_rules.yaml", strict=True)
with TraceSession() as session, sandbox:
    my_agent()          # real tool bodies never run; steps are flagged is_mocked=True
```

### 3. Fork-replay a fixed agent with `ReplayEngine`

```python
from agent_probe import ReplayEngine
from agent_probe.adapters.json_loader import load_trajectory

baseline = load_trajectory("trace.json")
engine = ReplayEngine(baseline, fork_at_step=2, sandbox=sandbox)
result = engine.run(fixed_agent)            # or: await engine.arun(async_fixed_agent)

result.replayed_step_ids                    # [1]  -> served from frozen history, 0 ms
result.candidate_trajectory.fork_step_id    # 2
result.diff.divergence_step_id              # first state/tool divergence vs baseline
```

## CLI Commands

```bash
# Evaluate a single trajectory against a declarative YAML suite
agent-probe run examples/baseline_trace.json --suite examples/eval_suite.yaml --html report.html

# Compare baseline vs candidate trajectories with optional suite assertions
agent-probe diff examples/baseline_trace.json examples/candidate_trace_loop.json \
  --suite examples/eval_suite.yaml --html diff_report.html

# Run the built-in zero-config demo
agent-probe demo

# Run the sandbox fault-injection & fork-replay demo (v0.2.0)
agent-probe harness-demo --html probe_harness_report.html

# Fork-replay your own agent against a recorded baseline (v0.2.0)
agent-probe replay trace.json --agent my_pkg.agents:fixed_agent --fork-at-step 2 \
  --sandbox examples/mock_rules.yaml --strict-sandbox --html replay_report.html
```

## Architecture

```text
src/agent_probe/
├── cli.py                   # Typer CLI commands (run, diff, demo, harness-demo, replay)
├── core/
│   ├── schema.py            # Pydantic v2 trajectory, report, and diff contracts
│   ├── engine.py            # Declarative YAML suite evaluation engine
│   └── differ.py            # Step-aligned trajectory diff, divergence & fork detector
├── evaluators/
│   ├── base.py              # BaseEvaluator interface
│   └── deterministic.py     # Five deterministic guard & path evaluators
├── adapters/
│   └── json_loader.py       # Anti-corruption JSON/YAML trace loader
├── sandbox/
│   ├── tracer.py            # TraceSession + @trace_tool (contextvars, sync/async)
│   ├── mocker.py            # MockRule + ToolSandbox (stubs & fault injection)
│   └── replay.py            # ReplayEngine + ReplayResult (time-travel fork replay)
└── reporter/
    ├── console.py           # Rich terminal tables & panels
    ├── html_generator.py    # Self-contained HTML generator
    └── templates/
        └── report.html.j2   # Inline CSS/JS trace diff UI template (with v0.2.0 badges)
```

## Live Demo

View the latest auto-deployed single-file trajectory diff report on GitHub Pages:
[https://ztwo02.github.io/Agent-probe/](https://ztwo02.github.io/Agent-probe/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
