# Agent-Probe

[![CI Status](https://github.com/Ztwo02/Agent-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/Ztwo02/Agent-probe/actions)
[![Live Demo Report](https://img.shields.io/badge/Live_Demo-HTML_Trace_Diff-blue)](https://ztwo02.github.io/Agent-probe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | 简体中文

轻量级、本地优先（Local-First）的 AI Agent 白盒轨迹评测与沙盒回放底座（Agent 执行路径领域的 `pytest` + `Playwright Trace Viewer`）。

## 核心特性

### v0.2.0 — 动态 Agent 沙盒与断点回放（Sandbox & Fork Replay）

- **无侵入式轨迹录制器（`TraceSession` + `@trace_tool`）**：基于 `contextvars` 实现的同步/异步工具调用采集器。自动捕获入参、耗时、返回值与异常（`error_type`），并实时组装为标准 `Trajectory` 对象。未处于活跃会话时，被装饰工具原样透传、零副作用。
- **确定性工具打桩沙盒（`ToolSandbox`）**：支持声明式 YAML / `MockRule` 规则配置。参数子集匹配最具体的规则优先命中，支持按工具配置默认兜底规则；支持故障注入（`TimeoutError`、`ConnectionError` 及自定义异常名）以及严格模式（`strict=True` 阻断未打桩的真实外部调用）。命中打桩的步骤会自动标记 `is_mocked`。
- **时间旅行断点分叉重跑（`ReplayEngine`）**：冻结基准轨迹前 `K-1` 步的历史状态（耗时 `0 ms`、不执行真实工具、原样重抛历史异常），仅从第 `K` 步开始运行修改后的新 Agent 逻辑。自动输出携带血缘信息（`parent_trace_id`、`fork_step_id`）的 `ReplayResult` 与 Phase 1 `TrajectoryDiff`。
- **可视化徽章**：HTML 报告支持渲染 `[MOCKED 🛡️]`、`[REPLAYED ⏪]` 与 `[FORK / DIVERGENCE POINT ⚡]` 视觉标记。

### v0.1.0 — 离线白盒轨迹评测与 Diff 对比

- **严格轨迹数据契约（`Pydantic v2`）**：校验每一步的状态、思考过程（Thought）、工具调用、观测结果（Observation）、耗时与 Token 消耗预算。
- **五大确定性评测算子**：
  - `MaxStepGuard`：强制限制最大执行步数，并定位超限步骤 ID。
  - `TokenBudgetGuard`：守卫总 Token 消耗预算，并定位突破阈值的步骤。
  - `LoopDetectionGuard`：检测连续多次调用同名同参数的工具（死循环检测），精确标记循环步骤 ID。
  - `RequiredToolCallMatch`：校验必调工具是否被调用，并支持递归参数子集匹配。
  - `TrajectoryPathMatch`：校验工具或状态执行序列是否符合预期顺序（`subsequence` 子序列或 `exact` 精确匹配）。
- **轨迹 Diff 对比引擎**：逐步对齐 Baseline 与 Candidate 轨迹，精准定位首个分叉点 `divergence_step_id`，并计算步数、Token 与耗时差值。
- **零配置双端报告**：输出 Rich 终端高亮表格，并导出零外部静态资源依赖的单文件自包含 HTML 对比报告。

## 快速开始

```bash
pip install -e ".[dev]"
pytest -v
agent-probe demo           # v0.1.0：离线轨迹 Diff 与白盒断言演示
agent-probe harness-demo   # v0.2.0：沙盒故障注入与断点分叉重跑演示
```

运行 `agent-probe demo` 将使用内置的 `examples/baseline_trace.json`、`examples/candidate_trace_loop.json` 与 `examples/eval_suite.yaml` 进行对比评测，并生成 `probe_demo_report.html`。

运行 `agent-probe harness-demo` 将直接执行 [`examples/demo_agent_harness.py`](examples/demo_agent_harness.py)（无需任何大模型 API Key），整个流程分为三个阶段：

1. 在 `ToolSandbox` 故障注入环境下运行存在缺陷的 v1 售后 Agent 并录制轨迹：当 `query_logistics` 触发超时异常后，v1 连续重复调用 3 次最终因 `LOOP_ABORTED` 中止。
2. 通过 `ReplayEngine` 从第 2 步对修复后的 v2 Agent 进行断点分叉重跑：第 1 步直接复用冻结的历史结果，第 2 步命中同样的注入超时后，v2 立即降级调用 `issue_apology_coupon` 并成功完成任务。
3. 自动对比并评分两次运行轨迹，生成可视化报告 `probe_harness_report.html`。

## Phase 2 动态沙盒快速上手

### 1. 使用 `@trace_tool` + `TraceSession` 录制轨迹

```python
from agent_probe import TraceSession, trace_tool

@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict:
    return carrier_api.get(order_id, realtime=realtime_refresh)

@trace_tool
async def search_kb(query: str) -> list[str]:  # 同样支持异步工具函数
    ...

with TraceSession(agent_version="aftersales-v1", task_input="我的订单到哪了？") as session:
    session.record_thought("先查询物流状态。", prompt_tokens=220, completion_tokens=45)
    query_logistics("ORD-20261116-8891")          # 自动记录为第 1 步（并吸收上一条思考）
    session.set_final_output("您的包裹预计明天送达。")

session.save_json("trace.json")                   # 导出的 JSON 可直接被 `agent-probe run/diff` 加载
```

`record_thought()` 会暂存思考内容并在下一次调用 `@trace_tool` 时合并为 ReAct 风格的单步记录；若后续没有工具调用，则会单独落盘为一个独立步骤。当工具函数抛出异常时，录制器会自动记录 `is_error=True` 及 `error_type`，随后继续向上抛出异常。

### 2. 使用 `ToolSandbox` 进行工具打桩与故障注入

```yaml
# examples/mock_rules.yaml 示例片段
rules:
  - tool_name: query_logistics
    match_args: {order_id: ORD-20261116-8891}
    return_value: {status: CARRIER_SYNC_PENDING}
  - tool_name: query_logistics            # 匹配键更多，因此在 realtime_refresh=true 时优先命中
    match_args: {order_id: ORD-20261116-8891, realtime_refresh: true}
    raise_error: TimeoutError
    error_message: "504 Gateway Timeout from carrier SF-Express"
  - tool_name: query_logistics            # 默认 "*" 兜底规则
    return_value: {error: ORDER_NOT_FOUND}
```

```python
from agent_probe import ToolSandbox, TraceSession

sandbox = ToolSandbox.from_yaml("examples/mock_rules.yaml", strict=True)
with TraceSession() as session, sandbox:
    my_agent()          # 不会执行真实工具体；步骤自动标记 is_mocked=True
```

### 3. 使用 `ReplayEngine` 进行断点分叉重跑

```python
from agent_probe import ReplayEngine
from agent_probe.adapters.json_loader import load_trajectory

baseline = load_trajectory("trace.json")
engine = ReplayEngine(baseline, fork_at_step=2, sandbox=sandbox)
result = engine.run(fixed_agent)            # 异步 Agent 可使用：await engine.arun(async_fixed_agent)

result.replayed_step_ids                    # [1]  -> 从冻结历史直接返回，耗时 0 ms
result.candidate_trajectory.fork_step_id    # 2
result.diff.divergence_step_id              # 对比基准轨迹的首个状态/工具分叉步骤 ID
```

## CLI 命令行指令

```bash
# 使用声明式 YAML 规则套件评测单条轨迹
agent-probe run examples/baseline_trace.json --suite examples/eval_suite.yaml --html report.html

# 对比基准轨迹与待测轨迹，并可选附带断言套件评分
agent-probe diff examples/baseline_trace.json examples/candidate_trace_loop.json \
  --suite examples/eval_suite.yaml --html diff_report.html

# 运行内置零配置离线对比演示
agent-probe demo

# 运行沙盒故障注入与断点分叉重跑演示（v0.2.0）
agent-probe harness-demo --html probe_harness_report.html

# 从指定断点步骤对自己的 Agent 函数进行分叉重跑（v0.2.0）
agent-probe replay trace.json --agent my_pkg.agents:fixed_agent --fork-at-step 2 \
  --sandbox examples/mock_rules.yaml --strict-sandbox --html replay_report.html
```

## 项目架构

```text
src/agent_probe/
├── cli.py                   # Typer CLI 命令行入口 (run, diff, demo, harness-demo, replay)
├── core/
│   ├── schema.py            # Pydantic v2 轨迹、评测报告与 Diff 数据契约
│   ├── engine.py            # 声明式 YAML 评测套件调度引擎
│   └── differ.py            # 逐步对齐轨迹 Diff、分叉与断点检测算法
├── evaluators/
│   ├── base.py              # BaseEvaluator 评测算子抽象基类
│   └── deterministic.py     # 五大确定性守卫与路径匹配算子
├── adapters/
│   └── json_loader.py       # 防腐层 JSON/YAML 轨迹加载与校验器
├── sandbox/
│   ├── tracer.py            # TraceSession + @trace_tool（contextvars 并发隔离，兼容同步/异步）
│   ├── mocker.py            # MockRule + ToolSandbox（确定性打桩与故障注入）
│   └── replay.py            # ReplayEngine + ReplayResult（时间旅行断点分叉重跑）
└── reporter/
    ├── console.py           # Rich 终端高亮表格与面板输出
    ├── html_generator.py    # 自包含单文件 HTML 报告生成器
    └── templates/
        └── report.html.j2   # 内联 CSS/JS 轨迹对比模板（含 v0.2.0 视觉徽章）
```

## 在线演示

点击查看由 GitHub Actions 自动部署的最新版单文件轨迹对比报告：
[https://ztwo02.github.io/Agent-probe/](https://ztwo02.github.io/Agent-probe/)

## 开源协议

本项目基于 MIT 协议开源，详见 [LICENSE](LICENSE)。
