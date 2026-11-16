# Agent-Probe Phase 2 (v0.2.0): 动态沙盒打桩与断点回放底座实施计划

> **给 AI 编程助手 (Antigravity) 的全局执行指令：**
>
> 1. **前置检查：** 在开始编写任何代码前，请先扫描当前仓库的目录结构、`pyproject.toml`、`src/agent_probe/` 下的现有模块以及 `@GitHubSpecifications.md`，确保 Phase 2 的模块路径与 Phase 1 (`v0.1.0`) 保持 $100\%$ 兼容。
> 2. **微步提交铁律：** 严格按照本文档的 **Task 2.1 $\rightarrow$ Task 2.2 $\rightarrow$ Task 2.3 $\rightarrow$ Task 2.4** 顺序推进。每完成一个 Task，必须运行 `pytest -v` 和 `ruff check` 且全部通过后，按照 `@GitHubSpecifications.md` 的 Conventional Commits 规范执行一次 `git commit`。
> 3. **零外部服务依赖：** 所有新增的录制器、Mock 沙盒与断点重跑引擎必须基于 Python 标准库（如 `contextvars`, `inspect`, `functools`, `asyncio`）与现有依赖实现，严禁引入外部数据库或重度第三方框架。

---

## 1. Phase 2 (v0.2.0) 核心目标与架构演进

Phase 1 (`v0.1.0`) 解决了“离线静态轨迹 JSON 的评测与 Diff 可视化”；Phase 2 (`v0.2.0`) 则将 `Agent-probe` 升级为真正的 **动态白盒试验台 (Agent Harness)**，解决以下三大核心工程痛点：

1. **无侵入轨迹采集 (`Tracer`)：** 开发者只需添加一行装饰器 `@trace_tool` 或上下文管理器 `with TraceSession():`，即可自动拦截同步/异步函数调用、耗时、异常与思考过程，实时组装成标准 `Trajectory` 对象。
2. **确定性工具打桩与故障注入 (`ToolSandbox`)：** 在评测时自动拦截真实外部 API 调用（如查库、搜索、发邮件），根据入参返回固定沙盒数据，或主动注入超时/异常（Fault Injection），以验证 Agent 的容错与自我纠错（Self-Correction）能力。
3. **时间旅行与断点分叉重跑 (`ReplayEngine`)：** 针对多步复杂轨迹，支持冻结前 $K-1$ 步的历史状态与工具返回结果（零 Token 消耗、零耗时回放），仅从第 $K$ 步开始执行修改后的新 Agent 逻辑（Fork Replay），并自动生成分叉前后的对比报告。

---

## 2. 目录结构与模块增量清单

请在保持 `v0.1.0` 现有文件向后兼容的基础上，新增或扩展以下文件（若现有仓库将模块置于 `src/agent_probe/sandbox/` 下，请统一在该包内实现并在 `src/agent_probe/__init__.py` 中导出核心 API）：

```text
Agent-probe/
├── plan_phase2.md                         # 本阶段工程蓝图
├── examples/
│   ├── mock_rules.yaml                    # [新增] 声明式工具打桩与异常注入配置示例
│   └── demo_agent_harness.py              # [新增] 零 API Key 依赖的沙盒故障注入与断点重跑演示脚本
├── src/
│   └── agent_probe/
│       ├── __init__.py                    # [更新] 导出 TraceSession, trace_tool, ToolSandbox, ReplayEngine
│       ├── core/
│       │   ├── schema.py                  # [兼容扩展] 新增 is_replayed 与 fork_step_id 字段支持
│       │   └── differ.py                  # [兼容扩展] 支持识别分叉重跑起点
│       ├── sandbox/
│       │   ├── __init__.py
│       │   ├── tracer.py                  # [新增] Task 2.1: 基于 contextvars 的同步/异步轨迹录制器
│       │   ├── mocker.py                  # [新增] Task 2.2: 确定性工具拦截与故障注入沙盒
│       │   └── replay.py                  # [新增] Task 2.3: 历史快照冻结与断点分叉重跑引擎
│       ├── reporter/
│       │   └── templates/report.html.j2   # [更新] Task 2.4: 新增 [MOCKED] / [REPLAYED] / [FORK POINT] 视觉标签
│       └── cli.py                         # [更新] Task 2.4: 新增 `agent-probe harness-demo` 与 `replay` 子命令
└── tests/
    ├── test_tracer.py                     # [新增] Task 2.1 单元测试
    ├── test_sandbox.py                    # [新增] Task 2.2 单元测试
    └── test_replay.py                     # [新增] Task 2.3 单元测试
```

---

## 3. 数据契约向后兼容扩展 (`schema.py`)

在不破坏 Phase 1 所有已有测试的前提下，对核心 Pydantic 模型进行以下带默认值的字段扩展：

1. **`Observation` 模型扩展：**
   * `is_mocked: bool = False`（标识该观测结果是否来自 `ToolSandbox` 打桩拦截）
   * `is_replayed: bool = False`（标识该观测结果是否来自 `ReplayEngine` 的历史轨迹回放）
   * `error_type: Optional[str] = None`（当 `is_error=True` 时，记录异常类名如 `"TimeoutError"`）
2. **`Trajectory` 模型扩展：**
   * `parent_trace_id: Optional[str] = None`（若为分叉重跑生成的轨迹，记录基准轨迹的 `trace_id`）
   * `fork_step_id: Optional[int] = None`（记录从哪一步开始发生分叉重跑）

---

## 4. 分阶段实施任务清单 (Step-by-Step Tasks)

### Task 2.1: 实现无侵入式轨迹采集器 (`src/agent_probe/sandbox/tracer.py`)

* **分支操作：** 从 `main` 分支切出 `feat/p2-tracer-sdk`
* **核心实现规格：**
  1. **上下文隔离：** 使用 Python 标准库 `contextvars.ContextVar` 维护当前活跃的 `TraceSession` 实例，确保多线程与 `asyncio` 并发安全。
  2. **`TraceSession` 上下文管理器：**
     * 初始化参数：`trace_id: Optional[str] = None`, `agent_version: str = "v0.2.0"`, `task_input: str = ""`
     * 支持同步 `with TraceSession(...) as session:` 与异步 `async with TraceSession(...) as session:`。
     * 提供显式记录接口：`session.record_thought(thought: str, state: str = "PLANNING", prompt_tokens: int = 0, completion_tokens: int = 0)`。
     * 提供结果设置接口：`session.set_final_output(output: str, status: str = "SUCCESS")`。
     * 在退出上下文时，自动汇总所有 Step 并构造完整的 `Trajectory` 对象（可通过 `session.trajectory` 获取，并支持 `session.save_json(filepath)`）。
  3. **`@trace_tool` 装饰器：**
     * 使用 `inspect.iscoroutinefunction` 自动识别并同时兼容普通函数 `def` 与异步函数 `async def`。
     * 若当前不在 `TraceSession` 上下文中，直接原样执行原函数（零副作用透传）。
     * 若处于活跃的 `TraceSession` 中：自动绑定入参字典 `arguments`、计时计算 `latency_ms`、捕获返回值封装为 `Observation`，并自动递增 `step_id` 追加到当前会话的 `steps` 列表中。
     * **异常捕获机制：** 若被装饰的工具抛出异常，自动记录一个 `is_error=True`、`error_type=type(e).__name__`、`output=str(e)` 的 Step，随后将异常继续向上抛出，交由上层 Agent 逻辑处理。
* **验证与提交标准：**
  * 编写并运行 `pytest tests/test_tracer.py -v`（覆盖同步函数、异步函数、嵌套调用、异常捕获及 JSON 序列化）。
  * 运行 `ruff check src/ tests/` 无警告。
  * 提交 Commit：`feat(sandbox): implement contextvars-based sync and async trajectory tracer`

---

### Task 2.2: 实现确定性工具打桩与故障注入沙盒 (`src/agent_probe/sandbox/mocker.py`)

* **分支操作：** 合并上一分支或基于其切出 `feat/p2-tool-mocker`
* **核心实现规格：**
  1. **声明式打桩规则模型 (`MockRule`)：**
     * `tool_name: str`：目标工具函数名。
     * `match_args: Optional[dict[str, Any]] = None`：参数子集匹配规则（若为 `None` 或空字典，则作为该工具的默认兜底规则 `*`）。
     * `return_value: Any = None`：命中规则时返回的确定性数据。
     * `raise_error: Optional[str] = None`：故障注入异常类型（如 `"TimeoutError"`, `"ConnectionError"`, `"ValueError"`）及错误提示信息 `error_message: str = "Mocked fault injection"`。
     * `simulated_latency_ms: float = 0.0`：模拟记录的耗时。
  2. **`ToolSandbox` 管理器：**
     * 支持通过 `ToolSandbox.from_yaml(filepath)` 或 `ToolSandbox(rules=[...])` 初始化。
     * 支持 `with sandbox:` 上下文激活。
     * **拦截联动：** 当 `@trace_tool` 装饰的函数在激活的 `ToolSandbox` 内被调用时，优先查询沙盒匹配规则（精确参数子集匹配优先于默认兜底匹配）：
       * 若命中 `raise_error`，自动构造对应异常并记录 `is_mocked=True, is_error=True` 的 Step 后抛出该异常。
       * 若命中 `return_value`，跳过真实函数体执行，直接返回预设值，并在录制的 Step 中标记 `Observation(output=return_value, is_mocked=True)`。
  3. **示例规则文件：** 创建 `examples/mock_rules.yaml`，包含电商物流查询的正常 Mock 返回、未知单号返回以及模拟接口超时（`TimeoutError`）的故障注入规则。
* **验证与提交标准：**
  * 编写并运行 `pytest tests/test_sandbox.py -v`（验证真实函数未被执行、参数匹配优先级、故障注入异常触发、`is_mocked` 标记正确性）。
  * 运行 `ruff check src/ tests/` 无警告。
  * 提交 Commit：`feat(sandbox): add deterministic tool mocking and fault injection engine`

---

### Task 2.3: 实现断点时间旅行与分叉重跑引擎 (`src/agent_probe/sandbox/replay.py`)

* **分支操作：** 切出特性分支 `feat/p2-replay-engine`
* **核心实现规格：**
  1. **`ReplayEngine` 核心类：**
     * 初始化参数：`baseline: Trajectory`, `fork_at_step: int`, `sandbox: Optional[ToolSandbox] = None`。
     * **工作原理（两阶段状态机）：**
       * **阶段 A —— 历史回放阶段 ($\text{current\_step} < \text{fork\_at\_step}$)：** 当被测 Agent 调用 `@trace_tool` 工具时，若当前步数小于 `fork_at_step` 且工具名称与 `baseline.steps` 中对应步骤一致，直接拦截调用并原样返回基准轨迹中的历史 `observation.output`（若历史步骤为异常则重放该异常），同时将该 Step 标记为 `is_replayed=True`，耗时计为 $0.0\text{ ms}$。
       * **阶段 B —— 分叉重跑阶段 ($\text{current\_step} \ge \text{fork\_at\_step}$)：** 到达断点步数后，自动解除历史锁定。后续的工具调用交由绑定的 `ToolSandbox` 或真实工具函数执行，记录新的推理与行动轨迹。
  2. **重跑结果封装 (`ReplayResult`)：**
     * 包含 `baseline_trajectory: Trajectory`、`candidate_trajectory: Trajectory`（自动设置 `parent_trace_id = baseline.trace_id` 和 `fork_step_id = fork_at_step`）。
     * 自动调用 Phase 1 的 `diff_trajectories(baseline, candidate)` 生成 `TrajectoryDiff` 对象。
* **验证与提交标准：**
  * 编写并运行 `pytest tests/test_replay.py -v`（构造一个 4 步轨迹，设置 `fork_at_step=3`，验证前 2 步严格返回历史数据且 `is_replayed=True`，第 3 步开始走向新分支并正确生成 Diff）。
  * 运行 `ruff check src/ tests/` 无警告。
  * 提交 Commit：`feat(sandbox): implement time-travel snapshot and fork replay engine`

---

### Task 2.4: CLI 升级、HTML 报告视觉徽章与端到端演示 (`cli.py` & `reporter/`)

* **分支操作：** 切出特性分支 `feat/p2-cli-and-docs`
* **核心实现规格：**
  1. **升级 HTML 报告模板 (`report.html.j2`)：**
     * 在步骤卡片上，若 `step.observation.is_mocked == True`，渲染醒目的紫/蓝色徽章 `[MOCKED 🛡️]`。
     * 若 `step.observation.is_replayed == True`，渲染灰色/青色徽章 `[REPLAYED ⏪]`。
     * 若该步等于 `fork_step_id` 或 `divergence_step_id`，在步骤左侧渲染醒目的高亮横幅 `[FORK / DIVERGENCE POINT ⚡]`。
  2. **构建端到端自包含沙盒演示 (`examples/demo_agent_harness.py`)：**
     * 编写一个纯本地模拟的售后 Agent 函数（无需任何外部 LLM API Key）：
       * **v1 缺陷版 Agent：** 在遇到 `query_logistics` 抛出 `TimeoutError`（由 `ToolSandbox` 故障注入触发）时，缺乏降级逻辑，连续重试 3 次陷入死循环。
       * **v2 修复版 Agent（通过 `ReplayEngine` 在第 2 步分叉重跑）：** 复用第 1 步历史状态，在第 2 步捕获超时异常后，立即切换调用备用工具 `issue_apology_coupon` 并优雅完成任务。
  3. **扩展 CLI 命令 (`src/agent_probe/cli.py`)：**
     * 新增子命令 `agent-probe harness-demo [--html probe_harness_report.html]`：一键运行上述沙盒故障注入与断点分叉重跑对比，在终端打印 Rich 报表并生成带 `[MOCKED]`、`[REPLAYED]` 与 `[FORK POINT]` 徽章的 HTML 报告。
  4. **版本升级与文档更新：**
     * 将 `pyproject.toml` 中的 `version` 更新为 `"0.2.0"`。
     * 更新 `README.md`，补充 Phase 2 的 `@trace_tool`、`ToolSandbox` 与 `ReplayEngine` 快速上手代码示例。
* **验证与提交标准：**
  * 运行 `pytest -v` 确保 Phase 1 与 Phase 2 的全量单元测试 $100\%$ 通过。
  * 执行 `agent-probe harness-demo` 验证终端输出与 `probe_harness_report.html` 生成无误。
  * 提交 Commit：`feat(cli): add harness-demo command and visual badges for v0.2.0`

---

## 5. Phase 2 交付质量守卫 (Definition of Done for v0.2.0)

在完成全部 4 个 Task 并准备合并回 `main` 分支发布 `v0.2.0` 前，必须依次执行以下验收命令并确保全部返回 Exit Code $0$：

```bash
# 1. 静态代码规范检查
ruff check src/ tests/

# 2. 全量单元测试（含 Phase 1 + Phase 2）
pytest -v

# 3. 离线轨迹评测演示（验证 Phase 1 回归正常）
agent-probe demo

# 4. 动态沙盒与断点重跑演示（验证 Phase 2 新增能力）
agent-probe harness-demo
```