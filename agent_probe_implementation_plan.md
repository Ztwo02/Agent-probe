# Agent-Probe: 轻量级 Agent 轨迹白盒评测与回放底座 (Implementation Plan)

> **给 AI 编程助手 (Antigravity) 的全局指令：**
> 1. 本项目遵循 **Local-First（本地优先）、零外部数据库依赖、防腐层解耦** 的核心工程原则。
> 2. 请严格按照本计划中的 **Phase 1 -> Phase 2 -> Phase 3** 顺序分步实施。每完成一个子任务（Task），必须运行对应的验证命令（`pytest` 或 CLI 命令）确保通过后，再进入下一个任务。
> 3. 所有对外暴露的数据结构必须使用 `Pydantic v2` 进行严格类型校验。
> 4. 可视化报告采用**单文件自包含 HTML（Single-File HTML Report）**架构（内联 Tailwind CSS 与原生 JS/Chart），避免在 Python 仓库中引入繁重的 Node.js 构建工具链，确保用户零配置开箱即用。

---

## 1. 项目定位与核心价值

**`agent-probe`** 是一个面向 AI Agent 开发者的轻量级白盒轨迹测试与沙盒回放工具（Agent 领域的 `pytest` + `Playwright Trace Viewer`）。

* **核心痛点解决：** 改变传统黑盒评测只看最终输出的局限，对 Agent 的**状态流转路径（State Trajectory）、工具调用参数（Tool Call Arguments）、Token 预算及死循环异常**进行确定性断言与版本间 Diff 对比。
* **零门槛体验：** 内置真实场景的成功与失败轨迹示例（Mock Data），克隆仓库后无需配置任何大模型 API Key，一行命令即可跑通评测并生成可视化轨迹对比报告。

---

## 2. 技术栈与环境约束 (Tech Stack)

* **语言版本：** Python $\ge 3.10$
* **包管理与构建：** 标准 `pyproject.toml` (支持 `pip install -e .` 及 `uv`)
* **核心依赖库（保持极简）：**
  * `pydantic>=2.0`：核心轨迹协议与评测报告的数据结构校验
  * `typer>=0.12.0` + `rich>=13.0.0`：构建高可读性的命令行交互与终端表格输出
  * `pyyaml>=6.0`：解析声明式评测规则与 Mock 工具配置
  * `jinja2>=3.1.0`：渲染零依赖的单文件交互式 HTML 轨迹对比报告
* **开发与测试依赖：**
  * `pytest>=8.0.0`
  * `ruff` (代码规范与格式化)

---

## 3. 目标目录结构 (Target Directory Structure)

```text
agent-probe/
├── pyproject.toml                  # 项目元数据、依赖声明与 CLI 入口配置 (注册命令 `agent-probe`)
├── README.md                       # 门面文档（含快速开始与架构说明）
├── plan.md                         # 工程演进蓝图（本文件）
├── examples/                       # 开箱即用的零配置示例数据
│   ├── baseline_trace.json         # 基准版本运行轨迹（黄金标准成功案例）
│   ├── candidate_trace_loop.json   # 待测版本轨迹（包含工具重复调用死循环与超预算的缺陷案例）
│   └── eval_suite.yaml             # 声明式评测规则配置文件
├── src/
│   └── agent_probe/
│       ├── __init__.py
│       ├── cli.py                  # Typer CLI 入口 (check, diff, report, init)
│       ├── core/                   # 核心领域层（纯逻辑，零外部耦合）
│       │   ├── __init__.py
│       │   ├── schema.py           # 标准轨迹契约 (Trajectory, Step, ToolCall) 与报告模型
│       │   ├── engine.py           # 评测执行引擎（调度算子、汇总得分）
│       │   └── differ.py           # 轨迹 Diff 算法（对比两个版本的路径分叉、Token 增减与耗时差异）
│       ├── evaluators/             # 评测算子插件层
│       │   ├── __init__.py
│       │   ├── base.py             # 评测算子抽象基类 (BaseEvaluator)
│       │   └── deterministic.py    # 确定性算子：MaxStepGuard, TokenBudgetGuard, ToolCallMatch, PathOrderMatch
│       ├── sandbox/                # [Phase 2] 沙盒拦截与断点回放层
│       │   ├── __init__.py
│       │   ├── mocker.py           # @mock_tool 工具确定性打桩装饰器
│       │   └── tracer.py           # @trace_agent 轨迹自动录制装饰器
│       ├── adapters/               # 防腐接入层（转换第三方日志为标准 Schema）
│       │   ├── __init__.py
│       │   └── json_loader.py      # 本地 JSON/YAML 轨迹加载与格式校验器
│       └── reporter/               # 表现层
│           ├── __init__.py
│           ├── console.py          # Rich 终端高亮报表输出
│           ├── html_generator.py   # 单文件自包含 HTML 报告生成器
│           └── templates/
│               └── report.html.j2  # 内置轨迹流转图与 Diff 视图的前端模板
└── tests/                          # 单元测试与集成测试
    ├── test_schema.py
    ├── test_evaluators.py
    ├── test_differ.py
    └── test_cli.py
```

---

## 4. 核心数据契约设计 (`src/agent_probe/core/schema.py`)

请严格按照以下字段定义 Pydantic 模型：

### 4.1 轨迹数据模型 (Trajectory Models)
* `TokenUsage`:
  * `prompt_tokens: int = 0`
  * `completion_tokens: int = 0`
  * `total_tokens: int` (支持自动计算或校验 $\text{total} = \text{prompt} + \text{completion}$)
* `ToolCall`:
  * `tool_name: str`
  * `arguments: dict[str, Any]`
* `Observation`:
  * `output: Any`
  * `is_error: bool = False`
  * `is_mocked: bool = False`
* `TrajectoryStep`:
  * `step_id: int`
  * `state: str` (例如 `"PLANNING"`, `"TOOL_EXECUTION"`, `"RESPONDING"`, `"ERROR"`)
  * `thought: Optional[str] = None`
  * `action: Optional[ToolCall] = None`
  * `observation: Optional[Observation] = None`
  * `latency_ms: float = 0.0`
  * `tokens: TokenUsage`
* `Trajectory`:
  * `trace_id: str`
  * `agent_version: str`
  * `task_input: str`
  * `final_output: Optional[str] = None`
  * `status: Literal["SUCCESS", "FAILED", "TIMEOUT", "LOOP_ABORTED"]`
  * `steps: list[TrajectoryStep]`
  * 计算属性 (Computed Fields)：`total_steps: int`, `total_tokens: int`, `total_latency_ms: float`, `tool_sequence: list[str]`

### 4.2 评测结果与 Diff 模型 (Evaluation & Diff Models)
* `AssertionResult`:
  * `rule_name: str`
  * `passed: bool`
  * `score: float` (范围 $[0.0, 1.0]$)
  * `reason: str`
  * `failed_step_ids: list[int] = []`
* `EvalReport`:
  * `trace_id: str`
  * `agent_version: str`
  * `overall_passed: bool`
  * `overall_score: float`
  * `assertions: list[AssertionResult]`
* `TrajectoryDiff`:
  * `baseline_trace_id: str`
  * `candidate_trace_id: str`
  * `divergence_step_id: Optional[int]` (首次发生状态或工具调用分叉的步骤 ID)
  * `step_delta: int` ($\text{Candidate Steps} - \text{Baseline Steps}$)
  * `token_delta: int` ($\text{Candidate Tokens} - \text{Baseline Tokens}$)
  * `latency_delta_ms: float`
  * `step_comparisons: list[dict[str, Any]]` (对齐每一步的状态、工具名与参数差异)

---

## 5. 分阶段实施任务清单 (Step-by-Step Implementation Tasks)

### Phase 1: v0.1 最小闭环 MVP（离线评测引擎 + 轨迹 Diff + 静态可视化报告）

#### Task 1.1: 初始化项目脚手架与核心 Schema
* **操作：**
  1. 创建 `pyproject.toml`，配置项目名称 `agent-probe` 及 CLI 入口 `agent-probe = "agent_probe.cli:app"`。
  2. 实现 `src/agent_probe/core/schema.py` 中的所有 Pydantic 模型。
  3. 编写 `src/agent_probe/adapters/json_loader.py`，支持读取本地 JSON 文件并转化为 `Trajectory` 对象，提供清晰的校验错误提示。
* **验证标准：** 编写并运行 `pytest tests/test_schema.py`，确保合法 JSON 能正确解析，非法字段能触发 Pydantic ValidationError。

#### Task 1.2: 实现 5 个核心确定性评测算子 (`src/agent_probe/evaluators/`)
* **操作：**
  1. 在 `base.py` 中定义抽象基类 `BaseEvaluator`，包含接口 `evaluate(trajectory: Trajectory) -> AssertionResult`。
  2. 在 `deterministic.py` 中实现以下算子：
     * `MaxStepGuard(max_steps: int)`：检查总步数是否超过上限。
     * `TokenBudgetGuard(max_tokens: int)`：检查总 Token 消耗是否超标。
     * `LoopDetectionGuard(max_consecutive_repeats: int = 2)`：检测是否连续多次调用同名且同参数的工具（死循环检测），并精准记录触发死循环的 `failed_step_ids`。
     * `RequiredToolCallMatch(tool_name: str, expected_args_subset: Optional[dict])`：验证轨迹中是否至少调用了一次指定工具，且关键参数匹配。
     * `TrajectoryPathMatch(expected_sequence: list[str], mode: Literal["subsequence", "exact"] = "subsequence")`：验证工具或状态序列是否遵循预设顺序（如先调用 `query_order` 再调用 `issue_refund`）。
  3. 在 `core/engine.py` 中实现 `EvaluationEngine`，支持从 `eval_suite.yaml` 加载算子配置并对 `Trajectory` 执行批量断言。
* **验证标准：** 运行 `pytest tests/test_evaluators.py`，覆盖通过与失败的边界用例。

#### Task 1.3: 实现轨迹对比引擎 (`src/agent_probe/core/differ.py`)
* **操作：**
  1. 实现 `diff_trajectories(baseline: Trajectory, candidate: Trajectory) -> TrajectoryDiff`。
  2. 找出两个轨迹首次出现分叉的步骤 `divergence_step_id`（即 `state` 或 `action.tool_name` 不一致的第一步）。
  3. 计算总步数差值、Token 差值、耗时差值，并生成逐步对齐的对比数组。
* **验证标准：** 运行 `pytest tests/test_differ.py` 验证分叉点识别与指标差值计算的准确性。

#### Task 1.4: 构建示例数据集 (`examples/`)
* **操作：**
  1. 创建 `examples/baseline_trace.json`：模拟一个电商售后 Agent 正常完成“查询物流 -> 判断延误超过 48 小时 -> 生成赔偿券 -> 回复用户”的 4 步轨迹。
  2. 创建 `examples/candidate_trace_loop.json`：模拟修改 Prompt 后变笨的待测版本，在第 2 步查询物流失败后陷入连续 3 次重复调用同一工具的死循环，导致 Token 超支。
  3. 创建 `examples/eval_suite.yaml`：配置针对该场景的 5 条断言规则。

#### Task 1.5: 实现 CLI 终端交互与单文件 HTML 报告生成器
* **操作：**
  1. 实现 `reporter/console.py`：使用 `rich.table` 和 `rich.panel` 在终端输出评测通过率、死循环告警步数以及 Baseline vs Candidate 的 Diff 摘要。
  2. 实现 `reporter/html_generator.py` 与 `templates/report.html.j2`：
     * 生成一个完全独立、无外部本地静态资源依赖的 `.html` 文件。
     * 报告顶部展示核心指标卡片（得分、总步数对比、Token 变化率、首次分叉点高亮）。
     * 报告中部展示评测规则断言明细表（标注 PASS / FAIL 及错误原因）。
     * 报告底部左右并排展示 **Baseline vs Candidate 轨迹时间轴 Diff 视图**，可点击展开每一步的 `thought`、`action.arguments` 与 `observation`，对分叉或死循环步骤用醒目红色高亮。
  3. 在 `cli.py` 中实现 3 个核心子命令：
     * `agent-probe run <trace_file> --suite <yaml_file> [--html <output.html>]`
     * `agent-probe diff <baseline_file> <candidate_file> [--suite <yaml_file>] [--html <diff_report.html>]`
     * `agent-probe demo`：一键加载内置 `examples/` 数据，在终端打印对比报告并自动生成 `probe_demo_report.html`。
* **验证标准：** 运行 `pytest tests/test_cli.py`，并在命令行执行 `agent-probe demo` 确认能正常生成终端表格与 `probe_demo_report.html`。

---

### Phase 2: v0.2 沙盒拦截器与录制装饰器 (`src/agent_probe/sandbox/`)

> **注意：** 仅在 Phase 1 所有测试 100% 通过后开启本阶段。

* **Task 2.1: 轨迹自动录制器 (`tracer.py`)**
  * 提供 `@trace_session` 与 `@trace_step` 装饰器（或 Context Manager `with ProbeTracer() as tracer:`），让外部 Python Agent 代码无需侵入修改核心逻辑即可自动导出符合 `Trajectory` 标准的 JSON 文件。
* **Task 2.2: 确定性工具打桩器 (`mocker.py`)**
  * 支持从 `mock_tools.yaml` 读取预设的输入参数匹配规则与返回值（包括模拟超时或抛出异常）。
  * 提供 `@mock_tool` 装饰器，在评测模式下自动拦截真实 API 请求并将 `Observation.is_mocked` 标记为 `True`。

---

### Phase 3: v1.0 启发式语义算子与 CI 集成

* **Task 3.1: 可选的 LLM-as-a-Judge 算子 (`evaluators/llm_judge.py`)**
  * 通过标准 OpenAI 兼容接口（支持传入自定义 `base_url` 与 `api_key`）评估“规划合理性（Planning Coherence）”与“工具参数幻觉（Argument Hallucination）”。
* **Task 3.2: Markdown CI 报告导出**
  * 支持 `--format markdown` 参数，生成可直接粘贴到 GitHub Pull Request 评论区的评测升降分对比表。

---

## 6. 交付质量守卫 (Definition of Done for Phase 1)

在向用户交付 Phase 1 代码前，请依次执行以下检查命令并确保全部返回成功（Exit Code 0）：

1. 安装可编辑包：`pip install -e ".[dev]"`
2. 运行全量单元测试：`pytest -v`
3. 运行零配置演示命令：`agent-probe demo`
4. 验证生成的 `probe_demo_report.html` 文件非空且包含完整的 Baseline 与 Candidate 步骤对比数据。