# Agent-Probe GitHub 工程化与提交规范 (GitHub Specifications)

> **仓库地址：** `https://github.com/Ztwo02/Agent-probe.git`  
> **在线演示预留地址 (GitHub Pages)：** `https://ztwo02.github.io/Agent-probe/`  
>
> **给 AI 编程助手 (Antigravity) 的协作指令：**
> 1. 在执行任何 `git commit` 操作前，必须严格遵守本文档第 3 节的 **Conventional Commits（约定式提交）** 格式规范，严禁使用诸如 `"update code"`、`"fix bug"` 等模糊的提交信息。
> 2. 遵循 **“一个子任务（Task）一次原子提交”** 原则：每完成 `plan.md` 中的一个 Task 并通过第 4 节的质量门禁（`pytest` 与 `ruff`）后，再进行提交。
> 3. 严禁将 `.env` 密钥、虚拟环境目录（`.venv/`）、`__pycache__/` 或临时生成的测试 HTML 报告提交至版本库。

---

## 1. 本地仓库初始化与远程绑定 (Repository Initialization)

在本地新建项目文件夹并放入 `plan.md` 与 `GitHubSpecifications.md` 后，按以下标准流程完成与远程仓库 `Ztwo02/Agent-probe` 的绑定：

```bash
# 1. 初始化本地 Git 仓库并设置默认主分支名为 main
git init -b main

# 2. 绑定你的 GitHub 远程仓库地址
git remote add origin https://github.com/Ztwo02/Agent-probe.git

# 3. 若远程仓库创建时已勾选 README.md 或 LICENSE，先拉取并合并基础文件
git pull origin main --allow-unrelated-histories

# 4. 提交工程蓝图与规范文档作为项目的第一个基石 Commit
git add plan.md GitHubSpecifications.md
git commit -m "docs(core): initialize project implementation plan and GitHub specifications"
git push -u origin main
```

---

## 2. 分支管理策略 (Branching Strategy)

本项目采用轻量级的 **GitHub Flow** 分支模型，所有分支命名与 `plan.md` 的阶段任务严格对齐：

* **`main`（受保护主干分支）：** 永远处于 **可安装（Installable）、测试 $100\%$ 通过（All Tests Green）** 的稳定状态。
* **开发分支命名规范：** `<type>/<phase>-<short-description>`

### 推荐分支演进路线图：

```text
main (稳定主干)
 │
 ├─► feat/p1-schema-and-loader    (对应 Plan Task 1.1: 核心 Pydantic Schema 与 JSON 加载器)
 ├─► feat/p1-deterministic-evals  (对应 Plan Task 1.2: 5 个核心确定性评测算子与引擎)
 ├─► feat/p1-trajectory-differ    (对应 Plan Task 1.3: 轨迹分叉点检测与指标 Diff 算法)
 ├─► feat/p1-cli-and-html-report  (对应 Plan Task 1.4 ~ 1.5: 示例数据集、CLI 与单文件 HTML 报告)
 │     └── 合并入 main 并发布标签 ──► Release v0.1.0
 │
 ├─► feat/p2-sandbox-tracer       (对应 Plan Task 2.1: @trace_session 轨迹录制装饰器)
 └─► feat/p2-deterministic-mocker (对应 Plan Task 2.2: @mock_tool 工具确定性打桩拦截器)
       └── 合并入 main 并发布标签 ──► Release v0.2.0
```

### 日常切分支与合并标准命令：

```bash
# 开启新阶段任务前，从 main 切出特性分支
git checkout main
git pull origin main
git checkout -b feat/p1-schema-and-loader

# 完成该分支下所有子任务并通过测试后，推送到远程并发起 Pull Request (或本地快进合并)
git push -u origin feat/p1-schema-and-loader
```

---

## 3. Git Commit 提交规范 (Conventional Commits)

所有 Commit Message 必须遵循以下标准格式，以便 GitHub Releases 自动生成结构化的 Changelog：

```text
<type>(<scope>): <subject>

[optional body]
```

### 3.1 `<type>` 必填类型枚举

| 类型 (`type`) | 适用场景说明 | 典型示例 |
| :--- | :--- | :--- |
| **`feat`** | 新增功能、算子、数据模型或 CLI 子命令 | `feat(evaluators): implement LoopDetectionGuard operator` |
| **`fix`** | 修复 Bug、边界异常或类型校验错误 | `fix(differ): handle trajectories with unequal step lengths` |
| **`test`** | 新增或完善 `tests/` 目录下的单元测试 | `test(schema): add unit tests for Trajectory computed fields` |
| **`docs`** | 新增或修改文档（`README.md`, `plan.md`, 注释等） | `docs(readme): add quickstart guide and architecture diagram` |
| **`refactor`** | 代码重构（既不新增外部功能，也不修复 Bug） | `refactor(engine): decouple YAML parser from evaluation runner` |
| **`style`** | 仅调整代码格式、缩进或前端 HTML/CSS 视觉样式 | `style(reporter): improve diff highlight contrast in HTML report` |
| **`chore`** | 构建配置、依赖管理（`pyproject.toml`）、`.gitignore` 或 CI 脚本 | `chore(ci): configure GitHub Actions for pytest and Pages deploy` |

### 3.2 `<scope>` 模块作用域枚举（针对 `agent-probe` 定制）

* `schema`：核心数据结构（`src/agent_probe/core/schema.py`）
* `engine`：评测调度引擎（`src/agent_probe/core/engine.py`）
* `differ`：轨迹对比算法（`src/agent_probe/core/differ.py`）
* `evaluators`：确定性与启发式评测算子（`src/agent_probe/evaluators/`）
* `sandbox`：录制装饰器与工具 Mock 沙盒（`src/agent_probe/sandbox/`）
* `adapters`：外部 JSON/日志防腐接入层（`src/agent_probe/adapters/`）
* `reporter`：Rich 终端输出与单文件 HTML 报告生成器（`src/agent_probe/reporter/`）
* `cli`：Typer 命令行入口（`src/agent_probe/cli.py`）
* `examples`：内置演示轨迹与规则配置（`examples/`）
* `ci`：GitHub Actions 自动化工作流（`.github/workflows/`）

### 3.3 `<subject>` 书写铁律

1. 使用**英文祈使句**（动词原形开头，如 `add`, `implement`, `fix`, `update`，不要用过去式 `added` 或第三人称 `adds`）。
2. 首字母**小写**，结尾**不加句号**。
3. 长度控制在 $72$ 个字符以内，直观表达“本次提交做了什么”。

---

## 4. 提交前质量门禁 (Pre-Commit Quality Gates)

无论是由开发者手动提交，还是由 Antigravity 代为提交，在执行 `git commit` 之前**必须依次通过以下 3 项本地检查**：

```bash
# 门禁 1：代码规范与静态检查（零警告通过）
ruff check src/ tests/

# 门禁 2：全量单元测试（确保新增代码未破坏已有逻辑）
pytest -v

# 门禁 3：端到端冒烟验证（Phase 1 Task 1.5 完成后启用）
agent-probe demo
```

---

## 5. 标准 `.gitignore` 与 CI/CD 自动化配置

### 5.1 根目录 `.gitignore` 规范

在项目根目录创建 `.gitignore`，确保工作区干净：

```gitignore
# Python 虚拟环境与字节码缓存
.venv/
venv/
__pycache__/
*.py[cod]
*$py.class

# 打包与构建产物
dist/
build/
*.egg-info/

# 测试与代码检查缓存
.pytest_cache/
.ruff_cache/
.coverage
htmlcov/

# 本地生成的临时评测报告（注意：examples/ 下的数据集需要保留）
probe_demo_report.html
reports/
*.eval.html
public/

# 环境变量与密钥（严禁上传）
.env
.env.*
!.env.example

# 编辑器与 AI 助手本地缓存
.vscode/
.idea/
.antigravity/
.DS_Store
```

### 5.2 GitHub Actions 自动测试与在线 Demo 部署 (`.github/workflows/ci.yml`)

在仓库中创建 `.github/workflows/ci.yml`。每次推送到 `main` 分支时，自动运行测试并将生成的演示报告发布到 `https://ztwo02.github.io/Agent-probe/`：

```yaml
name: CI & Live Demo Deployment

on:
  push:
    branches: [ "main" ]
  pull_request:
    branches: [ "main" ]

permissions:
  contents: read
  pages: write
  id-token: write

jobs:
  test-and-build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Ztwo02/Agent-probe
        uses: actions/checkout@v4

      - name: Install uv and setup Python 3.11
        uses: astral-sh/setup-uv@v4
        with:
          python-version: "3.11"

      - name: Install package and dev dependencies
        run: uv pip install --system -e ".[dev]"

      - name: Lint with Ruff
        run: ruff check src/ tests/

      - name: Run Unit Tests with Pytest
        run: pytest -v

      - name: Generate Live Demo HTML Report
        if: github.ref == 'refs/heads/main'
        run: |
          mkdir -p public
          agent-probe demo
          mv probe_demo_report.html public/index.html

      - name: Upload GitHub Pages Artifact
        if: github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v3
        with:
          path: ./public

  deploy-pages:
    needs: test-and-build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

---

## 6. 里程碑版本发布规范 (Semantic Versioning & Releases)

本项目遵循语义化版本规范 `v<Major>.<Minor>.<Patch>`：

1. **`v0.1.0` (MVP Release)：** 完成 `plan.md` 中 Phase 1 的全部 5 个子任务，支持离线轨迹导入、5 大确定性算子评测、轨迹分叉 Diff 以及单文件 HTML 报告导出。
2. **`v0.2.0` (Sandbox Release)：** 完成 Phase 2，新增 `@trace_session` 自动轨迹录制与 `@mock_tool` 工具确定性打桩能力。
3. **`v1.0.0` (Production Release)：** 完成 Phase 3，支持可选的 LLM-as-a-Judge 启发式算子与 GitHub PR Markdown 评测报告导出。

### 发布操作标准流程：

```bash
# 1. 确保当前处于 main 分支且所有测试通过
git checkout main
git pull origin main
pytest -v

# 2. 创建带注释的版本标签并推送到 Ztwo02/Agent-probe
git tag -a v0.1.0 -m "Release v0.1.0: White-box trajectory evaluation engine and HTML diff viewer"
git push origin v0.1.0
```
推送标签后，前往 `https://github.com/Ztwo02/Agent-probe/releases/new`，选择 `v0.1.0` 标签，点击 **Generate release notes** 自动汇总你的规范 Commit 记录，并附上 `probe_demo_report.html` 的运行截图完成发布。

---

## 7. GitHub 仓库门面配置清单 (Repository Setup Checklist)

完成首次代码推送后，请在 GitHub 网页端完成以下 3 项配置，最大化提升开源项目的专业度：

1. **配置右侧 About 信息栏（点击仓库首页右上角齿轮图标）：**
   * **Description:** `Lightweight, local-first white-box trajectory evaluation & replay harness for AI Agents. 轻量级 Agent 轨迹白盒评测与回放底座`
   * **Website:** 勾选 `Use your GitHub Pages website`（自动填入 `https://ztwo02.github.io/Agent-probe/`）。
   * **Topics:** 依次添加 `ai-agents`、`agent-evaluation`、`trajectory-diff`、`agent-harness`、`llm-testing`、`python`、`white-box-testing`。
2. **开启 GitHub Pages 自动部署：**
   * 进入仓库顶部的 `Settings` $\rightarrow$ 左侧菜单点击 `Pages` $\rightarrow$ 在 `Build and deployment` 下的 `Source` 下拉框中选择 **`GitHub Actions`**。
3. **在 `README.md` 顶部挂载实时状态徽章：**
   ```markdown
   [![CI Status](https://github.com/Ztwo02/Agent-probe/actions/workflows/ci.yml/badge.svg)](https://github.com/Ztwo02/Agent-probe/actions)
   [![Live Demo Report](https://img.shields.io/badge/Live_Demo-HTML_Trace_Diff-blue)](https://ztwo02.github.io/Agent-probe/)
   [![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
   [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
   ```