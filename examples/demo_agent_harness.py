"""Zero-API-key Agent-Probe harness demo: fault injection + breakpoint fork replay.

Scenario (e-commerce after-sales agent, fully local, no LLM / network calls):

* ``ToolSandbox`` (``examples/mock_rules.yaml``, strict mode) stubs every external tool and
  injects a ``TimeoutError`` whenever ``query_logistics`` is called with
  ``realtime_refresh=True``.
* **v1 (defective)** has no fallback: after the injected timeout it retries the identical
  call 3 more times and finally aborts with ``LOOP_ABORTED``.
* **v2 (fixed)** is re-run with ``ReplayEngine(fork_at_step=2)``: step 1 is replayed from
  v1's frozen history (0 ms, no tool execution), step 2 hits the same injected timeout, and
  v2 immediately degrades to ``issue_apology_coupon`` and finishes successfully.

Run it directly::

    python examples/demo_agent_harness.py [--html probe_harness_report.html]

or through the CLI::

    agent-probe harness-demo
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_probe import ReplayEngine, ReplayResult, ToolSandbox, TraceSession, trace_tool
from agent_probe.core.engine import EvaluationEngine
from agent_probe.core.schema import EvalReport, Trajectory
from agent_probe.sandbox import get_current_session

EXAMPLES_DIR = Path(__file__).resolve().parent
MOCK_RULES_PATH = EXAMPLES_DIR / "mock_rules.yaml"

ORDER_ID = "ORD-20261116-8891"
TASK_INPUT = f"我的订单 {ORDER_ID} 已经下单三天了还没送到，怎么处理？"
BASELINE_TRACE_ID = "harness-v1-baseline"
CANDIDATE_TRACE_ID = "harness-v2-fork-replay"
DEFAULT_FORK_STEP = 2
V1_MAX_RETRIES = 3

# Declarative SLA suite evaluated against both runs (Phase 1 evaluators).
HARNESS_SUITE: dict[str, Any] = {
    "suite_name": "aftersales-harness-fault-tolerance-suite",
    "rules": [
        {"type": "max_steps", "name": "max_steps_within_5", "params": {"max_steps": 5}},
        {
            "type": "loop_detection",
            "name": "no_consecutive_tool_loop",
            "params": {"max_consecutive_repeats": 2},
        },
        {
            "type": "required_tool_call",
            "name": "must_fallback_to_apology_coupon",
            "params": {
                "tool_name": "issue_apology_coupon",
                "expected_args_subset": {"reason": "logistics_delay_timeout"},
            },
        },
        {
            "type": "path_match",
            "name": "logistics_before_coupon_path",
            "params": {
                "expected_sequence": ["query_logistics", "issue_apology_coupon"],
                "mode": "subsequence",
            },
        },
    ],
}


class RealExternalCallError(RuntimeError):
    """Raised if a real external API would be hit (the strict sandbox prevents this)."""


# --------------------------------------------------------------------------- tools
@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict[str, Any]:
    """Real SF-Express carrier API client (never reached inside the sandbox)."""
    raise RealExternalCallError("query_logistics would call the real carrier API")


@trace_tool
def issue_apology_coupon(
    order_id: str, amount: int = 20, reason: str = "logistics_delay_timeout"
) -> dict[str, Any]:
    """Real coupon service client (never reached inside the sandbox)."""
    raise RealExternalCallError("issue_apology_coupon would call the real coupon service")


def _session() -> TraceSession:
    session = get_current_session()
    if session is None:
        raise RuntimeError("Agents must run inside a TraceSession / ReplayEngine context.")
    return session


# --------------------------------------------------------------------------- agents
def aftersales_agent_v1(order_id: str = ORDER_ID) -> None:
    """Defective agent: retries the timed-out realtime query blindly and loops."""
    session = _session()
    session.record_thought(
        f"先调用 query_logistics 查询订单 {order_id} 的基础物流状态。",
        prompt_tokens=220,
        completion_tokens=45,
    )
    query_logistics(order_id)

    session.record_thought(
        "基础物流未返回延误时长，开启 realtime_refresh=true 查询承运商实时节点。",
        prompt_tokens=310,
        completion_tokens=52,
    )
    try:
        query_logistics(order_id, realtime_refresh=True)
        session.set_final_output("物流正常，请耐心等待。")
        return
    except TimeoutError:
        pass

    for attempt in range(1, V1_MAX_RETRIES + 1):
        session.record_thought(
            f"实时查询超时，没有降级策略，使用相同参数第 {attempt} 次重试。",
            prompt_tokens=360 + 40 * attempt,
            completion_tokens=40,
        )
        try:
            query_logistics(order_id, realtime_refresh=True)
            session.set_final_output("物流正常，请耐心等待。")
            return
        except TimeoutError:
            continue

    session.record_thought(
        "连续重试仍然超时且陷入重复调用，触发熔断中止执行。",
        state="ERROR",
        prompt_tokens=520,
        completion_tokens=30,
    )
    session.set_final_output(
        "Execution aborted: consecutive identical tool loop after carrier timeout",
        status="LOOP_ABORTED",
    )


def aftersales_agent_v2(order_id: str = ORDER_ID) -> None:
    """Fixed agent: on carrier timeout, degrade to an apology coupon immediately."""
    session = _session()
    session.record_thought(
        f"先调用 query_logistics 查询订单 {order_id} 的基础物流状态。",
        prompt_tokens=220,
        completion_tokens=45,
    )
    query_logistics(order_id)

    session.record_thought(
        "基础物流未返回延误时长，开启 realtime_refresh=true 查询承运商实时节点。",
        prompt_tokens=310,
        completion_tokens=52,
    )
    try:
        query_logistics(order_id, realtime_refresh=True)
        session.set_final_output("物流正常，请耐心等待。")
        return
    except TimeoutError:
        session.record_thought(
            "承运商实时接口超时：停止重试，降级为发放致歉优惠券并安抚客户。",
            prompt_tokens=340,
            completion_tokens=48,
        )

    coupon = issue_apology_coupon(order_id, amount=20, reason="logistics_delay_timeout")
    session.set_final_output(
        f"非常抱歉您的订单 {order_id} 物流延迟，承运商系统暂时繁忙。"
        f"已为您发放 {coupon['amount']} 元致歉优惠券 {coupon['coupon_code']}，"
        "我们会持续跟进物流进度。"
    )


# --------------------------------------------------------------------------- harness
@dataclass
class HarnessDemoOutcome:
    """Everything produced by one harness demo run."""

    baseline: Trajectory
    replay: ReplayResult
    baseline_report: EvalReport
    candidate_report: EvalReport

    @property
    def candidate(self) -> Trajectory:
        return self.replay.candidate_trajectory


def load_sandbox(rules_path: Path = MOCK_RULES_PATH) -> ToolSandbox:
    """Strict sandbox: any tool without a mock rule is blocked instead of hitting prod."""
    return ToolSandbox.from_yaml(rules_path, strict=True)


def record_baseline(sandbox: ToolSandbox) -> Trajectory:
    """Run the defective v1 agent under fault injection and record its trajectory."""
    with TraceSession(
        trace_id=BASELINE_TRACE_ID,
        agent_version="aftersales-agent-v1.0-no-fallback",
        task_input=TASK_INPUT,
    ) as session, sandbox:
        aftersales_agent_v1()
    return session.trajectory


def fork_replay(
    baseline: Trajectory, sandbox: ToolSandbox, fork_at_step: int = DEFAULT_FORK_STEP
) -> ReplayResult:
    """Re-run the fixed v2 agent from ``fork_at_step`` on top of v1's frozen history."""
    engine = ReplayEngine(
        baseline,
        fork_at_step=fork_at_step,
        sandbox=sandbox,
        candidate_trace_id=CANDIDATE_TRACE_ID,
        agent_version="aftersales-agent-v2.0-timeout-fallback",
    )
    return engine.run(aftersales_agent_v2)


def run_harness_demo(
    fork_at_step: int = DEFAULT_FORK_STEP, rules_path: Path = MOCK_RULES_PATH
) -> HarnessDemoOutcome:
    """Full pipeline: fault-injected baseline -> fork replay -> diff -> SLA evaluation."""
    sandbox = load_sandbox(rules_path)
    baseline = record_baseline(sandbox)
    replay = fork_replay(baseline, sandbox, fork_at_step=fork_at_step)
    engine = EvaluationEngine.from_config(HARNESS_SUITE)
    return HarnessDemoOutcome(
        baseline=baseline,
        replay=replay,
        baseline_report=engine.evaluate(baseline),
        candidate_report=engine.evaluate(replay.candidate_trajectory),
    )


def main(argv: list[str] | None = None) -> int:
    from agent_probe.reporter.console import render_diff_report
    from agent_probe.reporter.html_generator import generate_html_report

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--html", default="probe_harness_report.html")
    parser.add_argument("--fork-at-step", type=int, default=DEFAULT_FORK_STEP)
    args = parser.parse_args(argv)

    outcome = run_harness_demo(fork_at_step=args.fork_at_step)
    render_diff_report(
        outcome.replay.diff,
        baseline=outcome.baseline,
        candidate=outcome.candidate,
        baseline_report=outcome.baseline_report,
        candidate_report=outcome.candidate_report,
    )
    out = generate_html_report(
        args.html,
        baseline=outcome.baseline,
        candidate=outcome.candidate,
        diff=outcome.replay.diff,
        baseline_report=outcome.baseline_report,
        candidate_report=outcome.candidate_report,
        title="Agent-Probe Harness Demo: Fault Injection & Fork Replay",
    )
    print(f"Harness HTML report saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
