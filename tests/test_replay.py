"""Unit tests for the time-travel snapshot replay and fork re-execution engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_probe import MockRule, ReplayEngine, ToolSandbox, TraceSession, trace_tool
from agent_probe.adapters.json_loader import load_trajectory
from agent_probe.core.differ import diff_trajectories
from agent_probe.core.schema import Trajectory
from agent_probe.sandbox import ReplayDivergenceError, get_current_session

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
REAL_CALLS: list[str] = []
_TICK = {"n": 0}


@trace_tool
def lookup_order(order_id: str) -> dict:
    REAL_CALLS.append("lookup_order")
    _TICK["n"] += 1
    return {"order_id": order_id, "paid": True, "tick": _TICK["n"]}


@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict:
    REAL_CALLS.append("query_logistics")
    _TICK["n"] += 1
    return {"order_id": order_id, "status": "DELAYED", "tick": _TICK["n"]}


@trace_tool
def escalate_to_human(order_id: str) -> str:
    REAL_CALLS.append("escalate_to_human")
    return "ticket-001"


@trace_tool
def issue_apology_coupon(order_id: str, amount: int = 10) -> dict:
    REAL_CALLS.append("issue_apology_coupon")
    return {"coupon": f"SORRY-{amount}"}


@trace_tool
async def async_lookup(order_id: str) -> dict:
    REAL_CALLS.append("async_lookup")
    return {"order_id": order_id, "live": True}


@trace_tool
def flaky_gateway(order_id: str) -> dict:
    REAL_CALLS.append("flaky_gateway")
    raise TimeoutError("504 Gateway Timeout")


def agent_v1(order_id: str) -> None:
    session = get_current_session()
    assert session is not None
    session.record_thought("Look up the order.", prompt_tokens=100, completion_tokens=10)
    lookup_order(order_id)
    session.record_thought("Check logistics.", prompt_tokens=150, completion_tokens=12)
    query_logistics(order_id)
    session.record_thought("Retry with realtime refresh.", prompt_tokens=200, completion_tokens=15)
    query_logistics(order_id, realtime_refresh=True)
    session.record_thought("Escalate.", prompt_tokens=250, completion_tokens=8)
    escalate_to_human(order_id)
    session.set_final_output("Escalated to a human agent.")


def agent_v2(order_id: str) -> None:
    session = get_current_session()
    assert session is not None
    session.record_thought("Look up the order.", prompt_tokens=100, completion_tokens=10)
    lookup_order(order_id)
    session.record_thought("Check logistics.", prompt_tokens=150, completion_tokens=12)
    query_logistics(order_id)
    session.record_thought(
        "Delayed: compensate the customer.", prompt_tokens=180, completion_tokens=9
    )
    issue_apology_coupon(order_id, amount=20)
    session.set_final_output("Sorry for the delay, here is a 20 CNY coupon.")


@pytest.fixture(autouse=True)
def _reset() -> None:
    REAL_CALLS.clear()
    _TICK["n"] = 0


@pytest.fixture()
def baseline() -> Trajectory:
    with TraceSession(trace_id="base-001", agent_version="v1", task_input="ORD-1 is late") as s:
        agent_v1("ORD-1")
    REAL_CALLS.clear()
    return s.trajectory


def test_fork_at_step_3_replays_history_then_branches(baseline: Trajectory) -> None:
    assert baseline.total_steps == 4
    engine = ReplayEngine(baseline, fork_at_step=3)
    result = engine.run(agent_v2, "ORD-1")

    cand = result.candidate_trajectory
    # Phase A: steps 1-2 never touched the real tools and returned frozen history.
    assert REAL_CALLS == ["issue_apology_coupon"]
    assert result.replayed_step_ids == [1, 2]
    for step_id in (1, 2):
        c_step, b_step = cand.steps[step_id - 1], baseline.steps[step_id - 1]
        assert c_step.observation.is_replayed is True
        assert c_step.observation.output == b_step.observation.output
        assert c_step.latency_ms == 0.0
    assert cand.steps[1].observation.output["tick"] == 2  # historical, not re-computed

    # Phase B: step 3 executed live on the new branch.
    step3 = cand.steps[2]
    assert step3.action.tool_name == "issue_apology_coupon"
    assert step3.observation.is_replayed is False
    assert step3.observation.output == {"coupon": "SORRY-20"}

    # Lineage + diff.
    assert cand.parent_trace_id == "base-001"
    assert cand.fork_step_id == 3
    assert cand.trace_id == "base-001-fork3"
    assert cand.task_input == "ORD-1 is late"
    assert cand.status == "SUCCESS"
    assert result.baseline_trajectory is baseline
    assert result.fork_step_id == 3
    assert result.early_divergence_step_id is None
    assert result.diff.divergence_step_id == 3
    assert result.diff.fork_step_id == 3
    assert result.diff.step_delta == -1
    flags = [(c["is_replayed"], c["is_fork_point"]) for c in result.diff.step_comparisons]
    assert flags == [(True, False), (True, False), (False, True), (False, False)]
    assert engine.result is result


def test_in_replay_phase_flag_tracks_state_machine(baseline: Trajectory) -> None:
    engine = ReplayEngine(baseline, fork_at_step=2)
    phases: list[bool] = []
    assert engine.in_replay_phase is False
    with engine as session:
        phases.append(engine.in_replay_phase)
        lookup_order("ORD-1")
        phases.append(engine.in_replay_phase)
        query_logistics("ORD-1")
        assert session is engine.session
    assert phases == [True, False]
    assert engine.session is None
    assert engine.result.replayed_step_ids == [1]
    assert REAL_CALLS == ["query_logistics"]


def test_replayed_error_is_reraised_with_original_type() -> None:
    with TraceSession(trace_id="err-base") as s:
        with pytest.raises(TimeoutError):
            flaky_gateway("ORD-1")
        escalate_to_human("ORD-1")
    REAL_CALLS.clear()

    def agent() -> None:
        try:
            flaky_gateway("ORD-1")
        except TimeoutError:
            issue_apology_coupon("ORD-1")

    result = ReplayEngine(s.trajectory, fork_at_step=2).run(agent)
    assert REAL_CALLS == ["issue_apology_coupon"]
    obs = result.candidate_trajectory.steps[0].observation
    assert obs.is_replayed and obs.is_error
    assert obs.error_type == "TimeoutError"
    assert obs.output == "504 Gateway Timeout"
    assert result.diff.divergence_step_id == 2


def test_tool_mismatch_before_fork_falls_through_to_live_execution(
    baseline: Trajectory,
) -> None:
    def agent() -> None:
        lookup_order("ORD-1")
        issue_apology_coupon("ORD-1")  # baseline step 2 was query_logistics
        query_logistics("ORD-1")

    result = ReplayEngine(baseline, fork_at_step=4).run(agent)
    assert result.early_divergence_step_id == 2
    assert result.replayed_step_ids == [1, 3]
    assert REAL_CALLS == ["issue_apology_coupon"]


def test_strict_mode_raises_on_pre_fork_divergence(baseline: Trajectory) -> None:
    engine = ReplayEngine(baseline, fork_at_step=4, strict=True)

    def agent() -> None:
        lookup_order("ORD-1")
        escalate_to_human("ORD-1")

    with pytest.raises(ReplayDivergenceError, match="Step 2"):
        engine.run(agent)
    assert engine.result is not None
    assert engine.result.candidate_trajectory.status == "FAILED"
    assert engine.result.candidate_trajectory.steps[1].observation.error_type == (
        "ReplayDivergenceError"
    )
    assert REAL_CALLS == []


def test_phase_b_calls_are_served_by_bound_sandbox(baseline: Trajectory) -> None:
    sandbox = ToolSandbox(
        rules=[
            MockRule(tool_name="lookup_order", return_value={"mocked": True}),
            MockRule(tool_name="issue_apology_coupon", return_value={"coupon": "MOCK"}),
        ]
    )
    result = ReplayEngine(baseline, fork_at_step=3, sandbox=sandbox).run(agent_v2, "ORD-1")
    steps = result.candidate_trajectory.steps
    assert REAL_CALLS == []
    # replay takes precedence over the sandbox for pre-fork steps
    assert steps[0].observation.is_replayed and not steps[0].observation.is_mocked
    assert steps[0].observation.output["paid"] is True
    assert steps[2].observation.is_mocked and steps[2].observation.output == {"coupon": "MOCK"}
    assert not sandbox.is_active


def test_async_arun() -> None:
    async def record() -> Trajectory:
        async with TraceSession(trace_id="async-base") as s:
            await async_lookup("A")
            await async_lookup("B")
        return s.trajectory

    async def agent() -> None:
        await async_lookup("A")
        await async_lookup("C")

    async def main() -> tuple[Trajectory, object]:
        base = await record()
        REAL_CALLS.clear()
        return base, await ReplayEngine(base, fork_at_step=2).arun(agent)

    base, result = asyncio.run(main())
    assert REAL_CALLS == ["async_lookup"]
    assert result.replayed_step_ids == [1]
    assert result.diff.divergence_step_id is None  # same tools -> ARGS_DIFF only
    assert result.diff.step_comparisons[1]["status"] == "ARGS_DIFF"


def test_replay_from_phase1_json_fixture() -> None:
    baseline = load_trajectory(EXAMPLES_DIR / "candidate_trace_loop.json")

    @trace_tool(name="query_logistics")
    def live_query(order_id: str, realtime_refresh: bool = False) -> dict:
        REAL_CALLS.append("live")
        return {}

    def agent() -> None:
        first = live_query("ORD-20261116-8891")
        assert first["status"] == "CARRIER_SYNC_PENDING"
        with pytest.raises(RuntimeError, match="504 Gateway Timeout"):
            live_query("ORD-20261116-8891", realtime_refresh=True)
        issue_apology_coupon("ORD-20261116-8891")

    result = ReplayEngine(baseline, fork_at_step=3).run(agent)
    assert REAL_CALLS == ["issue_apology_coupon"]
    assert result.replayed_step_ids == [1, 2]
    assert result.diff.divergence_step_id == 3
    assert result.diff.fork_step_id == 3


def test_validation_and_diff_lineage_guard(baseline: Trajectory) -> None:
    with pytest.raises(ValueError):
        ReplayEngine(baseline, fork_at_step=0)

    unrelated = baseline.model_copy(update={"trace_id": "other", "fork_step_id": 2})
    other = baseline.model_copy(update={"parent_trace_id": "zzz", "fork_step_id": 2})
    assert diff_trajectories(baseline, other).fork_step_id is None
    assert diff_trajectories(unrelated, unrelated).fork_step_id == 2
