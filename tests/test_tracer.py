"""Unit tests for the contextvars-based TraceSession and @trace_tool tracer."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from agent_probe import TraceSession, trace_tool
from agent_probe.adapters.json_loader import load_trajectory
from agent_probe.core.schema import Observation, Trajectory
from agent_probe.sandbox import get_current_session

CALLS: list[str] = []


@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict:
    CALLS.append(order_id)
    return {"order_id": order_id, "status": "IN_TRANSIT", "realtime": realtime_refresh}


@trace_tool(name="search_kb", state="RETRIEVAL")
async def async_search(query: str, top_k: int = 3) -> list[str]:
    await asyncio.sleep(0.001)
    return [f"{query}-{i}" for i in range(top_k)]


@trace_tool
def flaky_gateway(order_id: str) -> dict:
    raise TimeoutError(f"carrier gateway timed out for {order_id}")


@trace_tool
async def async_flaky(order_id: str) -> dict:
    raise ConnectionError("connection reset by peer")


@trace_tool
def handle_ticket(order_id: str) -> str:
    info = query_logistics(order_id)
    return f"ticket for {info['order_id']}"


@trace_tool
def with_kwargs(order_id: str, **extra: object) -> str:
    return order_id


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    CALLS.clear()


def test_passthrough_without_active_session() -> None:
    assert get_current_session() is None
    result = query_logistics("ORD-1")
    assert result["order_id"] == "ORD-1"
    assert CALLS == ["ORD-1"]
    assert query_logistics.__name__ == "query_logistics"


def test_sync_tool_records_step_with_thought_and_latency() -> None:
    with TraceSession(trace_id="t-sync", task_input="where is ORD-1?") as session:
        assert get_current_session() is session
        session.record_thought(
            "Query logistics first.", prompt_tokens=100, completion_tokens=20
        )
        out = query_logistics("ORD-1", realtime_refresh=True)
        session.set_final_output("Arrives tomorrow.")

    assert get_current_session() is None
    assert out["status"] == "IN_TRANSIT"
    traj = session.trajectory
    assert isinstance(traj, Trajectory)
    assert traj.trace_id == "t-sync"
    assert traj.agent_version == "v0.2.0"
    assert traj.status == "SUCCESS"
    assert traj.final_output == "Arrives tomorrow."
    assert traj.total_steps == 1

    step = traj.steps[0]
    assert step.step_id == 1
    assert step.state == "TOOL_EXECUTION"
    assert step.thought == "Query logistics first."
    assert step.action is not None
    assert step.action.tool_name == "query_logistics"
    assert step.action.arguments == {"order_id": "ORD-1", "realtime_refresh": True}
    assert step.observation == Observation(output=out)
    assert step.latency_ms >= 0.0
    assert step.tokens.total_tokens == 120


def test_async_tool_and_async_session() -> None:
    async def agent() -> TraceSession:
        async with TraceSession(task_input="kb lookup") as session:
            session.record_thought("Search the KB.")
            hits = await async_search("refund", top_k=2)
            assert hits == ["refund-0", "refund-1"]
            await async_search("policy")
        return session

    session = asyncio.run(agent())
    traj = session.trajectory
    assert traj.trace_id.startswith("trace-")
    assert traj.tool_sequence == ["search_kb", "search_kb"]
    assert [s.step_id for s in traj.steps] == [1, 2]
    assert traj.steps[0].state == "RETRIEVAL"
    assert traj.steps[0].thought == "Search the KB."
    assert traj.steps[1].thought is None
    assert traj.steps[0].action.arguments == {"query": "refund", "top_k": 2}
    assert traj.steps[1].action.arguments == {"query": "policy"}
    assert traj.steps[0].latency_ms > 0.0


def test_concurrent_async_sessions_are_isolated() -> None:
    async def run_one(label: str) -> Trajectory:
        async with TraceSession(trace_id=label) as session:
            for _ in range(3):
                await async_search(label, top_k=1)
        return session.trajectory

    async def main() -> list[Trajectory]:
        return await asyncio.gather(run_one("a"), run_one("b"))

    traj_a, traj_b = asyncio.run(main())
    assert traj_a.total_steps == 3 and traj_b.total_steps == 3
    assert all(s.action.arguments["query"] == "a" for s in traj_a.steps)
    assert all(s.action.arguments["query"] == "b" for s in traj_b.steps)


def test_nested_tool_calls_are_ordered_by_call_start() -> None:
    with TraceSession() as session:
        assert handle_ticket("ORD-9") == "ticket for ORD-9"

    traj = session.trajectory
    assert traj.tool_sequence == ["handle_ticket", "query_logistics"]
    assert [s.step_id for s in traj.steps] == [1, 2]
    assert traj.steps[0].latency_ms >= traj.steps[1].latency_ms


def test_sync_exception_is_recorded_and_reraised() -> None:
    with TraceSession() as session:
        with pytest.raises(TimeoutError, match="timed out"):
            flaky_gateway("ORD-7")
        session.set_final_output("Escalated to human.", status="FAILED")

    step = session.trajectory.steps[0]
    assert step.observation.is_error is True
    assert step.observation.error_type == "TimeoutError"
    assert step.observation.output == "carrier gateway timed out for ORD-7"
    assert step.observation.is_mocked is False
    assert session.trajectory.status == "FAILED"


def test_async_exception_is_recorded_and_reraised() -> None:
    async def agent() -> TraceSession:
        async with TraceSession() as session:
            with pytest.raises(ConnectionError):
                await async_flaky("ORD-8")
        return session

    step = asyncio.run(agent()).trajectory.steps[0]
    assert step.observation.is_error is True
    assert step.observation.error_type == "ConnectionError"


def test_unhandled_exception_marks_session_failed() -> None:
    session = TraceSession()
    with pytest.raises(TimeoutError):
        with session:
            flaky_gateway("ORD-X")
    assert session.trajectory.status == "FAILED"
    assert session.trajectory.total_steps == 1
    assert get_current_session() is None


def test_standalone_thoughts_are_flushed_as_steps() -> None:
    with TraceSession() as session:
        session.record_thought("Plan the approach.", state="PLANNING", prompt_tokens=10)
        session.record_thought("Decide to look up logistics.", completion_tokens=5)
        query_logistics("ORD-2")
        session.record_thought("Compose the answer.", state="RESPONDING", prompt_tokens=7)
        session.set_final_output("Done.")

    steps = session.trajectory.steps
    assert [s.step_id for s in steps] == [1, 2, 3]
    assert steps[0].state == "PLANNING" and steps[0].action is None
    assert steps[1].action.tool_name == "query_logistics"
    assert steps[1].thought == "Decide to look up logistics."
    assert steps[2].state == "RESPONDING" and steps[2].tokens.total_tokens == 7


def test_varkw_arguments_are_flattened() -> None:
    with TraceSession() as session:
        with_kwargs("ORD-3", channel="email", priority=1)
    assert session.trajectory.steps[0].action.arguments == {
        "order_id": "ORD-3",
        "channel": "email",
        "priority": 1,
    }


def test_invalid_final_status_rejected() -> None:
    with TraceSession() as session:
        with pytest.raises(ValueError):
            session.set_final_output("x", status="DONE")


def test_save_json_roundtrip(tmp_path: Path) -> None:
    with TraceSession(trace_id="t-json", task_input="订单查询") as session:
        session.record_thought("查询物流")
        query_logistics("ORD-中文")
        with pytest.raises(TimeoutError):
            flaky_gateway("ORD-中文")
        session.set_final_output("已处理")

    out = session.save_json(tmp_path / "nested" / "trace.json")
    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["trace_id"] == "t-json"
    assert raw["steps"][1]["observation"]["error_type"] == "TimeoutError"

    loaded = load_trajectory(out)
    assert loaded.trace_id == "t-json"
    assert loaded.tool_sequence == ["query_logistics", "flaky_gateway"]
    assert loaded.steps[0].thought == "查询物流"
    assert loaded.final_output == "已处理"
