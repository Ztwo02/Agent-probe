"""Unit tests for the deterministic ToolSandbox mocking and fault injection engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_probe import MockRule, ToolSandbox, TraceSession, trace_tool
from agent_probe.sandbox import SandboxConfigError, SandboxViolationError
from agent_probe.sandbox.mocker import resolve_error_class

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
REAL_CALLS: list[str] = []


@trace_tool
def query_logistics(order_id: str, realtime_refresh: bool = False) -> dict:
    REAL_CALLS.append(f"query_logistics:{order_id}")
    return {"order_id": order_id, "source": "real"}


@trace_tool
async def async_query(order_id: str) -> dict:
    REAL_CALLS.append(f"async_query:{order_id}")
    return {"order_id": order_id, "source": "real"}


@trace_tool
def send_email(to: str, body: str) -> dict:
    REAL_CALLS.append(f"send_email:{to}")
    return {"delivered": True, "source": "real"}


@pytest.fixture(autouse=True)
def _reset_calls() -> None:
    REAL_CALLS.clear()


def _logistics_sandbox(**kwargs: object) -> ToolSandbox:
    return ToolSandbox(
        rules=[
            MockRule(tool_name="query_logistics", return_value={"status": "DEFAULT"}),
            MockRule(
                tool_name="query_logistics",
                match_args={"order_id": "ORD-1"},
                return_value={"status": "IN_TRANSIT"},
                simulated_latency_ms=120.0,
            ),
            {
                "tool_name": "query_logistics",
                "match_args": {"order_id": "ORD-1", "realtime_refresh": True},
                "raise_error": "TimeoutError",
                "error_message": "504 Gateway Timeout",
                "simulated_latency_ms": 1200.0,
            },
        ],
        **kwargs,
    )


def test_mocked_return_skips_real_function_and_marks_step() -> None:
    sandbox = _logistics_sandbox()
    with TraceSession() as session, sandbox:
        result = query_logistics("ORD-1")

    assert result == {"status": "IN_TRANSIT"}
    assert REAL_CALLS == []
    step = session.trajectory.steps[0]
    assert step.observation.is_mocked is True
    assert step.observation.is_error is False
    assert step.observation.output == {"status": "IN_TRANSIT"}
    assert step.latency_ms == 120.0
    assert step.action.arguments == {"order_id": "ORD-1"}
    assert len(sandbox.hits) == 1
    assert sandbox.hits[0].rule.match_args == {"order_id": "ORD-1"}


def test_specific_match_beats_default_and_most_specific_wins() -> None:
    sandbox = _logistics_sandbox()
    assert sandbox.find_rule("query_logistics", {"order_id": "ORD-404"}).is_default
    assert sandbox.find_rule(
        "query_logistics", {"order_id": "ORD-1", "realtime_refresh": False}
    ).match_args == {"order_id": "ORD-1"}
    assert sandbox.find_rule(
        "query_logistics", {"order_id": "ORD-1", "realtime_refresh": True}
    ).raise_error == "TimeoutError"
    assert sandbox.find_rule("unknown_tool", {}) is None

    with sandbox:
        assert query_logistics("ORD-404") == {"status": "DEFAULT"}
    assert REAL_CALLS == []


def test_default_argument_values_participate_in_matching() -> None:
    sandbox = ToolSandbox(
        rules=[
            MockRule(
                tool_name="query_logistics",
                match_args={"realtime_refresh": False},
                return_value="cached",
            )
        ]
    )
    with TraceSession() as session, sandbox:
        assert query_logistics("ORD-2") == "cached"
    # recorded arguments stay as explicitly passed by the agent
    assert session.trajectory.steps[0].action.arguments == {"order_id": "ORD-2"}


def test_fault_injection_raises_and_records_mocked_error_step() -> None:
    with TraceSession() as session, _logistics_sandbox():
        with pytest.raises(TimeoutError, match="504 Gateway Timeout"):
            query_logistics("ORD-1", realtime_refresh=True)

    assert REAL_CALLS == []
    obs = session.trajectory.steps[0].observation
    assert obs.is_mocked is True
    assert obs.is_error is True
    assert obs.error_type == "TimeoutError"
    assert obs.output == "504 Gateway Timeout"
    assert session.trajectory.steps[0].latency_ms == 1200.0


def test_unmatched_tool_runs_real_function_unflagged() -> None:
    with TraceSession() as session, _logistics_sandbox():
        result = send_email("a@b.c", "hi")
    assert result["source"] == "real"
    assert REAL_CALLS == ["send_email:a@b.c"]
    assert session.trajectory.steps[0].observation.is_mocked is False


def test_strict_sandbox_blocks_unmocked_calls() -> None:
    with TraceSession() as session, _logistics_sandbox(strict=True):
        with pytest.raises(SandboxViolationError, match="send_email"):
            send_email("a@b.c", "hi")
    assert REAL_CALLS == []
    obs = session.trajectory.steps[0].observation
    assert obs.is_error and obs.is_mocked
    assert obs.error_type == "SandboxViolationError"


def test_sandbox_without_trace_session_still_mocks() -> None:
    with _logistics_sandbox():
        assert query_logistics("ORD-1") == {"status": "IN_TRANSIT"}
    assert REAL_CALLS == []
    # deactivated sandbox -> real function again
    assert query_logistics("ORD-1")["source"] == "real"


def test_async_tool_is_mocked_inside_async_contexts() -> None:
    sandbox = ToolSandbox(
        rules=[
            MockRule(tool_name="async_query", match_args={"order_id": "A"}, return_value="ok"),
            MockRule(tool_name="async_query", raise_error="ConnectionError"),
        ]
    )

    async def agent() -> TraceSession:
        async with TraceSession() as session, sandbox:
            assert await async_query("A") == "ok"
            with pytest.raises(ConnectionError, match="Mocked fault injection"):
                await async_query("B")
        return session

    steps = asyncio.run(agent()).trajectory.steps
    assert REAL_CALLS == []
    assert [s.observation.is_mocked for s in steps] == [True, True]
    assert steps[1].observation.error_type == "ConnectionError"


def test_nested_sandboxes_innermost_wins() -> None:
    outer = ToolSandbox(rules=[MockRule(tool_name="send_email", return_value="outer")])
    inner = ToolSandbox(rules=[MockRule(tool_name="send_email", return_value="inner")])
    with outer:
        with inner:
            assert send_email("x", "y") == "inner"
        assert send_email("x", "y") == "outer"
    assert not outer.is_active and not inner.is_active


def test_custom_and_synthetic_error_types() -> None:
    assert resolve_error_class("ValueError") is ValueError
    assert resolve_error_class("json.JSONDecodeError").__name__ == "JSONDecodeError"
    synthetic = resolve_error_class("RateLimitError")
    assert issubclass(synthetic, RuntimeError)
    assert resolve_error_class("RateLimitError") is synthetic

    sandbox = ToolSandbox(rules=[MockRule(tool_name="send_email", raise_error="RateLimitError")])
    with TraceSession() as session, sandbox:
        with pytest.raises(RuntimeError):
            send_email("x", "y")
    assert session.trajectory.steps[0].observation.error_type == "RateLimitError"

    with pytest.raises(ValueError):
        MockRule(tool_name="send_email", raise_error="not a valid name")


def test_from_yaml_example_rules() -> None:
    sandbox = ToolSandbox.from_yaml(EXAMPLES_DIR / "mock_rules.yaml")
    assert sandbox.strict is False
    assert {r.tool_name for r in sandbox.rules} >= {"query_logistics", "issue_apology_coupon"}

    with TraceSession() as session, sandbox:
        known = query_logistics("ORD-20261116-8891")
        unknown = query_logistics("ORD-UNKNOWN")
        with pytest.raises(TimeoutError, match="504"):
            query_logistics("ORD-20261116-8891", realtime_refresh=True)

    assert known["status"] == "CARRIER_SYNC_PENDING"
    assert unknown["error"] == "ORDER_NOT_FOUND"
    assert REAL_CALLS == []
    assert all(s.observation.is_mocked for s in session.trajectory.steps)


def test_from_yaml_rejects_invalid_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("rules: {tool_name: x}", encoding="utf-8")
    with pytest.raises(SandboxConfigError):
        ToolSandbox.from_yaml(bad)

    bad.write_text("- return_value: 1", encoding="utf-8")
    with pytest.raises(SandboxConfigError):
        ToolSandbox.from_yaml(bad)

    with pytest.raises(FileNotFoundError):
        ToolSandbox.from_yaml(tmp_path / "missing.yaml")

    ok = tmp_path / "ok.yaml"
    ok.write_text("- tool_name: send_email\n  return_value: sent\n", encoding="utf-8")
    assert ToolSandbox.from_yaml(ok, strict=True).strict is True
