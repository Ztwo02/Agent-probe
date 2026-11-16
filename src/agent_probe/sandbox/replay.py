"""Time-travel snapshot replay and fork re-execution engine.

``ReplayEngine`` re-runs a (modified) agent against a recorded baseline ``Trajectory``:

* **Phase A - historical replay** (``step_id < fork_at_step``): every ``@trace_tool`` call
  whose tool name matches the baseline step with the same ``step_id`` is short-circuited
  and answered with the frozen historical observation (errors are re-raised with their
  original exception type). Replayed steps cost 0 ms and are flagged
  ``Observation.is_replayed = True``.
* **Phase B - fork execution** (``step_id >= fork_at_step``): history is unlocked and calls
  flow to the bound ``ToolSandbox`` (if any) or the real tool functions.

Usage::

    engine = ReplayEngine(baseline, fork_at_step=2, sandbox=ToolSandbox.from_yaml(...))
    result = engine.run(agent_v2, order_id="ORD-1")       # or: await engine.arun(...)
    result.diff.divergence_step_id

    # or as a context manager around arbitrary agent code:
    with engine as session:
        agent_v2(session)
    result = engine.result
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from contextlib import ExitStack
from types import TracebackType
from typing import Any

from pydantic import BaseModel, Field

from agent_probe.core.differ import diff_trajectories
from agent_probe.core.schema import Trajectory, TrajectoryDiff, TrajectoryStep
from agent_probe.sandbox.mocker import ToolSandbox, resolve_error_class
from agent_probe.sandbox.tracer import (
    InterceptResult,
    ToolInterceptor,
    ToolInvocation,
    TraceSession,
    pop_interceptor,
    push_interceptor,
)


class ReplayDivergenceError(RuntimeError):
    """Raised in ``strict`` mode when the agent deviates from history before the fork."""


class ReplayResult(BaseModel):
    """Outcome of a fork replay: both trajectories plus their Phase 1 diff."""

    baseline_trajectory: Trajectory
    candidate_trajectory: Trajectory
    diff: TrajectoryDiff
    fork_step_id: int = Field(..., ge=1)
    early_divergence_step_id: int | None = None

    @property
    def replayed_step_ids(self) -> list[int]:
        """Candidate step ids answered from the frozen baseline history."""
        return [
            step.step_id
            for step in self.candidate_trajectory.steps
            if step.observation is not None and step.observation.is_replayed
        ]


class ReplayEngine(ToolInterceptor):
    """Two-phase (replay -> fork) state machine for breakpoint re-execution."""

    def __init__(
        self,
        baseline: Trajectory,
        fork_at_step: int,
        sandbox: ToolSandbox | None = None,
        *,
        strict: bool = False,
        candidate_trace_id: str | None = None,
        agent_version: str | None = None,
        task_input: str | None = None,
    ) -> None:
        if fork_at_step < 1:
            raise ValueError("fork_at_step must be >= 1")
        self.baseline = baseline
        self.fork_at_step = fork_at_step
        self.sandbox = sandbox
        self.strict = strict
        self.candidate_trace_id = candidate_trace_id or f"{baseline.trace_id}-fork{fork_at_step}"
        self.agent_version = agent_version or f"{baseline.agent_version}-fork"
        self.task_input = baseline.task_input if task_input is None else task_input

        self._baseline_by_id: dict[int, TrajectoryStep] = {
            step.step_id: step for step in baseline.steps
        }
        self._session: TraceSession | None = None
        self._stack: ExitStack | None = None
        self.early_divergence_step_id: int | None = None
        self.result: ReplayResult | None = None

    # ------------------------------------------------------------------ state machine
    @property
    def session(self) -> TraceSession | None:
        """The candidate ``TraceSession`` while the engine is active."""
        return self._session

    @property
    def in_replay_phase(self) -> bool:
        """True while the next step is still served from frozen history.

        Agent code can use this to skip expensive LLM calls during Phase A.
        """
        if self._session is None:
            return False
        return self._session.next_step_id < self.fork_at_step

    def baseline_step(self, step_id: int) -> TrajectoryStep | None:
        """Return the frozen baseline step with ``step_id`` (if any)."""
        return self._baseline_by_id.get(step_id)

    def intercept(self, invocation: ToolInvocation) -> InterceptResult | None:
        if invocation.session is not self._session or invocation.step_id is None:
            return None
        step_id = invocation.step_id
        if step_id >= self.fork_at_step:
            return None  # Phase B: unlocked, defer to sandbox / real tool.

        historical = self._baseline_by_id.get(step_id)
        hist_tool = (
            historical.action.tool_name
            if historical is not None and historical.action is not None
            else None
        )
        if (
            historical is None
            or historical.observation is None
            or hist_tool != invocation.tool_name
        ):
            if self.early_divergence_step_id is None:
                self.early_divergence_step_id = step_id
            if self.strict:
                return InterceptResult(
                    error=ReplayDivergenceError(
                        f"Step {step_id}: agent called {invocation.tool_name!r} but baseline "
                        f"recorded {hist_tool!r} before fork point {self.fork_at_step}"
                    )
                )
            return None

        observation = historical.observation
        if observation.is_error:
            error_cls = resolve_error_class(observation.error_type or "RuntimeError")
            output = observation.output
            message = output if isinstance(output, str) else str(output)
            return InterceptResult(error=error_cls(message), latency_ms=0.0, is_replayed=True)
        return InterceptResult(output=observation.output, latency_ms=0.0, is_replayed=True)

    # ------------------------------------------------------------------ activation
    def __enter__(self) -> TraceSession:
        if self._session is not None:
            raise RuntimeError("ReplayEngine is already active.")
        self.early_divergence_step_id = None
        self.result = None
        session = TraceSession(
            trace_id=self.candidate_trace_id,
            agent_version=self.agent_version,
            task_input=self.task_input,
        )
        session.parent_trace_id = self.baseline.trace_id
        session.fork_step_id = self.fork_at_step

        stack = ExitStack()
        try:
            stack.enter_context(session)
            if self.sandbox is not None:
                stack.enter_context(self.sandbox)
            token = push_interceptor(self)  # innermost -> highest priority
            stack.callback(pop_interceptor, token)
        except BaseException:
            stack.close()
            raise
        self._session = session
        self._stack = stack
        return session

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        session, stack = self._session, self._stack
        self._session, self._stack = None, None
        if stack is not None:
            stack.__exit__(exc_type, exc, tb)
        if session is not None:
            candidate = session.trajectory
            self.result = ReplayResult(
                baseline_trajectory=self.baseline,
                candidate_trajectory=candidate,
                diff=diff_trajectories(self.baseline, candidate),
                fork_step_id=self.fork_at_step,
                early_divergence_step_id=self.early_divergence_step_id,
            )

    async def __aenter__(self) -> TraceSession:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------------ convenience runners
    def run(self, agent: Callable[..., Any], *args: Any, **kwargs: Any) -> ReplayResult:
        """Execute a synchronous agent callable under replay and return the result."""
        with self:
            agent(*args, **kwargs)
        assert self.result is not None
        return self.result

    async def arun(
        self, agent: Callable[..., Awaitable[Any]], *args: Any, **kwargs: Any
    ) -> ReplayResult:
        """Execute an async agent callable under replay and return the result."""
        async with self:
            await agent(*args, **kwargs)
        assert self.result is not None
        return self.result
