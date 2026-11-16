"""Non-intrusive, contextvars-based trajectory tracer for sync and async agent code.

Usage::

    from agent_probe.sandbox import TraceSession, trace_tool

    @trace_tool
    def query_logistics(order_id: str) -> dict: ...

    with TraceSession(task_input="Where is my order?") as session:
        session.record_thought("Look up the logistics status first.")
        query_logistics("ORD-1")
        session.set_final_output("Your parcel arrives tomorrow.")

    trajectory = session.trajectory

Thought recording semantics: ``record_thought`` buffers a *pending* thought. The next
``@trace_tool`` call made inside the session absorbs it (thought text + token usage) into
its own step, producing ReAct-style ``thought -> action -> observation`` steps that match
the Phase 1 trajectory contract. A pending thought that is not followed by a tool call is
flushed as a standalone step (with the given ``state``) when another thought is recorded,
when ``set_final_output`` is called, or when the session exits.
"""

from __future__ import annotations

import functools
import inspect
import json
import threading
import time
import uuid
from collections.abc import Callable
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, TypeVar, overload

from agent_probe.core.schema import (
    Observation,
    TokenUsage,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)

F = TypeVar("F", bound=Callable[..., Any])

DEFAULT_TOOL_STATE = "TOOL_EXECUTION"
_VALID_STATUSES = frozenset({"SUCCESS", "FAILED", "TIMEOUT", "LOOP_ABORTED"})

_current_session: ContextVar[TraceSession | None] = ContextVar(
    "agent_probe_current_trace_session", default=None
)


def get_current_session() -> TraceSession | None:
    """Return the ``TraceSession`` active in the current context, if any."""
    return _current_session.get()


@dataclass
class _PendingThought:
    thought: str
    state: str
    prompt_tokens: int
    completion_tokens: int


class TraceSession:
    """Context manager that records tool calls and thoughts into a ``Trajectory``.

    Supports both ``with TraceSession(...)`` and ``async with TraceSession(...)``. The active
    session is stored in a ``ContextVar`` so concurrent asyncio tasks and threads (that copy
    the context, e.g. ``asyncio.to_thread``) are isolated from each other.
    """

    def __init__(
        self,
        trace_id: str | None = None,
        agent_version: str = "v0.2.0",
        task_input: str = "",
    ) -> None:
        self.trace_id = trace_id or f"trace-{uuid.uuid4().hex[:12]}"
        self.agent_version = agent_version
        self.task_input = task_input
        self.final_output: str | None = None
        self.status: str | None = None
        self.parent_trace_id: str | None = None
        self.fork_step_id: int | None = None

        self._steps: list[TrajectoryStep] = []
        self._pending_thought: _PendingThought | None = None
        self._next_step_id = 1
        self._lock = threading.RLock()
        self._token: Token[TraceSession | None] | None = None
        self._trajectory: Trajectory | None = None

    # ------------------------------------------------------------------ context protocol
    def __enter__(self) -> TraceSession:
        if self._token is not None:
            raise RuntimeError("TraceSession is already active and cannot be re-entered.")
        self._trajectory = None
        self._token = _current_session.set(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self._flush_pending_thought()
            if self.status is None:
                self.status = "FAILED" if exc_type is not None else "SUCCESS"
            self._trajectory = self._build_trajectory()
        finally:
            if self._token is not None:
                _current_session.reset(self._token)
                self._token = None

    async def __aenter__(self) -> TraceSession:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)

    # ------------------------------------------------------------------ public API
    @property
    def is_active(self) -> bool:
        """Whether this session is currently entered."""
        return self._token is not None

    @property
    def steps(self) -> list[TrajectoryStep]:
        """Snapshot of the steps recorded so far, ordered by ``step_id``."""
        with self._lock:
            return sorted(self._steps, key=lambda s: s.step_id)

    @property
    def next_step_id(self) -> int:
        """The ``step_id`` that will be assigned to the next recorded step."""
        with self._lock:
            return self._next_step_id

    def record_thought(
        self,
        thought: str,
        state: str = "PLANNING",
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Buffer an agent thought; it is attached to the next traced tool call."""
        with self._lock:
            self._flush_pending_thought()
            self._pending_thought = _PendingThought(
                thought=thought,
                state=state,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )

    def set_final_output(self, output: str, status: str = "SUCCESS") -> None:
        """Set the trajectory's final answer and terminal status."""
        if status not in _VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}; expected one of {sorted(_VALID_STATUSES)}"
            )
        with self._lock:
            self._flush_pending_thought()
            self.final_output = output
            self.status = status

    @property
    def trajectory(self) -> Trajectory:
        """The recorded ``Trajectory`` (a live snapshot while the session is active)."""
        if self._trajectory is not None and not self.is_active:
            return self._trajectory
        return self._build_trajectory()

    def save_json(self, filepath: str | Path) -> Path:
        """Serialize the recorded trajectory to a JSON file loadable by ``agent-probe run``."""
        out = Path(filepath)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = self.trajectory.model_dump()
        out.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        return out

    # ------------------------------------------------------------------ recording internals
    def _allocate_step(self) -> tuple[int, _PendingThought | None]:
        """Reserve the next step id and pop any pending thought for a tool call."""
        with self._lock:
            step_id = self._next_step_id
            self._next_step_id += 1
            pending = self._pending_thought
            self._pending_thought = None
            return step_id, pending

    def _record_tool_step(
        self,
        *,
        step_id: int,
        pending: _PendingThought | None,
        tool_name: str,
        arguments: dict[str, Any],
        observation: Observation,
        latency_ms: float,
        state: str,
    ) -> TrajectoryStep:
        tokens = (
            TokenUsage(
                prompt_tokens=pending.prompt_tokens,
                completion_tokens=pending.completion_tokens,
            )
            if pending is not None
            else TokenUsage()
        )
        step = TrajectoryStep(
            step_id=step_id,
            state=state,
            thought=pending.thought if pending is not None else None,
            action=ToolCall(tool_name=tool_name, arguments=arguments),
            observation=observation,
            latency_ms=max(latency_ms, 0.0),
            tokens=tokens,
        )
        with self._lock:
            self._steps.append(step)
        return step

    def _flush_pending_thought(self) -> None:
        with self._lock:
            pending = self._pending_thought
            if pending is None:
                return
            self._pending_thought = None
            step_id = self._next_step_id
            self._next_step_id += 1
            self._steps.append(
                TrajectoryStep(
                    step_id=step_id,
                    state=pending.state,
                    thought=pending.thought,
                    tokens=TokenUsage(
                        prompt_tokens=pending.prompt_tokens,
                        completion_tokens=pending.completion_tokens,
                    ),
                )
            )

    def _build_trajectory(self) -> Trajectory:
        with self._lock:
            steps = sorted(self._steps, key=lambda s: s.step_id)
            return Trajectory(
                trace_id=self.trace_id,
                agent_version=self.agent_version,
                task_input=self.task_input,
                final_output=self.final_output,
                status=self.status or "SUCCESS",  # type: ignore[arg-type]
                steps=steps,
                parent_trace_id=self.parent_trace_id,
                fork_step_id=self.fork_step_id,
            )


# ---------------------------------------------------------------------- @trace_tool


def _bind_arguments(
    signature: inspect.Signature | None, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Map a call's positional/keyword arguments to a flat ``{param: value}`` dict."""
    if signature is None:
        merged: dict[str, Any] = {f"arg{i}": value for i, value in enumerate(args)}
        merged.update(kwargs)
        return merged
    try:
        bound = signature.bind(*args, **kwargs)
    except TypeError:
        merged = {f"arg{i}": value for i, value in enumerate(args)}
        merged.update(kwargs)
        return merged

    arguments: dict[str, Any] = {}
    for name, value in bound.arguments.items():
        param = signature.parameters[name]
        if name in {"self", "cls"} and param.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        }:
            continue
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            arguments.update(value)
        elif param.kind is inspect.Parameter.VAR_POSITIONAL:
            arguments[name] = list(value)
        else:
            arguments[name] = value
    return arguments


def _error_observation(exc: Exception, **flags: bool) -> Observation:
    return Observation(output=str(exc), is_error=True, error_type=type(exc).__name__, **flags)


@overload
def trace_tool(func: F) -> F: ...


@overload
def trace_tool(
    func: None = None, *, name: str | None = None, state: str = DEFAULT_TOOL_STATE
) -> Callable[[F], F]: ...


def trace_tool(
    func: F | None = None,
    *,
    name: str | None = None,
    state: str = DEFAULT_TOOL_STATE,
) -> F | Callable[[F], F]:
    """Decorate a sync or async tool so its calls are recorded into the active session.

    Can be used bare (``@trace_tool``) or with options (``@trace_tool(name="search")``).
    Outside of an active ``TraceSession`` the wrapped function runs untouched.
    """

    def decorator(fn: F) -> F:
        tool_name = name or fn.__name__
        try:
            signature: inspect.Signature | None = inspect.signature(fn)
        except (TypeError, ValueError):
            signature = None

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                session = _current_session.get()
                if session is None:
                    return await fn(*args, **kwargs)

                arguments = _bind_arguments(signature, args, kwargs)
                step_id, pending = session._allocate_step()
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                except Exception as exc:
                    session._record_tool_step(
                        step_id=step_id,
                        pending=pending,
                        tool_name=tool_name,
                        arguments=arguments,
                        observation=_error_observation(exc),
                        latency_ms=(time.perf_counter() - start) * 1000.0,
                        state=state,
                    )
                    raise
                session._record_tool_step(
                    step_id=step_id,
                    pending=pending,
                    tool_name=tool_name,
                    arguments=arguments,
                    observation=Observation(output=result),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    state=state,
                )
                return result

            async_wrapper.__agent_probe_tool_name__ = tool_name  # type: ignore[attr-defined]
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            session = _current_session.get()
            if session is None:
                return fn(*args, **kwargs)

            arguments = _bind_arguments(signature, args, kwargs)
            step_id, pending = session._allocate_step()
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                session._record_tool_step(
                    step_id=step_id,
                    pending=pending,
                    tool_name=tool_name,
                    arguments=arguments,
                    observation=_error_observation(exc),
                    latency_ms=(time.perf_counter() - start) * 1000.0,
                    state=state,
                )
                raise
            session._record_tool_step(
                step_id=step_id,
                pending=pending,
                tool_name=tool_name,
                arguments=arguments,
                observation=Observation(output=result),
                latency_ms=(time.perf_counter() - start) * 1000.0,
                state=state,
            )
            return result

        sync_wrapper.__agent_probe_tool_name__ = tool_name  # type: ignore[attr-defined]
        return sync_wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)
    return decorator
