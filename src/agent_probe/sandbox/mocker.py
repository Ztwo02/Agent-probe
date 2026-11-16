"""Deterministic tool mocking and fault injection sandbox.

A ``ToolSandbox`` intercepts calls to ``@trace_tool``-decorated functions while it is
active (``with sandbox:`` / ``async with sandbox:``). Calls that match a ``MockRule`` never
reach the real function body: they either return the rule's deterministic
``return_value`` or raise the configured fault (``raise_error``). Recorded steps are
flagged with ``Observation.is_mocked = True``.

Rule resolution for a tool call:

1. Rules with a non-empty ``match_args`` whose key/value pairs are all present in the call
   arguments (parameter defaults included) are *specific* matches. The most specific one
   (most matched keys) wins; ties are broken by declaration order.
2. Otherwise the first *default* rule (``match_args`` is ``None`` / ``{}``) for the tool wins.
3. Otherwise the real tool runs, unless the sandbox is ``strict`` in which case a
   ``SandboxViolationError`` is raised to block an unmocked external call.
"""

from __future__ import annotations

import builtins
import importlib
import threading
from contextvars import Token
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from agent_probe.sandbox.tracer import (
    InterceptResult,
    ToolInterceptor,
    ToolInvocation,
    pop_interceptor,
    push_interceptor,
)


class SandboxConfigError(ValueError):
    """Raised when a mock rules file or rule definition is invalid."""


class SandboxViolationError(RuntimeError):
    """Raised by a ``strict`` sandbox when a tool call has no matching mock rule."""


_SYNTHETIC_ERRORS: dict[str, type[Exception]] = {}
_SYNTHETIC_LOCK = threading.Lock()


def resolve_error_class(error_name: str) -> type[Exception]:
    """Resolve a fault-injection error name to an exception class.

    Resolution order: Python builtins (``"TimeoutError"``), dotted import paths
    (``"requests.exceptions.ConnectTimeout"``), then a cached synthetic subclass of
    ``RuntimeError`` carrying the requested name (``"RateLimitError"``), so YAML rules can
    simulate SDK-specific errors without importing the SDK.
    """
    name = error_name.strip()
    if not name:
        raise SandboxConfigError("raise_error must be a non-empty exception name")

    builtin = getattr(builtins, name, None)
    if isinstance(builtin, type) and issubclass(builtin, Exception):
        return builtin

    if "." in name:
        module_name, _, attr = name.rpartition(".")
        try:
            candidate = getattr(importlib.import_module(module_name), attr)
        except (ImportError, AttributeError):
            candidate = None
        if isinstance(candidate, type) and issubclass(candidate, Exception):
            return candidate
        name = attr

    if not name.isidentifier():
        raise SandboxConfigError(f"Invalid exception name for raise_error: {error_name!r}")
    with _SYNTHETIC_LOCK:
        if name not in _SYNTHETIC_ERRORS:
            _SYNTHETIC_ERRORS[name] = type(name, (RuntimeError,), {"__module__": __name__})
        return _SYNTHETIC_ERRORS[name]


class MockRule(BaseModel):
    """Declarative stub / fault-injection rule for a single tool."""

    tool_name: str = Field(..., min_length=1)
    match_args: dict[str, Any] | None = None
    return_value: Any = None
    raise_error: str | None = None
    error_message: str = "Mocked fault injection"
    simulated_latency_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("raise_error")
    @classmethod
    def _validate_error_name(cls, value: str | None) -> str | None:
        if value is not None:
            resolve_error_class(value)
        return value

    @property
    def is_default(self) -> bool:
        """Whether this rule is the catch-all (``*``) rule for its tool."""
        return not self.match_args

    def matches(self, arguments: dict[str, Any]) -> bool:
        """Return True when every ``match_args`` pair equals the call's argument value."""
        if self.is_default:
            return True
        assert self.match_args is not None
        return all(
            key in arguments and arguments[key] == expected
            for key, expected in self.match_args.items()
        )

    def build_error(self) -> Exception:
        """Instantiate the configured fault-injection exception."""
        if self.raise_error is None:
            raise ValueError("Rule has no raise_error configured")
        return resolve_error_class(self.raise_error)(self.error_message)


@dataclass(frozen=True)
class SandboxHit:
    """A tool call answered by the sandbox (useful for assertions in tests)."""

    tool_name: str
    arguments: dict[str, Any]
    rule: MockRule


class ToolSandbox(ToolInterceptor):
    """Context-activated interceptor that stubs tools and injects faults deterministically."""

    def __init__(self, rules: list[MockRule | dict[str, Any]] | None = None, strict: bool = False):
        self.rules: list[MockRule] = [
            rule if isinstance(rule, MockRule) else MockRule.model_validate(rule)
            for rule in (rules or [])
        ]
        self.strict = strict
        self.hits: list[SandboxHit] = []
        self._tokens: list[Token[Any]] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_yaml(cls, filepath: str | Path, strict: bool | None = None) -> ToolSandbox:
        """Load rules from YAML: either a top-level list or a mapping with a ``rules`` key."""
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Mock rules file not found: {path}")
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise SandboxConfigError(f"Invalid YAML in {path}: {exc}") from exc

        file_strict = False
        if isinstance(data, dict):
            file_strict = bool(data.get("strict", False))
            raw_rules = data.get("rules")
        else:
            raw_rules = data
        if not isinstance(raw_rules, list):
            raise SandboxConfigError(
                f"{path}: expected a list of rules or a mapping with a 'rules' list"
            )
        try:
            rules = [MockRule.model_validate(item) for item in raw_rules]
        except ValidationError as exc:
            raise SandboxConfigError(f"{path}: invalid mock rule: {exc}") from exc
        return cls(rules=rules, strict=file_strict if strict is None else strict)

    def add_rule(self, rule: MockRule | dict[str, Any]) -> MockRule:
        """Append a rule at runtime and return the validated ``MockRule``."""
        validated = rule if isinstance(rule, MockRule) else MockRule.model_validate(rule)
        self.rules.append(validated)
        return validated

    # ------------------------------------------------------------------ matching
    def find_rule(self, tool_name: str, arguments: dict[str, Any]) -> MockRule | None:
        """Return the rule that should answer ``tool_name(**arguments)``, if any."""
        best_specific: MockRule | None = None
        fallback: MockRule | None = None
        for rule in self.rules:
            if rule.tool_name != tool_name:
                continue
            if rule.is_default:
                if fallback is None:
                    fallback = rule
                continue
            if rule.matches(arguments) and (
                best_specific is None
                or len(rule.match_args or {}) > len(best_specific.match_args or {})
            ):
                best_specific = rule
        return best_specific or fallback

    def intercept(self, invocation: ToolInvocation) -> InterceptResult | None:
        rule = self.find_rule(invocation.tool_name, invocation.full_arguments)
        if rule is None:
            if self.strict:
                return InterceptResult(
                    error=SandboxViolationError(
                        f"Strict sandbox blocked unmocked tool call: {invocation.tool_name}"
                    ),
                    is_mocked=True,
                )
            return None

        with self._lock:
            self.hits.append(SandboxHit(invocation.tool_name, dict(invocation.arguments), rule))
        if rule.raise_error is not None:
            return InterceptResult(
                error=rule.build_error(),
                latency_ms=rule.simulated_latency_ms,
                is_mocked=True,
            )
        return InterceptResult(
            output=rule.return_value,
            latency_ms=rule.simulated_latency_ms,
            is_mocked=True,
        )

    # ------------------------------------------------------------------ activation
    @property
    def is_active(self) -> bool:
        return bool(self._tokens)

    def __enter__(self) -> ToolSandbox:
        self._tokens.append(push_interceptor(self))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        pop_interceptor(self._tokens.pop())

    async def __aenter__(self) -> ToolSandbox:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.__exit__(exc_type, exc, tb)
