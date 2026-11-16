"""Local JSON and YAML trajectory loader with clear validation diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from agent_probe.core.schema import Trajectory


class TraceLoadError(ValueError):
    """Raised when a trace file cannot be read or decoded into a JSON/YAML mapping."""


def format_validation_error(
    exc: ValidationError, source: str | Path | None = None
) -> str:
    """Format a Pydantic ValidationError into a readable multi-line diagnostic."""
    header = f"Invalid trajectory schema in '{source}':" if source else "Invalid trajectory schema:"
    lines = [header]
    for err in exc.errors():
        loc = ".".join(str(part) for part in err.get("loc", ())) or "<root>"
        msg = err.get("msg", "validation error")
        err_type = err.get("type", "unknown")
        lines.append(f"  - {loc}: {msg} [{err_type}]")
    return "\n".join(lines)


def load_trajectory_from_dict(data: dict[str, Any]) -> Trajectory:
    """Validate an in-memory dictionary into a Trajectory model."""
    return Trajectory.model_validate(data)


def load_trajectory(file_path: str | Path) -> Trajectory:
    """Load and validate a local JSON or YAML trace file into a Trajectory.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist or is not a file.
        TraceLoadError: If decoding fails or the root document is not a mapping.
        pydantic.ValidationError: If the payload violates the Trajectory schema.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Trajectory file not found: {path}")

    raw_text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            payload = yaml.safe_load(raw_text)
        except yaml.YAMLError as exc:
            raise TraceLoadError(f"Failed to parse YAML file '{path}': {exc}") from exc
    else:
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise TraceLoadError(
                f"Failed to parse JSON file '{path}' at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc

    if not isinstance(payload, dict):
        raise TraceLoadError(
            f"Expected a JSON/YAML object at root of '{path}', got {type(payload).__name__}."
        )

    return load_trajectory_from_dict(payload)
