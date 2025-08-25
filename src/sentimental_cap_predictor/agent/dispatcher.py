from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, ValidationError, create_model

from .command_registry import get_registry

# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


@dataclass
class DispatchResult:
    """Container returned from :func:`dispatch`.

    Attributes
    ----------
    ok:
        Indicates whether the command executed successfully.
    message:
        Human readable status message.
    artifacts:
        Paths or identifiers of produced artifacts.
    metrics:
        Mapping of metric names to values returned by the command.
    """

    ok: bool
    message: str = ""
    artifacts: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_TYPE_NAMESPACE = {
    "Path": Path,
    "Sequence": Sequence,
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "dict": dict,
    "list": list,
    "None": type(None),
}


def _build_model(schema: Mapping[str, str]) -> type[BaseModel]:
    """Create a Pydantic model from the command schema."""

    fields: dict[str, tuple[Any, Any]] = {}
    for name, type_str in schema.items():
        annotation = eval(type_str, _TYPE_NAMESPACE)
        default = None if "None" in type_str else ...
        fields[name] = (annotation, default)
    return create_model(
        "Params",
        __config__=ConfigDict(extra="forbid"),
        **fields,
    )  # type: ignore[misc]


def _get_attr(obj: Any, name: str, default: Any | None = None) -> Any:
    """Return ``name`` from ``obj`` supporting both dicts and attributes."""

    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def dispatch(intent: Mapping[str, Any] | Any) -> DispatchResult:
    """Execute ``intent`` using the registered command handler.

    Parameters
    ----------
    intent:
        Mapping or object containing at least a ``command`` field and optional
        parameters.
    """

    try:
        if isinstance(intent, Mapping):
            command_name = intent["command"]
        else:
            command_name = intent.command
    except Exception:
        return DispatchResult(ok=False, message="missing command")

    registry = get_registry()
    command = registry.get(command_name)
    if not command:
        return DispatchResult(
            ok=False,
            message=f"unknown command: {command_name}",
        )

    params: Mapping[str, Any] | None = _get_attr(intent, "params")
    if params is None and isinstance(intent, Mapping):
        params = {k: v for k, v in intent.items() if k != "command"}
    if params is None:
        params = {}

    try:
        if command.params_schema:
            model = _build_model(command.params_schema)
            params = model(**params).model_dump()
    except ValidationError as exc:
        return DispatchResult(ok=False, message=str(exc))
    except Exception as exc:  # pragma: no cover - unexpected validation error
        return DispatchResult(ok=False, message=f"validation error: {exc}")

    try:
        output = command.handler(**params)
    except Exception as exc:
        return DispatchResult(ok=False, message=str(exc))

    message = _get_attr(output, "summary") or _get_attr(output, "message", "")
    metrics_obj = _get_attr(output, "metrics", {}) or {}
    artifacts_obj = _get_attr(output, "artifacts", []) or []
    reasoning = _get_attr(output, "reasoning", "")
    ok_flag = _get_attr(output, "ok", True)

    if isinstance(metrics_obj, Mapping):
        metrics = dict(metrics_obj)
    else:
        metrics = dict(metrics_obj)

    if isinstance(artifacts_obj, Mapping):
        artifacts = [str(v) for v in artifacts_obj.values()]
    elif isinstance(artifacts_obj, (str, Path)):
        artifacts = [str(artifacts_obj)]
    else:
        artifacts = [str(a) for a in artifacts_obj]

    if not message:
        if isinstance(output, Mapping) and not metrics and not artifacts:
            metrics = dict(output)
        elif output is not None:
            message = str(output)

    try:
        ok = bool(ok_flag)
    except Exception:  # pragma: no cover - non-boolean ok
        ok = True

    return DispatchResult(
        ok=ok,
        message=message,
        artifacts=artifacts,
        metrics=metrics,
        reasoning=reasoning,
    )
