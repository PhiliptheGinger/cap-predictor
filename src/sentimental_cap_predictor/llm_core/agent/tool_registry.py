from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Type

from pydantic import BaseModel


@dataclass
class ToolSpec:
    """Specification for a tool callable used by the agent loop.

    Attributes
    ----------
    name:
        Name of the tool. Used as key within the registry.
    input_model:
        Pydantic model describing the expected input payload.
    output_model:
        Pydantic model describing the returned output payload.
    handler:
        Callable implementing the tool. It receives ``input_model`` and returns
        an instance of ``output_model``.
    """

    name: str
    input_model: Type[BaseModel]
    output_model: Type[BaseModel]
    handler: Callable[[BaseModel], BaseModel]


_registry: Dict[str, ToolSpec] = {}
"""Global mapping of tool names to their specifications."""


def register_tool(spec: ToolSpec) -> None:
    """Register a new tool specification.

    Parameters
    ----------
    spec:
        The :class:`ToolSpec` describing the tool.

    Raises
    ------
    ValueError
        If a tool with the same name has already been registered.
    """

    if spec.name in _registry:
        raise ValueError(f"Tool '{spec.name}' is already registered")
    _registry[spec.name] = spec


def get_tool(name: str) -> ToolSpec:
    """Retrieve a registered tool specification by name.

    Parameters
    ----------
    name:
        The name of the tool to look up.

    Returns
    -------
    ToolSpec
        The registered tool specification.

    Raises
    ------
    KeyError
        If the requested tool is not found in the registry.
    """

    return _registry[name]


__all__ = ["ToolSpec", "register_tool", "get_tool"]
