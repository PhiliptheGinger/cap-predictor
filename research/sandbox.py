"""Execute strategy code in a restricted Python subprocess sandbox."""

# ruff: noqa: E402

from __future__ import annotations

import ast
import base64
import os
import pickle
import subprocess
import textwrap

from sentimental_cap_predictor.data_bundle import DataBundle
from sentimental_cap_predictor.research.types import (
    BacktestContext,
    BacktestResult,
    Idea,
)


class SandboxError(RuntimeError):
    """Raised when sandboxed execution fails or is unsafe."""


_DENY_IMPORTS: set[str] = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "requests",
    "http",
    "pathlib",
    "open",
    "eval",
    "exec",
}

_DENY_ATTRS: set[str] = {"__subclasses__", "__mro__", "__globals__"}


class _SafetyVisitor(ast.NodeVisitor):
    """AST visitor that rejects dangerous imports and attribute access."""

    def visit_Import(self, node: ast.Import) -> None:  # pragma: no cover
        for alias in node.names:
            if alias.name.split(".")[0] in _DENY_IMPORTS:
                raise SandboxError(f"Import '{alias.name}' denied")
            self.generic_visit(node)

    def visit_ImportFrom(
        self,
        node: ast.ImportFrom,
    ) -> None:  # pragma: no cover
        if node.module and node.module.split(".")[0] in _DENY_IMPORTS:
            raise SandboxError(f"Import '{node.module}' denied")
        self.generic_visit(node)

    def visit_Name(
        self,
        node: ast.Name,
    ) -> None:  # pragma: no cover
        if node.id in _DENY_IMPORTS:
            raise SandboxError(f"Use of '{node.id}' denied")
        self.generic_visit(node)

    def visit_Attribute(
        self,
        node: ast.Attribute,
    ) -> None:  # pragma: no cover
        if node.attr in _DENY_ATTRS:
            raise SandboxError(f"Access to '{node.attr}' denied")
        if node.attr in {"eval", "exec"}:
            raise SandboxError(f"Use of '{node.attr}' denied")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # pragma: no cover
        func = node.func
        if isinstance(func, ast.Name) and func.id in {"eval", "exec"}:
            raise SandboxError(f"Use of '{func.id}' denied")
        if isinstance(func, ast.Attribute) and func.attr in {"eval", "exec"}:
            raise SandboxError(f"Use of '{func.attr}' denied")
        self.generic_visit(node)


def _ast_safety_check(source: str) -> None:
    tree = ast.parse(source)
    _SafetyVisitor().visit(tree)


def _build_runner(
    payload: str,
    cpu_limit: int,
    mem_limit: int,
    op_limit: int,
) -> str:
    """Return the Python source executed in the sandboxed subprocess."""

    deny_imports = [i for i in _DENY_IMPORTS if i != "open"]
    return textwrap.dedent(
        f"""
        import base64
        import builtins
        import pickle
        import resource
        import sys

        _banned = (
            {deny_imports!r}
        )

        def _blocked_import(
            name,
            globals=None,
            locals=None,
            fromlist=(),
            level=0,
        ):
            root = name.split('.')[0]
            if root in _banned or any(item in _banned for item in fromlist):
                raise ImportError(f"blocked module: {{name}}")
            return __import__(name, globals, locals, fromlist, level)

        builtins.__import__ = _blocked_import
        builtins.open = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError('open disabled')
        )
        builtins.eval = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError('eval disabled')
        )
        builtins.exec = lambda *a, **k: (_ for _ in ()).throw(
            PermissionError('exec disabled')
        )

        resource.setrlimit(resource.RLIMIT_CPU, ({cpu_limit}, {cpu_limit}))
        resource.setrlimit(resource.RLIMIT_AS, ({mem_limit}, {mem_limit}))

        _op_limit = {op_limit}
        _ops = 0

        def _trace(frame, event, arg):
            global _ops
            if event == 'line':
                _ops += 1
                if _ops > _op_limit:
                    raise RuntimeError('operation limit exceeded')
            return _trace

        sys.settrace(_trace)

        source, data, idea, ctx = pickle.loads(base64.b64decode({payload!r}))
        globals_dict = {{'__builtins__': builtins}}
        locals_dict = {{}}
        exec(source, globals_dict, locals_dict)
        strategy = locals_dict.get('strategy')
        if strategy is None:
            raise RuntimeError('strategy not defined')
        from sentimental_cap_predictor.research.engine import simple_backtester
        backtester = simple_backtester(strategy)
        result = backtester(data, idea, ctx)
        sys.settrace(None)
        sys.stdout.write(base64.b64encode(pickle.dumps(result)).decode())
        """
    )


def run_strategy_source(
    source: str,
    data: DataBundle,
    idea: Idea,
    ctx: BacktestContext | None = None,
    cpu_seconds: int = 5,
    memory_bytes: int = 256 * 1024 * 1024,
    max_ops: int = 100_000,
) -> BacktestResult:
    """Execute ``source`` defining a ``strategy`` in a sandbox.

    Parameters
    ----------
    source:
        Python source code defining a ``strategy`` variable implementing the
        :class:`~sentimental_cap_predictor.research.types.Strategy` protocol.
    data, idea, ctx:
        Inputs forwarded to the backtester.
    cpu_seconds, memory_bytes:
        Resource limits for the subprocess.
    max_ops:
        Maximum number of traced line events allowed during execution.
    """

    _ast_safety_check(source)
    if ctx is None:
        ctx = BacktestContext()

    payload_data = pickle.dumps((source, data, idea, ctx))
    payload = base64.b64encode(payload_data).decode()
    runner = _build_runner(payload, cpu_seconds, memory_bytes, max_ops)

    env = {"PYTHONPATH": os.getcwd(), **os.environ}
    proc = subprocess.run(
        ["python", "-S", "-"],
        input=runner,
        text=True,
        capture_output=True,
        env=env,
        timeout=cpu_seconds + 1,
    )
    if proc.returncode != 0:
        raise SandboxError(proc.stderr.strip())
    try:
        return pickle.loads(base64.b64decode(proc.stdout.strip()))
    except Exception as exc:  # pragma: no cover - defensive
        raise SandboxError("Failed to deserialize result") from exc
