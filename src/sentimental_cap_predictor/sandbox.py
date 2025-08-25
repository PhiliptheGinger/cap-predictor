"""Execute untrusted strategy code in a restricted sandbox."""

from __future__ import annotations

import ast
import builtins
import multiprocessing as mp
import psutil
import time
from typing import Dict

# flake8: noqa


ALLOWED_MODULES = {"pandas", "numpy"}
DISALLOWED_BUILTINS = {
    "eval",
    "exec",
    "compile",
    "open",
    "input",
    "help",
    "dir",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__import__",
}
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in dir(builtins)
    if name not in DISALLOWED_BUILTINS
}

MAX_LOOP_ITERATIONS = 10_000


def _exec(code: str, conn: mp.connection.Connection, cpu_time: int, mem_limit: int) -> None:
    proc = psutil.Process()
    try:
        if hasattr(psutil, "RLIMIT_CPU"):
            proc.rlimit(psutil.RLIMIT_CPU, (cpu_time, cpu_time))
        if hasattr(psutil, "RLIMIT_AS"):
            proc.rlimit(psutil.RLIMIT_AS, (mem_limit, mem_limit))
    except Exception:
        pass
    env: Dict[str, object] = {"__builtins__": SAFE_BUILTINS}
    try:
        exec(code, env)
        env.pop("__builtins__", None)
        conn.send(env)
    except Exception as exc:  # pragma: no cover - relay exception to parent
        conn.send(exc)


def _validate(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name.split(".")[0]
            if module not in ALLOWED_MODULES:
                raise ValueError(f"Import of module '{module}' is not allowed")
        if isinstance(node, ast.Attribute):
            if (
                isinstance(node.value, ast.Name) and node.value.id == "__builtins__"
            ) or node.attr.startswith("__"):
                raise ValueError("Attribute access is not allowed")
        if isinstance(node, ast.While):
            raise ValueError("While loops are not allowed")
        if isinstance(node, ast.For):
            iter_call = node.iter
            if not (
                isinstance(iter_call, ast.Call)
                and isinstance(iter_call.func, ast.Name)
                and iter_call.func.id == "range"
            ):
                raise ValueError("For loops must use range with constant bounds")
            args = iter_call.args
            for arg in args:
                if not isinstance(arg, ast.Constant) or not isinstance(arg.value, int):
                    raise ValueError("Loop bounds must be integer constants")
            if len(args) == 1:
                iterations = args[0].value
            elif len(args) == 2:
                iterations = args[1].value - args[0].value
            elif len(args) == 3:
                iterations = (args[1].value - args[0].value) // args[2].value
            else:  # pragma: no cover - ast guarantees max 3 args
                iterations = MAX_LOOP_ITERATIONS + 1
            if iterations > MAX_LOOP_ITERATIONS:
                raise ValueError("Loop iteration count exceeds limit")


def run_code(
    code: str,
    timeout: int = 20,
    *,
    cpu_time: int | None = None,
    mem_limit: int = 500_000_000,
) -> Dict[str, object]:
    """Run ``code`` in a subprocess with limited builtins and imports."""

    _validate(code)
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    cpu = cpu_time or timeout
    proc = ctx.Process(target=_exec, args=(code, child_conn, cpu, mem_limit))
    proc.start()
    child_conn.close()
    proc_ps = psutil.Process(proc.pid)
    start = time.time()
    while proc.is_alive() and time.time() - start < timeout:
        if proc_ps.memory_info().rss > mem_limit:
            proc.terminate()
            raise RuntimeError("Strategy execution exceeded memory limit")
        proc.join(timeout=0.1)
    if proc.is_alive():
        proc.terminate()
        raise TimeoutError("Strategy execution timed out")
    if not parent_conn.poll():
        if proc.exitcode and proc.exitcode != 0:
            raise RuntimeError("Sandbox terminated unexpectedly")
        return {}
    result = parent_conn.recv()
    if isinstance(result, Exception):
        raise result
    return result
