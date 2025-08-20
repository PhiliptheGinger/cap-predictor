"""Execute untrusted strategy code in a restricted sandbox."""

from __future__ import annotations

import ast
import builtins
import multiprocessing as mp
from typing import Dict

ALLOWED_MODULES = {"pandas", "numpy"}
SAFE_BUILTINS = {name: getattr(builtins, name) for name in [
    "abs",
    "float",
    "int",
    "len",
    "min",
    "max",
    "range",
]}


def _exec(code: str, queue: mp.Queue) -> None:
    env: Dict[str, object] = {"__builtins__": SAFE_BUILTINS}
    exec(code, env)
    queue.put(env)


def _validate(code: str) -> None:
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module = node.names[0].name.split(".")[0]
            if module not in ALLOWED_MODULES:
                raise ValueError(f"Import of module '{module}' is not allowed")


def run_code(code: str, timeout: int = 5) -> Dict[str, object]:
    """Run ``code`` in a subprocess with limited builtins and imports."""

    _validate(code)
    queue: mp.Queue = mp.Queue()
    proc = mp.Process(target=_exec, args=(code, queue))
    proc.start()
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        raise TimeoutError("Strategy execution timed out")
    return queue.get() if not queue.empty() else {}
