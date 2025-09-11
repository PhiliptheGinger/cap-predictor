"""Execute Python code in a sandboxed ``agent_work`` subprocess."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import List, Optional, Dict


def python_run(*, code: Optional[str] = None, path: Optional[str] = None, timeout: int = 10) -> Dict[str, object]:
    """Execute the provided Python ``code`` or file at ``path``.

    Exactly one of ``code`` or ``path`` must be supplied. The code runs in a
    subprocess with its working directory set to ``agent_work``. Basic network
    access is disabled by monkey patching :mod:`socket` in the executed script.

    Parameters
    ----------
    code:
        The Python source code to execute.
    path:
        Relative path below ``agent_work`` pointing to a Python file containing
        the code to execute.
    timeout:
        Maximum execution time in seconds for the subprocess.

    Returns
    -------
    dict
        A mapping with ``stdout``, ``stderr`` and ``paths`` keys where
        ``paths`` contains any new files created during execution (relative to
        ``agent_work``).
    """

    if (code is None) == (path is None):
        raise ValueError("Provide exactly one of 'code' or 'path'")

    base = Path("agent_work").resolve()
    base.mkdir(parents=True, exist_ok=True)

    target_code: str
    if path is not None:
        target = (base / path).resolve()
        if not str(target).startswith(str(base)):
            raise ValueError("path must reside within 'agent_work/'")
        if not target.exists():
            raise FileNotFoundError(f"File not found: {target}")
        target_code = target.read_text()
    else:
        target_code = code or ""

    preamble = (
        "import socket\n"
        "def _deny(*args, **kwargs):\n    raise RuntimeError('Network access disabled')\n"
        "for attr in ('socket', 'create_connection', 'create_server', 'getaddrinfo'):\n"
        "    setattr(socket, attr, _deny)\n"
    )
    script = preamble + "\n" + target_code

    script_file = base / "__temp_exec.py"
    script_file.write_text(script)

    before_files = {p.relative_to(base) for p in base.rglob('*') if p.is_file()}

    proc = subprocess.run(
        [sys.executable, str(script_file)],
        cwd=base,
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    after_files = {p.relative_to(base) for p in base.rglob('*') if p.is_file()}
    script_file.unlink(missing_ok=True)
    new_files = [str(p) for p in sorted(after_files - before_files) if "__pycache__" not in p.parts]

    return {"stdout": proc.stdout, "stderr": proc.stderr, "paths": new_files}


__all__ = ["python_run"]

# Optional agent tool registration
try:  # pragma: no cover - registration is optional at runtime
    from pydantic import BaseModel, root_validator

    from sentimental_cap_predictor.llm_core.agent.tool_registry import (
        ToolSpec,
        register_tool,
    )

    class PythonRunInput(BaseModel):
        code: Optional[str] = None
        path: Optional[str] = None

        @root_validator
        def _check_one(cls, values):  # type: ignore[override]
            if (values.get("code") is None) == (values.get("path") is None):
                raise ValueError("Provide exactly one of 'code' or 'path'")
            return values

    class PythonRunOutput(BaseModel):
        stdout: str
        stderr: str
        paths: List[str]

    def _python_run_handler(payload: PythonRunInput) -> PythonRunOutput:
        result = python_run(code=payload.code, path=payload.path)
        return PythonRunOutput(**result)

    register_tool(
        ToolSpec(
            name="python.run",
            input_model=PythonRunInput,
            output_model=PythonRunOutput,
            handler=_python_run_handler,
        )
    )
except Exception:  # pragma: no cover - silently ignore registration issues
    pass
