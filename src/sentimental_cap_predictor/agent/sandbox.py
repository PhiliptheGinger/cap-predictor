from __future__ import annotations

import os
import shlex
import subprocess
from typing import Sequence

from .dispatcher import DispatchResult

# Allowed executables for safe_shell
_ALLOWED_BINARIES = {
    "python",
    "pytest",
    "ls",
    "dir",
    "cat",
    "type",
    "echo",
    "nvidia-smi",
    "pip",
    "git",
}


def run_subprocess(args: Sequence[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    """Execute ``args`` in a restricted subprocess environment.

    The subprocess receives no stdin, only a minimal environment, and its
    combined stdout/stderr is captured and returned.
    """

    env = {"PATH": os.environ.get("PATH", "")}
    if "PYTHONPATH" in os.environ:
        env["PYTHONPATH"] = os.environ["PYTHONPATH"]

    return subprocess.run(
        list(args),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        timeout=timeout,
        check=False,
    )


def safe_shell(cmd: str, timeout: int = 60) -> DispatchResult:
    """Execute ``cmd`` using a restricted whitelist-based policy.

    Parameters
    ----------
    cmd:
        Command string to execute. Only simple commands invoking whitelisted
        binaries without shell control operators are permitted.
    timeout:
        Maximum time in seconds the command is allowed to run.
    """

    if not cmd.strip():
        return DispatchResult(ok=False, message="empty command")

    forbidden = {"|", "&", ";", ">", "<"}
    if any(token in cmd for token in forbidden):
        return DispatchResult(ok=False, message="pipes and redirects are not allowed")

    try:
        args = shlex.split(cmd)
    except Exception as exc:
        return DispatchResult(ok=False, message=f"parse error: {exc}")

    binary = os.path.basename(args[0])
    if binary not in _ALLOWED_BINARIES:
        return DispatchResult(ok=False, message=f"command '{binary}' is not allowed")

    proc = run_subprocess(args, timeout=timeout)
    output = proc.stdout.rstrip()
    if proc.returncode != 0:
        return DispatchResult(ok=False, message=output)
    return DispatchResult(ok=True, message=output)
