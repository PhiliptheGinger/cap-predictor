"""Minimal CLI used for lightweight smoke testing.

The CLI intentionally avoids heavy dependencies and network access.  It simply
logs the project root using :mod:`sentimental_cap_predictor.config` and exercises
``extract_cmd`` from :mod:`sentimental_cap_predictor.cmd_utils`.
"""
from __future__ import annotations

import argparse

from . import config
from .cmd_utils import extract_cmd


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Lightweight smoke test CLI")
    parser.add_argument(
        "text",
        nargs="?",
        default="CMD: echo hello",
        help="Text input to parse for a command or question",
    )
    args = parser.parse_args(argv)

    # Demonstrate that configuration loading works
    config.log_project_root()

    cmd, question = extract_cmd(args.text)
    print(f"cmd={cmd!r} question={question!r}")


if __name__ == "__main__":  # pragma: no cover
    main()
