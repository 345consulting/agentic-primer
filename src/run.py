# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Run one chapter, mock or live, and write its page.

    python -m run ch01_single_call [live]

Scaffolding, never a lesson, and deliberately not in `support/`: it imports a
chapter by name, and nothing in `support/` may import from `chapters/`. A
chapter is a lesson with no opinion about how it is invoked, so the entry
point lives here once rather than at the foot of every chapter.
"""

from support.trace import ModelKind, Trace, summary_line
from support.view import write

import importlib
import sys
from pathlib import Path


def main() -> None:
    """Mock unless told otherwise -- a live run costs money and is deliberate."""
    if len(sys.argv) < 2:
        raise SystemExit("usage: python -m run <chapter> [live]")
    chapter = sys.argv[1]
    model_kind: ModelKind = "live" if "live" in sys.argv[2:] else "mock"

    run = importlib.import_module(f"chapters.{chapter}").run
    trace: Trace = run(model_kind)
    print(f"{trace.chapter}: {summary_line(trace.summary)}")
    print(f"  {write(trace, Path('out'))}")


if __name__ == "__main__":
    main()
