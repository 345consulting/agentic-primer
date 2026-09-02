# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""How a chapter is run from the command line.

Scaffolding, never a lesson. Parsing one argument is not a mechanism of
agency, so it does not have to be built by hand in an earlier chapter first --
the graduation rule constrains what a chapter uses to *work*, and this is the
door, not the room.
"""

from support.trace import ModelKind, Trace, summary_line
from support.view import write

import sys
from collections.abc import Callable
from pathlib import Path


def main(run: Callable[[ModelKind], Trace]) -> None:
    """`python -m chapters.chNN [live]` -- mock unless told otherwise.

    Mock is the default because it costs nothing and is identical every time.
    A live run is a deliberate act, and spelling it out is the whole of the
    ceremony.
    """
    model_kind: ModelKind = "live" if "live" in sys.argv[1:] else "mock"
    trace = run(model_kind)
    # Writing the page and printing the path belong to whoever ran the
    # chapter. The recorder used to do it, which made `trace` import `view`
    # and `view` import `trace` -- a cycle broken only by a deferred import.
    path = write(trace, Path("out"))
    print(f"{trace.chapter}: {summary_line(trace.summary)}")
    print(f"  {path}")
