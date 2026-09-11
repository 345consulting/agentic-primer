# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The seam between a mocked program and a real one.

Scaffolding, never a lesson. The third of these -- `models.py` switches the
model, `service.py` switches an HTTP service, and this switches a subprocess --
and they are the same shape on purpose: one function, so the only difference
between a mock run and a live run is which side of it you are on.

A program is recorded by hand rather than by hooks. There is no wire here: an
exit code, whatever the program wrote to each stream, and how long it took.
That is the whole of what a process boundary hands back, and it is less than
either of the other two boundaries gives you.
"""

from support.trace import ModelKind, Span

import shlex
import subprocess
from dataclasses import dataclass

# Long enough that a working program finishes, short enough that a hanging one
# is caught while you are still watching.
TIMEOUT_SECONDS = 2


@dataclass(frozen=True)
class Outcome:
    """What a process boundary gives back. Less than a response, and rougher."""

    code: int | None
    out: str
    err: str


def run_program(model_kind: ModelKind, command: str, span: Span, canned: Outcome | None) -> Outcome:
    """Run `command`, or return what the mock says it would have done.

    `command` is a string and is run through a shell, which is the point of
    `ch08_tool_program` rather than an oversight: it is how these tools are
    usually written, and it is why an argument the model chose can become a
    command it was never offered.
    """
    span.add_note("command", line=command)
    if model_kind == "mock":
        if canned is None:
            raise RuntimeError("a mock run needs a canned outcome; none was given")
        outcome = canned
    else:
        outcome = _execute(command)
    # `code` is None when nothing ran to completion: the program was killed on
    # a timeout, or was never there to start.
    span.add_note("exit", code=outcome.code, out=outcome.out[:500], err=outcome.err[:500])
    return outcome


def _execute(command: str) -> Outcome:
    """The real thing. Shell on, timeout on, output captured."""
    try:
        done = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            # An empty environment, so a program a model chose to run cannot
            # read the model provider's key or the service's out of ours.
            env={"PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        # Killed. A timed-out request may not have been received; a killed
        # process was certainly started, and may have finished its work.
        return Outcome(code=None, out="", err=f"killed after {TIMEOUT_SECONDS}s")
    return Outcome(code=done.returncode, out=done.stdout.strip(), err=done.stderr.strip())


def quoted(argument: str) -> str:
    """One argument, made safe to put in a command line.

    `shlex.quote` is the fix for the situation `argument_is_a_command`
    demonstrates. It is not used by default in that chapter, which is the
    difference between a tool and a shell.
    """
    return shlex.quote(argument)
