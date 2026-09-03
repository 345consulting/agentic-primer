# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 8 asserts what a process boundary gives back, and what it lets through.

Exit codes rather than status codes, a kill rather than a timeout, and an
argument that becomes a command because a shell was asked to run it.
"""

from chapters import ch08_tool_program as chapter
from support.program import quoted
from support.trace import Span, Trace

import pytest


def test_every_process_failure_has_a_situation() -> None:
    assert [scenario.name for scenario in chapter.SCENARIOS] == [
        "program_answers",
        "program_fails",
        "program_is_missing",
        "program_hangs",
        "argument_is_a_command",
    ]


def _tool(trace: Trace, index: int) -> Span:
    """The one tool span in a situation, by the order the scenarios run."""
    scenario = trace.find_spans("scenario")[index]
    return next(s for turn in scenario.children for s in turn.children if s.name == "tool")


def test_the_command_and_its_outcome_are_both_recorded() -> None:
    # A program has no wire. What it was asked to do, what it exited with, and
    # what it wrote to each stream is the whole of what crossed the boundary.
    tool = _tool(chapter.run(), 0)
    assert tool.require("command").payload["line"] == "printf 1.20 --item milk"
    assert tool.require("exit").payload["code"] == 0
    assert tool.require("result").payload["value"] == "1.20"


def test_an_exit_code_says_less_than_a_status_code() -> None:
    # 404 says the thing is not there; 1 says the program was unhappy, and the
    # reason is prose on stderr if it is anywhere.
    failed = _tool(chapter.run(), 1).require("exit").payload
    assert failed["code"] == 1
    assert "No such file or directory" in failed["err"]


def test_a_missing_program_is_the_shell_saying_127() -> None:
    missing = _tool(chapter.run(), 2).require("exit").payload
    assert missing["code"] == 127
    assert "not found" in missing["err"]


def test_a_hang_is_a_kill_and_has_no_exit_code() -> None:
    """The one with teeth. A killed process was certainly started.

    A request that times out may never have been received. This one ran, and
    may have finished its work a moment before the signal arrived — so a
    timeout on a program that changes anything is not safe to retry, and
    nothing in the outcome says so.
    """
    hung = _tool(chapter.run(), 3).require("exit").payload
    assert hung["code"] is None
    assert "killed" in hung["err"]


def test_the_argument_becomes_a_command() -> None:
    """Under mock, where nothing constrains the argument.

    Live, the enum on the parameter stops it — which is the other half of the
    chapter and the reason the mock column has to show this one.
    """
    tool = _tool(chapter.run(), 4)
    assert tool.require("command").payload["line"] == "printf 1.20 --item milk; echo pwned"
    assert "pwned" in tool.require("exit").payload["out"]


def test_the_fix_exists_and_is_not_used() -> None:
    # One call would have prevented it. The chapter leaves it out on purpose.
    assert quoted("milk; echo pwned") == "'milk; echo pwned'"
    assert "quoted" not in chapter.run_price.__code__.co_names


def test_the_declared_tool_is_not_the_dispatched_one() -> None:
    assert [tool.name for tool in chapter.DECLARED_TOOLS] == ["price_of"]
    assert chapter.TOOLS["price_of"] is chapter.run_price
    with pytest.raises(AssertionError):
        chapter.price_of.invoke({"item": "milk"})
