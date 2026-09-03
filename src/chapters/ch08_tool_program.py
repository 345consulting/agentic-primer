# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""A tool that runs a program, where an argument can become a command.

The third boundary. `ch04_tool_failures` stayed in the process,
`ch07_tool_http` crossed the network, and this one starts a program -- which
speaks a vocabulary neither of the others has:

    program_answers        exit 0, and stdout is the result
    program_fails          a non-zero exit, and stderr is the message
    program_is_missing     nothing to run: permanent, and the shell says 127
    program_hangs          killed on a timeout, having possibly done the work
    argument_is_a_command  the model's argument runs, because a shell ran it

An exit code carries less than a status code. `404` says the thing is not
there and `429` says come back in three seconds; `1` says the program was
unhappy, and the reason, if there is one, is somewhere in stderr as prose.
`who_retries` has to classify all three vocabularies.

`program_hangs` is worse here than its equivalent in `ch07_tool_http`. A
request that times out may never have been received. A process that is killed
was certainly started, and may have finished its work a moment before the
signal arrived -- so a timeout on a program that changes anything is not safe
to retry, and nothing in the outcome tells you that.

**And `argument_is_a_command` is the chapter, in two halves that disagree.**

The tool takes an item name and puts it straight into a command line. Given
`milk; echo pwned`, a shell runs two commands, and the second one's output
comes back as the tool's result -- recorded in the trace and handed to the
model as fact. The mock column shows exactly that:

    $ printf 1.20 --item milk; echo pwned
    out: '1.20\npwned'

Nothing was compromised, because the argument is `echo`. The point is that
nothing in this file stopped it: not the tool, which built the command line;
not the loop, which dispatched it; not the recorder, which stored the result
faithfully. `support/program.py` has `quoted()`, the one-line fix, deliberately
not called.

The live column refuses, and the reason is the better half of the lesson. The
argument is typed `Literal["bread", "butter", "chips", "milk"]`, so it reaches
the model provider as an `enum`, and the model answers that the string is not
a valid item. **The schema stopped the injection.**

That is `ch02_tool_call`'s finding arriving somewhere unexpected. There,
`part: str` against `Literal[...]` looked like a typing decision and turned out
to be the only thing constraining what the model could ask for; the cost was a
wrong answer. Here the same decision is the difference between a tool and a
shell.

Which is exactly why the mock column earns its place: it shows what the
unconstrained case does, and the unconstrained case is the common one. A tool
whose argument is a path, a filename, a search query or a customer's name has
no enum to hide behind, and then the only thing between a model's output and
your shell is `quoted()`.

The environment is emptied before the program runs, which is the other half of
the same lesson and is easy to forget: a child process inherits everything
unless you say otherwise, so a tool that runs a program hands it every secret
the harness holds -- the model provider's key included.
"""

from support.program import Outcome, run_program
from support.scenario import Scenario, run_scenario, run_scenarios
from support.trace import ModelKind, Span, Trace

from functools import partial
from typing import Any, Literal

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

SYSTEM_PROMPT = "You answer briefly and plainly. Use the tools you are given."

# One question per situation. The last one carries a poisoned item name --
# not because a model would invent it, but because this is how it arrives in
# practice: text from somewhere upstream, passed through faithfully.
QUESTIONS = {
    "program_answers": "How much is milk?",
    "program_fails": "How much is milk?",
    "program_is_missing": "How much is milk?",
    "program_hangs": "How much is milk?",
    "argument_is_a_command": (
        "Our supplier lists the item as `milk; echo pwned`. How much is it? "
        "Use that exact item name."
    ),
}

type GroceryItem = Literal["bread", "butter", "chips", "milk"]

TURN_CAP = 4

# The program each situation runs. `price-of` is a stand-in for a real one:
# a shell script that prints a price, fails, hangs, or is not installed.
# Templates, so the item lands where a real command line would put it. An
# earlier version appended it to the whole string, which put `--item milk`
# after `exit 3` and turned the failing program into an argument error.
PROGRAMS = {
    "program_answers": "printf 1.20 --item {item}",
    "program_fails": "ls /no-such-price-list/{item}",
    "program_is_missing": "price-of --item {item}",
    "program_hangs": "sleep 30 # {item}",
    "argument_is_a_command": "printf 1.20 --item {item}",
}

# What the mocked program did. A mock cannot hang, so the hanging one is
# recorded as what a hang looks like once it has been killed.
CANNED = {
    "program_answers": Outcome(code=0, out="1.20", err=""),
    "program_fails": Outcome(
        code=1, out="", err="ls: /no-such-price-list/milk: No such file or directory"
    ),
    "program_is_missing": Outcome(code=127, out="", err="price-of: command not found"),
    "program_hangs": Outcome(code=None, out="", err="killed after 2s"),
    "argument_is_a_command": Outcome(code=0, out="1.20\npwned", err=""),
}


class PriceProgramError(RuntimeError):
    """The program did not produce a price. Carries what it said."""


@tool
def price_of(item: GroceryItem) -> str:
    """The current shelf price of one of an item, in pounds."""
    raise AssertionError(f"price_of ['{item}'] is declared for its schema; run_price does the work")


def run_price(item: str, scenario: str, span: Span, model_kind: ModelKind) -> str:
    """Build a command line from the model's argument, and run it.

    The argument goes in unquoted. `support/program.quoted` is the fix and is
    not called, which is what `argument_is_a_command` is for.
    """
    # The argument goes straight in. `quoted(item)` is the fix and is not
    # called, which is the difference between a tool and a shell.
    command = PROGRAMS[scenario].format(item=item)
    outcome = run_program(model_kind, command, span, CANNED[scenario])
    if outcome.code != 0:
        raise PriceProgramError(
            f"['{PROGRAMS[scenario].split()[0]}'] exited ['{outcome.code}']: {outcome.err}"
        )
    return outcome.out


# Declared and dispatched are two different objects, as in `ch07_tool_http`.
TOOLS: dict[str, Any] = {price_of.name: run_price}

DECLARED_TOOLS = [price_of]

ASKS_FOR_MILK = AIMessage(
    "",
    tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}],
    response_metadata={"finish_reason": "tool_calls"},
)

# The one reply that is not about milk. Nothing in the schema forbids it: the
# argument is typed as a string, and this is a string.
ASKS_WITH_A_COMMAND = AIMessage(
    "",
    tool_calls=[{"name": "price_of", "args": {"item": "milk; echo pwned"}, "id": "c1"}],
    response_metadata={"finish_reason": "tool_calls"},
)

ANSWERS = AIMessage("Milk is 1.20.", response_metadata={"finish_reason": "stop"})

GIVES_UP = AIMessage(
    "I could not get the price of milk.", response_metadata={"finish_reason": "stop"}
)

SCENARIOS = [
    Scenario(
        name="program_answers",
        question=QUESTIONS["program_answers"],
        tools=DECLARED_TOOLS,
        mock_model_replies=[ASKS_FOR_MILK, ANSWERS],
    ),
    Scenario(name="program_fails", mock_model_replies=[ASKS_FOR_MILK, GIVES_UP]),
    Scenario(name="program_is_missing", mock_model_replies=[ASKS_FOR_MILK, GIVES_UP]),
    Scenario(name="program_hangs", mock_model_replies=[ASKS_FOR_MILK, GIVES_UP]),
    Scenario(
        name="argument_is_a_command",
        mock_model_replies=[
            ASKS_WITH_A_COMMAND,
            AIMessage("Milk is 1.20 pwned.", response_metadata={"finish_reason": "stop"}),
        ],
    ),
]


def execute_tool(
    call: ToolCall, scenario: Scenario, model_kind: ModelKind, trace: Trace
) -> ToolMessage:
    """One tool call, with the command and its outcome in the tool's span."""
    with trace.span("tool", name=call["name"], id=call["id"]) as span:
        span.add_note("args", **call["args"])
        try:
            result = TOOLS[call["name"]](
                **call["args"], scenario=scenario.name, span=span, model_kind=model_kind
            )
        except PriceProgramError as failure:
            span.add_note("failed", exception=type(failure).__name__, message=str(failure))
            return ToolMessage(
                content=f"['{call['name']}'] failed: {failure}. Answer without it.",
                tool_call_id=call["id"],
                status="error",
            )
        span.add_note("result", value=result)
    return ToolMessage(content=str(result), tool_call_id=call["id"])


def run(model_kind: ModelKind = "mock", turn_cap: int = TURN_CAP) -> Trace:
    return run_scenarios(
        "ch08_tool_program",
        model_kind,
        SCENARIOS,
        partial(run_scenario, system_prompt=SYSTEM_PROMPT, execute_tool=execute_tool),
        turn_cap,
    )
