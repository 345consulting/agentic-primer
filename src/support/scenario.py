# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""One run of a chapter's loop, under one condition.

Scaffolding, never a lesson. A scenario is how a chapter observes the same
loop under conditions it chooses -- the same category as `Trace` and the wire
hooks, which is why it may be imported rather than built by hand in an earlier
chapter first. It does not participate in the loop. It names what varies.

Built by hand in `ch05_loop_endings`, which ran one loop twice to show
that two replies with no tool calls can mean different things, and graduated
here for every chapter after it.

The fields grow one at a time, as a chapter needs one. There is no bag of
conditions: a field that is named can be grepped, and a chapter that varies
something new should have to say so here.
"""

from support.agent import ask_model, run_turns
from support.trace import ENDED, ModelKind, Trace

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import BaseTool


@dataclass(frozen=True)
class Scenario:
    """What varies between two runs of the same loop.

    `mock_model_replies` is what the mock model says, ignored on a live run,
    exactly as it is everywhere else. Anything else here is a condition the
    chapter imposes on the run rather than on the model.
    """

    name: str
    mock_model_replies: Sequence[AIMessage]
    # The question is part of the situation, not a constant beside it: ch04
    # and ch08 kept per-scenario question tables on the side before this.
    question: str = ""
    # ch04_tool_failures: the toolbox itself is what varies, so one scenario
    # can declare a tool that another must not see. Empty means the chapter's
    # own DECLARED_TOOLS, which is what every chapter before ch04 used.
    tools: Sequence[BaseTool] = field(default_factory=tuple)
    # ch05_loop_endings: low enough and the reply comes back empty.
    max_tokens: int | None = None

    def attributes(self) -> dict[str, str | int | None]:
        """What the page shows about this scenario, in the order it reads.

        The name first, because it is what the scenario *is*; conditions after
        it, and only the ones this scenario set. A tool list is named rather
        than rendered, since the objects would be noise.
        """
        return {
            "name": self.name,
            "tools": ", ".join(tool.name for tool in self.tools) or None,
            "max_tokens": self.max_tokens,
        }


# What a chapter's own loop returns: how many turns it ran, how long the list
# got, and how it ended.
type Outcome = tuple[int, int, str]

# The one thing a chapter with no loop of its own still has to provide.
type ExecuteTool = Callable[[ToolCall, Scenario, ModelKind, Trace], ToolMessage]

type RunOne = Callable[[ModelKind, Scenario, Trace, int], Outcome]


def run_scenario(
    model_kind: ModelKind,
    scenario: Scenario,
    trace: Trace,
    turn_cap: int,
    system_prompt: str,
    execute_tool: ExecuteTool,
) -> Outcome:
    """One situation, run through `support/agent.py`'s loop.

    Bookkeeping, and it was identical in four chapters: assemble the two
    opening messages, close over the scenario so the model gets its tools and
    its canned replies, hand the chapter's own `execute_tool` to the loop.

    What is left in a chapter is `execute_tool`, because deciding what to
    catch and what the model is told is the lesson wherever there is one.
    """
    messages: list[BaseMessage] = [SystemMessage(system_prompt), HumanMessage(scenario.question)]

    def ask(turn: int, so_far: list[BaseMessage]) -> AIMessage:
        return ask_model(
            trace,
            model_kind,
            scenario.mock_model_replies[turn - 1 :],
            so_far,
            scenario.tools,
            scenario.max_tokens,
        )

    def execute(call: ToolCall) -> ToolMessage:
        return execute_tool(call, scenario, model_kind, trace)

    return run_turns(messages, trace, turn_cap, ask, execute)


def run_scenarios(
    chapter: str,
    model_kind: ModelKind,
    scenarios: Sequence[Scenario],
    run_one: RunOne,
    turn_cap: int,
) -> Trace:
    """Run a chapter's loop once per scenario, and record the run around it.

    This is bookkeeping, not agency. Opening a span, noting an ending, adding
    up turns and closing the trace teach nothing about agents, and were
    identical in every chapter that had scenarios -- so they live here once.

    What stays in the chapter is `run_one`: the loop, and every decision it
    makes. Traceability is not the agent's job, and this is the part that is
    not the agent.
    """
    trace = Trace(chapter=chapter, model_kind=model_kind)
    turns = messages = 0
    endings: set[str] = set()

    for scenario in scenarios:
        with trace.span("scenario", **scenario.attributes()) as span:
            ran, held, ended = run_one(model_kind, scenario, trace, turn_cap)
            # The ending belongs to the scenario that produced it, not to a run
            # summary that would otherwise carry one per scenario.
            span.add_note(ENDED, reason=ended)
        turns += ran
        messages += held
        endings.add(ended)

    # The distinct endings. Each scenario records its own, so repeating them
    # here would be the same fact twice.
    trace.close(turns=turns, messages=messages, ended="; ".join(sorted(endings)))
    return trace
