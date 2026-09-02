# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""One invoke: no tools, no loop, no framework.

The smallest thing that is still an agent talking, and every later chapter is
this plus one more idea.

Read the trace for three things: what went to the model provider, what came back,
and who appended it to the history.

Five things a live run shows that a mock run cannot, all of them visible in
`out/ch01_single_call.live.json`, beside the mock run in `.mock.json`:

1. The request carries no session and no conversation id. Two messages, a
   model name, `stream: false`. The model provider remembers nothing between calls,
   which is why the whole history is sent every time.
2. The model reasons, and the reasoning arrives in `additional_kwargs`
   ["reasoning_content"], not in `content`. It is billed -- 23 of 55 output
   tokens on one run -- so a record that omits it under-accounts the call.
3. The message id is LangChain's invention (`lc_run--...`), not the
   model provider's (`a1a4aaa9-...`). This matters later: LangGraph's `add_messages`
   reduces by id, so the identity the state machine dedupes on has no meaning
   to the model provider at all.
4. `finish_reason` is the model provider's own account of why it stopped. This loop
   infers termination from empty `tool_calls` instead. They agree here; they
   will not always, because `length` and `content_filter` also produce a reply
   with no tool calls.
5. There is a `refusal` field. It is null here.
"""

from support.cli import main
from support.models import build_model
from support.trace import ModelKind, Trace

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# The system prompt is just a message. It has no special status in the
# protocol -- it is first in the list, and that is the whole of it.
SYSTEM_PROMPT = "You answer briefly and plainly."

# `user` rather than `human` because that is the word on the wire. LangChain
# calls the class `HumanMessage` and the trace prints `human`; the request body
# says `"role": "user"`. The name here agrees with the model provider.
USER_PROMPT = (
    "In one sentence, and per The Hitchhiker's Guide to the Galaxy: what is the "
    "Answer to the Ultimate Question of Life, the Universe, and Everything?"
)

# What the mock model replies. A live run ignores it and answers for itself;
# everything else about the chapter is identical either way. Named for what it
# holds, like the two prompts above -- never for the mock that consumes it.
MOCK_MODEL_REPLIES = [
    AIMessage("Forty-two, though the question is generally held to be the harder half.")
]


def run(model_kind: ModelKind = "mock") -> Trace:
    # The trace is ours, not the framework's. Nothing below records itself.
    trace = Trace(chapter="ch01_single_call", model_kind=model_kind)

    # The context. A plain list we assembled -- no store, no retrieval, no
    # selection. What is in this list is what the model will see.
    messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(USER_PROMPT)]

    # A turn is one model invocation plus any tools it asks for. This one asks
    # for none, so the turn is the invocation alone.
    with trace.span("turn", number=1):
        with trace.span("model", model_kind=model_kind) as span:
            # The exact list going to the model provider, recorded before it goes.
            span.add_context(messages)

            # The only network call in this chapter. It is stateless: the
            # model provider remembers nothing between calls, which is why
            # the whole list is sent every time.
            # The same `span` records both layers. On a live run it carries the
            # raw HTTP either side of the library as well as the normalized
            # reply -- what DeepSeek actually received, and what the library
            # made of what came back.
            reply = build_model(model_kind, MOCK_MODEL_REPLIES, span).invoke(messages)

            # What came back -- one message, and crucially no tool calls.
            span.add_reply(reply)

        # invoke() is typed as returning BaseMessage. Narrowing it is for the
        # type checker, not for correctness -- worth seeing rather than hiding
        # in a cast.
        assert isinstance(reply, AIMessage)

        # We append it. The model did not update a conversation; there is no
        # conversation object. The caller owns the history, always.
        messages.append(reply)

    # Why the program stops here, read off the reply rather than asserted: the
    # model asked for nothing further. That single condition is the entire
    # agentic loop, absent the loop -- and this chapter has no loop to run, so
    # a reply that did ask for something would have nowhere to go.
    ended = "no tool_calls" if not reply.tool_calls else "nowhere to put a tool call"
    trace.close(turns=1, messages=len(messages), ended=ended)
    return trace


if __name__ == "__main__":
    main(run)
