"""Chapter 1 -- a single call.

One model invocation. No tools, no loop, no framework. This is the smallest
thing that is still an agent talking, and every later chapter is this plus
one more idea.

Read the trace for three things: what went to the provider, what came back,
and who appended it to the history.

Five things a live run shows that a mock run cannot, all of them visible in
`out/ch01_single_call.json`:

1. The request carries no session and no conversation id. Two messages, a
   model name, `stream: false`. The provider remembers nothing between calls,
   which is why the whole history is sent every time.
2. The model reasons, and the reasoning arrives in `additional_kwargs`
   ["reasoning_content"], not in `content`. It is billed -- 23 of 55 output
   tokens on one run -- so a record that omits it under-accounts the call.
3. The message id is LangChain's invention (`lc_run--...`), not the
   provider's (`a1a4aaa9-...`). This matters later: LangGraph's `add_messages`
   reduces by id, so the identity the state machine dedupes on has no meaning
   to the provider at all.
4. `finish_reason` is the provider's own account of why it stopped. This loop
   infers termination from empty `tool_calls` instead. They agree here; they
   will not always, because `length` and `content_filter` also produce a reply
   with no tool calls.
5. There is a `refusal` field. It is null here.
"""

from support.models import Kind, model
from support.trace import Trace

import sys

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

# The system prompt is just a message. It has no special status in the
# protocol -- it is first in the list, and that is the whole of it.
SYSTEM = "You answer briefly and plainly."

QUESTION = (
    "In one sentence, and per The Hitchhiker's Guide to the Galaxy: what is the "
    "Answer to the Ultimate Question of Life, the Universe, and Everything?"
)

# What the scripted model will say. A live run ignores this and answers for
# itself; everything else about the chapter is identical either way.
SCRIPT = [AIMessage("Forty-two, though the question is generally held to be the harder half.")]


def run(kind: Kind = "mock") -> Trace:
    # The trace is ours, not the framework's. Nothing below records itself.
    trace = Trace(chapter="ch01_single_call", kind=kind)

    # The context. A plain list we assembled -- no store, no retrieval, no
    # selection. What is in this list is what the model will see.
    messages: list[BaseMessage] = [SystemMessage(SYSTEM), HumanMessage(QUESTION)]

    # A turn is one model invocation plus any tools it asks for. This one asks
    # for none, so the turn is the invocation alone.
    with trace.span("turn", number=1):
        with trace.span("model", kind=kind) as span:
            # The exact list going to the provider, recorded before it goes.
            span.context(messages)

            # The only network call in this chapter. It is stateless: the
            # provider remembers nothing between calls, which is why the whole
            # list is sent every time.
            # `span` is handed in twice over: once as the thing recording the
            # normalized call, and once as the wire recorder. On a live run the
            # same span then carries both layers -- what DeepSeek received, and
            # what the library made of what came back.
            reply = model(kind, SCRIPT, wire=span).invoke(messages)

            # What came back -- one message, and crucially no tool calls.
            span.reply(reply)

        # We append it. The model did not update a conversation; there is no
        # conversation object. The caller owns the history, always.
        messages.append(reply)

    # invoke() is typed as returning BaseMessage. Narrowing it is for the type
    # checker, not for correctness -- worth seeing rather than hiding in a cast.
    assert isinstance(reply, AIMessage)

    # Why the program stops here: the model asked for nothing further. That
    # single condition is the entire agentic loop, absent the loop.
    trace.close(turns=1, messages=len(messages), ended="no tool_calls")
    return trace


if __name__ == "__main__":
    kind: Kind = "live" if "live" in sys.argv[1:] else "mock"
    run(kind).report()
