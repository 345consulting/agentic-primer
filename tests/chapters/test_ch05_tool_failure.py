# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 4 asserts the decision, and the line between what we keep and what we send.

The mechanics are chapter 2's. What is new is that a tool raises, that we
choose to report rather than stop, and that two different texts come out of
one failure: the exception for the trace, a sentence for the model.
"""

from chapters import ch05_tool_failure

import pytest


def test_the_tool_really_raises() -> None:
    # Not a stub returning an error string -- the chapter is about catching.
    with pytest.raises(ch05_tool_failure.PriceServiceError):
        ch05_tool_failure.price_of.invoke({"item": "milk"})


def test_the_failure_is_recorded_as_a_decision_we_took() -> None:
    trace = ch05_tool_failure.run()
    (tool,) = trace.find_spans("tool")
    failed = tool.require("failed").payload
    assert failed["exception"] == "PriceServiceError"
    assert "no retry" in failed["decision"]
    assert tool.find("result") is None


def test_the_exception_text_stays_in_the_trace_and_never_reaches_the_model() -> None:
    """The line this chapter draws.

    An exception is written for a developer reading a stack trace and can carry
    a host, a path or a credential. LangGraph's default template formats it
    with `repr` and sends that to the model provider.
    """
    trace = ch05_tool_failure.run()
    (tool,) = trace.find_spans("tool")
    recorded = tool.require("failed").payload["message"]
    assert "pricing service unreachable" in recorded

    sent_onward = trace.find_spans("model")[1].context[-1]["content"]
    assert recorded not in sent_onward
    assert "price_of" in sent_onward


def test_the_message_to_the_model_says_what_to_do_and_not_to_retry() -> None:
    # The error text is a prompt. LangGraph's default says "Please fix your
    # mistakes"; this one says the opposite, on purpose.
    trace = ch05_tool_failure.run()
    told = trace.find_spans("model")[1].context[-1]
    assert told["role"] == "tool"
    assert "will not succeed on a retry" in told["content"]
    assert "Answer without it" in told["content"]


def test_the_run_answers_without_the_tool_and_stops() -> None:
    trace = ch05_tool_failure.run()
    answer = trace.find_spans("model")[1].reply
    assert not answer.get("tool_calls")
    assert "cannot get the price" in answer["content"]
    assert trace.summary["ended"] == "no tool_calls"


def test_every_span_that_was_entered_was_exited_even_though_one_raised() -> None:
    # The tool span is opened, the call raises inside it, and the span still
    # closes -- because the `with` block owns the exit, not the happy path.
    trace = ch05_tool_failure.run()
    assert all(span.exited_at is not None for _, span in trace.walk())
