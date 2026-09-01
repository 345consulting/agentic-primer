"""Chapter 1 asserts the smallest set of facts: one call, and why it stopped."""

from chapters import ch01_single_call


def test_one_invocation_is_one_turn() -> None:
    trace = ch01_single_call.run()
    turns = [span for _, span in trace.walk() if span.name == "turn"]
    assert len(turns) == 1
    assert trace.summary["turns"] == 1


def test_the_model_call_happens_inside_the_turn() -> None:
    trace = ch01_single_call.run()
    depths = {span.name: depth for depth, span in trace.walk()}
    assert depths["turn"] == 0
    assert depths["model"] == 1


def test_the_context_sent_is_exactly_what_we_assembled() -> None:
    trace = ch01_single_call.run()
    (model_span,) = [span for _, span in trace.walk() if span.name == "model"]
    (context,) = [note for note in model_span.notes if note.label == "context"]
    assert [m["role"] for m in context.payload["messages"]] == ["system", "human"]


def test_the_loop_would_end_because_the_reply_asks_for_no_tools() -> None:
    trace = ch01_single_call.run()
    (model_span,) = [span for _, span in trace.walk() if span.name == "model"]
    (reply,) = [note for note in model_span.notes if note.label == "reply"]
    assert "tool_calls" not in reply.payload["message"]
    assert trace.summary["ended"] == "no tool_calls"


def test_the_caller_appended_the_reply_to_the_history() -> None:
    # Two messages went in, three exist after. The provider appended nothing.
    trace = ch01_single_call.run()
    (model_span,) = [span for _, span in trace.walk() if span.name == "model"]
    (context,) = [note for note in model_span.notes if note.label == "context"]
    assert len(context.payload["messages"]) == 2
    assert trace.summary["messages"] == 3


def test_every_span_that_was_entered_was_exited() -> None:
    trace = ch01_single_call.run()
    assert all(span.exited_at is not None for _, span in trace.walk())
