"""Chapter 2 asserts the round trip: the model asks, we execute, turn two knows.

The facts here that chapter 1 could not have: a turn that contains work, a
result carried back as a message, and an id tying the two together.
"""

from chapters import ch02_tool_call
from support.trace import Json, Span, Trace


def test_a_turn_is_one_invocation_plus_the_tools_it_asked_for() -> None:
    trace = ch02_tool_call.run()
    turns = [span for _, span in trace.walk() if span.name == "turn"]
    assert len(turns) == 2
    # The tool ran inside turn one, not between the turns. Executing a tool is
    # the consequence of an invocation, not a turn of its own.
    assert [child.name for child in turns[0].children] == ["model", "tool"]
    assert [child.name for child in turns[1].children] == ["model"]


def test_the_first_reply_is_a_request_and_carries_no_answer() -> None:
    trace = ch02_tool_call.run()
    first = _reply(trace, turn=0)
    assert first["content"] == ""
    assert [call["name"] for call in first["tool_calls"]] == ["stock_on_hand"]


def test_we_dispatched_the_tool_ourselves() -> None:
    trace = ch02_tool_call.run()
    (tool,) = [span for _, span in trace.walk() if span.name == "tool"]
    assert tool.attrs["name"] == "stock_on_hand"
    assert dict(tool.notes[0].payload) == {"part": "flange"}
    assert tool.notes[1].payload["value"] == 17


def test_the_result_re_enters_the_context_as_a_message() -> None:
    # The provider is stateless. The only way turn two learns what the tool
    # said is that the result is in the list turn two sends.
    trace = ch02_tool_call.run()
    sent = _context(trace, turn=1)
    assert [m["role"] for m in sent] == ["system", "human", "ai", "tool"]
    assert sent[-1]["content"] == "17"


def test_the_id_ties_the_result_to_the_request_that_asked_for_it() -> None:
    trace = ch02_tool_call.run()
    asked = _reply(trace, turn=0)["tool_calls"][0]["id"]
    answered = _context(trace, turn=1)[-1]["tool_call_id"]
    assert answered == asked


def test_turn_two_asks_for_nothing_further_and_that_is_why_it_ends() -> None:
    trace = ch02_tool_call.run()
    assert "tool_calls" not in _reply(trace, turn=1)
    assert trace.summary["ended"] == "no tool_calls"


def test_every_span_that_was_entered_was_exited() -> None:
    trace = ch02_tool_call.run()
    assert all(span.exited_at is not None for _, span in trace.walk())


def _model_spans(trace: Trace) -> list[Span]:
    return [span for _, span in trace.walk() if span.name == "model"]


def _reply(trace: Trace, turn: int) -> Json:
    (note,) = [n for n in _model_spans(trace)[turn].notes if n.label == "reply"]
    message: Json = note.payload["message"]
    return message


def _context(trace: Trace, turn: int) -> list[Json]:
    (note,) = [n for n in _model_spans(trace)[turn].notes if n.label == "context"]
    messages: list[Json] = note.payload["messages"]
    return messages


def test_the_reason_it_stopped_is_read_off_the_reply_not_asserted() -> None:
    # The chapter has exactly two turns because two are written out. Saying
    # "no tool_calls" without looking would be true today and a lie the first
    # time a model asks for a second tool.
    trace = ch02_tool_call.run()
    assert trace.summary["ended"] == "no tool_calls"
    assert "tool_calls" not in _reply(trace, turn=1)
