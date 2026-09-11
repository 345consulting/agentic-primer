# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 14 asserts a reply reassembled from pieces, not returned whole."""

from chapters import ch14_stream as chapter
from support.models import MockModel
from support.trace import Span

from langchain_core.messages import AIMessage, HumanMessage


def stream_span() -> Span:
    trace = chapter.run()
    (scenario,) = trace.find_spans("scenario")
    (model,) = [c for c in scenario.children if c.name == "model"]
    (stream,) = [c for c in model.children if c.name == "stream"]
    return stream


def test_the_reply_arrives_as_more_than_one_chunk() -> None:
    span = stream_span()
    chunks = [note for note in span.notes if note.label == "chunk"]
    assert len(chunks) > 1


def test_the_accumulated_reply_matches_what_invoke_would_have_returned() -> None:
    trace = chapter.run()
    (scenario,) = trace.find_spans("scenario")
    (model,) = [c for c in scenario.children if c.name == "model"]
    reply = model.reply
    assert reply["content"] == chapter.MOCK_REPLY.content


def test_the_reply_note_carries_the_ordinary_ai_role_not_the_chunk_class() -> None:
    # AIMessageChunk.type is "AIMessageChunk", not "ai" -- the accumulated
    # result must be converted back, or the page's role column breaks.
    trace = chapter.run()
    (scenario,) = trace.find_spans("scenario")
    (model,) = [c for c in scenario.children if c.name == "model"]
    assert model.reply["role"] == "ai"


def test_chunks_are_notes_not_spans() -> None:
    # A chunk has no duration and nothing nested inside it -- it belongs on
    # Span.add_note, the same shape WIRE_REQUEST/WIRE_RESPONSE already use.
    span = stream_span()
    assert span.children == []


def test_first_and_last_token_ms_are_a_real_split_not_one_number() -> None:
    span = stream_span()
    streamed = span.require("streamed").payload
    assert streamed["first_token_ms"] <= streamed["last_token_ms"]
    assert streamed["chunks"] > 1


def test_text_is_chunked_by_word_and_carries_no_tool_calls() -> None:
    # This chapter's shape: content arrives in pieces, each one useful on its
    # own. A call's arguments are the opposite, and are ch15_stream_tools'.
    model = MockModel(replies=[AIMessage("Milk is 1.20.")])
    chunks = list(model.stream([HumanMessage("hi")]))
    assert len(chunks) > 1
    assert not any(chunk.tool_call_chunks for chunk in chunks)
    assert "".join(str(chunk.content) for chunk in chunks) == "Milk is 1.20."
