# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 14 asserts a reply reassembled from pieces, not returned whole."""

from chapters import ch14_stream as chapter
from support.models import MockModel
from support.trace import Span

import pytest
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


def test_the_mock_model_chunks_a_reply_it_was_not_asked_to() -> None:
    # MockModel.stream() refuses tool calls -- ch15_stream_tools chunks
    # those, a different shape from splitting plain text.
    tool_call_reply = AIMessage(
        "", tool_calls=[{"name": "price_of", "args": {"item": "milk"}, "id": "c1"}]
    )
    model = MockModel(replies=[tool_call_reply])
    with pytest.raises(AssertionError):
        list(model.stream([HumanMessage("hi")]))
