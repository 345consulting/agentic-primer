# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 7 asserts what a tool brings back when it leaves the process.

Status codes with meanings, latency in the tool span, a second credential --
and the credential arriving home in a place the recorder does not guard.
"""

from chapters import ch07_tool_http as chapter
from support.service import SERVICE_KEY

import pytest


def test_every_failure_class_has_a_situation() -> None:
    # These are the specimens ch08_retry_policy classifies. Two transient, one
    # permanent, and one that is transient by nature and unsafe by consequence.
    assert [scenario.name for scenario in chapter.SCENARIOS] == [
        "service_answers",
        "service_rate_limits",
        "service_is_broken",
        "service_says_no",
        "service_is_slow",
    ]


def test_the_tool_records_its_own_wire() -> None:
    trace = chapter.run()
    (answered, *_) = trace.find_spans("scenario")
    (tool,) = [span for turn in answered.children for span in turn.children if span.name == "tool"]
    assert tool.find("wire request") is not None
    assert tool.require("wire response").payload["status"] == 200


def test_the_recorder_refuses_the_credential_on_the_way_out() -> None:
    trace = chapter.run()
    (answered, *_) = trace.find_spans("scenario")
    (tool,) = [span for turn in answered.children for span in turn.children if span.name == "tool"]
    headers = tool.require("wire request").payload["headers"]
    assert "authorization" in headers
    assert headers["authorization"] is None


def test_and_the_credential_comes_home_in_the_response_body() -> None:
    """The chapter. One boundary is guarded and the other is not.

    The allowlist protects request headers. A response body is recorded whole,
    and this service echoes what it was sent -- so the value withheld at one
    end arrives at the other, and every check we have still passes.
    """
    trace = chapter.run()
    (answered, *_) = trace.find_spans("scenario")
    (tool,) = [span for turn in answered.children for span in turn.children if span.name == "tool"]
    echoed = tool.require("wire response").payload["body"]["headers"]["Authorization"]
    assert SERVICE_KEY in echoed


def test_a_failing_service_is_reported_with_what_it_said() -> None:
    trace = chapter.run()
    _, rate_limited, broken, refused, slow = trace.find_spans("scenario")
    said = {
        scenario.attributes["name"]: next(
            span.require("failed").payload["message"]
            for turn in scenario.children
            for span in turn.children
            if span.name == "tool"
        )
        for scenario in (rate_limited, broken, refused, slow)
    }
    assert "429" in said["service_rate_limits"]
    assert "500" in said["service_is_broken"]
    assert "404" in said["service_says_no"]
    assert "did not answer in time" in said["service_is_slow"]


def test_nothing_retries() -> None:
    # Every situation takes exactly two turns: the call, then an answer or an
    # apology. Classification arrives in ch08, the attempt in ch09.
    trace = chapter.run()
    assert [len(scenario.children) for scenario in trace.find_spans("scenario")] == [2] * 5


def test_the_declared_tool_is_not_the_dispatched_one() -> None:
    # Deliberate here, unlike ch04_tool_failures where it was the failure: the
    # model is told about price_of, and fetch_price does the work, because the
    # work needs a span and a situation that a tool call cannot carry.
    assert [tool.name for tool in chapter.DECLARED_TOOLS] == ["price_of"]
    assert chapter.TOOLS["price_of"] is chapter.fetch_price
    with pytest.raises(AssertionError):
        chapter.price_of.invoke({"item": "milk"})
