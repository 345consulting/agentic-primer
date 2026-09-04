# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Chapter 22 asserts a classification, checked against three requests that
are strict supersets of each other -- and that the recorder itself now
carries the tool schema it used to drop.
"""

from chapters import ch22_prompt_types as chapter
from support.trace import Span


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def buckets(span: Span) -> list[str]:
    (note,) = [n for n in span.notes if n.label == "classifications"]
    return [f["bucket"] for f in note.payload["found"]]


def test_classify_labels_a_system_and_human_message_per_conversation() -> None:
    messages = [{"role": "system", "content": "x"}, {"role": "human", "content": "y"}]
    found = chapter.classify(messages, [])
    assert [f["bucket"] for f in found] == ["per_conversation", "per_conversation"]


def test_classify_labels_a_tool_declaration_design_time() -> None:
    found = chapter.classify([], [{"name": "price_of", "description": "d", "schema": {}}])
    assert found == [
        {
            "what": "tool description: price_of",
            "bucket": "design_time",
            "reviewed": (
                "once, when the function was written -- or never, if a dependency wrote it"
            ),
        }
    ]


def test_classify_labels_a_tool_result_and_a_prior_reply_during_run() -> None:
    messages = [{"role": "ai", "content": ""}, {"role": "tool", "content": "1.20"}]
    found = chapter.classify(messages, [])
    assert [f["bucket"] for f in found] == ["during_run", "during_run"]


def test_the_first_scenario_has_no_tools_and_no_design_time_content() -> None:
    assert buckets(scenario_span("authored_per_conversation")) == [
        "per_conversation",
        "per_conversation",
    ]


def test_the_second_scenario_adds_design_time_but_not_during_run() -> None:
    found = buckets(scenario_span("authored_at_design_time"))
    assert "design_time" in found
    assert "during_run" not in found


def test_the_third_scenario_has_all_three_buckets() -> None:
    found = set(buckets(scenario_span("authored_during_the_run")))
    assert found == {"per_conversation", "design_time", "during_run"}


def test_each_scenario_is_a_strict_superset_of_the_one_before() -> None:
    first = set(buckets(scenario_span("authored_per_conversation")))
    second = set(buckets(scenario_span("authored_at_design_time")))
    third = set(buckets(scenario_span("authored_during_the_run")))
    assert first < second < third


def test_the_recorded_schema_includes_the_enum_not_just_the_ref() -> None:
    # The bug this chapter's own fix guards against: `.args` alone would
    # have shown a `$ref` and dropped the `$defs` block the enum lives in.
    trace = chapter.run()
    (model,) = [trace.find_spans("model")[-1]]
    ctx = model.require("context").payload
    (declared,) = ctx["tools"]
    assert "enum" in str(declared["schema"])
    assert "milk" in str(declared["schema"])
