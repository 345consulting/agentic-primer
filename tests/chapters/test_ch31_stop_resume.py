# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""Chapter 31 asserts a genuinely separate process can resume a stopped
run from nothing but a jsonl snapshot on disk, and that the two ways a
naive version breaks -- wrong timing, wrong recovery point -- do too.
"""

from chapters import ch31_stop_resume as chapter
from support.trace import Span

from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


def scenario_span(name: str) -> Span:
    trace = chapter.run()
    (span,) = [s for s in trace.find_spans("scenario") if s.attributes["name"] == name]
    return span


def test_a_new_process_resumes_from_the_last_snapshot() -> None:
    span = scenario_span("a_snapshot_after_every_turn_lets_a_new_process_resume")
    note = span.require("resumed_in_a_new_process")
    assert note.payload["resumed_from_turn"] == 2
    assert "XYZ001" in note.payload["final_reply"]


def test_the_checkpoint_file_actually_exists_on_disk() -> None:
    span = scenario_span("a_snapshot_after_every_turn_lets_a_new_process_resume")
    note = span.require("snapshot_written")
    assert Path(note.payload["path"]).exists()


def test_mid_turn_snapshot_check_is_skipped_under_mock() -> None:
    span = scenario_span("snapshotting_mid_turn_would_orphan_a_tool_call")
    note = span.require("resume_check")
    assert note.payload["checked"] is False


def test_a_crash_before_the_snapshot_recovers_the_prior_complete_turn() -> None:
    span = scenario_span("a_crash_before_the_next_snapshot_loses_at_most_one_turn")
    note = span.require("recovery_point")
    assert note.payload["recovered_turn"] == 2
    assert note.payload["turn_three_was_lost"] is True


def test_turn_three_is_redone_cleanly_after_recovery() -> None:
    span = scenario_span("a_crash_before_the_next_snapshot_loses_at_most_one_turn")
    note = span.require("turn_three_redone_cleanly")
    assert Path(note.payload["path"]).exists()


def test_two_run_ids_never_see_each_others_history() -> None:
    span = scenario_span("separate_run_ids_do_not_collide")
    note = span.require("isolation_check")
    assert note.payload["run_a_mentions_a100"] is True
    assert note.payload["run_a_mentions_b200"] is False
    assert note.payload["run_b_mentions_b200"] is True
    assert note.payload["run_b_mentions_a100"] is False


def test_serialize_then_deserialize_round_trips_a_tool_call() -> None:
    original = AIMessage(
        "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
    )
    restored = chapter.deserialize_message(chapter.serialize_message(original))
    assert isinstance(restored, AIMessage)
    assert restored.tool_calls[0]["name"] == "check_order"


def test_serialize_then_deserialize_round_trips_a_tool_reply() -> None:
    original = ToolMessage(content="order A100: shipped", tool_call_id="c1")
    restored = chapter.deserialize_message(chapter.serialize_message(original))
    assert isinstance(restored, ToolMessage)
    assert restored.tool_call_id == "c1"


def test_deserialize_rejects_an_unknown_role() -> None:
    with pytest.raises(ValueError, match="unknown role"):
        chapter.deserialize_message({"role": "narrator", "content": "once upon a time"})


def test_read_last_snapshot_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert chapter.read_last_snapshot(tmp_path, "no-such-run") is None


def test_write_snapshot_appends_rather_than_overwrites(tmp_path: Path) -> None:
    messages: list[BaseMessage] = [SystemMessage("x"), HumanMessage("y")]
    chapter.write_snapshot(tmp_path, "run", 1, messages)
    chapter.write_snapshot(tmp_path, "run", 2, [*messages, AIMessage("z")])
    lines = (tmp_path / "run.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    last = chapter.read_last_snapshot(tmp_path, "run")
    assert last is not None
    assert last["turn"] == 2
