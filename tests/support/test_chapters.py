# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The files are the list. These assert that nothing else has to be."""

from support.chapters import CHAPTERS_DIR, UNKNOWN_SOURCE, chapters, source_hash, summary_of


def test_the_chapters_are_the_files_in_numeric_order() -> None:
    found = [name for name, _ in chapters()]
    assert found == sorted(found)
    assert found == sorted(path.stem for path in CHAPTERS_DIR.glob("ch*.py"))


def test_a_chapter_describes_itself_in_its_own_first_docstring_line() -> None:
    """The one place a chapter is described is the chapter itself.

    A second copy is a second thing to keep true, and it goes stale silently.
    This asserts the summary is usable as a table-of-contents entry -- present,
    one line, and saying something the filename does not already say.
    """
    for name, summary in chapters():
        assert summary, name
        assert "\n" not in summary, name
        assert summary.lower() != name.lower(), name
        assert len(summary.split()) >= 4, name


def test_a_chapter_that_does_not_exist_has_no_summary_and_no_hash() -> None:
    assert summary_of("ch99_nonexistent") == ""
    assert source_hash("ch99_nonexistent") == UNKNOWN_SOURCE
