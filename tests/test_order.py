"""The reading order and the files on disk must agree.

The agreement is one-directional. Every chapter on disk must be listed, so a
new chapter cannot be silently orphaned. A listed chapter with no file is the
intended ladder -- the book renders it as "not written" rather than hiding it.
"""

from order import ORDER

from pathlib import Path

CHAPTERS = Path(__file__).parent.parent / "src" / "chapters"


def test_every_chapter_on_disk_appears_in_the_reading_order() -> None:
    on_disk = {p.stem for p in CHAPTERS.glob("ch*.py")}
    listed = {module for module, _ in ORDER}
    assert on_disk <= listed, f"not in the reading order: {sorted(on_disk - listed)}"


def test_the_reading_order_is_the_numeric_order() -> None:
    listed = [module for module, _ in ORDER]
    assert listed == sorted(listed)


def test_no_chapter_is_listed_twice() -> None:
    listed = [module for module, _ in ORDER]
    assert len(listed) == len(set(listed))


def test_every_chapter_has_a_summary() -> None:
    # The summary is what the book shows for a chapter that is not written
    # yet, so an empty one makes the table of contents useless exactly where
    # it is doing the most work.
    assert all(summary.strip() for _, summary in ORDER)
