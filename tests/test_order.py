"""The reading order and the files on disk must agree."""

from order import ORDER

from pathlib import Path

CHAPTERS = Path(__file__).parent.parent / "src" / "chapters"


def test_every_chapter_on_disk_appears_in_the_reading_order() -> None:
    on_disk = {p.stem for p in CHAPTERS.glob("ch*.py")}
    listed = {module for module, _ in ORDER}
    assert on_disk == listed


def test_the_reading_order_is_the_numeric_order() -> None:
    listed = [module for module, _ in ORDER]
    assert listed == sorted(listed)
