# Copyright (c) 2022-2026 345 Consulting, LLC
# Licensed under the MIT License. See LICENSE in the project root.

"""The chapters on disk: what they are called, what they say, and their hash.

Scaffolding, never a lesson. There is no list of chapters anywhere -- the
files are the list. They are numbered, so sorting the filenames is the reading
order, and each one already describes itself in its first docstring line. A
second copy of either fact is a second thing to keep true.

Read, never imported. Nothing in `support/` may import from `chapters/`, and
`ast` gets the docstring without running the module.
"""

import ast
import hashlib
from pathlib import Path

CHAPTERS_DIR = Path(__file__).resolve().parent.parent / "chapters"

# A trace is a record of a past run, not of the current code. Two columns can
# look coherent while describing different versions of a chapter, which is the
# same class of lie as a recorder that omits a field. The stamp is what lets
# the page notice.
UNKNOWN_SOURCE = "unknown"


def chapters() -> list[tuple[str, str]]:
    """Every chapter, in reading order, with the first line of its docstring."""
    return [(path.stem, summary_of(path.stem)) for path in sorted(CHAPTERS_DIR.glob("ch*.py"))]


def source_hash(chapter: str) -> str:
    """The chapter's source, as twelve hex characters.

    A chapter with no file is stamped `unknown` rather than raising, because a
    test may build a trace for a chapter that was never written.
    """
    path = CHAPTERS_DIR / f"{chapter}.py"
    if not path.is_file():
        return UNKNOWN_SOURCE
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def summary_of(chapter: str) -> str:
    """A chapter's one-line summary: the first line of its own docstring.

    The chapter describes itself, in the file it describes. Anywhere else is a
    second description to keep in step with the first.
    """
    path = CHAPTERS_DIR / f"{chapter}.py"
    if not path.is_file():
        return ""
    docstring = ast.get_docstring(ast.parse(path.read_text())) or ""
    return docstring.splitlines()[0].strip() if docstring else ""


def main() -> None:
    for chapter, summary in chapters():
        print(f"{chapter:<24} {summary}")


if __name__ == "__main__":
    main()
