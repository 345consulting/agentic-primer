"""The reading order, as data.

Chapters are numbered in their filenames so `ls` answers "what comes next".
This list answers the same question for the tooling, and one test asserts the
two agree -- so a new chapter cannot be silently orphaned.
"""

ORDER: list[tuple[str, str]] = [
    ("ch01_single_call", "one invoke; no tools, no loop, no framework"),
    ("ch02_tool_call", "the model asks; we execute; turn two knows"),
]


def main() -> None:
    for module, summary in ORDER:
        print(f"{module:<24} {summary}")


if __name__ == "__main__":
    main()
