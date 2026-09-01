"""The reading order, as data.

Chapters are numbered in their filenames so `ls` answers "what comes next".
This list answers the same question for the tooling, and one test asserts that
nothing on disk is missing from it -- so a new chapter cannot be silently
orphaned.

Entries with no file yet are the intended ladder, not a promise. They are here
because the order is the argument: chapter 5 is a complete agentic loop in
twelve lines of plain Python, and every chapter after it exists to answer one
question -- what did this buy over chapter 5? A ladder kept only in someone's
head cannot be checked against the code, so it is kept here and the book
renders it as a table of contents.
"""

ORDER: list[tuple[str, str]] = [
    ("ch01_single_call", "one invoke; no tools, no loop, no framework"),
    ("ch02_tool_call", "the model asks; we execute; turn two knows"),
    ("ch03_missing_tool", "the model asks for what is not there; who decides?"),
    ("ch04_two_tools", "two calls in one reply -- still ONE turn"),
    ("ch05_the_loop", "the whole loop, twelve lines of plain Python"),
    # Admission, eviction, persistence. Three names for one problem: the list
    # is the state, it grows without bound, and there is a budget. Still plain
    # Python, so the framework answers later can be asked what they bought.
    ("ch06_skills", "a tool whose result is instructions, not data"),
    ("ch07_compression", "the list is too long; what do you drop, and what does it cost?"),
    ("ch08_memory", "what survives when the list is thrown away"),
    # From here the same behaviour, rebuilt on the framework. Every chapter
    # answers the one question chapter 5 earns the right to ask.
    ("ch09_graph", "the same behaviour as a StateGraph -- what did it buy?"),
    ("ch10_limits", "recursion_limit at the boundary"),
    ("ch11_checkpoint", "MemorySaver, thread_id, resume -- and why that is not ch08"),
    ("ch12_interrupt", "interrupt and Command(resume=...) as an approval gate"),
    ("ch13_subgraph", "a graph as a node -- a supervisor, from the ground up"),
    ("ch14_parallel", "fan-out, Send, join, and the order things merge in"),
]


def main() -> None:
    for module, summary in ORDER:
        print(f"{module:<24} {summary}")


if __name__ == "__main__":
    main()
