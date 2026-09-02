# Agentic Primer

A 345c-owned learning project. The mechanics of an agentic harness, built from
the ground up, one chapter at a time — no framework abstractions hiding the
loop, and a trace of every entry and exit so the decision points are readable
rather than inferred.

Not a product, not a client project, not HiQ. It exists to be read.

## Build

Run every recipe from this directory.

```
just list                      the reading order
just run ch01_single_call      mock run; writes out/<chapter>.html
just run-live ch01_single_call the same chapter against a real provider
just gate                      ruff + ruff format --check + mypy --strict + pytest
```

`just gate` is the quality gate: zero violations, zero errors, zero findings.
Teaching code is gated like any other code — an untyped example teaches the
wrong habit.

## Layout

```
src/
  order.py       the reading order, as data
  support/       scaffolding — models, trace, view. Never a lesson.
  chapters/      the lessons, numbered
```

Three rules keep the scaffolding honest:

- **Nothing in `support/` may import from `chapters/`.**
- **Every mechanism a chapter imports from `support/` was built by hand in an
  earlier chapter first.** Teach it inline, then graduate it. A chapter that
  imports a mechanism no earlier chapter explained means there is a hidden
  mechanism — the thing this primer exists to eliminate.
- **Recording is scaffolding, even when it is interesting.** `Trace`, `view`,
  and the httpx wire hooks observe; they never participate. The second rule
  constrains mechanisms a chapter uses to work, not the instruments watching it.

Comment density tapers. Chapter 1 explains everything because nothing is
established; later chapters comment only what is new. The comments are the
delta from the previous chapter.

## The ladder

The chapters on disk are the reading order — they are numbered, so `ls` and
`just list` answer "what comes next", and each one's first docstring line is
its summary. There is no list of chapters in code, because a second copy of
either fact is a second thing to keep true.

What is not written yet is a plan, and a plan is prose:

```
ch01 single_call      one invoke; no tools, no loop, no framework      written
ch02 tool_call        the model asks; we execute; turn two knows       written
ch03 missing_tool     the model asks for what is not there; who decides?
ch04 two_tools        two calls in one reply -- still ONE turn
ch05 the_loop         the whole loop, twelve lines of plain Python
ch06 skills           a tool whose result is instructions, not data
ch07 compression      the list is too long; what do you drop, and what does it cost?
ch08 memory           what survives when the list is thrown away
ch09 graph            the same behaviour as a StateGraph -- what did it buy?
ch10 limits           recursion_limit at the boundary
ch11 checkpoint       MemorySaver, thread_id, resume -- and why that is not ch08
ch12 interrupt        interrupt and Command(resume=...) as an approval gate
ch13 subgraph         a graph as a node -- a supervisor, from the ground up
ch14 parallel         fan-out, Send, join, and the order things merge in
```

Chapter 5 is a complete agentic loop in twelve lines of plain Python. Every
chapter after it answers one question: *what did this buy over chapter 5?*
Chapters 6 to 8 are admission, eviction and persistence — one problem, which
is that the list is the state and the budget is finite — and they stay in
plain Python so the framework's answers from ch09 on can be asked what they
bought.

## Documented deviations from the code standard

Per the convention-conflict rule in `~/claude-and-i/docs/standards/code-style.md`,
a deviation is only valid if it is written down. These two are.

**Function length — "under 40 lines" is measured in code, not in lines.**
`_describe`, `_generate`, `book` and chapter 1's `run` exceed 40 lines as
files; none exceeds 20 lines of code. The remainder is comments, and in this
repo the comments are the lesson — a chapter that fits the limit by deleting
its explanation has lost the only thing it was for. The limit still binds on
code: if any of these needs scrolling to follow the *logic*, it gets split.

**"No undocumented returns" does not apply to three-line functions.**
The standard asks every return path to carry a comment saying why. Here most
functions are a guard and a return, under a docstring that already says what
comes back — a per-return comment would restate the signature, which the
"comment the why, never the what" rule forbids. The rule is honoured where it
earns its keep: any function with more than one return path explains why each
one is taken.

**Chapters are ordered by the reading, `support/` by the standard.**
Modules under `support/` follow "statics on top, public API, then private
helpers, alphabetical within each". Chapters cannot: `TOOLS` is built from the
tool function and `DECLARED` from `TOOLS`, so statics and definitions
necessarily interleave. Past that constraint a chapter is read top to bottom —
prompts, tools, mock replies, then the functions in the order they run — and
alphabetising it would destroy the sequence the file exists to teach.

## Findings

`FINDINGS.md` — things learned by running the chapters that outlive the chapter
that produced them. Appendable, dated. Chapter-specific observations live in
that chapter's docstring instead.

## Model

`deepseek-v4-flash`, chosen for price: cheap enough that running a chapter live
is routine rather than an event. Live runs need `DEEPSEEK_API_KEY` in the
environment and there is no fallback — absent the key, the run fails.

Per the entity rules this is a `345c.*` secret and belongs in GCP Secret
Manager, not in HiQ's SSM. **Currently it is neither** — it lives in a mode-600
file in /tmp, which is a stopgap, not a home.

## Gotchas

- **The mock and the live model do not prove the same things.** A scripted
  model lets a test assert exact mechanics; a live one can only be asserted
  against invariants that hold for any model. Anything provable only under mock
  was the mock's cooperation, not the framework's guarantee — and the harness
  must not assume it.
- **LangChain normalizes, and the normalization is lossy in both directions.**
  The message id you get is LangChain's invention, not the provider's, which
  matters because LangGraph's `add_messages` reduces by id.
- **The wire hooks record request header values by allowlist, not by
  denylist.** A denylist has to be right about every auth header that will
  ever exist; an allowlist that is wrong records nothing, and a trace is a
  file on disk. Response values are recorded less a credential deny-set —
  that side guards the provider's secrets, not ours.
