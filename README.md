# The agentic primer — the mechanics of an agentic harness

> An agent is a loop that sends the whole context to a model, executes whatever
> the model requests, appends the results to that same context, and repeats —
> until the model asks for nothing further. The harness owns the list and every
> decision about it; the model only ever gets a vote.

That is the whole subject. Chapter 5 is that paragraph in twelve lines of plain
Python, chapters 1 to 4 build up to it one idea at a time, and every chapter
after it answers a single question: *what did this buy over chapter 5?*

Two words in that definition do the work. **Requests** — the model never
executes anything and never decides anything; it emits a name and some
arguments and stops. **Owns** — the list is the only state there is, so
whoever owns the list owns the agent.

A standalone project, owned by 345 Consulting, LLC. It exists to be read.

    just list                 every chapter on disk, in order
    just run ch01_single_call runs it, writes out/<chapter>.html, prints the path
    just run-live ch01_...    the same chapter against a real provider
    just book                 every chapter that has been run, on one page
    just gate                 ruff + format + mypy + pytest

## How it is laid out

    src/
      support/       scaffolding — models, trace, view. Never a lesson.
      chapters/      the lessons, numbered

Three rules keep the scaffolding honest:

- **Nothing in `support/` may import from `chapters/`.**
- **Every symbol a chapter imports from `support/` was built by hand in an
  earlier chapter first.** Teach it inline, then graduate it. If a chapter
  imports a mechanism no earlier chapter explained, there is a hidden
  mechanism — which is the thing this primer exists to eliminate.
- **Recording is scaffolding, even when it is interesting.** `Trace`, `view`
  and the httpx wire hooks observe; they never participate. The rule above
  constrains mechanisms a chapter uses to *work*, not the instruments
  watching it.

## Reading a trace

The HTML is a projection; `out/<chapter>.json` is the record. Every span is an
entry with a matching exit, nested by depth. Model spans carry two notes: the
exact context sent to the provider, and the reply that came back.

## Mock and live

Chapters take the model as an argument. The scripted model replies from a fixed
list, so a chapter's trace is identical on every run and the mechanics can be
read rather than guessed at. A live run proves the same mechanics against a real
provider — and shows you which parts of the trace were the mock's cooperation.

Live runs require `DEEPSEEK_API_KEY` in the environment. There is no fallback:
absent the key, the run fails.

The two do not prove the same things. A scripted model lets a test assert exact
mechanics; a live one can only be asserted against invariants that hold for any
model. Anything provable only under mock was the mock's cooperation, not a
guarantee — see `FINDINGS.md`, where the sharpest finding so far was invisible
to every passing test because the script always asked the right question.
