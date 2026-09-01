# Primer — the mechanics of an agentic harness

A standalone project. It shares nothing with `../src` and never imports from
it. Read the chapters in order; each one adds exactly one idea to the last.

    just list                 the reading order
    just run ch01_single_call runs it, writes out/<chapter>.html, prints the path
    just run-live ch01_...    the same chapter against a real provider
    just gate                 ruff + format + mypy + pytest

## How it is laid out

    primer/
      order.py       the reading order, as data
      support/       scaffolding — models, trace, view. Never a lesson.
      chapters/      the lessons, numbered

Two rules keep the scaffolding honest:

- **Nothing in `support/` may import from `chapters/`.**
- **Every symbol a chapter imports from `support/` was built by hand in an
  earlier chapter first.** Teach it inline, then graduate it. If a chapter
  imports a mechanism no earlier chapter explained, there is a hidden
  mechanism — which is the thing this primer exists to eliminate.

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
