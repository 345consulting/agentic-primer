# The agentic primer — the mechanics of an agentic harness

> **An agent is a loop whose exit condition is set by the model.**

Unpacked:

> It sends the whole context to a model, executes whatever the model requests,
> appends the results to that same context, and repeats — until the model asks
> for nothing further. The harness owns the list and every decision about it;
> the model only ever gets a vote.

That is the whole subject, and the first three chapters are that sentence, one
clause each:

    a loop that sends the whole context to a model             ch01_single_call
    executes whatever the model requests, appends the results  ch02_tool_call
    and repeats until the model asks for nothing further       ch03_the_loop

Remove any one and it is not an agent. Everything after chapter 3 answers a
single question: *what did this buy over the loop?*

Two words do the work. **Requests** — the model never executes anything and
never decides anything; it emits a name and some arguments and stops. **Owns**
— the list is the only state there is, so whoever owns the list owns the agent.

And the one-line version is what separates an agent from everything else it
resembles: a workflow's exit condition is set at write time, an agent's at run
time, by something else. A run has seven ways to end and only that one means
finished; the other six — a cap, a truncated reply, a spent budget, a
deadline, a full context window, no progress, a veto — are limits you imposed.
Any loop can have limits. Only an agent asks something else whether to
continue.

A standalone project, owned by 345 Consulting, LLC. It exists to be read.

    just list                 every chapter on disk, in order
    just run ch01_single_call runs it, writes out/<chapter>.html, prints the path
    just run ch01_... live    the same chapter against the model provider
    just book                 every chapter that has been run, on one page
    just gate                 ruff + format + mypy + basedpyright + pytest

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
guarantee — see `LEARNINGS.md`, where the sharpest finding so far was invisible
to every passing test because the script always asked the right question.
