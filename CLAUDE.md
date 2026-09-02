# Agentic Primer

A 345c-owned learning project. The mechanics of an agentic harness, built from
the ground up, one chapter at a time — no framework abstractions hiding the
loop, and a trace of every entry and exit so the decision points are readable
rather than inferred.

Not a product, not a client project, not HiQ. It exists to be read.

## Build

Run every recipe from this directory.

```
just list                      every chapter on disk, in order
just run ch01_single_call      mock run; writes out/<chapter>.html
just run ch01_single_call live the same chapter against the model provider
just gate                      ruff + format + mypy + basedpyright + pytest
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
ch03 missing_tool     no tool covers the question, and nothing fails   written
ch04 tool_failure     the tool runs and raises; we decide what the model sees   written
ch05 the_loop         the whole loop, twelve lines of plain Python
ch06 two_tools        two calls in one reply -- still ONE turn
ch07 retry_policy     transient or permanent, and who is allowed to say so
ch08 retry_by_local   the harness calls again: no model, no tokens, backoff
ch09 retry_by_model   the model asks again: a full turn, and a longer list
ch10 retry_exhausted  out of strikes: escalate, and with what context?
ch11 supervisor       one loop calls another: a tool whose body is an agent
ch12 workflow         the same job with nothing deciding -- is the loop worth it?
ch13 stream           "stream": true -- a reply arrives in pieces
ch14 stream_tools     tool arguments arrive as fragments of a JSON string
ch15 hooks            the named points in the loop, and the three powers
ch16 guards           a hook that can say no, before dispatch
ch17 judge            a hook that reads the reply, per turn and not per run
ch18 skills           a tool whose result is instructions, not data
ch19 compression      the list is too long; what do you drop, and what does it cost?
ch20 memory           what survives when the list is thrown away
ch21 graph            the same behaviour as a StateGraph -- what did it buy?
ch22 limits           recursion_limit at the boundary
ch23 checkpoint       MemorySaver, thread_id, resume -- and why that is not ch20
ch24 interrupt        interrupt and Command(resume=...) as an approval gate
ch25 subgraph         the ch11 supervisor as a graph node -- what did it buy?
ch26 parallel         fan-out, Send, join, and the order things merge in
```

Chapter 5 is a complete agentic loop in twelve lines of plain Python. Every
chapter after it answers one question: *what did this buy over chapter 5?*
Everything through ch20 stays in plain Python, so the framework's answers from
ch21 on can be asked what they bought.

Chapters 3 and 4 are the failure pair, and they fail differently: in ch03
nothing goes wrong and the question is unanswered anyway; in ch04 something
goes wrong and we choose what the model is told. Neither retries, because
retrying needs a loop.

Chapters 7 to 10 are retry, following straight on from ch04's failure, and
split four ways because they are four different questions. **Policy** is
classification — transient or permanent, safe to call twice or not — and it is
declared by the tool author, because nobody else knows. **By local** is the
harness calling again: no model, no tokens, bounded by backoff, and correct
only when the outcome can change. **By model** is the model asking again after
reading an error: a full turn each time, on a list that has grown by a request
and a failure, so attempt three costs more than attempt one. **Exhausted** is
what happens when the strikes run out. The counter has to live outside the
model, because from inside the loop attempt four looks exactly like attempt
one — and the harder half is what gets handed to the human at 2am, since a run
that gave up with no account of what it tried is worse than one that never
started.

The organising question across retry is *who can change the outcome*. Bad
arguments is the only case where the model retrying is right and the harness
retrying is useless — and it is the one case LangGraph's `ToolNode` handles by
default, reporting `ToolInvocationError` back and re-raising everything else.
There is no retry anywhere in LangGraph.

Chapters 11 and 12 are orchestration, and they differ only in who chooses the
sequence. **Supervisor** is the pattern with the cheapest mechanism: a tool
whose implementation is another loop. Nothing new is needed -- `execute_tool`
dispatches, and the thing it calls happens to run its own `while` and return a
string -- which is why multi-agent arrives here, in plain Python, rather than
with the framework. It is also the first chapter where nesting is not merely
depth, and where `seq` and `thread`, recorded independently of nesting since
chapter 1, start to earn their keep.

A supervisor is a pattern; a subgraph is one way to build it. Chapter 25 is
that way, and having ch11 first is what makes it answerable: call-and-return
against handoff, an isolated worker against a shared message list, one string
coming back against a worker's whole transcript merging into the parent's
context and its bill.

Chapter 12 is the counterweight to chapter 5 and asks the question the rest of
the primer assumes away: the same job as a fixed sequence, with nothing
deciding anything, is cheaper, deterministic and testable. A workflow needs no
framework either, so it stays plain Python — and it is the reason ch21 lands
as it does, because a `StateGraph` is a workflow engine of which the agent
loop is one special case.

Chapters 13 and 14 are streaming, and they change the recorder before they
change a chapter. Every request up to here carries `"stream": false`, and
`_build_wire_hooks` calls `response.read()` — which consumes a body that has
not finished arriving. So `support/` learns to record events as they arrive,
with their arrival times, and the chapters follow. A reply stops being an
object and becomes a sequence; `model_provider_ms` stops being one number and
becomes an interval with a first token somewhere inside it. In ch14 the tool
arguments arrive as fragments of a JSON string, so a tool call cannot be
parsed, judged or dispatched until the stream ends — which is the sharpest
governance question in the primer, because a judge wants the whole turn and a
streaming interface has already shown the user half of it.

Chapters 15 to 17 are the seam. **Hooks** is the mechanism: the named points
in the loop — before the model, after the reply, before dispatch, after the
result, around a turn, around the run — and the three powers a hook can have,
in increasing order of danger: observe, modify, veto. The `Trace` from chapter
1 is already the first of these; it observes at exactly those points and was
never called a hook. **Guards** adds veto, before dispatch. **Judge** adds a
verdict on the reply, per turn rather than per run, which is the whole point:
a finding that arrives after the run is a report, and a finding that arrives
during it is a decision.

Judge comes after retry rather than before it, so the strike machinery is
already built on the concrete case. A judge's rejection is a second trigger
for the counter from ch10, and it looks nothing like a tool that failed —
which is easier to see once the first trigger works.

Chapters 18 to 20 are admission, eviction and persistence — one problem, which
is that the list is the state and the budget is finite.

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
