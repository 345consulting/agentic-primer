# Agentic Primer

A 345c-owned learning project. The mechanics of an agentic harness, built from
the ground up, one chapter at a time — no framework abstractions hiding the
loop, and a trace of every entry and exit so the decision points are readable
rather than inferred.

A learning artifact, not a product and not a client project. It exists to be read.

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

Comment density tapers. `single_call` explains everything because nothing is
established; later chapters comment only what is new. The comments are the
delta from the previous chapter.

## The ladder

The chapters on disk are the reading order — they are numbered, so `ls` and
`just list` answer "what comes next", and each one's first docstring line is
its summary. There is no list of chapters in code, because a second copy of
either fact is a second thing to keep true.

What is not written yet is a plan, and a plan is prose:

```
ch01 single_call      one invoke; no tools, no loop, no framework   written
ch02 tool_call        the model asks; we execute; turn two knows   written
ch03 the_loop         the whole loop, and the two ways it is allowed to end   written
ch04 tool_failures    the toolbox lets you down in three ways; one of them looks like it   written
ch05 loop_endings     every way a run can stop, and only one means finished   written
ch06 two_tools        two calls in one reply, both succeed -- still ONE turn
ch07 tool_http        a tool that calls an API: latency, a second secret, real failures
ch08 retry_policy     transient or permanent, and who is allowed to say so
ch09 retry_by_local   the harness calls again: no model, no tokens, backoff
ch10 retry_by_model   the model asks again: a full turn, and a longer list
ch11 retry_exhausted  out of strikes: escalate, and with what context?
ch12 routing          the loop branches -- an `if`, in three different places
ch13 state            what travels besides messages, and who may write it
ch14 supervisor       one loop calls another: a tool whose body is an agent
ch15 workflow         the same job with nothing deciding -- is the loop worth it?
ch16 stream           "stream": true -- a reply arrives in pieces
ch17 stream_tools     tool arguments arrive as fragments of a JSON string
ch18 observability    instrumentation that observes and never participates
ch19 hooks            the named points in the loop, and the three powers
ch20 guards           a hook that can say no, before dispatch
ch21 loop_veto        stopped because forbidden, which is not stopped because done
ch22 judge            a hook that reads the reply, per turn and not per run
ch23 prompt_types     everything that enters the context is a prompt
ch24 mcp              a dispatch table you did not write
ch25 mcp_injection    descriptions you did not write, in a context you did
ch26 rag_injection    a document you did not write, telling the model what to do
ch27 skills           a tool whose result is instructions, not data
ch28 compression      the list is too long; what do you drop, and what does it cost?
ch29 memory           what survives when the list is thrown away
ch30 graph            the same behaviour as a StateGraph -- what did it buy?
ch31 limits           recursion_limit at the boundary
ch32 checkpoint       MemorySaver, thread_id, resume -- and why that is not memory
ch33 interrupt        interrupt and Command(resume=...) as an approval gate
ch34 subgraph         the supervisor as a graph node -- what did it buy?
ch35 parallel         fan-out, Send, join, and the order things merge in
ch36 reducers         two updates to one field: append, or replace
```

Chapters are named in prose and numbered only in that list. The numbers have
moved five times in two days; the names have not.

**`single_call`, `tool_call` and `the_loop` are the atomic agent, one clause
of the definition each.**

    a loop that sends the whole context to a model             single_call
    executes whatever the model requests, appends the results  tool_call
    and repeats until the model asks for nothing further       the_loop

Remove any one and the definition stops being satisfiable. `single_call` alone
is the degenerate case — zero iterations, the condition false the first time.
`tool_call` without `the_loop` is a guess about depth. `the_loop` without
tools never iterates.

The turn cap in `the_loop` is the one thing there that the definition does not
ask for. It is prudence, not part of the atomic agent: shipping an unbounded
`while` is irresponsible, and the model's own ending is still the only one
that means finished.

**Everything after it is a variation, an addition, or the framework.**
Variations are the same atomic agent under conditions it did not choose — no
tool covers the question, a tool raises, a table someone else wrote, a result
that is instructions, a reply that arrives in pieces. Additions are machinery
the atomic agent does not have — retry policy, routing, bounds, hooks, guards,
judge, compression, memory. `workflow` is neither: it is the contrast, an exit
condition set at write time. And `supervisor` is the atomic agent containing
itself, which is why it needs no new machinery.

Those groups are a way to read the ladder, not a way to sort it — the order
follows dependencies instead. A veto needs guards, an injection needs a table
to inject into, a subgraph needs a supervisor and a graph.

Everything up to `graph` stays in plain Python, so the framework's answers can
each be asked what they bought.

**A scenario is a situation.** Not the knob turned, not the value expected
back. Every chapter declares at least one, so a run is always
`run → scenario → turn → model/tool` and the page has a single shape. A
chapter whose lesson *is* the comparison holds several; a chapter that adds a
clause holds one. `tool_failures` is the first kind — its three situations put
a run with no failure note anywhere beside a run with one, which makes "some
failures do not fail" visible rather than asserted.

`loop_endings` is the same idea applied to stopping. A run ends for one of
several reasons and only the first means the work finished: the model asked
for nothing, we hit a cap, the reply was cut off, a budget or a deadline ran
out, the context filled, nothing progressed, a guard refused. Each is a
scenario rather than a chapter, because the lesson is that they are
indistinguishable unless the summary says which. A veto is the exception and
waits for `guards`, since it needs something to do the vetoing.

`two_tools` is the clean case of what `partial_failure` showed under duress:
two calls in one reply, both succeeding, one turn. Fan-out on its own decides
nothing — it is a fact about a trace — which is why the failing version came
first and this one reads as the baseline it was measured against. Ordering and
concurrency belong to `parallel`, where a framework runs the same two calls on
a pool and the only visible difference is `seq` and `thread`.

`tool_http` is the first tool that leaves the process, and it makes the retry
chapters concrete: there is no transient failure in a dict lookup. It brings
latency inside the tool span, real failure classes where timeout and 429 and
503 are transient and 404 and 401 are not, and a second secret — the first in
this primer that is not the model provider's.

**Retry is four questions, not one.** `retry_policy` is classification —
transient or permanent, safe to call twice or not — declared by the tool
author, because nobody else knows. `retry_by_local` is the harness calling
again: no model, no tokens, bounded by backoff, and correct only when the
outcome can change. `retry_by_model` is the model asking again after reading
an error: a full turn each time, on a list that has grown by a request and a
failure. `retry_exhausted` is escalation, and what a human is handed at 2am.

The organising question across all four is *who can change the outcome*. Bad
arguments is the only case where the model retrying is right and the harness
retrying is useless — and it is the one case LangGraph's `ToolNode` handles by
default, reporting `ToolInvocationError` back and re-raising everything else.
There is no retry anywhere in LangGraph.

`routing` and `state` are the shape of the loop itself. Routing is an `if` in
three different places — after the reply, before the model, on something we
carry — and `add_conditional_edges` is that `if` with a graph around it.
`state` is the correction to the definition: turns, endings and counters have
been travelling in `run()`'s locals all along, outside the list this primer
keeps calling the only state there is.

`supervisor` and `workflow` are orchestration, and differ only in who chooses
the sequence. A supervisor is a pattern with a cheap mechanism — a tool whose
implementation is another loop — which is why multi-agent arrives in plain
Python. A subgraph is one other way to build it, and having the cheap one
first is what makes `subgraph` answerable: call-and-return against handoff, an
isolated worker against a shared message list, one string coming back against
a worker's whole transcript merging into the parent's context and its bill.

`stream` and `stream_tools` change the recorder before they change a chapter.
Every request until then carries `"stream": false`, and the wire hooks call
`response.read()` — which consumes a body that has not finished arriving. A
reply stops being an object and becomes a sequence; `model_provider_ms` stops
being one number. Then tool arguments arrive as fragments of a JSON string, so
a call cannot be parsed, judged or dispatched until the stream ends — which a
judge and a streaming interface want in opposite directions.

**`observability` comes before `hooks`, and they are not the same thing.**
Instrumentation observes and never participates: delete every span and the
agent behaves identically, and you simply cannot see it. A hook participates —
a guard vetoes, a judge rejects — and deleting them leaves an agent that runs
ungoverned. It is easy to conflate because LangChain implements tracing
through callbacks, and that is how you end up with an audit trail a hook can
silently suppress. `observability` is also where the hand-written spans in
these chapters could stop being hand-written, and where the `gen_ai.*`
conventions belong.

`hooks`, `guards` and `judge` are the seam. Hooks is the mechanism: the named
points in the loop and the three powers, in increasing order of danger —
observe, modify, veto. Guards adds veto before dispatch. Judge adds a verdict
on the reply, per turn rather than per run, which is the whole point: a
finding that arrives after the run is a report, and one that arrives during it
is a decision. Judge comes after retry so the strike machinery is already
built; a rejection is a second trigger for the same counter and looks nothing
like a tool that failed.

`prompt_types` is the general claim the injection chapters are instances of:
everything that enters the context is a prompt, whatever field carries it. The
useful axis is not whether you control it — you control most of it — but when
it was authored and who has read it since.

| authored | examples | last reviewed |
| --- | --- | --- |
| per conversation | system, user | as it is written |
| at design time | tool descriptions, argument enums, error templates, skill bodies | once, months ago — or never, if a dependency wrote it |
| during the run | tool results, retrieved documents, MCP descriptions, web pages | never, by anyone |

The middle class is the one that surprises people, because it is theirs and
they still never look at it as text a model obeys. A tool description lives in
a docstring, and docstrings are not reviewed as prompt engineering — which is
exactly the `tool_call` learning, where `part: str` against `Literal[...]`
looked like a typing decision and was the only thing constraining what the
model could ask for.

`mcp`, `mcp_injection` and `rag_injection` are where a third party writes into
a context we own. MCP is a dispatch table discovered at runtime, so the
declared list stops being a literal and its order becomes whatever a server
returned — quietly forfeiting the stable prefix the cache learning depends on.
They come after guards on purpose: the seam should exist before a stranger is
plugged into it, and "which of these tools can I actually gate" is a better
question than "what is a guard".

`skills`, `compression` and `memory` are admission, eviction and persistence —
one problem, which is that the list is the state and the budget is finite.

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

## Learnings

`LEARNINGS.md` — things learned by running the chapters that outlive the chapter
that produced them. Appendable, dated. Chapter-specific observations live in
that chapter's docstring instead.

Learnings name chapters, never number them. A finding outlives the reading
order, and the ladder below has been renumbered four times in one day.

## Model

`deepseek-v4-flash`, chosen for price: cheap enough that running a chapter live
is routine rather than an event. Live runs need `DEEPSEEK_API_KEY` in the
environment and there is no fallback — absent the key, the run fails.

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
