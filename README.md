# The agentic primer — the mechanics of an agentic harness

Read it live: **[345consulting.github.io/agentic-primer](https://345consulting.github.io/agentic-primer/)**

## The definition

> **An agent is a loop whose exit condition is set by the model.**

Unpacked:

1. It sends the whole context to a model
2. Executes whatever the model requests, appends the results, and 
3. repeats until the model asks for nothing further

The harness owns the list and every decision about it; the model only ever gets a vote.

That is the whole subject, and the first three chapters are the **atomic
agent** — that one sentence, one clause each:

- a loop that sends the context to a model [`ch01_single_call.py`](src/chapters/ch01_single_call.py)
- executes a tool that the model requests [`ch02_tool_call.py`](src/chapters/ch02_tool_call.py)
- and repeats until the model asks for nothing further [`ch03_the_loop.py`](src/chapters/ch03_the_loop.py)

Remove any one and it is not an agent.

## Why this exists

My understanding of agentic flows was good, but it lacked detail.
Tutorials explain the overall behavior — what an agent does, how it
looks from the outside — but not the mechanics underneath. I wanted to
know the internals well enough to fine-tune any lever I needed, not
just call an SDK and trust it.

So every chapter here is a hand-built mechanism first, run, and checked
against its own trace — and only once that's understood does a
framework get to claim it does the same thing.

A standalone project, owned by 345 Consulting, LLC. It exists to be read.

## Running it

    just list                       every chapter on disk, in order
    just run ch01_single_call       runs it, writes out/ch01_single_call.html
    just run ch01_single_call live  the same chapter against the model provider
    just book                       every chapter that has been run, on one page
    just gate                       ruff + format + mypy + basedpyright + pytest

## Chapters

| chapter | summary |
|---|---|
| [`ch01_single_call.py`](src/chapters/ch01_single_call.py) | one invoke; no tools, no loop, no framework |
| [`ch02_tool_call.py`](src/chapters/ch02_tool_call.py) | the model asks; we execute; turn two knows |
| [`ch03_the_loop.py`](src/chapters/ch03_the_loop.py) | the whole loop, and the two ways it is allowed to end |
| [`ch04_tool_failures.py`](src/chapters/ch04_tool_failures.py) | the toolbox lets you down in three ways; one of them looks like it |
| [`ch05_loop_endings.py`](src/chapters/ch05_loop_endings.py) | every way a run can stop, and only one means finished |
| [`ch06_two_tools.py`](src/chapters/ch06_two_tools.py) | the same two calls, together or in sequence |
| [`ch07_tool_http.py`](src/chapters/ch07_tool_http.py) | a tool that calls an API: latency, a second secret, real failures |
| [`ch08_tool_program.py`](src/chapters/ch08_tool_program.py) | a tool that runs a program: exit codes, and an argument for command |
| [`ch09_who_retries.py`](src/chapters/ch09_who_retries.py) | four cells of one matrix: who can change the outcome |
| [`ch10_routing.py`](src/chapters/ch10_routing.py) | the loop branches -- an `if`, in three different places |
| [`ch11_state.py`](src/chapters/ch11_state.py) | what travels besides messages, and who may write it |
| [`ch12_supervisor.py`](src/chapters/ch12_supervisor.py) | one loop calls another: a tool whose body is an agent |
| [`ch13_workflow.py`](src/chapters/ch13_workflow.py) | the same job with nothing deciding -- is the loop worth it? |
| [`ch14_stream.py`](src/chapters/ch14_stream.py) | "stream": true -- a reply arrives in pieces |
| [`ch15_stream_tools.py`](src/chapters/ch15_stream_tools.py) | tool arguments arrive as fragments of a JSON string |
| [`ch16_observability.py`](src/chapters/ch16_observability.py) | instrumentation that observes and never participates |
| [`ch17_stream_producer.py`](src/chapters/ch17_stream_producer.py) | the harness becomes the server |
| [`ch18_hooks.py`](src/chapters/ch18_hooks.py) | the named points in the loop, and the three powers |
| [`ch19_guards.py`](src/chapters/ch19_guards.py) | a hook that can say no, before dispatch |
| [`ch20_loop_veto.py`](src/chapters/ch20_loop_veto.py) | stopped because forbidden, which is not stopped because done |
| [`ch21_judge.py`](src/chapters/ch21_judge.py) | a hook that reads the reply, per turn and not per run |
| [`ch22_prompt_types.py`](src/chapters/ch22_prompt_types.py) | everything that enters the context is a prompt |
| [`ch23_mcp.py`](src/chapters/ch23_mcp.py) | a dispatch table you did not write |
| [`ch24_mcp_injection.py`](src/chapters/ch24_mcp_injection.py) | descriptions you did not write, in a context you did |
| [`ch25_rag_injection.py`](src/chapters/ch25_rag_injection.py) | a document you did not write, telling the model what to do |
| [`ch26_skills.py`](src/chapters/ch26_skills.py) | a tool whose result is instructions, not data |
| [`ch27_compression.py`](src/chapters/ch27_compression.py) | the list is too long; what do you drop, and what does it cost? |
| [`ch28_caching.py`](src/chapters/ch28_caching.py) | what triggers a hit, what triggers a miss |
| [`ch29_memory.py`](src/chapters/ch29_memory.py) | what survives when the list is thrown away |
| [`ch30_roles.py`](src/chapters/ch30_roles.py) | same content, three roles -- who's saying it, and does it matter? |
| [`ch31_stop_resume.py`](src/chapters/ch31_stop_resume.py) | a jsonl snapshot after every turn, and picking back up |
| [`ch32_graph.py`](src/chapters/ch32_graph.py) | the same behaviour as a StateGraph -- what did it buy? |
| [`ch33_limits.py`](src/chapters/ch33_limits.py) | recursion_limit at the boundary |
| [`ch34_checkpoint.py`](src/chapters/ch34_checkpoint.py) | MemorySaver, thread_id, resume -- and why that is not memory |
| [`ch35_interrupt.py`](src/chapters/ch35_interrupt.py) | interrupt and Command(resume=...) as an approval gate |
| [`ch36_subgraph.py`](src/chapters/ch36_subgraph.py) | the supervisor as a graph node -- what did it buy? |
| [`ch37_parallel.py`](src/chapters/ch37_parallel.py) | fan-out, Send, join, and the order things merge in |
| [`ch38_reducers.py`](src/chapters/ch38_reducers.py) | two updates to one field: append, or replace |
| [`ch39_callbacks.py`](src/chapters/ch39_callbacks.py) | the audit trail and the hook are the same mechanism in LangChain |
| [`ch40_capstone.py`](src/chapters/ch40_capstone.py) | everything since graph, composed into one real scenario |
| [`ch41_result_cache.py`](src/chapters/ch41_result_cache.py) | a cache of answers, not of attention -- what's needed to hit it |
| [`ch42_model_tiers.py`](src/chapters/ch42_model_tiers.py) | the same task, two models -- getting a complexity signal right |

## How it is laid out

    src/
      support/       scaffolding — models, trace, view. Never a lesson.
      chapters/      the lessons, numbered
    tests/
      support/       tests for the scaffolding
      chapters/      one test file per chapter — 42, matching the ladder
    out/             generated by `just run`/`just book`, and committed --
                     it's the hosted book, and the receipts for it.

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

## License

MIT — see `LICENSE`. Copyright 345 Consulting, LLC. Third-party dependency
licenses are in `THIRD_PARTY_LICENSES.md`.

## How this was built

Built with [Claude Code](https://claude.com/claude-code), conversationally,
chapter by chapter — designed, built, run, and checked against its own
trace before moving on. The commit history is the actual record of that
process, findings and corrections included.
