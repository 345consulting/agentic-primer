# Learnings

Things learned that outlive the chapter that produced them. Most came from
running one and reading the trace; a few came from reading a dependency's
source, and one is a synthesis of the others — each says which, because the
provenance is part of the claim.

Chapters are named, never numbered: a learning outlives the reading order, and
the numbers move. Chapter-specific observations belong in that chapter's
docstring; this file is for what generalizes.

Append, never rewrite. Date each entry.

**Before writing an entry, ask what was learned.** Don't draft one from a
unilateral read of the conversation -- confirm the framing with Sanjeev first,
the way every entry so far was proposed and discussed before it was written.

---

## 2026-09-01 — An agent is a loop whose exit condition is set by the model

*Synthesis, and the first one. It came out of a conversation, was corrected
twice, and has been amended once since by what the chapters did.*

**The definition.**

> An agent is a loop that sends the whole context to a model, executes whatever
> the model requests, appends the results to that same context, and repeats —
> until the model asks for nothing further. The harness owns the list and every
> decision about it; the model only ever gets a vote.

**Two words carry it.** *Requests* — the model never executes and never
decides; it emits a name and some arguments and stops, and whether anything
runs is the harness's call. *Owns* — the list is the only state there is, so
whoever owns the list owns the agent. Language like "the agent decides to use
a tool" hides exactly the seam a harness needs.

**What it excludes.** Communication is not the subject; that is transport, and
it is solved. Tools are not the subject either — they are what a turn may
contain. The subject is the condition: go round again, or stop. A framework
does not add agency; `ch03_the_loop` will have all of it in twelve lines. What
a framework adds is management — persistence, concurrency, resumption, limits.

**`ch01_single_call` was already an agent**, in the degenerate case: zero
iterations, because the first reply asked for nothing. Nothing was missing; the
condition was false the first time.

**Amended 2026-09-02.** "Until the model asks for nothing further" is the happy
path, and it is one of seven ways a run ends. The others are a turn cap, a
truncated reply that the loop's own condition reads as completion, an
exhausted budget, a deadline, a full context window, no progress, and a veto.
Six of the seven are stops rather than completions, and a summary that does not
distinguish them reports a run that gave up as a run that finished.

So the definition holds. The six are limits, not the condition — any loop can
have a cap, and only an agent asks something else whether to continue. When a
limit fires the run did not finish, it was stopped, which is why the summary
has to say which.

That is also the line between an agent and a workflow, and the whole of
`ch13_workflow`: a workflow's exit condition is set at write time, an agent's
at run time, by something that is not you.

---

## 2026-09-01 — a recorder that omits a field looks exactly like a provider that never sent it

**What happened.** The first live run of `ch01_single_call` showed
`reasoning_content` on the wire and no reasoning in the normalized `AIMessage`.
The conclusion drawn was that `langchain-deepseek` discards it. That was wrong.
The adapter preserves it (`langchain_deepseek/chat_models.py:315`, into
`additional_kwargs`); the primer's own `trace._describe` recorded only id, role,
content and tool calls, and dropped everything else silently.

**Why it matters.** A defect was asserted about a dependency on the strength of
an absence in our own instrument. Nothing failed, no test went red, and the
trace looked complete — it had simply stopped carrying a field. For a product
whose deliverable is the audit trail, this is the failure that costs the most
and announces itself the least: the record stays plausible while it stops being
true.

**The rule.** An instrument is not trusted until it has been checked against a
source outside itself. Here that source is the wire — which is the argument for
the httpx hooks in `support/models.py` being worth their keep, beyond
curiosity. Any field a recorder does not capture must be a field it was
*decided* not to capture, written down as such. Silence is not a decision.

**Where it applies beyond the primer.** Any recorder that reads token usage
off a normalized message inherits the same assumption. Reasoning is a subset
of output tokens rather than an addition — `completion_tokens: 21` with
`reasoning_tokens: 18` — so a log keeping only `output_tokens` gets the bill
right and loses the breakdown. "21 tokens for a one-sentence answer" and
"3 tokens of answer, 18 of reasoning" are materially different records, and
only one of them is true.

---

## 2026-09-01 — An unconstrained tool argument turns a wrong premise into a fluent fact

**What happened.** `ch02_tool_call`'s tool was declared `stock_on_hand(part: str)` and
looked up `{"flange": 17, "grommet": 240}.get(part, 0)`. Asked about flanges,
the live model called it with `part="flanges"` — plural, which is what an
English sentence about more than one flange contains. The lookup returned its
default. Turn two then answered, in fluent and confident prose, *"We have 0
flanges on hand."*

Nothing failed. The tool ran, returned an `int`, the loop completed, both turns
were recorded, and every test passed — because the mock's script asks for
`"flange"` and always will.

**Why it matters.** Two separate defects compounded, and each alone would have
been survivable.

The schema said `str`, so every string was a legal request. A tool's parameter
schema is not documentation: it is the only thing constraining what the model
is able to ask for, and it is enforced by the provider before the call is ever
made. Declaring `str` declines that enforcement.

The lookup had a default, so an argument outside the intended set produced an
answer instead of an error. `0` is a plausible stock level, indistinguishable
from a true one, and by the time it reaches the model it has lost every trace
of being a fallback. The model is not at fault for believing it — it has no
other source.

**The rule.** Constrain the argument in the schema, and fail on anything the
schema let through anyway. `Literal["flange", "grommet"]` puts
`enum: [...]` in the request body; `STOCK[part]` raises rather than coercing.
A tool that answers a question it did not understand is worse than one that
fails, because a failure stops the loop and a wrong number does not.

**What the mock could not have told us.** A scripted model asks for exactly
what the script says. This was only findable live, and it is the sharpest
example so far of the standing warning that the two kinds do not prove the
same things.

---

## 2026-09-01 — The library cost is first-call warm-up, not per-call overhead

**What happened.** `ch01_single_call`'s first live run split as 1431ms provider against
663ms library, and 32% unexplained was left open with two candidate
explanations: one-time client construction, or real per-call adapter cost.
`ch02_tool_call` has two invocations in one process and settles it.

| run | turn | provider | library |
| --- | --- | --- | --- |
| ch01_single_call | 1 | 1431ms | 663ms |
| ch01_single_call | 1 | 1455ms | 237ms |
| ch02_tool_call | 1 | 2268ms | 267ms |
| ch02_tool_call | 2 | 1685ms | **14ms** |

The second invocation in the same process costs 14ms of library time. The
overhead is warm-up, not per-call.

**Why it matters.** It is *not* client construction, which was the leading
hypothesis. `ch02_tool_call` builds a fresh `ChatDeepSeek` and a fresh `httpx.Client`
for each turn — because the wire hooks bind to a span, and the span differs per
turn — and turn two still costs 14ms. What is amortised is process-level: the
pydantic model machinery, the tool-schema conversion, the SSL context. TLS
handshaking is not in this number; it falls inside `provider_ms`, between the
request hook and the response hook.

For a harness this is the difference between an optimisation target and a
startup cost to pay once and ignore. It also means any benchmark that measures
a single invocation per process is measuring warm-up, and will overstate
per-call library cost by an order of magnitude.

**Still open.** Whether the 663ms/237ms spread on two otherwise identical
`ch01_single_call` runs is ordinary variance or something else. Two samples.

---

## 2026-09-01 — You are billed on the sum of prefixes, and the tool schema is most of it

**What happened.** Measured across the two chapters' live runs, per turn:

| run | turn | input | output | cache hit |
| --- | --- | --- | --- | --- |
| ch01_single_call | 1 | 122 | 20 | 0 |
| ch02_tool_call | 1 | 397 | 74 | 0 |
| ch02_tool_call | 2 | 484 | 37 | 384 |

`ch02_tool_call`'s conversation ends at 484 tokens of context and costs **881 billed
input tokens** to get there. Nothing was re-read and nothing was retried.

**Where the tokens actually go.** The step from 122 to 397 is one tool
declaration — a name, a two-value enum, and a one-line docstring — costing
**275 tokens**, re-sent on every call whether the model uses it or not. The
tool round trip that did the actual work, request plus result, cost 87. The
declaration is three times the conversation, and it scales with the size of the
toolbox rather than with the length of the exchange. Twenty tools is the whole
context before anyone has said anything.

**Why it compounds.** Every turn re-sends the entire history, so cost is the
sum of the prefixes rather than the length of the final context. Ten turns of
this shape is not ten times one turn; it is the triangle.

**The mitigation, and what it costs to break.** Turn two hit DeepSeek's prefix
cache for 384 of 484 tokens. That cache is provider-side KV state keyed on an
exact token-prefix match, and it is possible *only because* the context is
append-only — tool calls and results being permanent members of the transcript
is precisely what keeps the prefix byte-stable.

The hit was 384 and not 397, which is 6 x 64 exactly: it caches in blocks, and
the 13 tokens in the trailing partial block were not reusable. Cache boundaries
land where the block ends, not where the message does.

**The rule this sets for the harness.** Anything that edits history invalidates
the cache from the edit point onward and pays full price for every token after
it. Pruning an old tool result, injecting a judge's note mid-transcript,
rewriting a tool call before dispatch — each is a cache miss for the remainder
of the run. Append-only is cheap; editing is not. Governance that must alter
the transcript should alter it as late in the list as possible.

Two structural corollaries: declare tools in a stable order, because reordering
the list changes the prefix and forfeits the hit for nothing; and keep the
system prompt free of anything that varies per call, such as a timestamp.

**A caution about the other kind of cache.** A harness-level *response* cache
(LangChain's `set_llm_cache`, semantic caches) is a different thing entirely.
A prefix-cache hit is invisible and always correct. A response-cache hit means
the model never ran, and in an agent loop that means the tool never ran either
— a stale answer served with a trace that looks healthy. `set_llm_cache` is
global; switched on here it would return instantly, the httpx hooks would
record nothing, and `provider_ms` would read as a fast provider rather than a
call that never happened. Cache the deterministic parts; never the decision.

---

## 2026-09-01 — The model retried the tool because we kept declaring it

**What happened.** An earlier draft of `ch04_missing_tool` declared a tool it could not
dispatch, reported the miss back as a `ToolMessage` with `status="error"`, and
watched the live model call the same tool again — twice, on two separate runs.
The obvious reading was that reporting a failure invites a retry, and that a
harness therefore needs a counter.

The trace said otherwise. Turn two's request body:

```
tools sent   = ['stock_on_hand', 'restock_eta']
tool message = "tool ['restock_eta'] is not available"
```

The tool list is sent on every call, so the request that told the model the
tool had failed *also told it the tool existed* — and a schema in the `tools`
array is a much stronger claim than one sentence in a message. The model was
not being stubborn. It was resolving a contradiction we sent it, in our favour.

**Why it matters.** The first conclusion was about the model, and it would have
produced the wrong fix: firmer error wording, or a retry budget to contain
behaviour that was never the model's to begin with. The real defect is ours,
and the fix is structural — stop advertising a tool that cannot run. Nothing
requires the declared list to be the same on every turn. `ch02_tool_call` established
that it is re-sent every time, which is exactly what makes it changeable.

A retry budget is still worth having. It is just not what this was.

**The rule.** When the model does something that looks irrational, read the
request body before theorising about the model. The context is the only thing
it has, and we assembled it.

**How it was found.** By asking what was in the tool list, rather than by
reading the reply again. The reply had already been read three times.

---

## 2026-09-01 — Some failures do not fail

**What happened.** `ch04_missing_tool` asks when the next delivery of milk is expected.
No tool answers that; the toolbox holds `stock_on_hand` and nothing else. The
live model did not refuse and did not invent a tool -- the declaration is
enforced by the model provider before a call exists, so a name we never sent
cannot come back. It called `stock_on_hand("milk")`, got 2, and answered:
"I don't have access to delivery schedules. There are currently 2 units of
milk in stock."

The run cost a turn, a tool call and two round trips. It answered a question
nobody asked and apologised in prose for the one they did.

**Why it matters.** Read the trace as a harness would. Every span opened and
closed. The tool call succeeded and returned a real number. The reply came
back `finish_reason: stop` with no tool calls, so the loop terminated the way
a finished conversation terminates. There is no error, no exception, no retry,
no status field set to anything but success.

A dashboard counting tool calls and completions scores this run as clean. So
would a judge that checks the loop terminated properly, and so would every
assertion in this repository before the ones written for this chapter.

The only evidence that anything went wrong is a sentence of English inside the
final message, and prose is not a field.

**The rule.** Tool coverage is not observable from the trace. A harness can
verify that a call succeeded, that a loop terminated, that every span closed --
and none of that distinguishes an answered question from an unanswerable one.
If coverage matters, something has to compare the question to the toolbox, and
that comparison is a judgement, not a metric.

**Where it applies beyond the primer.** This is the case a per-turn judge
exists for, and it is a sharper argument for one than a wrong answer would be:
a wrong answer at least produces something to disagree with. Here the model
behaved correctly at every step, the harness behaved correctly at every step,
and the user did not get an answer.

**Not reproducible on demand.** One live run in three skipped the tool call
entirely and answered directly. The substitution is a tendency, not a rule --
which means a harness cannot even count on the wasted call being there to see.

---

## 2026-09-01 — LangChain carries reasoning inbound and drops it outbound

**What happened.** `ch04_missing_tool`'s live run failed with a 400 from DeepSeek:

```
The `reasoning_content` in the thinking mode must be passed back to the API.
```

Intermittently: two failures and one success on the same code, while `ch02_tool_call`
passed. The trace has both halves of the explanation in one place.

```
turn 1 reply      carried reasoning_content = True
turn 2 sent back  assistant keys: ['content', 'role', 'tool_calls']
```

`langchain-deepseek` preserves `reasoning_content` **inbound**, into
`additional_kwargs`. `_convert_message_to_dict` in `langchain_openai` then
builds the outbound assistant message from known fields only — content, role,
tool calls — so the reasoning is dropped on the way back. DeepSeek's thinking
mode requires it, and rejects the request without it.

It is intermittent because whether turn one produced any reasoning at all is
the model's choice. A multi-turn tool-calling conversation therefore fails
some of the time, for reasons that have nothing to do with the conversation.

**Why it matters.** This is the same field as the finding at the top of this
file, found the same way, and it contradicts the conclusion drawn there. That
one recorded a mistake: a dependency defect asserted from an absence in our own
instrument, when the adapter had preserved the field correctly and the recorder
had dropped it. This time the defect is real and it is the adapter's — and the
difference between the two is not judgement, it is that both were checked
against the wire.

"LangChain normalizes, and the normalization is lossy in both directions" was
a slogan in CLAUDE.md. It is now a subclass in `support/models.py`.

**The fix.** `ThinkingModel` overrides `_get_request_payload`, pairs the
serialized messages with the ones that went in, and restores
`reasoning_content` on any assistant message that carried it. Three live runs,
no 400. It is a workaround for someone else's defect, so it lives in
`support/`, not in a chapter — a chapter carrying it would be teaching the
shape of a library bug.

**Worth reporting upstream.** Any provider whose API requires reasoning to be
echoed will hit this, and the failure mode is a 400 that appears only when the
model happens to think.

**Amended 2026-09-02: this is not a DeepSeek quirk.** Every provider that
exposes reasoning with tool use requires it back, under a different name, and
several sign or encrypt it so it cannot be fabricated or edited:

| provider | what must go back | if it is dropped |
| --- | --- | --- |
| DeepSeek | `reasoning_content` | 400, as above |
| Anthropic | `thinking` blocks with their `signature` | rejected; signatures are verified |
| Gemini | `thoughtSignature` on the function-call part | multi-turn function calling breaks |
| OpenAI | reasoning items / `encrypted_content`, Responses API only | degraded, not fatal — Chat Completions returns no reasoning to echo |

The shared reason: with tool use the chain of thought is *state for that turn*,
not a byproduct. The model reasoned, asked for a tool, and is resuming.

So a normalizing layer sits between a harness and four different opaque
artifacts, keeping only the fields it recognises. Anthropic is the one to
check first for any harness that uses it: dropping a `signature` is the same
defect with a stricter failure.

*Provider details from knowledge rather than a doc check, and this area moves
quickly. Verify before relying on a specific field name.*

---

## 2026-09-02 — Everything that enters the context is a prompt

*Synthesis. This one was not produced by a run: it generalises two entries
above and one afternoon of reading a dependency's source.*

**What it says.** The model reads one thing — the context — and every field
that contributes to it is a prompt, whatever the protocol calls it. The
`tools` array is not a message and is not called a prompt by anyone, and it is
275 tokens of description and enum values that the model obeys. A tool result
is data until it contains a sentence, and then it is instruction. A library's
error template is text someone else wrote into your context.

**The useful axis is not control.** You control nearly all of it. The axis is
*when it was authored, and who has read it since*:

| authored | examples | last reviewed |
| --- | --- | --- |
| per conversation | system, user | as it is written |
| at design time | tool descriptions, argument enums, error templates, skill bodies | once, months ago — or never, if a dependency wrote it |
| during the run | tool results, retrieved documents, MCP descriptions, web pages | never, by anyone |

**Why the middle class matters most.** It is yours, and nobody looks at it as
text a model obeys. A tool description lives in a docstring, and docstrings are
not reviewed as prompt engineering. That is exactly the `ch02_tool_call`
finding above: `part: str` against `Literal[...]` read as a typing decision and
was the only thing constraining what the model could ask for. LangGraph's
error template is in this class too, and nobody chose it.

**The rule.** Review the context, not the prompt. If a string can reach the
model, it is part of the instruction set, and its review cadence should match
its authorship — design-time text needs a design-time review, and run-time
text needs a guard, because nothing else will ever look at it.

---

## 2026-09-02 — There is exactly one request type

**What it says.** On the wire the model has a single way to ask for anything: a
function call — a name, an id, and a JSON string of arguments. Everything sold
as an agentic capability is that mechanism with different strings in the
dispatch table.

```
retrieval / RAG         a tool called search
handing off to an agent a tool called transfer_to_billing, whose body is a loop
asking the user         a tool called ask_user that blocks on input
structured output       a tool the framework forces and then unwraps
MCP                     a dispatch table populated over a protocol
```

There is no `model_wants_approval` field and no `model_asks_a_question` field.
Other providers spell the same thing differently — Anthropic returns a
`tool_use` block, Gemini a `functionCall` part — and newer APIs add genuinely
distinct kinds: a computer-use call whose arguments are UI actions, and an MCP
approval request that asks permission rather than execution.

**The part that bites a harness.** Provider-executed tools — `web_search`,
`code_interpreter`, Anthropic's `server_tool_use` — appear in the reply
looking exactly like calls and have already run. They are not requests to you.
A dispatch loop that assumes "a call-shaped thing means my table" will try to
execute something the model provider already executed, and a guard written to
gate tool use will not see them at all.

**The rule.** Group by who executes, not by what it is called. Your table
covers one group; a second group is already done when it reaches you; and the
loop's termination condition has to read `finish_reason` rather than assuming
the two groups are one.

---

## 2026-09-02 — LangGraph has no retry, and its default error text asks for one

**What happened.** Read from source while deciding what `ch05_tool_failure`
should do.

`ToolNode`'s default handler is four lines:

```python
def _default_handle_tool_errors(e: Exception) -> str:
    if isinstance(e, ToolInvocationError):
        return e.message
    raise e
```

The model's own mistakes — bad arguments — are reported back so it can correct
itself. Everything else re-raises and the graph stops. That is a real policy
and a defensible one, and its axis is *whose fault*, not *can this succeed
later*.

**Three things follow.**

There is no retry anywhere in it. No count, no backoff, no budget, no
classification. The only retry in a LangGraph agent is the model choosing to
call again after reading an error — uncounted and unbounded.

The default error text is a prompt, and it instructs the model to try again:
`TOOL_CALL_ERROR_TEMPLATE = "Error: {error}\n Please fix your mistakes."`

And it formats with `repr(e)`, so whatever a tool put in its exception message
— a path, a host, a connection string — is sent to the model provider. An
exception is written for a developer reading a stack trace, not for a third
party.

**The rule.** A framework gives you a signal and a sink. Policy — classify,
count, escalate — is yours whether or not you decide it, and the default is a
decision someone else made.

---

## 2026-09-02 — Retry is a matrix, and the question is who can change the outcome

**What it says.** "How many retries" is downstream of a classification nobody
does by default. Retrying only helps if the outcome can change, and who can
change it decides who should retry.

| failure | harness retry | model retry |
| --- | --- | --- |
| rate limit, 5xx, timeout — idempotent tool | backoff, 2-3 | pointless: nothing to fix |
| timeout — non-idempotent tool | **0** — it may have succeeded | 0 |
| bad arguments | 0 — same args, same result | **once** — it can fix them |
| permanent: not found, auth, no capability | 0 | 0 — say so and stop |

**Two cells carry the lesson.** Bad arguments is the only one where the model
retrying is right and the harness retrying is useless — and it is the single
cell LangGraph implements. Non-idempotent timeout is the one that bites: it is
transient by nature and unsafe by consequence, so a budget keyed only on
transient/permanent gets it wrong and charges someone twice.

**Where the marker belongs.** With the tool author, declared like a schema,
because nobody else knows whether a call is safe to repeat. Default to
permanent and opt into retryable: an unclassified failure retried is a loop
spending money on something that will never work, and an unclassified failure
not retried is one wasted call and an honest answer. Wrong in the cheap
direction, which is the same argument as the header allowlist.

**And two outputs, not one.** The harness needs a machine-readable class; the
model needs prose telling it what to do instead. Telling a model "transient"
invites it to retry, which is the harness's decision.

---

## 2026-09-02 — `run` produces evidence, a test produces a verdict

**What it says.** They call the same function. `run(model_kind)` returns a
`Trace`; the runner writes the page and prints a path, and a test asserts
against the same object and writes nothing. The difference is the output: a
document for a person, or a boolean.

**Why it is worth stating.** A test can only confirm what someone already
thought to assert. Every entry in this file came from reading a trace —
`arguments` double-encoded on the wire, 275 tokens of schema, a cache hit of
exactly 6 x 64 blocks, 267ms against 14ms, a model calling a tool it had just
been told was unavailable. Not one of them would have been caught by an
assertion, because you cannot assert a fact you do not have yet.

So `run` is the product and the tests protect it. That is also why the trace
records fields nothing asserts — `reasoning_content`, `refusal`,
`finish_reason`, the defaulted parameters, the raw wire. They are there so the
next finding is findable.

**And the axes are independent.** Mock/live is not run/test:

```
              run                          test
mock    read the mechanics           exact assertions
live    read what really happens     invariants only
```

Three of those four are occupied. The empty one is live tests, which means
`build_live_model`, the wire hooks and every wire note are verified by a person
reading a page — an instrument checked only against itself, which is the first
finding in this file.

---

## 2026-09-02 — Turns are the depth of the question, not a property of the agent

**What happened.** `ch03_the_loop` needed a question the loop could not answer
in two turns. The first attempt was *"we need four milk for the week — if we
are short, what will the rest cost?"* with `stock_on_hand` and `price_of`
declared. The mock, following a script someone wrote, took three turns. The
live model took two: it called **both tools in one reply**.

It was right to. `price_of("milk")` was answerable from the question alone —
the word *milk* was in it — so nothing had to wait. That is fan-out, not a
chain, and the script had encoded a sequence the model did not need.

The question was changed so the second call's *argument* is the first call's
*result*: `lowest_stock_item()` takes no arguments and returns a name;
`price_of` needs that name. Mock and live then agreed, turn for turn.

**The rule.** A call whose arguments are already known goes now. A call whose
arguments come from a result must wait. The model issues everything it can at
once and waits only where it must, so:

    turns ≈ the depth of the data dependencies in the question

The number of tools does not enter into it. Neither does the harness.

**Depth costs; breadth is nearly free.** Each level of depth is another full
context resend, and cost is the sum of prefixes:

| turn | input | output | cached |
| --- | --- | --- | --- |
| 1 | 438 | 66 | 384 |
| 2 | 518 | 50 | 384 |
| 3 | 583 | 17 | 512 |

1539 billed input tokens for a conversation that ends at 583. The same three
calls as fan-out would have been two turns and roughly half the input.

**No framework can fix depth.** `ToolNode` runs a turn's calls concurrently
through `executor.map`, which helps breadth. Nothing parallelises a
dependency — that is what "depends on" means, not an implementation limit. So
a parallel-execution chapter can only ever address half of this.

**Two levers, and neither is the framework.**

*The question.* Naming `milk` collapsed depth 2 to depth 1 by supplying an
argument the model would otherwise have had to fetch. Phrasing that looks like
prompt style is the run's shape.

*The tools.* Two narrow tools force depth 2; one `price_of_lowest_stock_item`
answers in one turn. This is the chatty-API trade, except each round trip
costs a full context resend, so the penalty is far steeper than an HTTP call.
Tool granularity is a latency and cost decision before it is an API-design
one.

**And it is a mock/live finding as much as a design one.** The script asserted
a sequence that was never necessary, and it passed. Only the live column
disagreed — the mock will always confirm whatever plan its author imagined.

---

## 2026-09-02 — An empty tool_calls list is not a claim that anything was done

**What happened.** `ch05_loop_endings` sets `max_tokens: 24` — the first
of the seven CONFIGURABLE parameters this primer has ever set, after five
chapters of reporting all of them as chosen by the model provider.

The reply came back with **empty content, no tool calls, and
`finish_reason: length`**. The budget was spent before the model emitted
anything at all; with thinking on, the reasoning took it.

The loop's condition is `if not reply.tool_calls`. It fired, exactly as
designed, and would have recorded a completed run that produced nothing
whatsoever — and would have been within its rights, because an empty list is
precisely what "the model asked for nothing further" looks like.

**Why it matters.** Termination is inferred from an absence, and three
different things produce the same absence:

| finish_reason | what it means | tool_calls |
| --- | --- | --- |
| `stop` | the model chose to stop | empty |
| `length` | we cut it off | empty |
| `content_filter` | it was refused | empty |

Only the first means finished. The provider states which in the same response,
in a field a loop that infers from `tool_calls` never reads. `ch01_single_call`
noted this as point 4 of its docstring, as a footnote. In a loop it is not a
footnote: it ends runs early and reports them as complete.

**The rule.** Read the provider's own account before trusting your inference.
An inference from absence is the weakest evidence available, and here a
stronger claim is sitting in the same response for free.

**Two things the sharp version teaches that the tidy one would not.** The
first draft of the chapter scripted a sentence cut off mid-word, which is what
truncation looks like when you imagine it. What actually happens at a tight
budget is emptier and worse: nothing at all comes back, and the run looks like
a model that had nothing to say.

And the budget is spent on reasoning before any output exists, so `max_tokens`
does not bound the answer — it bounds the answer *plus the thinking*, and the
thinking goes first.

---

## 2026-09-02 — The same two calls cost 59% more when one waits for the other

**What happened.** `ch06_two_tools` runs one toolbox against two questions.
Both produce exactly two tool calls; only the dependency differs.

| situation | shape | turns | billed input |
| --- | --- | --- | --- |
| both arguments in the question | one reply, two calls | 2 | 1141 |
| second argument from a result | two replies, one call each | 3 | 1813 |

Fifty-nine per cent more input tokens for identical work, measured live.

**Why.** Cost is the sum of prefixes, so every level of depth is another full
context resend — and the resends get more expensive as they go, because the
list has grown by a request and a result each time. Breadth adds a tool span
and a `ToolMessage`; depth adds a whole turn.

**What it confirms.** This is the earlier entry — *turns are the depth of the
question, not a property of the agent* — with a controlled comparison rather
than an accident. The earlier one came from a chapter whose question was
accidentally shallow; this one holds the toolbox fixed and varies only the
question, and gets the same answer.

**And it bounds what a framework can do for you.** `ToolNode` dispatches a
turn's calls through `executor.map`, so a framework can make *breadth*
concurrent. Nothing parallelises a dependency, because that is what "depends
on" means. So the whole of concurrency addresses the cheap half.

**The levers are the question and the tools.** Naming `milk` in the question
supplied an argument the model would otherwise have fetched. A single
`price_of_lowest_stock_item` would collapse the chain to one turn. Tool
granularity is a latency and cost decision before it is an API-design one, and
the penalty per round trip is a full context resend rather than an HTTP call.

---

## 2026-09-02 — A tool that leaves the process brings back four things

**What happened.** `ch07_tool_http` is the first chapter whose tool makes a
real HTTP call. Five situations against httpbin, live:

| situation | tool span | status | outcome |
| --- | --- | --- | --- |
| service_answers | 225ms | 200 | a price |
| service_rate_limits | 226ms | 429 | transient, and `Retry-After` says how long |
| service_is_broken | 199ms | 500 | transient, and it says nothing |
| service_says_no | 200ms | 404 | permanent; retrying is pure cost |
| service_is_slow | 2331ms | — | timeout, and the call may still have landed |

**1. Latency lands somewhere new.** A turn's cost stopped splitting two ways.
`model_provider_ms` and `library_ms` are joined by a service that is nothing to
do with either, and it is the largest number on the row for the timeout case.

**2. Failures arrive with meanings.** These are real status codes, not
exceptions someone invented, and no two want the same treatment. The timeout
is the one with teeth: transient by nature and unsafe by consequence, because
you do not know whether the work happened. No status code tells you that —
only the tool's author knows whether calling twice is safe.

**3. The credential is guarded at one boundary and not the other.** The
request recorder refuses `Authorization` by allowlist, and that held:

    request   authorization  None

httpbin echoes request headers in its *response body*, which is recorded
whole:

    response  "Authorization": "Bearer primer-demo-key"

So the value withheld at one end arrived at the other, and every check in this
repository still passed. The key here is fabricated, which is the only reason
this is safe to demonstrate; a real one would be sitting in `out/*.json` in
plaintext. **Redaction is per-boundary, and a boundary nobody thought about is
not redacted.**

**4. The model retried a failure that can never succeed.** Told the 404 had
failed and to answer without the tool, the live run called it again anyway —
eleven turns against the mock's ten. It received a sentence, and trying again
is a reasonable inference from a sentence. Which is why classification belongs
to the tool and the counter belongs outside the loop: neither is something a
model can be persuaded into.

**And two bugs in our own recorder, found by the first non-DeepSeek response.**
It assumed every response body was JSON, so a 429 with an empty body crashed
the run — a recorder that only works on the happy path is not a recorder. And
`httpx.codes.OK` types as a `(200, 'OK')` tuple, so the status comparison was
confusing at best; `response.is_success` is what httpx actually offers.
basedpyright caught the second, mypy passed it, which is now the third time.

---

## 2026-09-02 — The argument schema is a shell-injection defence, by accident

**What happened.** `ch08_tool_program` runs a program from a command line built
out of the model's argument, unquoted, deliberately. The situation
`argument_is_a_command` asks for an item named `milk; echo pwned`, arriving
the way it does in practice — as upstream text the model passes through, not
as something a model invents.

The mock column runs it:

    $ printf 1.20 --item milk; echo pwned
    out: '1.20\npwned'

The live column refuses:

> "the price tool only accepts the exact catalog items `bread`, `butter`,
> `chips`, or `milk`. The string `milk; echo pwned` isn't a valid item"

The parameter is `Literal["bread", "butter", "chips", "milk"]`, so it reaches
the model provider as an `enum`. **The schema stopped it.**

**Why it matters.** This is the `ch02_tool_call` finding arriving somewhere
else entirely. There, `part: str` against `Literal[...]` looked like a typing
decision, and the cost of getting it wrong was a fluent wrong answer. Here the
same decision is the difference between a tool and a shell.

Nobody writes an argument type as a security control, and in this case it is
one. Which also means the defence is accidental and therefore fragile: it
holds only for arguments that happen to be enumerable.

**And that is the common case going the other way.** A tool whose argument is
a path, a filename, a search query or a customer's name has no enum to hide
behind. Then the only thing between a model's output and a shell is quoting,
and nothing in the loop, the tool or the recorder will tell you it is missing:
the mock column shows the command running, the exit code is 0, the result is
recorded faithfully, and every check in this repository passes.

**What a process boundary gives back, compared with the other two.**

| boundary | vocabulary | on a timeout |
| --- | --- | --- |
| in-process | exceptions, with types and messages | n/a |
| network | status codes with agreed meanings, `Retry-After` | the request may never have been received |
| process | an exit code, and prose on stderr if you are lucky | the process certainly ran, and may have finished |

`404` says the thing is not there. `1` says the program was unhappy. A killed
process is the worst of the three to reason about: it started, so a timeout on
a program that changes anything is not safe to retry, and nothing in the
outcome says so.

---

## 2026-09-02 — A tool cannot enforce a limit, because it never sees the sequence

*From the live runs of `who_retries`, whose mock column scripted the wrong
answer and was believed until the model disagreed with it. Reproduced: two
live runs, same split, same total.*

The chapter's third cell is "bad arguments: the model retries, once". The tool
rejects `quantity=20` against a bound of 12, the harness does not retry —
retrying an identical call would produce an identical rejection — and the
sentence handed back invites the model to correct the arguments.

The mock model corrects them, to 12, and says so. **The live model ordered 20.**

> turn 2 · asked for: place_order, place_order
>   place_order(item=milk, quantity=12) → ordered 12 x milk
>   place_order(item=milk, quantity=8)  → ordered 8 x milk
> turn 3 · "Ordered 20 bottles of milk total (12 + 8), as the per-order
>           maximum is 12."

**Nothing malfunctioned.** The error message named the bound, which is what
made it fixable. The tool enforced the bound. Both calls were inside it. The
harness dispatched what it was asked for, as it does. And the user's stated
intent — twenty bottles — was satisfied exactly, by a model that treated the
limit as a fact about calls rather than a fact about orders.

**Which it is.** `if not 1 <= quantity <= MAX_PER_ORDER` is a bound on one
invocation. A tool is called with arguments and returns a result; it does not
know it was called before, is about to be called again, or is one of two calls
in the same reply. It cannot enforce a total, a rate, a budget or a quota,
because none of those are properties of a call — they are properties of a
sequence, and the sequence is the harness's.

**The idempotence claim is narrower than it reads.** `place_order` is declared
not idempotent, and this chapter spends its argument on what that forbids: the
harness must not retry a call that may already have run. It forbids exactly
that and nothing else. The model called the non-idempotent tool twice, in one
reply, deliberately — and there was no rule to break, because the rule was
about retries and this was not a retry.

**The general form.** Three parties can produce a second call, and they are
governed in three different places:

| who calls again | why | what stops it |
| --- | --- | --- |
| the harness | the failure was transient | `retryable and idempotent` |
| the model | the sentence invited it | nothing, before `guards` |
| the model, differently | it inferred a way around | nothing, and no sentence would |

Only the first is retry. The other two are the model doing what it is for, and
a matrix organised around "who retries" quietly assumes the second row is a
repeat of the same call. It need not be.

**What this costs the reader who does not notice.** A tool author writes a
bound, tests it, sees it reject, and believes the system is bounded. It is
bounded per call. Every quota in a harness — spend, rate, rows written, emails
sent — has this shape, and every one of them is safe only if it is counted
somewhere that outlives a single invocation. That place is the loop, which is
the argument for `guards`: a check before dispatch, holding state across calls,
able to say no to the second one.

**And a note on mocks, again.** The mock column asserted a scripted correction
and passed. It was not wrong about the mechanism — the harness really does
behave that way — it was wrong about the model, which is the one thing a
scripted model can never be right about by construction. This is the second
time the live column has contradicted a scripted assumption about model
behaviour, after `tool_program`'s injection, and both times the live answer was
the more interesting one.

---

## 2026-09-03 — LangGraph's conditional edge is a checkpoint function, not a new primitive

*Synthesis, from working through `ch10_routing` and then asking what
LangGraph's own vocabulary was hiding.*

`ch10_routing` needed three `if`s in three different places -- after the
reply, before the model, on a turn counter already carried. Mapping each one
onto LangGraph turned into a rabbit hole of terms -- node, edge, conditional
edge, entry point, `path_map`, `BranchSpec` -- that felt like separate
machinery. It is one idea, worn three ways.

**A node is a function. An edge is "what runs next."** `add_node("model",
ask_model)` is naming a function; the string and the function are the same
thing seen from two sides -- the name is what an edge can point at, the
function is what actually runs when it does. `add_edge("tools", "model")` is
a fixed answer to "what runs next": always this, decided when the graph was
built.

**A conditional edge is the same slot, with the fixed answer replaced by a
function call.** `add_conditional_edges("model", route, path_map)` stores a
function plus a dict, attached to the `"model"` node. When `"model"` finishes,
the function runs against whatever is around -- the reply, in this case --
and returns a key; the dict turns that key into a real destination. Nothing
about this is a different kind of edge under the hood: it is the same "what
runs next" question, answered at run time instead of at build time.

**The map is not capped at two.** `tools_condition` only ever returns
`"tools"` or `"__end__"`, so its map only needs two entries -- which makes it
easy to mistake for the shape of the mechanism rather than one instance of
it. `route()` can return as many distinct answers as the logic needs, and the
map just grows to match: `{"done": "__end__", "confirm": "confirm_node",
"dispatch": "tools"}` is exactly as valid as a two-entry map, same call, same
storage.

**`set_conditional_entry_point` is the identical thing at a different
attachment point.** Not after a node -- before the graph's first node runs at
all. Same shape, function plus map; the only difference is where it is
registered. This is `ch10`'s `the_question_needs_no_model`: the function
decides before `"model"` is ever entered, so `ask_model` never runs, and
there is no reply to have branched on in the first place.

**The plain-language version, arrived at last and worth keeping over the
vocabulary:** a conditional edge is a checkpoint function dropped at a named
spot -- after a node, before one, or nowhere in particular, just on state the
loop is carrying. It looks at whatever is around, returns a name, and the
name says what runs next. Everything else -- "edge," "branch," "entry
point" -- is packaging around that one idea, repeated at three attachment
points.

**What this buys over writing the `if` by hand, and what it costs.** The `if`
becomes data the engine can inspect -- drawable (`get_graph().draw_mermaid()`),
and destinations decouple from callers, who no longer need to know which
function handles `"tools"`, only that something registered under that name
does. The cost is that control flow stops being readable top to bottom in one
file: `ch10`'s three `if`s sit exactly where they take effect; the graph
version scatters the same three decisions across `add_node` calls, a
`path_map`, and whatever `route()` looks like, none of which has to be near
the others. `graph` is where this primer puts a number on that trade rather
than asserting it.

---

## 2026-09-03 — A live model batches what a script spaced out on purpose

*From the live run of `ch11_state`, and the third time the live column has
declined to cooperate with a scenario's script -- after `tool_program`'s
injection and `who_retries`' split order.*

`a_running_total_survives_between_calls` is scripted as two turns, one
`place_order` in each, because that shape is what proves the lesson: a value
that outlives a single turn, which a local scoped to one turn's dispatch loop
cannot do. The live model put both calls in one reply instead -- the
`ch06_two_tools` fan-out shape, not the scripted sequence -- and answered in
one turn.

`state.total_ordered` still came out right, 5 + 3 = 8, because
`increment_order_total` runs once per call and does not care how many turns
they are spread across. Nothing broke. But this particular run does not
demonstrate what the scenario exists to demonstrate: it never needed a value
that survives between turns, because the model never spread its calls across
more than one.

**The general shape, now three times over.** A scripted model proves the
sharp version of a claim because it is told to behave the sharp way. A live
model proves whatever it actually does, and what it actually does is smarter
than the script whenever a shortcut is available: `tool_program` found the
model refusing an injected argument the schema happened to block;
`who_retries` found it splitting a rejected order into two calls that
together satisfied the user's intent; here it found the model skipping the
very turn boundary the scenario was built to require. In every case the
model was not wrong -- it did what a competent agent should do -- and in
every case the mock's scripted shape was the only reason the lesson was
visible at all.

**What this means for a scenario, going forward.** A mock scenario proves a
mechanism can do what the docstring claims. It does not prove a live model
will exercise that mechanism the way the scenario expects, and a chapter
whose claim depends on the model behaving one specific way should say so --
this one now does, in its own docstring -- rather than let a live run that
takes a smarter path read as a contradiction.

---

## 2026-09-03 — A model cannot call another agent, only ever a tool

*Synthesis, from building `ch12_supervisor` and the conversation that
followed it -- the same finding as `ch02_tool_call`'s, proven true one level
up rather than discovered anew.*

`ch02_tool_call` established this at the first level: the model emits a name
and some arguments, never executes anything, never knows what is behind the
name. `ch12_supervisor` is the same fact one level up. From the model's side,
`consult_inventory_expert` is indistinguishable from `price_of` -- a name in
its toolbox, nothing more. It has no concept of "delegating to another
agent," because the model was never given that concept; it only has
`tool_calls`, and `tool_calls` does not know what a tool's body contains.

So "agents calling agents" is not a new capability the model gained. It is
the same one-line mechanism from `ch02` -- the model asks, we decide what
runs -- with our decision, this once, being "run a whole other `run_turns`."
Nothing about the model's side of the exchange changed at all, which is why
`ch12` needed no new mechanism: `consult_inventory_expert` is declared the
same way every tool since `ch04_tool_failures` has been, and `Trace.span`'s
existing stack nests the inner agent's spans for free, because it does not
know or care that this call's implementation happens to be another loop.

**The corollary, worth keeping distinct from this.** A supervisor is defined
by having *at least one* tool whose implementation is another agent -- not by
every tool being one. `ch12`'s two tools both happen to be sub-agents because
that made the clean example; nothing about the pattern requires it. And a
supervisor still needs its own model call to decide *whether* and *which*
sub-agent to invoke -- a fixed list of agents run in a fixed order, with
nothing deciding, is `ch13_workflow`, not a supervisor.

---

## 2026-09-03 — A stream's granularity belongs to whoever produces it

*Sanjeev's, from building and running `stream` -- the consuming half -- and
working out what it does and does not tell you about the producing half.*

`stream`'s live runs came back with 129 chunks one time and 54 another, for
answers of similar length. Nothing in the request asks for a chunk size --
`"stream": true` is a boolean, and everything about how finely the reply gets
sliced is the provider's choice. A consumer takes what it is given and
accumulates.

That asymmetry inverts when the harness is the producer. Pushing progress to
a user of the harness is a connection we own at both ends, so granularity
stops being something to absorb and becomes something to design: run,
scenario, turn, span, or token -- each a different answer to "how much does
the person watching want to see."

Exposing that as a user-facing control -- a "thinking" knob, showing more or
less of what the harness is doing -- only works if the levels were designed in
from the start, because they are a filter over which span-closes get
forwarded, and a filter needs something structured to filter.

**The mistake to avoid is assuming the consuming side taught you anything
about the producing side.** One is a firehose you accumulate; the other is a
decision about what someone should see.

---

## 2026-09-03 — A reply can be a message and a tool request at once

*Sanjeev's, from asking to see content and a tool call in the same response
on the actual page, then asking whether reasoning ever overlapped either.*

Confirmed directly, not assumed: a live call returned `content="I'll help you
find the price of milk -- let me look that up for you now."` and
`tool_calls=[price_of(item="milk")]` on the same `AIMessage`. Every mock
script in this primer, since `ch04_tool_failures`, has written tool-call
replies with `content=""` -- a convention that quietly became an assumption
nobody had tested against a real model.

**One frame never mixes them.** In a streamed trace built to check this, each
`delta` carried exactly one of `reasoning_content`, `content`, or
`tool_calls` -- never two at once. But a *reply*, accumulated across many
frames, can and does end up with both: 31 reasoning chunks, then 10 content
chunks, then 11 tool-call chunks, cleanly separated in arrival order, merging
into one message with both fields non-empty.

**And every `execute_tool` in this codebase discards the content half.** It
reads `call["args"]`, never `reply.content` -- so the model's own
explanation, sitting right beside the call it is explaining, has been
silently thrown away in every chapter since `ch04`, with nothing ever
surfacing that it was happening.

---

## 2026-09-03 — A hook is a pre-registered rule at a named point, for a specific actor

*Sanjeev's, from building `ch18_hooks` and then asking whether `guards`,
`loop_veto` and `judge` are anything more than this.*

Three powers, in order of danger: observe (reads, changes nothing), modify
(reads, returns a replacement), veto (reads, can refuse). Not every point
gets every power for free -- `pre` and `post` do not offer the same veto.
Before an action, nothing has happened, so a veto refuses the action itself.
After, the action already occurred; a veto there can only stop its
*consequence* -- the result never reaching the message list, the loop ending
instead of continuing on it. Same word, two different things, depending on
which side of the action the hook sits.

**And a rule only proves it is real once more than one is attached to the
same point.** One hook per point cannot show whether hooks see the original
input or each other's output. Chaining observe -> modify -> veto on one call
showed the veto firing on the *normalized* value, not the one the model
sent -- proof that hooks are a pipeline, not a set of independent checks
running against the same untouched input.

**The actors it attaches to are not a fixed list of five.** Tool, model,
agent and step are the primitives with real pre/post moments. Supervisor and
workflow are shapes built from those four, not additional kinds needing their
own hook category -- confirmed against this primer's own earlier findings: a
supervisor is an ordinary agent nested in a tool, and a workflow does not
act, a step does.

**`guards` and `judge`, checked against this, are mostly scenarios of it.**
`guards`' mechanism -- a pre-hook with veto power -- is fully built here;
what it adds is a real cross-call condition (`ch11`'s `state.total_ordered`)
in place of the single-call catalog check this chapter kept deliberately
simple. `judge` is the same shape at a different actor -- `model`'s post
hook, reading a reply instead of a tool's result -- which this chapter's own
scope explicitly deferred. Neither needs new machinery.

**`loop_veto` is the one exception, and the gap is real.** This chapter has
no loop -- every scenario is one isolated call, no `run_turns`, no model, no
multiple turns. A veto here returns an error and stops there; nothing has
ever tested what happens *next* -- whether the model sees the refusal and
tries something else, or the run ends immediately. Every ending this primer
has built so far is "the model asked for nothing" or "we hit a cap."
"Stopped because forbidden" is a third kind that does not exist anywhere yet,
and it needs an actual multi-turn loop to be a question at all.

**This is the same shape as `supervisor`/`workflow` one level up.** Those are
configurations of `agent`/`tool`/`step`, not new primitives -- a supervisor is
an agent nested in a tool, a workflow is a step wrapping whatever is inside.
`guards`/`judge` are the identical move at the hook layer: specific, named
conditions built from `pre`/`post` x {observe, modify, veto}, not new powers
or new points. The ladder has now made this move twice -- primitives first,
then named patterns that turn out to be configurations of them, not
additions to them.

---

## 2026-09-03 — The hook is the guarantee; the model's manners are not

*Sanjeev's, from the live run of `loop_veto`, where the model asked
permission instead of routing around a refusal the way `ch09`'s did.*

`ch20`'s live model stopped after the refusal and asked permission --
*"Would you like me to order 4 more instead?"* -- rather than quietly
retrying with a smaller number the way `ch09`'s live model split 20 into
12+8 without asking anyone. Same shape of situation, same kind of model,
opposite choice. Nothing distinguishes in advance which behavior a given run
will produce.

**That is the case for the hook being belt-and-suspenders, not redundant.**
If the model always asked first, the hook would just be enforcing what good
behavior already provides. If the model always routed around silently, the
hook would be the only thing stopping it. This session proved both happen
from the same setup -- so the hook has to hold regardless of which one shows
up, and it did: the guard recomputes the real total on every attempt,
whether that attempt arrives as a polite question or a silent retry. The
model's manners are not the control. The hook is.

---

## 2026-09-03 — Everything sent to the model is a prompt

*Sanjeev's, from `prompt_types`, in his own words: "everything sent to the
model, whether a human prompt, list of tools, outputs from tools, outputs
from model... literally anything, including request for tools from the
model, is a prompt."*

There is no privileged field. `system` and `human` feel like "the prompt"
because they are the one row someone reviews every time it changes, but a
tool's declared description, its argument enum, a tool's result, and the
model's own prior reply all land in the same list and get read the same
way -- the model does not know or care which field carried a given piece of
text into its context. Calling only `system`/`human` "the prompt" is a
convention of who wrote it and when, not a fact about what the model does
with it.

**The useful split is not "do we control it" -- almost all of it, we do --
it's when it was authored and who has reviewed it since.** Per-conversation
(written now, read now), design-time (written once, reviewed rarely or
never), during-the-run (produced by this run's own calls, reviewed by
nobody, ever -- not even the model, which has already moved on by the time
its own prior reply comes back as context). The middle row is the one that
surprises people, because a tool's docstring lives in code, not in a prompt
file, and so it never gets audited as one -- which is exactly `ch02`'s old
finding (`part: str` vs. `Literal[...]`) wearing a different name.

---

## 2026-09-03 — MCP is more than a tool list and a call

*Sanjeev's, from `mcp`, in his own words: "mcp is more than just a tool
list and call; resource get's remote content to the model; prompts are
the same, injected from remote."*

Easy to hear "MCP" and think "remote tools" and stop there -- `tools/list`
and `tools/call` are the familiar shape, the one every chapter since `ch02`
already has a mental slot for. Resources and prompts are not a smaller
version of the same idea; they are content from somewhere else landing in
the model's context with no round trip and no model involved in the
decision at all. A resource is remote content the *client* fetched and
handed over. A prompt is a remote-defined conversation the client seeded
before turn one. Both are injected in the plain sense of the word -- text
placed into context by something other than the person or model currently
in the loop -- and neither leaves a mark saying so once it's there.

---

## 2026-09-03 — Injection changes behavior without consent

*Sanjeev's, from `mcp_injection`, in his own words: "my first exposure to
how injection can change the behavior w/o consent, and how to approach
it."*

`ch24`'s live model appended a marker string to an unrelated "say hello"
reply, having never called the tool whose description carried the
instruction. No one asked it to. No one told it to. The text was declared
alongside a question about something else entirely, and the model acted on
it anyway -- consent was never in the loop, because nothing about a tool's
`description` field looks like a place instructions could come from. The
approach that actually holds isn't "detect bad text" in general -- that's
unbounded, someone always phrases it differently -- it's moving the guard
to the earliest point the hostile content exists as data and has not yet
become something the model reads: before declaration, not before dispatch.
By the time a `tool_call` exists, the description already did its work.

---

## 2026-09-03 — Retrieved content joins the message the same way a prompt does

*Sanjeev's, from `rag_injection`, in his own words: "I've always wondered
how content makes it into the model, turns out, it's literally the same
way as the user prompt, maybe with an indicator that it is a reference
content."*

No indicator, confirmed directly across every chapter that spliced
something in -- a resource's text, a retrieved document, all landed as a
plain `HumanMessage`, same `content` field a typed question uses, nothing
marking it as fetched rather than authored. The one real exception is
native multimodal content: an image or PDF sent as actual bytes carries a
typed content block (`type: "image_url"`, etc.) that routes it to a
different encoder -- but that tag means "decode this differently," not
"trust this less." Text extracted *from* an image or PDF (OCR, transcription)
collapses right back into the no-indicator case, which is the common one:
almost every real RAG and document pipeline extracts to text first.

**Confirmed live, not just described, once both media types were actually
sent.** An image used `type: "image_url"`; a PDF and a DOCX both use the
same generic `type: "file"` block, distinguished only by the MIME type
inside the data URI -- no separate "document" type exists. Proven the hard
way: `type: "document"` and `type: "input_file"` both came back a `400`
naming exactly what the schema accepts -- `text`, `image_url`, `file`, and
nothing else. The indicator isn't a menu the model reads; it's a closed
contract the provider enforces before the request is even accepted.

---

## 2026-09-03 — Skills work because the model is smart enough to follow them

*Sanjeev's, from `skills`, in his own words: "skills are tools that return
a set of instructions that the model is intelligent enough to follow."*

That capability is exactly what makes `ch24`'s injection and `ch26`'s
skill the same mechanism aimed in different directions -- and this
chapter's own live testing showed the model is more capable of following
an instruction than the design assumed. `ch26` scripted a minimal pair
expecting a command to be followed and a description not to be; live, the
description was followed seven times out of eight anyway, because the
model didn't just parse grammar, it noticed a stated pattern about
"outputs" sitting next to its own output and completed it. The same
intelligence that makes a skill usable on the strength of plain English
instructions is what makes an unlabeled, undefended piece of text from
anywhere just as persuasive.

---

## 2026-09-04 — Compression is a method, not a black box

*Sanjeev's, from `compression`, in his own words: "I thought of
compression as some blackbox full of magic, and what I am learning is
there's a clearly defined method to this witchcraft, and it makes so much
sense."*

Ten scenarios, and every one of them was a concrete, nameable decision --
truncate by position (and break something), truncate by whole turns
instead, target by role, target by age, summarize and pay for it in a
model call, protect what's pinned, filter what's declared. None of it
required guessing at what a provider does internally. The "magic" was
always a specific choice about what to drop and when, checkable the same
way everything else in this primer is checkable -- by asking the model a
question only the dropped thing could answer, and seeing whether it still
can.

---

## 2026-09-04 — Caching is token-by-token, in complete blocks, prefix-based

*Sanjeev's, from `caching`, in his own words (refined together): "Caching
works token by token, in complete blocks, prefix-based -- not character by
character. A change within or before the last fully-cached block breaks
everything back to zero, from compression or an intentional edit alike. A
change after that boundary costs nothing, since that part was never
cached at all."*

Confirmed with real numbers, not the exact byte count: a 211-token
identical repeat reported `cache_read: 128`, not 211 -- the true matching
prefix, rounded down to a complete 128-token block. A 339-token
conversation cleared two blocks (`cache_read: 256`). Editing the very
first message broke the match back to `0`; editing the second-to-last
message left `cache_read` at `256`, completely unchanged, because that
trailing stretch was never part of any cached block to begin with, edited
or not.

---

## 2026-09-04 — Memory is just text content, and the role is your call

*Sanjeev's, from discussing `memory` before building it, in his own words:
"is it just another content of type=text, and role=user?" and, confirming
the role isn't fixed either -- "you mean I get to define system/user/
assistant or whatever?" -- "and this is precisely my learning."*

A recalled memory fact has no dedicated slot. It is `type: "text"`, the
same as everything else `ch25` already proved the schema is closed to
(`text`, `image_url`, `file` -- nothing else exists, confirmed by a real
`400`). And the role it rides in -- `system`, framed as background the
assistant already knows, or `user`, framed as something just restated, or
even a fabricated prior `AIMessage`/`ToolMessage` pretending a lookup just
happened -- is entirely the harness's own choice, the same choice `ch23`'s
resource splice and `ch26`'s skill loader already made explicitly. The API
has no opinion. "Memory" is not a mechanism the model or the protocol
provides; it is a decision the harness makes about which role to hand a
piece of text to, dressed up in a bigger name.

---

## 2026-09-04 — Each role means something different to the model

*Sanjeev's, from `roles`, in his own words: "each role type is different,
with different meaning, and different responsibilities that is
interpreted by the model."*

True even though `ch30`'s own live numbers showed no compliance
difference across them -- fifteen out of fifteen, one clear imperative,
every role. What differs is the *mechanism*, not necessarily the
outcome: `system` carries documented authority, `user` is the direct
request, a fabricated `assistant` turn works through self-consistency --
the model not contradicting a prior turn attributed to itself, a wholly
different lever than being told or asked. Three different reasons the
model went along with it, converging on the same answer because the
instruction was unambiguous. `ch26`'s own finding -- that even
declarative, non-commanding phrasing got imitated seven times out of
eight -- is the hint that these mechanisms would likely separate for a
weaker or more ambiguous instruction, where only some of the three
reasons would still carry it.

---

## 2026-09-04 — Resume is a deserialize step, not an operation the API knows about

*Sanjeev's, from `stop_resume`, in his own words, confirming my own
description back as the learning itself: "'resume' isn't a special
operation the model or the API knows about -- it's just: turn the last
saved line back into real message objects, and keep calling ask_model
like nothing happened. The illusion of continuity lives entirely in that
deserialize step."*

Nothing about `ask_model` changes, nothing about the request changes,
nothing on the wire says "this conversation was interrupted." A resumed
run and an uninterrupted one produce the identical request shape once
`deserialize_message` has turned the last jsonl line's plain dicts back
into real `SystemMessage`/`HumanMessage`/`AIMessage`/`ToolMessage`
objects. The API has no concept of resumption to support, because from
its side there is nothing to resume -- just a list of messages, same as
every call since `ch01`. The entire mechanism lives on this side of the
wire, in the two functions that convert between text on disk and objects
in memory.

---

## 2026-09-04 — LangGraph is a framework for calling what we give it, plus housekeeping the primer had already built

*Sanjeev's, from `graph`, in his own words: "LG is just a framework for
calling what we give it, and to do some housekeeping which the primer has
already done."*

`StateGraph` runs functions we wrote, on an order we declared with
`add_edge`/`add_conditional_edges`, and merges each one's return dict into
`state` by key, using a reducer we chose (`add_messages`, or the default
overwrite). Nothing under `state["messages"]` is LangGraph's -- the name is
ours, the contents are whatever `call_model` and `call_tool` return, and
the appending-not-replacing behavior only happens because `ch32` annotated
that field with `add_messages`. Every piece of that housekeeping --
appending results to a running list, deciding whether to loop again,
carrying tool declarations alongside but outside the message list -- is
something the primer built by hand first, from `ch01` through `ch11`.
`graph` doesn't introduce a new capability; it re-implements the same
loop's bookkeeping as a schedule of node calls and dict merges, which is
exactly why it was answerable by comparing it against a hand-built version
that already existed.

---

## 2026-09-04 — A super-step is a single execution unit, and recursion_limit counts units, not node calls

*Sanjeev's, from a conversation about `limits`, in his own words: a
super-step is "a single execution unit" -- and by his own gloss, that
phrasing already carries the parallel case, since to him "exec unit"
means one scheduled round whether it holds one node or several running
together.*

`recursion_limit` is a misnomer -- there's no call stack anywhere in a
compiled graph, just a step counter borrowed from Pregel/bulk-synchronous-
parallel terminology, incremented once per super-step and checked against
the limit each time. In `ch32`'s graphs, with no fan-out anywhere, a
super-step and a single node execution are the same thing, so the counter
happens to equal "how many node calls has this run made." That equivalence
breaks the moment nodes fan out from the same predecessor (`ch37_parallel`,
via `Send`): several nodes can run within the same synchronized round, and
the counter still only advances by one for the whole round, not once per
node inside it.

The same logic extends across a graph boundary, not just within one round.
If a node's body is itself a compiled subgraph (`ch36_subgraph`), the
child graph's own super-steps are counted against the *parent's*
`recursion_limit`, not a separate budget of their own -- unless the child
is invoked as its own independent `.invoke()` call, which is precisely the
"isolated worker with its own context" distinction `supervisor` already
drew against `subgraph`'s "call-and-return, one shared limit." A subgraph
nested by composition spends the parent's recursion budget; a subgraph
invoked as a separate run spends its own.

---

## 2026-09-04 — LangGraph's storage is swappable

*Sanjeev's, from `checkpoint`: "LangGraph's swappable storage."*

`InMemorySaver`, `SqliteSaver`, `PostgresSaver` all implement the same
contract -- `get_state`/`update_state`/commit-on-node-return -- and only
`graph.compile(checkpointer=...)` changes to swap between them. Proven
directly, not assumed: bolting our own jsonl persistence onto
`InMemorySaver` -- reading `get_state().values`, writing it out with
`ch31`'s own message format, then seeding a brand-new checkpointer via
`update_state()` before its first `invoke()` -- worked without touching
`thread_id`, resume, or any node logic. The checkpointer is a slot; what's
plugged into it never leaks into the graph's own code.

---

## 2026-09-04 — StateGraph is a general workflow engine, not an agent framework

*Sanjeev's, from `subgraph`: "LG StateGraph is going to be a major player
in workflows. Period."*

Nothing proven since `graph` -- schema-mapping between subgraphs, swappable
checkpoint storage, a step-counted recursion boundary, automatic field
merging by declared schema -- is agentic in nature; every one of those is
generic graph/workflow machinery wearing this ladder's vocabulary. That
generality is exactly why it scales past the agent use case: the same
primitives that build a model-calling loop build any node-and-edge
workflow with typed state, checkpointing, and composition, which is the
argument for `StateGraph` becoming a major player in workflow
orchestration generally, not just LLM agents.

---

## 2026-09-04 — A reducer's location is the schema, not the node

*Sanjeev's, from `reducers`: "the location of a reducer and custom logic
is key within the graph."*

`Annotated[T, reducer]` is declared once, at the schema level -- every
node that touches that field inherits the same merge behavior
automatically, with no per-node choice and no way to opt out locally.
`ch38`'s `left`/`right` nodes never coordinate or know about each other;
the merge logic that resolves their simultaneous writes lives entirely in
`FanState`'s declaration, not in either node's body. Put custom logic
(`operator.add`, or anything else) in the wrong place -- inside a node
instead of the schema -- and it only governs that one node's return, not
what happens when two different nodes write the same field.

---

## 2026-09-06 — LangChain's callbacks cover the common cases, not every need

*Sanjeev's, from `callbacks`: "LG callbacks are a good set of pre/post
hooks, but may require custom ones also depending on my needs."*

The built-in surface (`on_chain_start/end`, `on_tool_start/end`,
`on_chat_model_start`, `on_llm_end`, `on_tool_error`, etc.) covers every
standard lifecycle boundary, and any `BaseCallbackHandler` subclass plugs
into all of them via one `config={"callbacks": [...]}` list -- no
framework code to modify. But it's a fixed menu: it can only observe an
event that already exists, and it only has two of `ch18`'s three hook
powers (observe, abort via `raise_error`) -- never modify. Any need past
"watch these fixed points and possibly bail out" -- reshaping data in
flight, adding a new kind of checkpoint, anything
`HookVerdict.replacement`-shaped -- has to be built as custom logic in the
graph itself, not as a callback.

---

## 2026-09-06 — interrupt() adds scheduling markers to the checkpoint, not just data

*Sanjeev's, from `interrupt`: "an interrupt adds additional markers in
the checkpoint so a resume goes directly to the next step."*

A checkpoint was always a snapshot of state (`ch34`'s subject);
`interrupt()` additionally records the pending-task queue (`snap.next`,
`snap.tasks`) -- which node was mid-execution when it paused -- and the
specific `Interrupt` object(s) that node raised, each with its own `id`.
Resuming reads that queue directly and dispatches straight to the pending
node; it never re-derives "where to go" from `START`'s edges, the same
way an OS scheduler reads a process table rather than replaying a process
from its source. Completed nodes are simply absent from the pending
list, so they never re-run; only whatever node was in-flight does, from
its own top -- anything in that node before the `interrupt()` call
executes twice on resume.
