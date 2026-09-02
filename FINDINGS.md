# Findings

Things learned by running the chapters that outlive the chapter that produced
them. Chapters are named, never numbered, because a finding outlives the
reading order and the numbers move. Chapter-specific observations belong in
that chapter's docstring; this file is for what generalizes.

Append, never rewrite. Date each entry.

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

**What happened.** An earlier draft of `ch03_missing_tool` declared a tool it could not
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

**What happened.** `ch03_missing_tool` asks when the next delivery of milk is expected.
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

**What happened.** `ch03_missing_tool`'s live run failed with a 400 from DeepSeek:

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
