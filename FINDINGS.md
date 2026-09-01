# Findings

Things learned by running the chapters that outlive the chapter that produced
them. Chapter-specific observations belong in that chapter's docstring; this
file is for what generalizes.

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

**Where it applies beyond the primer.** `src/audit/callbacks.py` records
`gen_ai.*` on model events and was written against the same assumption about
what an `AIMessage` carries. Unverified as of this date. Reasoning tokens are
billed and may be unaccounted for.
