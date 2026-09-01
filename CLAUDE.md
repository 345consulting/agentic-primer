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
