# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The record of a run: what was entered, what was exited, and why.

Scaffolding, never a lesson. No chapter needs you to read this file to be
understood -- it exists so that nine chapters do not each carry sixty lines
of rendering. Nothing here may import from `chapters`.
"""

from support.chapters import source_hash

from collections.abc import Generator, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import current_thread
from typing import Any, Literal, get_args

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

type Json = dict[str, Any]

# Mock or live. The distinction originates at the model -- `models.model()` is
# the one switch -- and a run is mock or live because its model was. It lives
# here because the trace records it and both other support modules already
# depend on this one; declaring it twice is how the page and the seam drift.
type ModelKind = Literal["mock", "live"]

MODEL_KINDS: tuple[ModelKind, ...] = get_args(ModelKind.__value__)

# Note labels, named rather than written out at each site. The recorder writes
# them, the page dispatches on them, and `Span.model_provider_ms` matches two of them
# across a module boundary -- so a renamed literal would not fail, it would
# quietly stop finding the notes and drop the model provider/library split off the
# page. Chapter-authored labels stay literals in their chapter: those are the
# lesson, and nothing else reads them.
CHUNK = "chunk"

CONTEXT = "context"

ENDED = "ended"

REPLY = "reply"

WIRE_FRAME = "wire frame"

WIRE_REQUEST = "wire request"

WIRE_RESPONSE = "wire response"


@dataclass
class Note:
    """Something worth recording that is not itself an entry or an exit."""

    label: str
    payload: Json


@dataclass
class Span:
    """One entered-and-exited thing. Every enter has a matching exit."""

    name: str
    attributes: Json
    entered_at: datetime
    # Sequence and thread are recorded separately from nesting on purpose. A
    # tree says what contains what; it cannot say what ran beside what. From
    # the chapter where tool calls run on a pool, that difference is the whole
    # lesson, and inferring it from timestamps is guesswork.
    seq: int = 0
    thread: str = ""
    exited_at: datetime | None = None
    notes: list[Note] = field(default_factory=list)
    children: list[Span] = field(default_factory=list)
    # `notes` and `children` are each correctly ordered among themselves, but
    # separately -- a note added between two child spans has no way to say
    # so. `_entries` is the one true recorded order, notes and children
    # mixed, and it exists because `ch21_judge` was the first chapter to put
    # a note between two children of the same span and the page rendered
    # both notes before either child, showing a verdict before the reply it
    # judged. Not serialized directly; `as_json`'s `body` is built from it.
    _entries: list[Note | Span] = field(default_factory=list, repr=False)

    def add_note(self, label: str, /, **payload: Any) -> None:
        # `label` is positional-only because a caller records payload keys it
        # does not choose -- a tool's arguments are named by the model, and a
        # tool with a parameter called `label` would otherwise collide with
        # this signature and raise. Same reason as `Trace.span`.
        note = Note(label, payload)
        self.notes.append(note)
        self._entries.append(note)

    def add_context(self, messages: Sequence[BaseMessage], tools: Sequence[Any] = ()) -> None:
        """The exact list handed to the model provider, recorded before it goes
        -- and, since `ch22_prompt_types`, the tool declarations sent beside
        it. A tool's docstring becomes its schema `description`; a `Literal`
        becomes an `enum`. Both are text the model reads on every call, and
        until now neither ever appeared in a `context` note -- this recorder
        was itself an instance of `ch22`'s finding, not just an illustration
        of it: a piece of the prompt nobody had reviewed as one.

        `t.tool_call_schema.model_json_schema()`, not `t.args`: a `Literal`
        declared as a reusable type alias serializes as a `$ref` into a
        `$defs` block, and `.args` alone does not include it -- recording
        that would have been the exact failure this chapter is named for,
        a field dropped while the trace still looked complete.
        """
        self.add_note(
            CONTEXT,
            messages=[_describe(m) for m in messages],
            tools=[
                {
                    "name": t.name,
                    "description": t.description,
                    "schema": t.tool_call_schema.model_json_schema(),
                }
                for t in tools
            ],
        )

    def add_reply(self, message: BaseMessage) -> None:
        """What came back, including whether it asked for any tools."""
        self.add_note(REPLY, message=_describe(message))

    def find(self, label: str) -> Note | None:
        """The first note with this label, or None. One way to look one up."""
        return next((n for n in self.notes if n.label == label), None)

    def require(self, label: str) -> Note:
        """The note with this label, or an error naming what was recorded.

        Reading a trace back is as much a part of the recorder's job as
        writing it -- the page does it from JSON, the tests do it from the
        objects. A missing note is a defect in whoever wrote the span, so it
        says so rather than returning None for the caller to trip over.
        """
        note = self.find(label)
        if note is None:
            recorded = ", ".join(n.label for n in self.notes) or "nothing"
            raise KeyError(
                f"span ['{self.name}'] has no ['{label}'] note; recorded: ['{recorded}']"
            )
        return note

    @property
    def context(self) -> list[Json]:
        """The messages this span sent. Written by `add_context`."""
        messages: list[Json] = self.require(CONTEXT).payload["messages"]
        return messages

    @property
    def reply(self) -> Json:
        """The message that came back. Written by `add_reply`."""
        message: Json = self.require(REPLY).payload["message"]
        return message

    @property
    def model_provider_ms(self) -> float | None:
        """Time between the request leaving and the response arriving."""
        sent = self.find(WIRE_REQUEST)
        got = self.find(WIRE_RESPONSE)
        if sent is None or got is None:
            return None
        return float(got.payload["at_ms"]) - float(sent.payload["at_ms"])

    @property
    def library_ms(self) -> float | None:
        """Everything the span spent that was not waiting on the model provider.

        Building the request, parsing the response, and any adapter work either
        side of it. A single elapsed number hides all of that behind the
        network.
        """
        if self.elapsed_ms is None or self.model_provider_ms is None:
            return None
        return self.elapsed_ms - self.model_provider_ms

    @property
    def elapsed_ms(self) -> float | None:
        if self.exited_at is None:
            return None
        return (self.exited_at - self.entered_at).total_seconds() * 1000

    def as_json(self) -> Json:
        return {
            "name": self.name,
            "attributes": self.attributes,
            "seq": self.seq,
            "thread": self.thread,
            "elapsed_ms": self.elapsed_ms,
            "model_provider_ms": self.model_provider_ms,
            "library_ms": self.library_ms,
            "notes": [{"label": n.label, **n.payload} for n in self.notes],
            "children": [c.as_json() for c in self.children],
            # The one field the page actually renders from -- `notes` and
            # `children` above stay exactly as they were for everything that
            # already reads them (every chapter's tests included); this is
            # the two, interleaved in the order they were really recorded.
            "body": [
                {"kind": "note", "value": {"label": e.label, **e.payload}}
                if isinstance(e, Note)
                else {"kind": "span", "value": e.as_json()}
                for e in self._entries
            ],
        }


@dataclass
class Trace:
    """A chapter's whole run. Build it with `span`, finish it with `close`."""

    chapter: str
    model_kind: ModelKind = "mock"
    # Stamped at construction, so the hash is the source that produced the run.
    source: str = ""
    spans: list[Span] = field(default_factory=list)
    summary: Json = field(default_factory=dict)
    _open: list[Span] = field(default_factory=list, repr=False)
    _seq: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.source:
            self.source = source_hash(self.chapter)

    @contextmanager
    def span(self, name: str, /, **attributes: Any) -> Generator[Span]:
        # `name` is positional-only so that an attr may also be called `name`.
        # A span recording a tool call wants exactly that, and a recorder that
        # cannot record a field because of its own signature is a bad recorder.
        self._seq += 1
        span = Span(
            name=name,
            attributes=attributes,
            entered_at=datetime.now(UTC),
            seq=self._seq,
            thread=current_thread().name,
        )
        parent = self._open[-1] if self._open else None
        (parent.children if parent else self.spans).append(span)
        if parent is not None:
            parent._entries.append(span)
        self._open.append(span)
        try:
            yield span
        finally:
            span.exited_at = datetime.now(UTC)
            self._open.pop()

    def close(self, *, turns: int, messages: int, ended: str, **extra: Any) -> None:
        """Finish the run. Raises if a span was entered and never exited.

        The three named parameters are the shape of any run -- how many turns,
        how long the list got, and why it stopped -- so they are pinned rather
        than passed by convention, and a typo in one is caught at the gate
        instead of rendering a wrong summary at the top of the page. `extra`
        is for what a particular chapter also has to say.
        """
        if self._open:
            still_open = ", ".join(s.name for s in self._open)
            raise RuntimeError(
                f"chapter ['{self.chapter}'] closed with spans still open: ['{still_open}']"
            )
        self.summary = {"turns": turns, "messages": messages, "ended": ended, **extra}

    def as_json(self) -> Json:
        return {
            "chapter": self.chapter,
            "model_kind": self.model_kind,
            "source": self.source,
            "summary": self.summary,
            "spans": [s.as_json() for s in self.spans],
        }

    def find_spans(self, name: str) -> list[Span]:
        """Every span with this name, in the order they were entered."""
        return [span for _, span in self.walk() if span.name == name]

    def walk(self) -> Iterator[tuple[int, Span]]:
        """Every span, depth-first, with its depth. The order you read it in."""

        def descend(spans: list[Span], depth: int) -> Iterator[tuple[int, Span]]:
            for span in spans:
                yield depth, span
                yield from descend(span.children, depth + 1)

        yield from descend(self.spans, 0)


def summary_line(items: Json) -> str:
    """`a=1 \u00b7 b=2` -- the one summary format, so the page and the terminal
    line cannot drift apart.

    A middle dot rather than spaces: several `k=v` pairs separated by spaces
    read as one run-on phrase, and the eye has nothing to stop on.
    """
    return " \u00b7 ".join(f"{k}={v}" for k, v in items.items())


def _describe(message: BaseMessage) -> Json:
    """One message as the trace sees it: role, content, and any tool calls."""
    described: Json = {
        "id": message.id,
        "role": message.type,
        "content": str(message.content),
    }
    if isinstance(message, AIMessage) and message.tool_calls:
        described["tool_calls"] = [
            {"name": call["name"], "args": call["args"], "id": call["id"]}
            for call in message.tool_calls
        ]
    # What request this result answers. Position is not the tie -- several
    # tool calls can be in flight at once, and the id is all that survives.
    if isinstance(message, ToolMessage):
        described["tool_call_id"] = message.tool_call_id
    # Anything the adapter carried across that is not `content`. Reasoning
    # arrives here, and a recorder that omits it is indistinguishable from a
    # model provider that never sent it -- which is how a trace starts lying.
    if message.additional_kwargs:
        described["additional_kwargs"] = message.additional_kwargs
    # Why the model stopped, in the model provider's own words. Our loop infers
    # termination from empty tool_calls; this is what it actually said.
    if finish := message.response_metadata.get("finish_reason"):
        described["finish_reason"] = finish
    if isinstance(message, AIMessage) and message.usage_metadata:
        described["usage"] = dict(message.usage_metadata)
    return described
