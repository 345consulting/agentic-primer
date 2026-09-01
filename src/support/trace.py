"""The record of a run: what was entered, what was exited, and why.

Scaffolding, never a lesson. No chapter needs you to read this file to be
understood -- it exists so that nine chapters do not each carry sixty lines
of rendering. Nothing here may import from `chapters`.
"""

import hashlib
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import current_thread
from typing import Any, Self

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

type Json = dict[str, Any]

CHAPTERS = Path(__file__).resolve().parent.parent / "chapters"

# A trace is a record of a past run, not of the current code. Two columns can
# look coherent while describing different versions of a chapter -- which is
# the same class of lie as a recorder that omits a field. The stamp is what
# lets the page notice.
UNKNOWN_SOURCE = "unknown"


def source_hash(chapter: str) -> str:
    """The chapter's source, as twelve hex characters.

    Reading the file, not importing it -- `support` may not import `chapters`.
    A chapter with no file on disk is stamped `unknown` rather than raising,
    because a test may build a trace for a chapter that was never written.
    """
    path = CHAPTERS / f"{chapter}.py"
    if not path.is_file():
        return UNKNOWN_SOURCE
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


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
    # provider that never sent it -- which is how a trace starts lying.
    if message.additional_kwargs:
        described["additional_kwargs"] = message.additional_kwargs
    # Why the model stopped, in the provider's own words. Our loop infers
    # termination from empty tool_calls; this is what it actually said.
    if finish := message.response_metadata.get("finish_reason"):
        described["finish_reason"] = finish
    if isinstance(message, AIMessage) and message.usage_metadata:
        described["usage"] = dict(message.usage_metadata)
    return described


@dataclass
class Note:
    """Something worth recording that is not itself an entry or an exit."""

    label: str
    payload: Json


@dataclass
class Span:
    """One entered-and-exited thing. Every enter has a matching exit."""

    name: str
    attrs: Json
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

    def note(self, label: str, **payload: Any) -> Self:
        self.notes.append(Note(label, payload))
        return self

    def context(self, messages: Sequence[BaseMessage]) -> Self:
        """The exact list handed to the provider, recorded before it goes."""
        return self.note("context", messages=[_describe(m) for m in messages])

    def reply(self, message: BaseMessage) -> Self:
        """What came back, including whether it asked for any tools."""
        return self.note("reply", message=_describe(message))

    @property
    def provider_ms(self) -> float | None:
        """Time between the request leaving and the response arriving."""
        sent = next((n for n in self.notes if n.label == "wire request"), None)
        got = next((n for n in self.notes if n.label == "wire response"), None)
        if sent is None or got is None:
            return None
        return float(got.payload["at_ms"]) - float(sent.payload["at_ms"])

    @property
    def library_ms(self) -> float | None:
        """Everything the span spent that was not waiting on the provider.

        Building the request, parsing the response, and any adapter work either
        side of it. A single elapsed number hides all of that behind the
        network.
        """
        if self.elapsed_ms is None or self.provider_ms is None:
            return None
        return self.elapsed_ms - self.provider_ms

    @property
    def elapsed_ms(self) -> float | None:
        if self.exited_at is None:
            return None
        return (self.exited_at - self.entered_at).total_seconds() * 1000

    def as_json(self) -> Json:
        return {
            "name": self.name,
            "attrs": self.attrs,
            "seq": self.seq,
            "thread": self.thread,
            "elapsed_ms": self.elapsed_ms,
            "provider_ms": self.provider_ms,
            "library_ms": self.library_ms,
            "notes": [{"label": n.label, **n.payload} for n in self.notes],
            "children": [c.as_json() for c in self.children],
        }


@dataclass
class Trace:
    """A chapter's whole run. Build it with `span`, finish it with `close`."""

    chapter: str
    kind: str = "mock"
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
    def span(self, name: str, /, **attrs: Any) -> Iterator[Span]:
        # `name` is positional-only so that an attr may also be called `name`.
        # A span recording a tool call wants exactly that, and a recorder that
        # cannot record a field because of its own signature is a bad recorder.
        self._seq += 1
        span = Span(
            name=name,
            attrs=attrs,
            entered_at=datetime.now(UTC),
            seq=self._seq,
            thread=current_thread().name,
        )
        parent = self._open[-1] if self._open else None
        (parent.children if parent else self.spans).append(span)
        self._open.append(span)
        try:
            yield span
        finally:
            span.exited_at = datetime.now(UTC)
            self._open.pop()

    def close(self, **summary: Any) -> None:
        if self._open:
            still_open = ", ".join(s.name for s in self._open)
            raise RuntimeError(f"{self.chapter}: closed with spans still open [{still_open}]")
        self.summary = summary

    def as_json(self) -> Json:
        return {
            "chapter": self.chapter,
            "kind": self.kind,
            "source": self.source,
            "summary": self.summary,
            "spans": [s.as_json() for s in self.spans],
        }

    def walk(self) -> Iterator[tuple[int, Span]]:
        """Every span, depth-first, with its depth. The order you read it in."""

        def descend(spans: list[Span], depth: int) -> Iterator[tuple[int, Span]]:
            for span in spans:
                yield depth, span
                yield from descend(span.children, depth + 1)

        yield from descend(self.spans, 0)

    def report(self, out: Path = Path("out")) -> Path:
        """Write the trace, print one line, and hand back the path to read."""
        from support.view import write

        path: Path = write(self, out)
        summary = "  ".join(f"{k}={v}" for k, v in self.summary.items())
        print(f"{self.chapter}: {summary}")
        print(f"  {path}")
        return path
