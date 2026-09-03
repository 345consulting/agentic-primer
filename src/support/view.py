# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Trace to JSON, and JSON to a page you can read in a browser.

Scaffolding, never a lesson. The JSON is canonical; the page is a projection
of it. Collapsing is `<details>`, so the page needs no script and no server --
open the file.
"""

from support.chapters import UNKNOWN_SOURCE, chapters, source_hash
from support.trace import CONTEXT, ENDED, MODEL_KINDS, REPLY, Json, Trace, summary_line

import html
import json
import re
from pathlib import Path

_STYLE = """
:root { color-scheme: light dark; --line: #8884; --dim: #8888; }
body { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; margin: 0 1.5rem 2rem; }
/* The title and the column toggles stay put: at thirty-five chapters you are
   always scrolled away from them, and hiding a column is something you want
   to do from wherever you are. `Canvas` is the system background, so it
   follows the viewer's theme without a token of its own. */
header.top { position: sticky; top: 0; z-index: 2; background: Canvas;
             padding: 1.25rem 0 .6rem; margin-bottom: 1rem;
             border-bottom: 1px solid var(--line); }
header.top h1 { display: inline; margin-right: .75rem; }
/* The toggles get their own line: they are a control, and reading them as a
   continuation of the subtitle is the first thing anyone notices. */
header.top .toggles { display: block; margin-top: .5rem; }
h1 { font-size: 1.1rem; margin: 0 0 .25rem; }
.summary { color: var(--dim); margin-bottom: 1.5rem; }
details { border-left: 1px solid var(--line); margin: 0 0 0 .5rem; padding-left: .75rem; }
summary { cursor: pointer; padding: .15rem 0; }
summary::marker { color: var(--dim); }
.name { font-weight: 600; }
.meta, .ms { color: var(--dim); font-weight: 400; }
.note { margin: .35rem 0 .5rem 1rem; }
.label { color: var(--dim); }
table { border-collapse: collapse; margin: .25rem 0 .5rem; width: 100%; }
td, th { border-bottom: 1px solid var(--line); padding: .2rem .5rem; text-align: left;
         vertical-align: top; font-weight: 400; }
th { color: var(--dim); font-size: .85em; }
.role { white-space: nowrap; }
.calls { color: #b26; }
.content { white-space: pre-wrap; }
input[type=checkbox] { display: none; }
.toggles { margin-bottom: .75rem; color: var(--dim); }
.toggles label { cursor: pointer; border: 1px solid var(--line); border-radius: 3px;
                 padding: .1rem .5rem; margin-left: .4rem; user-select: none; }
#hide-mock:checked ~ .toggles label[for=hide-mock],
#hide-live:checked ~ .toggles label[for=hide-live] { opacity: .35; text-decoration: line-through; }
#hide-mock:checked ~ * .col-mock { display: none; }
#hide-live:checked ~ * .col-live { display: none; }
.row { display: flex; gap: 1.25rem; align-items: flex-start; }
.row.head { border-bottom: 1px solid var(--line); padding-bottom: .4rem; margin-bottom: .75rem; }
.col { flex: 1 1 0; min-width: 0; }
.col.missing { color: var(--dim); font-style: italic; padding: .5rem 0; }
.turn { margin-bottom: 1.5rem; }
.turnno { color: var(--dim); font-size: .85em; text-transform: uppercase;
          letter-spacing: .06em; margin-bottom: .3rem; }
.stale { color: #b26; font-weight: 600; }
/* The outcome is what a reader is scanning for, so it is the one part of a
   row at full strength. The name is a label and the rest is metadata. */
.digest { color: inherit; }
.name { font-weight: 400; color: var(--dim); }
.dot { color: var(--line); }
details.chapter > summary { padding: .6rem 0; border-top: 1px solid var(--line);
                            font-size: 1rem; margin-top: 1rem; }
details.chapter { border: 0; margin: 0; padding: 0; }
details.chapter > summary .why { color: var(--dim); font-weight: 400; font-size: .9rem; }
details.chapter > summary .facts { color: var(--dim); font-weight: 400; font-size: .85rem;
                                   display: block; margin-left: 1rem; }
h2 { font-size: 1rem; margin: 2rem 0 .25rem; padding-top: .75rem;
     border-top: 1px solid var(--line); }
h2 .why { color: var(--dim); font-weight: 400; }
h2 a.top { color: var(--dim); font-weight: 400; font-size: .8em; float: right; }
nav { margin: 0 0 1.5rem; }
nav a { display: block; padding: .12rem 0; text-decoration: none; }
nav a:hover { text-decoration: underline; }
nav .why { color: var(--dim); }
nav .not-run { color: var(--dim); font-style: italic; }
/* A wire note is a handful of short fields and one long one. As a run-on line
   the long field swallows the rest, so each gets its own row and the keys
   line up in a column you can scan down. */
.wirefields { display: grid; grid-template-columns: max-content minmax(0, 1fr);
              gap: .1rem .75rem; margin: .35rem 0; }
.wirefields dt { color: var(--dim); white-space: nowrap; }
.wirefields dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.wirefields dd.bulk { color: var(--dim); font-size: .92em; }
.wirefields details { border: 0; margin: 0; padding: 0; }
.wirefields summary { padding: 0; }
table.headers { margin: .25rem 0 .5rem; }
table.headers td { font-size: .92em; }
table.headers td.hname { color: var(--dim); white-space: nowrap; width: 1%; }
table.headers td.hvalue { overflow-wrap: anywhere; }
table.headers td.unrecorded { color: var(--dim); font-style: italic; }
.wirefields dd.defaulted { color: #b26; }
@media (prefers-color-scheme: dark) { .wirefields dd.defaulted { color: #f9a; } }
/* Wrap rather than scroll: the page is read at whatever width the window is,
   and a horizontal scrollbar inside a column hides the end of every line. */
pre.wire .k { color: #06c; }
pre.wire .s { color: #0a7; }
pre.wire .n { color: #b26; }
pre.wire .l { color: #a60; font-style: italic; }
@media (prefers-color-scheme: dark) {
  pre.wire .k { color: #7bf; } pre.wire .s { color: #6d9; }
  pre.wire .n { color: #f9a; } pre.wire .l { color: #fc7; }
}
pre.wire { background: #8881; padding: .5rem .75rem; margin: .35rem 0; border-radius: 3px;
           white-space: pre-wrap; overflow-wrap: anywhere; }
"""


# One pass over pretty-printed JSON: strings (a key if a colon follows it),
# numbers, and the three literals. Everything else falls through as punctuation.
_TOKENS = re.compile(
    r'(?P<str>"(?:[^"\\]|\\.)*")(?P<colon>\s*:)?'
    r"|(?P<num>-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    r"|(?P<lit>\btrue\b|\bfalse\b|\bnull\b)"
)


def write_book(out: Path = Path("out")) -> Path:
    """Every chapter that has been run, on one page, in reading order.

    The order and the summaries come from the chapter files themselves, so a
    chapter that exists is listed the moment it is written, before it has ever
    been run.
    """
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    for chapter, why in chapters():
        traces = _load(chapter, out)
        # Two states. A chapter with no trace has been written and not run --
        # there is no third state any more, because the list comes from the
        # files, so a chapter that is listed exists by definition.
        if traces:
            state, body = "", _render_chapter(chapter, traces)
        else:
            state = "not run"
            body = f'<div class="col missing">{state} &mdash; just run {chapter}</div>'
        sections.append(
            f'<details class="chapter" id="{chapter}">'
            # The count goes on its own line under the name and summary:
            # a chapter's row is a title, and the size of it is a caption.
            f"<summary>{html.escape(chapter)} "
            f'<span class="why">{html.escape(why)}</span>'
            f'<span class="facts">{_facts(traces, state)}</span></summary>'
            f"{body}</details>"
        )

    page = out / "index.html"
    page.write_text(
        f"<!doctype html><meta charset=utf-8><title>the agentic primer</title>"
        f"<style>{_STYLE}</style>{_render_toggles()}"
        f"{_render_header('the agentic primer', 'every chapter that has been run, in order')}"
        f"{''.join(sections)}"
    )
    return page


def render(traces: dict[str, Json]) -> str:
    """Every kind that has been run, turn beside turn.

    Turns are the unit of comparison, not spans: a mock run and a live run
    agree on how many turns there were or they do not, and that disagreement
    is the most interesting thing either trace can tell you.
    """
    chapter = next(iter(traces.values()))["chapter"]
    return (
        f"<!doctype html><meta charset=utf-8>"
        f"<title>{html.escape(chapter)}</title><style>{_STYLE}</style>"
        f"{_render_toggles()}{_render_header(chapter, '')}"
        f"{_render_chapter(chapter, traces)}"
    )


def write(trace: Trace, out: Path) -> Path:
    """Write this run, then re-render the page over every kind on disk.

    A run never discards another kind's record. That is what makes the page a
    comparison rather than a snapshot -- run the mock, run it live, read both.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{trace.chapter}.{trace.model_kind}.json").write_text(
        json.dumps(trace.as_json(), indent=2)
    )

    page = out / f"{trace.chapter}.html"
    page.write_text(render(_load(trace.chapter, out)))
    # The book is regenerated on every run, so it can never be older than the
    # chapter pages it collects.
    write_book(out)
    return page


def _render_attributes(attributes: Json) -> str:
    """A span's identity as a value, anything else as a pair.

    The first attribute is what the span *is* -- a scenario's name, a turn's
    number, a tool's name -- and its key is guessable from the span's own
    name, so `partial_failure` says as much as `name=partial_failure`. Every
    later attribute is a qualifier and keeps its key, because `24` alone does
    not say `max_tokens`.

    An attribute with no value is dropped rather than printed as `None`: the
    contrast between a row that shows `max_tokens=24` and one that shows
    nothing is the same fact, without a Python word leaking onto the page.

    A tool call's id is dropped outright -- forty opaque characters on the
    busiest row on the page, and it is in the notes and the JSON, where the
    pairing rule needs it.
    """
    shown = [(key, value) for key, value in attributes.items() if key != "id" and value is not None]
    if not shown:
        return ""
    (_, identity), *rest = shown
    parts = [str(identity)] + [f"{key}={value}" for key, value in rest]
    return f'<span class="meta">{html.escape(" \u00b7 ".join(parts))}</span>'


def _head_summary(data: Json) -> Json:
    """A column's summary, minus what its own rows already say.

    A chapter that ran scenarios states each ending on the scenario's row, so
    repeating them here says it twice per column and four times per chapter.
    A chapter that ran turns has nowhere else to put it, so it keeps it.
    """
    summary: Json = dict(data["summary"])
    spans = data["spans"]
    if spans and spans[0]["name"] == "scenario":
        summary.pop("ended", None)
    return summary


def _render_chapter(chapter: str, traces: dict[str, Json]) -> str:
    """One chapter's head row and its turns. The same markup on both pages."""
    current = source_hash(chapter)
    heads, bodies = [], []
    for kind in MODEL_KINDS:
        data = traces.get(kind)
        summary = summary_line(_head_summary(data)) if data else "not run"
        # The kind is the label; the summary and the source stamp are two
        # separate facts about it.
        facts = ' <span class="dot">\u00b7</span> '.join(
            part
            for part in (
                f'<span class="meta">{html.escape(summary)}</span>',
                _render_source(data, current),
            )
            if part
        )
        heads.append(f'<div class="col col-{kind}"><b>{kind}</b> {facts}</div>')

    units = max((len(data["spans"]) for data in traces.values()), default=0)
    for index in range(units):
        cells = "".join(_render_column(kind, traces.get(kind), index) for kind in MODEL_KINDS)
        bodies.append(
            f'<div class="turn"><div class="turnno">{_unit_label(traces, index)}</div>'
            f'<div class="row">{cells}</div></div>'
        )

    return f'<div class="row head">{"".join(heads)}</div>{"".join(bodies)}'


def _render_column(kind: str, data: Json | None, index: int) -> str:
    """One kind's view of one unit -- or a note that this kind has not one."""
    if data is None:
        return f'<div class="col col-{kind} missing">not run</div>'
    spans = data["spans"]
    if index >= len(spans):
        return f'<div class="col col-{kind} missing">nothing here</div>'
    return f'<div class="col col-{kind}">{_render_span(spans[index])}</div>'


def _unit_label(traces: dict[str, Json], index: int) -> str:
    """What the chapter opened at the top level, named as the chapter named it.

    The comparison unit used to be hardcoded as "turn N", which was true only
    by accident: every chapter so far opens turns at the top level. A chapter
    running the same loop under several conditions opens `scenario` spans
    instead, and the page should say so rather than call them turns.
    """
    for kind in MODEL_KINDS:
        data = traces.get(kind)
        if data and index < len(data["spans"]):
            span = data["spans"][index]
            # The heading is the span's name and what it *is* -- the first
            # attribute -- and nothing else. The qualifiers are on the span's
            # own row underneath, where they have their keys; here they would
            # be bare values with nothing to attach to, and an unset one would
            # print as `None`.
            identity = next(
                (str(value) for value in span["attributes"].values() if value is not None),
                "",
            )
            return html.escape(f"{span['name']} {identity}".strip())
    return f"#{index + 1}"


def _render_field(key: str, value: str, css: str = "") -> str:
    klass = f' class="{css}"' if css else ""
    return f"<dt>{html.escape(key)}</dt><dd{klass}>{html.escape(value)}</dd>"


def _header_values(note: Json) -> dict[str, str | None]:
    """Headers as name to value, `None` where the value was not recorded.

    Traces written before the values were recorded carry `header_names`
    instead. Reading both means an old record still renders -- as names with
    no values, which is exactly what it is.
    """
    headers: dict[str, str | None] = note.get("headers") or {}
    if not headers:
        headers = dict.fromkeys(note.get("header_names") or [])
    return headers


def _render_headers(headers: dict[str, str | None]) -> str:
    """Collapsed by default: the longest field in the note and rarely the point.

    A value of `None` renders as "not recorded" rather than as blank. A header
    that was present with its value withheld is a different fact from a header
    that was never sent, and a blank cell cannot tell them apart.
    """
    rows = []
    for name, value in headers.items():
        cell = (
            f'<td class="hvalue">{html.escape(value)}</td>'
            if value is not None
            else '<td class="hvalue unrecorded">not recorded</td>'
        )
        rows.append(f'<tr><td class="hname">{html.escape(name)}</td>{cell}</tr>')
    return (
        f'<details><summary>show</summary><table class="headers">{"".join(rows)}</table></details>'
    )


def _highlight(text: str) -> str:
    """Colour JSON without a library. The page stays one self-contained file.

    Tokenise the raw text and escape each piece as it is emitted. Escaping
    first would turn every quote into `&quot;` and leave the string pattern
    matching nothing -- which highlights the digits inside string values and
    nothing else.
    """
    out: list[str] = []
    at = 0
    for match in _TOKENS.finditer(text):
        out.append(html.escape(text[at : match.start()]))
        if raw := match.group("str"):
            kind = "k" if match.group("colon") else "s"
            out.append(f'<span class="{kind}">{html.escape(raw)}</span>')
            out.append(match.group("colon") or "")
        elif raw := match.group("num"):
            out.append(f'<span class="n">{raw}</span>')
        else:
            out.append(f'<span class="l">{match.group("lit")}</span>')
        at = match.end()
    out.append(html.escape(text[at:]))
    return "".join(out)


def _load(chapter: str, out: Path) -> dict[str, Json]:
    """Every kind of this chapter that has been run and left a record."""
    traces: dict[str, Json] = {}
    for kind in MODEL_KINDS:
        record = out / f"{chapter}.{kind}.json"
        if record.exists():
            traces[kind] = json.loads(record.read_text())
    return traces


def _render_messages(messages: list[Json]) -> str:
    rows = []
    for index, message in enumerate(messages):
        calls = message.get("tool_calls") or []
        asked = (
            '<div class="calls">'
            + "<br>".join(
                f"{html.escape(c['name'])}({html.escape(json.dumps(c['args']))})" for c in calls
            )
            + "</div>"
            if calls
            else ""
        )
        rows.append(
            f"<tr><td>[{index}]</td>"
            f'<td class="role">{html.escape(message["role"])}</td>'
            f'<td class="content">{html.escape(message["content"])}{asked}</td></tr>'
        )
    head = "<tr><th></th><th>role</th><th>content</th></tr>"
    return f"<table>{head}{''.join(rows)}</table>"


def _render_note(note: Json) -> str:
    label = note["label"]
    if label == CONTEXT:
        count = len(note["messages"])
        return (
            f'<div class="note"><span class="label">context sent &mdash; '
            f"{count} message(s)</span>{_render_messages(note['messages'])}</div>"
        )
    if label == REPLY:
        return (
            f'<div class="note"><span class="label">reply</span>'
            f"{_render_messages([note['message']])}</div>"
        )
    if label.startswith("wire "):
        return (
            f'<div class="note"><span class="label">{html.escape(label)}</span>'
            f"{_render_wire_fields(note)}"
            f'<pre class="wire">{_highlight(json.dumps(note.get("body"), indent=2))}</pre></div>'
        )
    rest = {k: v for k, v in note.items() if k != "label"}
    return (
        f'<div class="note"><span class="label">{html.escape(label)}</span> '
        f"{html.escape(json.dumps(rest))}</div>"
    )


def _render_source(data: Json | None, current: str) -> str:
    """Whether this column still describes the chapter as it is on disk.

    A stored trace records a past run. Two columns can agree turn for turn and
    still have come from different versions of the chapter -- the page looks
    coherent while quietly comparing two different things. The stamp is the
    only thing that can say so.
    """
    if data is None:
        return ""
    stamp = data.get("source", UNKNOWN_SOURCE)
    if stamp == current or current == UNKNOWN_SOURCE:
        # A stamp that matches is twelve hex characters saying nothing. It
        # is worth a row only when the two columns came from different code.
        return ""
    return (
        f'<span class="stale">recorded from different source '
        f"({html.escape(stamp)} &ne; {html.escape(current)})</span>"
    )


def _facts(traces: dict[str, Json], state: str) -> str:
    """A chapter's one line while it is closed: how many of whatever it ran.

    Deliberately not a dashboard. At thirty-five rows the questions are which
    chapter this is, whether it has run, and how much of it there is. Turns,
    messages and endings are detail, and detail belongs on the rows below --
    which is where a reader goes once something has caught their eye.

    Turns or scenarios, never both: a chapter opens one or the other, and the
    count of the thing it opened is the size of the chapter. When the two
    kinds disagree it says so as `3/2`, because a disagreement between the
    columns is the one thing at this level worth interrupting for.
    """
    if state:
        return state
    counts = {kind: len(data["spans"]) for kind, data in traces.items()}
    if not counts:
        return ""
    unit = next(iter(traces.values()))["spans"][0]["name"]
    shown = sorted(set(counts.values()), reverse=True)
    total = "/".join(str(count) for count in shown)
    return f"{total} {unit}{'' if shown == [1] else 's'}"


ROW_TEXT = 90


def _shorten(text: str) -> str:
    """The first line's worth, and an honest count of what was left out.

    A closed row is one line, and a model can answer with several hundred
    characters. Cutting silently leaves a reader unsure whether the sentence
    ended or the page gave up, so the row says which.
    """
    text = " ".join(text.split())
    if len(text) <= ROW_TEXT:
        return text
    return f"{text[:ROW_TEXT]}... ({len(text) - ROW_TEXT} more characters)"


def _digest(span: Json) -> str:
    """What came out of a span, said in the span's own terms.

    Every row on the page reads left to right as: what it is, what it cost,
    what came out. This is the last of those, and it is labelled because
    position alone does not say whether `price_of` on a model row is what the
    model said or what it asked for.

    Nothing here is computed that the trace did not record.
    """
    notes = {note["label"]: note for note in span["notes"]}

    if reply := notes.get(REPLY):
        message = reply["message"]
        parts = []
        usage = message.get("usage") or {}
        if usage.get("input_tokens"):
            parts.append(f"{usage['input_tokens']} in / {usage['output_tokens']} out")
        if calls := message.get("tool_calls") or []:
            parts.append("asked for: " + ", ".join(call["name"] for call in calls))
        if content := message["content"]:
            parts.append(f"responded with: {_shorten(content)}")
        # Empty content and no tool calls is not a rendering artefact: it is
        # what a reply cut off before it produced anything looks like, and it
        # is the whole of ch05_loop_endings.
        if not calls and not content:
            parts.append("responded with: nothing")
        return " \u00b7 ".join(parts)

    if ended := notes.get(ENDED):
        # A scenario has one note. How much it ran and how it stopped is the
        # whole of what it has to say while closed.
        inner = span["children"][0]["name"] if span["children"] else "turn"
        count = len(span["children"])
        turns = f"{count} {inner}{'' if count == 1 else 's'}"
        return f"{turns} \u00b7 ended: {ended['reason']}"

    if "args" in notes:
        given = ", ".join(f"{k}={v}" for k, v in notes["args"].items() if k != "label")
        parts = [f"given: {given}"] if given else []
        # A call that was retried and then worked leaves no trace in the
        # message list -- ch09_who_retries is about the model never learning
        # that. The page is not the model, and this is where it can say so.
        tries = sum(1 for note in span["notes"] if note["label"] == "attempt")
        if tries > 1:
            parts.append(f"attempts: {tries}")
        if failed := notes.get("failed"):
            # The exception's own message carries a colon, so it goes in
            # parentheses rather than after a second one. A chapter whose
            # failures are all one class records no type name, and then the
            # message is the whole of what happened.
            named = failed.get("exception")
            message = failed["message"][:60]
            parts.append(f"raised: {named} ({message})" if named else f"raised: {message}")
        elif result := notes.get("result"):
            parts.append(f"returned: {result['value']}")
        return " \u00b7 ".join(parts)

    if span["children"]:
        # A turn is one invocation plus its tools, so what came out of it is
        # what came out of the model. The tools are one row down; repeating
        # them here was a preview of the thing directly underneath.
        model = next((c for c in span["children"] if c["name"] == "model"), None)
        return _digest(model) if model else ""

    return ""


def _render_span(span: Json) -> str:
    # Sequence and thread are recorded on every span and shown on none of
    # them: until a chapter runs tools concurrently there is one thread and
    # seq is document order, so the columns are noise. They are in the JSON,
    # and the chapter that needs them can ask the page to show them.
    where = ""
    # Every part of a span's row is a separate fact -- what it is, what it was
    # given, where it ran, how long it took, what it did -- and space-separated
    # they read as one phrase. The dot gives the eye somewhere to stop.
    parts = [
        part
        for part in (_render_attributes(span["attributes"]), where, _render_timing(span))
        if part
    ]
    if digest := _digest(span):
        parts.append(f'<span class="digest">{html.escape(digest)}</span>')
    facts = ' <span class="dot">\u00b7</span> '.join(parts)
    # The name is a label rather than one of the facts, so a space follows
    # it. Everything after it is a separate fact and takes a separator.
    head = f'<span class="name">{html.escape(span["name"])}</span> {facts}'
    body = "".join(_render_note(n) for n in span["notes"]) + "".join(
        _render_span(c) for c in span["children"]
    )
    return f"<details><summary>{head}</summary>{body}</details>"


def _render_timing(span: Json) -> str:
    """Total, and where it went. A single number hides the library behind the network."""
    total = span.get("elapsed_ms")
    if total is None:
        return ""
    model_provider, library = span.get("model_provider_ms"), span.get("library_ms")
    if model_provider is None or library is None:
        return f'<span class="ms">{total:.1f}ms</span>'
    return (
        f'<span class="ms">{total:.1f}ms '
        f"(model provider {model_provider:.1f} \u00b7 library {library:.1f})</span>"
    )


def _render_toggles() -> str:
    """The checkboxes alone.

    Hiding a column is CSS on a sibling selector rather than script, so these
    must precede every column they hide and cannot move into the header. Their
    labels can: `for=` reaches an input from anywhere on the page.
    """
    return '<input type="checkbox" id="hide-mock"><input type="checkbox" id="hide-live">'


def _render_header(title: str, subtitle: str) -> str:
    """Title and column toggles, stuck to the top of the window.

    At thirty-five chapters you are always scrolled away from them, and
    hiding a column is something you want to do from wherever you are.
    """
    return (
        f'<header class="top"><h1>{html.escape(title)}</h1>'
        f'<span class="summary">{html.escape(subtitle)}</span>'
        f'<span class="toggles">show: '
        f'<label for="hide-mock">mock</label><label for="hide-live">live</label>'
        f"</span></header>"
    )


def _render_wire_fields(note: Json) -> str:
    """One row per field, keys in a column you can scan down.

    Headers collapse, because eighteen of them is the longest thing in the
    note and rarely what you came for. The defaulted parameters do not: every
    one is a value the model provider chose and the record cannot recover, so the
    run is not reproducible from its own trace, and that should be in the way.
    """
    skip = ("label", "body", "at_ms", "headers", "header_names", "defaulted_by_model_provider")
    rows = [_render_field(k, str(v)) for k, v in note.items() if k not in skip]
    if defaulted := note.get("defaulted_by_model_provider"):
        rows.append(
            _render_field(f"defaulted ({len(defaulted)})", ", ".join(defaulted), "defaulted")
        )
    if headers := _header_values(note):
        rows.append(f"<dt>headers ({len(headers)})</dt><dd>{_render_headers(headers)}</dd>")
    return f'<dl class="wirefields">{"".join(rows)}</dl>'


if __name__ == "__main__":
    print(write_book())
