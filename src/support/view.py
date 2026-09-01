"""Trace to JSON, and JSON to a page you can read in a browser.

Scaffolding, never a lesson. The JSON is canonical; the page is a projection
of it. Collapsing is `<details>`, so the page needs no script and no server --
open the file.
"""

from support.trace import UNKNOWN_SOURCE, Json, Trace, source_hash

import html
import json
import re
from pathlib import Path

_STYLE = """
:root { color-scheme: light dark; --line: #8884; --dim: #8888; }
body { font: 13px ui-monospace, SFMono-Regular, Menlo, monospace; margin: 1.5rem; }
h1 { font-size: 1.1rem; margin: 0 0 .25rem; }
.summary { color: var(--dim); margin-bottom: 1.5rem; }
details { border-left: 1px solid var(--line); margin: 0 0 0 .5rem; padding-left: .75rem; }
summary { cursor: pointer; padding: .15rem 0; }
summary::marker { color: var(--dim); }
.name { font-weight: 600; }
.attrs, .ms { color: var(--dim); font-weight: 400; }
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


def _attrs(attrs: Json) -> str:
    if not attrs:
        return ""
    body = "  ".join(f"{k}={v}" for k, v in attrs.items())
    return f' <span class="attrs">{html.escape(body)}</span>'


def _messages(messages: list[Json]) -> str:
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


def _header_map(note: Json) -> dict[str, str | None]:
    """Headers as name to value, `None` where the value was not recorded.

    Traces written before the values were recorded carry `header_names`
    instead. Reading both means an old record still renders -- as names with
    no values, which is exactly what it is.
    """
    headers: dict[str, str | None] = note.get("headers") or {}
    if not headers:
        headers = dict.fromkeys(note.get("header_names") or [])
    return headers


def _headers(headers: dict[str, str | None]) -> str:
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


def _field(key: str, value: str, css: str = "") -> str:
    klass = f' class="{css}"' if css else ""
    return f"<dt>{html.escape(key)}</dt><dd{klass}>{html.escape(value)}</dd>"


def _wire_fields(note: Json) -> str:
    """One row per field, keys in a column you can scan down.

    Headers collapse, because eighteen of them is the longest thing in the
    note and rarely what you came for. The defaulted parameters do not: every
    one is a value the provider chose and the record cannot recover, so the
    run is not reproducible from its own trace, and that should be in the way.
    """
    skip = ("label", "body", "at_ms", "headers", "header_names", "defaulted_by_provider")
    rows = [_field(k, str(v)) for k, v in note.items() if k not in skip]
    if defaulted := note.get("defaulted_by_provider"):
        rows.append(_field(f"defaulted ({len(defaulted)})", ", ".join(defaulted), "defaulted"))
    if headers := _header_map(note):
        rows.append(f"<dt>headers ({len(headers)})</dt><dd>{_headers(headers)}</dd>")
    return f'<dl class="wirefields">{"".join(rows)}</dl>'


def _note(note: Json) -> str:
    label = note["label"]
    if label == "context":
        count = len(note["messages"])
        return (
            f'<div class="note"><span class="label">context sent &mdash; '
            f"{count} message(s)</span>{_messages(note['messages'])}</div>"
        )
    if label == "reply":
        return (
            f'<div class="note"><span class="label">reply</span>'
            f"{_messages([note['message']])}</div>"
        )
    if label.startswith("wire "):
        return (
            f'<div class="note"><span class="label">{html.escape(label)}</span>'
            f"{_wire_fields(note)}"
            f'<pre class="wire">{_highlight(json.dumps(note.get("body"), indent=2))}</pre></div>'
        )
    rest = {k: v for k, v in note.items() if k != "label"}
    return (
        f'<div class="note"><span class="label">{html.escape(label)}</span> '
        f"{html.escape(json.dumps(rest))}</div>"
    )


def _timing(span: Json) -> str:
    """Total, and where it went. A single number hides the library behind the network."""
    total = span.get("elapsed_ms")
    if total is None:
        return ""
    provider, library = span.get("provider_ms"), span.get("library_ms")
    if provider is None or library is None:
        return f' <span class="ms">{total:.1f}ms</span>'
    return (
        f' <span class="ms">{total:.1f}ms '
        f"(provider {provider:.1f} &middot; library {library:.1f})</span>"
    )


def _span(span: Json) -> str:
    thread = html.escape(str(span.get("thread", "")))
    where = f' <span class="attrs">#{span.get("seq", "")} {thread}</span>'
    head = (
        f'<span class="name">{html.escape(span["name"])}</span>'
        f"{_attrs(span['attrs'])}{where}{_timing(span)}"
    )
    body = "".join(_note(n) for n in span["notes"]) + "".join(_span(c) for c in span["children"])
    return f"<details open><summary>{head}</summary>{body}</details>"


KINDS = ("mock", "live")


def _column(kind: str, data: Json | None, index: int) -> str:
    """One kind's view of one turn -- or a note that this kind has not been run."""
    if data is None:
        return f'<div class="col col-{kind} missing">not run</div>'
    spans = data["spans"]
    if index >= len(spans):
        return f'<div class="col col-{kind} missing">no turn {index + 1}</div>'
    return f'<div class="col col-{kind}">{_span(spans[index])}</div>'


def _source(data: Json | None, current: str) -> str:
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
        return f' <span class="attrs">source {html.escape(stamp)}</span>'
    return (
        f' <span class="stale">recorded from different source '
        f"({html.escape(stamp)} &ne; {html.escape(current)})</span>"
    )


def render(traces: dict[str, Json]) -> str:
    """Every kind that has been run, turn beside turn.

    Turns are the unit of comparison, not spans: a mock run and a live run
    agree on how many turns there were or they do not, and that disagreement
    is the most interesting thing either trace can tell you.
    """
    chapter = next(iter(traces.values()))["chapter"]
    current = source_hash(chapter)
    heads, bodies = [], []
    for kind in KINDS:
        data = traces.get(kind)
        summary = "  ".join(f"{k}={v}" for k, v in data["summary"].items()) if data else "not run"
        heads.append(
            f'<div class="col col-{kind}"><b>{kind}</b> '
            f'<span class="attrs">{html.escape(summary)}</span>'
            f"{_source(data, current)}</div>"
        )

    turns = max((len(d["spans"]) for d in traces.values()), default=0)
    for index in range(turns):
        cells = "".join(_column(kind, traces.get(kind), index) for kind in KINDS)
        bodies.append(
            f'<div class="turn"><div class="turnno">turn {index + 1}</div>'
            f'<div class="row">{cells}</div></div>'
        )

    return (
        f"<!doctype html><meta charset=utf-8>"
        f"<title>{html.escape(chapter)}</title><style>{_STYLE}</style>"
        f"<h1>{html.escape(chapter)}</h1>"
        # Checkbox toggles rather than script: hiding a column is CSS on a
        # sibling selector, so the page stays one file with nothing to load.
        f'<input type="checkbox" id="hide-mock"><input type="checkbox" id="hide-live">'
        f'<div class="toggles">show: '
        f'<label for="hide-mock">mock</label><label for="hide-live">live</label></div>'
        f'<div class="row head">{"".join(heads)}</div>{"".join(bodies)}'
    )


def write(trace: Trace, out: Path) -> Path:
    """Write this run, then re-render the page over every kind on disk.

    A run never discards another kind's record. That is what makes the page a
    comparison rather than a snapshot -- run the mock, run it live, read both.
    """
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{trace.chapter}.{trace.kind}.json").write_text(json.dumps(trace.as_json(), indent=2))

    traces: dict[str, Json] = {}
    for kind in KINDS:
        record = out / f"{trace.chapter}.{kind}.json"
        if record.exists():
            traces[kind] = json.loads(record.read_text())

    page = out / f"{trace.chapter}.html"
    page.write_text(render(traces))
    return page
