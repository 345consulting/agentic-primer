# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""Instrumentation that observes and never participates.

Every mechanism this chapter would formalize already exists by hand,
chapter after chapter -- `Trace`/`Span` in `support/trace.py`, the httpx
wire hooks in `support/models.py`, the model-provider-vs-library timing
split, `view.py`'s rendering. What was missing was not a mechanism; it
was `ch39_callbacks`' own subject settled first -- whether a framework's
real tracing can be the same mechanism as a hook, strong enough to
suppress the audit trail. It can. This chapter is the other half: does
*our* instrumentation ever risk that, checked against a real
OpenTelemetry SDK instead of asserted about our own.

Four scenarios:

    a_real_otel_span_carries_the_gen_ai_attributes
        a real `opentelemetry.sdk.trace` span, not a hand-mapped dict --
        `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.*`, using
        the actual attribute-name constants `opentelemetry-semantic-
        conventions` ships, not strings typed by hand. Exported to a real
        file on disk by a real `SpanExporter`, openable like `ch31` and
        `ch34`'s own artifacts
    otel_nests_spans_via_contextvars_the_same_way_our_stack_does
        `tracer.start_as_current_span(...)` nests a tool span inside a
        model span inside a scenario span with no parent ever passed
        explicitly -- the same contextvar-propagation family `ch39` found
        LangGraph using for callbacks. The exported spans' parent/child
        `span_id` chain is checked directly against what it should be,
        the same chain `Trace._open`'s stack produces by hand
    a_broken_span_exporter_cannot_break_the_call
        the direct mirror of `ch39`'s sharpest finding, from the
        instrumentation side instead of the hook side: an exporter that
        raises on every export, and the real work underneath completes
        anyway -- because OTel's own span/exporter split was built so an
        exporter failing can never reach back into what it was only
        supposed to record
    usage_and_reasoning_tokens_map_to_the_spec
        `ch01`/`ch28`'s own input/output/reasoning token split, carried
        as real span attributes under `gen_ai.usage.input_tokens`,
        `gen_ai.usage.output_tokens`, and `gen_ai.usage.reasoning
        .output_tokens` -- present live, since a mock reply carries no
        `usage_metadata` to map in the first place

**Why OTel here and not a hand-mapped dictionary.** Every chapter since
`ch32` checks a hand-built claim against the real framework it names --
`StateGraph` against the hand-built loop, `BaseCallbackHandler` against
`ch18`'s hooks. A dictionary shaped like `gen_ai.*` would only prove that
strings can be typed correctly. A real `TracerProvider`, a real
`SpanExporter`, and the actual attribute constants the OpenTelemetry
project ships prove the mapping survives contact with the thing it is
claiming to be compatible with.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Trace

import json
from collections.abc import Sequence
from pathlib import Path
from typing import override

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes as gen_ai
from opentelemetry.trace import Tracer


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


OUT_ROOT = Path("out")


class JsonFileSpanExporter(SpanExporter):
    """Every exported span, one JSON object per line -- ch31's own
    snapshot format, for spans instead of messages.
    """

    def __init__(self, path: Path) -> None:
        self.path: Path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # A fresh file per run, not an ever-growing one across repeated
        # `just run` invocations -- this mirrors ch31's own "one snapshot
        # per run_id", minus the run_id, since a scenario's whole point
        # here is one export, not a resumable sequence of them.
        self.path.write_text("")

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self.path.open("a") as f:
            for span in spans:
                f.write(f"{span.to_json(indent=None)}\n")
        return SpanExportResult.SUCCESS

    @override
    def shutdown(self) -> None:
        pass


class RaisingSpanExporter(SpanExporter):
    """An exporter that fails every time -- the instrumentation-side
    mirror of ch39's FaultyCallback, to find out whether OTel's own
    architecture lets a broken exporter reach back into the call it was
    only supposed to record.
    """

    @override
    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        raise RuntimeError("exporter blew up on every export, on purpose")

    @override
    def shutdown(self) -> None:
        pass


def _tracer_writing_to(path: Path) -> Tracer:
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonFileSpanExporter(path)))
    return provider.get_tracer("agentic-primer")


def a_real_otel_span_carries_the_gen_ai_attributes(trace: Trace, model_kind: ModelKind) -> None:
    """A real model call, wrapped in a real OTel span -- gen_ai.system,
    gen_ai.request.model, and (live only) the usage split, using the
    actual attribute-name constants the semantic-conventions package
    ships rather than strings typed by hand.
    """
    name = "a_real_otel_span_carries_the_gen_ai_attributes"
    with trace.span("scenario", name=name) as span:
        snapshot_path = OUT_ROOT / "ch16_observability" / name / "spans.jsonl"
        tracer = _tracer_writing_to(snapshot_path)

        with (
            tracer.start_as_current_span("gen_ai.chat") as model_span,
            trace.span("turn", number=1),
        ):
            model_span.set_attribute(gen_ai.GEN_AI_SYSTEM, "deepseek")
            model_span.set_attribute(gen_ai.GEN_AI_REQUEST_MODEL, "deepseek-v4-flash")
            # The real call happens inside the span, not after it -- the
            # span's own start/end times what actually ran, the same way
            # ch01's model span always has.
            reply = ask_model(
                trace, model_kind, [AIMessage("hello.")], [HumanMessage("Say hello.")], []
            )
            usage = reply.usage_metadata
            if usage:
                model_span.set_attribute(gen_ai.GEN_AI_USAGE_INPUT_TOKENS, usage["input_tokens"])
                model_span.set_attribute(gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, usage["output_tokens"])

        span.add_note("otel_span_exported", path=str(snapshot_path))
        span.add_note(
            "attributes_used_real_constants",
            attribute_keys=[gen_ai.GEN_AI_SYSTEM, gen_ai.GEN_AI_REQUEST_MODEL],
            reply=str(reply.content),
            usage_present=usage is not None,
        )


def otel_nests_spans_via_contextvars_the_same_way_our_stack_does(
    trace: Trace, model_kind: ModelKind
) -> None:
    """No parent is ever passed to start_as_current_span -- the exported
    spans' parent_id/span_id chain is built entirely through a contextvar,
    the same propagation family ch39 found LangGraph using for callbacks.
    Checked directly against the chain it should form, not assumed.
    """
    name = "otel_nests_spans_via_contextvars_the_same_way_our_stack_does"
    with trace.span("scenario", name=name) as span:
        snapshot_path = OUT_ROOT / "ch16_observability" / name / "spans.jsonl"
        tracer = _tracer_writing_to(snapshot_path)

        # Real work inside every span, not attributes set on an empty
        # one -- a model call that actually asks for the tool, and a
        # real Tool.invoke() for the tool span to wrap.
        script = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            )
        ]
        with (
            tracer.start_as_current_span("scenario"),
            tracer.start_as_current_span("gen_ai.chat") as model_span,
            trace.span("turn", number=1),
        ):
            model_span.set_attribute(gen_ai.GEN_AI_SYSTEM, "deepseek")
            reply = ask_model(
                trace,
                model_kind,
                script,
                [HumanMessage("What's the status of A100?")],
                [check_order],
            )
            with tracer.start_as_current_span("gen_ai.tool.check_order") as tool_span:
                tool_span.set_attribute(gen_ai.GEN_AI_TOOL_NAME, "check_order")
                assert reply.tool_calls
                call = reply.tool_calls[0]
                tool_span.set_attribute(gen_ai.GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(call["args"]))
                execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))

        exported = [json.loads(line) for line in snapshot_path.read_text().splitlines()]
        by_name = {e["name"]: e for e in exported}
        tool_parent_is_model = (
            by_name["gen_ai.tool.check_order"]["parent_id"]
            == by_name["gen_ai.chat"]["context"]["span_id"]
        )
        model_parent_is_scenario = (
            by_name["gen_ai.chat"]["parent_id"] == by_name["scenario"]["context"]["span_id"]
        )
        scenario_has_no_parent = by_name["scenario"]["parent_id"] is None
        same_trace = len({e["context"]["trace_id"] for e in exported}) == 1

        span.add_note("otel_spans_exported", path=str(snapshot_path))
        span.add_note(
            "the_chain_matches_our_own_stacks_nesting",
            tool_parent_is_model=tool_parent_is_model,
            model_parent_is_scenario=model_parent_is_scenario,
            scenario_has_no_parent=scenario_has_no_parent,
            all_spans_share_one_trace_id=same_trace,
        )


def a_broken_span_exporter_cannot_break_the_call(trace: Trace, model_kind: ModelKind) -> None:
    """ch39's sharpest finding, from the instrumentation side: an
    exporter that raises on every export -- and the real model call
    underneath completes anyway. Delete or break this exporter and the
    agent's behavior does not change, which is exactly ch16's own
    definition of what instrumentation is supposed to be.
    """
    name = "a_broken_span_exporter_cannot_break_the_call"
    with trace.span("scenario", name=name) as span:
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(RaisingSpanExporter()))
        tracer = provider.get_tracer("agentic-primer")

        raised = False
        reply_content = ""
        try:
            with tracer.start_as_current_span("gen_ai.chat") as model_span:
                model_span.set_attribute(gen_ai.GEN_AI_SYSTEM, "deepseek")
                with trace.span("turn", number=1):
                    reply = ask_model(
                        trace,
                        model_kind,
                        [AIMessage("fine, despite the exporter.")],
                        [HumanMessage("go")],
                        [],
                    )
                reply_content = str(reply.content)
        except RuntimeError:
            raised = True

        span.add_note(
            "the_call_completed_despite_the_broken_exporter",
            raised=raised,
            reply=reply_content,
        )


def usage_and_reasoning_tokens_map_to_the_spec(trace: Trace, model_kind: ModelKind) -> None:
    """ch01/ch28's own input/output/reasoning split, carried as real span
    attributes under the spec's own gen_ai.usage.* keys. Mock carries no
    usage_metadata at all -- there is nothing to map in the first place,
    which is itself the honest finding, not a gap in this scenario.
    """
    name = "usage_and_reasoning_tokens_map_to_the_spec"
    with trace.span("scenario", name=name) as span:
        snapshot_path = OUT_ROOT / "ch16_observability" / name / "spans.jsonl"
        tracer = _tracer_writing_to(snapshot_path)

        with trace.span("turn", number=1):
            reply = ask_model(
                trace,
                model_kind,
                [AIMessage("A short reply, to keep reasoning tokens visible.")],
                [HumanMessage("Explain briefly why the sky is blue.")],
                [],
            )

        usage = reply.usage_metadata
        with tracer.start_as_current_span("gen_ai.chat") as model_span:
            if usage:
                model_span.set_attribute(gen_ai.GEN_AI_USAGE_INPUT_TOKENS, usage["input_tokens"])
                model_span.set_attribute(gen_ai.GEN_AI_USAGE_OUTPUT_TOKENS, usage["output_tokens"])
                reasoning = (usage.get("output_token_details") or {}).get("reasoning", 0)
                model_span.set_attribute(gen_ai.GEN_AI_USAGE_REASONING_OUTPUT_TOKENS, reasoning)

        span.add_note("otel_span_exported", path=str(snapshot_path))
        span.add_note(
            "usage_mapped_under_mock_only_when_present",
            usage_present=usage is not None,
            usage=dict(usage) if usage else None,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch16_observability", model_kind=model_kind)

    a_real_otel_span_carries_the_gen_ai_attributes(trace, model_kind)
    otel_nests_spans_via_contextvars_the_same_way_our_stack_does(trace, model_kind)
    a_broken_span_exporter_cannot_break_the_call(trace, model_kind)
    usage_and_reasoning_tokens_map_to_the_spec(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
