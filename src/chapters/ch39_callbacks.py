# Copyright (c) 2022-2026 345 Consulting, LLC
# Proprietary and Confidential. All rights reserved.

"""The audit trail and the hook are the same mechanism in LangChain.

`ch18_hooks` built a taxonomy from nothing -- observe, modify, veto, in
increasing order of danger -- and warned, in prose, that a framework
implementing tracing through the same mechanism as its hooks risks a
hook silently suppressing the audit trail. `ch16_observability` was
skipped rather than written because that warning was untested. This
chapter tests it directly, against `langchain_core`'s real
`BaseCallbackHandler`.

Six scenarios:

    a_tracer_observes_without_participating
        a callback handler attached to a real StateGraph -- ch32's own
        call_model/call_tool shape, with call_tool dispatching a real
        Tool object -- recording chain, tool, and model start/end events.
        A bare model.invoke() only ever fires the model pair; this is the
        graph this primer already built, checked against the full set of
        events instead of the two easiest ones
    callbacks_propagate_ambiently_without_being_passed
        found live: a node function that calls model.invoke() with no
        config argument at all still gets instrumented, because
        LangGraph threads the callback list through a contextvar. This
        primer's own Trace never does this -- every function here takes
        `trace` as a real parameter, because ambient propagation is
        exactly the kind of hidden mechanism this primer exists to avoid
    a_default_callback_error_is_isolated
        a handler that raises does not crash the run and does not stop
        a second handler from firing -- the safe default
    a_tracer_can_turn_a_completed_answer_into_a_crash
        found live: raise_error=True on a handler that fails in
        on_llm_end -- after the model has already answered -- aborts the
        whole call and erases a legitimate tracer registered after it.
        The workflow's outcome should never depend on whether recording
        it succeeded; this is the one flag that makes it depend on
        exactly that
    a_callback_can_only_observe_or_abort_never_modify
        found live: a callback that mutates the messages it is handed
        has zero effect on what the model actually processes -- of
        ch18's three powers, callbacks only ever have two
    a_broken_handler_can_replace_the_real_tool_error_with_its_own
        found live: on_tool_error is a distinct event from on_tool_start/
        on_tool_end -- it fires when the tool itself raises, not when a
        handler does. By default it is isolated exactly like scenario
        three. With raise_error=True it is worse than scenario four: the
        caller does not just lose the audit trail, the exception that
        propagates out is the broken handler's own error, not the tool's
        -- ch09_who_retries' own classification (transient vs permanent
        tool failure) is exactly what a caller loses the ability to make
        when the failure they see is not the failure that happened

**The actual fix for scenario four is not a workaround, it is a rule.**
Register the tracer first in the callbacks list -- handlers dispatch in
list order per event, and only handlers *after* a raising one in that
same dispatch are skipped, so a tracer registered earlier still records
before a later handler aborts anything. And do not set raise_error=True
on a handler that was not deliberately built as a circuit-breaker. But
the deeper answer is architectural, not a convention to remember
correctly every time: `support/trace.py`'s `Trace` was never made a peer
in a shared, order-sensitive, exception-vulnerable handler list the way
a `BaseCallbackHandler` tracer is. There is no list for a broken hook to
starve, because the audit trail was never sharing a mechanism with
anything that has veto power in the first place.
"""

from support.agent import ask_model, execute_tool
from support.trace import ModelKind, Trace

import uuid
from typing import Annotated, TypedDict, override

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


@tool
def check_order(order_id: str) -> str:
    """Check the status of an order by id."""
    return f"order {order_id}: shipped"


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _route_on_tool_calls(state: State) -> str:
    last = state["messages"][-1]
    assert isinstance(last, AIMessage)
    return "call_tool" if last.tool_calls else END


def a_tracer_observes_without_participating(trace: Trace, model_kind: ModelKind) -> None:
    """A callback handler attached to a real StateGraph -- the same
    call_model/call_tool shape ch32 through ch38 already built -- with
    call_tool dispatching a real Tool object through check_order.invoke()
    instead of a plain function call. A bare model.invoke() only ever
    fires the model pair; on_chain_start and on_tool_start need an actual
    graph and an actual Tool invoked through it, which this already is.
    ch16's claim, checked against the full set of events instead of the
    two easiest ones.
    """
    name = "a_tracer_observes_without_participating"
    with trace.span("scenario", name=name) as span:
        events: list[str] = []

        class CallbackRecorder(BaseCallbackHandler):
            @override
            def on_chain_start(self, *_a: object, **_k: object) -> None:
                events.append("on_chain_start")

            @override
            def on_tool_start(self, *_a: object, **_k: object) -> None:
                events.append("on_tool_start")

            @override
            def on_tool_end(self, *_a: object, **_k: object) -> None:
                events.append("on_tool_end")

            @override
            def on_chat_model_start(self, *_a: object, **_k: object) -> None:
                events.append("on_chat_model_start")

            @override
            def on_llm_end(self, *_a: object, **_k: object) -> None:
                events.append("on_llm_end")

            @override
            def on_chain_end(self, *_a: object, **_k: object) -> None:
                events.append("on_chain_end")

        script = [
            AIMessage(
                "", tool_calls=[{"name": "check_order", "args": {"order_id": "A100"}, "id": "c1"}]
            ),
            AIMessage("Order A100 has shipped."),
        ]
        turn = [0]

        def call_model(state: State) -> State:
            turn[0] += 1
            with trace.span("turn", number=turn[0]) as turn_span:
                before = len(events)
                reply = ask_model(
                    trace, model_kind, [script[turn[0] - 1]], state["messages"], [check_order]
                )
                turn_span.add_note("callbacks_fired_here", events=events[before:])
            return {"messages": [reply]}

        def call_tool(state: State) -> State:
            with trace.span("turn", number=turn[0]) as turn_span:
                before = len(events)
                last = state["messages"][-1]
                assert isinstance(last, AIMessage)
                call = last.tool_calls[0]
                message = execute_tool(call, trace, lambda _n, a, _s: check_order.invoke(a))
                turn_span.add_note("callbacks_fired_here", events=events[before:])
                return {"messages": [message]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_node("call_tool", call_tool)
        graph.add_edge(START, "call_model")
        graph.add_conditional_edges(
            "call_model", _route_on_tool_calls, {"call_tool": "call_tool", END: END}
        )
        graph.add_edge("call_tool", "call_model")
        compiled = graph.compile()

        result = compiled.invoke(
            {
                "messages": [
                    HumanMessage("What's the status of order A100?"),
                ]
            },
            config={"callbacks": [CallbackRecorder()]},
        )

        span.add_note(
            "the_tracer_saw_every_edge",
            saw_chain="on_chain_start" in events and "on_chain_end" in events,
            saw_tool="on_tool_start" in events and "on_tool_end" in events,
            saw_model="on_chat_model_start" in events and "on_llm_end" in events,
            reply=str(result["messages"][-1].content),
        )


def callbacks_propagate_ambiently_without_being_passed(trace: Trace, model_kind: ModelKind) -> None:
    """A node function calls model.invoke() with no config at all --
    LangGraph threads the graph's own callbacks in through a contextvar,
    invisibly. This primer's Trace has never worked this way: `trace` is
    a real parameter every function accepts, on purpose.
    """
    name = "callbacks_propagate_ambiently_without_being_passed"
    with trace.span("scenario", name=name) as span:
        events: list[str] = []

        class CallbackRecorder(BaseCallbackHandler):
            @override
            def on_chat_model_start(self, *_a: object, **_k: object) -> None:
                events.append("on_chat_model_start")

        def call_model(state: State) -> State:
            # No config, no callbacks argument -- nothing here mentions
            # the CallbackRecorder at all. ask_model's own config
            # parameter is left at its default, None.
            with trace.span("turn", number=1) as turn_span:
                turn_span.add_note("callbacks_before_the_call", events=list(events))
                reply = ask_model(
                    trace, model_kind, [AIMessage("graph reply.")], state["messages"], []
                )
                turn_span.add_note("callbacks_after_the_call", events=list(events))
            return {"messages": [reply]}

        graph = StateGraph(State)
        graph.add_node("call_model", call_model)
        graph.add_edge(START, "call_model")
        graph.add_edge("call_model", END)
        compiled = graph.compile()

        compiled.invoke(
            {"messages": [HumanMessage("hi")]}, config={"callbacks": [CallbackRecorder()]}
        )

        span.add_note(
            "the_node_never_mentioned_the_recorder",
            events=events,
            recorder_fired_anyway="on_chat_model_start" in events,
        )


def a_default_callback_error_is_isolated(trace: Trace, model_kind: ModelKind) -> None:
    """A handler that raises, by default -- raise_error unset, so False.
    The exception is logged and swallowed; a second, well-behaved handler
    still fires, and the call itself still succeeds.
    """
    name = "a_default_callback_error_is_isolated"
    with trace.span("scenario", name=name) as span:
        events: list[str] = []

        class FaultyCallback(BaseCallbackHandler):
            @override
            def on_llm_end(self, *_a: object, **_k: object) -> None:
                events.append("faulty:on_llm_end")
                raise RuntimeError("faulty handler blew up")

        class TracingCallback(BaseCallbackHandler):
            @override
            def on_llm_end(self, *_a: object, **_k: object) -> None:
                events.append("tracer:on_llm_end")

        raised = False
        reply = ""
        try:
            with trace.span("turn", number=1) as turn_span:
                turn_span.add_note("callbacks_before_the_call", events=list(events))
                result = ask_model(
                    trace,
                    model_kind,
                    [AIMessage("fine.")],
                    [HumanMessage("go")],
                    [],
                    config={"callbacks": [FaultyCallback(), TracingCallback()]},
                )
                turn_span.add_note("callbacks_after_the_call", events=list(events))
            reply = str(result.content)
        except RuntimeError:
            raised = True

        span.add_note(
            "the_call_survived_the_broken_handler",
            raised=raised,
            reply=reply,
            tracer_still_fired="tracer:on_llm_end" in events,
        )


def a_tracer_can_turn_a_completed_answer_into_a_crash(trace: Trace, model_kind: ModelKind) -> None:
    """raise_error=True changes what "isolated" meant in the scenario
    above. Faulty fails in on_llm_end -- after the model has already
    produced a real answer -- and the exception propagates out of
    invoke() itself. Tracer, registered after Faulty, never records the
    call at all: not because nothing happened, but because something did
    and the record of it was erased on the way out.

    This is the failure ch16's own test says it should never be able to
    cause: delete or break the tracer and the agent's behavior changed,
    from "answered" to "crashed." A workflow's outcome must never depend
    on whether recording it succeeded, and this flag is the one place in
    the real API where it does.
    """
    name = "a_tracer_can_turn_a_completed_answer_into_a_crash"
    with trace.span("scenario", name=name) as span:
        events: list[str] = []

        class FaultyCallback(BaseCallbackHandler):
            raise_error: bool = True

            @override
            def on_llm_end(self, *_a: object, **_k: object) -> None:
                events.append("faulty:on_llm_end")
                # The model has already produced its reply by the time
                # this fires -- on_llm_end runs after generation, not
                # before it. Raising here does not stop an answer from
                # existing; it stops anyone downstream from finding out.
                raise RuntimeError("faulty handler blew up after the model answered")

        class TracingCallback(BaseCallbackHandler):
            @override
            def on_llm_end(self, *_a: object, **_k: object) -> None:
                events.append("tracer:on_llm_end")

        raised = False
        with trace.span("turn", number=1) as turn_span:
            turn_span.add_note("callbacks_before_the_call", events=list(events))
            try:
                ask_model(
                    trace,
                    model_kind,
                    [AIMessage("the real answer.")],
                    [HumanMessage("go")],
                    [],
                    config={"callbacks": [FaultyCallback(), TracingCallback()]},
                )
            except RuntimeError:
                raised = True
            finally:
                turn_span.add_note("callbacks_after_the_call", events=list(events))

        span.add_note(
            "a_completed_answer_became_an_unrecorded_crash",
            raised=raised,
            faulty_fired="faulty:on_llm_end" in events,
            tracer_suppressed="tracer:on_llm_end" not in events,
        )


def a_callback_can_only_observe_or_abort_never_modify(trace: Trace, model_kind: ModelKind) -> None:
    """A callback that mutates the messages it is handed has no effect on
    what the model actually processes -- of ch18's three powers, a
    callback only ever has two. HookVerdict.replacement has no equivalent
    here; a callback's return value is discarded outright.
    """
    name = "a_callback_can_only_observe_or_abort_never_modify"
    with trace.span("scenario", name=name) as span:
        events: list[str] = []

        class MeddlingCallback(BaseCallbackHandler):
            @override
            def on_chat_model_start(
                self,
                serialized: dict[str, object],
                messages: list[list[BaseMessage]],
                *,
                run_id: uuid.UUID,
                parent_run_id: uuid.UUID | None = None,
                tags: list[str] | None = None,
                metadata: dict[str, object] | None = None,
                **kwargs: object,
            ) -> str:
                events.append("meddler:on_chat_model_start")
                # Attempt to blank out the prompt the model is about to
                # see, and return something -- both are ignored.
                messages[0].clear()
                return "this return value is discarded"

        with trace.span("turn", number=1) as turn_span:
            turn_span.add_note("callbacks_before_the_call", events=list(events))
            result = ask_model(
                trace,
                model_kind,
                [AIMessage("saw the real prompt.")],
                [HumanMessage("the real, unmutated question")],
                [],
                config={"callbacks": [MeddlingCallback()]},
            )
            turn_span.add_note("callbacks_after_the_call", events=list(events))

        span.add_note(
            "the_mutation_attempt_changed_nothing",
            reply=str(result.content),
        )


@tool
def broken_order_lookup(order_id: str) -> str:
    """A tool that always fails -- the real failure this scenario needs
    a handler to obscure.
    """
    raise RuntimeError(f"order lookup backend is down, order_id={order_id}")


def a_broken_handler_can_replace_the_real_tool_error_with_its_own(
    trace: Trace, _model_kind: ModelKind
) -> None:
    """on_tool_error is its own event, distinct from on_tool_start/
    on_tool_end -- it fires when the tool itself raises. By default it
    is isolated the same safe way as scenario three: both handlers
    record it, and the tool's own exception is what propagates.

    With raise_error=True, the failure mode is sharper than scenario
    four. There the tracer was merely erased; here the exception the
    caller actually catches is the broken handler's own RuntimeError,
    not the tool's -- ch09_who_retries' whole classification (was this
    transient or permanent, retry or don't) is unanswerable when the
    exception in hand does not describe what actually failed.
    """
    name = "a_broken_handler_can_replace_the_real_tool_error_with_its_own"
    with trace.span("scenario", name=name) as span:
        default_events: list[str] = []

        class FaultyOnError(BaseCallbackHandler):
            @override
            def on_tool_error(self, *_a: object, **_k: object) -> None:
                default_events.append("faulty:on_tool_error")
                raise RuntimeError("faulty handler blew up handling the tool error")

        class TracingCallback(BaseCallbackHandler):
            @override
            def on_tool_error(self, *_a: object, **_k: object) -> None:
                default_events.append("tracer:on_tool_error")

        call: ToolCall = {"name": "broken_order_lookup", "args": {"order_id": "A100"}, "id": "c1"}

        default_error = ""
        with trace.span("turn", number=1) as turn_span:
            turn_span.add_note("callbacks_before_the_call", events=list(default_events))
            try:
                execute_tool(
                    call,
                    trace,
                    lambda _n, a, _s: broken_order_lookup.invoke(
                        a, config={"callbacks": [FaultyOnError(), TracingCallback()]}
                    ),
                )
            except RuntimeError as exc:
                default_error = str(exc)
            finally:
                turn_span.add_note("callbacks_after_the_call", events=list(default_events))

        raising_events: list[str] = []

        class RaisingFaultyOnError(BaseCallbackHandler):
            raise_error: bool = True

            @override
            def on_tool_error(self, *_a: object, **_k: object) -> None:
                raising_events.append("faulty:on_tool_error")
                raise RuntimeError("faulty handler blew up handling the tool error")

        class RaisingTracingCallback(BaseCallbackHandler):
            @override
            def on_tool_error(self, *_a: object, **_k: object) -> None:
                raising_events.append("tracer:on_tool_error")

        raising_error = ""
        with trace.span("turn", number=2) as turn_span:
            turn_span.add_note("callbacks_before_the_call", events=list(raising_events))
            try:
                execute_tool(
                    call,
                    trace,
                    lambda _n, a, _s: broken_order_lookup.invoke(
                        a,
                        config={"callbacks": [RaisingFaultyOnError(), RaisingTracingCallback()]},
                    ),
                )
            except RuntimeError as exc:
                raising_error = str(exc)
            finally:
                turn_span.add_note("callbacks_after_the_call", events=list(raising_events))

        span.add_note(
            "default_is_isolated",
            tracer_still_fired="tracer:on_tool_error" in default_events,
            caller_saw_the_real_tool_error="order lookup backend is down" in default_error,
        )
        span.add_note(
            "raise_error_substitutes_the_wrong_exception",
            tracer_suppressed="tracer:on_tool_error" not in raising_events,
            caller_saw_the_real_tool_error="order lookup backend is down" in raising_error,
            caller_saw_the_handlers_own_error="faulty handler blew up" in raising_error,
        )


def run(model_kind: ModelKind = "mock") -> Trace:
    trace = Trace(chapter="ch39_callbacks", model_kind=model_kind)

    a_tracer_observes_without_participating(trace, model_kind)
    callbacks_propagate_ambiently_without_being_passed(trace, model_kind)
    a_default_callback_error_is_isolated(trace, model_kind)
    a_tracer_can_turn_a_completed_answer_into_a_crash(trace, model_kind)
    a_callback_can_only_observe_or_abort_never_modify(trace, model_kind)
    a_broken_handler_can_replace_the_real_tool_error_with_its_own(trace, model_kind)

    trace.close(turns=0, messages=0, ended="n/a")
    return trace
