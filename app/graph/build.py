"""Assemble and compile the LangGraph state machine.

Topology:

    START
      -> guard ---------------(suspicious)------------------> END
           |(clean)
           v
      classify_intent
           |-- qa     --> retrieve --> generate_response --> self_critique --> END
           |-- sql    --> sql_agent ------------> generate_response --> self_critique --> END
           |-- action --> action_tool  (interrupts for human approval) ------------> END

The compiled graph uses a `MemorySaver` checkpointer so (a) conversation state
persists across turns per thread_id and (b) the human-in-the-loop `interrupt()`
in `action_tool` can pause and later resume.
"""
from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.graph import nodes
from app.graph.router import route_by_intent
from app.graph.state import AgentState


def build_graph():
    g = StateGraph(AgentState)

    g.add_node("guard", nodes.guard)
    g.add_node("classify_intent", nodes.classify_intent_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("sql_agent", nodes.sql_agent_node)
    g.add_node("action_tool", nodes.action_tool_node)
    g.add_node("generate_response", nodes.generate_response_node)
    g.add_node("self_critique", nodes.self_critique_node)

    g.add_edge(START, "guard")
    g.add_conditional_edges(
        "guard",
        nodes.route_after_guard,
        {"blocked": END, "classify_intent": "classify_intent"},
    )
    g.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {"qa": "retrieve", "sql": "sql_agent", "action": "action_tool"},
    )
    g.add_edge("retrieve", "generate_response")
    g.add_edge("sql_agent", "generate_response")
    g.add_edge("generate_response", "self_critique")
    g.add_edge("self_critique", END)
    g.add_edge("action_tool", END)

    return g


@lru_cache(maxsize=1)
def get_graph():
    """Compiled graph singleton with an in-memory checkpointer."""
    return build_graph().compile(checkpointer=MemorySaver())
