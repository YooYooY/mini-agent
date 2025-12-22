from __future__ import annotations

from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph


# ========= Constant =========
Route = Literal[
    "direct_answer",
    "needs_thinking",
    "continue_thinking",
    "end",
]

MAX_STEPS = 3


# ========= State =========
class AgentState(TypedDict):
    input: str
    messages: list[str]
    step_count: int
    route: Route


# ========= Nodes =========
def analyze_node(state: AgentState) -> dict:
    messages = state["messages"] + [f"[analyze] analyzing input: {state['input']}"]

    route: Route = "needs_thinking"

    return {
        "messages": messages,
        "route": route,
        "step_count": state["step_count"] + 1,
    }


def thinking_node(state: AgentState) -> dict:
    messages = state["messages"] + ["[thinking] reasoning about the problem..."]

    return {
        "messages": messages,
        "step_count": state["step_count"] + 1,
    }


def decide_next_node(state: AgentState) -> dict:
    """
    decide：continue or end
    """
    if state["step_count"] >= MAX_STEPS:
        route: Route = "end"
        messages = state["messages"] + ["[decide] reached max steps, finishing"]
    else:
        route = "continue_thinking"
        messages = state["messages"] + ["[decide] need more thinking"]

    return {
        "messages": messages,
        "route": route,
        "step_count": state["step_count"] + 1,
    }


# ========= Routers =========
def route_after_decide(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    # Nodes
    graph.add_node("analyze_node", analyze_node)
    graph.add_node("thinking_node", thinking_node)
    graph.add_node("decide_node", decide_next_node)

    # Entry
    graph.add_edge(START, "analyze_node")

    # analyze → thinking
    graph.add_edge("analyze_node", "thinking_node")

    # thinking → decide
    graph.add_edge("thinking_node", "decide_node")

    # conditional loop / end
    graph.add_conditional_edges(
        "decide_node",
        route_after_decide,
        {
            "continue_thinking": "analyze_node",
            "end": END,
        },
    )

    return graph.compile()


# ========= Run =========
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "input": "Explain how LangGraph enables agent loops",
            "messages": [],
            "step_count": 0,
            "route": "needs_thinking",
        }
    )

    print("\n=== FINAL STATE ===")
    for m in result["messages"]:
        print(m)
