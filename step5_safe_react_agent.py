from typing import Literal, TypedDict
from huggingface_hub import Agent
from langgraph.graph import START, END, StateGraph

# ========= Constant =========
Route = Literal["continue", "end"]
ActionType = Literal["search", "calculator", "none"]

MAX_STEPS = 10
MAX_TOOL_ERRORS = 2


# ========= State =========
class AgentState(TypedDict):
    input: str
    messages: list[str]
    step_count: int
    route: Route

    thought: str
    action: ActionType
    observation: str

    tool_error_count: int
    final_answer: str | None


# ========= Mock Tools =========
def search_tool(query: str) -> str:
    if "fail" in query:
        raise RuntimeError("Search tool failed")
    return f"[tool: search] Found info about {query}"


def calculator_tool(expr: str) -> str:
    return f"[tool: calculator] Result of '{expr}' is 42"


# ========= Nodes =========
def thinking_node(state: AgentState) -> dict:
    thought = f"I am thinking about: {state["input"]}"

    messages = state["messages"] + [f"[thought] {thought}"]

    step_count = state["step_count"] + 1

    if "calculate" in thought:
        action = "calculator"
    elif "search" in thought:
        action = "search"
    else:
        action = "none"

    return {"messages": messages, "step_count": step_count, "action": action}


def action_node(state: AgentState) -> dict:

    action = state["action"]
    input = state["input"]
    tool_error_count = state["tool_error_count"]

    try:
        if action == "calculator":
            observation = calculator_tool(input)
        elif action == "search":
            observation = search_tool(input)
        else:
            observation = "No tool needed."

    except Exception as e:
        observation = f"[tool_error] {str(e)}"
        tool_error_count = tool_error_count + 1

    messages = state["messages"] + [f"[action] {action}"]

    step_count = state["step_count"] + 1

    return {
        "messages": messages,
        "observation": observation,
        "step_count": step_count,
        "tool_error_count": tool_error_count,
    }


def observation_node(state: AgentState) -> dict:
    messages = state["messages"] + [f"[observation] {state['observation']}"]

    if "Found info" in state["observation"]:
        final_answer = "Here is the answer base on search results."
    else:
        final_answer = None

    step_count = state["step_count"] + 1

    return {
        "messages": messages,
        "final_answer": final_answer,
        "step_count": step_count,
    }


def decide_node(state: AgentState) -> dict:
    messages = state["messages"]

    if state["final_answer"] is not None:
        route: Route = "end"
        messages = messages + ["[decide] I have a final answer."]
    elif state["step_count"] >= MAX_STEPS:
        route: Route = "end"
        messages = messages + ["[decide] Max steps reached. Fallback."]
    elif state["tool_error_count"] >= MAX_TOOL_ERRORS:
        route: Route = "end"
        messages = messages + ["[decide] Too many tool errors. Fallback."]
    else:
        route = "continue"
        messages = messages + ["[decide] Continue reasoning."]

    return {
        "messages": messages,
        "route": route,
        "step_count": state["step_count"] + 1,
    }


def route_after_decide(state: AgentState) -> Route:
    return state["route"]


# ========= Build Graph =========
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("thinking_node", thinking_node)
    graph.add_node("action_node", action_node)
    graph.add_node("observation_node", observation_node)
    graph.add_node("decide_node", decide_node)

    graph.add_edge(START, "thinking_node")
    graph.add_edge("thinking_node", "action_node")
    graph.add_edge("action_node", "observation_node")
    graph.add_edge("observation_node", "decide_node")

    graph.add_conditional_edges(
        "decide_node", route_after_decide, {"continue": "thinking_node", "end": END}
    )

    return graph.compile()


# ========= Run =========
if __name__ == "__main__":
    app = build_graph()

    result = app.invoke(
        {
            "input": "search LangGraph success agent example",
            # "input": "search LangGraph fail agent example",
            "messages": [],
            "step_count": 0,
            "route": "continue",
            "thought": "",
            "action": "none",
            "observation": "",
            "tool_error_count": 0,
            "final_answer": None,
        }
    )

    print("\n=== FINAL TRACE ===")
    for msg in result["messages"]:
        print(msg)

    print("\nFINAL ANSWER:", result["final_answer"])
