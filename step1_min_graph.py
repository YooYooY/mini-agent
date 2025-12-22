from typing import Literal, TypedDict
from langgraph.graph import START, END, StateGraph


class State(TypedDict):
    text: str
    mode: Literal["question", "shout"]


def prepare(state: State) -> dict:
    raw = state["text"].strip()
    mode = "question" if raw.endswith("?") else "shout"
    return {"text": raw, "mode": mode}


def shout_node(state: State) -> dict:
    return {"text": state["text"].upper() + "!!!"}


def question_node(state: State) -> dict:
    return {"text": state["text"] + " 🤔"}


def route_condition(state: State) -> Literal["shout", "question"]:
    return state["mode"]


def build_graph():
    graph = StateGraph(State)

    graph.add_node("prepare_node", prepare)
    graph.add_node("shout_node", shout_node)
    graph.add_node("question_node", question_node)

    graph.add_edge(START, "prepare_node")

    graph.add_conditional_edges(
        "prepare_node",
        route_condition,
        {"shout": "shout_node", "question": "question_node"},
    )

    graph.add_edge("shout_node", END)
    graph.add_edge("question_node", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()

    out1 = app.invoke({"text": "Hello LangGraph", "mode": "shout"})

    print("CASE 1 OUTPUT", out1)

    out2 = app.invoke({"text": "What is LangGraph?", "mode": "shout"})

    print("CASE 2 OUTPUT", out2)
