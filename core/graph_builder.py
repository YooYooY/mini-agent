from langgraph.graph import StateGraph, END
from core.types import TaskState

from planner.planner_node import planner_node
from retriever.retriever_node import retriever_node
from executor.executor_node import executor_node
from critic.critic_node import critic_node
from fallback.fail_answer_node import fail_answer_node


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("fail_answer_node", fail_answer_node)

    graph.set_entry_point("planner_node")

    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        lambda s: {
            "pass": "end",
            "revise": "retriever_node",
            "fail": "fail_answer_node",
        }[s["critic_result"]["status"]],
        {
            "retriever_node": "retriever_node",
            "fail_answer_node": "fail_answer_node",
            "end": END,
        },
    )

    return graph.compile()
