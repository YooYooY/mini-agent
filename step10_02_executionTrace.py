from typing import TypedDict
import uuid
from langgraph.graph import StateGraph, END


class IntentContext(TypedDict):
    topic: str
    intent: str
    task_plan: list


class RetrievalContext(TypedDict):
    doc_scope: list
    retriever_hits: list


class ExecutionTrace(TypedDict):
    step: str
    tool: str
    input: dict
    output: dict
    status: str
    error: str | None


class TaskState(TypedDict):
    task_id: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    executionTrace: list[ExecutionTrace]


memory_store = {}


def append_trace(
    task_id, step, tool, input_data, output=None, status="success", error=None
):
    trace_item = {
        "step": step,
        "tool": tool,
        "input": input_data,
        "output": output,
        "status": status,
        "error": error,
    }

    memory_store[task_id]["execution_trace"].append(trace_item)


def init_task_memory():
    task_id = str(uuid.uuid4())
    memory_store[task_id] = {
        "task_meta": {"task_id": task_id},
        "intent_context": {},
        "retrieval_context": {},
        "execution_trace": [],
    }
    return task_id


def planner_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    topic = "API / orders"
    intent = "查询接口说明并生成示例代码"
    task_plan = ["检索文档", "抽取参数", "生成代码示例"]

    task_memory["intent_context"] = {
        "topic": topic,
        "intent": intent,
        "task_plan": task_plan,
    }

    state["intent_context"] = task_memory["intent_context"]

    return state


def retriever_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    query = "订单查询 API"  # demo 模拟

    hits = [
        {"doc_id": "api_orders_query", "score": 0.92},
        {"doc_id": "api_orders_common", "score": 0.88},
        {"doc_id": "auth_intro", "score": 0.75},
    ]

    task_memory["retrieval_context"] = {
        "doc_scope": ["orders", "api"],
        "retriever_hits": hits,
    }

    append_trace(
        task_id=task_id,
        step="retriever",
        tool="doc_retriever",
        input_data={"query", query},
        output={"hits": hits},
    )

    state["retrieval_context"] = task_memory["retrieval_context"]

    return state


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent = task_memory["intent_context"]["intent"]
    hits = task_memory["retrieval_context"]["retriever_hits"]

    answer = f"根据 {intent}，基于 {len(hits)} 个文档生成回答"

    append_trace(
        task_id=task_id,
        step="executor",
        tool="answer_generator",
        input_data={"intent": intent, "hits": hits},
        output={"answer": answer},
    )

    state["answer"] = answer

    return state


def build_graph():
    graph = StateGraph(TaskState)
    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)

    graph.set_entry_point("planner_node")

    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", END)

    return graph.compile()


if __name__ == "__main__":

    app = build_graph()

    task_id = init_task_memory()

    result = app.invoke(
        {
            "task_id": task_id,
            "intent_context": {},
            "retrieval_context": {},
            "answer": "",
        }
    )

    print("\n=== 最终答案 ===")
    print(result["answer"])

    print("\n=== 任务级 Memory ===")
    print(memory_store[task_id])

    print("\n=== Execution Trace ===")
    for step in memory_store[task_id]["execution_trace"]:
        print(step)
