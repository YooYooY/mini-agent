from typing import Literal, TypedDict
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


class CriticResult(TypedDict):
    status: Literal["pass", "revise", "fail"]
    reason: str
    critic_count: int


class TaskState(TypedDict):
    task_id: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    execution_trace: list[ExecutionTrace]
    critic_result: CriticResult


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
        "critic_result": {"critic_count": 0, "status": "pass", "reason": ""},
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

    query = "订单查询 API"

    hits = []

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


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    hits = task_memory["retrieval_context"]["retriever_hits"]
    trace = task_memory["execution_trace"]
    critic_count = task_memory["critic_result"]["critic_count"]

    problems = []

    if len(hits) == 0:
        problems.append("retriever returned no documents")

    executed_steps = [t["step"] for t in trace]
    if "executor" not in executed_steps:
        problems.append("executor was never called")

    if state.get("answer") is None:
        problems.append("no answer was generated")

    if critic_count >= 2:
        critic = {
            "status": "fail",
            "reason": "critic count exceed",
            "critic_count": critic_count,
        }
    elif problems:
        critic = {
            "status": "revise",
            "reason": "; ".join(problems),
            "critic_count": critic_count + 1,
        }
    else:
        critic = {
            "status": "pass",
            "reason": "pipleine executed correctly",
            "critic_count": 0,
        }

    task_memory["critic_result"] = critic
    state["critic_result"] = critic

    return state


def fail_answer_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    critic = task_memory["critic_result"]

    reason = critic.get("reason", "unknown error")

    answer = f"⚠️ 当前查询未能成功处理（已终止）\n原因：{reason}"

    append_trace(
        task_id=task_id,
        step="fail_answer",
        tool="system_fallback",
        input_data={"critic": critic},
        output={"answer": answer},
    )

    state["answer"] = answer
    return state


def route_after_critic(state: TaskState) -> str:
    status = state["critic_result"]["status"]

    if status == "pass":
        return "end"

    if status == "revise":
        return "retriever"

    if status == "fail":
        return "fail_answer"


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
        route_after_critic,
        {
            "retriever": "retriever_node",
            "fail_answer": "fail_answer_node",
            "end": END,
        },
    )

    graph.add_edge("fail_answer_node", END)

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
            "critic_result": {"critic_count": 0, "status": "pass", "reason": ""},
        }
    )

    print("\n=== 最终答案 ===")
    print(result["answer"])

    print("\n=== 任务级 Memory ===")
    print(memory_store[task_id])

    print("\n=== Execution Trace ===")
    for step in memory_store[task_id]["execution_trace"]:
        print(step)
