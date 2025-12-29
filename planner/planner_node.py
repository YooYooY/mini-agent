from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import IntentContext, TaskState


def planner_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    user_query = state["user_query"]

    if "订单" in user_query or "order" in user_query.lower():
        topic = "订单 API"
        intent = "查询订单相关接口信息并生成示例代码"
    else:
        topic = "未知主题"
        intent = f"查询与「{user_query}」相关的信息"

    intent_context: IntentContext = {
        "topic": topic,
        "intent": intent,
        "task_plan": ["检索文档", "语义审查", "生成回答"],
    }

    task_memory["intent_context"] = intent_context
    state["intent_context"] = intent_context

    next_step = "retriever_node"

    append_trace(
        task_id,
        "planner_node",
        "intent_planner",
        {"user_query": user_query},
        {"intent_context": intent_context},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, "planner_node", next_step)

    return state
