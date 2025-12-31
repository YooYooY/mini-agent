from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState
from planner.intent_detector import detect_intent


def planner_node(state: TaskState, *, config) -> TaskState:

    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    user_query = state["user_query"]

    intent_context = detect_intent(user_query)

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
