from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.memory import memory_store
from core.types import TaskState


def fail_answer_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    critic = memory_store[task_id]["critic_result"]

    answer = f"⚠️ 查询终止\n原因：{critic.get('reason','unknown')}"

    state["answer"] = answer

    append_trace(
        task_id,
        "fail_answer_node",
        "system_fallback",
        {"critic": critic},
        {"answer": answer},
        status="warning",
        next_step="end",
    )

    save_checkpoint(task_id, state, "fail_answer_node", "end")

    return state
