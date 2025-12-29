from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent = task_memory["intent_context"].get("intent", "")
    hits = task_memory["retrieval_context"].get("retriever_hits", [])

    doc_titles = [h["title"] for h in hits]
    answer = f"根据意图「{intent}」并参考文档：{', '.join(doc_titles)}，生成的示例回答（模拟 LLM）。"

    state["answer"] = answer
    next_step = "critic_node"

    append_trace(
        task_id,
        "executor_node",
        "answer_generator",
        {"intent": intent, "hit_count": len(hits)},
        {"answer_preview": answer[:80]},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, "executor_node", next_step)

    return state
