# executor/executor_node.py
from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState
from .answer_engine import compress_evidence, generate_answer


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    user_query = state["user_query"]
    intent = task_memory["intent_context"].get("intent", "")
    hits = task_memory["retrieval_context"].get("retriever_hits", [])

    # 1) 先用 LLM 压缩 / 过滤 evidence
    compressed_evidence = compress_evidence(
        user_query=user_query,
        intent=intent,
        hits=hits,
    )

    # print("compressed_evidence=>", compressed_evidence)

    # 2) 基于压缩后的 evidence 生成最终回答
    answer = generate_answer(
        user_query=user_query,
        intent=intent,
        compressed_evidence=compressed_evidence,
        hits=hits,
    )

    # print("answer=>", answer)

    state["answer"] = answer

    next_step = "critic_node"

    # 3) 写入 trace（记录 evidence 使用情况）
    append_trace(
        task_id=task_id,
        step="executor_node",
        tool="rag_answer_executor",
        input_data={
            "intent": intent,
            "hit_count": len(hits),
        },
        output={
            "answer_preview": answer[:120],
            "compressed_evidence_preview": compressed_evidence[:120],
        },
        next_step=next_step,
    )

    # 4) checkpoint，便于从 executor 之后继续恢复
    save_checkpoint(task_id, state, last_step="executor_node", next_step=next_step)

    return state
