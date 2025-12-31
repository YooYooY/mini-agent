# executor/executor_node.py
from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState
from .answer_engine import (
    compress_evidence,
    build_answer_prompt,
    generate_answer_stream,
)


def executor_node(state: TaskState, *, config):
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    user_query = state["user_query"]
    intent = task_memory["intent_context"].get("intent", "")
    hits = task_memory["retrieval_context"].get("retriever_hits", [])

    llm = config["configurable"]["llm"]

    # 1) 先用 LLM 压缩 / 过滤 evidence
    compressed_evidence = compress_evidence(
        user_query=user_query,
        intent=intent,
        hits=hits,
    )

    yield {
        "type": "evidence_ready",
        "task_id": task_id,
        "compressed_preview": compressed_evidence[:120],
    }

    # —— 2) 进入 Streaming Answer 阶段 ——
    yield {
        "type": "stage_start",
        "stage": "executor_answer",
        "task_id": task_id,
    }

    prompt = build_answer_prompt(
        user_query=user_query,
        intent=intent,
        compressed_evidence=compressed_evidence,
        hits=hits,
    )
    print("\n[prompt] BUILT len =", len(prompt))
    chunks = []
    token_i = 0

    for token in generate_answer_stream(llm, prompt):
        chunks.append(token)
        token_i += 1
        print(f"[stream]  token[{token_i}] =>", repr(token))
        yield {
            "type": "answer_chunk",
            "task_id": task_id,
            "token": token,
        }

    print("\n[stream] DONE, total tokens =", token_i)
    # —— 3) 拼接最终 Answer（用于 critic / trace / checkpoint） ——
    answer = "".join(chunks)
    state["answer"] = answer

    yield {
        "type": "answer_complete",
        "task_id": task_id,
        "answer": answer,
    }

    next_step = "critic_node"

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
