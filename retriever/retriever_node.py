from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState
from retriever.vector_store_chroma import vector_index

DEFAULT_TOP_K = 3

def retriever_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    rc = state["retrieval_context"]
    round = rc["round"]
    query = rc["query"]

    # --- 2) 根据 critic 结果动态调整 top_k ---
    top_k = DEFAULT_TOP_K + round
    hits = _rerank_hits(query,  vector_index.search(query, k=top_k))

    task_memory["retrieval_context"]["retriever_hits"] = hits
    state["retrieval_context"] = rc

    next_step="executor_node"

    append_trace(
        task_id,
        "retriever_node",
        "vector_retriever",
        {
            "round": round,
            "query": query,
            "query_source": rc["query_source"],
        },
        {"hit_count": len(hits)},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, "retriever_node", next_step)

    return state


def _rerank_hits(user_query, hits):
    """
    MVP 级 rerank：
    - 先按 score
    - 再按是否出现 query keywords
    """

    def score(h):
        bonus = 0
        if any(k in h["chunk"] for k in user_query.split()):
            bonus += 0.1
        return h["score"] + bonus

    return sorted(hits, key=score, reverse=True)
