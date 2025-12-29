from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from core.types import TaskState, RetrievalContext
from .vector_store_chroma import vector_index


def retriever_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    query = state["user_query"]
    hits = vector_index.search(query, k=3)

    retrieval_context: RetrievalContext = {
        "query": query,
        "doc_scope": ["orders", "api"],
        "retriever_hits": hits,
    }

    task_memory["retrieval_context"] = retrieval_context
    state["retrieval_context"] = retrieval_context

    evidence_preview = [
        {"doc_id": h["doc_id"], "title": h["title"], "score": h["score"]} for h in hits
    ]

    next_step = "executor_node"

    append_trace(
        task_id,
        "retriever_node",
        "chroma_vector_retriever",
        {"query": query},
        {"hit_count": len(hits), "evidence": evidence_preview},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, "retriever_node", next_step)

    return state
