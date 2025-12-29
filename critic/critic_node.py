from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from critic.sanity_rules import retrieval_sanity_check
from critic.llm_critic import run_llm_critic
from core.types import TaskState


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    hits = task_memory["retrieval_context"].get("retriever_hits", [])
    critic_prev = task_memory["critic_result"]
    critic_count = critic_prev["critic_count"]

    user_query = state["user_query"]
    intent = task_memory["intent_context"].get("intent", "")
    draft_answer = state.get("answer", "")

    if critic_count >= 2:
        critic_result = {
            "status": "fail",
            "reason": "critic count exceeded",
            "action": "stop",
        }
    else:
        sanity = retrieval_sanity_check(hits)
        llm_result = run_llm_critic(user_query, intent, hits, draft_answer)

        # rule 仅阻止 false-pass
        if sanity is False and llm_result["status"] == "pass":
            critic_result = {
                "status": "revise",
                "reason": "retriever unhealthy – avoid false pass",
                "action": "redo_retriever",
            }
        else:
            critic_result = llm_result

    critic_result["critic_count"] = (
        0 if critic_result["status"] == "pass" else critic_count + 1
    )

    task_memory["critic_result"] = critic_result
    state["critic_result"] = critic_result

    routing = {
        "pass": "end",
        "revise": "retriever_node",
        "fail": "fail_answer_node",
    }[critic_result["status"]]

    append_trace(
        task_id,
        "critic_node",
        "llm_semantic_critic",
        {"query": user_query, "intent": intent, "hit_count": len(hits)},
        {"critic_result": critic_result},
        next_step=routing,
    )

    save_checkpoint(task_id, state, "critic_node", routing)

    return state
