# critic_node.py

from core.memory import memory_store
from core.trace import append_trace
from core.checkpoint import save_checkpoint
from critic.policy_mapper import map_critic_policy
from critic.llm_critic import run_llm_critic
from core.types import TaskState


def critic_node(state: TaskState, *, config) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    rc = task_memory["retrieval_context"]

    user_query = state["user_query"]
    intent = task_memory["intent_context"].get("intent", "")
    hits = rc.get("retriever_hits", [])
    draft_answer = state.get("answer", "")

    critic_prev = task_memory["critic_result"]
    critic_count = critic_prev.get("critic_count", 0)
    
    draft_answer = state.get("answer", "")

    llm = config["configurable"]["llm"]

    # --- Stage 1: LLM semantic critic ---
    semantic_result = run_llm_critic(
        user_query=user_query,
        intent=intent,
        hits=hits,
        draft_answer=draft_answer,
        llm=llm,
    )

    # --- Stage 2: Map to agent policy ---
    critic_result = map_critic_policy(
        semantic_result=semantic_result,
        critic_count=critic_count,
    )
    
    task_memory["critic_result"] = critic_result
    state["critic_result"] = critic_result

    routing = {
        "pass": "end",
        "revise_retry": "retriever_node",
        "revise_rewrite": "query_rewrite_node",
        "fail": "fail_answer_node",
    }[critic_result["status"]]

    append_trace(
        task_id,
        "critic_node",
        "llm_semantic_critic",
        {
            "round": rc["round"],
            "query": rc["query"],
            "hit_count": len(hits),
        },
        {"critic_result": critic_result},
        next_step=routing,
    )

    save_checkpoint(task_id, state, "critic_node", routing)

    return state
