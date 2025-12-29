from __future__ import annotations
from typing import TypedDict, Literal, Optional, List, Dict
import uuid
import json
import os

import dotenv
from langgraph.graph import StateGraph, END
from vector_store_chroma import vector_index
from langchain_openai import ChatOpenAI


llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

dotenv.load_dotenv()

# ========= 类型定义 =========


class IntentContext(TypedDict, total=False):
    topic: str
    intent: str
    task_plan: List[str]


class RetrievalContext(TypedDict, total=False):
    query: str
    doc_scope: List[str]
    retriever_hits: List[Dict]


class ExecutionTrace(TypedDict, total=False):
    step: str
    tool: str
    input: Dict
    output: Optional[Dict]
    status: str  # success / warning / error
    error: Optional[str]
    critic_round: int
    next_step: Optional[str]


class CriticResult(TypedDict, total=False):
    status: Literal["pass", "revise", "fail"]
    reason: str
    critic_count: int
    action: Optional[str]  # e.g. "redo_retriever", "stop"


class TaskState(TypedDict, total=False):
    task_id: str
    user_query: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    execution_trace: List[ExecutionTrace]
    critic_result: CriticResult
    resume_next_step: Optional[str]


# ========= 全局 Memory =========

memory_store: Dict[str, Dict] = {}

# ========= Checkpoint =========

CHECKPOINT_DIR = "./checkpoints"


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path(task_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{task_id}.json")


def save_checkpoint(task_id: str, state: TaskState, last_step: str, next_step: str):
    """保存当前任务 checkpoint，包括 next_step"""
    ensure_checkpoint_dir()
    payload = {
        "task_id": task_id,
        "last_step": last_step,
        "next_step": next_step,
        "state": state,
        "memory": memory_store[task_id],
    }
    with open(checkpoint_path(task_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_checkpoint(task_id: str):
    path = checkpoint_path(task_id)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def has_checkpoint(task_id: str) -> bool:
    return os.path.exists(checkpoint_path(task_id))


# ========= Trace 工具 =========


def append_trace(
    task_id: str,
    step: str,
    tool: str,
    input_data: Dict,
    output: Optional[Dict] = None,
    status: str = "success",
    error: Optional[str] = None,
    next_step: Optional[str] = None,
):
    critic_round = memory_store[task_id]["critic_result"]["critic_count"]
    trace_item: ExecutionTrace = {
        "step": step,
        "tool": tool,
        "input": input_data,
        "output": output,
        "status": status,
        "error": error,
        "critic_round": critic_round,
        "next_step": next_step,
    }
    memory_store[task_id]["execution_trace"].append(trace_item)


# ========= 任务初始化 =========


def init_task_memory(task_id: Optional[str] = None) -> str:
    if task_id is None:
        task_id = str(uuid.uuid4())
    memory_store[task_id] = {
        "task_meta": {"task_id": task_id},
        "intent_context": {},
        "retrieval_context": {},
        "execution_trace": [],
        "critic_result": {
            "critic_count": 0,
            "status": "pass",
            "reason": "",
            "action": None,
        },
    }
    return task_id


def create_init_state(task_id: str, user_query: str) -> TaskState:
    return TaskState(
        task_id=task_id,
        user_query=user_query,
        intent_context={},
        retrieval_context={},
        answer="",
        execution_trace=[],
        critic_result={
            "critic_count": 0,
            "status": "pass",
            "reason": "",
            "action": None,
        },
    )


# ========= Node 实现 =========


def entry_node(state: TaskState) -> TaskState:
    # 入口不做事，仅负责交给 route_from_entry 决定起点
    return state


def route_from_entry(state: TaskState) -> str:
    resume_next = state.get("resume_next_step")
    if resume_next:
        return resume_next
    return "planner_node"


def planner_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    # 简单地用 user_query 推出 topic / intent
    user_query = state["user_query"]

    if "订单" in user_query or "order" in user_query.lower():
        topic = "订单 API"
        intent = "查询订单相关接口信息并生成示例代码"
    else:
        topic = "未知主题"
        intent = f"查询与「{user_query}」相关的信息"

    task_plan = ["检索文档", "验证领域是否匹配", "生成回答"]

    intent_context: IntentContext = {
        "topic": topic,
        "intent": intent,
        "task_plan": task_plan,
    }

    task_memory["intent_context"] = intent_context
    state["intent_context"] = intent_context

    next_step = "retriever_node"

    append_trace(
        task_id=task_id,
        step="planner_node",
        tool="intent_planner",
        input_data={"user_query": user_query},
        output={"intent_context": intent_context},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="planner_node", next_step=next_step)

    return state


def retriever_node(state: TaskState) -> TaskState:
    """
    使用 Chroma 的真实向量检索：
    - retriever 不再 mock
    - 返回真实 chunk
    - evidence 写入 trace
    - checkpoint 记录状态
    """
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

    # evidence 进入 trace，只写精简版
    evidence_preview = [
        {
            "doc_id": h["doc_id"],
            "title": h["title"],
            "score": h["score"],
        }
        for h in hits
    ]

    next_step = "executor_node"

    append_trace(
        task_id=task_id,
        step="retriever_node",
        tool="chroma_vector_retriever",
        input_data={"query": query},
        output={"hit_count": len(hits), "evidence": evidence_preview},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="retriever_node", next_step=next_step)

    return state


def executor_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    intent = task_memory["intent_context"].get("intent", "")
    hits = task_memory["retrieval_context"].get("retriever_hits", [])

    # 用 evidence 文本生成一个“假回答”（真实工程里换成 LLM 调用）
    doc_titles = [h["title"] for h in hits]
    answer = f"根据意图「{intent}」，并参考文档：{', '.join(doc_titles)}，生成的示例回答（这里省略 LLM 调用）。"

    state["answer"] = answer

    next_step = "critic_node"

    append_trace(
        task_id=task_id,
        step="executor_node",
        tool="answer_generator",
        input_data={"intent": intent, "hit_count": len(hits)},
        output={"answer_preview": answer[:80]},
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="executor_node", next_step=next_step)

    return state


# ======== 域匹配规则（模拟“语义”检测）========


def _run_llm_critic(user_query: str, intent: str, hits: list[dict], draft_answer: str):
    """
    返回一个结构化裁决结果（LLM 助手角色 only）
    """

    evidence_text = "\n\n".join(
        f"[{h['title']}]\n{h['chunk']}\n(score={h['score']})"
        for h in hits[:3]  # 控制长度
    )

    prompt = f"""
You are a retrieval quality auditor for an AskMyDocs RAG system.

Task:
Evaluate whether the retrieved evidence is semantically relevant
to the user query and the planned task intent.

User Query:
{user_query}

Task Intent:
{intent}

Draft Answer (generated from evidence):
{draft_answer}

Retrieved Evidence Chunks:
{evidence_text}

You must reason carefully and output a structured JSON with fields:

- status:
  - "pass"   → evidence matches query & task domain
  - "revise" → evidence seems partially relevant or mismatched, redo retriever
  - "fail"   → critically wrong retrieval or cannot recover

- reason:
  short natural language justification

- action:
  - "redo_retriever"
  - "stop"
  - null if status="pass"
"""

    resp = llm.invoke(prompt)

    # 预期 LLM 输出 json 或接近 json
    import json

    try:
        result = json.loads(resp.content)
    except Exception:
        result = {
            "status": "revise",
            "reason": "LLM critic returned invalid response, fallback to revise",
            "action": "redo_retriever",
        }

    return result


def _retrieval_sanity_check(
    user_query: str, hits: list[dict]
) -> Literal[True, False, None]:
    """
    返回:
        True  -> 明确健康（允许 LLM critic 决定 pass/revise）
        False -> 明确不健康（至少 revise）
        None  -> 不确定（完全交给 LLM critic）

    ⚠️ 这里不做“业务语义判断”
    只负责：系统稳定性 / 反直觉 / 结构异常
    """

    # 1) 完全未召回 → 至少 revise
    if len(hits) == 0:
        return False

    # 2) evidence 均为重复 chunk → retriever 明显异常
    chunks = [h["chunk"] for h in hits]

    print(
        "len(set(chunks)) == 1 and len(chunks) > 1",
        len(set(chunks)) == 1 and len(chunks) > 1,
    )

    if len(set(chunks)) == 1 and len(chunks) > 1:
        return False

    # 3) evidence 过短（可能是标题 / 噪声）
    if all(len(c) < 30 for c in chunks):
        return False

    # ⭐️ 正常情况：交给 LLM 决定
    return None


def _merge_sanity_and_llm(rule_state: bool | None, llm_result: dict):
    """
    rule_state:
        True  -> evidence 健康（LLM 有最终决定权）
        False -> evidence 明显异常（LLM 不能直接 PASS）
        None  -> 交给 LLM critic

    重点：
    - rule 兜底 = 只阻止 LLM“过度自信 PASS”
    - 不强推 PASS / 不做业务推断
    """

    # evidence 明显异常 → 至少 revise
    if rule_state is False:
        if llm_result.get("status") == "pass":
            return {
                "status": "revise",
                "reason": "sanity check failed (retriever unhealthy), avoid false-pass",
                "action": "redo_retriever",
            }
        return llm_result

    # 正常 / 不确定 → 完全尊重 LLM 语义 critic
    return llm_result


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    hits = task_memory["retrieval_context"].get("retriever_hits", [])
    critic_prev = task_memory["critic_result"]
    critic_count = critic_prev["critic_count"]
    user_query = state["user_query"]

    intent = task_memory["intent_context"].get("intent", "")
    draft_answer = state.get("answer", "")

    # critic 上限防死循环
    if critic_count >= 2:
        critic_result = {
            "status": "fail",
            "reason": "critic count exceeded",
            "action": "stop",
        }
    else:
        rule_state = _retrieval_sanity_check(user_query, hits)
        llm_result = _run_llm_critic(
            user_query=user_query,
            intent=intent,
            hits=hits,
            draft_answer=draft_answer,
        )

        critic_result = _merge_sanity_and_llm(rule_state, llm_result)

    # --- critic_count 更新策略 ---
    if critic_result["status"] == "pass":
        new_count = 0
    else:
        new_count = critic_count + 1

    critic_result = {
        **critic_result,
        "critic_count": new_count,
    }

    task_memory["critic_result"] = critic_result
    state["critic_result"] = critic_result

    # --- 路由决策 ---
    routing = {
        "pass": "end",
        "revise": "retriever_node",
        "fail": "fail_answer_node",
    }[critic_result["status"]]

    # --- trace 写入 evidence-aware 语义日志 ---
    append_trace(
        task_id=task_id,
        step="critic_node",
        tool="llm_semantic_critic",
        input_data={
            "query": user_query,
            "intent": intent,
            "hit_count": len(hits),
        },
        output={"critic_result": critic_result},
        next_step=routing,
    )

    save_checkpoint(task_id, state, "critic_node", routing)

    return state


def fail_answer_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    critic = task_memory["critic_result"]
    reason = critic.get("reason", "unknown error")

    answer = "⚠️ 当前查询未能成功处理（已终止）。\n" f"原因：{reason}"

    state["answer"] = answer

    next_step = "end"

    append_trace(
        task_id=task_id,
        step="fail_answer_node",
        tool="system_fallback",
        input_data={"critic": critic},
        output={"answer": answer},
        status="warning",
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="fail_answer_node", next_step=next_step)

    return state


# ========= Trace-Driven Resume =========


def resume_from_checkpoint(app, task_id: str) -> Optional[TaskState]:
    ckpt = load_checkpoint(task_id)
    if not ckpt:
        print(f"❌ no checkpoint for task {task_id}")
        return None

    print(f"🔁 Resuming from checkpoint for task: {task_id}")
    print(f"   last_step: {ckpt['last_step']}")
    print(f"   next_step: {ckpt['next_step']}")

    memory_store[task_id] = ckpt["memory"]
    state: TaskState = ckpt["state"]
    state["resume_next_step"] = ckpt["next_step"]

    result: TaskState = app.invoke(state)
    return result


# ========= 构建 Graph =========


def build_graph():
    graph = StateGraph(TaskState)

    graph.add_node("entry_node", entry_node)
    graph.add_node("planner_node", planner_node)
    graph.add_node("retriever_node", retriever_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("critic_node", critic_node)
    graph.add_node("fail_answer_node", fail_answer_node)

    graph.set_entry_point("entry_node")

    graph.add_conditional_edges(
        "entry_node",
        route_from_entry,
        {
            "planner_node": "planner_node",
            "retriever_node": "retriever_node",
            "executor_node": "executor_node",
            "critic_node": "critic_node",
            "fail_answer_node": "fail_answer_node",
            "end": END,
        },
    )

    graph.add_edge("planner_node", "retriever_node")
    graph.add_edge("retriever_node", "executor_node")
    graph.add_edge("executor_node", "critic_node")

    graph.add_conditional_edges(
        "critic_node",
        lambda s: {
            "pass": "end",
            "revise": "retriever_node",
            "fail": "fail_answer_node",
        }[s["critic_result"]["status"]],
        {
            "retriever_node": "retriever_node",
            "fail_answer_node": "fail_answer_node",
            "end": END,
        },
    )

    graph.add_edge("fail_answer_node", END)

    return graph.compile()


# ========= 两个测试用例 =========


def print_trace(task_id: str, label: str):
    print(f"\n=== {label} · Execution Trace ===")
    for step in memory_store[task_id]["execution_trace"]:
        print(
            f"- step={step['step']} | tool={step['tool']} "
            f"| next={step.get('next_step')} | status={step['status']}"
        )


def test_high_relevance(app):
    print("\n================= 测试 1：高相关文档（订单查询 API） =================")
    task_id = init_task_memory()
    state = create_init_state(task_id, user_query="请帮我查询订单查询 API 的接口说明")
    result = app.invoke(state)

    critic = result["critic_result"]
    print("\n[测试 1] Critic 结果：")
    print(critic)
    print("\n[测试 1] 最终答案：")
    print(result["answer"])
    # print_trace(task_id, "测试 1")

    # 预期：
    # status = pass
    # reason = retrieval semantically matched task
    # action = None


def test_domain_mismatch(app):
    print(
        "\n================= 测试 2：故意无关 query（退款 UI layout） ================="
    )
    task_id = init_task_memory()
    state = create_init_state(task_id, user_query="设计一下退款流程的 UI layout")
    result = app.invoke(state)

    critic = result["critic_result"]
    print("\n[测试 2] Critic 结果：")
    print(critic)
    print("\n[测试 2] 最终答案：")
    print(result["answer"])
    # print_trace(task_id, "测试 2")

    # 预期：
    # status = revise
    # reason = evidence does not match task domain
    # action = redo_retriever
    # 并且 retriever 会在下一轮被重新调用（有 critic_count 上限保护）


if __name__ == "__main__":
    # 先往 Chroma 里塞一点订单 API 文档，用作向量检索的真实底库
    sample_docs = [
        {
            "id": "orders-api-001",
            "title": "订单查询接口",
            "text": """
GET /api/orders/{order_id}

参数：
- order_id: 订单ID

功能：
根据订单ID返回订单详情，包括状态、价格、物流信息。
""",
        },
        {
            "id": "orders-api-002",
            "title": "订单列表查询接口",
            "text": """
GET /api/orders?user_id={uid}

参数：
- user_id: 用户ID

功能：
返回用户最近 50 条订单，支持状态筛选、时间范围筛选。
""",
        },
    ]

    vector_index.add_documents(sample_docs)

    app = build_graph()

    # 测试 1：高相关（订单查询 API）
    # test_high_relevance(app)

    # 测试 2：故意无关（退款 UI layout）
    test_domain_mismatch(app)
