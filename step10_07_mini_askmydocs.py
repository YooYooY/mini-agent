from __future__ import annotations
from typing import TypedDict, Literal, Optional, List, Dict
import uuid
import json
import os

import dotenv
from langgraph.graph import StateGraph, END
from sympy import print_glsl
from vector_store_chroma import vector_index

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


def _domain_matches_task(user_query: str, hits: list[dict]) -> bool:
    """
    最小可用“语义判断”：
    - 如果 query 里面明显是「订单 / order / api」相关
    - 且 evidence chunk/title 同样是订单 API 相关
    → 认为域匹配
    - 如果 query 明显是退款 / UI / layout
    - 但 evidence 全是订单 API
    → 认为域不匹配
    """

    query = user_query.lower()

    print("query=>", query)
    print("hits=>", hits)

    order_keywords = ["订单", "order", "/api/orders"]
    refund_ui_keywords = ["退款", "refund", "layout"]

    # 判断 query 是不是「订单 API 领域」
    is_order_query = any(k in query for k in order_keywords)
    is_refund_ui_query = any(k in query for k in refund_ui_keywords)
    print("is_order_query=>", is_order_query)
    print("is_refund_ui_query=>", is_refund_ui_query)

    text_blob = " ".join(h["title"] + " " + h["chunk"] for h in hits).lower()
    print("text_blob=>", text_blob)
    has_order_evidence = any(k in text_blob for k in order_keywords)
    has_refund_ui_evidence = any(k in text_blob for k in refund_ui_keywords)
    print("has_order_evidence=>", has_order_evidence)
    print("has_refund_ui_evidence=>", has_refund_ui_evidence)

    # 规则 1：订单查询 + 订单 evidence → 匹配
    if is_order_query and has_order_evidence and not has_refund_ui_evidence:
        return True

    # 规则 2：明显退款/UI 请求，但 evidence 明显还是订单 API → 不匹配
    if is_refund_ui_query and has_order_evidence and not has_refund_ui_evidence:
        return False

    # 其他情况先当作不匹配（保守一点）
    return False


def critic_node(state: TaskState) -> TaskState:
    task_id = state["task_id"]
    task_memory = memory_store[task_id]

    hits = task_memory["retrieval_context"].get("retriever_hits", [])
    critic_prev = task_memory["critic_result"]
    critic_count = critic_prev["critic_count"]
    user_query = state["user_query"]

    problems: list[str] = []

    if len(hits) == 0:
        problems.append("retriever returned no documents")

    if not state.get("answer"):
        problems.append("no answer was generated")

    # critic 上限防死循环
    if critic_count >= 2:
        status: Literal["fail"] = "fail"
        reason = "critic count exceeded"
        action = "stop"
    else:
        # 先做域匹配检查
        match = _domain_matches_task(user_query, hits)
        print("match===>", match)
        if hits and match:
            status = "pass"
            reason = "retrieval semantically matched task"
            action = None
        else:
            # 没 hits 或域不匹配 → 建议重新检索
            status = "revise"
            if not hits:
                reason = "no evidence retrieved; need to redo retriever"
            else:
                reason = "evidence does not match task domain"
            action = "redo_retriever"

    # 更新 critic_count
    if status == "pass":
        new_critic_count = 0
    else:
        new_critic_count = critic_count + 1

    critic_result: CriticResult = {
        "status": status,
        "reason": reason,
        "critic_count": new_critic_count,
        "action": action,
    }

    task_memory["critic_result"] = critic_result
    state["critic_result"] = critic_result

    # 根据 critic_result 决定下一跳（同时写入 trace / checkpoint）
    if status == "pass":
        next_step = "end"
    elif status == "revise":
        next_step = "retriever_node"
    else:
        next_step = "fail_answer_node"

    append_trace(
        task_id=task_id,
        step="critic_node",
        tool="rule_domain_critic",
        input_data={"hit_count": len(hits), "user_query": user_query},
        output={"critic_result": critic_result},
        status="success",
        next_step=next_step,
    )

    save_checkpoint(task_id, state, last_step="critic_node", next_step=next_step)

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
    test_high_relevance(app)

    # 测试 2：故意无关（退款 UI layout）
    test_domain_mismatch(app)
