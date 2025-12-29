from typing import TypedDict, Optional, List, Dict, Literal


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
    status: str
    error: Optional[str]
    critic_round: int
    next_step: Optional[str]


class CriticResult(TypedDict, total=False):
    status: Literal["pass", "revise", "fail"]
    reason: str
    critic_count: int
    action: Optional[str]


class TaskState(TypedDict, total=False):
    task_id: str
    user_query: str
    intent_context: IntentContext
    retrieval_context: RetrievalContext
    answer: str
    execution_trace: List[ExecutionTrace]
    critic_result: CriticResult
    resume_next_step: Optional[str]
