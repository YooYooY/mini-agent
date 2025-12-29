import uuid
from .types import TaskState

memory_store: dict = {}


def init_task_memory(task_id: str | None = None) -> str:
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
