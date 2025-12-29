from .memory import memory_store
from .types import ExecutionTrace


def append_trace(
    task_id: str,
    step: str,
    tool: str,
    input_data,
    output=None,
    status="success",
    error=None,
    next_step=None,
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
