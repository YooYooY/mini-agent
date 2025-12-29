import os, json
from .memory import memory_store

CHECKPOINT_DIR = "./checkpoints"


def ensure_checkpoint_dir():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def checkpoint_path(task_id: str):
    return os.path.join(CHECKPOINT_DIR, f"{task_id}.json")


def save_checkpoint(task_id, state, last_step, next_step):
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


def has_checkpoint(task_id: str):
    return os.path.exists(checkpoint_path(task_id))
