from core.memory import memory_store
from core.checkpoint import load_checkpoint


def resume_from_checkpoint(app, task_id: str):
    ckpt = load_checkpoint(task_id)
    if not ckpt:
        print("❌ no checkpoint")
        return None

    print("🔁 resume checkpoint:", task_id)

    memory_store[task_id] = ckpt["memory"]
    state = ckpt["state"]
    state["resume_next_step"] = ckpt["next_step"]

    return app.invoke(state)
