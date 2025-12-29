from core.graph_builder import build_graph
from core.memory import init_task_memory, create_init_state
# from resume.resume_runner import resume_from_checkpoint

if __name__ == "__main__":

    app = build_graph()

    user_query = "请帮我查询订单查询 API 的接口说明"

    task_id = init_task_memory()
    state = create_init_state(task_id, user_query)

    result = app.invoke(state)

    print("\n=== 最终答案 ===")
    print(result["answer"])
