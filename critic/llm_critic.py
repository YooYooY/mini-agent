import dotenv
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

dotenv.load_dotenv()

def run_llm_critic(user_query: str, intent: str, hits: list[dict], draft_answer: str):
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
