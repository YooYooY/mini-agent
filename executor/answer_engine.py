from typing import List, Dict
from langchain_openai import ChatOpenAI


# 可以用便宜一点的模型，先保证可用性
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0.2,
)


def _format_evidence_for_llm(hits: List[Dict]) -> str:
    """
    把 retriever 返回的 hits 转成 LLM 能看懂的文本块：
    - 控制长度
    - 保留 doc_id / title / chunk
    """
    blocks = []
    for i, h in enumerate(hits[:5]):  # 最多带 5 个 chunk 进来
        blocks.append(
            f"""[DOC {i+1}] id={h.get('doc_id')}
title={h.get('title')}
content:
{h.get('chunk')}
"""
        )
    return "\n\n".join(blocks)


def compress_evidence(user_query: str, intent: str, hits: List[Dict]) -> str:
    """
    用 LLM 对 evidence 做“压缩+筛选”：
    - 只保留和当前 query / intent 强相关的信息
    - 限制在一个短文本里，减少后面 answer 的上下文长度
    """
    if not hits:
        return ""

    raw_evidence = _format_evidence_for_llm(hits)

    prompt = f"""
You are a retrieval assistant.

Goal:
Given a user query and several documentation chunks,
select and compress only the most relevant information
that is needed to answer the question.

User Query:
{user_query}

Task Intent:
{intent}

Raw Evidence:
{raw_evidence}

Instructions:
- Only keep information that is directly useful for answering the query.
- You may paraphrase or shorten long sentences.
- Do NOT invent new facts.
- Output a concise Chinese summary, within about 300–500 Chinese characters.
- If nothing is relevant, output "无有效证据".
"""

    resp = llm.invoke(prompt)
    compressed = resp.content.strip()
    return compressed


def generate_answer(
    user_query: str,
    intent: str,
    compressed_evidence: str,
    hits: List[Dict],
) -> str:
    """
    使用压缩过的 evidence 生成最终给用户的答案：
    - 主体回答
    - 附带“参考文档列表”（人类可读）
    """
    # 如果没有有效证据，给一个诚实的、可被 critic 识别的回答
    if not hits or not compressed_evidence or "无有效证据" in compressed_evidence:
        return (
            f"目前在已索引的文档中，没有找到能直接回答「{user_query}」的可靠内容。\n"
            "建议：\n"
            "1. 检查知识库是否已经收录相关接口或文档；\n"
            "2. 补充上传与该问题直接相关的接口说明 / 设计文档；\n"
            "3. 再次尝试提问。"
        )

    evidence_refs = []
    seen = set()
    for h in hits[:5]:
        key = (h.get("doc_id"), h.get("title"))
        if key in seen:
            continue
        seen.add(key)
        evidence_refs.append(f"- 《{h.get('title')}》 (id: {h.get('doc_id')})")

    refs_block = "\n".join(evidence_refs)

    prompt = f"""
你是一个 API 文档助手，负责基于「提供的证据」回答用户问题。

【用户问题】
{user_query}

【任务意图】
{intent}

【经过筛选和压缩后的证据】
{compressed_evidence}

要求：
- 必须严格基于上面的“证据”回答，不要凭空编造接口或字段。
- 回答语言使用简体中文。
- 结构清晰，适合给前端 / 后端工程师直接使用。
- 如果证据中缺少关键细节，请明确指出“文档中未给出 xxx 信息”。

请给出最终回答。
"""

    resp = llm.invoke(prompt)
    answer_main = resp.content.strip()

    full_answer = f"{answer_main}\n\n" "———\n" "参考文档：\n" f"{refs_block}"

    return full_answer
