from typing import List, Dict
from langchain_openai import ChatOpenAI


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


def build_answer_prompt(
    user_query: str,
    intent: str,
    compressed_evidence: str,
    hits: List[Dict],
) -> str:

    evidence_refs = []
    seen = set()

    for h in hits[:5]:
        key = (h.get("doc_id"), h.get("title"))
        if key in seen:
            continue
        seen.add(key)
        evidence_refs.append(f"- 《{h.get('title')}》 (id: {h.get('doc_id')})")

    refs_block = "\n".join(evidence_refs) if evidence_refs else "（无文档引用）"

    return f"""
你是一个 AskMyDocs 文档问答助手。

请仅基于提供的证据信息回答问题，
如果证据不足，请给出温和、可靠、无幻觉的回答。

【用户问题】
{user_query}

【任务类型】
{intent}

【压缩后的证据信息】
{compressed_evidence}

【检索命中文档参考】
{refs_block}

请输出最终回答内容。
"""


def generate_answer_stream(llm, prompt: str):
    """
    负责：
    - 调用 llm.stream(prompt)
    - 按 token 输出
    - 不负责拼接最终答案
    """
    for delta in llm.stream(prompt):
        token = delta.content or ""
        if not token:
            continue
        yield token
