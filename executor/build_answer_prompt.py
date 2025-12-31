def build_answer_prompt(user_query, hits):
    evidence = "\n\n".join(f"[{h['title']}]\n{h['chunk']}" for h in hits[:3])

    return f"""
You are an AskMyDocs assistant.

Answer the user question based ONLY on the evidence below.
If evidence is missing or unrelated, say so safely.

User Question:
{user_query}

Retrieved Evidence:
{evidence}
"""
