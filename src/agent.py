from typing import Dict, List, Tuple, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage

from src.tools import rag_search, classify_image

# -----------------------------
# Tools registry
# -----------------------------
TOOLS = {
    "rag_search": rag_search,
    "classify_image": classify_image,
}

# -----------------------------
# LLM (bind tools)
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
).bind_tools(list(TOOLS.values()))

# -----------------------------
# System prompt
# -----------------------------
SYSTEM_PROMPT = """
You are a multimodal ReACT agent.

Rules:
- You may receive text and images.
- Use tools when external knowledge or processing is required.
- When you have enough information, provide a final answer.
"""


def run_agent(
    query: str,
    image_url: Optional[str] = None,
) -> Tuple[str, List[str]]:
    """
    Run a ReACT agent with optional multimodal input.

    Returns:
        answer: final answer text
        used_tools: list of tool names used during execution
    """
    used_tools: List[str] = []

    # -----------------------------
    # Build initial message (multimodal)
    # -----------------------------
    content: List[dict] = [
        {"type": "text", "text": SYSTEM_PROMPT},
        {"type": "text", "text": query},
    ]

    if image_url:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    messages = [HumanMessage(content=content)]

    # -----------------------------
    # ReACT loop
    # -----------------------------
    while True:
        response: AIMessage = llm.invoke(messages)

        # ----- Tool calls -----
        if response.tool_calls:
            print("TOOL CALLS:", response.tool_calls)
            for tc in response.tool_calls:
                tool_name = tc["name"]
                args = tc["args"]

                tool = TOOLS.get(tool_name)
                if not tool:
                    raise RuntimeError(f"Unknown tool: {tool_name}")

                # record tool usage
                used_tools.append(tool_name)

                result = tool.invoke(args)

                messages.append(response)
                messages.append(
                    ToolMessage(
                        content=str(result),
                        tool_call_id=tc["id"],
                    )
                )
            continue

        # ----- Final answer -----
        if response.content:
            return response.content.strip(), used_tools

        raise RuntimeError("Agent returned neither tool_calls nor content.")
