from langchain_core.tools import tool
from src.rag import retriever


@tool
def rag_search(query: str) -> str:
    """
    Search background knowledge using RAG.

    Use this tool when the question involves:
    - Technical frameworks
    - Libraries
    - Architecture explanations
    - Concepts you are not 100% certain about
    """
    docs = retriever.invoke(query)
    return "\n".join(d.page_content for d in docs)


@tool
def classify_image(label: str) -> str:
    """Classify an image (placeholder example)."""
    return f"The image is classified as: {label}"
