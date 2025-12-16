from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List

from src.agent import run_agent

app = FastAPI()


# -----------------------------
# Request / Response schemas
# -----------------------------
class AgentRequest(BaseModel):
    query: str
    image_url: Optional[str] = None


class AgentResponse(BaseModel):
    answer: str
    used_tools: List[str]
    status: str = "ok"


# -----------------------------
# API endpoint
# -----------------------------
@app.post("/agent/ask", response_model=AgentResponse)
def ask_agent(req: AgentRequest):
    answer, used_tools = run_agent(
        query=req.query,
        image_url=req.image_url,
    )

    return AgentResponse(
        answer=answer,
        used_tools=used_tools,
    )
