# 🧠 Mini-Agent

### A Hands-on ReACT + RAG + Multimodal Agent for Learning

---

## 📌 Overview

**Mini-Agent** is a **hands-on learning project** that explores how modern LLM agents can be built using:

* **ReACT (Reason + Act)** as a control-flow pattern
* **Tool Calling** for external actions
* **RAG (Retrieval-Augmented Generation)** for knowledge retrieval
* **Multimodal input (text + image)**
* **FastAPI** as a simple backend interface

This project is **not production-ready**.
Its purpose is to provide a **clear, runnable reference** for understanding how these pieces work together in practice.

---

## ✨ What This Project Focuses On

* Understanding **ReACT as a control-flow design**, not a prompt format
* Seeing how **LLMs decide when to use tools**
* Treating **RAG as an optional information source**, not a magic database
* Integrating **multimodal inputs** into an agent loop
* Exposing an agent through a **simple HTTP API** for experimentation

---

## 🧠 Core Ideas

* **ReACT is about control flow**
  Reason → Act → Observe → Decide again

* **Tool calling is structural, not textual**
  Tools are invoked via structured calls, not parsed from strings

* **RAG reduces uncertainty**
  The model does not know what is in the knowledge base—it decides when to retrieve

* **Multimodal input does not change the agent loop**
  Images are just another form of observation

---

## 🏗️ High-level Architecture

```
Client (for testing)
    ↓
FastAPI endpoint
    ↓
ReACT Agent Loop
    ↓
LLM decides:
  - answer directly
  - or call a tool (RAG, etc.)
    ↓
Observation
    ↓
Final answer
```

---

## 📂 Project Structure

```text
mini-agent/
├── src/
│   ├── __init__.py
│   ├── app.py        # FastAPI entry point
│   ├── agent.py      # ReACT agent logic
│   ├── rag.py        # RAG setup (vector store)
│   ├── tools.py      # Tool definitions
├── data/
│   └── langchain_intro.txt
├── .env              # Environment variables
└── requirements.txt
```

---

## ⚙️ Setup

### 1️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx
```

---

## ▶️ Run the Server

From the project root:

```bash
uvicorn src.app:app --reload
```

Then visit:

```
http://localhost:8000/docs
```

---

## 🔌 API Example

### Endpoint

```http
POST /agent/ask
```

### Request

```json
{
  "query": "What is LangChain?",
  "image_url": "https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg"
}
```

### Response

```json
{
  "answer": "LangChain is a framework for building applications powered by large language models.",
  "used_tools": ["rag_search"],
  "status": "ok"
}
```

---

## 🧩 Notes on Agent Behavior

* The LLM decides whether to call tools based on:

  * The system prompt
  * Tool descriptions
  * The semantics of the question

* During tool calls, the model may return an empty `content` field.
  This is expected behavior when using structured tool calling.

---

## 🔍 Why This Project Exists

Many tutorials focus on **prompt patterns** or **isolated features**.

This project was built to help answer questions like:

* How does an agent actually decide to use a tool?
* How do ReACT, RAG, and multimodal inputs fit into one loop?
* What does a minimal but realistic agent backend look like?

If you are learning AI application development or agent systems, this repository aims to be a **clear and honest reference**.

