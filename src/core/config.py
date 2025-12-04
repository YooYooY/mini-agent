from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# 路径相关（项目根目录）
# ---------------------------
# __file__ = src/core/config.py
# parents[2] = 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PERSIST_DIR = PROJECT_ROOT / "chroma_db"

# 确保目录存在
DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(exist_ok=True)
PERSIST_DIR.mkdir(exist_ok=True)

# ---------------------------
# 环境变量（.env）
# ---------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------------
# RAG 配置
# ---------------------------
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K = 3
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ---------------------------
# OCR 配置
# ---------------------------
TESSERACT_CMD = "/usr/local/bin/tesseract"
