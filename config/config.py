"""
Centralized configuration for the F1 RAG Bot.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============ Paths ============
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CHROMADB_DIR = DATA_DIR / "chromadb"
CACHE_DIR = DATA_DIR / "cache"
UPLOADS_DIR = DATA_DIR / "uploads"

# ============ Embedding Model ============
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
TOKENIZER = os.getenv("TOKENIZER", "sentence-transformers/all-MiniLM-L6-v2")
# ============ LLM Settings ============

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")  
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3-flash-preview")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ============ Retrieval Settings ============
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
DEFAULT_REGULATION_YEAR = int(os.getenv("DEFAULT_REGULATION_YEAR", "2026"))

# ============ Chunking Settings ============
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "480"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ============ Server Settings ============
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# ============ API Keys ============
HF_TOKEN = os.getenv("HF_TOKEN")
HF_BASE_URL=os.getenv("HF_BASE_URL")
GEMINI_KEY = os.getenv("GEMINI_KEY")
