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

# ============ Embedding Model ============
# Options: 
#   - "sentence-transformers/all-MiniLM-L6-v2" (fast, local)
#   - "sentence-transformers/all-mpnet-base-v2" (better quality, local)
#   - "nvidia/nv-embedqa-e5-v5" (NVIDIA API)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ============ LLM Settings ============
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "huggingface")  # "huggingface", "openai", "nvidia"
LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))

# ============ Retrieval Settings ============
RETRIEVER_K = int(os.getenv("RETRIEVER_K", "5"))
DEFAULT_REGULATION_YEAR = int(os.getenv("DEFAULT_REGULATION_YEAR", "2026"))

# ============ Chunking Settings ============
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))

# ============ Server Settings ============
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "8000"))

# ============ API Keys ============
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
