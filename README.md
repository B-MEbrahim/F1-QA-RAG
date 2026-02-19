# F1 RAG Bot 🏎️

A Retrieval-Augmented Generation (RAG) assistant for Formula 1 regulations. 

## Overview

This application allows users to ask questions about:
- **FIA F1 Regulations** (Technical, Sporting, Financial, Operational)
- **Custom PDF Rules** (upload your own documents for Q&A)

The system uses semantic search over chunked regulation PDFs and an LLM to generate grounded answers with source citations.

## Features

- 📄 **Document Ingestion**: PDF parsing → Markdown → Semantic chunking → Vector embeddings
- 🔍 **Semantic Search**: ChromaDB vector store with year-based collections
- 📤 **PDF Upload**: Users can upload custom F1 rules PDFs and ask questions about them
- 🛡️ **Guard-rails**: Input validation and output grounding checks
- 📊 **Evaluation**: Retrieval and answer quality metrics
- 🌐 **API Server**: FastAPI + LangServe endpoints
- 💬 **Chat UI**: Gradio interface with session management

## Project Structure

```
f1_rag_bot/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── chromadb/        # Vector store (ChromaDB)
│   ├── uploads/         # User-uploaded PDF documents
│   └── raw/             # FIA regulation PDFs
│       └── 2026/
├── src/
│   ├── config.py        # Centralized configuration
│   ├── chain.py         # Main RAG pipeline
│   ├── models.py        # Pydantic models
│   ├── app/
│   │   ├── server.py    # FastAPI/LangServe endpoints
│   │   └── ui.py        # Gradio UI with upload feature
│   ├── ingestion/
│   │   └── ingest.py    # PDF ingestion & chunking
│   ├── tools/
│   │   ├── retriever.py # Regulation search tool
│   │   ├── uploads.py   # Session-to-collection mapping
│   │   └── files/       # File utilities
│   ├── chain/
│   │   ├── chain.py     # RAG chain with collection routing
│   │   └── chat_history.py # Session history management
│   ├── guardrails/
│   │   └── checks.py    # Safety & factuality checks
│   └── evaluation/
│       └── evaluate.py  # Eval metrics pipeline
└── tests/
    └── test_chain.py    # Comprehensive test suite
```

## Setup

### 1. Create Environment

```bash
conda create -n f1-rag python=3.12 -y
conda activate f1-rag
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your NVIDIA API key (required)
# Get your key at: https://build.nvidia.com/
```

**Required:** API keys for LLM provider (Gemini recommended, or HuggingFace endpoint)

### 3. Ingest Regulations

Place FIA regulation PDFs in `data/raw/<year>/` then run:

```bash
python -m src.ingestion.ingest --dir data/raw/2026
```

### 4. Run the Application

**Terminal 1 - Start API Server:**
```bash
python -m src.app.server
# API docs at http://localhost:8000/docs
```

**Terminal 2 - Start Gradio UI:**
```bash
python -m src.app.ui
# Access UI at http://localhost:7860
```

## Usage Examples

### Via Python API
```python
from src.chain import get_answer

# Ask about default F1 regulations
answer = get_answer("What is the minimum weight of an F1 car in 2026?")
print(answer["answer"])
```

### Via Gradio UI
1. Open http://localhost:7860
2. Ask questions about F1 2026 regulations
3. **Upload Custom PDF**: Use the file upload widget to upload your own F1 rules document
4. Subsequent questions in that session will search the uploaded document
5. Click "New Session ID" to clear uploads and start fresh

## Architecture

```
┌──────────────────────────────────────────┐
│           User Query                     │
└──────────────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│    Session & Collection Routing                             │
│  (Check for uploaded PDF collection)                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
        ▼                             ▼
┌──────────────────────┐     ┌─────────────────────┐
│  Uploaded            │     │  Default Year       │
│  Collection          │     │  Collection         │
│  (ChromaDB)          │     │  (2026)             │
└──────────┬───────────┘     └─────────┬───────────┘
           │                           │
           └──────────────┬────────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │ Retrieve docs    │
                 │ (Semantic)       │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Answer LLM      │
                 │  (with context)  │
                 └────────┬─────────┘
                          │
                          ▼
                 ┌──────────────────┐
                 │  Guard-rails     │
                 │  + Citations     │
                 └──────────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain, LangServe |
| Vector Store | ChromaDB |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| LLM | Google Gemini / HuggingFace |
| PDF Processing | PyMuPDF4LLM |
| UI | Gradio |
| Server | FastAPI + Uvicorn |

## Testing

Run the comprehensive test suite:

```bash
pytest tests/test_chain.py -v
```

Test coverage includes:
- **Retriever Tests**: Document retrieval and k-parameter limiting
- **Upload Collection Tests**: Session mapping and uploaded collection routing
- **Guardrails Tests**: Prompt injection and on-topic detection
- **Chat History Tests**: History management and persistence
- **Pipeline Tests**: Full chain with default and uploaded collections
- **Edge Cases**: Error handling and graceful degradation

## Limitations

- English language only
- Requires internet for LLM API calls (Gemini/HuggingFace)
- Uploaded PDFs must follow FIA regulation naming convention for metadata extraction
- Session data stored in-memory (will be cleared on server restart)
- Single-server deployment (no horizontal scaling)

## Future Enhancements

- [ ] Multi-year regulation comparison
- [ ] Persistent session storage (database backend)
- [ ] Fine-tuned domain-specific embeddings
- [ ] Streaming responses for large answers
- [ ] Cloud deployment (Docker/Kubernetes)
- [ ] Support for non-FIA regulation documents
- [ ] Multi-language support
- [ ] Advanced metadata extraction for various document types

## License

MIT License - For educational purposes only.

## Acknowledgments

- [FIA for the F1 regulations](https://www.fia.com/regulation/category/110)

