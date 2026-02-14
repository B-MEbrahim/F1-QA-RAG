# F1 RAG Bot 🏎️

A Retrieval-Augmented Generation (RAG) assistant for Formula 1 regulations and race results. Built as a final project for the NVIDIA DLI RAG Agents course.

## Overview

This application allows users to ask questions about:
- **FIA F1 Regulations** (Technical, Sporting, Financial, Operational)
- **Race Results** (via FastF1 API)

The system uses semantic search over chunked regulation PDFs and an LLM to generate grounded answers with source citations.

## Features

- 📄 **Document Ingestion**: PDF parsing → Markdown → Semantic chunking → Vector embeddings
- 🔍 **Semantic Search**: ChromaDB vector store with year-based collections
- 🤖 **Intent Routing**: Structured LLM output to route queries to the right tool
- 🏁 **Race Results**: Live F1 data via FastF1 API
- 🛡️ **Guard-rails**: Input validation and output grounding checks
- 📊 **Evaluation**: Retrieval and answer quality metrics
- 🌐 **API Server**: FastAPI + LangServe endpoints
- 💬 **Chat UI**: Gradio/Streamlit interface

## Project Structure

```
f1_rag_bot/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── cache/           # FastF1 cache
│   ├── chromadb/        # Vector store
│   └── raw/             # FIA regulation PDFs
│       └── 2026/
├── src/
│   ├── config.py        # Centralized configuration
│   ├── chain.py         # Main RAG pipeline
│   ├── models.py        # Pydantic models
│   ├── server.py        # FastAPI/LangServe endpoints
│   ├── ui.py            # Gradio/Streamlit UI
│   ├── ingestion/
│   │   └── ingest.py    # PDF ingestion pipeline
│   ├── tools/
│   │   ├── retriever.py # Regulation search tool
│   │   └── f1_stats.py  # FastF1 race results tool
│   ├── guardrails/
│   │   └── checks.py    # Safety & factuality checks
│   └── evaluation/
│       └── evaluate.py  # Eval metrics pipeline
└── tests/
    └── test_chain.py
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

**Required:** `NVIDIA_API_KEY` for embeddings (nv-embedqa-e5-v5) and LLM (Mixtral-8x7B)

### 3. Ingest Regulations

Place FIA regulation PDFs in `data/raw/<year>/` then run:

```bash
python -m src.ingestion.ingest --dir data/raw/2026
```

### 4. Run the Application

**Option A: Gradio UI**
```bash
python -m src.ui
```

**Option B: API Server**
```bash
python -m src.server
# API docs at http://localhost:8000/docs
```

## Usage Examples

```python
from src.chain import get_answer

# Ask about regulations
answer = get_answer("What is the minimum weight of an F1 car in 2026?")

# Ask about race results
answer = get_answer("Who won the 2025 Bahrain Grand Prix?")
```

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   User      │────▶│  Router LLM  │────▶│  Tool Call  │
│   Query     │     │  (Intent)    │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
                    ┌───────────────────────────┼───────────────────────────┐
                    │                           │                           │
                    ▼                           ▼                           ▼
            ┌───────────────┐           ┌───────────────┐           ┌───────────────┐
            │  Regulations  │           │  Race Results │           │  General Chat │
            │  (Retriever)  │           │   (FastF1)    │           │               │
            └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
                    │                           │                           │
                    └───────────────────────────┼───────────────────────────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │  Answer LLM   │
                                        │  (with ctx)   │
                                        └───────┬───────┘
                                                │
                                                ▼
                                        ┌───────────────┐
                                        │  Guard-rails  │
                                        │  + Citations  │
                                        └───────────────┘
```

## Technology Stack

| Component | Technology |
|-----------|------------|
| Framework | LangChain, LangServe |
| Vector Store | ChromaDB |
| Embeddings | NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`) |
| LLM | NVIDIA NIM (`mistralai/mixtral-8x7b-instruct-v0.1`) |
| F1 Data | FastF1 |
| UI | Gradio |
| Server | FastAPI + Uvicorn |

## Evaluation

Run the evaluation pipeline:

```bash
python -m src.evaluation.evaluate
```

Metrics:
- **Retrieval Hit Rate**: % of queries where expected source was retrieved
- **Answer Relevance**: Keyword overlap between question and answer
- **Faithfulness**: Grounding score (answer vs. context)

## Limitations

- English language only
- 2026 regulations only (can add more years)
- Simple keyword-based guardrails (can upgrade to NLI models)
- Not production-ready (local deployment only)

## Future Enhancements

- [ ] Multi-year regulation comparison
- [ ] Driver/team statistics tool
- [ ] Fine-tuned domain-specific embeddings
- [ ] Streaming responses
- [ ] Cloud deployment (Docker/Kubernetes)

## License

MIT License - For educational purposes only.

## Acknowledgments

- NVIDIA DLI for the course materials
- FIA for the F1 regulations
- FastF1 for the race data API
