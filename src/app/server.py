"""
FastAPI server with LangServe endpoints for the F1 RAG Bot.

Endpoints:
- POST /chat: Simple chat endpoint
- POST /ask: Full response with metadata
- GET /health: Health check
- POST /clear: Clear chat history
"""
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langserve import add_routes
from src.chain import rag_chain
from src.chain import get_answer, chat, clear_history, clear_chat_history
from config.config import SERVER_HOST, SERVER_PORT, UPLOADS_DIR
from src.models import ChatRequest, ChatResponse, FullResponse, ClearRequest
from src.ingestion.ingest import ingest_pdf_to_collection
from src.tools.uploads import set_session_collection
from src.tools import extract_metadata_from_filename
from pathlib import Path


# ============ FastAPI App ============

app = FastAPI(
    title="F1 RAG Bot API",
    description="API for querying F1 regulations and race results using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)



# ============ Endpoints ============

@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "F1 RAG Bot API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Simple chat (returns answer only)",
            "/ask": "POST - Full response with metadata",
            "/clear": "POST - Clear chat history",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "f1-rag-bot"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Simple chat endpoint that returns just the answer.
    
    Args:
        request: ChatRequest with question and optional session_id
    
    Returns:
        ChatResponse with the answer
    """
    try:
        answer = chat(request.question, request.session_id)
        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=FullResponse)
async def ask_endpoint(request: ChatRequest):
    """
    Full ask endpoint that returns answer with metadata.
    
    Args:
        request: ChatRequest with question and optional session_id
    
    Returns:
        FullResponse with answer, intent, sources, and validation info
    """
    try:
        result = get_answer(request.question, request.session_id)
        return FullResponse(
            answer=result["answer"],
            sources=result["sources"],
            validation_info=result["validation_info"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear")
async def clear_endpoint(request: ClearRequest):
    """
    Clear chat history for a session.
    
    Args:
        request: ClearRequest with session_id
    
    Returns:
        Confirmation message
    """
    clear_chat_history(request.session_id)
    return {"message": f"Chat history cleared for session: {request.session_id}"}


@app.post("/upload")
async def upload_rules_file(
    session_id: str = Form(...),
    file: UploadFile = File(...)
):
    """
    Upload a PDF rules file and index it for the given session.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = UPLOADS_DIR / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = upload_dir / Path(file.filename).name

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    # Extract metadata from filename to get year
    metadata = extract_metadata_from_filename(str(file_path))
    year = metadata.get("regulation_year", "unknown")
    print(f"[upload] Extracted metadata: {metadata}")
    
    # Create collection name with year if available
    collection_name = f"upload_{session_id}_{year}"
    print(f"[upload] Creating collection: {collection_name}")
    
    chunk_count = ingest_pdf_to_collection(str(file_path), collection_name)
    set_session_collection(session_id, collection_name)
    print(f"[upload] Successfully ingested {chunk_count} chunks into {collection_name}")

    return {
        "message": "Upload complete",
        "file_name": file.filename,
        "chunk_count": chunk_count,
        "session_id": session_id,
        "collection_name": collection_name,
        "regulation_year": year
    }


@app.post("/clear-upload")
async def clear_upload(request: ClearRequest):
    """
    Clear uploaded files and collection mapping for a session.
    """
    import shutil
    from src.tools.uploads import clear_session_collection
    
    session_id = request.session_id
    
    # Clear in-memory collection mapping
    clear_session_collection(session_id)
    
    # Delete uploaded files directory
    upload_dir = UPLOADS_DIR / session_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir)
    
    return {"message": f"Uploaded files cleared for session: {session_id}"}


add_routes(
    app,
    rag_chain,
    path="/answer",
    enabled_endpoints=["invoke", "batch", "stream"]
)


# ============ Main Entry Point ============

if __name__ == "__main__":
    uvicorn.run(
        "src.app.server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=True
    )
