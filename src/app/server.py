"""
FastAPI server with LangServe endpoints for the F1 RAG Bot.

Endpoints:
- POST /chat: Simple chat endpoint
- POST /ask: Full response with metadata
- GET /health: Health check
- POST /clear: Clear chat history
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from langserve import add_routes
from src.chain import answer_chain
from src.chain import get_answer, chat, clear_history
from config.config import SERVER_HOST, SERVER_PORT
from src.models import ChatRequest, ChatResponse, FullResponse, ClearRequest


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
            intent=result["intent"],
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
    clear_history(request.session_id)
    return {"message": f"Chat history cleared for session: {request.session_id}"}


add_routes(
    app,
    answer_chain,
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
