import os
from openai import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from .chat_history import add_to_history, get_chat_history

# import tools
from src.tools import get_retriever
from src.guardrails.checks import validate_input, validate_output
from config.config import LLM_PROVIDER, LLM_MODEL, GEMINI_KEY, HF_TOKEN, HF_BASE_URL


# ============ LLM Setup ============
match LLM_PROVIDER:
    case "gemini":
        chat_llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            api_key=GEMINI_KEY
        )
    case "huggingface":
        chat_llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=HF_TOKEN,
            base_url=HF_BASE_URL
        )


# ============ Answer Chain (with chat history) ============
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an F1 Official Assistant. Answer the question based ONLY on the provided context.
If the context is empty or irrelevant, say you don't know.
Be concise but thorough. Always cite the specific regulation or source when possible.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

answer_chain = answer_prompt | chat_llm | StrOutputParser()

# ============ Main Pipeline ============
def get_answer(question: str, session_id: str = "default", year: int = 2026) -> dict:
    """
    RAG pipeline with vector database retrieval, guardrails, and chat history.
    
    Args:
        question: User's question
        session_id: Session identifier for chat history
        year: Year for F1 regulations (default: 2026)
    
    Returns:
        dict with keys: answer, sources, validation_info
    """
    print(f"\n--- Processing: {question} ---")
    
    # Input validation (guardrails)
    is_valid, validation_msg = validate_input(question)
    if not is_valid:
        return {
            "answer": f"I cannot process this request: {validation_msg}",
            "sources": [],
            "validation_info": {"input_blocked": True, "reason": validation_msg}
        }

    # Get retriever and fetch relevant documents
    print(f"--- Retrieving documents from vector database ({year}) ---")
    retriever = get_retriever(year)
    retrieved_docs = retriever.invoke(question)
    
    # Format context with sources
    context = "\n\n".join(
        [f"[Source: {d.metadata.get('source', 'Unknown')} | Rule: {d.metadata.get('rule_id', 'N/A')}]\n{d.page_content}" 
         for d in retrieved_docs]
    )
    
    # Get chat history
    history = get_chat_history(session_id)
    
    # Generate answer
    print(f"--- Context Length: {len(str(context))} chars ---")
    print(f"--- Retrieved {len(retrieved_docs)} documents ---")
    
    raw_answer = answer_chain.invoke({
        "context": context,
        "question": question,
        "chat_history": history
    })
    
    # Output validation (guardrails) - add citations if we have sources
    final_answer, validation_info = validate_output(raw_answer, retrieved_docs)
    
    # Add to chat history
    add_to_history(session_id, question, raw_answer)
    
    return {
        "answer": final_answer,
        "sources": [d.metadata for d in retrieved_docs] if retrieved_docs else [],
        "validation_info": validation_info
    }


# ============ Simple Interface ============
def chat(question: str, session_id: str = "default", year: int = 2026) -> str:
    """Simple chat interface that returns just the answer string."""
    result = get_answer(question, session_id, year)
    return result["answer"]


if __name__ == "__main__":
    # Test the pipeline
    # print("=" * 60)
    # print(chat("What is the maximum fuel mass flow for 2026?"))
    # print("=" * 60)
    # print(chat("Tell me more about the fuel regulations", session_id="test"))
    print("=" * 60)
    print(chat("What is the maximum enigne horse power of an f1 car?", session_id="test"))