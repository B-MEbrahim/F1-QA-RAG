import os
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from operator import itemgetter

# HuggingFace LLM (commented out - keeping for reference)
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# llm = HuggingFaceEndpoint(
#     repo_id=repo_id,
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     temperature=0.2,
# )
# chat_llm = ChatHuggingFace(llm=llm)

# import tools
from src.tools.retriever import search_f1_regulations, get_retriever
from src.tools.f1_stats import get_race_results_tool
from src.models import Race, Regulations, RouteQuery
from src.guardrails.checks import validate_input, validate_output

load_dotenv()

# ============ LLM Setup (NVIDIA NIM) ============
chat_llm = ChatNVIDIA(
    model="mistralai/mixtral-8x7b-instruct-v0.1",
    temperature=0.2,
    max_tokens=1024,
)
structured_llm = chat_llm.with_structured_output(RouteQuery)

# ============ Router Chain ============
router_system_prompt = """You are an expert F1 Assistant. 
Your task is to route the user's question and extract relevant parameters.
- For Rules/Regs (penalties, technical specs): Set intent='REGULATIONS' and fill 'regulations_query'. Default year is 2026.
- For Race Results (winners, podiums): Set intent='RACE_RESULTS' and fill 'race_query'.
- For anything else: Set intent='GENERAL_CHAT'.
"""

route_prompt = ChatPromptTemplate.from_messages([
    ("system", router_system_prompt),
    ("human", "{question}"),
])

router_chain = route_prompt | structured_llm

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

# ============ Chat History Store ============
# Simple in-memory store (for demo purposes)
chat_histories: dict[str, list] = {}

def get_chat_history(session_id: str) -> list:
    """Get chat history for a session."""
    return chat_histories.get(session_id, [])

def add_to_history(session_id: str, human_msg: str, ai_msg: str):
    """Add a message pair to chat history."""
    if session_id not in chat_histories:
        chat_histories[session_id] = []
    chat_histories[session_id].append(HumanMessage(content=human_msg))
    chat_histories[session_id].append(AIMessage(content=ai_msg))
    # Keep only last 10 exchanges (20 messages)
    if len(chat_histories[session_id]) > 20:
        chat_histories[session_id] = chat_histories[session_id][-20:]

def clear_history(session_id: str):
    """Clear chat history for a session."""
    if session_id in chat_histories:
        del chat_histories[session_id]

# ============ Main Pipeline ============
def get_answer(question: str, session_id: str = "default") -> dict:
    """
    Main RAG pipeline with routing, tool calls, guardrails, and chat history.
    
    Returns:
        dict with keys: answer, intent, sources, validation_info
    """
    print(f"\n--- Processing: {question} ---")
    
    # Input validation (guardrails)
    is_valid, validation_msg = validate_input(question)
    if not is_valid:
        return {
            "answer": f"I cannot process this request: {validation_msg}",
            "intent": "BLOCKED",
            "sources": [],
            "validation_info": {"input_blocked": True, "reason": validation_msg}
        }

    # Route and extract intent
    try:
        route_result: RouteQuery = router_chain.invoke({"question": question})
        intent = route_result.intent
        print(f"--- Detected Intent: {intent} ---")
    except Exception as e:
        print(f"Routing Error: {e}")
        return {
            "answer": "Sorry, I had trouble understanding your question. Please try rephrasing.",
            "intent": "ERROR",
            "sources": [],
            "validation_info": {"error": str(e)}
        }

    context = ""
    retrieved_docs = []

    # Invoke tools based on intent
    if intent == "REGULATIONS":
        year = route_result.regulations_query.year if route_result.regulations_query else 2026
        print(f"--- Tool Call: Searching Regulations ({year}) ---")
        
        # Get retriever and fetch docs
        retriever = get_retriever(year)
        retrieved_docs = retriever.invoke(question)
        
        # Format context with sources
        context = "\n\n".join(
            [f"[Source: {d.metadata.get('source', 'Unknown')} | Rule: {d.metadata.get('rule_id', 'N/A')}]\n{d.page_content}" 
             for d in retrieved_docs]
        )
    
    elif intent == "RACE_RESULTS":
        if route_result.race_query:
            year = route_result.race_query.year
            gp_name = route_result.race_query.gp_name
            print(f"--- Tool Call: FastF1 Results ({year} {gp_name}) ---")
            context = get_race_results_tool.invoke({"year": year, "gp_name": gp_name})
        else:
            context = "Error: Could not extract Race Name or Year from your question."

    else:  # GENERAL_CHAT
        context = "No specific F1 data retrieved. Answer based on general knowledge about Formula 1."

    # Get chat history
    history = get_chat_history(session_id)
    
    # Generate answer
    print(f"--- Context Length: {len(str(context))} chars ---")
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
        "intent": intent,
        "sources": [d.metadata for d in retrieved_docs] if retrieved_docs else [],
        "validation_info": validation_info
    }


# ============ Simple Interface ============
def chat(question: str, session_id: str = "default") -> str:
    """Simple chat interface that returns just the answer string."""
    result = get_answer(question, session_id)
    return result["answer"]


if __name__ == "__main__":
    # Test the pipeline
    print("=" * 60)
    print(chat("What is the maximum fuel mass flow for 2026?"))
    print("=" * 60)
    print(chat("Who won the Bahrain GP in 2025?"))
    print("=" * 60)
    # Test follow-up (chat history)
    print(chat("Tell me more about the fuel regulations", session_id="test"))
