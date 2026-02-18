# ============ Chat History Store ============
from langchain_core.messages import HumanMessage, AIMessage

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
        