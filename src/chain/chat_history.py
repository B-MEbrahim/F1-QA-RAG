# ============ Chat History Store ============
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory

# Simple in-memory store 
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
        
# history with state manager
stores: dict[str, BaseChatMessageHistory] = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Returns the message history object for a given session_id."""
    if session_id not in stores:
        stores[session_id] = ChatMessageHistory()
    return stores[session_id]

def add_to_history(session_id:str, human_msg: str, ai_msg: str):
    history = get_session_history(session_id)
    history.add_user_message(human_msg)
    history.add_ai_message(ai_msg)

def get_chat_history_list(session_id: str):
    """Returns the raw list of messages """
    return get_session_history(session_id).messages
 
def clear_history(session_id: str):
    if session_id in stores:
        stores[session_id].clear()
