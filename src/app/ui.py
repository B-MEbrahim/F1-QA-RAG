"""
Gradio UI for the F1 RAG Bot.

Provides a chat interface for asking questions about F1 regulations and race results.
"""
import gradio as gr
import uuid
import requests
import json
from config.config import SERVER_HOST, SERVER_PORT

# ============ Server Configuration ============
LANGSERVE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ANSWER_CHAIN_ENDPOINT = f"{LANGSERVE_URL}/answer/invoke"
CLEAR_HISTORY_ENDPOINT = f"{LANGSERVE_URL}/clear"


# ============ Chat Interface ============

def respond(message: str, history: list, session_id: str):
    """
    Process user message and return bot response via LangServe.
    
    Args:
        message: User's question
        history: Chat history (list of message dicts with role/content)
        session_id: Unique session identifier
    
    Returns:
        Updated history with new response
    """
    if not message.strip():
        return history, ""
    
    try:
        # Call answer_chain via LangServe
        payload = {
            "question": message,
            "chat_history": history,
            "session_id": session_id
        }
        
        response = requests.post(
            ANSWER_CHAIN_ENDPOINT,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        
        # Extract answer from LangServe response
        answer = result.get("output", result) if isinstance(result, dict) else str(result)
        
        # Add to history using messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        
        return history, ""
    
    except requests.exceptions.ConnectionError:
        error_msg = "❌ Could not connect to server. Make sure the FastAPI server is running."
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


def clear_chat(session_id: str):
    """Clear chat history via server endpoint."""
    try:
        requests.post(
            CLEAR_HISTORY_ENDPOINT,
            json={"session_id": session_id},
            timeout=10
        )
    except Exception as e:
        print(f"Warning: Could not clear history on server: {e}")
    return [], ""


def create_session_id():
    """Generate a new session ID."""
    return str(uuid.uuid4())[:8]


# ============ Build Gradio Interface ============

def create_demo():
    """Create and return the Gradio demo interface."""
    
    with gr.Blocks(
        title="F1 RAG Assistant",
        theme=gr.themes.Soft(primary_hue="red"),
        css="""
        .chatbot-container { height: 500px !important; }
        .header { text-align: center; margin-bottom: 20px; }
        """
    ) as demo:
        
        # Header
        gr.Markdown(
            """
            # 🏎️ F1 Regulations Assistant
            
            Ask questions about **FIA Formula 1 regulations** or **race results**.
            
            **Examples:**
            - "What is the minimum weight of an F1 car in 2026?"
            - "What are the DRS rules?"
            - "Who won the 2025 Bahrain Grand Prix?"
            - "What are the penalties for track limits violations?"
            """,
            elem_classes=["header"]
        )
        
        # Session state
        session_id = gr.State(create_session_id)
        
        # Chat interface
        chatbot = gr.Chatbot(
            label="Chat",
            height=450,
            elem_classes=["chatbot-container"],
            # type="messages"
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your Question",
                placeholder="Ask about F1 regulations or race results...",
                scale=4,
                show_label=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)
        
        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            new_session_btn = gr.Button("🔄 New Session", variant="secondary")
        
        # Intent legend
        gr.Markdown(
            """
            ---
            **Response Types:** 📜 Regulations | 🏁 Race Results | 💬 General | 🚫 Blocked | ❌ Error
            """
        )
        
        # Event handlers
        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )
        
        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )
        
        clear_btn.click(
            fn=clear_chat,
            inputs=[session_id],
            outputs=[chatbot, msg_input]
        )
        
        new_session_btn.click(
            fn=lambda: ([], "", create_session_id()),
            outputs=[chatbot, msg_input, session_id]
        )
    
    return demo


# ============ Main Entry Point ============

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
