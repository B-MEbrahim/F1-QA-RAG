"""
Gradio UI for the F1 RAG Bot.

Provides a chat interface for asking questions about F1 regulations and race results.
"""
import gradio as gr
import uuid
from src.chain import get_answer, clear_history

# ============ Chat Interface ============

def respond(message: str, history: list, session_id: str):
    """
    Process user message and return bot response.
    
    Args:
        message: User's question
        history: Chat history (list of [user, bot] pairs)
        session_id: Unique session identifier
    
    Returns:
        Updated history with new response
    """
    if not message.strip():
        return history, ""
    
    # Get answer from RAG pipeline
    result = get_answer(message, session_id=session_id)
    
    # Format response with intent indicator
    intent_emoji = {
        "REGULATIONS": "📜",
        "RACE_RESULTS": "🏁",
        "GENERAL_CHAT": "💬",
        "BLOCKED": "🚫",
        "ERROR": "❌"
    }
    
    emoji = intent_emoji.get(result["intent"], "🤖")
    response = f"{emoji} {result['answer']}"
    
    # Add to history
    history.append([message, response])
    
    return history, ""


def clear_chat(session_id: str):
    """Clear chat history."""
    clear_history(session_id)
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
            avatar_images=(None, "🏎️")
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
