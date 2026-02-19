"""
Gradio UI for the F1 RAG Bot.
"""
import gradio as gr
import uuid
import requests
from config.config import SERVER_HOST, SERVER_PORT

# ============ Server Configuration ============
LANGSERVE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ANSWER_CHAIN_ENDPOINT = f"{LANGSERVE_URL}/answer/invoke"
CLEAR_HISTORY_ENDPOINT = f"{LANGSERVE_URL}/clear"


# ============ Chat Interface ============

def respond(message: str, history: list, session_id: str):
    """
    Process user message and return bot response via LangServe.
    History is a list of {"role": ..., "content": ...} dicts.
    """
    if not message.strip():
        return history, ""

    try:
        payload = {
            "input": {
                "question": message,
            },
            "config": {
                "configurable": {
                    "session_id": session_id
                }
            }
        }

        print(f"Sending to {ANSWER_CHAIN_ENDPOINT} with session {session_id}")
        response = requests.post(
            ANSWER_CHAIN_ENDPOINT,
            json=payload,
            timeout=60
        )
        response.raise_for_status()

        result = response.json()

        # LangServe wraps output: {"output": <chain_output>, "metadata": {...}}
        # Chain may return a plain string OR a dict like {"answer": "..."}
        output = result.get("output", "Error: No output key in response")
        if isinstance(output, dict):
            answer = output.get("answer", str(output))
        else:
            answer = output

        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": answer})
        return history, ""

    except requests.exceptions.ConnectionError:
        error_msg = f"❌ Could not connect to {LANGSERVE_URL}. Is server.py running?"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
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

    with gr.Blocks(
        title="F1 RAG Assistant",
        theme=gr.themes.Soft(primary_hue="red"),
        css="""
        .chatbot-container { height: 600px !important; }
        .header { text-align: center; margin-bottom: 20px; }
        """
    ) as demo:

        gr.Markdown(
            """
            # F1 Regulations Assistant

            Ask questions about **FIA Formula 1 regulations**.
            """,
            elem_classes=["header"]
        )

        session_id = gr.State(create_session_id)

        chatbot = gr.Chatbot(
            label="Chat",
            height=550,
            elem_classes=["chatbot-container"],
        )

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="Ex: What is the penalty for changing an engine?",
                scale=4,
                show_label=False
            )
            submit_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("🗑️ Clear History", variant="secondary")
            new_session_btn = gr.Button("🔄 New Session ID", variant="secondary")

        session_display = gr.Markdown(value=lambda: f"Session ID: {create_session_id()}")

        msg_input.submit(
            fn=respond,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )

        submit_btn.click(
            fn=respond,
            inputs=[msg_input, chatbot, session_id],
            outputs=[chatbot, msg_input]
        )

        clear_btn.click(
            fn=clear_chat,
            inputs=[session_id],
            outputs=[chatbot, msg_input]
        )

        def reset_session():
            new_id = create_session_id()
            return [], "", new_id, f"Session ID: {new_id}"

        new_session_btn.click(
            fn=reset_session,
            outputs=[chatbot, msg_input, session_id, session_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )