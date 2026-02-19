"""
Gradio UI for the F1 RAG Bot.
"""
import gradio as gr
import uuid
import requests
from pathlib import Path
from config.config import SERVER_HOST, SERVER_PORT

# ============ Server Configuration ============
LANGSERVE_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
ANSWER_CHAIN_ENDPOINT = f"{LANGSERVE_URL}/answer/invoke"
CLEAR_HISTORY_ENDPOINT = f"{LANGSERVE_URL}/clear"
CLEAR_UPLOAD_ENDPOINT = f"{LANGSERVE_URL}/clear-upload"
CLEAR_UPLOAD_ENDPOINT = f"{LANGSERVE_URL}/clear-upload"


# ============ Chat Interface ============

def respond(message: str, history: list, session_id: str):
    """
    Process user message and return bot response via LangServe.
    History is a list of {"role": ..., "content": ...} dicts.
    """
    if not message.strip():
        return history, ""

    history = history or []

    try:
        payload = {
            "input": {
                "question": message,
                "session_id": session_id,
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


def upload_rules_file(file_obj, session_id: str):
    """Upload a PDF rules file to the server and index it for this session."""
    if file_obj is None:
        return "Please choose a PDF file first."

    file_path = Path(getattr(file_obj, "name", file_obj))
    if file_path.suffix.lower() != ".pdf":
        return "Only PDF files are supported."

    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                f"{LANGSERVE_URL}/upload",
                data={"session_id": session_id},
                files={"file": (file_path.name, f, "application/pdf")},
                timeout=120
            )
        response.raise_for_status()
        result = response.json()
        return f"Uploaded {result.get('file_name', file_path.name)} ({result.get('chunk_count', 0)} chunks)"
    except Exception as e:
        return f"❌ Upload failed: {e}"


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
            upload_file = gr.File(
                label="Upload PDF Rules",
                file_types=[".pdf"],
                scale=3
            )
            upload_btn = gr.Button("Upload", variant="secondary", scale=1)
            upload_status = gr.Textbox(
                label="Upload Status",
                interactive=False,
                scale=4
            )

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

        upload_btn.click(
            fn=upload_rules_file,
            inputs=[upload_file, session_id],
            outputs=[upload_status]
        )

        def reset_session_with_clear(current_session_id: str):
            """Reset session: clear uploads and create new session ID."""
            # Clear uploaded files for the current session
            try:
                requests.post(
                    CLEAR_UPLOAD_ENDPOINT,
                    json={"session_id": current_session_id},
                    timeout=10
                )
                print(f"Cleared uploads for session: {current_session_id}")
            except Exception as e:
                print(f"Warning: Could not clear uploads: {e}")
            
            # Generate new session
            new_id = create_session_id()
            return [], "", new_id, f"Session ID: {new_id}", None, ""

        new_session_btn.click(
            fn=reset_session_with_clear,
            inputs=[session_id],
            outputs=[chatbot, msg_input, session_id, session_display, upload_file, upload_status]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )