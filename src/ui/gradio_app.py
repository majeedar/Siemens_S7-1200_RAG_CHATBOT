"""Gradio-based chat interface for the S7-1200 RAG chatbot.

Provides a professional UI with chat, source display, settings panel,
query metadata, and user feedback.
"""

import json
import logging
import time
from typing import Optional

import gradio as gr

from src.config import settings
from src.chains.rag_chain import S7RAGChain, extract_citations
from src.retrieval.vector_store import QdrantVectorStore
from src.monitoring.feedback import FeedbackStore
from src.monitoring.metrics import start_metrics_server
from src.monitoring.tracing import configure_langsmith

logger = logging.getLogger(__name__)

CUSTOM_CSS = """
.header-container {
    text-align: center;
    padding: 1rem;
    background: linear-gradient(135deg, #1a237e 0%, #0d47a1 50%, #01579b 100%);
    border-radius: 12px;
    margin-bottom: 1rem;
    color: white;
}
.header-container h1 {
    margin: 0;
    font-size: 1.8rem;
    color: white !important;
}
.header-container p {
    margin: 0.3rem 0 0 0;
    opacity: 0.9;
    color: #e3f2fd !important;
}
.source-box {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 0.5rem;
    background: #fafafa;
    max-height: 500px;
    overflow-y: auto;
}
.metadata-box {
    font-family: monospace;
    font-size: 0.85rem;
}
.footer-text {
    text-align: center;
    font-size: 0.8rem;
    color: #666;
    padding: 0.5rem;
}
"""

EXAMPLE_QUERIES = [
    "How do I configure analog input AI_0?",
    "What is the maximum I/O capacity of S7-1200?",
    "Explain PID control function blocks",
    "How to connect HMI to S7-1200 CPU?",
    "What are the safety precautions for wiring?",
    "Show me example code for motion control",
    "What communication protocols does S7-1200 support?",
    "How do I configure PROFINET communication?",
]


class S7ChatbotUI:
    """Gradio Blocks application for the S7-1200 RAG chatbot."""

    def __init__(self, rag_chain: Optional[S7RAGChain] = None):
        self.vector_store = QdrantVectorStore()
        self.rag_chain = rag_chain or S7RAGChain(vector_store=self.vector_store)
        self.feedback_store = FeedbackStore()
        self.app: Optional[gr.Blocks] = None

        # Start monitoring
        if settings.ENABLE_METRICS:
            start_metrics_server(port=settings.METRICS_PORT)
        configure_langsmith(
            api_key=settings.LANGSMITH_API_KEY,
            project=settings.LANGSMITH_PROJECT,
        )

    def _get_db_info(self) -> str:
        """Get current database information."""
        info = self.vector_store.get_collection_info()
        if "error" in info:
            return f"Status: {info['status']}\nError: {info['error']}"
        return (
            f"Collection: {info['name']}\n"
            f"Documents: {info.get('points_count', 'N/A')}\n"
            f"Vectors: {info.get('vectors_count', 'N/A')}\n"
            f"Status: {info['status']}\n"
            f"Vector Size: {info['vector_size']}"
        )

    def build(self) -> gr.Blocks:
        """Construct and return the Gradio Blocks application."""
        with gr.Blocks(
            title="S7-1200 PLC Assistant",
            theme=gr.themes.Soft(
                primary_hue="blue",
                secondary_hue="indigo",
            ),
            css=CUSTOM_CSS,
        ) as app:
            # Header
            gr.HTML(
                """
                <div class="header-container">
                    <h1>S7-1200 PLC Technical Assistant</h1>
                    <p>RAG-powered chatbot for the Siemens S7-1200 System Manual</p>
                </div>
                """
            )

            # State for tracking last exchange (used by feedback buttons)
            last_query_state = gr.State("")
            last_answer_state = gr.State("")
            last_metadata_state = gr.State({})

            with gr.Row():
                # Left column: Chat interface (2/3 width)
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        label="Chat",
                        height=500,
                        type="messages",
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Ask a question about the S7-1200 PLC...",
                            label="Message",
                            scale=4,
                            container=False,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)

                    with gr.Row():
                        clear_btn = gr.Button("Clear Chat", variant="secondary")
                        retry_btn = gr.Button("Regenerate", variant="secondary")

                    with gr.Row():
                        gr.HTML("<b>Rate the last response:</b>")
                        thumbs_up_btn = gr.Button("👍", size="sm", min_width=60)
                        thumbs_down_btn = gr.Button("👎", size="sm", min_width=60)
                        flag_btn = gr.Button("🚩 Flag", size="sm", variant="stop", min_width=80)
                        feedback_status = gr.Textbox(
                            value="", label="", interactive=False, container=False,
                        )

                    gr.Markdown("### Example Questions")
                    examples = gr.Examples(
                        examples=[[q] for q in EXAMPLE_QUERIES],
                        inputs=[msg_input],
                        label="",
                    )

                # Right column: Settings and sources (1/3 width)
                with gr.Column(scale=1):
                    gr.Markdown("### Settings")
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=settings.OLLAMA_TEMPERATURE,
                        step=0.05,
                        label="Temperature",
                        info="Lower = more precise, Higher = more creative",
                    )
                    topk_slider = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=settings.RETRIEVAL_TOP_K,
                        step=1,
                        label="Top-K Sources",
                        info="Number of document chunks to retrieve",
                    )

                    gr.Markdown("### Database Info")
                    db_info = gr.Textbox(
                        label="",
                        value=self._get_db_info,
                        interactive=False,
                        lines=5,
                    )
                    refresh_btn = gr.Button("Refresh DB Info", size="sm")

                    gr.Markdown("### Retrieved Sources")
                    sources_display = gr.Markdown(
                        value="Sources will appear here after a query.",
                        label="",
                        elem_classes=["source-box"],
                    )

                    gr.Markdown("### Query Metadata")
                    metadata_display = gr.Code(
                        value="{}",
                        language="json",
                        label="",
                        interactive=False,
                    )

            # Footer
            gr.HTML(
                """
                <div class="footer-text">
                    <strong>S7-1200 RAG Chatbot</strong> |
                    Powered by LangChain, Qdrant, Ollama (Llama 3.1 8B), Gradio |
                    <em>Always verify critical information with official Siemens documentation</em>
                </div>
                """
            )

            # --- Event handlers ---
            def _history_to_pairs(chat_history):
                """Convert Gradio messages-format history to (user, assistant) pairs."""
                pairs = []
                user_buf = None
                for entry in (chat_history or []):
                    role = entry.get("role", "")
                    content = entry.get("content", "")
                    if role == "user":
                        user_buf = content
                    elif role == "assistant" and user_buf is not None:
                        pairs.append((user_buf, content))
                        user_buf = None
                return pairs

            def _on_submit(message, chat_history, temperature, top_k):
                if not message.strip():
                    return chat_history, "", "{}", "", "", "", {}

                self.rag_chain.update_settings(temperature=temperature, top_k=int(top_k))
                self.rag_chain.conversation_history = _history_to_pairs(chat_history)

                result = self.rag_chain.generate(message)

                sources_md = self.rag_chain.get_source_display(result["sources"])
                metadata_json = json.dumps(result["metadata"], indent=2)

                new_history = list(chat_history or [])
                new_history.append({"role": "user", "content": message})
                new_history.append({"role": "assistant", "content": result["answer"]})

                return (
                    new_history, sources_md, metadata_json, "",
                    message, result["answer"], result["metadata"],
                )

            def _on_clear():
                self.rag_chain.clear_history()
                return (
                    [], "Sources will appear here after a query.", "{}", "",
                    "", "", {},
                )

            def _on_retry(chat_history, temperature, top_k):
                if not chat_history or len(chat_history) < 2:
                    return chat_history, "", "{}", "", "", "", {}
                last_user_msg = None
                for entry in reversed(chat_history):
                    if entry.get("role") == "user":
                        last_user_msg = entry["content"]
                        break
                if not last_user_msg:
                    return chat_history, "", "{}", "", "", "", {}
                trimmed = chat_history[:-2]
                return _on_submit(last_user_msg, trimmed, temperature, top_k)

            def _on_feedback(feedback_type, last_query, last_answer, last_metadata):
                if not last_query:
                    return "No response to rate yet."
                self.feedback_store.record(
                    query=last_query,
                    answer=last_answer,
                    feedback_type=feedback_type,
                    metadata=last_metadata if isinstance(last_metadata, dict) else {},
                )
                label = {"thumbs_up": "👍 Thanks!", "thumbs_down": "👎 Noted.", "flag": "🚩 Flagged."}
                return label.get(feedback_type, "Recorded.")

            # Event bindings
            submit_inputs = [msg_input, chatbot, temperature_slider, topk_slider]
            submit_outputs = [
                chatbot, sources_display, metadata_display, msg_input,
                last_query_state, last_answer_state, last_metadata_state,
            ]

            msg_input.submit(
                fn=_on_submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )
            send_btn.click(
                fn=_on_submit,
                inputs=submit_inputs,
                outputs=submit_outputs,
            )
            clear_btn.click(
                fn=_on_clear,
                inputs=[],
                outputs=submit_outputs,
            )
            retry_btn.click(
                fn=_on_retry,
                inputs=[chatbot, temperature_slider, topk_slider],
                outputs=submit_outputs,
            )
            refresh_btn.click(
                fn=self._get_db_info,
                inputs=[],
                outputs=[db_info],
            )

            # Feedback buttons
            feedback_inputs = [last_query_state, last_answer_state, last_metadata_state]
            thumbs_up_btn.click(
                fn=lambda q, a, m: _on_feedback("thumbs_up", q, a, m),
                inputs=feedback_inputs,
                outputs=[feedback_status],
            )
            thumbs_down_btn.click(
                fn=lambda q, a, m: _on_feedback("thumbs_down", q, a, m),
                inputs=feedback_inputs,
                outputs=[feedback_status],
            )
            flag_btn.click(
                fn=lambda q, a, m: _on_feedback("flag", q, a, m),
                inputs=feedback_inputs,
                outputs=[feedback_status],
            )

        self.app = app
        return app

    def launch(self, **kwargs) -> None:
        """Build and launch the Gradio app."""
        if self.app is None:
            self.build()
        self.app.launch(
            server_name=kwargs.get("server_name", settings.GRADIO_SERVER_NAME),
            server_port=kwargs.get("server_port", settings.GRADIO_SERVER_PORT),
            share=kwargs.get("share", settings.GRADIO_SHARE),
        )


def create_app() -> gr.Blocks:
    """Factory function to create the Gradio application."""
    ui = S7ChatbotUI()
    return ui.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ui = S7ChatbotUI()
    ui.launch()
