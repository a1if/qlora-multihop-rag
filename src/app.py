"""
src/app.py
Gradio demo for the multi-hop RAG pipeline.

Uses the same environment configuration as the API service (QMR_LLM,
QMR_INDEX_DIR, ...), so the demo can run against a local model or any
OpenAI-compatible endpoint.

Usage:
    python -m src.app
    # Opens at http://localhost:7860
"""

import gradio as gr

from src.api import config_from_env
from src.rag_pipeline import RAGPipeline

# ─── Load Pipeline ────────────────────────────────────────────────────────────

config = config_from_env()
pipeline = RAGPipeline.from_config(config)


# ─── Inference ────────────────────────────────────────────────────────────────

def respond(message: str, history: list, mode: str, num_hops: int, top_k: int):
    """Answer one message; return updated history, sources markdown and timing."""
    if not message.strip():
        return history, "", "—"

    result = pipeline.answer(message, mode=mode, k=int(top_k), hops=int(num_hops), short=False)

    lines = []
    if result.queries:
        lines.append("**Search queries:**")
        lines += [f"{i}. {q}" for i, q in enumerate(result.queries, 1)]
    if result.passages:
        lines.append("\n**Retrieved passages:**")
        lines += [f"\n*[{i}] {p.get('title', '')}* — {p['text'][:300]}..." for i, p in enumerate(result.passages, 1)]
    sources = "\n".join(lines) or "_Closed-book answer — no retrieval used._"

    t = result.timings
    timing = (f"{result.latency_s:.2f}s total · retrieval {t['retrieval_s']:.2f}s · "
              f"LLM {t['llm_s']:.2f}s over {t['llm_calls']} call(s)")
    history.append((message, result.answer))
    return history, sources, timing


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-Hop RAG", theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate")) as demo:
    gr.Markdown(
        f"""
        # Multi-Hop RAG
        *LLM: `{pipeline.llm.name}` · index: `{config.index_dir}`*

        Compare retrieval strategies: closed-book, single-hop, iterative follow-up search,
        and up-front question decomposition.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480)
            msg_box = gr.Textbox(placeholder="Ask a multi-hop question...", label="Your message", lines=2)
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            with gr.Accordion("Settings", open=True):
                mode = gr.Radio(["closed", "single", "iterative", "decompose"], value="iterative", label="Mode")
                num_hops = gr.Slider(1, 4, value=config.num_hops, step=1, label="Hops / sub-questions")
                top_k = gr.Slider(1, 12, value=config.top_k, step=1, label="Passages given to the reader")

            timing_box = gr.Textbox(label="Timing", interactive=False)
            sources_box = gr.Markdown(value="*Sources will appear here...*")

    # ── Event Wiring ──────────────────────────────────────────────────────────

    state = gr.State([])

    def submit(message, history, mode, num_hops, top_k):
        history, sources, timing = respond(message, history, mode, num_hops, top_k)
        return history, history, "", sources, timing

    inputs = [msg_box, state, mode, num_hops, top_k]
    outputs = [chatbot, state, msg_box, sources_box, timing_box]
    submit_btn.click(submit, inputs=inputs, outputs=outputs)
    msg_box.submit(submit, inputs=inputs, outputs=outputs)
    clear_btn.click(lambda: ([], [], "", "*Sources will appear here...*", "—"), outputs=outputs)


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
