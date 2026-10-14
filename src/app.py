"""
src/app.py
Gradio chat demo for the multi-hop RAG pipeline.

Uses the same environment configuration as the API service (QMR_LLM,
QMR_INDEX_DIR, QMR_RETRIEVER, ...), so the demo can run the model in-process
or through a vLLM / Ollama server. scripts/run_chatbot.ps1 sets these for the
fine-tuned model and the HotpotQA index.

Usage:
    python -m src.app
    # Opens at http://127.0.0.1:7860
"""

import os

import gradio as gr

from src.api import config_from_env
from src.rag_pipeline import RAGPipeline
from src.text_utils import is_abstention, parse_sources

# ─── Load Pipeline ────────────────────────────────────────────────────────────

config = config_from_env()
pipeline = RAGPipeline.from_config(config)

EXAMPLES = [
    "Glenn Quagmire is voiced by the creator of what 2012 film?",
    "Venus is a play by an American playwright who won what prize?",
    "Which university was founded first, Duke University or Tufts University?",
]


# ─── Inference ────────────────────────────────────────────────────────────────

def respond(message: str, history: list, mode: str, num_hops: int, top_k: int):
    """Answer one message; return updated history, sources markdown and timing."""
    if not message.strip():
        return history, "", "—"

    result = pipeline.answer(message, mode=mode, k=int(top_k), hops=int(num_hops), short=True)
    cited = [result.passages[i - 1] for i in parse_sources(result.raw_output) if 0 < i <= len(result.passages)]

    if is_abstention(result.answer):
        reply = "I couldn't find enough evidence in the corpus to answer that."
    else:
        reply = f"**{result.answer}**"
        if cited:
            reply += "\n\nSources: " + ", ".join(p.get("title", "?") for p in cited)

    lines = []
    if result.queries:
        lines.append("**Searches run:**")
        lines += [f"{i}. {q}" for i, q in enumerate(result.queries, 1)]
    if result.passages:
        lines.append("\n**Passages given to the model** (cited ones marked ✓):")
        cited_ids = {id(p) for p in cited}
        lines += [f"\n{'✓ ' if id(p) in cited_ids else ''}*[{i}] {p.get('title', '')}*: {p['text'][:300]}..."
                  for i, p in enumerate(result.passages, 1)]
    sources = "\n".join(lines) or "_Closed-book answer: no retrieval used._"

    t = result.timings
    timing = (f"{result.latency_s:.2f}s total · retrieval {t['retrieval_s']:.2f}s · "
              f"LLM {t['llm_s']:.2f}s over {t['llm_calls']} call(s)")
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return history, sources, timing


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(title="Multi-Hop RAG") as demo:
    gr.Markdown(
        f"""
        # Multi-Hop RAG
        *LLM: `{pipeline.llm.name}` · retriever: `{config.retriever}` · index: `{config.index_dir}`*

        Ask questions that need two facts from two Wikipedia paragraphs. The corpus is the
        ~5,000 paragraphs behind the 500 HotpotQA benchmark questions, so ask about those topics.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480)
            msg_box = gr.Textbox(placeholder="Ask a multi-hop question and press Enter...",
                                 label="Your message", lines=1)
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn = gr.Button("Clear")
            gr.Examples(EXAMPLES, inputs=msg_box)

        with gr.Column(scale=2):
            with gr.Accordion("Settings", open=True):
                mode = gr.Radio(["closed", "single", "iterative", "decompose"], value="iterative", label="Mode")
                num_hops = gr.Slider(1, 4, value=config.num_hops, step=1, label="Hops / sub-questions")
                top_k = gr.Slider(1, 12, value=config.top_k, step=1, label="Passages given to the reader")

            timing_box = gr.Textbox(label="Timing", interactive=False)
            sources_box = gr.Markdown(value="*Searches and passages will appear here...*")

    # ── Event Wiring ──────────────────────────────────────────────────────────

    def submit(message, history, mode, num_hops, top_k):
        history, sources, timing = respond(message, history or [], mode, num_hops, top_k)
        return history, "", sources, timing

    inputs = [msg_box, chatbot, mode, num_hops, top_k]
    outputs = [chatbot, msg_box, sources_box, timing_box]
    submit_btn.click(submit, inputs=inputs, outputs=outputs)
    msg_box.submit(submit, inputs=inputs, outputs=outputs)
    clear_btn.click(lambda: ([], "", "*Searches and passages will appear here...*", "—"), outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name=os.environ.get("QMR_APP_HOST", "127.0.0.1"), server_port=7860,
                theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"))
