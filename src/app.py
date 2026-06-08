"""
src/app.py
Gradio chat interface for the QLoRA + Multi-Hop RAG system.

Features:
    - Conversational memory (multi-turn)
    - Source citation panel
    - Inference timing
    - Toggle between RAG and fine-tuned-only modes

Usage:
    python src/app.py
    # Opens at http://localhost:7860
"""

import time
import torch
import gradio as gr
from src.rag_pipeline import RAGPipeline
from configs.training_config import RAGConfig

# ─── Load Pipeline ────────────────────────────────────────────────────────────

config = RAGConfig()
pipeline = RAGPipeline(model_path="./new_merged_model_fp32", config=config)


# ─── Inference ────────────────────────────────────────────────────────────────

def respond(
    message: str,
    history: list,
    use_rag: bool,
    num_hops: int,
    top_k: int,
) -> tuple[list, str, str]:
    """
    Generate a response and return updated history, sources, and timing.

    Args:
        message:  Current user message.
        history:  List of (user, assistant) turn pairs.
        use_rag:  Whether to use multi-hop RAG or fine-tuned-only mode.
        num_hops: Number of retrieval hops.
        top_k:    Documents retrieved per hop.

    Returns:
        history:  Updated conversation history.
        sources:  Formatted source citations string.
        timing:   Inference time string.
    """
    if not message.strip():
        return history, "", "—"

    start = time.time()

    if use_rag:
        answer, contexts, subquestions, trace = pipeline.query(
            message, hops=num_hops, k=top_k
        )
        # Format sources
        source_lines = [f"**Sub-questions decomposed:**"]
        for i, sq in enumerate(subquestions, 1):
            source_lines.append(f"  {i}. {sq}")
        source_lines.append("\n**Retrieved contexts (snippets):**")
        for i, ctx in enumerate(contexts[:3], 1):
            source_lines.append(f"\n*Source {i}:* {ctx[:300]}...")
        sources_text = "\n".join(source_lines)
    else:
        # Fine-tuned model only (no retrieval)
        prompt = f"### Instruction:\n{message}\n\n### Response:"
        tokenizer = pipeline.tokenizer
        model = pipeline.model
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        answer = full[len(prompt):].strip()
        sources_text = "_Fine-tuned model only — no retrieval used._"

    elapsed = time.time() - start
    timing = f"⏱ {elapsed:.2f}s"

    history.append((message, answer))
    return history, sources_text, timing


# ─── Gradio UI ────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="QLoRA + Multi-Hop RAG",
    theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
) as demo:

    gr.Markdown(
        """
        # 🔬 QLoRA Fine-Tuning + Multi-Hop RAG
        *DeepSeek-R1-Distill-Qwen-1.5B · FAISS · arXiv Knowledge Base*

        Ask questions grounded in 5,000 arXiv paper abstracts.
        Toggle **RAG mode** off to query the fine-tuned model directly.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480, bubble_full_width=False)
            msg_box = gr.Textbox(
                placeholder="Ask something about NLP, LLMs, RAG, transformers...",
                label="Your message",
                lines=2,
            )
            with gr.Row():
                submit_btn = gr.Button("Send", variant="primary")
                clear_btn  = gr.Button("Clear")

        with gr.Column(scale=2):
            with gr.Accordion("⚙️ Settings", open=True):
                use_rag   = gr.Checkbox(value=True,  label="Use Multi-Hop RAG")
                num_hops  = gr.Slider(1, 3, value=2, step=1, label="Retrieval Hops")
                top_k     = gr.Slider(3, 12, value=8, step=1, label="Top-K Documents per Hop")

            timing_box = gr.Textbox(label="⏱ Inference Time", interactive=False)
            sources_box = gr.Markdown(label="📚 Sources & Reasoning Trace", value="*Sources will appear here...*")

    gr.Examples(
        examples=[
            ["What is the significance of attention mechanisms in transformers?", True, 2, 8],
            ["How does QLoRA reduce memory usage during fine-tuning?",           True, 2, 8],
            ["Explain the difference between dense and sparse retrieval.",       True, 2, 8],
            ["Tell me a fun fact about penguins.",                               False, 2, 8],
        ],
        inputs=[msg_box, use_rag, num_hops, top_k],
        label="Example queries",
    )

    # ── Event Wiring ──────────────────────────────────────────────────────────

    state = gr.State([])  # conversation history

    def submit(message, history, use_rag, num_hops, top_k):
        history, sources, timing = respond(message, history, use_rag, num_hops, top_k)
        return history, history, "", sources, timing

    submit_btn.click(
        submit,
        inputs=[msg_box, state, use_rag, num_hops, top_k],
        outputs=[chatbot, state, msg_box, sources_box, timing_box],
    )
    msg_box.submit(
        submit,
        inputs=[msg_box, state, use_rag, num_hops, top_k],
        outputs=[chatbot, state, msg_box, sources_box, timing_box],
    )
    clear_btn.click(
        lambda: ([], [], "", "*Sources will appear here...*", "—"),
        outputs=[chatbot, state, msg_box, sources_box, timing_box],
    )


if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
