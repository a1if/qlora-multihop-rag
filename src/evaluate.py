"""
src/evaluate.py
Three-way evaluation framework: Base Model vs Fine-Tuned vs Multi-Hop RAG.

Metrics computed:
    - BLEU            (n-gram precision vs fine-tuned reference)
    - ROUGE-1/2/L     (recall-oriented lexical overlap)
    - BERTScore F1    (contextual semantic similarity)
    - Cosine Similarity (embedding-space groundedness in retrieved context)
    - Response Length + Lexical Diversity (stylistic quality)

Usage:
    from src.evaluate import Evaluator
    from src.rag_pipeline import RAGPipeline
    from configs.training_config import FinetuneConfig

    evaluator = Evaluator(rag_pipeline, base_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    results_df, metrics = evaluator.run()
"""

import re
import gc
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from evaluate import load as load_metric
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


# ─── Default Test Questions ────────────────────────────────────────────────────

GENERAL_QUESTIONS = [
    "What is the capital of France?",
    "Tell me a fun fact about giraffes.",
]

RAG_QUESTIONS = [
    "Explain the main components of a transformer model in NLP.",
    "What is the significance of QLoRA in fine-tuning large language models?",
]


# ─── Text Utilities ───────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def response_length(text: str) -> int:
    return len(text.split())


def lexical_diversity(text: str) -> float:
    words = text.split()
    return len(set(words)) / len(words) if words else 0.0


# ─── Evaluator Class ──────────────────────────────────────────────────────────

class Evaluator:
    """
    Runs the three-way model comparison and computes all metrics.

    Args:
        rag_pipeline: An initialised RAGPipeline instance.
        base_model_name: HuggingFace model ID for the unmodified base model.
        questions: Optional list of test questions; uses defaults if None.
    """

    def __init__(self, rag_pipeline, base_model_name: str, questions: list[str] = None):
        self.rag = rag_pipeline
        self.questions = questions or (GENERAL_QUESTIONS + RAG_QUESTIONS)

        # Load metrics
        self.bleu      = load_metric("bleu")
        self.rouge     = load_metric("rouge")
        self.bertscore = load_metric("bertscore")
        self.embedder  = SentenceTransformer("all-MiniLM-L6-v2")

        # Load base model pipeline for comparison
        print(f"Loading base model: {base_model_name}")
        self.base_pipe = pipeline(
            "text-generation",
            model=base_model_name,
            model_kwargs={"torch_dtype": torch.float16, "device_map": "auto"},
        )

    # ── Inference Helpers ─────────────────────────────────────────────────────

    def _base_response(self, question: str) -> str:
        prompt = f"### Instruction:\n{question}\n### Response:"
        result = self.base_pipe(
            prompt, max_new_tokens=100, do_sample=True,
            temperature=0.7, top_k=50, top_p=0.95,
        )
        return result[0]["generated_text"][len(prompt):].strip()

    def _ft_response(self, question: str, context: str = None) -> str:
        if context:
            prompt = (
                f"### Instruction:\nAnswer using the context.\n\n"
                f"### Context:\n{context}\n\n"
                f"### Question:\n{question}\n\n### Response:"
            )
        else:
            prompt = f"### Instruction:\n{question}\n\n### Response:"

        model = self.rag.model
        tokenizer = self.rag.tokenizer
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                num_beams=5,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        full = tokenizer.decode(output[0], skip_special_tokens=True)
        return full[len(prompt):].strip()

    # ── Metric Computation ────────────────────────────────────────────────────

    def _cosine_groundedness(self, answer: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        context_text = " ".join(contexts)
        emb_a = self.embedder.encode(answer, convert_to_tensor=True)
        emb_c = self.embedder.encode(context_text, convert_to_tensor=True)
        return float(util.cos_sim(emb_a, emb_c))

    def _compute_metrics(self, prediction: str, reference: str, contexts: list[str]) -> dict:
        pred_n = normalize(prediction)
        ref_n  = normalize(reference)

        bleu_score   = self.bleu.compute(predictions=[pred_n], references=[ref_n])["bleu"]
        rouge_scores = self.rouge.compute(predictions=[pred_n], references=[ref_n])
        bert_out     = self.bertscore.compute(predictions=[pred_n], references=[ref_n], lang="en")
        bert_f1      = float(np.mean(bert_out["f1"]))
        cosine       = self._cosine_groundedness(pred_n, contexts)

        return {
            "bleu":      bleu_score,
            "rouge1":    rouge_scores["rouge1"],
            "rouge2":    rouge_scores["rouge2"],
            "rougeL":    rouge_scores["rougeL"],
            "bertscore": bert_f1,
            "cosine":    cosine,
            "length":    response_length(pred_n),
            "diversity": lexical_diversity(pred_n),
        }

    # ── Main Evaluation Run ───────────────────────────────────────────────────

    def run(self, verbose: bool = True) -> tuple[pd.DataFrame, dict]:
        """
        Evaluate all questions and return results DataFrame and aggregate metrics.
        """
        rows = []
        agg = {k: [] for k in ["bleu", "rouge1", "rouge2", "rougeL", "bertscore", "cosine", "length", "diversity"]}

        for q in self.questions:
            base_res = self._base_response(q)
            ft_res   = self._ft_response(q)
            rag_res, contexts, _, _ = self.rag.query(q)

            # Truncate and normalise RAG response
            rag_res = normalize(" ".join(rag_res.split()[:512]))
            ft_res  = normalize(ft_res)

            m = self._compute_metrics(rag_res, ft_res, contexts)
            for k, v in m.items():
                agg[k].append(v)

            rows.append({
                "Question":  q,
                "Base":      base_res[:80] + "...",
                "FineTuned": ft_res[:80] + "...",
                "RAG":       rag_res[:80] + "...",
                **{f"rag_{k}": v for k, v in m.items()},
            })

            # Free memory after each question
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        df = pd.DataFrame(rows)

        metrics = {
            "Avg BLEU":             np.mean(agg["bleu"]),
            "Avg ROUGE-1":          np.mean(agg["rouge1"]),
            "Avg ROUGE-2":          np.mean(agg["rouge2"]),
            "Avg ROUGE-L":          np.mean(agg["rougeL"]),
            "Avg BERTScore F1":     np.mean(agg["bertscore"]),
            "Avg Cosine Similarity":np.mean(agg["cosine"]),
            "Avg Response Length":  np.mean(agg["length"]),
            "Avg Lexical Diversity":np.mean(agg["diversity"]),
        }

        if verbose:
            self._print_results(rows, metrics)
            self._plot(metrics, agg["length"])

        return df, metrics

    # ── Display & Plotting ────────────────────────────────────────────────────

    def _print_results(self, rows: list, metrics: dict):
        table_data = [
            [r["Question"][:45], "Base Model",  r["Base"]]  for r in rows
        ] + [
            [r["Question"][:45], "Fine-Tuned",  r["FineTuned"]] for r in rows
        ] + [
            [r["Question"][:45],
             f"RAG (BLEU={r['rag_bleu']:.2f}, R1={r['rag_rouge1']:.2f})",
             r["RAG"]]
            for r in rows
        ]
        print("\n" + tabulate(
            table_data,
            headers=["Question", "Model", "Response Snippet"],
            tablefmt="fancy_grid",
            maxcolwidths=[45, 35, 70],
        ))
        print("\n" + tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="fancy_grid", floatfmt=".3f"))

    def _plot(self, metrics: dict, lengths: list):
        # Bar chart of semantic/textual quality metrics
        labels = ["BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BERTScore", "Cosine"]
        values = [
            metrics["Avg BLEU"], metrics["Avg ROUGE-1"], metrics["Avg ROUGE-2"],
            metrics["Avg ROUGE-L"], metrics["Avg BERTScore F1"], metrics["Avg Cosine Similarity"],
        ]
        colours = ["steelblue", "royalblue", "deepskyblue", "lightcoral", "violet", "mediumseagreen"]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].bar(labels, values, color=colours)
        axes[0].set_title("RAG Evaluation Metrics", fontsize=13)
        axes[0].set_ylabel("Score (0–1)")
        axes[0].set_ylim(0, 1)
        axes[0].grid(axis="y", linestyle="--", alpha=0.5)

        axes[1].boxplot(lengths, patch_artist=True,
                        boxprops=dict(facecolor="lightgreen", color="green"),
                        medianprops=dict(color="darkgreen"))
        axes[1].set_title("RAG Response Length Distribution", fontsize=13)
        axes[1].set_ylabel("Tokens per Response")
        axes[1].grid(axis="y", linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.savefig("evaluation_results.png", dpi=150, bbox_inches="tight")
        plt.show()
        print("Plot saved: evaluation_results.png")
