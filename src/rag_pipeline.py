"""
src/rag_pipeline.py
Multi-hop Retrieval-Augmented Generation pipeline.

Builds a FAISS index over arXiv paper abstracts and answers complex
queries by decomposing them into sub-questions and retrieving evidence
iteratively across multiple hops.

Usage:
    from src.rag_pipeline import build_index, RAGPipeline
    from configs.training_config import RAGConfig

    config = RAGConfig()

    # One-time: build the knowledge base
    build_index(config)

    # At inference time
    pipeline = RAGPipeline(model_path="./new_merged_model_fp32", config=config)
    answer, contexts, subqs, trace = pipeline.query("Your question here")
"""

import torch
import faiss
import numpy as np
from datasets import load_dataset, load_from_disk
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Index Building ───────────────────────────────────────────────────────────

def build_index(config):
    """
    Download arXiv abstracts, generate embeddings, build a FAISS L2 index,
    and persist both the index and the raw documents to disk.
    """
    print(f"Loading {config.num_documents} arXiv abstracts...")
    dataset = load_dataset(
        config.dataset_name, split=f"train[:{config.num_documents}]"
    )
    documents = dataset["abstract"]

    print(f"Encoding documents with {config.embedding_model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(config.embedding_model, device=device)
    embeddings = embedder.encode(
        documents, show_progress_bar=True, convert_to_tensor=False
    ).astype("float32")

    print("Building FAISS L2 index...")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, config.index_path)
    dataset.save_to_disk(config.docs_path)

    print(f"Index saved: {config.index_path}")
    print(f"Documents saved: {config.docs_path}")
    print(f"Index contains {index.ntotal} vectors of dimension {embeddings.shape[1]}")


# ─── RAG Pipeline Class ───────────────────────────────────────────────────────

class RAGPipeline:
    """
    Multi-hop RAG controller.

    Workflow per query:
        1. Decompose query into sub-questions (LLM-guided)
        2. For each hop: FAISS retrieval → reasoning trace
        3. Generate follow-up questions from accumulated trace
        4. Synthesise a final answer over all retrieved evidence
    """

    def __init__(self, model_path: str, config):
        self.config = config
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load LLM
        print(f"Loading model from {model_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        # Load FAISS index and documents
        print("Loading FAISS index and knowledge base...")
        self.index = faiss.read_index(config.index_path)
        dataset = load_from_disk(config.docs_path)
        self.documents = dataset["abstract"]

        # Load embedding model
        self.embedder = SentenceTransformer(config.embedding_model, device=device)
        print("RAGPipeline ready.")

    # ── LLM Generation ────────────────────────────────────────────────────────

    def _generate(self, prompt: str, max_new_tokens: int = None, temperature: float = None) -> str:
        max_new_tokens = max_new_tokens or self.config.max_new_tokens
        temperature = temperature or self.config.temperature

        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k_sampling,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _embed(self, texts: list) -> np.ndarray:
        return self.embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def _search(self, query: str, k: int = None) -> list[int]:
        k = k or self.config.top_k
        query_vec = self._embed([query])
        _, indices = self.index.search(query_vec, k)
        return [int(i) for i in indices[0]]

    def _format_context(self, doc_ids: list[int], max_chars: int = None) -> tuple[str, list[str]]:
        max_chars = max_chars or self.config.max_context_chars
        texts = [self.documents[i] for i in doc_ids]
        joined = "\n\n---\n\n".join(texts)
        return joined[:max_chars], texts

    # ── Sub-Question Decomposition ────────────────────────────────────────────

    def _decompose(self, question: str, hops: int) -> list[str]:
        prompt = (
            f"Break the following question into {hops} short, logical sub-questions "
            f"to answer sequentially:\n\nQuestion: {question}\nSub-questions:"
        )
        raw = self._generate(prompt, max_new_tokens=120, temperature=0.3)
        lines = [l.strip("- ").strip() for l in raw.splitlines() if l.strip()]
        subs = [l.split(".", 1)[-1].strip() if "." in l[:3] else l for l in lines]
        return subs[:hops] if subs else [question]

    def _make_followup(self, question: str, trace: list[str]) -> str:
        prompt = (
            f"You are solving a multi-hop question.\n"
            f"Original: {question}\n"
            f"Current reasoning:\n{'  '.join(trace)}\n\n"
            f"Suggest ONE next short sub-question:"
        )
        result = self._generate(prompt, max_new_tokens=60, temperature=0.3)
        return result.strip().splitlines()[0].strip()

    # ── Answer Synthesis ──────────────────────────────────────────────────────

    def _synthesise(self, question: str, evidence: list[str]) -> str:
        context = "\n\n---\n\n".join(evidence)
        prompt = (
            f"### Instruction:\nAnswer the question using the context below.\n\n"
            f"### Context:\n{context}\n\n"
            f"### Guidance:\n"
            f"- Reuse key phrases and terminology from the context.\n"
            f"- Keep the answer factual and concise.\n\n"
            f"### Question:\n{question}\n\n"
            f"### Response:"
        )
        return self._generate(prompt, max_new_tokens=self.config.max_new_tokens, temperature=0.3)

    # ── Main Query Interface ──────────────────────────────────────────────────

    def query(
        self,
        question: str,
        hops: int = None,
        k: int = None,
    ) -> tuple[str, list[str], list[str], list[str]]:
        """
        Answer a question using multi-hop RAG.

        Args:
            question: The user query.
            hops:     Number of retrieval hops (default: config.num_hops).
            k:        Top-k documents per hop (default: config.top_k).

        Returns:
            answer:       Final synthesised answer.
            contexts:     List of retrieved document snippets used.
            subquestions: Decomposed sub-questions for this query.
            trace:        Step-by-step reasoning trace.
        """
        hops = hops or self.config.num_hops
        k = k or self.config.top_k

        subquestions = self._decompose(question, hops)
        evidence_ids, trace = [], []
        current_q = subquestions[0] if subquestions else question

        for hop in range(hops):
            retrieved = self._search(current_q, k=k)
            evidence_ids += [i for i in retrieved if i not in evidence_ids]

            ctx_text, _ = self._format_context(retrieved, max_chars=800)
            trace.append(f"[Hop {hop + 1}] Q: {current_q}\nContext:\n{ctx_text[:500]}")

            if hop < hops - 1:
                current_q = self._make_followup(question, trace)

        _, contexts = self._format_context(evidence_ids, max_chars=self.config.max_context_chars)
        answer = self._synthesise(question, contexts)

        return answer, contexts, subquestions, trace
