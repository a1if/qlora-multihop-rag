"""
src/retriever.py
Dense passage retrieval with FAISS (cosine similarity via inner product
over L2-normalised embeddings).

Documents are dicts with at least a "text" key and optionally "title".
"""

import json
import os
import threading

import numpy as np


def doc_text(doc: dict) -> str:
    """Text used for both embedding and the reader prompt."""
    title = doc.get("title")
    return f"{title}: {doc['text']}" if title else doc["text"]


def merge_round_robin(ranked_lists: list[list[dict]], k: int) -> list[dict]:
    """
    Interleave several ranked lists (one per hop / query), dropping duplicates,
    and keep the first k. Every hop gets a fair share of the evidence budget.
    """
    merged, seen = [], set()
    for rank in range(max((len(r) for r in ranked_lists), default=0)):
        for ranked in ranked_lists:
            if rank < len(ranked):
                doc = ranked[rank]
                key = doc.get("id", doc_text(doc))
                if key not in seen:
                    seen.add(key)
                    merged.append(doc)
                    if len(merged) == k:
                        return merged
    return merged


class DenseRetriever:
    """FAISS IndexFlatIP over normalised sentence embeddings."""

    def __init__(self, docs: list[dict], index, embedder):
        self.docs = docs
        self.index = index
        self.embedder = embedder
        self._lock = threading.Lock()   # search() is called from many threads under batching

    # ── Encoding ──────────────────────────────────────────────────────────────

    @staticmethod
    def _encode(embedder, texts: list[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
        vecs = np.asarray(
            embedder.encode(texts, batch_size=batch_size, show_progress_bar=progress, convert_to_numpy=True),
            dtype="float32",
        )
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs / np.clip(norms, 1e-12, None)

    # ── Build / Persist ───────────────────────────────────────────────────────

    @classmethod
    def build(cls, docs: list[dict], embedder, batch_size: int = 64) -> "DenseRetriever":
        import faiss

        vecs = cls._encode(embedder, [doc_text(d) for d in docs], batch_size, progress=True)
        index = faiss.IndexFlatIP(vecs.shape[1])
        index.add(vecs)
        return cls(docs, index, embedder)

    def save(self, directory: str):
        import faiss

        os.makedirs(directory, exist_ok=True)
        faiss.write_index(self.index, os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.jsonl"), "w", encoding="utf-8") as f:
            for doc in self.docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str, embedder) -> "DenseRetriever":
        import faiss

        index = faiss.read_index(os.path.join(directory, "index.faiss"))
        with open(os.path.join(directory, "docs.jsonl"), encoding="utf-8") as f:
            docs = [json.loads(line) for line in f]
        return cls(docs, index, embedder)

    @staticmethod
    def exists(directory: str) -> bool:
        return all(
            os.path.exists(os.path.join(directory, name)) for name in ("index.faiss", "docs.jsonl")
        )

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, k: int) -> list[dict]:
        with self._lock:
            query_vec = self._encode(self.embedder, [query])
            scores, indices = self.index.search(query_vec, k)
        return [
            {**self.docs[i], "score": float(s)}
            for s, i in zip(scores[0], indices[0], strict=True)
            if i != -1
        ]
