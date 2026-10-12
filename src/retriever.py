"""
src/retriever.py
Passage retrieval: dense FAISS search (cosine similarity via inner product
over L2-normalised embeddings), BM25, and a hybrid of both with optional
cross-encoder reranking.

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


# ─── Sparse Retrieval (BM25) ──────────────────────────────────────────────────

_STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on that the to was were what which who whom "
    "whose when where why how did does do with this these those or not".split()
)


def tokenize(text: str) -> list[str]:
    import re

    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over doc_text (title + paragraph) with an inverted index."""

    def __init__(self, docs: list[dict], k1: float = 1.2, b: float = 0.75):
        from collections import Counter, defaultdict

        self.docs = docs
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = np.zeros(len(docs), dtype="float32")
        for i, doc in enumerate(docs):
            tokens = tokenize(doc_text(doc))
            lengths[i] = len(tokens)
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.norm = k1 * (1 - b + b * lengths / max(lengths.mean(), 1.0))
        n = len(docs)
        self.idf = {t: float(np.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))) for t, p in self.postings.items()}

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self.docs), dtype="float32")
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.norm[i])
        return scores

    def search(self, query: str, k: int) -> list[dict]:
        scores = self.scores(query)
        top = np.argsort(-scores)[:k]
        return [{**self.docs[i], "score": float(scores[i])} for i in top if scores[i] > 0]


# ─── Hybrid Retrieval ─────────────────────────────────────────────────────────

def reciprocal_rank_fusion(ranked_lists: list[list[dict]], k: int, c: int = 60) -> list[dict]:
    """Fuse rankings by summing 1 / (c + rank); robust to incomparable score scales."""
    fused, docs = {}, {}
    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked):
            key = doc.get("id", doc_text(doc))
            fused[key] = fused.get(key, 0.0) + 1.0 / (c + rank + 1)
            docs.setdefault(key, doc)
    order = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{**docs[key], "score": fused[key]} for key in order]


class HybridRetriever:
    """
    Dense + BM25 candidates fused with reciprocal rank fusion, optionally
    re-scored by a cross-encoder that reads query and passage together.

    Same search(query, k) interface as DenseRetriever, so the pipeline is unchanged.
    """

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever | None = None, reranker=None,
                 candidates: int = 20):
        self.dense = dense
        self.docs = dense.docs
        self.bm25 = bm25 or BM25Retriever(dense.docs)
        self.reranker = reranker
        self.candidates = candidates
        self._lock = threading.Lock()

    def search(self, query: str, k: int) -> list[dict]:
        n = max(self.candidates, k)
        pool = reciprocal_rank_fusion([self.dense.search(query, n), self.bm25.search(query, n)], n)
        if self.reranker is None:
            return pool[:k]
        with self._lock:
            scores = self.reranker.predict([(query, doc_text(d)) for d in pool])
        ranked = sorted(zip(scores, range(len(pool)), pool, strict=True), key=lambda x: -x[0])[:k]
        return [{**doc, "score": float(score)} for score, _, doc in ranked]


def make_retriever(kind: str, dense: DenseRetriever, rerank_model: str = None, candidates: int = 20):
    """kind: dense | hybrid | hybrid_rerank"""
    if kind == "dense":
        return dense
    if kind == "hybrid":
        return HybridRetriever(dense, candidates=candidates)
    if kind == "hybrid_rerank":
        from sentence_transformers import CrossEncoder

        return HybridRetriever(dense, reranker=CrossEncoder(rerank_model), candidates=candidates)
    raise ValueError(f"Unknown retriever {kind!r}; expected dense | hybrid | hybrid_rerank")
