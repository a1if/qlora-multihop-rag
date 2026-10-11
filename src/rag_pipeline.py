"""
src/rag_pipeline.py
Multi-hop Retrieval-Augmented Generation pipeline.

Retrieval modes (all RAG modes give the reader the same budget of k passages):
    closed     No retrieval; the model answers from its own weights.
    single     One dense retrieval with the original question.
    iterative  Retrieve with the question, ask the LLM for a follow-up search
               query given what was found (e.g. the bridge entity), retrieve
               again; repeat for `hops` rounds.
    decompose  Ask the LLM to split the question into sub-questions up front and
               retrieve for each one (falls back to the question if parsing fails).

Usage:
    from src.rag_pipeline import RAGPipeline, build_arxiv_index
    from configs.training_config import RAGConfig

    config = RAGConfig()
    build_arxiv_index(config)                       # one-time

    pipeline = RAGPipeline.from_config(config)      # LLM backend from config.llm
    result = pipeline.answer("Your question here", mode="iterative")
    print(result.answer, [p["title"] for p in result.passages], result.timings)
"""

import threading
import time
from dataclasses import dataclass, field

from src.llm import LLM, make_llm
from src.retriever import DenseRetriever, doc_text, merge_round_robin
from src.text_utils import extract_answer, extract_query, parse_subquestions, strip_reasoning

MODES = ("closed", "single", "iterative", "decompose")


# ─── Prompts ──────────────────────────────────────────────────────────────────

SHORT_ANSWER_RULE = (
    "Respond with only the short answer (an entity, date, number, or yes/no) "
    "on a single line in the form 'Answer: <answer>'."
)

READER_PROMPT = (
    "Answer the question using the passages below. {rule}\n\n"
    "Passages:\n{passages}\n\n"
    "Question: {question}"
)

CLOSED_PROMPT = "Answer the question. {rule}\n\nQuestion: {question}"

LONG_ANSWER_RULE = "Answer concisely and cite the passages you used like [1]."

FOLLOWUP_PROMPT = (
    "You are answering a multi-hop question one search at a time.\n\n"
    "Question: {question}\n\n"
    "Passages found so far:\n{passages}\n\n"
    "Write ONE short search query for the information still missing to answer "
    "the question (for example, the entity the passages above point to). "
    "Respond with only the query on a single line."
)

DECOMPOSE_PROMPT = (
    "Break the question into at most {n} simple sub-questions that can each be "
    "answered by a single Wikipedia search, in the order they must be answered. "
    "Write one sub-question per line.\n\n"
    "Question: {question}"
)


def format_passages(passages: list[dict], max_chars: int) -> str:
    return "\n".join(f"[{i}] {doc_text(p)[:max_chars]}" for i, p in enumerate(passages, 1))


# ─── RAG Pipeline ─────────────────────────────────────────────────────────────

@dataclass
class RAGResult:
    answer: str
    raw_output: str
    mode: str
    passages: list[dict] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    latency_s: float = 0.0
    timings: dict = field(default_factory=dict)   # retrieval_s, llm_s, llm_calls


class _StageTimer:
    """Accumulates wall-clock time and call counts per pipeline stage."""

    def __init__(self):
        self.totals = {"retrieval_s": 0.0, "llm_s": 0.0, "llm_calls": 0}

    def add(self, stage: str, seconds: float):
        self.totals[f"{stage}_s"] += seconds
        if stage == "llm":
            self.totals["llm_calls"] += 1


class RAGPipeline:
    def __init__(self, llm: LLM, retriever: DenseRetriever | None, config):
        self.llm = llm
        self.retriever = retriever
        self.config = config
        # per-thread stage timings, so concurrent answer() calls don't mix
        self._local = threading.local()

    @classmethod
    def from_config(cls, config, llm_spec: str = None, index_dir: str = None) -> "RAGPipeline":
        from sentence_transformers import SentenceTransformer

        embedder = SentenceTransformer(config.embedding_model)
        retriever = DenseRetriever.load(index_dir or config.index_dir, embedder)
        llm = make_llm(llm_spec or config.llm, config.prompt_style, config.reasoning, config.load_in_4bit)
        return cls(llm, retriever, config)

    # ── Instrumented Calls ────────────────────────────────────────────────────

    def _llm(self, instruction: str, max_new_tokens: int) -> str:
        c = self.config
        if c.reasoning:
            max_new_tokens += c.reasoning_token_budget
        start = time.perf_counter()
        try:
            return self.llm.generate(
                instruction, max_new_tokens, do_sample=c.do_sample,
                temperature=c.temperature, top_p=c.top_p, top_k=c.top_k_sampling,
            )
        finally:
            self._local.timer.add("llm", time.perf_counter() - start)

    def _search(self, query: str, k: int) -> list[dict]:
        start = time.perf_counter()
        try:
            return self.retriever.search(query, k)
        finally:
            self._local.timer.add("retrieval", time.perf_counter() - start)

    def _read(self, question: str, passages: list[dict], short: bool) -> tuple[str, str]:
        rule = SHORT_ANSWER_RULE if short else LONG_ANSWER_RULE
        if passages:
            prompt = READER_PROMPT.format(
                rule=rule, question=question,
                passages=format_passages(passages, self.config.max_passage_chars),
            )
        else:
            prompt = CLOSED_PROMPT.format(rule=rule, question=question)
        max_tokens = self.config.max_new_tokens if short else self.config.max_new_tokens_long
        raw = self._llm(prompt, max_tokens)
        return (extract_answer(raw) if short else strip_reasoning(raw)), raw

    def _followup_query(self, question: str, passages: list[dict]) -> str:
        prompt = FOLLOWUP_PROMPT.format(
            question=question, passages=format_passages(passages[:3], self.config.max_passage_chars)
        )
        return extract_query(self._llm(prompt, 32)) or question

    def _decompose(self, question: str, n: int) -> list[str]:
        raw = self._llm(DECOMPOSE_PROMPT.format(n=n, question=question), 96)
        return parse_subquestions(raw, n)

    # ── Retrieval Strategies ──────────────────────────────────────────────────

    def _retrieve(self, question: str, mode: str, k: int, hops: int) -> tuple[list[dict], list[str]]:
        if mode == "closed":
            return [], []
        if mode == "single":
            return self._search(question, k), [question]

        if mode == "decompose":
            queries = self._decompose(question, hops) or [question]
            return merge_round_robin([self._search(q, k) for q in queries], k), queries

        queries, ranked = [question], [self._search(question, k)]
        if mode == "iterative":
            for _ in range(hops - 1):
                evidence = merge_round_robin(ranked, k)
                query = self._followup_query(question, evidence)
                queries.append(query)
                ranked.append(self._search(query, k))
        else:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        return merge_round_robin(ranked, k), queries

    # ── Main Interface ────────────────────────────────────────────────────────

    def answer(self, question: str, mode: str = "iterative", k: int = None, hops: int = None,
               short: bool = True) -> RAGResult:
        """
        Args:
            question: The user query.
            mode:     One of MODES.
            k:        Passages given to the reader (same budget for every mode).
            hops:     Retrieval rounds for iterative / max sub-questions for decompose.
            short:    Short-answer extraction (benchmarks) vs free-form cited answer (demo).
        """
        if mode not in MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {MODES}")
        if mode != "closed" and self.retriever is None:
            raise ValueError(f"Mode {mode!r} needs a retriever")
        k = k or self.config.top_k
        hops = hops or self.config.num_hops

        self._local.timer = _StageTimer()
        start = time.perf_counter()
        passages, queries = self._retrieve(question, mode, k, hops)
        answer, raw = self._read(question, passages, short)
        return RAGResult(answer, raw, mode, passages, queries,
                         latency_s=time.perf_counter() - start, timings=dict(self._local.timer.totals))


# ─── arXiv Knowledge Base (demo corpus) ───────────────────────────────────────

def build_arxiv_index(config):
    """Download arXiv abstracts, embed them and save a DenseRetriever index."""
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    print(f"Loading {config.num_documents} arXiv abstracts...")
    dataset = load_dataset(config.dataset_name, split=f"train[:{config.num_documents}]")
    docs = [
        {"id": i, "title": " ".join(row["title"].split()), "text": " ".join(row["abstract"].split())}
        for i, row in enumerate(dataset)
    ]
    embedder = SentenceTransformer(config.embedding_model)
    retriever = DenseRetriever.build(docs, embedder)
    retriever.save(config.index_dir)
    print(f"Index with {retriever.index.ntotal} passages saved to {config.index_dir}")
