"""
Shared fakes: a deterministic bag-of-words embedder and a scripted LLM, so the
whole pipeline, API and benchmark harness run in CI without downloading models.
"""

import re
import zlib

import numpy as np
import pytest

from configs.training_config import RAGConfig

DIM = 512


class FakeEmbedder:
    """Hashes lowercase words into a fixed-size count vector."""

    def encode(self, texts, **kwargs):
        vecs = np.zeros((len(texts), DIM), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                vecs[row, zlib.crc32(word.encode()) % DIM] += 1.0
        return vecs


class FakeLLM:
    """
    Scripted responses keyed on which prompt template it receives.
    The reader only answers correctly if the bridge passage was retrieved.
    """

    name = "fake"

    def __init__(self):
        self.calls = []

    def generate(self, instruction, max_new_tokens, do_sample=False, temperature=0.3, top_p=0.9, top_k=40):
        self.calls.append(instruction)
        if "Write ONE short search query" in instruction:
            return "Query: Marie Curie birthplace"
        if "Break the question into" in instruction:
            return "1. Who discovered polonium?\n2. Where was Marie Curie born?"
        if QUESTION in instruction and "born in Warsaw" in instruction:
            return "<think>bridge found</think>Answer: Poland"
        return "Answer: France"

    def close(self):
        pass


CORPUS = [
    {"id": 0, "title": "Polonium", "text": "Polonium is a chemical element discovered by Marie Curie and Pierre Curie in 1898."},
    {"id": 1, "title": "Marie Curie", "text": "Marie Curie was born in Warsaw, Poland, and later moved to Paris."},
    {"id": 2, "title": "France", "text": "France is a country in Western Europe; many famous scientists were born and worked in France."},
    {"id": 3, "title": "Radium", "text": "Radium is a chemical element; the scientist who isolated it worked in France."},
    {"id": 4, "title": "Eiffel Tower", "text": "The Eiffel Tower is a landmark in Paris, the capital of France."},
]

QUESTION = "In which country was the scientist who discovered polonium born?"


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def retriever(embedder):
    from src.retriever import DenseRetriever

    return DenseRetriever.build(CORPUS, embedder)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def config():
    return RAGConfig(top_k=2, num_hops=2)


@pytest.fixture
def pipeline(fake_llm, retriever, config):
    from src.rag_pipeline import RAGPipeline

    return RAGPipeline(fake_llm, retriever, config)
