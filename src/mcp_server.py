"""
src/mcp_server.py
MCP server exposing the multi-hop RAG pipeline as tools.

Tools:
    search(query, k)          retrieve passages (hybrid BM25 + dense + rerank by default)
    get_passage(passage_id)   full text of one passage
    answer(question, mode)    the whole pipeline: multi-hop retrieval + reader with citations
Resource:
    corpus://info             corpus size, retriever and model in use

Any MCP client can use it: the agent in src/agent.py, Claude Desktop / Claude
Code, Cursor, etc. Configuration comes from the same environment variables as
the API service (QMR_LLM, QMR_INDEX_DIR, QMR_RETRIEVER, ...).

Usage:
    python -m src.mcp_server                                 # stdio (desktop clients)
    python -m src.mcp_server --transport streamable-http     # http://127.0.0.1:8765/mcp
"""

import argparse
import logging
from typing import Annotated, Literal

from pydantic import Field

from src.retriever import doc_text

logger = logging.getLogger("qmr.mcp")

INSTRUCTIONS = (
    "Tools for answering multi-hop questions over the indexed corpus. Use `search` to find "
    "passages; when a passage names an entity you still need facts about, search for that entity. "
    "Use `answer` to run the full multi-hop pipeline in one call."
)


def _passage(doc: dict, max_chars: int | None = None) -> dict:
    text = doc["text"] if max_chars is None else doc["text"][:max_chars]
    out = {"id": doc.get("id"), "title": doc.get("title"), "text": text}
    if "score" in doc:
        out["score"] = round(float(doc["score"]), 4)
    return out


def create_server(pipeline, preview_chars: int = 600):
    """Build the MCP server around an existing RAGPipeline (tests pass a fake one)."""
    from mcp.server.mcpserver import MCPServer

    server = MCPServer(name="multihop-rag", instructions=INSTRUCTIONS, version="1.0.0")
    by_id = {doc.get("id"): doc for doc in pipeline.retriever.docs}

    @server.tool(description="Search the corpus. Returns the top-k passages (id, title, text preview, score).")
    def search(query: Annotated[str, Field(min_length=1, max_length=500)],
               k: Annotated[int, Field(ge=1, le=20)] = 5) -> list[dict]:
        return [_passage(d, preview_chars) for d in pipeline.retriever.search(query, k)]

    @server.tool(description="Return the full text of one passage by its id (from search results).")
    def get_passage(passage_id: int) -> dict:
        doc = by_id.get(passage_id)
        if doc is None:
            raise ValueError(f"No passage with id {passage_id}")
        return {**_passage(doc), "formatted": doc_text(doc)}

    @server.tool(description=(
        "Answer a question with the full pipeline. mode: 'iterative' (multi-hop, default), 'single' "
        "(one search) or 'closed' (no retrieval). Returns the answer, cited passages and the searches run."
    ))
    def answer(question: Annotated[str, Field(min_length=1, max_length=2000)],
               mode: Literal["closed", "single", "iterative", "decompose"] = "iterative") -> dict:
        result = pipeline.answer(question, mode=mode, short=True)
        return {
            "answer": result.answer,
            "queries": result.queries,
            "passages": [_passage(p, preview_chars) for p in result.passages],
            "latency_s": round(result.latency_s, 3),
        }

    @server.resource("corpus://info", name="corpus-info", mime_type="application/json",
                     description="Corpus size and pipeline configuration")
    def corpus_info() -> dict:
        config = pipeline.config
        return {
            "passages": len(pipeline.retriever.docs),
            "retriever": getattr(config, "retriever", "dense"),
            "llm": getattr(pipeline.llm, "name", "unknown"),
            "top_k": config.top_k,
            "hops": config.num_hops,
        }

    return server


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    # stdout carries the MCP protocol over stdio, so logs go to stderr
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from src.api import config_from_env
    from src.rag_pipeline import RAGPipeline

    server = create_server(RAGPipeline.from_config(config_from_env()))
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
