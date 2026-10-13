"""
src/mcp_server.py
MCP server exposing the multi-hop RAG pipeline as tools.

Tools (typed inputs validated by Pydantic; typed, schema'd outputs):
    search(query, k)          retrieve passages (hybrid BM25 + dense + rerank, per config)
    get_passage(passage_id)   full text of one passage
    answer(question, mode)    full pipeline: multi-hop retrieval + reader with citations
Resource:
    corpus://info             corpus size, retriever and model in use

Search tools only need the index; the LLM loads on the first `answer` call,
so the server starts in seconds and can be explored without a GPU.

Observability: the MCP SDK wraps every request in an OpenTelemetry span and
the pipeline adds child spans (see src/telemetry.py); every request also
emits one JSON log line on stderr with the tool, latency, outcome and trace id.

Any MCP client can use it: the LangGraph agent, MCP Inspector, Claude Desktop /
Claude Code, Cursor. Configuration comes from the same environment variables as
the API service (QMR_LLM, QMR_INDEX_DIR, QMR_RETRIEVER, ...).

Usage:
    python -m src.mcp_server                                 # stdio (desktop clients, Inspector)
    python -m src.mcp_server --transport streamable-http     # http://127.0.0.1:8765/mcp
"""

import argparse
import json
import logging
import threading
import time
from collections.abc import Callable
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from src.retriever import doc_text
from src.telemetry import current_trace_id, setup_tracing, span
from src.text_utils import parse_sources

logger = logging.getLogger("qmr.mcp")

INSTRUCTIONS = (
    "Tools for answering multi-hop questions over the indexed corpus. Use `search` to find "
    "passages; when a passage names an entity you still need facts about, search for that entity. "
    "Use `get_passage` for the full text of a result, or `answer` to run the whole multi-hop "
    "pipeline in one call."
)
MAX_K = 20
PREVIEW_CHARS = 700   # = RAGConfig.max_passage_chars: clients see exactly what the reader saw


# ─── Schemas ──────────────────────────────────────────────────────────────────

class Passage(BaseModel):
    id: int = Field(description="Passage id, usable with get_passage")
    title: str | None = Field(None, description="Wikipedia article title")
    text: str = Field(description="Passage text (truncated to a preview in search results)")
    score: float | None = Field(None, description="Retrieval score (higher is more relevant)")


class SearchResult(BaseModel):
    query: str
    results: list[Passage]
    truncated_chars: int = Field(description="Characters per passage text in this result")


class PassageDetail(Passage):
    formatted: str = Field(description="'title: text', exactly as the reader model sees it")


class AnswerResult(BaseModel):
    answer: str = Field(description="Short answer, or 'insufficient evidence'")
    queries: list[str] = Field(description="Searches the pipeline ran, in order")
    passages: list[Passage] = Field(description="Evidence given to the reader")
    citations: list[int] = Field(description="Ids of the passages the answer cites (empty if none)")
    latency_s: float


QueryText = Annotated[str, Field(min_length=1, max_length=500, description="Search query: a question or entity name")]
TopK = Annotated[int, Field(ge=1, le=MAX_K, description=f"Number of passages to return (1-{MAX_K})")]
Question = Annotated[str, Field(min_length=1, max_length=2000, description="The question to answer")]
Mode = Annotated[Literal["closed", "single", "iterative", "decompose"], Field(
    description="'iterative' = multi-hop (default), 'single' = one search, 'closed' = no retrieval")]


def _passage(doc: dict, max_chars: int | None = None) -> Passage:
    text = doc["text"] if max_chars is None else doc["text"][:max_chars]
    score = doc.get("score")
    return Passage(id=doc["id"], title=doc.get("title"), text=text,
                   score=round(float(score), 4) if score is not None else None)


# ─── Request Logging ──────────────────────────────────────────────────────────

async def log_requests(ctx, call_next):
    """One JSON log line per MCP request; runs inside the SDK's tracing span, so it carries the trace id."""
    name = ctx.params.get("name") if ctx.params else None
    start = time.perf_counter()
    outcome = "ok"
    try:
        result = await call_next(ctx)
        # middleware sees the wire form (a dict with "isError"); objects are handled for safety
        is_error = result.get("isError") if isinstance(result, dict) else getattr(result, "is_error", False)
        if is_error:
            outcome = "tool_error"
        return result
    except Exception as exc:
        outcome = type(exc).__name__
        raise
    finally:
        logger.info(json.dumps({
            "event": "mcp_request", "method": ctx.method, "tool": name, "outcome": outcome,
            "duration_ms": round(1000 * (time.perf_counter() - start), 1), "trace_id": current_trace_id(),
        }))


# ─── Background Loading ───────────────────────────────────────────────────────

class BackgroundRetriever:
    """
    Loads the retriever (embedder, reranker, BM25 index: ~35 s from disk) in a
    thread, so the server answers the MCP handshake immediately. Tool calls that
    arrive before it's ready wait for it, up to timeout_s.
    """

    def __init__(self, load: Callable[[], object], timeout_s: float = 300.0):
        self.timeout_s = timeout_s
        self._ready = threading.Event()
        self._value, self._error = None, None
        threading.Thread(target=self._run, args=(load,), name="retriever-loader", daemon=True).start()

    def _run(self, load):
        start = time.perf_counter()
        try:
            self._value = load()
            logger.info(json.dumps({"event": "retriever_ready", "passages": len(self._value.docs),
                                    "load_s": round(time.perf_counter() - start, 1)}))
        except Exception as exc:
            self._error = exc
            logger.exception("retriever failed to load")
        finally:
            self._ready.set()

    def get(self):
        from mcp.server.mcpserver.exceptions import ToolError

        if not self._ready.wait(self.timeout_s):
            raise ToolError("The search index is still loading; retry in a few seconds.")
        if self._error is not None:
            raise ToolError(f"The search index failed to load: {self._error}")
        return self._value

    @property
    def docs(self):
        return self.get().docs

    def search(self, query: str, k: int):
        return self.get().search(query, k)


# ─── Server ───────────────────────────────────────────────────────────────────

def create_server(retriever, config, get_pipeline: Callable[[], object], llm_name: str = "lazy"):
    """
    Build the MCP server.

    Args:
        retriever:     object with .search(query, k) and .docs
        config:        RAGConfig
        get_pipeline:  returns a RAGPipeline; called on the first `answer` (tests pass a ready one)
    """
    from mcp.server.mcpserver import MCPServer
    from mcp.server.mcpserver.exceptions import ToolError

    server = MCPServer(name="multihop-rag", instructions=INSTRUCTIONS, version="1.0.0",
                       middleware=[log_requests])
    lookup: dict = {}

    def passages_by_id() -> tuple[dict, str]:
        # built on first use, so a background-loading retriever doesn't block startup
        if not lookup:
            by_id = {doc["id"]: doc for doc in retriever.docs}
            lookup.update(by_id=by_id, id_range=f"{min(by_id)}-{max(by_id)}" if by_id else "none")
        return lookup["by_id"], lookup["id_range"]

    @server.tool(description="Search the corpus. Returns the top-k passages with ids, titles, text previews and scores.")
    def search(query: QueryText, k: TopK = 5) -> SearchResult:
        if not query.strip():
            raise ToolError("query is empty: pass a question or an entity name to search for")
        with span("retrieval.search", retrieval__query=query[:200], retrieval__k=k) as s:
            hits = retriever.search(query, k)
            s.set_attribute("retrieval.results", len(hits))
        return SearchResult(query=query, results=[_passage(d, PREVIEW_CHARS) for d in hits],
                            truncated_chars=PREVIEW_CHARS)

    @server.tool(description="Return the full text of one passage by its id (ids come from search results).")
    def get_passage(passage_id: Annotated[int, Field(description="Passage id from a search result")]) -> PassageDetail:
        by_id, id_range = passages_by_id()
        doc = by_id.get(passage_id)
        if doc is None:
            raise ToolError(f"No passage with id {passage_id}. Valid ids are {id_range}; "
                            f"use the ids returned by `search`.")
        p = _passage(doc)
        return PassageDetail(**p.model_dump(), formatted=doc_text(doc))

    @server.tool(description=(
        "Answer a question with the full pipeline (multi-hop retrieval + fine-tuned reader). Returns the "
        "short answer, the searches it ran and the evidence passages. Loads the LLM on first use."
    ))
    def answer(question: Question, mode: Mode = "iterative") -> AnswerResult:
        if not question.strip():
            raise ToolError("question is empty")
        try:
            pipeline = get_pipeline()
            result = pipeline.answer(question, mode=mode, short=True)
        except ToolError:
            raise
        except Exception as exc:
            logger.exception("answer failed")
            raise ToolError(f"The answer pipeline failed ({type(exc).__name__}: {exc}). "
                            f"`search` still works for retrieving evidence directly.") from exc
        cited = [result.passages[i - 1]["id"] for i in parse_sources(result.raw_output)
                 if 0 < i <= len(result.passages)]
        return AnswerResult(answer=result.answer, queries=result.queries,
                            passages=[_passage(p, PREVIEW_CHARS) for p in result.passages],
                            citations=list(dict.fromkeys(cited)), latency_s=round(result.latency_s, 3))

    @server.resource("corpus://info", name="corpus-info", mime_type="application/json",
                     description="Corpus size and pipeline configuration")
    def corpus_info() -> dict:
        by_id, id_range = passages_by_id()
        return {"passages": len(by_id), "passage_ids": id_range, "retriever": config.retriever,
                "llm": llm_name, "top_k": config.top_k, "hops": config.num_hops, "max_k": MAX_K}

    return server


def lazy_pipeline(retriever, config) -> Callable[[], object]:
    """Build the RAGPipeline (and load the LLM) once, on first call, thread-safely."""
    lock, holder = threading.Lock(), {}

    def get():
        with lock:
            if "pipeline" not in holder:
                from src.llm import make_llm
                from src.rag_pipeline import RAGPipeline

                logger.info(json.dumps({"event": "loading_llm", "llm": config.llm}))
                llm = make_llm(config.llm, config.prompt_style, config.reasoning, config.load_in_4bit,
                               config.hf_max_batch_size)
                holder["pipeline"] = RAGPipeline(llm, retriever, config)
            return holder["pipeline"]

    return get


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args(argv)

    # stdout carries the MCP protocol over stdio, so logs go to stderr
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tracing = setup_tracing("multihop-rag-mcp")
    from src.api import config_from_env
    from src.rag_pipeline import load_retriever

    config = config_from_env()
    retriever = BackgroundRetriever(lambda: load_retriever(config))
    logger.info(json.dumps({"event": "ready", "retriever": config.retriever, "llm": config.llm,
                            "tracing": tracing, "index": "loading in background"}))
    server = create_server(retriever, config, lazy_pipeline(retriever, config), llm_name=config.llm)
    if args.transport == "stdio":
        server.run("stdio")
    else:
        server.run("streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
