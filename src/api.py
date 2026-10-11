"""
src/api.py
FastAPI service for the multi-hop RAG pipeline.

Endpoints:
    POST /v1/answer   answer a question (mode, top_k, hops, short/long answers)
    GET  /healthz     liveness
    GET  /readyz      readiness (pipeline loaded)
    GET  /metrics     Prometheus metrics: request counts, latency by stage

Configuration comes from environment variables (see Settings below), so the
same image serves a local model or proxies a vLLM / Ollama / hosted endpoint.

Usage:
    QMR_LLM=openai:deepseek-r1:1.5b@http://localhost:11434/v1 \\
        uvicorn src.api:app --host 0.0.0.0 --port 8080
"""

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from configs.training_config import RAGConfig

logger = logging.getLogger("qmr.api")


# ─── Settings ─────────────────────────────────────────────────────────────────

def config_from_env() -> RAGConfig:
    defaults = RAGConfig()
    return RAGConfig(
        llm=os.environ.get("QMR_LLM", defaults.llm),
        index_dir=os.environ.get("QMR_INDEX_DIR", defaults.index_dir),
        prompt_style=os.environ.get("QMR_PROMPT_STYLE", defaults.prompt_style),
        reasoning=os.environ.get("QMR_REASONING", "false").lower() == "true",
        top_k=int(os.environ.get("QMR_TOP_K", defaults.top_k)),
        num_hops=int(os.environ.get("QMR_HOPS", defaults.num_hops)),
    )


# ─── Schemas ──────────────────────────────────────────────────────────────────

Mode = Literal["closed", "single", "iterative", "decompose"]


class AnswerRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    mode: Mode = "iterative"
    top_k: int | None = Field(None, ge=1, le=20)
    hops: int | None = Field(None, ge=1, le=4)
    answer_style: Literal["short", "long"] = "long"


class Passage(BaseModel):
    title: str | None = None
    text: str
    score: float | None = None


class AnswerResponse(BaseModel):
    request_id: str
    answer: str
    mode: Mode
    queries: list[str]
    passages: list[Passage]
    latency_ms: float
    timings: dict


# ─── Metrics ──────────────────────────────────────────────────────────────────

class Metrics:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "qmr_requests_total", "Answer requests", ["mode", "status"], registry=self.registry
        )
        self.latency = Histogram(
            "qmr_stage_seconds", "Time spent per pipeline stage", ["stage"], registry=self.registry,
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
        )


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app(pipeline=None, config: RAGConfig | None = None) -> FastAPI:
    """Pass a ready pipeline (tests) or let the lifespan build one from env config."""
    config = config or config_from_env()
    metrics = Metrics()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if app.state.pipeline is None:
            from src.rag_pipeline import RAGPipeline

            logger.info(json.dumps({"event": "loading_pipeline", "llm": config.llm, "index": config.index_dir}))
            app.state.pipeline = RAGPipeline.from_config(config)
        yield
        app.state.pipeline.llm.close()

    app = FastAPI(title="QLoRA Multi-Hop RAG", version="1.0.0", lifespan=lifespan)
    app.state.pipeline = pipeline

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = rid
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        return response

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz():
        if app.state.pipeline is None:
            raise HTTPException(503, "pipeline not loaded")
        return {"status": "ready", "llm": app.state.pipeline.llm.name}

    @app.get("/metrics")
    def prometheus():
        return PlainTextResponse(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    # Sync handler: FastAPI runs it in a worker thread, so blocking model calls
    # don't stall the event loop.
    @app.post("/v1/answer", response_model=AnswerResponse)
    def answer(req: AnswerRequest, request: Request):
        pipeline = app.state.pipeline
        if pipeline is None:
            raise HTTPException(503, "pipeline not loaded")
        rid = request.state.request_id
        start = time.perf_counter()
        try:
            result = pipeline.answer(
                req.question, mode=req.mode, k=req.top_k, hops=req.hops,
                short=req.answer_style == "short",
            )
        except Exception:
            metrics.requests.labels(req.mode, "error").inc()
            logger.exception(json.dumps({"event": "answer_failed", "request_id": rid, "mode": req.mode}))
            raise HTTPException(502, "generation failed") from None

        elapsed = time.perf_counter() - start
        metrics.requests.labels(req.mode, "ok").inc()
        metrics.latency.labels("total").observe(elapsed)
        metrics.latency.labels("retrieval").observe(result.timings.get("retrieval_s", 0.0))
        metrics.latency.labels("llm").observe(result.timings.get("llm_s", 0.0))
        logger.info(json.dumps({
            "event": "answer", "request_id": rid, "mode": req.mode,
            "latency_ms": round(1000 * elapsed, 1), **result.timings,
        }))
        return AnswerResponse(
            request_id=rid,
            answer=result.answer,
            mode=result.mode,
            queries=result.queries,
            passages=[Passage(title=p.get("title"), text=p["text"], score=p.get("score")) for p in result.passages],
            latency_ms=1000 * elapsed,
            timings=result.timings,
        )

    return app


logging.basicConfig(level=os.environ.get("QMR_LOG_LEVEL", "INFO"), format="%(message)s")

# The model loads in the lifespan hook at server start, so importing this is cheap.
app = create_app()
