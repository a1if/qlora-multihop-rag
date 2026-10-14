"""
src/telemetry.py
OpenTelemetry tracing for the pipeline and the MCP server.

The MCP SDK already wraps every tool call in a SERVER span (GenAI semantic
conventions: gen_ai.tool.name, ...). The pipeline adds child spans:

    tools/call search                  (MCP SDK)
    └─ retrieval.search                query, k, results
       └─ retrieval.rerank             candidates
    tools/call answer                  (MCP SDK)
    └─ rag.answer                      mode, k, hops
       ├─ retrieval.search             (one per hop)
       └─ llm.generate                 model, max_new_tokens, prompt/output size

Where spans go is chosen by environment variables, so nothing is sent
anywhere unless configured:

    OTEL_EXPORTER_OTLP_ENDPOINT   OTLP/HTTP collector: Jaeger, Grafana Tempo, Langfuse, ...
    QMR_TRACE_FILE                append spans as JSON lines to a local file
    QMR_TRACE_CONSOLE=1           print spans to stderr

Usage:
    from src.telemetry import setup_tracing, span
    setup_tracing("multihop-rag-mcp")
    with span("retrieval.search", query=q, k=5) as s:
        ...
        s.set_attribute("retrieval.results", len(hits))

    python -m src.telemetry traces.jsonl      # per-span latency summary
"""

import contextlib
import json
import os
import sys

try:
    from opentelemetry import trace
except ImportError:  # tracing is optional; spans become no-ops
    trace = None

_TRACER_NAME = "qlora-multihop-rag"


class _NoSpan:
    def set_attribute(self, *_):
        pass

    def set_attributes(self, *_):
        pass

    def record_exception(self, *_):
        pass


@contextlib.contextmanager
def span(name: str, **attributes):
    """Start a child span; does nothing if OpenTelemetry isn't installed or configured."""
    if trace is None:
        yield _NoSpan()
        return
    clean = {k.replace("__", "."): v for k, v in attributes.items() if v is not None}
    with trace.get_tracer(_TRACER_NAME).start_as_current_span(name, attributes=clean) as s:
        yield s


def current_trace_id() -> str | None:
    if trace is None:
        return None
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


# ─── Exporters ────────────────────────────────────────────────────────────────

def _span_record(s) -> dict:
    ctx = s.get_span_context()
    return {
        "name": s.name,
        "trace_id": format(ctx.trace_id, "032x"),
        "span_id": format(ctx.span_id, "016x"),
        "parent_id": format(s.parent.span_id, "016x") if s.parent else None,
        "start_ns": s.start_time,
        "duration_ms": round((s.end_time - s.start_time) / 1e6, 3),
        "status": s.status.status_code.name,
        "attributes": dict(s.attributes or {}),
    }


def _jsonl_exporter(path: str):
    from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

    class JsonlSpanExporter(SpanExporter):
        def export(self, spans):
            with open(path, "a", encoding="utf-8") as f:
                for s in spans:
                    f.write(json.dumps(_span_record(s), default=str) + "\n")
            return SpanExportResult.SUCCESS

    return JsonlSpanExporter()


def setup_tracing(service_name: str) -> bool:
    """Install a tracer provider if any exporter is configured. Returns True if tracing is on."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    trace_file = os.environ.get("QMR_TRACE_FILE")
    console = os.environ.get("QMR_TRACE_CONSOLE") == "1"
    if trace is None or not (endpoint or trace_file or console):
        return False

    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    if trace_file:
        provider.add_span_processor(SimpleSpanProcessor(_jsonl_exporter(trace_file)))
    if console:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr)))
    trace.set_tracer_provider(provider)
    return True


# ─── Trace File Summary ───────────────────────────────────────────────────────

def summarize(path: str) -> list[dict]:
    """Per-span-name count and latency percentiles from a QMR_TRACE_FILE."""
    import numpy as np

    by_name: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_name.setdefault(rec["name"], []).append(rec["duration_ms"])
            errors[rec["name"]] = errors.get(rec["name"], 0) + (rec["status"] == "ERROR")
    rows = []
    for name, durations in sorted(by_name.items()):
        d = np.asarray(durations)
        rows.append({"span": name, "count": len(d), "errors": errors[name],
                     "p50_ms": float(np.percentile(d, 50)), "p95_ms": float(np.percentile(d, 95))})
    return rows


if __name__ == "__main__":
    rows = summarize(sys.argv[1])
    print("| Span | Count | Errors | p50 (ms) | p95 (ms) |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| `{r['span']}` | {r['count']} | {r['errors']} | {r['p50_ms']:.1f} | {r['p95_ms']:.1f} |")
