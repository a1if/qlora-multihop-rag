# MCP server: live verification with MCP Inspector

Session of 2026-10-13. The server ran over streamable HTTP against the real 4,943-paragraph benchmark index with hybrid BM25 + dense retrieval and cross-encoder reranking, and was driven by the official MCP Inspector CLI (v2.8.0).

```bash
QMR_INDEX_DIR=results/baseline/index_n500_seed0 QMR_RETRIEVER=hybrid_rerank \
QMR_TRACE_FILE=traces.jsonl python -m src.mcp_server --transport streamable-http --port 8765

npx @modelcontextprotocol/inspector@2.8.0 --cli http://127.0.0.1:8765/mcp --method tools/list
npx @modelcontextprotocol/inspector@2.8.0 --cli http://127.0.0.1:8765/mcp --method tools/call --tool-name search --tool-arg "query=..." --tool-arg k=3
```

## `tools/list`

```
- search      {'query': {'type': 'string', 'maxLength': 500}, 'k': {'type': 'integer', 'minimum': 1, 'maximum': 20, 'default': 5}} | outputSchema: True
- get_passage {'passage_id': {'type': 'integer'}} | outputSchema: True
- answer      {'question': {'type': 'string', 'maxLength': 2000}, 'mode': {'type': 'string', 'enum': ['closed', 'single', 'iterative', 'decompose'], 'default': 'iterative'}} | outputSchema: True
```

## Tool calls: both hops of a multi-hop question

```
=== search('Glenn Quagmire voice actor', k=3)
  [8] Glenn Quagmire  score=8.9744
  [3] Tiegs for Two  score=0.4191
  [0] Screams of Silence: The Story of Brenda Q  score=0.1505
=== search('Seth MacFarlane 2012 film', k=3)
  [2] Seth MacFarlane  score=7.4431
  [1321] Sing (2016 American film)  score=1.0053
  [195] Logan Lucky  score=0.8498
```

## Validation and error messages

```
=== search(k=50)
  ERROR: Error executing tool search: 1 validation error for searchArguments
  k  Input should be less than or equal to 20 [type=less_than_equal, input_value=50, input_type=int]
=== get_passage(99999)
  ERROR: Error executing tool get_passage: No passage with id 99999. Valid ids are 0-4942; use the ids returned by `search`.
=== search('   ')
  ERROR: Error executing tool search: query is empty: pass a question or an entity name to search for
```

## Server log: one JSON line per request

```json
{"event": "mcp_request", "method": "tools/call", "tool": "search", "outcome": "ok", "duration_ms": 4050.9, "trace_id": "ff6a531c8be792feefb68c3211a1246d"}
{"event": "mcp_request", "method": "tools/call", "tool": "search", "outcome": "ok", "duration_ms": 121.4, "trace_id": "65bb1240cb398fa8c4536d5d66410501"}
{"event": "mcp_request", "method": "tools/call", "tool": "search", "outcome": "tool_error", "duration_ms": 30.8, "trace_id": "3acfd2f4174ad1a98c35ca0486ae524b"}
{"event": "mcp_request", "method": "tools/call", "tool": "get_passage", "outcome": "tool_error", "duration_ms": 3.7, "trace_id": "13a42feb24bd64e2b34b49837ea4f94e"}
{"event": "mcp_request", "method": "tools/call", "tool": "search", "outcome": "tool_error", "duration_ms": 0.8, "trace_id": "df9ff2c9d73abb069f0b6cd949270ff0"}
```

The first search took 4 s because it waited for the index to finish loading in the background (the server answers the MCP handshake immediately).

## OpenTelemetry span summary (`python -m src.telemetry traces.jsonl`)

| Span | Count | Errors | p50 (ms) | p95 (ms) |
|---|---|---|---|---|
| `initialize` | 6 | 0 | 0.4 | 6.9 |
| `notifications/initialized` | 6 | 0 | 0.2 | 0.2 |
| `retrieval.rerank` | 2 | 0 | 466.6 | 813.2 |
| `tools/call get_passage` | 1 | 1 | 3.9 | 3.9 |
| `tools/call search` | 4 | 2 | 76.3 | 3461.6 |
| `tools/list` | 6 | 0 | 0.6 | 1.7 |

(The reranker ran while a training job shared the GPU, which inflates its latency.)
