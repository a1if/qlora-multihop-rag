HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=hybrid_rerank · embedder=all-MiniLM-L6-v2

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs q8b:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `q8b:single` | 51.8 | 63.4 [59.8, 67.2] | 49.0 | 63.0 | 86.0 | 72.6 | – | 1.47 | 7.33 / 26.49 | 1.0 | 95% |
| `q8b:iterative` | 56.8 | 69.3 [65.9, 72.8] | 55.0 | 64.0 | 93.4 | 87.4 | +5.8 [+3.3, +8.5] | 0.31 | 61.18 / 91.47 | 2.0 | 98% |
