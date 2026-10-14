HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=hybrid_rerank · embedder=all-MiniLM-L6-v2

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs ft:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `ft:single` | 47.8 | 59.9 [55.9, 63.7] | 43.8 | 64.0 | 86.0 | 72.6 | – | 3.62 | 3.99 / 5.25 | 1.0 | 89% |
| `ft:iterative` | 56.8 | 70.1 [66.6, 73.6] | 54.5 | 66.0 | 94.8 | 90.2 | +10.2 [+6.9, +14.0] | 2.47 | 6.40 / 7.32 | 2.0 | 91% |
