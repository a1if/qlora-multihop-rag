HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=hybrid_rerank · embedder=all-MiniLM-L6-v2

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs base:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base:single` | 40.0 | 50.6 [46.6, 54.8] | 38.5 | 46.0 | 86.0 | 72.6 | – | 5.64 | 2.50 / 3.87 | 1.0 | 87% |
| `base:iterative` | 43.6 | 54.4 [50.4, 58.5] | 42.2 | 49.0 | 87.7 | 75.8 | +3.8 [+1.3, +6.2] | 2.55 | 6.26 / 7.23 | 2.0 | 91% |
| `ft:iterative` | 58.4 | 71.2 [67.8, 74.7] | 57.2 | 63.0 | 94.2 | 89.0 | +20.6 [+16.4, +24.7] | 2.61 | 6.06 / 6.60 | 2.0 | 91% |
