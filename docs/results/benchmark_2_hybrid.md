HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=hybrid · embedder=all-MiniLM-L6-v2

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs base:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base:single` | 36.8 | 45.9 [41.9, 50.1] | 34.2 | 47.0 | 78.0 | 58.0 | – | 6.23 | 2.36 / 4.81 | 1.0 | 91% |
| `base:iterative` | 42.6 | 52.3 [48.2, 56.3] | 40.8 | 50.0 | 81.8 | 65.6 | +6.4 [+3.5, +9.2] | 2.37 | 6.65 / 8.79 | 2.0 | 93% |
| `ft:iterative` | 52.6 | 64.8 [61.1, 68.5] | 51.0 | 59.0 | 89.5 | 81.4 | +18.9 [+14.5, +23.6] | 2.68 | 6.04 / 7.30 | 2.0 | 96% |
