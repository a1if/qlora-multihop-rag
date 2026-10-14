HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=hybrid_rerank · embedder=D:/qmr/runs/retriever_minilm

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs base:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base:single` | 40.2 | 51.0 [47.0, 55.2] | 38.8 | 46.0 | 86.0 | 72.6 | – | 5.01 | 2.63 / 4.04 | 1.0 | 83% |
| `base:iterative` | 43.8 | 54.7 [50.6, 58.7] | 42.2 | 50.0 | 88.1 | 76.6 | +3.7 [+1.1, +6.3] | 2.39 | 6.63 / 7.78 | 2.0 | 91% |
| `ft:iterative` | 58.4 | 71.0 [67.5, 74.7] | 57.5 | 62.0 | 93.9 | 88.4 | +20.0 [+15.8, +24.3] | 2.59 | 6.10 / 6.80 | 2.0 | 91% |
