HotpotQA distractor dev · n=500 · seed=0 · k=5 · hops=2 · prompt=chat · reasoning=False · retriever=dense · embedder=D:/qmr/runs/retriever_minilm

| System | EM | F1 (95% CI) | EM bridge | EM comp. | Support R@5 | All-support@5 | ΔF1 vs base:single (95% CI) | Throughput (q/s) | p50 / p95 latency under load (s) | LLM calls | LLM share |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `base:single` | 39.2 | 49.6 [45.8, 53.8] | 38.2 | 43.0 | 84.5 | 70.8 | – | 6.78 | 2.18 / 4.67 | 1.0 | 91% |
| `base:iterative` | 42.0 | 52.8 [48.8, 57.0] | 42.5 | 40.0 | 85.1 | 72.6 | +3.2 [+0.7, +5.5] | 2.56 | 6.14 / 8.54 | 2.0 | 96% |
| `ft:iterative` | 53.6 | 65.6 [62.0, 69.4] | 52.0 | 60.0 | 90.9 | 83.4 | +15.9 [+11.8, +20.2] | 2.40 | 6.80 / 7.80 | 2.0 | 94% |
