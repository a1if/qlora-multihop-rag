Training runs (RTX 5070, 12 GB). From each run's metrics.json.

| Run | Examples / task | Steps | Train loss | Eval loss | Tokens/s | Peak VRAM (GB) | Time (min) |
|---|---|---|---|---|---|---|---|
| `sft_10k` | 10k reader/abstain/query (final pipeline model) | 625 | 0.307 | 0.234 | 2993 | 5.0 | 34 |
| `sft_full` | 29.7k reader/abstain/query | 1856 | 0.253 | 0.201 | 2911 | 5.1 | 105 |
| `agent_sft` | 7,040 agent steps, 10% abstain (final agent) | 440 | 0.247 | 0.145 | 2319 | 6.6 | 57 |
| `agent_sft_v2` | 7,040 agent steps, 3% abstain | 441 | 0.276 | 0.178 | 2327 | 6.5 | 57 |

Retriever fine-tune (`retriever_minilm`): 39,882 triplets, loss 0.546, 107 s.
