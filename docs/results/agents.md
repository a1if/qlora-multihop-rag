Agent benchmark: first 100 of the 500 HotpotQA dev questions, hybrid retrieval + rerank, k=5.

| Agent | EM | F1 (95% CI) | Both gold seen | Searches / q | Abstain | ΔF1 vs fixed pipeline (95% CI) |
|---|---|---|---|---|---|---|
| Base Qwen3-1.7B agent | 36 | 44.4 [35.5, 53.9] | 53% | 1.09 | 6% | -24.6 [-34.8, -13.7] |
| Base agent, thinking mode | 29 | 33.7 [24.3, 42.3] | 30% | 0.83 | 6% | -35.3 [-45.3, -24.7] |
| Tool-use fine-tuned agent v1 (10% abstain traces, final) | 55 | 64.1 [54.8, 72.5] | 93% | 1.97 | 17% | -4.9 [-14.0, +4.2] |
| Agent v2 (3% abstain traces), no guardrails | 0 | 0.0 [0.0, 0.0] | 0% | 0.00 | 0% | -69.0 [-76.6, -60.8] |
| Agent v2 + guardrails + forced answer | 38 | 47.6 [38.6, 56.6] | 97% | 5.00 | 36% | -21.4 [-30.6, -12.3] |
| Fine-tuned fixed 2-hop pipeline (reference) | 58 | 69.0 [60.8, 76.6] | 84% | 2.00 | 5% | – |
