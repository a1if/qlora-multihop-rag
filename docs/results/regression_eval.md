Regression eval (CI): 25 golden questions, outputs recorded through the MCP `answer` tool.

| Recording | Metric | Value | CI gate |
|---|---|---|---|
| `sft_10k_hybrid_rerank` | questions | 25 | – |
| `sft_10k_hybrid_rerank` | em | 0.680 | min 0.6 |
| `sft_10k_hybrid_rerank` | f1 | 0.739 | min 0.66 |
| `sft_10k_hybrid_rerank` | retrieval_all_gold | 0.880 | min 0.8 |
| `sft_10k_hybrid_rerank` | retrieval_any_gold | 1.000 | – |
| `sft_10k_hybrid_rerank` | faithfulness | 1.000 | min 0.95 |
| `sft_10k_hybrid_rerank` | citation_precision | 0.958 | min 0.9 |
| `sft_10k_hybrid_rerank` | abstention_rate | 0.040 | – |
| `sft_10k_hybrid_rerank` | abstentions_justified | 1.000 | min 1.0 |
| `sft_10k_hybrid_rerank` | wrong_when_evidence_missing | 0.333 | max 0.5 |
