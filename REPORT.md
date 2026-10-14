# Improving a 1.7B LLM for Multi-Hop Question Answering: Technical Report

**Model:** Qwen3-1.7B · **Task:** HotpotQA multi-hop QA · **Hardware:** one RTX 5070 (12 GB), Windows · **Code:** this repository

Every number below comes from saved prediction files and can be regenerated with `python scripts/export_results.py`; the source table for each claim is linked in [docs/results/](docs/results/).

---

## 1. Summary

One small open model was improved one technique at a time and measured at each step on the same 500 questions:

| Stage | Technique | EM | F1 (95% CI) |
|---|---|---|---|
| 0 | Model alone (closed-book) | 10.4 | 13.9 [11.2, 16.8] |
| 1 | + RAG, single dense search | 34.0 | 43.2 [39.1, 47.4] |
| 2 | + iterative multi-hop retrieval | 37.2 | 47.8 [43.6, 52.0] |
| 3 | + hybrid BM25 + dense, cross-encoder rerank | 43.6 | 54.4 [50.4, 58.5] |
| 5 | + LoRA fine-tuned reader and query writer | **58.4** | **71.2 [67.8, 74.7]** |

- **5.6× the base model's exact match** (10.4 → 58.4).
- The fine-tuned 1.7B model **matches a 4.7× larger Qwen3-8B** (71.2 vs 69.3 F1, difference not significant) at **~8× the throughput**.
- A LangGraph agent driving the system through an MCP server went from 44.4 to 64.1 F1 after **tool-use fine-tuning**.
- Training was made **2.8× faster with 55% less memory**; evaluation **3.3× faster**.

Sources: [ladder (dense)](docs/results/benchmark_1_ladder_dense.md), [hybrid + rerank](docs/results/benchmark_3_hybrid_rerank.md), [8B reference](docs/results/benchmark_6_reference_qwen3_8b_4bit.md), [agents](docs/results/agents.md).

---

## 2. Setup

**Task.** HotpotQA questions need two facts from two Wikipedia paragraphs (e.g. *"Glenn Quagmire is voiced by the creator of what 2012 film?"* needs Quagmire → Seth MacFarlane → *Ted*).

**Benchmark.** 500 questions sampled from the HotpotQA distractor dev set (seed 0). Their paragraphs (2 gold + 8 distractors each) are pooled into one corpus of **4,943 paragraphs**, so retrieval must find the right 2 among thousands. The reader always gets **k = 5 passages**, whatever the retrieval strategy, so strategies are compared on an equal evidence budget. Decoding is greedy.

**Metrics.**
- **EM / F1** with the official HotpotQA answer normalisation.
- **Both-gold recall:** whether both supporting paragraphs were among the passages the reader saw.
- **95% bootstrap confidence intervals**, and **paired bootstrap** differences between systems on the same questions. A difference whose CI excludes 0 is called significant.

**Training data** comes only from the HotpotQA *train* split; the benchmark uses *dev*.

---

## 3. Method

### 3.1 Retrieval
- **Dense:** all-MiniLM-L6-v2 embeddings, FAISS inner product over normalised vectors (cosine).
- **BM25:** own inverted-index implementation over `title: text`.
- **Hybrid:** reciprocal rank fusion of dense and BM25 top-20 lists, then an optional **cross-encoder reranker** (ms-marco-MiniLM-L6-v2).
- **Multi-hop (iterative):** search the question, have the LLM write one follow-up query from the passages found, search again, and merge both result lists round-robin to k = 5. A **decompose** variant splits the question into sub-questions up front.

### 3.2 Fine-tuning the reader and query writer (`sft_10k`)
One LoRA adapter on Qwen3-1.7B, trained on three tasks built from HotpotQA train ([`src/data_sft.py`](src/data_sft.py)):

| Task | Input | Target |
|---|---|---|
| Reader | question + 2 gold + 3 hard-distractor passages (shuffled) | `Answer: X` + `Sources: [i] [j]` |
| Abstain | question with a gold passage removed | `Answer: insufficient evidence` |
| Query writer | question + hop-1 passages | the hop-2 bridge entity (e.g. "Seth MacFarlane") |

Prompts are produced by the **same functions the pipeline uses at inference**; a test checks they are byte-identical. Loss is computed on answer tokens only. bf16 LoRA, rank 32, all projection layers, 10k examples, 1 epoch, 34 minutes.

### 3.3 Tool-use fine-tuning for the agent (`agent_sft`)
A LangGraph ReAct agent calls the MCP server's `search` tool itself. Training traces from HotpotQA train show `search(question)` → `search(bridge entity)` → `Answer: X`, plus 15% early-stop and 10% abstain variants. Each step is rendered exactly as the live agent sees it: the real MCP tool schemas, the agent's system prompt and Qwen3's tool-call template. 7,040 steps, 57 minutes.

---

## 4. Results

### 4.1 Improvement ladder
See section 1. Notable details ([dense](docs/results/benchmark_1_ladder_dense.md), [hybrid](docs/results/benchmark_2_hybrid.md), [hybrid + rerank](docs/results/benchmark_3_hybrid_rerank.md)):
- **Multi-hop helps only when the follow-up query is good.** With the base model it adds +4.6 F1. The base model's follow-up searches changed the retrieved passages on 12 of 20 test questions but found no additional gold passages.
- **Fine-tuning the query writer is a large part of the fine-tuning gain:** with dense retrieval, both-gold recall rises from 59.8% (base) to 74.8% (fine-tuned).
- **Retrieval and fine-tuning stack:** fine-tuned multi-hop goes 62.1 → 64.8 (hybrid) → 71.2 F1 (+ reranker); both-gold recall reaches 89.0%.

### 4.2 Bigger model reference ([table](docs/results/benchmark_6_reference_qwen3_8b_4bit.md))

| Model | EM | F1 | Throughput |
|---|---|---|---|
| Qwen3-8B base, 4-bit | 56.8 | 69.3 [65.9, 72.8] | 0.31 q/s |
| Qwen3-1.7B fine-tuned | 58.4 | 71.2 [67.8, 74.7] | 2.61 q/s |

Paired difference (1.7B − 8B): **+1.6 EM [−2.4, +5.8], +2.0 F1 [−1.5, +5.4]**: a match, not a win. Caveats: the 8B is not fine-tuned and ran in 4-bit (required to fit 12 GB).

### 4.3 Grounding behaviour
- **Abstention** (dense retrieval, multi-hop): when retrieval missed a needed passage, the base model answered wrongly 83% of the time vs 29% for the fine-tuned model. Only 1.4% of questions were refused despite full evidence.
- **Citations:** precision 91%, recall 97% on the benchmark. On the 25-question regression set (recorded through the MCP server): **faithfulness 100%, citation precision 96%** ([table](docs/results/regression_eval.md)).

### 4.4 Agent ([table](docs/results/agents.md))

| Agent (100 questions) | EM | F1 | Both gold seen | Searches / q |
|---|---|---|---|---|
| Base Qwen3-1.7B | 36 | 44.4 | 53% | 1.09 |
| Base, thinking mode | 29 | 33.7 | 30% | 0.83 |
| **Tool-use fine-tuned (final)** | **55** | **64.1** | **93%** | 1.97 |
| Fine-tuned fixed pipeline (reference) | 58 | 69.0 | 84% | 2.00 |

Tool-use fine-tuning: **+19.7 F1** over the base agent (CI [+8.3, +30.8]). The final agent is statistically level with the fixed pipeline (−4.9 F1, CI [−14.0, +4.2]) and has the highest evidence recall of any setup.

### 4.5 Efficiency ([speed sweep](docs/results/training_speed_sweep.md), [training runs](docs/results/training_runs.md))

| | Before | After | How |
|---|---|---|---|
| Training throughput | 857 tok/s | 2,418 tok/s (**2.8×**) | LM head computed only over the answer span (`logits_to_keep`); loss and gradients verified identical |
| Training memory, batch 4 | 10.8 GB | 4.9 GB | same; batch 8 went from out-of-memory to 6.0 GB |
| Eval, multi-hop, 500 q | 11 min 23 s | 3 min 27 s (**3.3×**) | dynamic batching of concurrent generation calls; 95% identical answers |

QLoRA (4-bit) was 10% slower than bf16 LoRA at this model size; gradient checkpointing was required at every batch size.

---

## 5. Ablations and negative results

| Experiment | Result | Takeaway |
|---|---|---|
| 30k vs 10k training examples ([table](docs/results/benchmark_5_sft_full_30k.md)) | eval loss 0.234 → 0.201, but F1 −1.1 [−3.7, +1.6] | Validation loss isn't the goal; the cheaper 10k model is final |
| Decomposition vs iterative retrieval | +0.6 F1 (n.s.) vs +4.6 F1 | Writing the next query from evidence beats planning sub-questions blind |
| Fine-tuned embedder ([dense](docs/results/benchmark_4_tuned_embedder_dense.md), [hybrid](docs/results/benchmark_4_tuned_embedder_hybrid_rerank.md)) | single-hop both-gold 54.4% → 70.8% in 107 s of training; no gain on top of the reranker (71.0 vs 71.2) | Both fix the same misses; the tuned embedder is the cheap alternative to a reranker |
| Agent with the pipeline-tuned model | stopped calling tools entirely | Narrow fine-tuning overrides general behaviour; tool use needed its own data |
| Agent v2: abstain traces 10% → 3% | looped on its own answer entity until the step limit on all 100 questions (0 EM); guardrails recovered it to 38 EM | Abstain traces were also the model's only learned way to stop |
| Gemma 4 E4B reference | could not load: 5.6 GB unquantised embedding table, plus RAM and page-file limits with C: 98% full | Needs a larger GPU or freed system memory |

---

## 6. Engineering

- **Serving:** FastAPI (`/v1/answer`, Prometheus metrics) and an **MCP server** (`search`, `get_passage`, `answer`) with Pydantic input/output schemas, limits (k ≤ 20), actionable `ToolError` messages, a lazily loaded LLM, and background index loading so clients connect immediately. Verified live with the MCP Inspector ([session](docs/results/mcp_inspector.md)).
- **Observability:** OpenTelemetry spans per MCP request (SDK middleware) with `rag.answer` → `retrieval.search` / `retrieval.rerank` / `llm.generate` children, trace context propagated from the agent into the server, and a JSON log line per request.
- **Robustness:**
  - On CUDA out-of-memory, batches are split in half and retried. The retry runs outside the `except` block, because the live exception's traceback kept the failed attempt's tensors alive (memory grew 9.8 → 11.6 GB before this fix).
  - Agent guardrails: repeated-search blocking, a per-run search budget, and a forced final answer with the reply prefilled as `Answer:`.
  - Per-question error isolation in evaluation.
- **Windows constraints:**
  - A constant-memory safetensors resharder (a 16 GB single-file checkpoint exceeded the commit limit).
  - A GPU environment on a second drive.
  - transformers 5 API changes handled (`dtype`, `warmup_steps`, `train_sampling_strategy`).
- **Quality:**
  - 132 tests that run in seconds with a fake embedder and a scripted LLM, plus a real in-process MCP client.
  - A 25-question regression eval in GitHub Actions: live BM25 hit rate, plus gates on recorded faithfulness, citation precision, EM and abstention.

---

## 7. Example behaviour

Chatbot (`scripts/run_chatbot.ps1`), fine-tuned model, hybrid + rerank:

| Question | Answer | Correct? |
|---|---|---|
| Glenn Quagmire is voiced by the creator of what 2012 film? | **Ted** (sources: Glenn Quagmire, Seth MacFarlane) | ✓ |
| Avengers, who directed it | "I couldn't find enough evidence in the corpus" | ✓ (not in the corpus) |
| Which university was founded first, Duke or Tufts? | **Tufts** (sources: Tufts University, Duke University) | ✗ The cited passages say Duke 1838, Tufts 1852 |

The last one is a typical remaining error: retrieval and citations are right, but the 1.7B model compares the two dates wrongly. Comparison questions remain weaker than bridge questions in the benchmark too.

---

## 8. Limitations

- One benchmark (HotpotQA), one sample of 500 dev questions; agent results use 100. The corpus is the questions' own paragraph pool, not all of Wikipedia.
- The 8B reference is not fine-tuned and runs in 4-bit.
- Faithfulness is checked lexically (answer words appear in a cited passage), not with an LLM judge.
- All runs use greedy decoding with a single seed; training-seed variance was not measured.

## 9. Reproduce

```powershell
.\scripts\setup_gpu_env.ps1 -EnvRoot D:\qmr
python -m src.eval_hotpotqa --n 500 --retriever hybrid_rerank --systems base:closed base:single base:iterative
python -m src.data_sft --n 20000 --out data/sft
python -m src.finetune --set run_name=sft_10k --set max_train_examples=10000
python -m src.eval_hotpotqa --n 500 --retriever hybrid_rerank --model ft=hf:D:/qmr/runs/sft_10k/merged --systems ft:iterative
python -m src.data_agent --n 2500 --abstain-ratio 0.10 --out data/agent
python scripts/export_results.py
```
