# Task Tracker: Resume Upgrade

Branch: `eval-hotpotqa` · PR #1 open, CI green · Tests: 81 passing · Lint: clean

Goal: take **one** small open model (Qwen3-1.7B) and improve it step by step for multi-hop QA with RAG and fine-tuning, measuring every step on HotpotQA. No model comparisons, no cloud models. The resume story is the skills: RAG design, fine-tuning (QLoRA/LoRA, retriever tuning), evaluation, and shipping it as a service.

Legend: `[x]` done · `[ ]` todo · `[~]` written but not checked against real models/data

---

## Resume here (paused 2026-10-13, evening)

Done this session: sft_full benchmark (no gain over sft_10k → sft_10k is final), Qwen3-8B reference (fine-tuned 1.7B matches it at ~8× throughput, +2.0 F1 n.s.), recorded eval outputs + CI gates, Gemma 4 E4B attempt (failed on RAM / page file: C: is 98% full; needs freed RAM and fewer concurrent questions), agent benchmark + tool-use fine-tune (+19.7 F1). README has all results.

Next ideas:
1. Agent over-abstains (17%): fewer abstain traces (10% → 5%) or require 2 searches before abstaining; retrain (~1 h)
2. Retry Gemma 4 E4B reference with WSL/other apps closed and `--workers 4` (`D:\qmr\models\gemma-4-E4B-it` is resharded)
3. Final resume bullets; merge PR

## Older resume notes (2026-10-13, afternoon)

Improvement ladder so far (Qwen3-1.7B, HotpotQA dev n=500, k=5, iterative multi-hop from stage 2):

| Stage | EM | F1 (95% CI) | Both gold retrieved |
|---|---|---|---|
| 0 closed-book | 10.4 | 13.9 [11.2, 16.8] | – |
| 1 single-hop RAG | 34.0 | 43.2 [39.1, 47.4] | 54.4% |
| 2 iterative multi-hop | 37.2 | 47.8 [43.6, 52.0] | 59.8% |
| 3 + hybrid BM25 + dense + rerank | 43.6 | 54.4 [50.4, 58.5] | 75.8% |
| 5 + LoRA SFT (sft_10k) | 58.4 | 71.2 [67.8, 74.7] | 89.0% |

State at pause:
- `sft_full` (all 29,683 examples) finished: 1,856 steps in 1 h 45 min, 2,911 tok/s, peak 5.1 GB, train loss 0.253, **eval loss 0.201** (sft_10k: 0.234). Adapter + merged model in `D:\qmr\runs\sft_full\`. Its benchmark was NOT run (stopped on request).
- MCP server, telemetry, regression eval, LangGraph agent: code done and tested (see Phase 2b).

Next steps, in order (GPU):
1. Benchmark sft_full on the best retrieval:
   `python -m src.eval_hotpotqa --n 500 --retriever hybrid_rerank --model ft=hf:D:/qmr/runs/sft_full/merged --systems ft:single ft:iterative --out results/full`
   Compare with sft_10k (58.4 / 71.2) → data-size point; use the better model below.
2. Gemma 4 E4B reference row (4-bit, local snapshot on C:, no fine-tuning, same retrieval):
   `python -m src.eval_hotpotqa --n 500 --retriever hybrid_rerank --load-in-4bit --model gemma=hf:C:/Users/alift/.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2 --systems gemma:single gemma:iterative --out results/full`
3. Record golden outputs for CI, then set `recorded` gates in `evals/thresholds.json` just below the measured values:
   `python -m evals.record --llm hf:D:/qmr/runs/sft_full/merged --retriever hybrid_rerank --index results/full/index_n500_seed0 --name sft_full_hybrid_rerank`
4. Agent benchmark (base and fine-tuned) vs fixed pipeline:
   `python -m src.agent --eval --n 500 --llm hf:D:/qmr/runs/sft_full/merged --index results/full/index_n500_seed0 --name ft_agent`
   (and `--llm hf:Qwen/Qwen3-1.7B --name base_agent`); summarise EM/F1, tool calls/question, throughput
5. README results update (final ladder, agent row, Gemma reference, eval gates), resume bullets

Environment (every new shell):
```powershell
$env:HF_HOME = "D:\qmr\hf-cache"; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"; $env:QMR_RUNS_DIR = "D:\qmr\runs"
& D:\qmr\venv\Scripts\Activate.ps1
cd C:\Users\alift\qlora-multihop-rag
```
Local, gitignored artifacts: `results/` (baseline, batched, full: predictions + summaries + indexes), `data/sft/` (29,715 train / 743 val), runs in `D:\qmr\runs\` (sft_10k, sft_full, retriever_minilm, speed_*). Regenerate the data with `python -m src.data_sft --n 20000 --out data/sft`.

Note: free RAM is tight (~2 GB free during training). The CI-level test suite passes; a transient numpy MemoryError once came from that.

---

## Files

### Source: new
| File | Status | Purpose |
|---|---|---|
| `src/llm.py` | [~] | `LLM` protocol; `HFLLM` (in-process) and `OpenAICompatibleLLM` (own checkpoint served by vLLM / Ollama); `make_llm(spec)` |
| `src/retriever.py` | [x] | FAISS inner-product retriever over normalised embeddings, save/load, round-robin evidence merge |
| `src/text_utils.py` | [x] | `<think>` stripping; answer / query / sub-question parsing |
| `src/metrics.py` | [x] | HotpotQA EM/F1, support recall, bootstrap CI, paired-bootstrap delta |
| `src/hotpotqa.py` | [~] | Loads HotpotQA distractor dev set and pools its paragraphs into one corpus (dataset ID and fields not yet checked against the real dataset) |
| `src/eval_hotpotqa.py` | [~] | Ablation benchmark CLI: model specs, concurrent workers, resumable JSONL, summary.md/csv |
| `src/api.py` | [~] | FastAPI service: `/v1/answer`, `/healthz`, `/readyz`, `/metrics`, request IDs, JSON logs |
| `src/batching.py` | [~] | Dynamic batching of concurrent generate() calls (tested with fakes; GPU check pending) |
| `src/build_index.py` | [~] | Index build CLI (arXiv or HotpotQA) |
| `src/data_sft.py` | [x] | SFT data from HotpotQA train: reader (gold + hard distractors → answer + citations), abstain, hop-2 query writer |

### Source: modified
| File | Status | Change |
|---|---|---|
| `src/rag_pipeline.py` | [~] | Rewritten: 4 modes (closed / single / iterative / decompose), same passage budget for all, per-thread stage timings; fixed prompt-in-output, L2/cosine mismatch and sampling bugs |
| `src/app.py` | [~] | Gradio demo on the new pipeline, configured from env vars |
| `configs/training_config.py` | [x] | `RAGConfig` gains `llm` spec, prompt style, reasoning, generation settings; GPU-specific comments removed |
| `src/finetune.py` | [~] | Rewritten for the SFT data: LoRA/QLoRA, answer-only loss, speed settings, merge for serving (not yet run on GPU) |
| `src/evaluate.py` | [x] | **Deleted**: graded RAG against another model's output, not correct answers |

### Tests
| File | Status | Covers |
|---|---|---|
| `tests/conftest.py` | [x] | Fake embedder, scripted fake LLM, toy corpus |
| `tests/test_text_utils.py` | [x] | Parsing edge cases |
| `tests/test_metrics.py` | [x] | EM/F1, recall, CIs |
| `tests/test_retriever.py` | [x] | Ranking, cosine scores, save/load, merge |
| `tests/test_pipeline.py` | [x] | Each mode, evidence budget, thread-safe timings |
| `tests/test_api.py` | [x] | Endpoints, validation, request IDs, metrics, 502 on backend failure |
| `tests/test_llm.py` | [x] | HTTP payloads, API key, spec parsing |
| `tests/test_eval_harness.py` | [x] | End-to-end benchmark run, resume, reports |
| `tests/test_batching.py` | [x] | Batching, max size, group isolation, error propagation |
| `tests/test_finetune.py` | [x] | Config overrides, label masking, prompt parity with inference, end-of-turn token, data cache key |
| `tests/test_data_sft.py` | [x] | Task mix, gold/citation correctness, abstain support, bridge targets, train/inference prompt parity |

### Infra / docs
| File | Status | Purpose |
|---|---|---|
| `pyproject.toml` | [x] | Project metadata, pytest + ruff config |
| `requirements.txt` | [x] | Full set (training + serving); old BLEU/ROUGE deps removed |
| `requirements-serve.txt` | [x] | API + benchmark, no GPU |
| `requirements-dev.txt` | [x] | Tests + lint |
| `scripts/setup_gpu_env.ps1` | [x] | Windows GPU env: venv + HF cache on another drive, PyTorch CUDA 12.8 |
| `Dockerfile` | [x] | CPU API image, non-root, healthcheck (builds in CI) |
| `docker-compose.yml` | [~] | Ollama + indexer + API stack (not run yet) |
| `.dockerignore` | [x] | |
| `.github/workflows/ci.yml` | [x] | Ruff + pytest on Python 3.10/3.12, Docker build (green on PR #1) |
| `LICENSE` | [x] | MIT; holder "a1if", year 2026 (check these) |
| `README.md` | [x] | Rewritten around the engineering story; results placeholder |
| `.gitignore` | [x] | Added `indexes/` |
| `notebooks/qlora_multihop_rag.ipynb` | [x] | Outputs and Colab widget state cleared |

---

## Phases

### Phase 1: Fix and restructure (done)
- [x] Fix multi-hop bugs (prompt echo, index metric mismatch, sampling, `<think>` leakage)
- [x] Model backend layer (in-process HF, or own checkpoint behind vLLM / Ollama)
- [x] FastAPI service with metrics and logging
- [x] HotpotQA benchmark harness
- [x] Tests, lint, CI, Docker, README
- [x] Commit and open PR: https://github.com/a1if/qlora-multihop-rag/pull/1
- [x] CI green on the PR (tests on 3.10 + 3.12, Docker build)
- [x] Clear notebook outputs
- [x] Reframe docs around fine-tuning (no hosted-API pitch)
- [ ] Check `LICENSE` holder/year

### Phase 2: Improvement ladder (the core story)
One model, Qwen3-1.7B. Each stage adds one technique on top of the previous one and gets a row in the results table (EM, F1, retrieval recall@k, latency).
Compute: local RTX 5070 (12 GB), models straight from Hugging Face (`hf:` backend). Needs PyTorch built for CUDA 12.8+; bf16 supported.

Setup
- [x] Model: **Qwen3-1.7B** (thinking toggled via `enable_thinking`; bf16)
- [x] Local GPU environment at `D:\qmr` (C: is full): torch 2.11+cu128, transformers 5.19, peft 0.21, trl 1.15, bitsandbytes 0.50 (`scripts/setup_gpu_env.ps1`)
- [x] Smoke run on the real model (`--n 20`): pipeline works end to end. Fixed `torch_dtype` → `dtype` for transformers 5
  - finding: base model's hop-2 queries changed retrieval on 12/20 questions but found 0 extra gold passages, which motivates Stages 4–5
- [x] Full baseline n=500 for Stages 0–2 (`results/baseline/summary.md`)
- [ ] Split hygiene: train on HotpotQA *train*, report on a fixed *dev* sample (n=500)
- [ ] Experiment tracking (W&B or MLflow): config, loss curves, eval per run

The ladder
- [x] Stage 0: no retrieval (closed-book baseline): EM 10.4 · F1 13.9
- [x] Stage 1: + RAG, single-hop dense retrieval: EM 34.0 · F1 43.2 · support R@5 75.3%
- [x] Stage 2: + multi-hop retrieval: iterative EM 37.2 · F1 47.8 (+4.6 F1 vs single, 95% CI [+1.7, +7.5]) · support R@5 77.8%. Decompose: EM 34.6 · F1 43.8 (+0.6, n.s.), so iterative is the multi-hop strategy going forward
- [x] Stage 3: + hybrid BM25 + dense (RRF) and cross-encoder rerank: base iterative EM 43.6 · F1 54.4 (both-gold 75.8%); ft_10k iterative EM 58.4 · F1 71.2 (both-gold 89.0%). Hybrid alone: base 52.3 / ft 64.8 F1
- [x] Stage 4: + fine-tuned embedder (MNRL, 39,882 triplets, 107 s): dense single-hop both-gold 54.4% → 70.8%, EM 34.0 → 39.2; dense ft iterative 62.1 → 65.6 F1. With hybrid + rerank: no further gain (71.0 vs 71.2), overlaps with the reranker
- [~] Stage 5: + fine-tuned model, LoRA SFT. First run `sft_10k` (10k examples, 34 min, 2,993 tok/s, eval loss 0.234): ft:iterative EM 50.2 · F1 62.1 (+14.3 F1 vs base iterative, CI [+10.1, +18.6]); both-gold retrieval 59.8% → 74.8%; wrong answers when evidence missing 83% → 29%; citation precision 91% / recall 97%. ft:single EM 36.8 · F1 46.7 (abstains 31%, mostly when gold missing). Full 30k run next. one LoRA adapter trained on two tasks:
  - reader: question + gold and distractor passages → short answer with citations, "insufficient evidence" when gold is missing
  - query writer: question + hop-1 passages → hop-2 search query (bridge entity)
- [ ] Stage 6 (optional): + DPO on grounded vs unsupported answers sampled from the Stage 5 model

Code needed for the ladder
- [x] Data builder for the SFT tasks (`src/data_sft.py`): 20k train questions → 29,715 examples (17,847 reader · 2,031 abstain · 9,837 query), 743 val; prompts shared with the pipeline so train = inference text
- [x] Rewrite `src/finetune.py`: same chat prompt as inference, loss on answer tokens only, over-length examples dropped (never truncated), tokenized-data cache, JSON/`--set` config per run, adapter + merged model + run_config.json + metrics.json (tokens/s, peak VRAM)
- [ ] Throughput sweep on the 5070 (after baseline finishes): LoRA vs QLoRA, gradient checkpointing on/off, batch size → pick fastest setting that fits; record speed-up
- [x] Retriever fine-tuning script (`src/finetune_retriever.py`)
- [x] BM25 + RRF + cross-encoder reranker in `src/retriever.py`
- [x] Harness: `--retriever dense|hybrid|hybrid_rerank`, `--embedding-model <path>`
- [ ] Extra metrics: citation precision/recall, hallucination rate on unanswerable variants

Speed
- [x] Dynamic batching for local-model evaluation and the API (`src/batching.py`, `--workers 16` default)
- [x] Batching verified on GPU: single 4m10s → 1m36s (2.6×), iterative 11m23s → 3m27s (3.3×); 95–98% identical answers, EM within ±0.2
- [x] Training throughput sweep (`scripts/speed_sweep.py`, 20 steps, effective batch 16):

  | Setup | Tokens/s | Speed | Peak VRAM |
  |---|---|---|---|
  | LoRA, full logits, bs4 (before) | 857 | 1.00× | 10.8 GB |
  | **LoRA, answer-span logits, bs4** | **2418** | **2.82×** | **4.9 GB** |
  | LoRA, answer-span, bs8 | 2158 | 2.52× | 6.0 GB |
  | QLoRA 4-bit, answer-span, bs8 | 1954 | 2.28× | 5.6 GB |
  | LoRA, answer-span, bs16 | 1236 | 1.44× | 9.1 GB |
  | no gradient checkpointing (bs4, bs8) | OOM | – | – |

  Answer-span logits (LM head only over answer tokens) fixed an OOM at bs8 (>13.5 GB → 6.0 GB) and is the main speed-up. Defaults set to LoRA bs4 × GA4.

Understanding why it works (small ablations on Stage 5)
- [ ] LoRA rank 8 vs 32 vs 128
- [ ] QLoRA 4-bit vs 16-bit LoRA (quality vs memory)
- [ ] Training-set size 1k / 5k / 20k
- [ ] Error analysis: bridge vs comparison questions, before/after examples

### Phase 2b: MCP server + agent (user's plan, 2026-10-13)
- [x] 1. MCP docs / Python SDK: mcp 2.2.0 (`MCPServer`, the renamed `FastMCP`; in-process `Client` for tests)
- [x] 2. Server with three tools (`search`, `get_passage`, `answer`), typed inputs with constraints + descriptions, Pydantic output models → output schemas (`src/mcp_server.py`)
- [x] 3. Wraps the hybrid FAISS/BM25 retriever: k ≤ 20, length limits, `ToolError` messages that say what to do, lazy LLM load so search works without a GPU
- [x] 4. Tested with MCP Inspector 2.8.0 (CLI, streamable HTTP) against the live server and real index: tools/list shows schemas (k 1–20, enums, output schemas); search('Glenn Quagmire voice actor') → Glenn Quagmire, search('Seth MacFarlane 2012 film') → Seth MacFarlane; k=50, unknown id, blank query → clear errors; every call logged with latency + trace id. Server now answers the handshake immediately and loads the index (~35 s) in the background
- [x] 5. Observability: OpenTelemetry (SDK's per-request span + `rag.answer` / `retrieval.search` / `retrieval.rerank` / `llm.generate` children; OTLP / JSONL / console), JSON log per request with trace id, `python -m src.telemetry` latency summary; tested that one call = one connected trace
- [~] 6. Eval set: 25 questions + 250 paragraphs in `evals/`, live BM25 hit-rate gates in CI; recorded-output faithfulness/citation/EM gates need the real-model recording (`python -m evals.record`, after training)
- [x] 7. Dockerfile + compose `mcp` service, README: architecture diagram, MCP client setup (Inspector, Claude Desktop), tool table, observability, regression eval
- [x] Stretch: LangGraph ReAct agent over the MCP tools. Results (100 q): base agent EM 36 / F1 44.4 (1.09 searches/q); thinking mode 29 / 33.7; tool-use LoRA (`agent_sft`, 7,040 steps from `src/data_agent.py`, 57 min, eval loss 0.145) EM 55 / F1 64.1, both-gold 93%, 1.97 searches/q, +19.7 F1 vs base agent [+8.3, +30.8]; vs fixed fine-tuned pipeline −4.9 F1 n.s.; over-abstains 17%. Earlier notes: (`src/agent.py`): local Qwen3 via a LangChain chat model on Qwen3's tool-call template, own MCP 2.x → LangChain tool adapter (langchain-mcp-adapters pins mcp<2), trace context into the server, step limit; 6 tests with a scripted model. Benchmark vs fixed pipeline pending (GPU)

### Phase 3: Ship the final model
- [ ] Serve the best checkpoint with vLLM or Ollama; API `curl /v1/answer` works
- [ ] Optional: `docker compose` stack (Docker daemon currently returns 500). Not needed for training or benchmarking.

### Phase 4: Write-up
- [ ] README results: ladder table (Stage 0 → final), ablation table, data-size plot
- [ ] README "What I learned" section
- [ ] Publish adapter to Hugging Face Hub with a model card
- [ ] Final resume bullets with real numbers

### Optional
- [ ] Streaming responses in the API
